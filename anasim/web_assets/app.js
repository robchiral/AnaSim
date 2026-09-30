// Page controller: loading, setup, and the running session.
import { Controls } from "./controls.js";
import { Monitor } from "./monitor.js";
import { connect } from "./transport.js";

const $ = (id) => document.getElementById(id);
let transport;

let catalog = null;
let info = null;
let monitor = null;
let controls = null;
let last = null;
let arrestShown = false;
let cshtTimer = null;
const pending = new Map();
let nextId = 1;

function show(screen) {
  for (const id of ["loading", "setup", "app"]) $(id).hidden = id !== screen;
}

function send(name, args = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    transport.postMessage({ type: "cmd", id, name, args });
  });
}

// Commands from controls surface failures in a dialog instead of rejecting silently.
function command(name, args) {
  return send(name, args).catch((message) => alertDialog("Command failed", message));
}

function onMessage({ data }) {
  switch (data.type) {
    case "load_error":
      $("loading-status").textContent = `AnaSim could not load: ${data.message}`;
      $("loading-status").classList.add("error");
      document.querySelector(".loading-bar").hidden = true;
      break;
    case "ready":
      catalog = data.catalog;
      buildSetup();
      show("setup");
      break;
    case "created":
      startSession(data.info);
      break;
    case "create_error":
      setupError(data.message);
      break;
    case "tick":
      onTick(JSON.parse(data.snap));
      break;
    case "result": {
      const request = pending.get(data.id);
      pending.delete(data.id);
      if (!request) break;
      if ("error" in data) request.reject(data.error);
      else request.resolve(data.value);
      break;
    }
    case "error":
      for (const request of pending.values()) request.reject(data.message);
      pending.clear();
      clearInterval(cshtTimer);
      if (last) {
        last = { ...last, running: false, ended: true, recording: false };
        updateBar(last);
      }
      alertDialog("Simulation stopped", data.message);
      break;
  }
}

// --- Setup ------------------------------------------------------------

function buildSetup() {
  const form = $("setup-form");
  $("version").textContent = `AnaSim ${catalog.version}`;
  for (const [name, [min, max]] of Object.entries(catalog.ranges)) {
    Object.assign(form.elements[name], { min, max });
  }
  for (const [name, { options, default: value }] of Object.entries(catalog.models)) {
    const select = form.elements[name];
    select.replaceChildren(...options.map((option) => new Option(option, option)));
    select.value = value;
  }
  form.elements.scenario_id.replaceChildren(
    ...catalog.scenarios.map((scenario) => new Option(scenario.label, scenario.id)),
  );

  const syncGuided = () => {
    const guided = form.elements.session.value === "guided";
    $("scenario-row").hidden = !guided;
    $("initial-state").disabled = guided;
    if (guided) {
      const scenario = catalog.scenarios.find((s) => s.id === form.elements.scenario_id.value);
      form.elements.mode.value = scenario.mode;
      form.elements.maint_type.value = scenario.maint_type;
    }
  };
  for (const radio of form.elements.session) radio.onchange = syncGuided;
  form.elements.scenario_id.onchange = syncGuided;
  syncGuided();

  form.onsubmit = (event) => {
    event.preventDefault();
    const params = readSetup(form);
    if (!params) return;
    $("setup-error").hidden = true;
    $("setup-start").disabled = true;
    $("setup-start").textContent = "Preparing session…";
    transport.postMessage({ type: "create", params });
  };
  $("setup-cancel").onclick = () => show("app");
}

function readSetup(form) {
  const f = form.elements;
  const params = {};
  for (const name of ["age", "weight", "height", "baseline_hb"]) {
    const input = f[name];
    const value = Number(input.value);
    if (input.value === "" || !Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) {
      const label = input.closest("label").firstChild.textContent.trim();
      setupError(`${label} must be between ${input.min} and ${input.max}.`);
      input.focus();
      return null;
    }
    params[name] = value;
  }
  params.sex = f.sex.value;
  params.renal_function = Number(f.renal_function.value);
  params.hepatic_function = Number(f.hepatic_function.value);
  for (const name of Object.keys(catalog.models)) params[name] = f[name].value;
  params.arterial_line_enabled = f.arterial_line_enabled.checked;
  params.end_on_cardiac_arrest = f.end_on_cardiac_arrest.checked;
  if (f.session.value === "guided") {
    params.scenario_id = f.scenario_id.value;
  } else {
    params.mode = f.mode.value;
    params.maint_type = f.maint_type.value;
  }
  return params;
}

function setupError(message) {
  $("setup-error").textContent = message;
  $("setup-error").hidden = false;
  $("setup-start").disabled = false;
  $("setup-start").textContent = "Start simulation";
}

// --- Session ------------------------------------------------------------

function startSession(sessionInfo) {
  info = sessionInfo;
  monitor?.destroy();
  last = null;
  arrestShown = false;
  delete $("step-instruction").dataset.instruction;
  $("setup-start").disabled = false;
  $("setup-start").textContent = "Start simulation";
  $("setup-cancel").hidden = false;
  show("app");

  monitor = new Monitor(info);
  controls = new Controls(info, command);
  controls.onMedicationsShown = refreshCsht;
  clearInterval(cshtTimer);
  cshtTimer = setInterval(() => controls.currentTab === "Medications" && refreshCsht(), 5000);

  const speed = $("speed");
  [speed.min, speed.max] = info.speed_range;
  speed.value = "1";
  $("record").title = info.recordings_dir
    ? `Record time-series data to a CSV file in ${info.recordings_dir}.`
    : "Record time-series data and download it as CSV when you stop.";

  const scenario = info.scenario;
  $("scenario").hidden = !scenario;
  if (scenario) $("scenario-name").textContent = scenario.name;
}

function refreshCsht() {
  send("csht").then((values) => controls.showCsht(values)).catch(() => {});
}

function onTick(snap) {
  last = snap;
  monitor.update(snap);
  controls.sync(snap.controls);
  updateBar(snap);
  if (info.scenario) updateScenario(snap.scenario);
  if (snap.download) download(snap.download.filename, snap.download.csv);
  if (snap.notice) alertDialog("Recording failed", snap.notice);
  if (snap.arrest_reason && !arrestShown) {
    arrestShown = true;
    confirmDialog(
      "Simulation endpoint reached",
      `Cardiac arrest: ${snap.arrest_reason}.\n\nResuscitation is not modeled, so the session ends here.`,
      [["Review final state", "outlined"], ["Start a new session", "primary"]],
    ).then((choice) => choice === 1 && newSession());
  }
}

function updateBar(snap) {
  const run = $("run");
  const status = $("status");
  let label, variant;
  if (snap.ended) {
    [label, variant] = ["Session ended", "primary"];
  } else if (snap.running) {
    [label, variant] = ["Pause simulation", "outlined"];
  } else if (snap.time > 0) {
    [label, variant] = ["Resume simulation", "primary outlined"];
  } else {
    [label, variant] = ["Start simulation", "primary"];
  }
  if (run.textContent !== label) run.textContent = label;
  run.className = `btn ${variant}`;
  run.disabled = snap.ended;
  if (document.activeElement !== $("speed")) $("speed").value = String(snap.speed);
  status.textContent = snap.running && snap.lagging ? `Actual speed ${snap.achieved_speed}×` : "";

  const record = $("record");
  record.setAttribute("aria-pressed", String(snap.recording));
  record.textContent = snap.recording ? "Stop recording" : "Record CSV";

  const total = Math.floor(snap.time);
  const pad = (n) => String(n).padStart(2, "0");
  $("clock").textContent = `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
}

function updateScenario(step) {
  const total = info.scenario.total;
  const status = $("step-status");
  const next = $("step-next");
  const target = $("step-target");
  if (step.complete) {
    $("step-title").textContent = "Scenario complete";
    $("step-instruction").textContent = info.scenario.description;
    $("scenario-count").textContent = "";
    status.textContent = "";
    target.hidden = true;
    next.hidden = true;
    return;
  }
  next.hidden = false;
  if ($("step-title").textContent !== step.title) {
    $("step-title").textContent = step.title;
  }
  // Instructions are authored in anasim/scenarios with simple <b>, <i>, and <br> markup.
  if ($("step-instruction").dataset.instruction !== step.instruction) {
    $("step-instruction").innerHTML = step.instruction;
    $("step-instruction").dataset.instruction = step.instruction;
  }
  $("scenario-count").textContent = `Step ${step.index + 1} of ${total}`;
  target.hidden = !step.target_tab;
  if (step.target_tab) target.textContent = `Open ${step.target_tab.toLowerCase()}`;
  status.textContent = step.met ? "" : step.status;
  next.disabled = !step.met;
  next.textContent = step.met && step.index === total - 1 ? "Finish scenario" : "Continue";
}

$("step-next").onclick = () => command("scenario_next");
$("step-target").onclick = () => {
  const tab = last?.scenario?.target_tab;
  if (tab) controls.openTab(tab);
};

$("run").onclick = () => last && command("run", { running: !last.running });
$("measure-nibp").onclick = () => command("nibp");
$("speed").onchange = () => {
  const input = $("speed");
  const value = Math.min(Math.max(Number(input.value) || 1, Number(input.min)), Number(input.max));
  input.value = String(value);
  command("speed", { value });
};
$("record").onclick = async () => {
  if (!last) return;
  const result = await command("record", { active: !last.recording });
  if (result && result.csv) download(result.filename, result.csv);
};
$("new-session").onclick = () => newSession();

async function newSession() {
  if (last?.recording) {
    const result = await command("record", { active: false });
    if (result && result.csv) download(result.filename, result.csv);
  }
  // The old session may already have stopped; the next one replaces it either way.
  await send("run", { running: false }).catch(() => {});
  show("setup");
}

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --- Dialogs ------------------------------------------------------------

function confirmDialog(title, text, buttons) {
  const dialog = $("dialog");
  $("dialog-title").textContent = title;
  $("dialog-text").textContent = text;
  const row = $("dialog-buttons");
  row.replaceChildren();
  return new Promise((resolve) => {
    buttons.forEach(([label, variant], index) => {
      const button = Object.assign(document.createElement("button"), {
        type: "button",
        className: `btn ${variant}`,
        textContent: label,
      });
      button.onclick = () => {
        dialog.close();
        resolve(index);
      };
      row.append(button);
    });
    dialog.oncancel = () => resolve(-1);
    if (!dialog.open) dialog.showModal();
  });
}

function alertDialog(title, text) {
  return confirmDialog(title, text, [["OK", "primary"]]);
}

try {
  transport = await connect(onMessage);
} catch (error) {
  onMessage({ data: { type: "load_error", message: error.message } });
}
