// Page controller: loading, setup, and the running session.
import { Controls } from "./controls.js";
import { Monitor } from "./monitor.js";
import { Trends } from "./trends.js";
import { connect } from "./transport.js";

const $ = (id) => document.getElementById(id);

// Per-tick writes must skip unchanged text: replacing a button's text node
// mid-press makes Safari drop the click.
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

let transport;

let catalog = null;
let info = null;
let monitor = null;
let controls = null;
let trends = null;
let trendReturnTab = "Machine";
let last = null;
let arrestShown = false;
let cshtTimer = null;
const pending = new Map();
let nextId = 1;

function show(screen) {
  $("session-options").open = false;
  for (const id of ["loading", "setup", "app"]) $(id).hidden = id !== screen;
}

function send(name, args = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    transport.postMessage({ type: "cmd", id, name, args });
  });
}

// Show control command failures in a dialog.
function command(name, args) {
  return send(name, args).catch((message) => { alertDialog("Command failed", message); });
}

function onMessage({ data }) {
  switch (data.type) {
    case "load_progress": {
      const progress = $("loading-progress");
      progress.setAttribute("aria-valuenow", String(data.stage - 1));
      progress.setAttribute("aria-valuetext", `${data.label}, step ${data.stage} of 3`);
      for (const [index, segment] of [...progress.children].entries()) {
        segment.dataset.state = index < data.stage - 1 ? "complete" : index === data.stage - 1 ? "active" : "";
      }
      $("loading-status").textContent = data.label;
      $("loading-stage").textContent = `${data.stage} / 3`;
      break;
    }
    case "load_error":
      $("loading-status").textContent = `AnaSim could not load: ${data.message}`;
      $("loading-status").classList.add("error");
      $("loading-progress").hidden = true;
      $("loading-stage").hidden = true;
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
  form.elements.scenario_id.replaceChildren(
    ...catalog.scenarios.map((scenario) => new Option(scenario.label, scenario.id)),
  );

  const syncOpioidPlan = () => {
    const maintenance = form.elements.mode.value === "steady_state";
    $("maintenance-row").hidden = !maintenance;
    form.elements.concomitant_opioids.disabled = maintenance;
    if (maintenance) form.elements.concomitant_opioids.checked = true;
  };
  for (const radio of form.elements.mode) radio.onchange = syncOpioidPlan;
  const syncGuided = () => {
    const guided = form.elements.session.value === "guided";
    $("scenario-row").hidden = !guided;
    $("initial-state").disabled = guided;
    $("initial-state").hidden = guided;
    $("scenario-initial-state").hidden = !guided;
    if (guided) {
      const scenario = catalog.scenarios.find((s) => s.id === form.elements.scenario_id.value);
      form.elements.concomitant_opioids.checked = true;
      form.elements.mode.value = scenario.mode;
      form.elements.maint_type.value = scenario.maint_type;
      const start = scenario.mode === "awake"
        ? "Awake before induction"
        : `Anesthetized · ${form.elements.maint_type.selectedOptions[0].textContent}`;
      setText($("scenario-start-summary"), `${start} · Opioids planned`);
    }
    syncOpioidPlan();
  };
  for (const radio of form.elements.session) radio.onchange = syncGuided;
  form.elements.scenario_id.onchange = syncGuided;
  syncGuided();

  const syncPregnancy = () => {
    const female = form.elements.sex.value === "female";
    $("pregnancy-choice").hidden = !female;
    if (!female) form.elements.pregnant.checked = false;
    const pregnant = form.elements.pregnant.checked;
    form.elements.weight.closest("label").firstChild.textContent = pregnant ? "Pre-pregnancy weight " : "Weight ";
    $("gestation-row").hidden = !pregnant;
    $("position-row").hidden = !pregnant;
  };
  form.elements.sex.onchange = syncPregnancy;
  form.elements.pregnant.onchange = syncPregnancy;
  syncPregnancy();

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
  const params = { gestational_age_weeks: 0, position: "supine" };
  const numeric = ["age", "weight", "height", "baseline_hb"];
  if (f.creatinine_clearance_ml_min.value !== "") numeric.push("creatinine_clearance_ml_min");
  if (f.pregnant.checked) numeric.push("gestational_age_weeks");
  for (const name of numeric) {
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
  params.coronary_flow_reserve = Number(f.coronary_flow_reserve.value);
  params.heart_failure = f.heart_failure.value;
  params.copd_severity = Number(f.copd_severity.value);
  if (f.pregnant.checked) params.position = f.position.value;
  params.arterial_line_enabled = f.arterial_line_enabled.checked;
  params.end_on_cardiac_arrest = f.end_on_cardiac_arrest.checked;
  params.tci_enabled = f.tci_enabled.checked;
  params.concomitant_opioids = f.concomitant_opioids.checked;
  params.patient_variability = f.patient_variability.checked;
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
  controls?.destroy();
  trends?.destroy();
  last = null;
  arrestShown = false;
  delete $("step-instruction").dataset.instruction;
  $("setup-start").disabled = false;
  $("setup-start").textContent = "Start simulation";
  $("setup-cancel").hidden = false;
  show("app");

  monitor = new Monitor(info);
  controls = new Controls(info, command);
  controls.onViewChanged = () => {
    if (last?.scenario) updateScenario(last.scenario);
  };
  trends = new Trends(info);
  trendReturnTab = "Machine";
  controls.onMedicationsShown = refreshCsht;
  clearInterval(cshtTimer);
  cshtTimer = setInterval(refreshCsht, 5000);

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
  if ($("app").hidden || !controls.needsCsht()) return;
  send("csht").then((values) => controls.showCsht(values)).catch(() => {});
}

function onTick(snap) {
  last = snap;
  monitor.update(snap);
  controls.sync(snap.controls);
  trends.update(snap);
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
    [label, variant] = ["Pause", "outlined"];
  } else if (snap.time > 0) {
    [label, variant] = ["Resume", "primary outlined"];
  } else {
    [label, variant] = ["Start", "primary"];
  }
  setText(run, label);
  run.className = `btn ${variant}`;
  run.disabled = snap.ended;
  if (document.activeElement !== $("speed")) $("speed").value = String(snap.speed);
  setText(status, snap.running && snap.lagging ? `Actual speed ${snap.achieved_speed}×` : "");

  const record = $("record");
  record.setAttribute("aria-pressed", String(snap.recording));
  setText(record, snap.recording ? "Stop recording" : "Record CSV");
  $("recording-status").hidden = !snap.recording;

  const total = Math.floor(snap.time);
  const pad = (n) => String(n).padStart(2, "0");
  setText($("clock"), `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`);
}

function updateScenario(step) {
  const total = info.scenario.total;
  const status = $("step-status");
  const next = $("step-next");
  const target = $("step-target");
  if (step.complete) {
    setText($("step-title"), "Scenario complete");
    setText($("step-instruction"), info.scenario.description);
    setText($("scenario-count"), "");
    setText(status, "");
    target.hidden = true;
    next.hidden = true;
    return;
  }
  next.hidden = false;
  setText($("step-title"), step.title);
  // Instructions are authored in anasim/scenarios with simple <b>, <i>, and <br> markup.
  if ($("step-instruction").dataset.instruction !== step.instruction) {
    $("step-instruction").innerHTML = step.instruction;
    $("step-instruction").dataset.instruction = step.instruction;
  }
  setText($("scenario-count"), `Step ${step.index + 1} of ${total}`);
  const sectionVisible = !step.target_section || controls.eventSections.get(step.target_section).section.open;
  target.hidden = !step.target_tab || (controls.currentTab === step.target_tab && sectionVisible);
  if (step.target_tab) {
    const tab = controls.tabs.find((button) => button.dataset.tab === step.target_tab);
    setText(target, `Open ${tab.textContent.toLowerCase()}`);
  }
  const normalize = (text) => text.replace(/\s+/g, " ").trim().toLowerCase();
  const repeated = normalize($("step-instruction").textContent).includes(normalize(step.status));
  setText(status, step.met || repeated ? "" : step.status);
  next.disabled = !step.met;
  setText(next, step.met && step.index === total - 1 ? "Finish scenario" : "Continue");
}

$("step-next").onclick = () => command("scenario_next");
$("step-target").onclick = () => {
  const tab = last?.scenario?.target_tab;
  if (tab) controls.openTab(tab);
  const section = last?.scenario?.target_section;
  if (section) controls.openEventSection(section);
};

$("run").onclick = () => last && command("run", { running: !last.running });
$("measure-nibp").onclick = () => command("nibp");
$("nmb-mode").onchange = () => command("nmb_mode", { mode: $("nmb-mode").value });
$("nmb-repeat").onchange = () => command("nmb_repeat", { seconds: Number($("nmb-repeat").value) });
$("nmb-measure").onclick = () => command("nmb_measure");
$("speed").onchange = () => {
  const input = $("speed");
  const value = Math.min(Math.max(Number(input.value) || 1, Number(input.min)), Number(input.max));
  input.value = String(value);
  command("speed", { value });
};
$("record").onclick = async () => {
  $("session-options").open = false;
  if (!last) return;
  const result = await command("record", { active: !last.recording });
  if (result && result.csv) download(result.filename, result.csv);
};
$("new-session").onclick = () => {
  $("session-options").open = false;
  newSession();
};
$("show-trends").onclick = () => {
  $("session-options").open = false;
  if (controls.currentTab !== "Trends") trendReturnTab = controls.currentTab;
  controls.openTab("Trends");
  trends.requestDraw();
  $("close-trends").focus();
};
$("close-trends").onclick = () => {
  controls.openTab(trendReturnTab);
  $("session-options").querySelector("summary").focus();
};
$("trends-panel").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    $("close-trends").click();
  }
});
document.addEventListener("pointerdown", (event) => {
  const options = $("session-options");
  if (!options.contains(event.target)) options.open = false;
});
$("session-options").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("session-options").open = false;
    $("session-options").querySelector("summary").focus();
  }
});

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
