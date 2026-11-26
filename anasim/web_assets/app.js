// Page controller: loading, setup, and the running session.
import { $, setText } from "./display.js";
import { Controls } from "./controls.js";
import { Monitor } from "./monitor.js";
import { MonitorAudio } from "./audio.js";
import { Trends } from "./trends.js";
import { connect } from "./transport.js";
import { CasePanel } from "./case.js";

let transport;
const audio = new MonitorAudio();
const narrowWorkspace = window.matchMedia("(max-width: 900px)");
const workspace = $("workspace");
const toolsToggle = $("session-tools-toggle");
const sessionTools = $("session-tools");
const speed = $("speed");
const speedPresets = [...document.querySelectorAll(".speed-preset")];
const audioOptions = $("session-options");
const compactMonitor = window.matchMedia("(max-width: 720px)");
function syncMonitorDetails() {
  for (const id of ["respiratory-loops", "nmb-history"]) $(id).open = !compactMonitor.matches;
}
compactMonitor.addEventListener("change", syncMonitorDetails);
syncMonitorDetails();

let catalog = null;
let info = null;
let monitor = null;
let controls = null;
let casePanel = null;
let trends = null;
let last = null;
let arrestShown = false;
let cshtTimer = null;
const pending = new Map();
let nextId = 1;

function show(screen) {
  const wasHidden = $(screen).hidden;
  audioOptions.open = false;
  setToolsOpen(false);
  if (screen !== "app") {
    $("trends-dialog").close();
    audio.stopPlayback();
  }
  for (const id of ["loading", "setup", "app"]) $(id).hidden = id !== screen;
  if (wasHidden) {
    window.scrollTo(0, 0);
    $(screen === "setup" ? "setup-title" : "run").focus({ preventScroll: true });
  }
}

function setToolsOpen(open) {
  if (open) audioOptions.open = false;
  toolsToggle.setAttribute("aria-expanded", String(open));
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
      break;
    }
    case "load_error":
      $("loading-status").textContent = `AnaSim could not load: ${data.message}`;
      $("loading-status").classList.add("error");
      $("loading-progress").hidden = true;
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
      onTick(data.snap);
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
      audio.resetSession(info?.sample_interval);
      for (const request of pending.values()) request.reject(data.message);
      pending.clear();
      clearInterval(cshtTimer);
      if (last) {
        last = { ...last, running: false, ended: true, recording: false };
        updateBar(last);
      }
      casePanel?.freezeControls();
      alertDialog("Simulation stopped", data.message);
      break;
  }
}

// --- Setup ------------------------------------------------------------

const PATIENT_PRESETS = {
  healthy: {
    age: 40, sex: "male", weight: 70, height: 170, baseline_hb: 13.5,
    renal_function: "1.0", creatinine_clearance_ml_min: "", hepatic_function: "1.0",
    coronary_flow_reserve: "5", heart_failure: "none", copd_severity: "0",
    surgical_site: "other", pregnant: false, gestational_age_weeks: 28, position: "supine",
  },
  elderly_cad: {
    age: 76, sex: "male", weight: 74, height: 172, baseline_hb: 11.8,
    renal_function: "0.8", creatinine_clearance_ml_min: "", hepatic_function: "1.0",
    coronary_flow_reserve: "2", heart_failure: "hfpef", copd_severity: "0",
    surgical_site: "upper_abdominal", pregnant: false, gestational_age_weeks: 28, position: "supine",
  },
  obese_copd: {
    age: 64, sex: "male", weight: 128, height: 174, baseline_hb: 14.8,
    renal_function: "1.0", creatinine_clearance_ml_min: "", hepatic_function: "1.0",
    coronary_flow_reserve: "5", heart_failure: "none", copd_severity: "0.65",
    surgical_site: "lower_abdominal", pregnant: false, gestational_age_weeks: 28, position: "supine",
  },
  obstetric: {
    age: 31, sex: "female", weight: 64, height: 165, baseline_hb: 11.2,
    renal_function: "1.0", creatinine_clearance_ml_min: "", hepatic_function: "1.0",
    coronary_flow_reserve: "5", heart_failure: "none", copd_severity: "0",
    surgical_site: "lower_abdominal", pregnant: true, gestational_age_weeks: 38, position: "supine",
  },
};

function buildCaseBrowser(scenarios) {
  const browser = $("case-browser");
  browser.replaceChildren();
  const groups = [
    { key: "routine", title: "Routine" },
    { key: "crisis", title: "Crisis" },
  ];
  let first = true;
  for (const group of groups) {
    const matching = scenarios.filter((s) => (s.category ?? "routine") === group.key);
    if (!matching.length) continue;
    const section = document.createElement("div");
    section.className = "case-group";
    const heading = document.createElement("div");
    heading.className = "case-group-title";
    heading.textContent = group.title;
    section.append(heading);
    const list = document.createElement("div");
    list.className = "case-list";
    for (const scenario of matching) {
      const item = document.createElement("label");
      item.className = "case-item";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "scenario_id";
      radio.value = scenario.id;
      if (first) {
        radio.checked = true;
        first = false;
      }
      const title = document.createElement("span");
      title.className = "case-item-title";
      title.textContent = scenario.label;
      const meta = document.createElement("span");
      meta.className = "case-item-meta";
      const startTag = scenario.start_mode === "awake" ? "Awake" : "Maintenance";
      const maintTag = scenario.maint_type === "balanced" ? "Sevo" : "TIVA";
      meta.textContent = `${startTag} · ${maintTag}`;
      item.append(radio, title, meta);
      list.append(item);
    }
    section.append(list);
    browser.append(section);
  }
}

function buildSetup() {
  const form = $("setup-form");
  $("version").textContent = `v${catalog.version}`;
  for (const [name, [min, max]] of Object.entries(catalog.ranges)) {
    Object.assign(form.elements[name], { min, max });
  }
  buildCaseBrowser(catalog.scenarios);

  const syncMaintenance = () => {
    $("maintenance-row").hidden = form.elements.mode.value !== "steady_state";
  };
  for (const radio of form.elements.mode) radio.onchange = syncMaintenance;
  syncMaintenance();
  // Cases set their own starting state; open simulation asks for it.
  const syncSession = () => {
    const isCase = form.elements.session.value === "case";
    $("case-fields").hidden = !isCase;
    $("initial-state").hidden = isCase;
    const selectedId = form.elements.scenario_id.value;
    const scenario = catalog.scenarios.find((s) => s.id === selectedId) ?? catalog.scenarios[0];
    if (scenario) setText($("case-summary"), scenario.summary);
  };
  for (const radio of form.elements.session) radio.onchange = syncSession;
  $("case-browser").addEventListener("change", syncSession);
  syncSession();

  const syncPregnancy = () => {
    const female = form.elements.sex.value === "female";
    $("pregnancy-choice").hidden = !female;
    if (!female) form.elements.pregnant.checked = false;
    const pregnant = form.elements.pregnant.checked;
    setText($("weight-label"), pregnant ? "Pre-preg weight" : "Weight");
    $("gestation-row").hidden = !pregnant;
    $("position-row").hidden = !pregnant;
  };
  form.elements.sex.onchange = syncPregnancy;
  form.elements.pregnant.onchange = syncPregnancy;
  syncPregnancy();

  const presetButtons = [...form.querySelectorAll("[data-preset]")];
  const syncPresetHighlight = () => {
    const f = form.elements;
    for (const button of presetButtons) {
      const preset = PATIENT_PRESETS[button.dataset.preset];
      const matches = Object.entries(preset).every(([key, val]) => {
        if (!preset.pregnant && (key === "gestational_age_weeks" || key === "position")) return true;
        const field = f[key];
        if (!field) return true;
        return field.type === "checkbox" ? field.checked === Boolean(val) : String(field.value) === String(val);
      });
      button.setAttribute("aria-pressed", String(matches));
    }
  };
  for (const button of presetButtons) {
    button.onclick = () => {
      const preset = PATIENT_PRESETS[button.dataset.preset];
      const f = form.elements;
      for (const [key, val] of Object.entries(preset)) {
        const field = f[key];
        if (!field) continue;
        if (field.type === "checkbox") field.checked = Boolean(val);
        else field.value = String(val);
      }
      syncPregnancy();
      syncSummaries();
      $("setup-error").hidden = true;
    };
  }

  const syncSummaries = () => {
    const f = form.elements;
    const severity = (field) => field.selectedOptions[0].textContent.split(" (")[0].toLowerCase();
    const list = (items, empty) => {
      const text = items.join(", ");
      return text ? text.charAt(0).toUpperCase() + text.slice(1) : empty;
    };
    const w = f.weight.valueAsNumber;
    const h = f.height.valueAsNumber;
    const bmi = w > 0 && h > 0 ? w / ((h / 100) ** 2) : NaN;
    setText($("setup-bmi"), Number.isFinite(bmi) ? bmi.toFixed(1) : "--");

    const organs = [];
    if (Number(f.renal_function.value) !== 1) organs.push(`${severity(f.renal_function)} renal impairment`);
    if (f.creatinine_clearance_ml_min.value !== "") organs.push(`CrCl ${f.creatinine_clearance_ml_min.value} mL/min`);
    if (Number(f.hepatic_function.value) !== 1) organs.push(`${severity(f.hepatic_function)} hepatic impairment`);
    setText($("organ-function-summary"), list(organs, "Normal"));
    const conditions = [];
    if (Number(f.coronary_flow_reserve.value) < 5) conditions.push(`${severity(f.coronary_flow_reserve)} coronary disease`);
    if (f.heart_failure.value !== "none") conditions.push(f.heart_failure.value === "hfref" ? "HFrEF" : "HFpEF");
    if (Number(f.copd_severity.value) > 0) conditions.push(`${severity(f.copd_severity)} COPD`);
    if (f.surgical_site.value !== "other") conditions.push(`${severity(f.surgical_site)} surgery`);
    if (f.pregnant.checked) conditions.push(`${f.gestational_age_weeks.value} weeks pregnant`);
    setText($("clinical-conditions-summary"), list(conditions, "None"));
    syncPresetHighlight();
  };
  const onSetupEdit = () => {
    syncSummaries();
    $("setup-error").hidden = true;
  };
  form.addEventListener("input", onSetupEdit);
  form.addEventListener("change", onSetupEdit);
  syncSummaries();

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
      const labelEl = input.closest("label");
      const label = (labelEl.querySelector(".param-head > span:first-child") ?? labelEl.firstChild).textContent.trim();
      setupError(`${label} must be between ${input.min} and ${input.max}.`);
      for (let details = input.closest("details"); details; details = details.parentElement.closest("details")) {
        details.open = true;
      }
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
  params.surgical_site = f.surgical_site.value;
  if (f.pregnant.checked) params.position = f.position.value;
  params.arterial_line_enabled = f.arterial_line_enabled.checked;
  params.tci_enabled = f.tci_enabled.checked;
  params.patient_variability = f.patient_variability.checked;
  if (f.session.value === "case") {
    params.scenario_id = f.scenario_id.value;
    params.practice_mode = f.guidance.value;
  } else {
    params.mode = f.mode.value;
    params.maint_type = f.maint_type.value;
  }
  return params;
}

const SETUP_START_LABEL = "Open session";

function setupError(message) {
  $("setup-error").textContent = message;
  $("setup-error").hidden = false;
  $("setup-start").disabled = false;
  $("setup-start").textContent = SETUP_START_LABEL;
}

// --- Session ------------------------------------------------------------

function startSession(sessionInfo) {
  info = sessionInfo;
  syncMonitorDetails();
  audio.resetSession(info.sample_interval);
  monitor?.destroy();
  controls?.destroy();
  casePanel?.destroy();
  trends?.destroy();
  last = null;
  arrestShown = false;
  delete $("step-instruction").dataset.instruction;
  $("setup-start").disabled = false;
  $("setup-start").textContent = SETUP_START_LABEL;
  $("setup-cancel").hidden = false;
  show("app");
  workspace.scrollTop = 0;

  monitor = new Monitor(info);
  controls = new Controls(info, command);
  casePanel = new CasePanel(command, newSession, info);
  trends = new Trends(info);
  controls.onViewChanged = (tabChanged) => {
    if (last?.scenario) updateScenario(last.scenario);
    if (tabChanged && narrowWorkspace.matches) {
      const panel = workspace.querySelector(".tab-panel:not([hidden])");
      const tabsEdge = workspace.getBoundingClientRect().top + controls.tabList.offsetHeight;
      if (panel.getBoundingClientRect().top < tabsEdge) panel.scrollIntoView({ block: "start" });
    }
  };
  controls.onMedicationsShown = refreshCsht;
  clearInterval(cshtTimer);
  cshtTimer = setInterval(refreshCsht, 5000);

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
  audio.update(snap);
  monitor.update(snap);
  controls.sync(snap.controls, snap.time, snap.ended || Boolean(snap.scenario?.complete));
  trends.update(snap);
  updateBar(snap);
  if (info.scenario) updateScenario(snap.scenario);
  if (snap.ended) casePanel.freezeControls();
  if (snap.download) download(snap.download.filename, snap.download.csv);
  if (snap.notice) alertDialog("Recording failed", snap.notice);
  if (snap.ended && snap.arrest_reason && !arrestShown && !snap.scenario?.debrief) {
    arrestShown = true;
    confirmDialog(
      "Simulation endpoint reached",
      `Cardiac arrest: ${snap.arrest_reason}.\n\nProlonged ischemia has left the heart unable to recover.`,
      [["Review final state", "outlined"], ["Start a new session", "primary"]],
    ).then((choice) => choice === 1 && newSession());
  }
}

function updateBar(snap) {
  const run = $("run");
  const status = $("status");
  let label, variant;
  if (snap.ended) {
    [label, variant] = ["Ended", "primary"];
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
  if (document.activeElement !== speed) speed.value = String(snap.speed);
  speed.disabled = snap.ended;
  for (const preset of speedPresets) {
    preset.setAttribute("aria-pressed", String(Number(preset.dataset.speed) === snap.speed));
    preset.disabled = snap.ended;
  }
  const achieved = snap.achieved_speed >= 10 ? Math.round(snap.achieved_speed) : snap.achieved_speed;
  setText(status, snap.running && snap.lagging ? `Actual ${achieved}×` : "");

  const record = $("record");
  record.setAttribute("aria-pressed", String(snap.recording));
  setText(record, snap.recording ? "Stop recording" : "Record");
  $("recording-status").hidden = !snap.recording;

  const total = Math.floor(snap.time);
  const pad = (n) => String(n).padStart(2, "0");
  setText($("clock"), `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`);
}

function normalizeStepText(text) {
  return text.replace(/\s+/g, " ").trim().toLowerCase().replace(/\.$/, "");
}

function updateScenario(step) {
  casePanel.update(step);
  if (step.mode === "independent" || step.complete) return;
  const total = info.scenario.total;
  const status = $("step-status");
  const next = $("step-next");
  const target = $("step-target");
  const instructionEl = $("step-instruction");
  setText($("step-title"), step.title);
  // Instructions are authored in anasim/scenarios with simple <b>, <i>, and <br> markup.
  if (instructionEl.dataset.instruction !== step.instruction) {
    instructionEl.innerHTML = step.instruction;
    instructionEl.dataset.instruction = step.instruction;
    instructionEl.dataset.normalizedInstruction = normalizeStepText(instructionEl.textContent);
  }
  setText($("scenario-count"), `Step ${step.index + 1} of ${total}`);
  target.hidden = !step.target_tab || controls.currentTab === step.target_tab;
  if (step.target_tab) {
    const tab = controls.tabs.find((button) => button.dataset.tab === step.target_tab);
    setText(target, `Open ${tab.textContent.toLowerCase()}`);
  }
  const instruction = instructionEl.dataset.normalizedInstruction ?? normalizeStepText(instructionEl.textContent);
  const cue = normalizeStepText(step.status);
  const repeated = instruction === cue || instruction.startsWith(`${cue} `) || instruction.startsWith(`${cue}.`);
  setText(status, step.met || repeated ? "" : step.status);
  next.disabled = !step.met;
  setText(next, step.met && step.index === total - 1 ? "Finish and review" : "Continue");
}

$("step-next").onclick = () => command("scenario_next");
$("step-target").onclick = () => {
  const target = last?.scenario?.target_tab;
  if (!target) return;
  controls.openTab(target);
  controls.tabs.find((tab) => tab.dataset.tab === target).focus();
};

$("run").onclick = () => last && command("run", { running: !last.running });
$("measure-nibp").onclick = () => command("nibp");
$("nmb-mode").onchange = () => command("nmb_mode", { mode: $("nmb-mode").value });
$("nmb-repeat").onchange = () => command("nmb_repeat", { seconds: Number($("nmb-repeat").value) });
$("nmb-measure").onclick = () => command("nmb_measure");
speed.onchange = () => {
  const value = Math.min(Math.max(Number(speed.value) || 1, Number(speed.min)), Number(speed.max));
  speed.value = String(value);
  command("speed", { value });
};
speed.onkeydown = (event) => {
  if (event.key === "Enter") speed.blur();
};
for (const preset of speedPresets) {
  preset.onclick = () => {
    const value = Number(preset.dataset.speed);
    speed.value = String(value);
    command("speed", { value });
  };
}
$("record").onclick = async () => {
  audioOptions.open = false;
  if (narrowWorkspace.matches) toolsToggle.focus();
  setToolsOpen(false);
  if (!last) return;
  const result = await command("record", { active: !last.recording });
  if (result && result.csv) download(result.filename, result.csv);
};
$("new-session").onclick = () => {
  audioOptions.open = false;
  setToolsOpen(false);
  newSession();
};
toolsToggle.onclick = () => {
  const open = toolsToggle.getAttribute("aria-expanded") !== "true";
  setToolsOpen(open);
  if (open) $("record").focus();
};
document.addEventListener("focusin", (event) => {
  if (event.target !== toolsToggle && !sessionTools.contains(event.target)) setToolsOpen(false);
});
narrowWorkspace.addEventListener("change", () => setToolsOpen(false));
sessionTools.parentElement.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || toolsToggle.getAttribute("aria-expanded") !== "true") return;
  setToolsOpen(false);
  toolsToggle.focus();
});
document.addEventListener("pointerdown", (event) => {
  if (!audioOptions.contains(event.target)) audioOptions.open = false;
  if (!toolsToggle.contains(event.target) && !sessionTools.contains(event.target)) setToolsOpen(false);
});
audioOptions.querySelector("summary").onclick = () => setToolsOpen(false);
audioOptions.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !audioOptions.open) return;
  event.preventDefault();
  audioOptions.open = false;
  audioOptions.querySelector("summary").focus();
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
