import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { Controls } from "../anasim/web_assets/controls.js";

const { states: DEFIB_STATES, pyState: TCI_STOP_STATE } = JSON.parse(execFileSync("python3", ["-B", "-c", `
import json
from anasim.web import WebSession
s = WebSession(dict(age=40, weight=70, height=170, sex="male", mode="awake", tci_enabled=True))
states = []
def save(): states.append(s.snapshot()["controls"]["defibrillator"])
save()
s.command("charge_defibrillator")
save()
s.engine.start()
for _ in range(50): s.engine.step(0.1)
save()
s.command("shock", '{"synchronized": true}')
save()
for _ in range(10): s.engine.step(0.1)
save()
s.command("drug_target", '{"key": "propofol", "target": 3.0}')
on_state = s.snapshot()["controls"]["drugs"]["propofol"]
s.command("drug_target", '{"key": "propofol", "target": 0}')
zero_controls = s.snapshot()["controls"]
zero_state = zero_controls["drugs"]["propofol"]
print(json.dumps({
    "states": states,
    "pyState": [on_state["is_tci"], zero_state["is_tci"], zero_state["rate"], zero_controls],
}))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));

test("lipid controls submit doses and clear the receipt for a new session", async () => {
  const elements = new Map();
  for (const id of ["c-lipid-bolus", "c-lipid-give", "c-lipid-feedback", "c-lipid-rate",
    "c-lipid-stop", "fluid-feedback", "c-maint"]) {
    elements.set(id, { value: "0", disabled: false, hidden: false, textContent: "" });
  }
  globalThis.document = { getElementById: id => elements.get(id), querySelectorAll: () => [] };
  const commands = [];
  const controls = Object.create(Controls.prototype);
  controls.editSettings = () => ({ cancel() {} });
  controls.send = async (name, args) => {
    commands.push([name, args]);
    return { fluid: { text: "90 mL 20% lipid emulsion over 2.5 min started" } };
  };
  controls.bindFluids(60);
  assert.equal(elements.get("c-lipid-bolus").value, "90");
  await elements.get("c-lipid-give").onclick();
  assert.deepEqual(commands[0], ["fluid", { kind: "lipid", volume_ml: 90 }]);
  assert.equal(elements.get("c-lipid-feedback").hidden, false);
  assert.match(elements.get("c-lipid-feedback").textContent, /90 mL/);
  await elements.get("c-lipid-stop").onclick();
  assert.deepEqual(commands[1], ["lipid_rate", { ml_min: 0 }]);
  controls.bindFluids(80);
  assert.equal(elements.get("c-lipid-bolus").value, "100");
  assert.equal(elements.get("c-lipid-feedback").hidden, true);
});

test("defibrillator controls follow charging, pending R waves, and actual delivery", async () => {
  const elements = new Map();
  for (const id of ["c-compressions", "c-charge", "c-shock", "c-shock-feedback", "c-defibrillator-status"]) {
    elements.set(id, { disabled: false, hidden: false, textContent: "" });
  }
  const modes = [false, true].map(value => ({
    dataset: { value: String(value) }, disabled: false,
    getAttribute() {}, setAttribute() {},
  }));
  elements.set("c-sync", { querySelectorAll: () => modes });
  globalThis.document = { getElementById: id => elements.get(id) };
  const states = DEFIB_STATES;
  const controls = Object.create(Controls.prototype);
  const commands = [];
  controls.send = async name => { commands.push(name); };
  controls.state = { defibrillator: states[0] };
  controls.bindResuscitation();
  controls.syncDefibrillator(states[0]);
  assert.equal(elements.get("c-shock").disabled, true);
  await elements.get("c-charge").onclick();
  assert.deepEqual(commands, ["charge_defibrillator"]);

  controls.state = { defibrillator: states[1] };
  controls.syncDefibrillator(states[1]);
  assert.match(elements.get("c-defibrillator-status").textContent, /Charging/);
  controls.syncDefibrillator({ ...states[1], charge_remaining: 5 + 1e-12 });
  assert.equal(elements.get("c-defibrillator-status").textContent, "Charging, 5 s remaining");
  assert.equal(elements.get("c-shock").disabled, true);
  await elements.get("c-charge").onclick();
  assert.equal(commands.at(-1), "disarm_defibrillator");

  controls.syncDefibrillator(states[2]);
  assert.equal(elements.get("c-shock").disabled, false);
  controls.synchronized = false;
  controls.syncDefibrillator(states[3]);
  assert.equal(controls.synchronized, true);
  assert.equal(elements.get("c-shock").disabled, true);
  assert.ok(modes.every(button => button.disabled));
  assert.equal(elements.get("c-shock-feedback").hidden, true);
  controls.syncDefibrillator(states[4]);
  assert.match(elements.get("c-shock-feedback").textContent, /Synchronized shock 200 J delivered/);
  assert.equal(elements.get("c-shock").disabled, true);
  assert.ok(modes.every(button => !button.disabled));
  clearTimeout(controls.shockFeedbackTimer);
});

test("stopping a TCI infusion resets target to null and clears is_tci in engine", async () => {
  const stubs = new Map();
  globalThis.document = {
    activeElement: null,
    getElementById: id => {
      if (!stubs.has(id)) {
        stubs.set(id, {
          value: "", disabled: false, hidden: false, checked: false,
          className: "", textContent: "", querySelectorAll: () => [],
          append() {}, getAttribute() { return null; }, setAttribute() {}, focus() {},
        });
      }
      return stubs.get(id);
    },
    createElement: () => {
      const children = new Map();
      return {
        className: "",
        hidden: false,
        contains: () => false,
        set innerHTML(_v) {
          children.set(".infusion-name", { textContent: "", title: "" });
          children.set(".infusion-adjust", { onclick: null, setAttribute() {} });
          children.set(".infusion-stop", { onclick: null, setAttribute() {} });
          children.set(".infusion-editor", { id: "", hidden: true, append() {} });
          children.set("output", { textContent: "" });
        },
        querySelector: sel => children.get(sel),
      };
    },
  };
  const sent = [];
  const controls = Object.create(Controls.prototype);
  controls.performAction = async (_btn, name, args) => {
    sent.push([name, args]);
    return {};
  };
  const w = {
    spec: { generic_name: "Propofol", name: "Propofol 10 mg/mL" },
    card: { open: false },
    form: {},
    mode: { value: "tci" },
    target: { value: "", disabled: false, focus() {} },
    rate: { value: "", disabled: false, focus() {} },
    editor: { pending: false, values: new Map(), cancel() {} },
  };
  controls.buildRunningInfusion("propofol", w);
  const stopBtn = w.runningRow.querySelector(".infusion-stop");
  controls.state = { drugs: { propofol: { is_tci: true } } };
  await stopBtn.onclick();
  assert.deepEqual(sent[0], ["drug_target", { key: "propofol", target: null }]);
  controls.state = { drugs: { propofol: { is_tci: false } } };
  await stopBtn.onclick();
  assert.deepEqual(sent[1], ["drug_rate", { key: "propofol", rate: 0 }]);

  const pyState = TCI_STOP_STATE;
  assert.deepEqual(pyState.slice(0, 3), [true, false, 0]);

  controls.weight = 70;
  controls.lipidMaximum = 840;
  controls.drugs = { propofol: w };
  controls.showBloodGases = () => {};
  controls.showClinicalFeedback = () => {};
  controls.maskSealEditor = { cancel() {} };
  controls.syncDefibrillator = () => {};
  controls.applyVentMode = () => {};
  controls.syncRhythm = () => {};
  controls.syncEventSection = () => {};
  controls.sync(pyState[3], 0);
  assert.equal(w.mode.value, "tci");
  assert.equal(w.target.disabled, false);
  assert.equal(w.rate.disabled, true);
  assert.equal(stubs.get("c-apl-state").textContent, "Standby", "APL stays available for presetting without an airway");
  assert.equal(stubs.get("c-bag-breath").disabled, true, "single breath disabled without airway");
  controls.sync({ ...pyState[3], airway: "Mask", bag_mask: false, vent: { ...pyState[3].vent, on: false } }, 0);
  assert.equal(stubs.get("c-apl-state").textContent, "Open", "APL applies during spontaneous breathing on mask");
  assert.equal(stubs.get("c-bag-breath").disabled, false, "single breath available on mask with vent and bagging off");
  assert.equal(stubs.get("c-bag-state").textContent, "Normal");
  controls.sync({ ...pyState[3], airway: "Mask", bag_mask: false, vent: { ...pyState[3].vent, on: true } }, 0);
  assert.equal(stubs.get("c-apl-state").textContent, "Standby", "the ventilator uses its own pressure setting");
  assert.equal(stubs.get("c-bag-breath").disabled, true, "single breath disabled while ventilator is on");
  controls.sync({ ...pyState[3], airway: "Mask", bag_mask: true, apl: 0, vent: { ...pyState[3].vent, on: false } }, 0);
  assert.equal(stubs.get("c-apl-state").textContent, "Open", "APL also limits bag ventilation");
  assert.equal(stubs.get("c-apl-state").className, "warn");
  assert.equal(stubs.get("c-bag-state").textContent, "No pressure");
  assert.equal(stubs.get("c-bag-breath").disabled, true, "single breath disabled during continuous bagging");
  controls.sync({
    ...pyState[3],
    airway: "ETT",
    bag_mask: true,
    apl: 70,
    o2_connected: false,
    o2_flush: false,
    reservoir: { fill: 1, pressure: 48, state: "high" },
    vent: { ...pyState[3].vent, on: false },
  }, 0);
  assert.equal(stubs.get("c-bag-state").textContent, "High · 48 cmH₂O");
  assert.equal(stubs.get("c-bag-state").className, "bad");
  assert.equal(stubs.get("c-o2-flush").disabled, true, "O2 flush disabled when O2 supply disconnected");
  controls.sync({
    ...pyState[3],
    airway: "Mask",
    bag_mask: false,
    o2_connected: true,
    o2_flush: true,
    reservoir: { fill: 0, pressure: 0, state: "empty" },
    vent: { ...pyState[3].vent, on: false },
  }, 0);
  assert.equal(stubs.get("c-bag-state").textContent, "Empty");
  assert.equal(stubs.get("c-o2-flush").disabled, false);
  assert.equal(stubs.get("c-o2-flush").textContent, "O₂ flushing…");

  // O2 fail-safe and 25% O2 guard cap delivered N2O in total FGF and flowmeter bar.
  for (const id of ["fgf-bar-o2", "fgf-bar-air", "fgf-bar-n2o"]) {
    const props = {};
    stubs.set(id, { style: { setProperty(k, v) { props[k] = v; } }, props, dataset: {} });
  }
  controls.sync({
    ...pyState[3],
    fgf: { o2: 6, air: 0, n2o: 4 },
    o2_connected: false,
  }, 0);
  assert.equal(stubs.get("c-total-fgf").textContent, "0.0 L/min");
  assert.equal(stubs.get("fgf-bar-n2o").props["--fill"], "0%");
  controls.sync({
    ...pyState[3],
    fgf: { o2: 1, air: 0, n2o: 6 },
    o2_connected: true,
  }, 0);
  assert.equal(stubs.get("c-total-fgf").textContent, "4.0 L/min");
  assert.equal(stubs.get("fgf-bar-n2o").props["--fill"], "20%");
});

test("defibrillator sync buttons reset per session and hemorrhage select preserves observed rates", () => {
  const pressed = new Map();
  const modes = [false, true].map(value => ({
    dataset: { value: String(value) },
    disabled: false,
    getAttribute(k) { return k === "aria-pressed" ? pressed.get(value) : null; },
    setAttribute(k, v) { if (k === "aria-pressed") pressed.set(value, v); },
  }));
  const elements = new Map([
    ["c-compressions", {}],
    ["c-charge", {}],
    ["c-shock", {}],
    ["c-shock-feedback", { hidden: false }],
    ["c-sync", { querySelectorAll: () => modes }],
  ]);
  globalThis.Option = class {
    constructor(text, value) { this.text = text; this.value = value; this.dataset = {}; }
  };
  const hemOptions = [new Option("500", "500"), new Option("1000", "1000")];
  elements.set("c-hem-rate", {
    value: "500",
    options: hemOptions,
    add(opt) { hemOptions.push(opt); },
    querySelector: () => hemOptions.find(o => o.dataset.observed === "true") ?? null,
  });
  globalThis.document = { getElementById: id => elements.get(id), activeElement: null };
  const controls = Object.create(Controls.prototype);
  controls.bindResuscitation();
  assert.equal(pressed.get(false), "true");
  assert.equal(pressed.get(true), "false");
  modes[1].onclick();
  assert.equal(controls.synchronized, true);
  controls.bindResuscitation();
  assert.equal(controls.synchronized, false);
  assert.equal(pressed.get(false), "true");
  assert.equal(pressed.get(true), "false");

  controls.syncHemorrhageRate(750);
  assert.equal(elements.get("c-hem-rate").value, "750");
  assert.equal(hemOptions.some(o => o.value === "750" && o.dataset.observed === "true"), true);

  const rhythmOptions = [new Option("Sinus rhythm", "Sinus rhythm")];
  for (const opt of rhythmOptions) opt.remove = () => rhythmOptions.splice(rhythmOptions.indexOf(opt), 1);
  globalThis.Option = class {
    constructor(text, value) {
      this.text = text;
      this.value = value;
      this.dataset = {};
      this.remove = () => rhythmOptions.splice(rhythmOptions.indexOf(this), 1);
    }
  };
  elements.set("c-rhythm", {
    value: "Sinus rhythm",
    options: rhythmOptions,
    add(opt) { rhythmOptions.push(opt); },
  });
  controls.syncRhythm("Agonal rhythm");
  assert.equal(elements.get("c-rhythm").value, "Agonal rhythm");
  assert.equal(rhythmOptions.some(o => o.value === "Agonal rhythm" && o.dataset.observed === "true"), true);
  // User manually picks Sinus rhythm in the dropdown before snapshot arrives.
  elements.get("c-rhythm").value = "Sinus rhythm";
  controls.syncRhythm("Sinus rhythm");
  assert.equal(rhythmOptions.some(o => o.dataset.observed === "true"), false);
});

test("applyVentMode disables hidden ventilator inputs so unused invalid values cannot block Apply", () => {
  const tvInput = { disabled: false };
  const psInput = { disabled: false };
  const rows = [
    { dataset: { vent: "tv" }, hidden: false, querySelectorAll: () => [tvInput] },
    { dataset: { vent: "p_support" }, hidden: false, querySelectorAll: () => [psInput] },
  ];
  const labels = new Map([
    ["c-rr-label", { textContent: "" }],
    ["c-pinsp-label", { textContent: "" }],
  ]);
  globalThis.document = {
    getElementById: id => labels.get(id),
    querySelectorAll: sel => (sel === "#c-vent-settings [data-vent]" ? rows : []),
  };
  const controls = Object.create(Controls.prototype);
  controls.ventEditor = { pending: false };
  controls.applyVentMode("CPAP");
  assert.equal(rows[0].hidden, true);
  assert.equal(tvInput.disabled, true);
  assert.equal(psInput.disabled, true);
  controls.applyVentMode("PSV");
  assert.equal(rows[0].hidden, true);
  assert.equal(tvInput.disabled, true);
  assert.equal(rows[1].hidden, false);
  assert.equal(psInput.disabled, false);
});

test("performAction does not re-enable shock after status becomes idle or session ends", async () => {
  const elements = new Map();
  for (const id of ["c-compressions", "c-charge", "c-shock", "c-shock-feedback", "c-defibrillator-status"]) {
    elements.set(id, { id, disabled: false, hidden: false, textContent: "" });
  }
  const modes = [false, true].map(value => ({
    dataset: { value: String(value) }, disabled: false,
    getAttribute() {}, setAttribute() {},
  }));
  elements.set("c-sync", { querySelectorAll: () => modes });
  globalThis.document = { getElementById: id => elements.get(id) };
  const controls = Object.create(Controls.prototype);
  controls.ended = false;
  controls.state = { defibrillator: DEFIB_STATES[2] };
  controls.send = async () => {
    controls.state = { defibrillator: DEFIB_STATES[0] };
    controls.syncDefibrillator(DEFIB_STATES[0]);
  };
  await controls.performAction(elements.get("c-shock"), "shock", { synchronized: false });
  assert.equal(elements.get("c-shock").disabled, true);

  const genericBtn = { id: "c-generic", disabled: false };
  controls.send = async () => {
    controls.ended = true;
    genericBtn.disabled = true;
  };
  await controls.performAction(genericBtn, "fluid", { kind: "crystalloid", volume_ml: 250 });
  assert.equal(genericBtn.disabled, true);
});

test("blood gas H/L flags match displayed rounded values and strict upper bounds", () => {
  const makeEl = (tag) => ({
    tag, className: "", textContent: "", children: [],
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
  });
  const tHead = makeEl("thead");
  const tBody = makeEl("tbody");
  const table = { tHead, tBodies: [tBody] };
  const elements = new Map([
    ["abg-pending", { hidden: false, textContent: "" }],
    ["abg-results", { hidden: false, querySelector: () => table }],
  ]);
  globalThis.document = {
    getElementById: (id) => elements.get(id),
    createElement: makeEl,
  };
  const controls = Object.create(Controls.prototype);
  controls.time = 120;
  controls.shownBloodGases = null;
  controls.bloodGases = [{
    drawn_at: 0, ready_at: 120,
    ph: 7.348, pa_co2: 45.4, pao2: 100, hco3: 24.0, base_excess: 0.0,
    lactate: 1.98, sodium: 140, potassium: 4.0, chloride: 104,
    ionized_calcium: 1.20, sao2: 99.0, hb: 13.5, fio2: 0.5,
  }];
  controls.showBloodGases();
  const rowByName = new Map(tBody.children.map((tr) => [tr.children[0].textContent, tr.children[2]]));
  assert.equal(rowByName.get("pH").textContent, "7.35");
  assert.equal(rowByName.get("pH").className, "");
  assert.equal(rowByName.get("PaCO₂").textContent, "45");
  assert.equal(rowByName.get("PaCO₂").className, "");
  assert.equal(rowByName.get("Lactate").textContent, "2.0");
  assert.equal(rowByName.get("Lactate").className, "abg-high");
  assert.equal(rowByName.get("Lactate").children[0].textContent, "H");
});

test("defibrillator sync mode auto-resets to unsynchronized after shock delivery", () => {
  const pressed = new Map();
  const modes = [false, true].map(value => ({
    dataset: { value: String(value) },
    disabled: false,
    getAttribute(k) { return k === "aria-pressed" ? pressed.get(value) : null; },
    setAttribute(k, v) { if (k === "aria-pressed") pressed.set(value, v); },
  }));
  const badge = { hidden: true };
  const elements = new Map([
    ["c-compressions", {}],
    ["c-charge", { textContent: "" }],
    ["c-shock", { disabled: false }],
    ["c-shock-feedback", { hidden: true, textContent: "" }],
    ["c-defibrillator-status", { textContent: "" }],
    ["defib-sync-badge", badge],
    ["c-sync", { querySelectorAll: () => modes }],
  ]);
  globalThis.document = { getElementById: id => elements.get(id) ?? null };
  const controls = Object.create(Controls.prototype);
  controls.bindResuscitation();
  modes[1].onclick();
  assert.equal(controls.synchronized, true);
  assert.equal(badge.hidden, false);

  controls.syncDefibrillator({ status: "waiting_for_r", synchronized: true, charge_remaining: 0, last_shock: null });
  assert.equal(controls.synchronized, true);
  controls.syncDefibrillator({
    status: "idle",
    synchronized: false,
    charge_remaining: 0,
    last_shock: { time: 12.5, text: "Synchronized shock 200 J delivered" },
  });
  assert.equal(controls.synchronized, false);
  assert.equal(pressed.get(false), "true");
  assert.equal(pressed.get(true), "false");
  assert.equal(badge.hidden, true);
});

test("extracorporeal controls and site-tagged blood gas headers update cleanly", async () => {
  const makeEl = (tag = "div") => ({
    tag, value: "", disabled: false, hidden: false, open: false, checked: false, className: "", textContent: "",
    dataset: {}, children: [], attrs: {},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
    getAttribute(k) { return this.attrs[k] ?? null; },
    setAttribute(k, v) { this.attrs[k] = v; },
    closest: () => ({}),
    querySelectorAll: () => [],
  });
  const tHead = makeEl("thead");
  const tBody = makeEl("tbody");
  const table = { tHead, tBodies: [tBody] };
  const elements = new Map([
    ["ecmo-module", makeEl("details")],
    ["ecmo-summary", makeEl("output")],
    ["ecmo-start-panel", makeEl("div")],
    ["ecmo-active-panel", makeEl("div")],
    ["ecmo-readout", makeEl("p")],
    ["c-abg-site-row", makeEl("div")],
    ["c-abg-site", { ...makeEl("select"), value: "right_radial" }],
    ["c-abg", makeEl("button")],
    ["abg-pending", makeEl("p")],
    ["abg-results", { ...makeEl("div"), querySelector: () => table }],
    ["c-ecmo-flow", makeEl("input")],
    ["c-ecmo-sweep", makeEl("input")],
    ["c-ecmo-fio2", makeEl("input")],
    ["c-ecmo-water", makeEl("input")],
    ["c-ecmo-lv-vent", makeEl("input")],
    ["c-ecmo-sevo", makeEl("input")],
    ["c-ecmo-venous-clamp", makeEl("input")],
    ["c-ecmo-bg-strategy", makeEl("select")],
    ["ecmo-lv-vent-row", makeEl("div")],
    ["ecmo-sevo-row", makeEl("div")],
    ["ecmo-venous-row", makeEl("div")],
    ["ecmo-bg-row", makeEl("div")],
    ["ecmo-cpb-controls", makeEl("div")],
    ["c-ecmo-hc", makeEl("button")],
  ]);
  const eventSelectors = [];
  globalThis.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
  globalThis.document = {
    activeElement: null,
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, makeEl("div"));
      return elements.get(id);
    },
    createElement: makeEl,
    querySelectorAll: (sel) => {
      eventSelectors.push(sel);
      return sel === ".event-section[data-event]" ? [] : [elements.get("ecmo-module")];
    },
  };
  const sent = [];
  const controls = Object.create(Controls.prototype);
  controls.time = 120;
  controls.shownBloodGases = null;
  controls.bloodGases = [];
  controls.destroyed = false;
  controls.editSettings = () => ({ cancel() {} });
  controls.bindEvents({ disturbances: [], rhythms: [], patient: { gestational_age_weeks: 0 } });
  assert.deepEqual(eventSelectors, [".event-section[data-event]"]);
  assert.equal(controls.eventSections.size, 0);
  controls.performAction = async (_btn, name, args) => {
    sent.push([name, args]);
    return {
      blood_gas: {
        drawn_at: 60, ready_at: 120, site: args?.site ?? "arterial",
        ph: 7.38, pa_co2: 40, pao2: 65, hco3: 24, base_excess: 0,
        lactate: 1.5, sodium: 140, potassium: 4.0, chloride: 104,
        ionized_calcium: 1.2, sao2: 91.0, hb: 12.0, fio2: 0.5,
      },
    };
  };
  controls.bindLabs();
  elements.get("c-abg-site-row").hidden = false;
  elements.get("c-abg-site").value = "right_radial";
  await elements.get("c-abg").onclick();
  assert.deepEqual(sent[0], ["blood_gas", { site: "right_radial" }]);
  assert.equal(tHead.children[0].children[2].textContent, "1:00 (R rad)");

  controls.syncExtracorporeal({
    active: true, mode: "va_ecmo", target_flow: 4.0, flow: 4.0, sweep: 4.0, fio2: 100,
    water_target_c: 37.0, heater_cooler_on: true, lv_vent: 0.5,
    pre_so2: 72.0, reservoir_low: false,
  });
  assert.equal(elements.get("ecmo-module").open, true);
  assert.equal(elements.get("ecmo-start-panel").hidden, true);
  assert.equal(elements.get("ecmo-active-panel").hidden, false);
  assert.equal(elements.get("ecmo-lv-vent-row").hidden, false);
  assert.equal(elements.get("c-ecmo-lv-vent").disabled, false);
  assert.equal(elements.get("c-ecmo-sevo").disabled, true);
  assert.equal(elements.get("c-ecmo-venous-clamp").disabled, true);
  assert.equal(elements.get("c-ecmo-bg-strategy").disabled, true);
  assert.equal(elements.get("ecmo-cpb-controls").hidden, true);
  assert.equal(elements.get("c-abg-site-row").hidden, false);
  assert.match(elements.get("ecmo-readout").textContent, /VA ECMO 4\.00 L\/min · Sweep 4\.0 L\/min · FiO₂ 100% · Pre-O₂ SvO₂ 72%/);
  assert.doesNotMatch(elements.get("ecmo-readout").textContent, /R rad|Fem/);

  // CPB enables CPB-only controls and disables VA-only LV vent input
  controls.syncExtracorporeal({
    active: true, mode: "cpb", target_flow: 4.5, flow: 4.5, sweep: 3.0, fio2: 80,
    water_target_c: 37.0, heater_cooler_on: true, sevo: 1.5, venous_clamp: 100,
    blood_gas_strategy: "alpha_stat", reservoir_ml: 900, reservoir_low: false,
  });
  assert.equal(elements.get("c-ecmo-lv-vent").disabled, true);
  assert.equal(elements.get("c-ecmo-sevo").disabled, false);
  assert.equal(elements.get("c-ecmo-venous-clamp").disabled, false);
  assert.equal(elements.get("c-ecmo-bg-strategy").disabled, false);

  // Stopping support hides and resets ABG site selector back to arterial
  controls.syncExtracorporeal({ active: false, mode: "none" });
  assert.equal(elements.get("c-abg-site-row").hidden, true);
  assert.equal(elements.get("c-abg-site").value, "arterial");

  // ACT measurement display in Labs tab
  const actBody = makeEl("tbody");
  elements.set("c-act", makeEl("button"));
  elements.set("act-pending", makeEl("p"));
  elements.set("act-results", { ...makeEl("div"), querySelector: () => ({ tBodies: [actBody] }) });
  controls.acts = [];
  controls.shownActs = null;
  controls.performAction = async (_btn, name, args) => {
    sent.push([name, args]);
    return { act: { drawn_at: 60, ready_at: 120, act_s: 512 } };
  };
  controls.bindLabs();
  await elements.get("c-act").onclick();
  assert.equal(sent.at(-1)[0], "act");
  assert.equal(actBody.children.length, 1);
  assert.equal(actBody.children[0].children[1].textContent, "512");
  assert.equal(actBody.children[0].children[1].className, "abg-high");
});
