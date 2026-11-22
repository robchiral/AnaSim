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
  controls.syncDefibrillator(states[3]);
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
