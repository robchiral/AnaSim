import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { Controls } from "../anasim/web_assets/controls.js";

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
  const states = JSON.parse(execFileSync("python3", ["-B", "-c", `
import json
from anasim.web import WebSession
s = WebSession(dict(age=40,weight=70,height=170,sex="male",mode="awake"))
states = []
def save(): states.append(s.snapshot()["controls"]["defibrillator"])
save()
s.command("charge_defibrillator")
save()
s.engine.start()
for _ in range(500): s.engine.step(.01)
save()
s.command("shock", '{"synchronized": true}')
save()
for _ in range(100): s.engine.step(.01)
save()
print(json.dumps(states))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));
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

  const pyState = JSON.parse(execFileSync("python3", ["-B", "-c", `
import json
from anasim.web import WebSession
s = WebSession(dict(age=40, weight=70, height=170, sex="male", mode="awake", tci_enabled=True))
s.command("drug_target", '{"key": "propofol", "target": 3.0}')
on_state = s.snapshot()["controls"]["drugs"]["propofol"]
s.command("drug_target", '{"key": "propofol", "target": 0}')
zero_controls = s.snapshot()["controls"]
zero_state = zero_controls["drugs"]["propofol"]
print(json.dumps([on_state["is_tci"], zero_state["is_tci"], zero_state["rate"], zero_controls]))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));
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
});
