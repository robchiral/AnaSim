import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Monitor } from "../anasim/web_assets/monitor.js";
import { NeuromuscularDisplay } from "../anasim/web_assets/neuromuscular.js";

function setupPage() {
  const page = readFileSync(new URL("../anasim/web_assets/index.html", import.meta.url), "utf8");
  const elements = new Map([...page.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)].map(
    ([, tag, id]) => [id, {
      id, tag, textContent: "", dataset: {}, style: { setProperty() {} }, attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
    }],
  ));
  globalThis.document = {
    getElementById: id => elements.get(id) ?? null,
    querySelector: selector => selector === ".monitor-body" ? { style: { setProperty() {} } } :
      elements.get(selector.slice(1)),
    querySelectorAll: selector => selector === ".monitor output" ?
      [...elements.values()].filter(el => el.tag === "output") : [],
  };
  globalThis.window = { devicePixelRatio: 1 };
  globalThis.ResizeObserver = class {
    constructor(callback) { this.callback = callback; }
    observe() {}
    disconnect() {}
  };
  const frames = new Map();
  let nextFrame = 0;
  globalThis.requestAnimationFrame = callback => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = id => frames.delete(id);
  return { elements, frames, drawFrame() {
    assert.equal(frames.size, 1);
    const [id, callback] = frames.entries().next().value;
    frames.delete(id);
    callback();
  } };
}

test("monitor renders sampled TOF counts, qualified ratios, and selected single twitch", () => {
  const { elements } = setupPage();

  const { info, snapshots } = JSON.parse(execFileSync("python3", ["-c", `
import json
from anasim.monitors.neuromuscular import NeuromuscularMonitor
from anasim.web import WebSession
session = WebSession({"age": 40, "weight": 70, "height": 170, "sex": "male"})
snapshots = []
for mode, t1, ratio, baseline in [("tof", 1, 1, True), ("tof", .2, .09, True),
                                  ("tof", .1, 1, True), ("single", .125, 1, True),
                                  ("single", .4, 1, False)]:
    monitor = NeuromuscularMonitor()
    if baseline:
        monitor.establish_baseline(1)
    monitor.set_mode(mode, 0)
    for time in [0, .5, 1, 1.5]:
        monitor.step(time, t1, ratio)
    session.engine.nmb = monitor
    session.engine.state.time = 1.5
    snapshots.append(session.snapshot())
print(json.dumps({"info": json.loads(session.info()), "snapshots": snapshots}))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));
  const monitor = new Monitor(info);
  monitor.update(snapshots[0]);
  assert.equal(elements.get("nmb-title").textContent, "TOF ratio");
  assert.equal(elements.get("nmb-value").textContent, "100%");
  assert.equal(elements.get("nmb-result").dataset.quality, "recovered");

  monitor.update(snapshots[1]);
  assert.equal(elements.get("nmb-title").textContent, "TOF count");
  assert.equal(elements.get("nmb-value").textContent, "2/4");
  assert.equal(elements.get("nmb-detail").textContent, "");

  monitor.update(snapshots[2]);
  assert.equal(elements.get("nmb-value").textContent, "4/4");
  assert.equal(elements.get("nmb-detail").textContent, "Low signal, ratio 100%");
  assert.equal(elements.get("nmb-result").dataset.quality, "low");

  monitor.update(snapshots[3]);
  assert.equal(elements.get("nmb-title").textContent, "T1");
  assert.equal(elements.get("nmb-value").textContent, "13%");
  assert.equal(elements.get("nmb-t2").hidden, true);

  monitor.update(snapshots[4]);
  assert.equal(elements.get("nmb-value").textContent, "--");
  assert.equal(elements.get("nmb-detail").textContent, "Baseline unavailable");
  monitor.destroy();
});

test("monitor blanks arterial systolic and diastolic numerics while retaining static MAP in arrest", () => {
  const { elements } = setupPage();
  const alarmClasses = new Set();
  elements.get("n-art").classList = {
    toggle(name, force) { force ? alarmClasses.add(name) : alarmClasses.delete(name); },
  };
  globalThis.document.querySelectorAll = selector => {
    if (selector === ".monitor output") return [...elements.values()].filter(el => el.tag === "output");
    if (selector === ".monitor [data-alarm]") return [elements.get("n-art")];
    return [];
  };

  const { info, cpr, stopped } = JSON.parse(execFileSync("python3", ["-c", `
import json
from anasim.web import WebSession
session = WebSession({"age": 40, "weight": 70, "height": 170, "sex": "male", "mode": "steady_state"})
session.command("run", json.dumps({"running": True}))
session.engine.set_rhythm("VFIB")
for _ in range(150):
    session.step(0.1)
session.command("compressions", json.dumps({"active": True}))
for _ in range(100):
    session.step(0.1)
cpr = session.snapshot()
session.command("compressions", json.dumps({"active": False}))
for _ in range(100):
    session.step(0.1)
stopped = session.snapshot()
print(json.dumps({"info": json.loads(session.info()), "cpr": cpr, "stopped": stopped}))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));

  const monitor = new Monitor(info);
  monitor.update(cpr);
  assert.match(elements.get("v-art").textContent, /^\d+\/\d+$/);

  monitor.update(stopped);
  assert.deepEqual(stopped.vitals.art.slice(0, 2), [null, null]);
  assert.equal(elements.get("v-art").textContent, "--/--");
  assert.match(elements.get("v-art-map").textContent, /^\d+$/);
  assert.equal(alarmClasses.has("alarm-low"), true);
  monitor.destroy();
});

test("monitor coalesces updates and redraws resized loops while paused without an idle frame loop", () => {
  const { frames, drawFrame } = setupPage();
  const monitor = new Monitor({
    arterial_line: true, sample_interval: 0.01, nmb_history: [],
    patient: { age: 40, sex: "male", weight: 70, renal_status: "Normal", hepatic_status: "Normal",
      coronary_flow_reserve: 5, heart_failure: "none", surgical_site: "other" },
  });
  drawFrame();
  assert.equal(frames.size, 0);
  monitor.updateNumerics = () => {};
  monitor.updateAlarms = () => {};
  monitor.nmb.update = () => {};
  let draws = 0;
  monitor.sweeps.ecg = {
    write() {}, draw() { draws++; }, observer: { disconnect() {} }, ctx: { clearRect() {} },
  };
  const snap = { waves: { ecg: [] }, loop: [], vitals: { nibp_cuff: null }, ended: false };
  monitor.update(snap);
  assert.equal(frames.size, 0, "unchanged snapshots must leave the renderer idle");
  for (let i = 0; i < 2; i++) monitor.update({ ...snap, waves: { ecg: [0.1] } });
  drawFrame();
  assert.equal(draws, 1, "updates before the next frame share one redraw");
  assert.equal(frames.size, 0);

  monitor.update({ ...snap, loop: [{ breath: 1, paw: [5, 8], flow: [0, 10], volume: [0, 100] }] });
  drawFrame();
  assert.equal(monitor.loops.dirty, false);
  monitor.loops.observer.callback();
  assert.equal(monitor.loops.dirty, true);
  drawFrame();
  assert.equal(frames.size, 0, "a paused resize must draw once");

  monitor.update({ ...snap, waves: { ecg: [0.2] } });
  monitor.destroy();
  assert.equal(frames.size, 0, "replaced sessions must cancel their pending redraw");
});

test("neuromuscular trends scroll retained readings, expire old results, and preserve mode and signal labels", () => {
  const { elements } = setupPage();
  const group = elements.get("nmb-trend-points");
  let markup = "", replacements = 0;
  Object.defineProperty(group, "innerHTML", {
    get: () => markup,
    set(value) { markup = value; replacements++; },
  });
  const display = new NeuromuscularDisplay([
    [100, "tof", 50, false], [200, "single", 25, false], [300, "tof", 75, true],
  ]);
  const nmb = { mode: "tof", reading: null, measuring: false, next: null,
    baseline: 1, repeat: 20, wait: 0 };
  const positions = () => {
    const offset = Number(group.attributes.transform.match(/translate\(([^ ]+)/)[1]);
    return [...markup.matchAll(/M([^ ]+) ([^h]+)h0/g)].map(([, x, y]) => [Number(x) + offset, Number(y)]);
  };
  display.update(nmb, [], 900, false);
  assert.deepEqual(positions(), [[20, 20], [60, 11.5]]);
  assert.match(markup, /class="low"/);
  display.update(nmb, [], 910, false);
  assert.deepEqual(positions(), [[18, 20], [58, 11.5]]);
  display.update(nmb, [[300, "tof", 75, true]], 910, false);
  assert.equal(replacements, 1, "clock updates and replayed readings must reuse the paths");

  display.update(nmb, [[400, "tof", 90, false]], 920, false);
  assert.equal(positions().length, 3);
  display.update({ ...nmb, mode: "single" }, [], 920, false);
  assert.deepEqual(positions(), [[36, 28.5]]);
  assert.doesNotMatch(markup, /class="low"/);
  assert.equal(elements.get("nmb-trend-label").textContent, "T1, past 15 min");
  assert.equal(elements.get("nmb-trend").attributes["aria-label"], "T1 over the past 15 minutes");
  display.update({ ...nmb, mode: "single" }, [], 1100, false);
  assert.deepEqual(positions(), [[0, 28.5]], "the left boundary remains inclusive");
  display.update({ ...nmb, mode: "single" }, [], 1101, false);
  assert.equal(markup, "", "results older than fifteen minutes must disappear");
  display.update(nmb, [], 1101, false);
  assert.equal(positions().length, 2, "switching modes must retain the other mode's recent readings");
});

test("monitor rounds integer numerics to nearest whole number", () => {
  const { elements } = setupPage();
  const monitor = new Monitor({
    arterial_line: true, sample_interval: 0.01, nmb_history: [],
    patient: { age: 40, sex: "male", weight: 70, renal_status: "Normal", hepatic_status: "Normal",
      coronary_flow_reserve: 5, heart_failure: "none", surgical_site: "other" },
  });
  monitor.update({
    time: 1.0,
    ended: false,
    waves: { ecg: [], pleth: [], capno: [], art: [], paw: [], flow: [], volume: [] },
    loop: [],
    alarms: {},
    vitals: {
      hr: 59.6,
      st_depression_mv: 0.0,
      spo2: 94.6,
      art: [119.6, 79.4, 59.6],
      nibp: [120.4, 80.5, 64.6],
      nibp_age: 0,
      nibp_cycling: false,
      nibp_failed: false,
      nibp_cuff: null,
      etco2: 34.6,
      rr: 11.6,
      rr_source: "airway",
      bis: 80.4,
      temp: 37.0,
      net_fluid: 0,
      fluid_in: 0,
      blood_in: 0,
      urine_out: 0,
      blood_out: 0,
      ppeak: 18.6,
      pplat: 14.4,
      peep: 4.6,
      pmean: 8.6,
      vte: 499.6,
      mv: 6.0,
      cdyn: 49.6,
      fio2: 39.6,
      eto2: 35.6,
      fi_n2o: 0.0,
      et_n2o: 0.0,
      fi_sevo: 2.0,
      et_sevo: 1.9,
      et_mac: 1.0,
    },
    nmb: { mode: "tof", reading: null, measuring: false, next: null, baseline: 1, repeat: 0, wait: 0 },
    nmb_trend: [],
  });
  assert.equal(elements.get("v-hr").textContent, "60");
  assert.equal(elements.get("v-spo2").textContent, "95");
  assert.equal(elements.get("v-art").textContent, "120/79");
  assert.equal(elements.get("v-art-map").textContent, "60");
  assert.equal(elements.get("v-etco2").textContent, "35");
  assert.equal(elements.get("v-bis").textContent, "80");
  monitor.destroy();
});

test("ecg sweep autoscales large CPR artifact and clamps drawn points inside lane bounds", () => {
  setupPage();
  const drawnY = [];
  const ctx = {
    setTransform() {}, clearRect() {}, beginPath() {}, stroke() {}, fillText() {},
    moveTo(_x, y) { drawnY.push(y); }, lineTo(_x, y) { drawnY.push(y); },
  };
  const canvas = { clientWidth: 200, clientHeight: 80, width: 200, height: 80, getContext: () => ctx };
  globalThis.document.querySelectorAll = selector => {
    if (selector === ".wave") {
      return ["ecg", "pleth", "art", "co2", "paw", "flow"].map(name => ({
        dataset: { wave: name }, hidden: false, querySelector: () => canvas,
      }));
    }
    return [];
  };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#0f0" });
  const monitor = new Monitor({
    arterial_line: true, sample_interval: 0.01, nmb_history: [],
    patient: { age: 40, sex: "male", weight: 70, renal_status: "Normal", hepatic_status: "Normal",
      coronary_flow_reserve: 5, heart_failure: "none", surgical_site: "other" },
  });
  const ecg = monitor.sweeps.ecg;
  ecg.resize();
  drawnY.length = 0;
  ecg.write([-1.2, 2.4]);
  assert.ok(ecg.range[0] <= -1.0 && ecg.range[1] >= 2.5);
  ecg.draw();
  assert.ok(drawnY.length >= 2 && drawnY.every(y => y >= 22 && y <= 78));
  monitor.destroy();
});
