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
