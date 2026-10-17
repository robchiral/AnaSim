import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Monitor } from "../anasim/web_assets/monitor.js";

test("monitor renders sampled TOF counts, qualified ratios, and selected single twitch", () => {
  const page = readFileSync(new URL("../anasim/web_assets/index.html", import.meta.url), "utf8");
  const elements = new Map([...page.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)].map(
    ([, tag, id]) => [id, { id, tag, textContent: "", dataset: {}, style: { setProperty() {} }, setAttribute() {} }],
  ));
  globalThis.document = {
    getElementById: id => elements.get(id) ?? null,
    querySelector: selector => selector === ".monitor-body" ? { style: { setProperty() {} } } :
      elements.get(selector.slice(1)),
    querySelectorAll: selector => selector === ".monitor output" ?
      [...elements.values()].filter(el => el.tag === "output") : [],
  };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  globalThis.requestAnimationFrame = () => 1;
  globalThis.cancelAnimationFrame = () => {};

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
  assert.equal(elements.get("nmb-detail").textContent, "(100%) · Low signal");
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
