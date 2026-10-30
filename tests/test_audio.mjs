import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { MonitorAudio } from "../anasim/web_assets/audio.js";

function setup(t) {
  let milliseconds = 0;
  const timers = new Map();
  const elements = new Map(["sound-toggle", "alarm-silence", "alarm-countdown", "sound-status", "pulse-volume",
    "alarm-volume", "pulse-volume-value", "alarm-volume-value"].map(id => [id, {
    textContent: "", hidden: false, value: id === "pulse-volume" ? "35" : "45",
    attributes: {}, setAttribute(name, value) { this.attributes[name] = value; },
  }]));
  const document = Object.assign(new EventTarget(), { hidden: false, getElementById: id => elements.get(id) });
  const contexts = [];
  class AudioContext {
    constructor() {
      this.currentTime = milliseconds / 1000;
      this.state = "suspended";
      this.destination = {};
      this.oscillators = [];
      contexts.push(this);
    }
    async resume() { this.state = "running"; }
    createGain() {
      return {
        gain: { value: 1, setValueAtTime(value) { this.value = value; }, linearRampToValueAtTime() {} },
        connect(destination) { this.destination = destination; }, disconnect() {},
      };
    }
    createOscillator() {
      const oscillator = {
        frequency: { value: 0 }, connected: true,
        connect(envelope) { this.envelope = envelope; },
        disconnect() { this.connected = false; },
        start(time) { this.startTime = time; },
        stop(time = milliseconds / 1000) { this.stopTime = time; },
      };
      this.oscillators.push(oscillator);
      return oscillator;
    }
  }
  const saved = {};
  for (const [name, value] of Object.entries({ document, window: new EventTarget(), AudioContext })) {
    saved[name] = globalThis[name];
    globalThis[name] = value;
  }
  t.after(() => {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete globalThis[name];
      else globalThis[name] = value;
    }
  });
  t.mock.method(performance, "now", () => milliseconds);
  t.mock.method(globalThis, "setInterval", callback => { const id = Symbol(); timers.set(id, callback); return id; });
  t.mock.method(globalThis, "clearInterval", id => timers.delete(id));
  const audio = new MonitorAudio();
  audio.resetSession(0.05);
  const notes = channel => contexts.flatMap(context => context.oscillators)
    .filter(note => note.envelope.destination === audio.channels[channel]);
  function advance(ms) {
    milliseconds += ms;
    for (const context of contexts) {
      context.currentTime = milliseconds / 1000;
      for (const note of context.oscillators) {
        if (note.stopTime <= context.currentTime && !note.ended) { note.ended = true; note.onended(); }
      }
    }
    for (const callback of timers.values()) callback();
  }
  const snap = (time = milliseconds / 1000, changes = {}) => ({
    time, running: true, ended: false, speed: 1, alarms: {},
    vitals: { spo2: 98 }, waves: { pleth: [] }, ...changes,
  });
  return { audio, elements, contexts, notes, advance, snap, document, timers };
}

const pleths = JSON.parse(execFileSync("python3", ["-c", `
import json
import numpy as np
from anasim.core.enums import RhythmType
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.spo2 import SpO2Monitor
result = []
for rhythm, hr, perfusion in [(RhythmType.SINUS, 70, 1), (RhythmType.AFIB, 95, .12)]:
    cycle = CardiacCycle(rng=np.random.default_rng(7))
    monitor = SpO2Monitor()
    samples, beats = [], []
    for index in range(1, 1001):
        time = index * .01
        beat = cycle.step(.01, hr, rhythm)
        value, _ = monitor.step(.01, beat, 98, perfusion=perfusion)
        if beat.beat_started:
            beats.append(time + monitor.peripheral_delay_s)
        if index % 5 == 0:
            samples.append([time, value])
    result.append({"samples": samples, "beats": [time for time in beats if 1 < time < 9.5]})
print(json.dumps(result))
`], { cwd: new URL("..", import.meta.url), encoding: "utf8" }));

test("sound starts off and follows modeled pulses, including low-perfusion AF and the dicrotic notch", async t => {
  const { audio, contexts, notes, advance, snap } = setup(t);
  audio.update(snap());
  assert.equal(contexts.length, 0);
  assert.equal(audio.toggle.attributes["aria-label"], "Enable sound");
  await audio.setEnabled(true);
  assert.equal(audio.toggle.attributes["aria-label"], "Mute sound");
  for (const fixture of pleths) {
    audio.resetSession(0.05);
    const start = notes("pulse").length;
    const origin = performance.now() / 1000;
    audio.update(snap(0));
    for (const [time, value] of fixture.samples) {
      advance(50);
      audio.update(snap(time, { waves: { pleth: [value] } }));
    }
    const times = notes("pulse").slice(start).map(note => note.startTime - origin).filter(time => time > 1 && time < 9.5);
    assert.equal(times.length, fixture.beats.length, "one tone per pulse, including notch-bearing beats");
    times.forEach((time, index) => assert.ok(Math.abs(time - fixture.beats[index]) < 0.12));
  }
});

test("pitch falls with saturation, signal loss stops pulses, and fast playback cannot queue a burst", async t => {
  const { audio, notes, advance, snap, elements } = setup(t);
  await audio.setEnabled(true);
  audio.update(snap(0));
  function pulse(time, spo2 = 98, speed = 1) {
    advance(300);
    audio.update(snap(time, { speed, vitals: { spo2 }, waves: { pleth: [1, 0.5, 0, 0.2, 1] } }));
  }
  pulse(0.25);
  const highPitch = notes("pulse").at(-1).frequency.value;
  pulse(0.5, 85);
  assert.ok(notes("pulse").at(-1).frequency.value < highPitch);
  const active = notes("pulse").at(-1);
  audio.update(snap(0.5, { vitals: { spo2: null } }));
  assert.equal(active.connected, false);
  const count = notes("pulse").length;
  pulse(0.75, null);
  assert.equal(notes("pulse").length, count);
  for (let index = 1; index <= 30; index++) {
    advance(50);
    audio.update(snap(0.75 + index * 2.5, {
      speed: 50, waves: { pleth: Array.from({ length: 50 }, (_, i) => [1, 0.5, 0, 0.2, 1][i % 5]) },
    }));
  }
  const fast = notes("pulse").slice(count);
  assert.ok(fast.length > 3 && fast.length <= 6);
  fast.slice(1).forEach((note, index) => assert.ok(note.startTime - fast[index].startTime >= 0.25));
  elements.get("pulse-volume").value = "0";
  elements.get("pulse-volume").oninput();
  pulse(76);
  assert.equal(notes("pulse").length, count + fast.length);
  assert.equal(elements.get("pulse-volume-value").textContent, "0%");
});

test("alarms use existing flags, clear promptly, and silence expires after two real minutes at any speed", async t => {
  const { audio, notes, advance, snap, elements } = setup(t);
  await audio.setEnabled(true);
  const alarm = () => snap(undefined, { speed: 50, alarms: { SpO2: "low", MAP: "low" } });
  audio.update(alarm());
  assert.equal(notes("alarm").length, 3, "simultaneous flags share a single alarm pattern");
  assert.equal(elements.get("alarm-silence").hidden, false);
  elements.get("alarm-silence").onclick();
  assert.ok(notes("alarm").every(note => !note.connected), "silence cancels future notes in the pattern");
  for (let second = 1; second < 120; second++) {
    advance(1000);
    audio.update(alarm());
  }
  assert.equal(notes("alarm").length, 3);
  assert.equal(elements.get("alarm-countdown").textContent, "0:01");
  advance(1000);
  audio.update(alarm());
  assert.equal(notes("alarm").length, 6);
  audio.update(snap());
  assert.ok(notes("alarm").every(note => !note.connected || note.ended));
  assert.equal(elements.get("alarm-silence").hidden, true);
  elements.get("alarm-volume").value = "0";
  elements.get("alarm-volume").oninput();
  audio.update(alarm());
  assert.equal(notes("alarm").length, 6);
  assert.ok(audio.channels.pulse.gain.value > 0, "alarm volume is independent of pulse volume");
});

test("the compact alarm control appears only for running alarms and retains silence across pause", async t => {
  const { audio, elements, advance, snap } = setup(t);
  await audio.setEnabled(true);
  const alarm = () => snap(undefined, { alarms: { SpO2: "low" } });
  audio.update(alarm());
  assert.equal(audio.silence.hidden, false);
  assert.equal(audio.countdown.hidden, true);
  assert.equal(audio.silence.attributes["aria-label"], "Silence alarms for 2 minutes");
  audio.silence.onclick();
  assert.equal(audio.countdown.textContent, "2:00");
  audio.update({ ...alarm(), running: false });
  assert.equal(audio.silence.hidden, true);
  advance(1000);
  audio.update(alarm());
  assert.equal(audio.silence.hidden, false);
  assert.equal(audio.countdown.textContent, "1:59");
  assert.match(audio.silence.attributes["aria-label"], /Resume alarms/);
  audio.update(snap());
  assert.equal(audio.silence.hidden, true, "resolved flags hide the control even while silence remains");
  audio.update(alarm());
  elements.get("alarm-silence").onclick();
  assert.equal(audio.countdown.hidden, true);
  assert.equal(audio.silence.attributes["aria-label"], "Silence alarms for 2 minutes");
});

test("pause, end, hidden pages, stale data, errors, and new sessions stop scheduled audio", async t => {
  const { audio, notes, advance, snap, document, timers } = setup(t);
  await audio.setEnabled(true);
  const alarm = () => snap(undefined, { alarms: { HR: "high" } });
  for (const stop of [
    () => audio.update({ ...alarm(), running: false }),
    () => audio.update({ ...alarm(), ended: true }),
    () => { document.hidden = true; document.dispatchEvent(new Event("visibilitychange")); },
    () => advance(1001),
    () => audio.resetSession(0.05),
    () => audio.stopPlayback(),
  ]) {
    document.hidden = false;
    audio.update(alarm());
    assert.ok(audio.notes.size > 0);
    stop();
    assert.equal(audio.notes.size, 0);
    const count = notes("alarm").length;
    advance(5000);
    assert.equal(notes("alarm").length, count, "stopped audio never catches up");
  }
  audio.update(alarm());
  await audio.setEnabled(false);
  assert.equal(audio.notes.size, 0);
  assert.equal(timers.size, 0);
});

test("failed audio activation leaves the simulation usable and can be retried", async t => {
  const { audio, contexts } = setup(t);
  const original = globalThis.AudioContext.prototype.resume;
  globalThis.AudioContext.prototype.resume = async () => { throw new Error("blocked"); };
  await audio.setEnabled(true);
  assert.equal(audio.enabled, false);
  assert.equal(audio.status.hidden, false);
  audio.update({ running: true, ended: false, vitals: { spo2: 98 }, alarms: {}, time: 0, waves: { pleth: [] } });
  globalThis.AudioContext.prototype.resume = original;
  await audio.setEnabled(true);
  assert.equal(audio.enabled, true);
  assert.equal(audio.status.hidden, true);
  assert.equal(contexts.length, 1);
});
