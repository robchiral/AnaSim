// Optional monitor tones, driven by the displayed pleth and alarm flags.
const PULSE_GAP_S = 0.25;
const ALARM_REPEAT_S = 5;
const SILENCE_MS = 120_000;
const STALE_MS = 1000;

export class MonitorAudio {
  constructor() {
    const $ = (id) => document.getElementById(id);
    this.toggle = $("sound-toggle");
    this.silence = $("alarm-silence");
    this.countdown = $("alarm-countdown");
    this.status = $("sound-status");
    this.sliders = { pulse: $("pulse-volume"), alarm: $("alarm-volume") };
    this.outputs = { pulse: $("pulse-volume-value"), alarm: $("alarm-volume-value") };
    this.enabled = false;
    this.context = null;
    this.channels = {};
    this.notes = new Map();
    this.timer = null;
    this.resetSession();

    this.toggle.disabled = !globalThis.AudioContext;
    if (this.toggle.disabled) this.toggle.title = "Sound is unavailable in this browser.";
    this.toggle.onclick = () => this.setEnabled(!this.enabled);
    this.silence.onclick = () => {
      this.silencedUntil = this.silencedUntil > performance.now() ? 0 : performance.now() + SILENCE_MS;
      this.stopNotes("alarm");
      this.nextAlarm = 0;
      this.refresh();
    };
    for (const [name, slider] of Object.entries(this.sliders)) {
      slider.oninput = () => {
        this.outputs[name].textContent = `${slider.value}%`;
        this.setVolume(name);
        if (Number(slider.value) === 0) this.stopNotes(name);
        this.refresh();
      };
    }
    document.addEventListener("visibilitychange", () => this.stopPlayback());
    window.addEventListener("pagehide", () => this.stopPlayback());
  }

  async setEnabled(enabled) {
    this.enabled = enabled;
    this.status.hidden = true;
    this.syncControls();
    if (!enabled) {
      clearInterval(this.timer);
      this.timer = null;
      this.stopNotes();
      return;
    }
    try {
      // Create and resume within the sound button's user gesture.
      if (!this.context) {
        this.context = new AudioContext();
        for (const name of Object.keys(this.sliders)) {
          this.channels[name] = this.context.createGain();
          this.channels[name].connect(this.context.destination);
          this.setVolume(name);
        }
        this.context.onstatechange = () => {
          if (this.context.state !== "running") this.stopPlayback();
        };
      }
      await this.context.resume();
      if (!this.enabled) return;
      if (this.timer === null) this.timer = setInterval(() => this.refresh(), 100);
      this.nextAlarm = 0;
      this.refresh();
    } catch {
      this.enabled = false;
      clearInterval(this.timer);
      this.timer = null;
      this.stopNotes();
      this.status.textContent = "Sound could not start. Try enabling sound again.";
      this.status.hidden = false;
      this.syncControls();
    }
  }

  setVolume(name) {
    if (this.context) {
      this.channels[name].gain.setValueAtTime((Number(this.sliders[name].value) / 100) ** 2, this.context.currentTime);
    }
  }

  resetSession(sampleInterval) {
    this.sampleInterval = sampleInterval;
    this.silencedUntil = 0;
    this.stopPlayback();
  }

  stopPlayback() {
    this.snapshot = null;
    this.previous = null;
    this.peak = 0;
    this.rising = false;
    this.lastSampleTime = -Infinity;
    this.lastPulse = -Infinity;
    this.nextAlarm = 0;
    this.stopNotes();
    this.syncControls();
  }

  update(snap) {
    const previousTime = this.snapshot?.time;
    this.snapshot = snap;
    this.receivedAt = performance.now();
    this.refresh();
    if (!this.playing() || snap.vitals.spo2 === null) {
      this.previous = null;
      this.peak = 0;
      this.rising = false;
      this.lastSampleTime = snap.time;
      this.stopNotes("pulse");
      return;
    }

    const values = snap.waves.pleth;
    let pulse = false;
    for (let i = 0; i < values.length; i++) {
      const time = snap.time - (values.length - 1 - i) * this.sampleInterval;
      // Replayed sweeps and delayed batches must not produce old pulse tones.
      if (previousTime === undefined || time <= this.lastSampleTime + 1e-6 ||
          time < snap.time - STALE_MS / 1000 * snap.speed) continue;
      const value = values[i];
      if (time - this.lastSampleTime > this.sampleInterval * 1.5) this.previous = null;
      // The trough must be below the dicrotic notch before another upstroke counts.
      if (this.previous !== null && value > this.previous && !this.rising && this.previous < this.peak * 0.35) {
        pulse = true;
        this.peak = value;
      }
      this.rising = this.previous !== null && value > this.previous;
      this.peak = Math.max(this.peak, value);
      this.previous = value;
      this.lastSampleTime = time;
    }
    const now = this.context.currentTime;
    if (pulse && Number(this.sliders.pulse.value) > 0 && now - this.lastPulse >= PULSE_GAP_S) {
      const saturation = Math.min(100, Math.max(50, snap.vitals.spo2));
      this.tone("pulse", 450 * 2 ** ((saturation - 90) / 30), now, 0.12);
      this.lastPulse = now;
    }
  }

  playing() {
    return this.enabled && this.context?.state === "running" && !document.hidden &&
      this.snapshot?.running && !this.snapshot.ended && performance.now() - this.receivedAt < STALE_MS;
  }

  refresh() {
    this.syncControls();
    if (!this.playing()) {
      this.stopNotes();
      this.nextAlarm = 0;
      return;
    }
    const alarm = Object.keys(this.snapshot.alarms).length > 0 &&
      Number(this.sliders.alarm.value) > 0 && performance.now() >= this.silencedUntil;
    if (!alarm) {
      this.stopNotes("alarm");
      this.nextAlarm = 0;
      return;
    }
    const now = this.context.currentTime;
    if (now >= this.nextAlarm) {
      for (const offset of [0, 0.22, 0.44]) this.tone("alarm", 780, now + offset, 0.14);
      this.nextAlarm = now + ALARM_REPEAT_S;
    }
  }

  syncControls() {
    this.toggle.setAttribute("aria-pressed", String(this.enabled));
    const soundLabel = this.enabled ? "Mute sound" : "Enable sound";
    this.toggle.setAttribute("aria-label", soundLabel);
    this.toggle.title = this.toggle.disabled ? "Sound is unavailable in this browser." : soundLabel;
    const remaining = Math.max(0, Math.ceil((this.silencedUntil - performance.now()) / 1000));
    this.silence.hidden = !this.playing() || Number(this.sliders.alarm.value) === 0 ||
      !Object.keys(this.snapshot.alarms).length;
    this.silence.setAttribute("aria-pressed", String(remaining > 0));
    const countdown = `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, "0")}`;
    this.countdown.hidden = remaining === 0;
    this.setText(this.countdown, remaining ? countdown : "");
    const alarmLabel = remaining ? `Resume alarms (${countdown} remaining)` : "Silence alarms for 2 minutes";
    this.silence.setAttribute("aria-label", alarmLabel);
    this.silence.title = alarmLabel;
  }

  setText(element, text) {
    if (element.textContent !== text) element.textContent = text;
  }

  tone(channel, frequency, time, duration) {
    const oscillator = this.context.createOscillator();
    const envelope = this.context.createGain();
    oscillator.frequency.value = frequency;
    envelope.gain.setValueAtTime(0, time);
    envelope.gain.linearRampToValueAtTime(0.18, time + 0.008);
    envelope.gain.setValueAtTime(0.18, time + duration - 0.025);
    envelope.gain.linearRampToValueAtTime(0, time + duration);
    oscillator.connect(envelope);
    envelope.connect(this.channels[channel]);
    this.notes.set(oscillator, channel);
    oscillator.onended = () => {
      oscillator.disconnect();
      envelope.disconnect();
      this.notes.delete(oscillator);
    };
    oscillator.start(time);
    oscillator.stop(time + duration);
  }

  stopNotes(channel) {
    for (const [oscillator, name] of this.notes) {
      if (channel && channel !== name) continue;
      oscillator.stop();
      // Disconnect immediately, including notes scheduled later in an alarm burst.
      oscillator.disconnect();
      this.notes.delete(oscillator);
    }
  }
}
