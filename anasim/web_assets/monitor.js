// Waveform sweeps and monitor numerics.

// Slower respiratory sweeps show several breaths.
const CARDIAC_SWEEP_S = 10;
const RESPIRATORY_SWEEP_S = 20;
const CHANNELS = {
  ecg: { color: "--ecg", range: [-0.5, 1.5], seconds: CARDIAC_SWEEP_S },
  pleth: { color: "--spo2", range: [-0.1, 1.4], seconds: CARDIAC_SWEEP_S },
  art: { color: "--abp", range: [0, 200], ticks: [0, 50, 100, 150, 200], seconds: CARDIAC_SWEEP_S },
  co2: { color: "--co2", range: [0, 60], ticks: [0, 20, 40, 60], seconds: RESPIRATORY_SWEEP_S },
  paw: { color: "--vent", range: [-5, 40], ticks: [0, 20, 40], seconds: RESPIRATORY_SWEEP_S },
  // Wide enough for pressure-control peak flows; zero shows incomplete exhalation.
  flow: { color: "--vent", range: [-90, 90], ticks: [-60, 0, 60], seconds: RESPIRATORY_SWEEP_S },
};
const AXIS_WIDTH = 35;
const GAP_FRACTION = 0.014;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const trunc = (value) => (value === null ? "--" : String(Math.trunc(value)));
const fixed = (value, digits) => (value === null ? "--" : value.toFixed(digits));

class Sweep {
  constructor(container, channel, sampleInterval) {
    this.canvas = container.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.channel = channel;
    this.color = css(channel.color);
    const size = Math.max(2, Math.round(channel.seconds / sampleInterval));
    this.data = new Float32Array(size).fill(NaN);
    this.gap = Math.min(size - 1, Math.max(1, Math.round(GAP_FRACTION * size)));
    this.writeIndex = 0;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const { clientWidth: w, clientHeight: h } = this.canvas;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = w;
    this.height = h;
    this.draw();
  }

  write(values) {
    const size = this.data.length;
    const start = this.writeIndex;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      this.data[(start + i) % size] = v === null ? NaN : v;
    }
    const end = start + values.length;
    for (let i = 0; i < this.gap; i++) this.data[(end + i) % size] = NaN;
    this.writeIndex = end % size;
  }

  draw() {
    const { ctx, width, height, channel, data } = this;
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);
    const [lo, hi] = channel.range;
    const pad = (hi - lo) * 0.03;
    const top = 4;
    const plotH = height - top - 4;
    const y = (v) => top + plotH * (1 - (v - (lo - pad)) / (hi - lo + 2 * pad));

    if (channel.ticks) {
      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const t of channel.ticks) {
        const ty = Math.round(y(t)) + 0.5;
        ctx.strokeStyle = "rgba(240, 244, 248, 0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(AXIS_WIDTH, ty);
        ctx.lineTo(width, ty);
        ctx.stroke();
        ctx.fillStyle = css("--text-dim");
        ctx.fillText(String(t), AXIS_WIDTH - 6, Math.min(Math.max(ty, 7), height - 7));
      }
    }

    const plotW = width - AXIS_WIDTH;
    const step = plotW / (data.length - 1);
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (Number.isNaN(v)) {
        pen = false;
        continue;
      }
      const px = AXIS_WIDTH + i * step;
      const py = y(v);
      if (pen) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
  }
}

// Pressure-volume and flow-volume loops overlay the current and previous breaths.
const LOOP_KEYS = ["paw", "flow", "volume"];
const LOOP_MAX_POINTS = 700; // One sweep of an open breath during apnea

class Loops {
  constructor() {
    this.canvases = [document.getElementById("loop-pv"), document.getElementById("loop-fv")];
    this.previous = null;
    this.current = null;
    this.dirty = true;
    this.observer = new ResizeObserver(() => { this.dirty = true; });
    for (const canvas of this.canvases) this.observer.observe(canvas);
  }

  destroy() {
    this.observer.disconnect();
  }

  update(runs) {
    for (const run of runs ?? []) {
      if (run.breath !== this.current?.breath) {
        if (this.current?.paw.length > 1) this.previous = this.current;
        this.current = { breath: run.breath, paw: [], flow: [], volume: [] };
      }
      for (const key of LOOP_KEYS) {
        const values = this.current[key];
        values.push(...run[key]);
        if (values.length > LOOP_MAX_POINTS) values.splice(0, values.length - LOOP_MAX_POINTS);
      }
      this.dirty = true;
    }
  }

  draw() {
    const [pv, fv] = this.canvases;
    const loops = [this.previous, this.current].filter(Boolean);
    const all = (key) => loops.flatMap((loop) => loop[key]);
    const maxVolume = niceCeil(Math.max(500, ...all("volume")), 250);
    const maxPaw = niceCeil(Math.max(20, ...all("paw")), 10);
    const maxFlow = niceCeil(Math.max(30, ...all("flow").map(Math.abs)), 30);
    const traces = (x, y) => [
      this.previous && { xs: this.previous[x], ys: this.previous[y], alpha: 0.35 },
      this.current && { xs: this.current[x], ys: this.current[y], alpha: 1, head: true },
    ].filter(Boolean);
    plotLoop(pv, traces("paw", "volume"), [-5, maxPaw], [0, maxVolume], "cmH₂O", "mL");
    plotLoop(fv, traces("volume", "flow"), [0, maxVolume], [-maxFlow, maxFlow], "mL", "L/min");
  }
}

function niceCeil(value, step) {
  return Math.ceil(value / step) * step;
}

function plotLoop(canvas, traces, [x0, x1], [y0, y1], xUnit, yUnit) {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  const width = Math.round(w * dpr), height = Math.round(h * dpr);
  if (canvas.width !== width || canvas.height !== height) [canvas.width, canvas.height] = [width, height];
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = 6;
  const px = (x) => pad + (w - 2 * pad) * (x - x0) / (x1 - x0);
  const py = (y) => h - pad - (h - 2 * pad) * (y - y0) / (y1 - y0);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(240, 244, 248, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px(Math.max(x0, 0)), pad);
  ctx.lineTo(px(Math.max(x0, 0)), h - pad);
  ctx.moveTo(pad, py(Math.max(y0, 0)));
  ctx.lineTo(w - pad, py(Math.max(y0, 0)));
  ctx.stroke();
  ctx.fillStyle = css("--text-dim");
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${x1} ${xUnit}`, w - pad, py(Math.max(y0, 0)) - 2);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`${y1} ${yUnit}`, px(Math.max(x0, 0)) + 3, pad);
  const color = css("--vent");
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  ctx.lineJoin = "round";
  for (const { xs, ys, alpha, head } of traces) {
    if (xs.length < 2) continue;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    xs.forEach((x, i) => (i ? ctx.lineTo(px(x), py(ys[i])) : ctx.moveTo(px(x), py(ys[i]))));
    ctx.stroke();
    if (head) {
      // Mark the newest point so the trace reads as live.
      ctx.beginPath();
      ctx.arc(px(xs.at(-1)), py(ys.at(-1)), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

export class Monitor {
  constructor(info) {
    this.arterialLine = info.arterial_line;
    this.dirty = false;
    this.sweeps = {};
    for (const container of document.querySelectorAll(".wave")) {
      const name = container.dataset.wave;
      container.hidden = name === "art" && !this.arterialLine;
      if (!container.hidden) this.sweeps[name] = new Sweep(container, CHANNELS[name], info.sample_interval);
    }
    document.getElementById("n-art").hidden = !this.arterialLine;
    document.getElementById("n-nibp").hidden = this.arterialLine;
    document.querySelector("#n-nibp").dataset.alarm = this.arterialLine ? "" : "MAP";
    document.querySelector("#n-art").dataset.alarm = this.arterialLine ? "MAP" : "";

    const p = info.patient;
    const parts = ["Simulated patient", `${Math.round(p.age)} y`, cap(p.sex), `${p.weight.toFixed(1)} kg`];
    if (p.renal_status.toLowerCase() !== "normal") parts.push(`Renal: ${p.renal_status}`);
    if (p.hepatic_status.toLowerCase() !== "normal") parts.push(`Hepatic: ${p.hepatic_status}`);
    document.getElementById("patient-info").textContent = parts.join("  ·  ");

    this.fields = {};
    for (const el of document.querySelectorAll(".numerics output, .vent-bar output")) this.fields[el.id] = el;
    this.fields["v-spo2-unit"] = document.getElementById("v-spo2-unit");
    this.fields["v-rr-title"] = document.getElementById("v-rr-title");
    this.alarmBoxes = [...document.querySelectorAll(".numeric[data-alarm]")];
    this.loops = new Loops();
    this.frame = requestAnimationFrame(() => this.render());
  }

  destroy() {
    cancelAnimationFrame(this.frame);
    this.loops.destroy();
    for (const sweep of Object.values(this.sweeps)) {
      sweep.observer.disconnect();
      sweep.ctx.clearRect(0, 0, sweep.width, sweep.height);
    }
  }

  update(snap) {
    const waves = snap.waves;
    if (waves.ecg.length) {
      for (const [name, sweep] of Object.entries(this.sweeps)) sweep.write(waves[name]);
      this.dirty = true;
    }
    this.loops.update(snap.loop);
    this.updateNumerics(snap.vitals);
    document.getElementById("measure-nibp").disabled = snap.ended || snap.vitals.nibp_cuff !== null;
    this.updateAlarms(snap.alarms);
  }

  render() {
    if (this.dirty) {
      for (const sweep of Object.values(this.sweeps)) sweep.draw();
      this.dirty = false;
    }
    if (this.loops.dirty) {
      this.loops.draw();
      this.loops.dirty = false;
    }
    this.frame = requestAnimationFrame(() => this.render());
  }

  set(id, text) {
    const el = this.fields[id];
    if (el.textContent !== text) el.textContent = text;
  }

  updateNumerics(v) {
    this.set("v-hr", trunc(v.hr));
    this.set("v-spo2", trunc(v.spo2));
    this.set("v-spo2-unit", v.spo2 === null ? "No signal" : "%");
    if (this.arterialLine) {
      this.set("v-art", `${trunc(v.art[0])}/${trunc(v.art[1])} (${trunc(v.art[2])})`);
    } else if (v.nibp === null) {
      this.set("v-nibp", "--/-- (--)");
    } else {
      this.set("v-nibp", `${trunc(v.nibp[0])}/${trunc(v.nibp[1])} (${trunc(v.nibp[2])})`);
    }
    if (!this.arterialLine) {
      const age = Math.floor(v.nibp_age);
      const reading = v.nibp_age === null ? "No reading" :
        `Last reading ${Math.floor(age / 60)}:${String(age % 60).padStart(2, "0")} ago`;
      const status = v.nibp_cuff !== null ? `Cuff ${trunc(v.nibp_cuff)} mmHg` :
        v.nibp_failed ? "Measurement failed" : "";
      this.set("v-nibp-status", status ? `${status} · ${reading.toLowerCase()}` : reading);
    }
    this.set("v-etco2", trunc(v.etco2));
    this.set("v-rr", trunc(v.rr));
    this.set("v-rr-title", v.rr_source === "impedance" ? "RR imp" : "RR");
    this.set("v-bis", trunc(v.bis));
    this.set("v-tof", `${trunc(v.tof)}%`);
    this.set("v-temp", fixed(v.temp, 1));

    const net = v.net_fluid ?? 0;
    this.set("v-net", `${net >= 0 ? "+" : "−"}${Math.abs(net).toFixed(0)} mL`);
    document.getElementById("v-io").textContent =
      `IV ${fixed(v.fluid_in, 0)}  PRBC ${fixed(v.blood_in, 0)}  ·  Urine ${fixed(v.urine_out, 0)}  Loss ${fixed(v.blood_out, 0)}`;

    this.set("v-ppeak", trunc(v.ppeak));
    this.set("v-pplat", trunc(v.pplat));
    this.set("v-peep", trunc(v.peep));
    this.set("v-pmean", trunc(v.pmean));
    this.set("v-vte", trunc(v.vte));
    this.set("v-mv", fixed(v.mv, 1));
    this.set("v-cdyn", trunc(v.cdyn));
    this.set("v-o2", `${trunc(v.fio2)}/${trunc(v.eto2)}`);
    this.set("v-n2o", `${trunc(v.fi_n2o)}/${trunc(v.et_n2o)}`);
    this.set("v-sevo", `${fixed(v.fi_sevo, 1)}/${fixed(v.et_sevo, 1)}`);
    this.set("v-mac", fixed(v.et_mac, 2));
  }

  updateAlarms(alarms) {
    for (const box of this.alarmBoxes) {
      const name = box.dataset.alarm;
      const level = name ? alarms[name] : undefined;
      box.classList.toggle("alarm-low", level === "low");
      box.classList.toggle("alarm-high", level === "high");
      const title = box.querySelector(".numeric-title");
      if (!title.dataset.base) title.dataset.base = title.textContent;
      const text = level ? `${title.dataset.base} ${level}` : title.dataset.base;
      if (title.textContent !== text) title.textContent = text;
    }
  }
}

function cap(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
