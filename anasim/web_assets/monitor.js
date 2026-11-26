// Waveform sweeps, alarm banner, and bedside monitor numerics.
import { minutesSeconds, setText } from "./display.js";
import { NeuromuscularDisplay } from "./neuromuscular.js";

// Slower respiratory sweeps show several breaths.
const CARDIAC_SWEEP_S = 10;
const RESPIRATORY_SWEEP_S = 20;
const CHANNELS = {
  ecg: { color: "--ecg", range: [-0.5, 1.5], autoScale: "ecg", scaleStep: 0.5, seconds: CARDIAC_SWEEP_S },
  pleth: { color: "--spo2", range: [-0.1, 1.4], seconds: CARDIAC_SWEEP_S },
  art: { color: "--abp", range: [0, 200], autoScale: "expand", scaleStep: 50, seconds: CARDIAC_SWEEP_S },
  co2: { color: "--co2", range: [0, 60], autoScale: "expand", scaleStep: 20, seconds: RESPIRATORY_SWEEP_S },
  paw: { color: "--paw", range: [-5, 40], autoScale: "pressure", seconds: RESPIRATORY_SWEEP_S },
  // Symmetric flow bounds keep zero centered and incomplete exhalation visible.
  flow: { color: "--flow", range: [-90, 90], autoScale: "flow", seconds: RESPIRATORY_SWEEP_S },
};
const AXIS_WIDTH = 36;
const GAP_FRACTION = 0.016;

const ALARM_LABELS = {
  HR: "HR",
  MAP: "MAP",
  SpO2: "SpO₂",
  EtCO2: "EtCO₂",
  RR: "RR",
  Ppeak: "Ppeak",
  MV: "MV",
  VTe: "VTe",
  FiO2: "FiO₂",
  BIS: "BIS",
  Temp: "Temp",
  ReservoirLow: "CPB reservoir",
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const whole = (value) => (value === null ? "--" : String(Math.round(value)));
const fixed = (value, digits) => (value === null ? "--" : value.toFixed(digits));

class Sweep {
  constructor(container, channel, sampleInterval) {
    this.canvas = container.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.channel = channel;
    this.range = [...channel.range];
    this.shrinkSamples = 0;
    this.color = css(channel.color);
    this.labelColor = css("--text-dim");
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
      const sample = values[i];
      this.data[(start + i) % size] = sample === null ? NaN : sample;
    }
    const end = start + values.length;
    for (let i = 0; i < this.gap; i++) this.data[(end + i) % size] = NaN;
    this.writeIndex = end % size;
    this.updateScale(values);
  }

  updateScale(values) {
    const samples = values.length;
    const { channel, data, range } = this;
    if (!channel.autoScale || !samples) return;
    let lo = Infinity, hi = -Infinity;
    const source = channel.autoScale === "expand" ? values : data;
    for (let i = 0; i < source.length; i++) {
      const value = source[i];
      if (value !== null && Number.isFinite(value)) {
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
    }
    if (lo === Infinity) {
      this.shrinkSamples = 0;
      return;
    }

    let desired;
    if (channel.autoScale === "flow") {
      const extent = niceCeil(Math.max(20, -lo, hi), 10);
      desired = [-extent, extent];
    } else if (channel.autoScale === "pressure") {
      const extent = Math.max(5, -lo, hi);
      const step = extent <= 10 ? 1 : extent <= 20 ? 2 : 5;
      desired = [-niceCeil(Math.max(2, -lo), step), niceCeil(Math.max(5, hi), step)];
    } else {
      desired = [Math.min(channel.range[0], -niceCeil(-lo, channel.scaleStep)),
                 Math.max(channel.range[1], niceCeil(hi, channel.scaleStep))];
    }

    if (desired[0] < range[0] || desired[1] > range[1]) {
      // New excursions must fit even while older, larger values remain visible.
      this.range = [Math.min(range[0], desired[0]), Math.max(range[1], desired[1])];
      this.shrinkSamples = 0;
    } else if (channel.autoScale !== "expand" && desired[1] - desired[0] <= 0.75 * (range[1] - range[0])) {
      // Shrink after a full sweep of simulation samples with headroom.
      this.shrinkSamples += samples;
      if (this.shrinkSamples >= data.length) {
        this.range = desired;
        this.shrinkSamples = 0;
      }
    } else {
      this.shrinkSamples = 0;
    }
  }

  draw() {
    const { ctx, width, height, channel, data } = this;
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);
    const [lo, hi] = this.range;
    const pad = (hi - lo) * 0.03;
    const top = 22;
    const plotH = height - top - 6;
    const y = (val) => top + plotH * (1 - (val - (lo - pad)) / (hi - lo + 2 * pad));

    if (channel.autoScale && channel !== CHANNELS.ecg) {
      const ticks = lo < 0 ? [lo, 0, hi] : channel === CHANNELS.art ? [lo, (lo + hi) / 2, hi] : [lo, hi];
      ctx.font = "10px system-ui, -apple-system, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      let lastLabelY = -Infinity;
      for (const t of ticks) {
        const ty = Math.round(y(t)) + 0.5;
        const isZeroLine = channel === CHANNELS.flow && t === 0;
        ctx.strokeStyle = isZeroLine ? "rgba(215, 220, 226, 0.24)" : "rgba(200, 206, 214, 0.09)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(AXIS_WIDTH - 3, ty);
        ctx.lineTo(width - 4, ty);
        ctx.stroke();
        const labelY = Math.min(Math.max(ty, 7), height - 7);
        // Skip crowding an intermediate zero label onto a nearby lower bound (e.g. Paw -2 and 0).
        if (t === 0 && lo < 0 && Math.abs(labelY - lastLabelY) < 11) continue;
        ctx.fillStyle = this.labelColor;
        ctx.fillText(String(t), AXIS_WIDTH - 6, labelY);
        lastLabelY = labelY;
      }
    }

    const plotW = width - AXIS_WIDTH;
    const step = plotW / (data.length - 1);
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.75;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      if (Number.isNaN(val)) {
        pen = false;
        continue;
      }
      const px = AXIS_WIDTH + i * step;
      const py = Math.min(Math.max(y(val), top), height - 2);
      if (pen) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();

    // Crisp write-head dot just before the sweep erase bar.
    const headIndex = (this.writeIndex - 1 + data.length) % data.length;
    const headVal = data[headIndex];
    if (ctx.arc && ctx.fill && !Number.isNaN(headVal)) {
      const hx = AXIS_WIDTH + headIndex * step;
      const hy = Math.min(Math.max(y(headVal), top), height - 2);
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(hx, hy, 1.9, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// Pressure-volume and flow-volume loops overlay the current and previous breaths.
const LOOP_KEYS = ["paw", "flow", "volume"];
const LOOP_MAX_POINTS = 700; // One sweep of an open breath during apnea

class Loops {
  constructor(onResize) {
    this.canvases = ["loop-pv", "loop-fv"].map((id) => document.getElementById(id));
    this.previous = null;
    this.current = null;
    this.ranges = null;
    this.newBreath = true;
    this.dirty = true;
    this.observer = new ResizeObserver(() => {
      this.dirty = true;
      onResize();
    });
    for (const canvas of this.canvases) this.observer.observe(canvas);
  }

  destroy() {
    this.observer.disconnect();
  }

  update(runs) {
    for (const run of runs ?? []) {
      if (run.reset) {
        this.previous = null;
        this.current = null;
        this.ranges = null;
        this.newBreath = true;
        this.dirty = true;
        continue;
      }
      if (run.breath !== this.current?.breath) {
        if (this.current?.paw.length > 1) this.previous = this.current;
        this.current = { breath: run.breath, paw: [], flow: [], volume: [] };
        this.newBreath = true;
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
    let minVolume = 0, maxVolume = 250, minPressure = 0, maxPressure = 2, flowExtent = 10;
    for (const loop of [this.previous, this.current]) {
      if (!loop) continue;
      const { volume, paw, flow } = loop;
      for (let i = 0; i < volume.length; i++) {
        const v = volume[i];
        if (v < minVolume) minVolume = v;
        if (v > maxVolume) maxVolume = v;
        const p = paw[i];
        if (p < minPressure) minPressure = p;
        if (p > maxPressure) maxPressure = p;
        const absFlow = Math.abs(flow[i]);
        if (absFlow > flowExtent) flowExtent = absFlow;
      }
    }
    maxVolume = niceCeil(maxVolume, 250);
    // Plot padding covers small excursions below the breath's starting volume.
    // Larger changes, such as an effort during mandatory expiration, need a negative scale.
    const volumeFloor = minVolume >= -LOOP_PADDING * maxVolume ? 0 : -niceCeil(-minVolume, 250);
    const pressureExtent = Math.max(-minPressure, maxPressure);
    const pressureStep = pressureExtent <= 5 ? 1 : pressureExtent <= 20 ? 2 : 5;
    const minPaw = -niceCeil(-minPressure, pressureStep);
    const maxPaw = niceCeil(maxPressure, pressureStep);
    const maxFlow = niceCeil(flowExtent, 10);
    const desired = { pressure: [minPaw, maxPaw], volume: [volumeFloor, maxVolume], flow: [-maxFlow, maxFlow] };
    // Expand immediately; shrink on a new breath using both retained loops.
    for (const key of Object.keys(desired)) {
      if (this.ranges && !this.newBreath) {
        desired[key] = [Math.min(this.ranges[key][0], desired[key][0]),
                        Math.max(this.ranges[key][1], desired[key][1])];
      }
    }
    this.ranges = desired;
    this.newBreath = false;
    const traces = (x, y) => [
      this.previous && { xs: this.previous[x], ys: this.previous[y], alpha: 0.30 },
      this.current && { xs: this.current[x], ys: this.current[y], alpha: 1, head: true },
    ].filter(Boolean);
    plotLoop(pv, traces("paw", "volume"), desired.pressure, desired.volume, "cmH₂O", "mL", "--paw");
    plotLoop(fv, traces("volume", "flow"), desired.volume, desired.flow, "mL", "L/min", "--flow");
  }
}

function niceCeil(value, step) {
  return Math.ceil(value / step) * step;
}

const LOOP_PADDING = 0.04;
const loopColors = {};

function plotLoop(canvas, traces, [x0, x1], [y0, y1], xUnit, yUnit, traceVar) {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  loopColors.label ??= css("--text-dim");
  const traceColor = (loopColors[traceVar] ??= css(traceVar) || "#DCE4EC");
  const width = Math.round(w * dpr), height = Math.round(h * dpr);
  if (canvas.width !== width || canvas.height !== height) [canvas.width, canvas.height] = [width, height];
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const left = 28, right = w - 8, top = 16, bottom = h - 18;
  const xPad = LOOP_PADDING * (x1 - x0), yPad = LOOP_PADDING * (y1 - y0);
  const px = (x) => left + (right - left) * (x - x0 + xPad) / (x1 - x0 + 2 * xPad);
  const py = (y) => bottom - (bottom - top) * (y - y0 + yPad) / (y1 - y0 + 2 * yPad);

  // Subtle calibrated reference frame.
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(200, 206, 214, 0.09)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(left + 0.5, top + 0.5, right - left - 1, bottom - top - 1);
  ctx.moveTo(left, Math.round((top + bottom) / 2) + 0.5);
  ctx.lineTo(right, Math.round((top + bottom) / 2) + 0.5);
  ctx.moveTo(Math.round((left + right) / 2) + 0.5, top);
  ctx.lineTo(Math.round((left + right) / 2) + 0.5, bottom);
  ctx.stroke();

  // Prominent zero axes.
  ctx.strokeStyle = "rgba(215, 220, 226, 0.24)";
  ctx.beginPath();
  ctx.moveTo(Math.round(px(0)) + 0.5, top);
  ctx.lineTo(Math.round(px(0)) + 0.5, bottom);
  ctx.moveTo(left, Math.round(py(0)) + 0.5);
  ctx.lineTo(right, Math.round(py(0)) + 0.5);
  ctx.stroke();

  ctx.fillStyle = loopColors.label;
  ctx.font = "10px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${x1} ${xUnit}`, right, h - 1);
  ctx.textAlign = "left";
  ctx.fillText(String(x0), left, h - 1);
  ctx.textBaseline = "top";
  ctx.fillText(`${y1} ${yUnit}`, left, 1);
  if (y0 < 0) ctx.fillText(String(y0), 2, bottom - 10);

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  ctx.strokeStyle = ctx.fillStyle = traceColor;
  ctx.lineWidth = 1.7;
  ctx.lineJoin = "round";
  for (const { xs, ys, alpha, head } of traces) {
    if (xs.length < 2) continue;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    xs.forEach((x, i) => (i ? ctx.lineTo(px(x), py(ys[i])) : ctx.moveTo(px(x), py(ys[i]))));
    ctx.stroke();
    if (head) {
      ctx.beginPath();
      ctx.arc(px(xs.at(-1)), py(ys.at(-1)), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
}

function formatLimitPair(minVal, maxVal, digits = 0) {
  const fmt = (num) => (digits ? Number(num).toFixed(digits) : String(Math.round(num)));
  if (minVal !== undefined && maxVal !== undefined) return `${fmt(maxVal)}\n${fmt(minVal)}`;
  if (maxVal !== undefined) return fmt(maxVal);
  if (minVal !== undefined) return fmt(minVal);
  return "";
}

const RHYTHM_ALARMS = [
  ["Asystole", "ASYSTOLE"],
  ["VFIB", "VFIB"],
  ["VTACH", "VTACH"],
  ["Arrest", "CARDIAC ARREST"],
];

function alarmSide(raw) {
  return typeof raw === "string" ? raw : raw?.low ? "low" : raw?.high ? "high" : "";
}

export class Monitor {
  constructor(info) {
    this.arterialLine = info.arterial_line;
    this.dirty = false;
    this.frame = null;
    this.sweeps = {};

    const arterialLane = document.getElementById("art-lane");
    arterialLane.hidden = !this.arterialLine;
    const lanes = [...document.querySelectorAll(".monitor-lane")].filter((lane) => !lane.hidden);
    lanes.forEach((lane, index) => lane.style.setProperty("--lane", index + 1));
    const body = document.querySelector(".monitor-body");
    body.style.setProperty("--lane-count", lanes.length);
    body.style.setProperty("--respiratory-start", lanes.length - 2);
    for (const container of document.querySelectorAll(".wave")) {
      const name = container.dataset.wave;
      container.hidden = name === "art" && !this.arterialLine;
      if (!container.hidden) this.sweeps[name] = new Sweep(container, CHANNELS[name], info.sample_interval);
    }
    document.querySelector("#n-nibp").dataset.alarm = this.arterialLine ? "" : "MAP";
    document.querySelector("#n-art").dataset.alarm = this.arterialLine ? "MAP" : "";

    const p = info.patient;
    const sexTag = p.sex === "female" ? "F" : "M";
    const bmi = p.height ? `BMI ${(p.weight / ((p.height / 100) ** 2)).toFixed(1)}` : "";
    const parts = [];
    if (bmi) parts.push(bmi);
    if (p.renal_status !== "Normal") parts.push(`${p.renal_status.toLowerCase()} renal`);
    if (p.creatinine_clearance_ml_min != null) parts.push(`CrCl ${Math.round(p.creatinine_clearance_ml_min)}`);
    if (p.hepatic_status !== "Normal") parts.push(`${p.hepatic_status.toLowerCase()} hepatic`);
    if (p.coronary_flow_reserve < 5) parts.push("CAD");
    if (p.heart_failure !== "none") parts.push(p.heart_failure === "hfref" ? "HFrEF" : "HFpEF");
    if (p.copd_severity > 0) parts.push("COPD");
    if (p.type2_diabetes) parts.push("T2D");
    if (p.baseline_glucose != null && Math.round(p.baseline_glucose) !== (p.type2_diabetes ? 150 : 90)) {
      parts.push(`BG ${Math.round(p.baseline_glucose)}`);
    }
    if (p.surgical_site !== "other") parts.push(`${p.surgical_site === "upper_abdominal" ? "upper" : "lower"} abd`);
    if (p.gestational_age_weeks > 0) parts.push(`G ${p.gestational_age_weeks} wk`);
    document.getElementById("session-patient-summary").textContent = `${Math.round(p.age)}y ${sexTag} · ${Number(p.weight.toFixed(1))} kg`;
    document.getElementById("session-patient-details").textContent = parts.join(" · ");

    this.populateLimits(info.alarm_thresholds ?? {});

    this.fields = {};
    for (const el of document.querySelectorAll(".monitor output")) this.fields[el.id] = el;
    this.fields["v-spo2-unit"] = document.getElementById("v-spo2-unit");
    this.fields["v-rr-title"] = document.getElementById("v-rr-title");
    this.fields["v-io"] = document.getElementById("v-io");
    this.alarmBoxes = [...document.querySelectorAll(".monitor [data-alarm]")];
    this.alarmBanner = document.getElementById("alarm-banner");
    this.alarmBannerText = document.getElementById("alarm-banner-text");
    this.measureNibpBtn = document.getElementById("measure-nibp");
    this.loops = new Loops(() => this.requestDraw());
    this.nmb = new NeuromuscularDisplay(info.nmb_history);
    this.requestDraw();
  }

  populateLimits(t) {
    const entries = [
      ["lim-hr", formatLimitPair(t.HR_min, t.HR_max)],
      ["lim-map", formatLimitPair(t.MAP_min, t.MAP_max)],
      ["lim-nibp-map", formatLimitPair(t.MAP_min, t.MAP_max)],
      ["lim-spo2", formatLimitPair(t.SpO2_min, t.SpO2_max)],
      ["lim-etco2", formatLimitPair(t.EtCO2_min, t.EtCO2_max)],
      ["lim-rr", formatLimitPair(t.RR_min, undefined)],
      ["lim-ppeak", formatLimitPair(undefined, t.Ppeak_max)],
      ["lim-mv", formatLimitPair(t.MV_min, undefined, 1)],
      ["lim-fio2", t.FiO2_min !== undefined ? `≥${Math.round(t.FiO2_min)}` : ""],
      ["lim-bis", formatLimitPair(t.BIS_min, t.BIS_max)],
      ["lim-temp", formatLimitPair(t.Temp_min, t.Temp_max, 1)],
    ];
    for (const [id, text] of entries) {
      const el = document.getElementById(id);
      if (el) setText(el, text);
    }
  }

  destroy() {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
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
    this.nmb.update(snap.nmb, snap.nmb_trend, snap.time, snap.ended);
    if (this.measureNibpBtn) this.measureNibpBtn.disabled = snap.ended || snap.vitals.nibp_cuff !== null;
    this.updateAlarms(snap.alarms, snap.controls, snap.ended, snap.vitals);
    if (this.dirty || this.loops.dirty) this.requestDraw();
  }

  requestDraw() {
    if (this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.render();
    });
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
  }

  set(id, text) {
    setText(this.fields[id], text);
  }

  updateNumerics(v) {
    this.set("v-hr", whole(v.hr));
    this.set("v-spo2", whole(v.spo2));
    this.set("v-spo2-unit", v.spo2 === null ? "No signal" : "%");
    if (this.arterialLine) {
      this.set("v-art", `${whole(v.art[0])}/${whole(v.art[1])}`);
      this.set("v-art-map", whole(v.art[2]));
    }
    if (v.nibp === null) {
      this.set("v-nibp", "--/-- (--)");
    } else {
      this.set("v-nibp", `${whole(v.nibp[0])}/${whole(v.nibp[1])} (${whole(v.nibp[2])})`);
    }
    const reading = v.nibp_age === null ? "No reading" :
      `${minutesSeconds(v.nibp_age)} ago`;
    const status = v.nibp_cuff !== null ? `Cuff ${whole(v.nibp_cuff)} mmHg` :
      v.nibp_failed ? "Measurement failed" : "";
    this.set("v-nibp-status", status ? `${status}, ${reading.toLowerCase()}` : reading);
    this.set("v-etco2", whole(v.etco2));
    this.set("v-rr", whole(v.rr));
    this.set("v-rr-title", v.rr_source === "impedance" ? "RR imp" : "RR");
    this.set("v-bis", whole(v.bis));
    this.set("v-temp", fixed(v.temp, 1));

    const net = v.net_fluid ?? 0;
    this.set("v-net", `${net >= 0 ? "+" : "−"}${Math.abs(net).toFixed(0)} mL`);
    this.set("v-io",
      `IV\u00a0${fixed(v.fluid_in, 0)} · PRBC\u00a0${fixed(v.blood_in, 0)} · Urine\u00a0${fixed(v.urine_out, 0)} · Loss\u00a0${fixed(v.blood_out, 0)}\u00a0mL`);

    this.set("v-ppeak", whole(v.ppeak));
    this.set("v-pplat", whole(v.pplat));
    this.set("v-peep", whole(v.peep));
    this.set("v-pmean", whole(v.pmean));
    this.set("v-vte", whole(v.vte));
    this.set("v-mv", fixed(v.mv, 1));
    this.set("v-cdyn", whole(v.cdyn));
    this.set("v-fio2", whole(v.fio2));
    this.set("v-eto2", whole(v.eto2));
    this.set("v-fi-n2o", whole(v.fi_n2o));
    this.set("v-et-n2o", whole(v.et_n2o));
    this.set("v-fi-sevo", fixed(v.fi_sevo, 1));
    this.set("v-et-sevo", fixed(v.et_sevo, 1));
    this.set("v-mac", fixed(v.et_mac, 1));
  }

  updateAlarms(alarms, controls = null, ended = false, vitals = null) {
    const active = alarms ?? {};
    const rhythmAlarm = RHYTHM_ALARMS.find(([key]) => active[key]);
    for (const box of this.alarmBoxes) {
      const name = box.dataset.alarm;
      const critical = name === "HR" && Boolean(rhythmAlarm);
      const side = name ? alarmSide(active[name]) : "";
      const level = critical ? "high" : side;
      box.classList?.toggle("alarm-low", !critical && level === "low");
      box.classList?.toggle("alarm-high", level === "high");
      box.classList?.toggle("alarm-critical", critical);
    }

    if (this.alarmBanner && this.alarmBannerText) {
      const highPriority = [];
      const mediumPriority = [];
      if (rhythmAlarm) highPriority.push(rhythmAlarm[1]);
      if (controls && controls.o2_connected === false) highPriority.push("O₂ SUPPLY DISCONNECTED");

      const spo2SignalLost = vitals?.spo2 === null;
      for (const [key, raw] of Object.entries(active)) {
        if (key === "Asystole" || key === "VFIB" || key === "VTACH" || key === "Arrest" || key === "O2Supply") continue;
        if (key === "HR" && rhythmAlarm) continue;
        const side = alarmSide(raw);
        if (!side) continue;
        if (key === "SpO2" && side === "low" && spo2SignalLost) {
          if (rhythmAlarm) mediumPriority.push("SpO₂ NO SIGNAL");
          else highPriority.push("SpO₂ NO SIGNAL");
          continue;
        }
        const label = ALARM_LABELS[key] ?? key;
        const text = `${label} ${side.toUpperCase()}`;
        if ((key === "SpO2" && side === "low") || (key === "FiO2" && side === "low") || key === "ReservoirLow") {
          highPriority.push(text);
        } else {
          mediumPriority.push(text);
        }
      }

      const allLabels = [...highPriority, ...mediumPriority];
      this.alarmBanner.title = allLabels.join(" · ");
      if (highPriority.length) {
        this.alarmBanner.dataset.severity = "critical";
        const count = mediumPriority.length;
        const extra = count ? ` · +${count} ${count === 1 ? "alarm" : "alarms"}` : "";
        setText(this.alarmBannerText, `${highPriority.join(" · ")}${extra}`);
      } else if (mediumPriority.length) {
        this.alarmBanner.dataset.severity = "warning";
        setText(this.alarmBannerText, mediumPriority.join(" · "));
      } else {
        this.alarmBanner.dataset.severity = ended ? "ended" : "normal";
        setText(this.alarmBannerText, ended ? "Session ended" : "");
      }
    }
  }
}
