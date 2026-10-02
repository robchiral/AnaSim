// Waveform sweeps and monitor numerics.

// Respiratory traces sweep at half the cardiac speed, as on anesthesia
// workstations, so each shows several breaths.
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
    this.alarmBoxes = [...document.querySelectorAll(".numeric[data-alarm]")];
    this.frame = requestAnimationFrame(() => this.render());
  }

  destroy() {
    cancelAnimationFrame(this.frame);
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
    this.updateNumerics(snap.vitals);
    document.getElementById("measure-nibp").disabled = snap.ended || snap.vitals.nibp_cuff !== null;
    this.updateAlarms(snap.alarms);
  }

  render() {
    if (this.dirty) {
      for (const sweep of Object.values(this.sweeps)) sweep.draw();
      this.dirty = false;
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
