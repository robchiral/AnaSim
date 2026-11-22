// Recorded vital signs from the patient monitor.
import { minutesSeconds } from "./display.js";

function trendTime(seconds) {
  if (seconds < 0) return `-${minutesSeconds(Math.abs(Math.round(seconds)))}`;
  return minutesSeconds(seconds);
}

const WINDOW_S = 1800;
const SERIES = {
  map: { color: "--abp", range: [40, 120], minKey: "MAP_min", maxKey: "MAP_max" },
  hr: { color: "--ecg", range: [30, 150], minKey: "HR_min", maxKey: "HR_max" },
  etco2: { color: "--co2", range: [0, 60], minKey: "EtCO2_min", maxKey: "EtCO2_max" },
  bis: { color: "--bis", range: [0, 100], minKey: "BIS_min", maxKey: "BIS_max" },
};

export class Trends {
  constructor(info) {
    this.dialog = document.getElementById("trends-dialog");
    this.window = document.getElementById("trend-window");
    this.window.value = "900";
    this.arterial = info.arterial_line;
    this.thresholds = info.alarm_thresholds ?? {};
    document.getElementById("trend-map-label").textContent = this.arterial ? "ART MAP" : "NIBP MAP";
    this.samples = [...info.trends];
    this.time = this.samples.at(-1).time;
    this.frame = null;
    this.canvases = [...this.dialog.querySelectorAll("canvas")];
    this.listeners = new AbortController();
    const signal = this.listeners.signal;
    const trigger = document.getElementById("show-trends");
    trigger.addEventListener("click", () => {
      trigger.focus({ preventScroll: true });
      this.dialog.showModal();
      this.requestDraw();
    }, { signal });
    this.window.addEventListener("change", () => this.requestDraw(), { signal });
    this.observer = new ResizeObserver(() => this.requestDraw());
    for (const canvas of this.canvases) this.observer.observe(canvas);
  }

  destroy() {
    this.listeners.abort();
    this.observer.disconnect();
    this.dialog.close();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
  }

  update(snap) {
    this.time = snap.time;
    for (const sample of snap.trends) {
      if (sample.time > this.samples.at(-1).time) this.samples.push(sample);
    }
    while (this.samples.length > 1 && this.samples[0].time < this.time - WINDOW_S) this.samples.shift();
    if (snap.trends.length) this.requestDraw();
  }

  requestDraw() {
    if (!this.dialog.open || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      if (this.dialog.open) this.draw();
    });
  }

  draw() {
    const style = getComputedStyle(document.documentElement);
    const font = `10px ${getComputedStyle(document.body).fontFamily}`;
    const color = (name) => style.getPropertyValue(name).trim();
    const earliest = this.samples.length ? this.samples[0].time : 0;
    const end = earliest < 0 ? Math.max(0, this.time) : Math.max(1, this.time);
    const start = Math.max(Math.min(0, earliest), end - Number(this.window.value));
    const samples = this.samples.filter((sample) => sample.time >= start);
    for (const canvas of this.canvases) {
      const key = canvas.dataset.trend;
      const series = SERIES[key];
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) continue;
      const dpr = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const discrete = key === "map" && !this.arterial;
      let [min, max] = series.range;
      let count = 0, lastSample;
      for (const sample of samples) {
        if (!Number.isFinite(sample[key])) continue;
        if (discrete && !(Number.isFinite(sample.map_time) && sample.map_time >= start && sample.map_time <= end)) continue;
        min = Math.min(min, sample[key]);
        max = Math.max(max, sample[key]);
        count++;
        lastSample = sample;
      }
      const low = Math.floor(min / 10) * 10;
      const high = Math.ceil(max / 10) * 10;
      const left = 34;
      const right = width - 8;
      const top = 12;
      const bottom = height - 22;
      const x = (time) => left + (time - start) / (end - start) * (right - left);
      const y = (value) => bottom - (value - low) / (high - low) * (bottom - top);

      ctx.font = font;
      ctx.lineWidth = 1;
      ctx.strokeStyle = color("--monitor-rule");
      ctx.fillStyle = color("--text-dim");
      ctx.textAlign = "right";
      for (const value of [low, (low + high) / 2, high]) {
        const py = Math.round(y(value)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(left, py);
        ctx.lineTo(right, py);
        ctx.stroke();
        ctx.fillText(String(value), left - 5, py + 3);
      }

      // Dashed alarm-limit guides when within the displayed range.
      const limitLo = this.thresholds[series.minKey];
      const limitHi = this.thresholds[series.maxKey];
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(242, 183, 5, 0.30)";
      for (const lim of [limitLo, limitHi]) {
        if (lim !== undefined && lim > low && lim < high) {
          const ly = Math.round(y(lim)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(left, ly);
          ctx.lineTo(right, ly);
          ctx.stroke();
        }
      }
      ctx.restore();

      const middle = Math.round((start + end) / 120) * 60;
      const labels = [[start, "left"], [end, "right"]];
      const clearance = (end - start) / 4;
      if (middle - start >= clearance && end - middle >= clearance) labels.push([middle, "center"]);
      for (const [time, alignment] of labels) {
        ctx.textAlign = alignment;
        ctx.fillText(trendTime(time), x(time), height - 6);
      }
      let connected = false;
      let cuffTime = null;
      ctx.strokeStyle = color(series.color);
      ctx.fillStyle = color(series.color);
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      for (const sample of samples) {
        const value = sample[key];
        const time = discrete ? sample.map_time : sample.time;
        if (!Number.isFinite(value) || time === null || time < start || time > end) {
          connected = false;
          continue;
        }
        if (discrete) {
          if (time !== cuffTime) {
            ctx.moveTo(x(time) + 2.5, y(value));
            ctx.arc(x(time), y(value), 2.5, 0, Math.PI * 2);
            cuffTime = time;
          }
        } else {
          if (connected) ctx.lineTo(x(time), y(value));
          else ctx.moveTo(x(time), y(value));
          connected = true;
        }
      }
      if (discrete) ctx.fill();
      else ctx.stroke();
      if (!discrete && count === 1) {
        ctx.beginPath();
        ctx.arc(x(lastSample.time), y(lastSample[key]), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!count) {
        ctx.fillStyle = color("--text-dim");
        ctx.textAlign = "center";
        ctx.fillText("No readings", (left + right) / 2, (top + bottom) / 2);
      }
    }
  }
}
