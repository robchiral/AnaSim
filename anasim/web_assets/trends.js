// Recorded monitor measurements, shown in the sidebar without covering the loops.
const WINDOW_S = 1800;
const SERIES = {
  map: { color: "--abp", range: [40, 120] },
  hr: { color: "--ecg", range: [30, 150] },
  etco2: { color: "--co2", range: [0, 60] },
  bis: { color: "--bis", range: [0, 100] },
};

const stamp = (seconds) => {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
};

export class Trends {
  constructor(info) {
    this.panel = document.getElementById("trends-panel");
    this.window = document.getElementById("trend-window");
    this.window.value = "900";
    this.arterial = info.arterial_line;
    document.getElementById("trend-map-label").textContent = this.arterial ? "ART MAP" : "NIBP MAP";
    this.samples = [...info.trends];
    this.time = this.samples.at(-1).time;
    this.frame = null;
    this.canvases = [...this.panel.querySelectorAll("canvas")];
    this.listeners = new AbortController();
    this.window.addEventListener("change", () => this.requestDraw(), { signal: this.listeners.signal });
    this.observer = new ResizeObserver(() => this.requestDraw());
    for (const canvas of this.canvases) this.observer.observe(canvas);
  }

  destroy() {
    this.listeners.abort();
    this.observer.disconnect();
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
    if (this.panel.hidden || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      if (!this.panel.hidden) this.draw();
    });
  }

  draw() {
    const style = getComputedStyle(document.documentElement);
    const color = (name) => style.getPropertyValue(name).trim();
    const end = Math.max(1, this.time);
    const start = Math.max(0, end - Number(this.window.value));
    const samples = this.samples.filter((sample) => sample.time >= start);
    for (const canvas of this.canvases) {
      const key = canvas.dataset.trend;
      const series = SERIES[key];
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) continue;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const discrete = key === "map" && !this.arterial;
      const values = samples.filter((sample) => !discrete || (
        Number.isFinite(sample.map_time) && sample.map_time >= start && sample.map_time <= end
      )).map((sample) => sample[key]).filter(Number.isFinite);
      const low = Math.floor(Math.min(series.range[0], ...values) / 10) * 10;
      const high = Math.ceil(Math.max(series.range[1], ...values) / 10) * 10;
      const left = 32;
      const right = width - 8;
      const top = 12;
      const bottom = height - 24;
      const x = (time) => left + (time - start) / (end - start) * (right - left);
      const y = (value) => bottom - (value - low) / (high - low) * (bottom - top);
      ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`;
      ctx.lineWidth = 1;
      ctx.strokeStyle = color("--monitor-rule");
      ctx.fillStyle = color("--text-dim");
      ctx.textAlign = "right";
      for (const value of [low, (low + high) / 2, high]) {
        ctx.beginPath();
        ctx.moveTo(left, y(value));
        ctx.lineTo(right, y(value));
        ctx.stroke();
        ctx.fillText(String(value), left - 5, y(value) + 4);
      }
      for (const [time, alignment] of [[start, "left"], [(start + end) / 2, "center"], [end, "right"]]) {
        ctx.textAlign = alignment;
        ctx.fillText(stamp(time), x(time), height - 7);
      }
      let connected = false;
      let cuffTime = null;
      ctx.strokeStyle = color(series.color);
      ctx.fillStyle = color(series.color);
      ctx.lineWidth = 1.5;
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
      if (!discrete && values.length === 1) {
        const sample = samples.find((point) => Number.isFinite(point[key]));
        ctx.beginPath();
        ctx.arc(x(sample.time), y(sample[key]), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!values.length) {
        ctx.fillStyle = color("--text-dim");
        ctx.textAlign = "center";
        ctx.fillText("No readings", (left + right) / 2, (top + bottom) / 2);
      }
    }
  }
}
