const $ = (id) => document.getElementById(id);
const TREND_S = 900;

export class NeuromuscularDisplay {
  constructor(history) {
    this.history = [...history];
    this.trendMode = null;
    this.trendTime = null;
  }

  update(nmb, rows, time, ended) {
    let trendChanged = this.trendMode !== nmb.mode;
    for (const row of rows) {
      if (row[0] > (this.history.at(-1)?.[0] ?? -Infinity)) {
        this.history.push(row);
        trendChanged = true;
      }
    }
    const reading = nmb.reading;
    const single = nmb.mode === "single";
    const label = single ? "T1" : "TOF ratio";
    let title = label, value = "--", detail = reading ? "" : "Awaiting measurement", quality = "";
    if (reading) {
      if (single) {
        value = reading.t1 === null ? "--" : `${Math.round(reading.t1)}%`;
        detail = reading.t1 === null ? "Baseline unavailable" : "Of baseline";
      } else if (reading.ratio === null || reading.low_signal) {
        title = "TOF count";
        value = `${reading.count}/4`;
        if (reading.low_signal) {
          detail = `Low signal, ratio ${Math.round(reading.ratio)}%`;
          quality = "low";
        }
      } else {
        value = `${Math.round(reading.ratio)}%`;
        quality = reading.ratio >= 90 ? "recovered" : "";
      }
    }
    $("nmb-title").textContent = title;
    $("nmb-value").textContent = value;
    $("nmb-detail").textContent = detail;
    // Automatic readings are at most one interval old, so show the countdown instead.
    $("nmb-status").textContent = nmb.measuring ? `Pulse ${nmb.pulses}/${single ? 1 : 4}` :
      nmb.next !== null ? `Next ${Math.ceil(nmb.next)} s` :
      reading ? `${Math.floor(Math.max(0, time - reading.time))} s ago` : "";
    $("nmb-result").dataset.quality = quality;
    $("nmb-mode").value = nmb.mode;
    $("nmb-mode").disabled = ended || nmb.measuring;
    $("nmb-repeat").value = String(nmb.repeat);
    $("nmb-repeat").disabled = ended;
    $("nmb-measure").disabled = ended || nmb.measuring || nmb.wait > 0;

    const reference = nmb.baseline ?? 1;
    $("nmb-baseline").hidden = nmb.baseline === null;
    for (let i = 0; i < 4; i++) {
      const bar = $(`nmb-t${i + 1}`);
      bar.hidden = single && i > 0;
      const amplitude = reading?.responses[i] ?? 0;
      bar.style.setProperty("--height", `${Math.min(100, 100 * amplitude / reference)}%`);
    }
    $("nmb-bars").setAttribute("aria-label", reading ? reading.responses.map(
      (response, index) => `T${index + 1} ${Math.round(100 * response / reference)}${nmb.baseline === null ? " units" : "% of baseline"}`,
    ).join(", ") : "No twitch responses measured");

    const cutoff = time - TREND_S;
    let expired = 0;
    while (expired < this.history.length && this.history[expired][0] < cutoff) expired++;
    if (expired) {
      this.history.splice(0, expired);
      trendChanged = true;
    }
    if (trendChanged) {
      const points = this.history.filter(([, mode, result]) => mode === nmb.mode && result !== null);
      $("nmb-trend-points").innerHTML = points.map(([timestamp, , result, low]) => {
        const x = timestamp / TREND_S * 180;
        const y = 37 - Math.min(100, Math.max(0, result)) / 100 * 34;
        return `<path d="M${x.toFixed(2)} ${y.toFixed(2)}h0"${low ? ' class="low"' : ""}/>`;
      }).join("");
      this.trendMode = nmb.mode;
      $("nmb-trend").setAttribute("aria-label", `${label} over the past 15 minutes`);
      $("nmb-trend-label").textContent = `${label}, past 15 min`;
    }
    if (this.trendTime !== time) {
      // Keep each reading's path and move the time window as a group.
      $("nmb-trend-points").setAttribute("transform", `translate(${-cutoff / TREND_S * 180} 0)`);
      this.trendTime = time;
    }
  }
}
