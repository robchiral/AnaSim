import { $, setText } from "./display.js";

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
    const ptc = nmb.mode === "ptc";
    const label = single ? "T1" : ptc ? "PTC" : "TOF ratio";
    let title = single ? "T1 / baseline" : ptc ? "PTC" : "TOF";
    let value = "--", detail = "", quality = "";
    if (reading) {
      if (single) {
        value = reading.t1 === null ? "--" : `${Math.round(reading.t1)}%`;
        if (reading.t1 === null) detail = "Baseline unavailable";
      } else if (ptc) {
        title = "PTC";
        value = `${reading.count}/15`;
        detail = reading.count === 0 ? "Intense block" : reading.count < 15 ? "Deep block" : "TOF detectable";
      } else if (reading.ratio === null || reading.low_signal) {
        title = "TOF count";
        value = `${reading.count}/4`;
        if (reading.low_signal) {
          detail = "Low signal";
          quality = "low";
        }
      } else {
        title = label;
        value = `${Math.floor(reading.ratio)}%`;
        const t1Depressed = reading.t1 !== null && reading.t1 < 90;
        if (t1Depressed) detail = `T1 ${Math.round(reading.t1)}%`;
        quality = reading.ratio >= 90 && !t1Depressed ? "recovered" : "";
      }
    }
    setText($("nmb-title"), title);
    setText($("nmb-value"), value);
    setText($("nmb-detail"), detail);
    // Automatic readings are at most one interval old, so show the countdown instead.
    const stimulation = single ? "Stimulating" : ptc ? "PTC stimulating" : `Pulse ${nmb.pulses}/4`;
    setText($("nmb-status"), nmb.measuring ? stimulation :
      nmb.next !== null ? `Next ${Math.ceil(nmb.next)} s` :
      reading ? `${Math.floor(Math.max(0, time - reading.time))} s ago` : "");
    $("nmb-result").dataset.quality = quality;
    $("nmb-mode").value = nmb.mode;
    $("nmb-mode").disabled = ended || nmb.measuring;
    $("nmb-repeat").value = String(nmb.repeat);
    $("nmb-repeat").disabled = ended;
    $("nmb-measure").disabled = ended || nmb.measuring || nmb.wait > 0;

    const reference = nmb.baseline ?? 1;
    const bars = $("nmb-bars");
    bars.hidden = !reading || ptc;
    for (let i = 0; i < 4; i++) {
      const bar = $(`nmb-t${i + 1}`);
      bar.hidden = single && i > 0;
      const amplitude = reading?.responses[i] ?? 0;
      bar.style.setProperty("--height", `${Math.min(100, 100 * amplitude / reference)}%`);
    }
    if (reading) {
      const scale = nmb.baseline === null ? "% of model maximum" : "% of baseline";
      bars.setAttribute("aria-label", reading.responses.map(
        (response, index) => `T${index + 1} ${Math.round(100 * response / reference)}${scale}`,
      ).join(", "));
    }

    const cutoff = time - TREND_S;
    let expired = 0;
    while (expired < this.history.length && this.history[expired][0] < cutoff) expired++;
    if (expired) {
      this.history.splice(0, expired);
      trendChanged = true;
    }
    if (trendChanged) {
      const points = this.history.filter(([, mode, result, low]) => mode === nmb.mode && result !== null && !low);
      const scaleMax = ptc ? 15 : 100;
      $("nmb-trend-points").innerHTML = points.map(([timestamp, , result]) => {
        const x = timestamp / TREND_S * 180;
        const y = 37 - Math.min(scaleMax, Math.max(0, result)) / scaleMax * 34;
        return `<path d="M${x.toFixed(2)} ${y.toFixed(2)}h0"/>`;
      }).join("");
      this.trendMode = nmb.mode;
      $("nmb-history").dataset.mode = nmb.mode;
      const metric = single ? "T1 as a percentage of baseline" : ptc ? "Post-tetanic count" : label;
      $("nmb-trend").setAttribute("aria-label", `${metric} over the past 15 minutes`);
      setText($("nmb-trend-label"), `${label} history`);
      setText($("nmb-trend-empty"), single ? "No T1 readings" : ptc ? "No PTC readings" : "No ratio readings");
      $("nmb-trend-body").hidden = points.length === 0;
      $("nmb-trend-empty").hidden = points.length > 0;
    }
    if (this.trendTime !== time) {
      // Keep each reading's path and move the time window as a group.
      $("nmb-trend-points").setAttribute("transform", `translate(${-cutoff / TREND_S * 180} 0)`);
      this.trendTime = time;
    }
  }
}
