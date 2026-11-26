// Case briefing, actions, hints, and debrief.
import { $, minutesSeconds, setText } from "./display.js";

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}

function pairs(rows, className) {
  const list = node("dl", undefined, className);
  for (const [term, detail] of rows) list.append(node("dt", term), node("dd", detail));
  return list;
}

function patientReportLine(info) {
  if (!info?.patient) return "";
  const p = info.patient;
  const sex = p.sex === "female" ? "Female" : "Male";
  const parts = [`${Math.round(p.age)}y ${sex}`, `${Number(p.weight.toFixed(1))} kg`];
  if (p.height) {
    const bmi = p.weight / ((p.height / 100) ** 2);
    parts.push(`${Math.round(p.height)} cm (BMI ${bmi.toFixed(1)})`);
  }
  if (p.renal_status !== "Normal") parts.push(`${p.renal_status} renal impairment`);
  if (p.hepatic_status !== "Normal") parts.push(`${p.hepatic_status} hepatic impairment`);
  if (p.coronary_flow_reserve < 5) parts.push("Coronary disease");
  if (p.heart_failure !== "none") parts.push(p.heart_failure === "hfref" ? "HFrEF" : "HFpEF");
  if (p.copd_severity > 0) parts.push("COPD");
  if (p.type2_diabetes) parts.push("Type 2 diabetes");
  const defaultBg = p.type2_diabetes ? 150 : 90;
  if (p.baseline_glucose != null && Math.abs(p.baseline_glucose - defaultBg) > 0.5) {
    parts.push(`BG ${Math.round(p.baseline_glucose)} mg/dL`);
  }
  if (p.gestational_age_weeks > 0) parts.push(`${p.gestational_age_weeks} wk pregnant`);
  if (info.version) parts.push(`AnaSim ${info.version}`);
  return parts.join(" · ");
}

export class CasePanel {
  constructor(send, newSession, info = null) {
    this.send = send;
    this.info = info;
    this.actionSignature = "";
    this.reviewRendered = false;
    this.disabledControls = new Map();
    this.listeners = new AbortController();
    let restoreTimeline = null;
    window.addEventListener("beforeprint", () => {
      if (!$("case-review-dialog").open) return;
      const timeline = $("case-review-content").querySelector(".case-timeline");
      if (timeline && !timeline.open) {
        timeline.open = true;
        restoreTimeline = timeline;
      }
    }, { signal: this.listeners.signal });
    window.addEventListener("afterprint", () => {
      if (restoreTimeline) {
        restoreTimeline.open = false;
        restoreTimeline = null;
      }
    }, { signal: this.listeners.signal });
    $("case-review-dialog").close();
    $("case-hint").onclick = () => send("case_hint");
    $("case-end").onclick = () => send("case_finish");
    $("case-review").onclick = () => $("case-review-dialog").showModal();
    $("case-review-close").onclick = () => $("case-review-dialog").close();
    $("case-review-print").onclick = () => window.print();
    $("case-new-session").onclick = () => { $("case-review-dialog").close(); newSession(); };
  }

  destroy() {
    this.listeners.abort();
    $("case-review-dialog").close();
    for (const [control, disabled] of this.disabledControls) control.disabled = disabled;
  }

  freezeControls() {
    $("clinical-feedback").hidden = true;
    for (const control of document.querySelectorAll('[data-panel] :is(input, select, button), #nmb-measure, #nmb-mode, #nmb-repeat, #measure-nibp, #speed, .speed-preset, #record')) {
      if (!this.disabledControls.has(control)) this.disabledControls.set(control, control.disabled);
      control.disabled = true;
    }
  }

  update(state) {
    const independent = state.mode === "independent";
    $("case-guide").hidden = independent || state.complete;
    $("case-navigation").hidden = independent || state.complete;
    $("case-briefing").hidden = !independent || state.complete;
    setText($("case-briefing"), state.briefing);
    const observation = state.observations.at(-1);
    const cue = observation?.text ?? "";
    $("case-observation").hidden = !cue || state.complete;
    setText($("case-observation"), cue);
    $("case-hint").hidden = !independent || state.complete;
    $("case-hint-text").hidden = !state.hint || state.complete;
    setText($("case-hint-text"), state.hint ?? "");
    $("case-end").hidden = state.complete;
    $("case-review").hidden = !state.complete;
    if (independent || state.complete) setText($("scenario-count"), state.complete ? "Case complete" : "Independent");
    const signature = JSON.stringify(state.actions);
    if (signature !== this.actionSignature) {
      this.actionSignature = signature;
      $("case-actions").replaceChildren(...state.actions.map((action) => {
        const button = node("button", action.label, "btn outlined small");
        button.type = "button";
        button.disabled = !action.enabled;
        button.onclick = async () => {
          button.disabled = true;
          if (await this.send("case_action", { action: action.id }) === undefined) button.disabled = false;
        };
        return button;
      }));
    }
    $("case-actions").hidden = state.complete;
    if (state.complete) {
      this.freezeControls();
      if (!this.reviewRendered) {
        this.reviewRendered = true;
        this.renderReview(state.debrief);
        $("case-review-dialog").showModal();
      }
    }
  }

  renderReview(report) {
    const independent = report.mode === "independent";
    const hints = `${report.hints_used} ${report.hints_used === 1 ? "hint" : "hints"} used`;
    setText($("case-review-title"), report.name);
    setText($("case-review-summary"), [
      independent ? "Independent" : "Guided",
      minutesSeconds(report.duration),
      ...(independent ? [hints] : []),
    ].join(" · "));
    setText($("case-review-patient"), patientReportLine(this.info));

    const content = [];
    if (report.reason === "cardiac_arrest") {
      content.push(node("p", "The case ended at irreversible cardiac arrest.", "case-arrest-banner"));
    }

    const { physiology } = report;
    const summaryStrip = node("dl", undefined, "case-summary-strip");
    const metrics = [
      ["Lowest MAP", `${physiology.lowest_map} mmHg`, physiology.lowest_map < 65],
      ["Lowest SpO₂", `${physiology.lowest_spo2}%`, physiology.lowest_spo2 < 90],
      ["MAP < 65 mmHg", minutesSeconds(physiology.hypotension_s), physiology.hypotension_s > 0],
      ["SpO₂ < 90%", minutesSeconds(physiology.hypoxemia_s), physiology.hypoxemia_s > 0],
    ];
    for (const [label, value, flagged] of metrics) {
      const cell = node("div", undefined, flagged ? "is-flagged" : undefined);
      cell.append(node("dt", label), node("dd", value));
      summaryStrip.append(cell);
    }
    content.push(summaryStrip);

    const outcomes = node("section");
    const outcomeList = node("ul", undefined, "case-outcomes");
    for (const outcome of report.outcomes) {
      const item = node("li", undefined, "case-outcome-row");
      const result = node("span", outcome.achieved ? "Met" : "Not met",
        `outcome-status ${outcome.achieved ? "is-met" : "is-missed"}`);
      item.append(node("span", outcome.label), result);
      outcomeList.append(item);
    }
    outcomes.append(node("h3", "Outcomes"), outcomeList);
    content.push(outcomes);

    const safety = node("section");
    safety.append(node("h3", "Safety events"));
    if (report.safety_events.length) {
      const box = node("div", undefined, "case-safety-box");
      box.append(pairs(report.safety_events.map((event) => [minutesSeconds(event.time), event.text]), "case-log case-safety"));
      safety.append(box);
    } else {
      safety.append(node("p", "No critical unsafe actions recorded.", "case-ok-note"));
    }
    content.push(safety);

    if (report.learning_points.length) {
      const review = node("section");
      const points = node("ul", undefined, "case-review-points");
      for (const point of report.learning_points) points.append(node("li", point));
      review.append(node("h3", "Clinical review"), points);
      content.push(review);
    }

    const timeline = node("details", undefined, "case-timeline");
    timeline.append(node("summary", `Action timeline (${report.timeline.length} events)`));
    timeline.append(pairs(report.timeline.map((entry) => [minutesSeconds(entry.time), entry.text]), "case-log"));
    content.push(timeline);

    $("case-review-content").replaceChildren(...content);
  }
}
