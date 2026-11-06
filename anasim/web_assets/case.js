// Case briefing, actions, hints, and the end-of-case review. Guided steps stay in app.js.
const $ = (id) => document.getElementById(id);
const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };

function caseTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

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

export class CasePanel {
  constructor(send, newSession) {
    this.send = send;
    this.actionSignature = "";
    this.reviewRendered = false;
    this.disabledControls = new Map();
    $("case-review-dialog").close();
    $("case-hint").onclick = () => send("case_hint");
    $("case-end").onclick = () => send("case_finish");
    $("case-review").onclick = () => $("case-review-dialog").showModal();
    $("case-review-close").onclick = () => $("case-review-dialog").close();
    $("case-new-session").onclick = () => { $("case-review-dialog").close(); newSession(); };
  }

  destroy() {
    $("case-review-dialog").close();
    for (const [control, disabled] of this.disabledControls) control.disabled = disabled;
  }

  update(state, waiting) {
    const independent = state.mode === "independent";
    $("case-guide").hidden = independent || state.complete;
    $("case-briefing").hidden = !independent || state.complete;
    setText($("case-briefing"), state.briefing);
    const observation = state.observations.at(-1);
    const cue = observation?.text ?? (waiting ? "Press Start to begin the case." : "");
    $("case-observation").hidden = !cue || state.complete;
    setText($("case-observation"), cue);
    $("case-hint").hidden = !independent || state.complete;
    $("case-hint-text").hidden = !state.hint || state.complete;
    setText($("case-hint-text"), state.hint ?? "");
    $("case-end").hidden = state.complete;
    $("case-review").hidden = !state.complete;
    if (independent || state.complete) setText($("scenario-count"), state.complete ? "Case complete" : "On my own");
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
      $("clinical-feedback").hidden = true;
      for (const control of document.querySelectorAll('[data-panel] input, [data-panel] select, [data-panel] button, #nmb-measure, #nmb-mode, #nmb-repeat, #measure-nibp, #record')) {
        if (!this.disabledControls.has(control)) this.disabledControls.set(control, control.disabled);
        control.disabled = true;
      }
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
      independent ? "On my own" : "Step-by-step guide", `${caseTime(report.duration)} elapsed`, ...(independent ? [hints] : []),
    ].join(", "));
    const content = [];
    if (report.reason === "cardiac_arrest") content.push(node("p", "The case ended at irreversible cardiac arrest.", "case-safety"));
    const { physiology } = report;
    content.push(pairs([
      ["Lowest MAP", `${physiology.lowest_map} mmHg`],
      ["Lowest oxygen saturation", `${physiology.lowest_spo2}%`],
      ["MAP below 65 mmHg", caseTime(physiology.hypotension_s)],
      ["Oxygen saturation below 90%", caseTime(physiology.hypoxemia_s)],
    ], "case-metrics"));
    content.push(node("h3", "Outcomes"));
    content.push(pairs(report.outcomes.map((outcome) => [outcome.label, outcome.achieved ? "Yes" : "No"]), "case-metrics"));
    content.push(node("h3", "Safety events"));
    if (report.safety_events.length) {
      content.push(pairs(report.safety_events.map((event) => [caseTime(event.time), event.text]), "case-log case-safety"));
    } else content.push(node("p", "No critical unsafe actions recorded."));
    if (report.learning_points.length) {
      content.push(node("h3", "Review"));
      const review = node("ul");
      for (const point of report.learning_points) review.append(node("li", point));
      content.push(review);
    }
    const timeline = node("details", undefined, "case-timeline");
    timeline.append(node("summary", "Timeline"));
    timeline.append(pairs(report.timeline.map((entry) => [caseTime(entry.time), entry.text]), "case-log"));
    content.push(timeline);
    $("case-review-content").replaceChildren(...content);
  }
}
