// Machine, medication, circulation, and event controls.
// Numeric settings remain drafts until explicitly applied.
import { SettingsEditor } from "./settings.js";

const $ = (id) => document.getElementById(id);
const editors = new WeakMap();

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function setValue(input, value) {
  if (editors.has(input)) {
    editors.get(input).sync(input, value);
    return;
  }
  if (document.activeElement === input) return;
  const text = String(value);
  if (input.value !== text) input.value = text;
}

function setPressed(button, pressed, text) {
  const value = String(pressed);
  if (button.getAttribute("aria-pressed") !== value) button.setAttribute("aria-pressed", value);
  if (text !== undefined) setText(button, text);
}

function readNumber(input) {
  return Number(input.value);
}

const round = (value, digits = 1) => Number(value.toFixed(digits));
const clock = (seconds) => {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
};
const signed = (value) => {
  const text = value.toFixed(1);
  return Number(text) > 0 ? `+${text}` : Number(text) === 0 ? "0.0" : text;
};
// Blood gas report rows: label, unit, result field, and display format.
const bloodGasRows = [
  ["pH", "", "ph", (value) => value.toFixed(2)],
  ["PaCO₂", "mmHg", "pa_co2", (value) => value.toFixed(0)],
  ["PaO₂", "mmHg", "pao2", (value) => value.toFixed(0)],
  ["HCO₃⁻", "mmol/L", "hco3", (value) => value.toFixed(1)],
  ["Base excess", "mmol/L", "base_excess", signed],
  ["Lactate", "mmol/L", "lactate", (value) => value.toFixed(1)],
  ["Na⁺", "mmol/L", "sodium", (value) => value.toFixed(0)],
  ["K⁺", "mmol/L", "potassium", (value) => value.toFixed(1)],
  ["Cl⁻", "mmol/L", "chloride", (value) => value.toFixed(0)],
  ["Ionized Ca²⁺", "mmol/L", "ionized_calcium", (value) => value.toFixed(2)],
  ["SaO₂", "%", "sao2", (value) => value.toFixed(1)],
  ["Hb", "g/dL", "hb", (value) => value.toFixed(1)],
  ["FiO₂", "%", "fio2", (value) => (value * 100).toFixed(0)],
];
// Four significant figures keep small rates visible without long decimals.
const rateText = (value) => String(Number(value.toPrecision(4)));
// Displayed rates: whole numbers from 10 up, two significant figures below.
const shortRate = (value) => (value >= 10 ? Math.round(value) : Number(value.toPrecision(2)));

// Settings each ventilator mode uses, as on a GE Aisys. PSV uses RR and Pinsp
// for apnea backup breaths and Tinsp as the longest supported breath.
const ventFields = {
  "VCV": ["rr", "tv", "ie", "pause", "p_max"],
  "PCV": ["rr", "p_insp", "ie"],
  "PCV-VG": ["rr", "tv", "ie", "p_max"],
  "SIMV-VC": ["rr", "tv", "t_insp", "pause", "p_max", "p_support", "trigger"],
  "SIMV-PC": ["rr", "p_insp", "t_insp", "p_support", "trigger"],
  "SIMV-VG": ["rr", "tv", "t_insp", "p_max", "p_support", "trigger"],
  "PSV": ["p_support", "trigger", "rr", "p_insp", "t_insp"],
  "CPAP": ["trigger"],
};
const ventInputs = {
  rr: "c-rr", tv: "c-tv", p_insp: "c-pinsp", p_support: "c-psupport", peep: "c-peep",
  t_insp: "c-tinsp", pause: "c-pause", p_max: "c-pmax", trigger: "c-trigger",
};

// Presentation order stays fixed while medications are administered.
const medicationGroups = [
  { name: "Anesthesia and sedation", keys: ["propofol", "midazolam", "ketamine", "etomidate"] },
  { name: "Antiarrhythmics", keys: ["lidocaine", "amiodarone"] },
  { name: "Opioids", keys: ["fentanyl", "remi", "hydromorphone", "morphine"] },
  { name: "Vasopressors and inotropes", keys: ["phenyl", "ephedrine", "nore", "epi", "vaso", "dobu", "milri"] },
  { name: "Heart rate and blood pressure", keys: ["atropine", "glyco", "esmolol", "labetalol"] },
  { name: "Neuromuscular blockade and reversal", keys: ["roc", "succinylcholine", "sugammadex"] },
  { name: "Rescue and antidotes", keys: ["dantrolene", "lipid", "calcium_chloride"] },
];
const medicationSearchTerms = {
  dantrolene: "malignant hyperthermia MH",
  lipid: "local anesthetic systemic toxicity LAST",
  calcium_chloride: "hypocalcemia",
};
const halfTimeDrugs = ["propofol", "remi", "fentanyl"];

export class Controls {
  constructor(info, send) {
    this.settingEditors = [];
    this.destroyed = false;
    this.feedbackTimers = new Map();
    this.history = [...info.medication_history];
    this.bloodGases = [...info.blood_gases];
    this.shownBloodGases = null;
    this.time = 0;
    this.weight = info.patient.weight;
    this.send = async (name, args) => {
      const result = await send(name, args);
      if (!this.destroyed && result && "feedback" in result) this.showClinicalFeedback(result.feedback);
      if (!this.destroyed && result?.medication) {
        this.history.push(result.medication);
        this.history = this.history.slice(-50);
        this.showMedicationHistory();
        this.showMedicationFeedback(result.medication);
      }
      return result;
    };
    this.state = null;
    $("clinical-feedback-dismiss").onclick = () => this.send("dismiss_feedback");
    this.bindTabs();
    this.bindMachine();
    this.buildDrugs(info.drugs);
    this.bindFluids(info.patient.weight);
    this.bindResuscitation();
    this.bindLabs();
    this.bindEvents(info);
    // Cases start their own events.
    this.tabs.find((tab) => tab.dataset.tab === "Events").hidden = Boolean(info.scenario);
    this.showMedicationHistory();
  }

  editSettings(fields, apply, container = fields[0].closest("fieldset"), onChange) {
    const editor = new SettingsEditor(container, fields, async () => (await apply()) !== undefined, onChange);
    for (const field of fields) editors.set(field, editor);
    this.settingEditors.push(editor);
    return editor;
  }

  destroy() {
    this.destroyed = true;
    clearTimeout(this.fluidFeedbackTimer);
    clearTimeout(this.shockFeedbackTimer);
    for (const timer of this.feedbackTimers.values()) clearTimeout(timer);
    for (const editor of this.settingEditors) {
      for (const field of editor.fields) editors.delete(field);
      editor.destroy();
    }
  }

  async performAction(button, name, args) {
    if (button.disabled) return;
    button.disabled = true;
    try {
      return await this.send(name, args);
    } finally {
      button.disabled = false;
    }
  }

  showMedicationHistory() {
    const list = $("medication-history-list");
    list.replaceChildren();
    $("medication-history").hidden = this.history.length === 0;
    for (const entry of this.history) {
      const row = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = clock(entry.time);
      row.append(time, entry.text);
      list.prepend(row);
    }
  }

  showMedicationFeedback(entry) {
    const feedback = this.medicationRows.get(entry.key).feedback;
    setText(feedback, entry.text);
    feedback.hidden = false;
    clearTimeout(this.feedbackTimers.get(entry.key));
    this.feedbackTimers.set(entry.key, setTimeout(() => {
      feedback.hidden = true;
      this.feedbackTimers.delete(entry.key);
    }, 6000));
  }

  // --- Tabs ---------------------------------------------------------

  bindTabs() {
    this.tabs = [...document.querySelectorAll(".tabs button")];
    for (const tab of this.tabs) tab.onclick = () => this.openTab(tab.dataset.tab);
    this.openTab("Machine");
  }

  openTab(name) {
    document.querySelector(".tabs").hidden = name === "Trends";
    for (const tab of this.tabs) tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
    for (const panel of document.querySelectorAll(".tab-panel")) panel.hidden = panel.dataset.panel !== name;
    this.currentTab = name;
    if (name === "Medications") this.onMedicationsShown?.();
    this.onViewChanged?.();
  }

  // --- Machine ------------------------------------------------------

  bindMachine() {
    $("c-bair").onchange = () => this.send("bair_hugger", { target_c: Number($("c-bair").value) });
    $("c-cooling").onclick = () => this.send("surface_cooling", { on: !this.state.surface_cooling });
    for (const button of $("c-airway").querySelectorAll("button")) {
      button.onclick = () => this.send("airway", { mode: button.dataset.value });
    }
    this.maskSealEditor = this.editSettings(
      [$("c-mask-seal")],
      () => this.send("mask_seal", { percent: readNumber($("c-mask-seal")) }),
      $("c-mask-seal").parentElement,
    );
    const fgf = () => this.send("fgf", { o2: readNumber($("c-o2")), air: readNumber($("c-air")), n2o: readNumber($("c-n2o")) });
    this.editSettings([$("c-o2"), $("c-air"), $("c-n2o")], fgf, $("c-o2").closest(".fresh-gas-settings"));
    this.editSettings([$("c-vap")], () => this.send("vaporizer", { percent: readNumber($("c-vap")) }));
    $("c-bag").onclick = () => this.send("bag_mask", { active: !this.state.bag_mask });
    this.editSettings(
      [$("c-bag-rr"), $("c-bag-tv"), $("c-bag-pmax")],
      () => this.send("bag_settings", {
        rr: readNumber($("c-bag-rr")), tv: readNumber($("c-bag-tv")), p_max: readNumber($("c-bag-pmax")),
      }),
      $("c-bag-settings"),
    );
    $("c-vent-power").onclick = () => this.send("vent_power", { on: !this.state.vent.on });
    this.editSettings(
      ["c-vent-mode", "c-ie", ...Object.values(ventInputs)].map($),
      () => this.send("vent", {
        mode: $("c-vent-mode").value, ie: $("c-ie").value,
        ...Object.fromEntries(Object.entries(ventInputs).map(([name, id]) => [name, readNumber($(id))])),
      }),
      $("c-vent-settings"),
      () => this.applyVentMode($("c-vent-mode").value),
    );
  }

  applyVentMode(mode) {
    if (this.displayedVentMode === mode) return;
    this.displayedVentMode = mode;
    const used = ventFields[mode];
    for (const el of document.querySelectorAll("#c-vent-settings [data-vent]")) {
      el.hidden = !used.includes(el.dataset.vent);
    }
    $("c-rr-label").textContent = mode === "PSV" ? "Backup RR" : "RR";
    $("c-pinsp-label").textContent = mode === "PSV" ? "Backup Pinsp" : "Pinsp";
  }

  // --- Medications --------------------------------------------------

  buildDrugs(drugs) {
    const container = $("drug-cards");
    const lipid = $("lipid-controls");
    container.replaceChildren();
    $("running-infusion-list").replaceChildren();
    $("running-infusions").hidden = true;
    $("drug-search").value = "";
    this.searchGroupState = null;
    this.medicationRows = new Map();
    this.medicationGroups = medicationGroups.map(({ name, keys }) => {
      const section = document.createElement("details");
      section.className = "medication-group";
      section.open = true;
      const heading = document.createElement("summary");
      heading.textContent = name;
      section.append(heading);
      container.append(section);
      return { section, name, keys };
    });
    this.drugs = {};
    for (const spec of drugs) {
      const infusion = spec.rate_unit !== null;
      const tci = spec.tci_unit !== null;
      const card = document.createElement("details");
      card.className = "drug-card";
      card.innerHTML = `
        <summary><span class="drug-name"></span><span class="drug-strength"></span></summary>
        <div class="drug-controls">
        ${tci ? `<label class="inline">Infusion mode
          <select class="infusion-mode"><option value="rate">Manual rate</option><option value="tci">TCI</option></select>
        </label>` : ""}
        ${infusion ? `<div class="inputs${tci ? "" : " manual-rate"}">
          <label>Infusion rate</label>
          ${tci ? '<label class="target-label"></label>' : ""}
          <span class="unit-input"><input class="rate" type="number" min="0" max="2000" step="any"><span class="rate-unit"></span></span>
          ${tci ? '<span class="unit-input"><input class="target" type="number" step="0.1"><span class="target-unit"></span></span>' : ""}
        </div>` : ""}
        ${infusion ? '<div class="infusion-actions"></div>' : ""}
        <div class="bolus">
          <span class="unit-input"><input class="bolus-amount" type="number" min="0" max="1000" step="any"><span class="bolus-unit"></span></span>
          <button type="button" class="btn outlined">Give bolus</button>
        </div>
        <p class="medication-feedback" role="status" hidden></p>
        <p class="csht" hidden></p>
        </div>`;
      card.querySelector(".drug-name").textContent = spec.generic_name;
      card.querySelector(".drug-strength").textContent = spec.name.slice(spec.generic_name.length).trim();
      card.querySelector(".bolus-unit").textContent = spec.bolus_unit;

      const w = {
        card,
        spec,
        form: card.querySelector(".drug-controls"),
        mode: card.querySelector(".infusion-mode"),
        rate: card.querySelector(".rate"),
        target: card.querySelector(".target"),
        bolus: card.querySelector(".bolus-amount"),
        give: card.querySelector(".bolus .btn"),
        csht: card.querySelector(".csht"),
      };
      w.bolus.value = String(spec.default_bolus);
      if (spec.key === "dantrolene") {
        const total = document.createElement("span");
        total.id = "c-dantrolene-total";
        total.className = "drug-strength";
        card.querySelector("summary").append(total);
      }
      w.bolus.setAttribute("aria-label", `${spec.name} bolus (${spec.bolus_unit})`);

      const key = spec.key;
      if (infusion) {
        card.querySelector(".rate-unit").textContent = spec.rate_unit;
        w.rate.value = "0";
        w.rate.setAttribute("aria-label", `${spec.name} infusion rate (${spec.rate_unit})`);
      }
      if (tci) {
        card.querySelector(".target-label").textContent = spec.target_label;
        card.querySelector(".target-unit").textContent = spec.tci_unit;
        w.target.min = String(spec.tci_range[0]);
        w.target.max = String(spec.tci_range[1]);
        w.target.value = "0";
        w.target.setAttribute("aria-label", `${spec.name} ${spec.target_label} (${spec.tci_unit})`);
      }
      if (infusion) {
        const fields = tci ? [w.mode, w.rate, w.target] : [w.rate];
        w.editor = this.editSettings(fields, () => w.mode?.value === "tci"
          ? this.send("drug_target", { key, target: readNumber(w.target) })
          : this.send("drug_rate", { key, rate: readNumber(w.rate) }),
          card.querySelector(".infusion-actions"), () => {
            w.rate.disabled = w.editor.pending || w.mode?.value === "tci";
            if (w.target) w.target.disabled = w.editor.pending || w.mode.value !== "tci";
          });
        if (w.mode) w.mode.setAttribute("aria-label", `${spec.name} infusion mode`);
      }
      w.bolus.required = true;
      w.bolus.min = "0.001";
      w.give.onclick = () => {
        if (!w.bolus.reportValidity()) return;
        return this.performAction(w.give, "drug_bolus", { key, amount: readNumber(w.bolus) });
      };
      this.drugs[key] = w;
      this.medicationRows.set(key, {
        card, name: `${spec.name} ${medicationSearchTerms[key] ?? ""}`, feedback: card.querySelector(".medication-feedback"),
      });
      if (infusion) this.buildRunningInfusion(key, w);
      card.ontoggle = () => {
        if (!card.open) return;
        if (infusion) this.closeInfusionEditor(w);
        if (halfTimeDrugs.includes(key)) this.onMedicationsShown?.();
      };
    }
    const reversal = document.createElement("details");
    reversal.className = "drug-card";
    reversal.innerHTML = `
      <summary>Sugammadex</summary>
      <div class="drug-controls">
        <button type="button" class="btn outlined full" data-sugammadex="2">Moderate block: 2 mg/kg</button>
        <button type="button" class="btn outlined full" data-sugammadex="4">Deep block: 4 mg/kg</button>
        <button type="button" class="btn outlined full" data-sugammadex="16">Immediate reversal: 16 mg/kg</button>
        <p class="medication-feedback" role="status" hidden></p>
      </div>`;
    this.medicationRows.set("sugammadex", {
      card: reversal, name: "Sugammadex", feedback: reversal.querySelector(".medication-feedback"),
    });
    const calcium = document.createElement("details");
    calcium.className = "drug-card";
    calcium.innerHTML = `
      <summary>Calcium chloride</summary>
      <div class="drug-controls">
        <button type="button" class="btn outlined full" data-calcium="1000">Give 1 g</button>
        <p class="medication-feedback" role="status" hidden></p>
      </div>`;
    this.medicationRows.set("calcium_chloride", {
      card: calcium, name: `Calcium chloride ${medicationSearchTerms.calcium_chloride}`, feedback: calcium.querySelector(".medication-feedback"),
    });
    this.medicationRows.set("lipid", {
      card: lipid, name: `Lipid emulsion 20% ${medicationSearchTerms.lipid}`, feedback: $("c-lipid-feedback"),
    });
    for (const group of this.medicationGroups) {
      for (const key of group.keys) group.section.append(this.medicationRows.get(key).card);
    }
    for (const button of document.querySelectorAll("[data-sugammadex]")) {
      button.onclick = () => this.performAction(button, "sugammadex", { mg_per_kg: Number(button.dataset.sugammadex) });
    }
    for (const button of document.querySelectorAll("[data-calcium]")) {
      button.onclick = () => this.performAction(button, "calcium_chloride", { mg: Number(button.dataset.calcium) });
    }
    $("drug-search").oninput = () => this.filterMedications();
    $("drug-search-clear").onclick = () => {
      $("drug-search").value = "";
      this.filterMedications();
      $("drug-search").focus();
    };
    this.filterMedications();
  }

  filterMedications() {
    const query = $("drug-search").value.trim().toLowerCase();
    if (query && this.searchGroupState === null) {
      this.searchGroupState = this.medicationGroups.map(({ section }) => section.open);
    }
    let found = false;
    this.medicationGroups.forEach((group, index) => {
      let groupMatches = false;
      for (const key of group.keys) {
        const row = this.medicationRows.get(key);
        const matches = `${row.name} ${group.name}`.toLowerCase().includes(query);
        row.card.hidden = !matches;
        groupMatches ||= matches;
      }
      group.section.hidden = !groupMatches;
      if (query && groupMatches) group.section.open = true;
      if (!query && this.searchGroupState !== null) group.section.open = this.searchGroupState[index];
      found ||= groupMatches;
    });
    if (!query) this.searchGroupState = null;
    $("drug-search-clear").hidden = !$("drug-search").value;
    $("drug-search-empty").hidden = found;
  }

  buildRunningInfusion(key, w) {
    const row = document.createElement("div");
    row.className = "running-infusion";
    row.hidden = true;
    row.innerHTML = `
      <div><span class="infusion-name"></span><output></output></div>
      <button type="button" class="btn small outlined infusion-adjust">Adjust</button>
      <button type="button" class="btn small outlined infusion-stop">Stop</button>
      <div class="infusion-editor" hidden></div>`;
    row.querySelector(".infusion-name").textContent = w.spec.generic_name;
    row.querySelector(".infusion-name").title = w.spec.name;
    const adjust = row.querySelector(".infusion-adjust");
    adjust.setAttribute("aria-label", `Adjust ${w.spec.name}`);
    w.runningEditor = row.querySelector(".infusion-editor");
    w.runningEditor.id = `infusion-editor-${key}`;
    w.runningAdjust = adjust;
    adjust.setAttribute("aria-controls", w.runningEditor.id);
    adjust.setAttribute("aria-expanded", "false");
    adjust.onclick = () => {
      if (!w.runningEditor.hidden) {
        this.closeInfusionEditor(w);
        return;
      }
      w.card.open = false;
      w.runningEditor.append(w.form);
      w.runningEditor.hidden = false;
      adjust.setAttribute("aria-expanded", "true");
      const input = w.mode?.value === "tci" ? w.target : w.rate;
      input.focus({ preventScroll: true });
    };
    const stop = row.querySelector(".infusion-stop");
    stop.setAttribute("aria-label", `Stop ${w.spec.name}`);
    // A manual rate command also disables TCI in the shared engine.
    stop.onclick = async () => {
      const result = await this.performAction(stop, "drug_rate", { key, rate: 0 });
      if (result !== undefined) w.editor.cancel();
    };
    w.runningRow = row;
    w.runningValue = row.querySelector("output");
    $("running-infusion-list").append(row);
  }

  closeInfusionEditor(w) {
    if (w.runningEditor.hidden) return;
    w.card.append(w.form);
    w.runningEditor.hidden = true;
    w.runningAdjust.setAttribute("aria-expanded", "false");
  }

  needsCsht() {
    return this.currentTab === "Medications" && halfTimeDrugs.some((key) => this.drugs[key].card.open);
  }

  showCsht(values) {
    for (const key of halfTimeDrugs) {
      const label = this.drugs[key].csht;
      const minutes = values[key];
      label.hidden = !(minutes > 0);
      if (minutes > 0) {
        setText(label, `Effect-site half-time ~${minutes.toFixed(0)} min`);
        label.title = "Estimated from the pump's population model, which omits hand boluses; not a guaranteed wake-up time.";
      }
    }
  }

  // --- Fluids -------------------------------------------------------

  bindFluids(weight) {
    this.lipidMaximum = 12 * weight;
    $("c-lipid-feedback").hidden = true;
    $("c-lipid-bolus").value = String(weight >= 70 ? 100 : round(1.5 * weight));
    const lipidGive = $("c-lipid-give");
    lipidGive.onclick = async () => {
      const result = await this.performAction(lipidGive, "fluid", {
        kind: "lipid", volume_ml: readNumber($("c-lipid-bolus")),
      });
      if (this.destroyed || result === undefined) return;
      setText($("c-lipid-feedback"), result.fluid.text);
      $("c-lipid-feedback").hidden = false;
    };
    const lipidRate = $("c-lipid-rate");
    const lipidEditor = this.editSettings(
      [lipidRate], () => this.send("lipid_rate", { ml_min: readNumber(lipidRate) }), $("c-lipid-actions"),
    );
    $("c-lipid-stop").onclick = async () => {
      const result = await this.performAction($("c-lipid-stop"), "lipid_rate", { ml_min: 0 });
      if (!this.destroyed && result !== undefined) lipidEditor.cancel();
    };
    $("fluid-feedback").hidden = true;
    for (const button of document.querySelectorAll("[data-fluid]")) {
      button.onclick = async () => {
        const result = await this.performAction(button, "fluid", {
          kind: button.dataset.fluid, volume_ml: Number(button.dataset.volume),
        });
        if (this.destroyed || result === undefined) return;
        const feedback = $("fluid-feedback");
        setText(feedback, result.fluid.text);
        feedback.hidden = false;
        clearTimeout(this.fluidFeedbackTimer);
        this.fluidFeedbackTimer = setTimeout(() => { feedback.hidden = true; }, 6000);
      };
    }
    this.editSettings([$("c-maint")], () => this.send("maintenance_fluid", { ml_hr: readNumber($("c-maint")) }));
  }

  // --- Resuscitation ------------------------------------------------

  bindResuscitation() {
    $("c-compressions").onclick = () => this.send("compressions", { active: !this.state.compressions.active });
    this.synchronized = false;
    for (const button of $("c-sync").querySelectorAll("button")) {
      button.onclick = () => {
        this.synchronized = button.dataset.value === "true";
        for (const other of $("c-sync").querySelectorAll("button")) setPressed(other, other === button);
      };
    }
    const shock = $("c-shock");
    const charge = $("c-charge");
    charge.onclick = () => this.performAction(charge,
      this.state.defibrillator.status === "idle" ? "charge_defibrillator" : "disarm_defibrillator");
    $("c-shock-feedback").hidden = true;
    shock.onclick = async () => {
      const result = await this.performAction(shock, "shock", { synchronized: this.synchronized });
      if (this.destroyed || result === undefined) return;
      shock.disabled = true;
      this.showShockReceipt(result.shock);
    };
  }

  syncDefibrillator(defibrillator) {
    const status = defibrillator.status;
    setText($("c-defibrillator-status"), status === "charging"
      ? `Charging, ${Math.ceil(defibrillator.charge_remaining)} s remaining`
      : { idle: "Not charged", ready: "Ready, 200 J", waiting_for_r: "Waiting for an R wave" }[status]);
    setText($("c-charge"), status === "idle" ? "Charge" : "Disarm");
    $("c-shock").disabled = status !== "ready";
    for (const button of $("c-sync").querySelectorAll("button")) button.disabled = status === "waiting_for_r";
    const lastShock = defibrillator.last_shock;
    if (lastShock && lastShock.time !== this.lastShockTime) {
      this.lastShockTime = lastShock.time;
      this.showShockReceipt(lastShock);
    }
  }

  showShockReceipt(receipt) {
    const feedback = $("c-shock-feedback");
    setText(feedback, `${clock(receipt.time)} ${receipt.text}`);
    feedback.hidden = false;
    clearTimeout(this.shockFeedbackTimer);
    this.shockFeedbackTimer = setTimeout(() => { feedback.hidden = true; }, 6000);
  }

  // --- Labs ---------------------------------------------------------

  bindLabs() {
    const button = $("c-abg");
    button.onclick = async () => {
      const result = await this.performAction(button, "blood_gas");
      if (this.destroyed || result === undefined) return;
      this.bloodGases.push(result.blood_gas);
      this.showBloodGases();
    };
    this.showBloodGases();
  }

  // Results post at their ready time on the simulation clock.
  showBloodGases() {
    const ready = this.bloodGases.filter((result) => result.ready_at <= this.time);
    const pending = this.bloodGases.length - ready.length;
    const status = $("abg-pending");
    status.hidden = pending === 0;
    if (pending > 0) {
      const next = clock(Math.ceil(this.bloodGases[ready.length].ready_at - this.time));
      setText(status, pending === 1 ? `Result in ${next}` : `${pending} samples in analysis; next result in ${next}`);
    }
    if (ready.length === this.shownBloodGases) return;
    this.shownBloodGases = ready.length;
    $("abg-results").hidden = ready.length === 0;
    const newest = ready.reverse();
    const cell = (tag, text, scope) => {
      const element = document.createElement(tag);
      if (scope) element.scope = scope;
      element.textContent = text;
      return element;
    };
    const table = $("abg-results").querySelector("table");
    const header = document.createElement("tr");
    header.append(cell("th", "Drawn", "col"), ...newest.map((result) => cell("th", clock(result.drawn_at), "col")));
    table.tHead.replaceChildren(header);
    table.tBodies[0].replaceChildren(...bloodGasRows.map(([label, unit, key, format]) => {
      const row = document.createElement("tr");
      const name = cell("th", label, "row");
      if (unit) {
        const span = cell("span", unit);
        span.className = "unit";
        name.append(span);
      }
      row.append(name, ...newest.map((result) => cell("td", result[key] === null ? "—" : format(result[key]))));
      return row;
    }));
  }

  // --- Events -------------------------------------------------------

  bindEvents(info) {
    this.eventSections = new Map([...document.querySelectorAll(".event-section")].map((section) => {
      section.open = false;
      section.dataset.active = "false";
      section.ontoggle = () => this.onViewChanged?.();
      const summary = section.querySelector("summary");
      summary.setAttribute("aria-disabled", "false");
      summary.tabIndex = 0;
      summary.onclick = (event) => {
        if (section.dataset.active === "true") event.preventDefault();
      };
      return [section.dataset.event, { section, summary, output: summary.querySelector("output") }];
    }));
    $("c-o2-supply").onclick = () => this.send("oxygen_supply", { connected: !this.state.o2_connected });

    const profiles = $("c-disturbance");
    profiles.replaceChildren(new Option("Off", ""));
    for (const { key, label } of info.disturbances) profiles.append(new Option(label, key));
    profiles.onchange = () => this.send("disturbance_profile", { profile: profiles.value || null });
    $("c-disturb").onclick = () => this.send("disturbance", { active: !this.state.disturbance.active });

    for (const [id, command] of [["c-obstruction", "obstruction"], ["c-bronchospasm", "bronchospasm"]]) {
      this.editSettings([$(id)], () => this.send(command, { percent: readNumber($(id)) }), $(id).parentElement);
    }
    $("c-auto-laryngo").onchange = () => this.send("auto_laryngospasm", { enabled: $("c-auto-laryngo").checked });

    $("c-hemorrhage").onclick = () => this.send("hemorrhage", {
      active: !this.state.hemorrhage.active,
      rate_ml_min: Number($("c-hem-rate").value),
    });
    const rhythms = $("c-rhythm");
    rhythms.replaceChildren(...info.rhythms.map((name) => new Option(name, name)));
    rhythms.onchange = () => this.send("rhythm", { name: rhythms.value });
    $("c-position-row").hidden = info.patient.gestational_age_weeks === 0;
    $("c-position").onchange = () => this.send("position", { position: $("c-position").value });
    $("c-anaphylaxis").onclick = () => this.send("anaphylaxis", { active: !this.state.anaphylaxis });
    $("c-sepsis").onclick = () => this.send("sepsis", { active: !this.state.sepsis });
    $("c-mh").onclick = () => this.send("malignant_hyperthermia", { active: !this.state.malignant_hyperthermia });
  }

  syncRhythm(rhythm) {
    const select = $("c-rhythm");
    const observed = select.querySelector("option[data-observed]");
    if (observed && observed.value !== rhythm) observed.remove();
    if (!Array.from(select.options).some(option => option.value === rhythm)) {
      const option = new Option(rhythm, rhythm);
      option.disabled = true;
      option.dataset.observed = "true";
      select.add(option);
    }
    // Rhythm is a live observation, including while this control has focus.
    select.value = rhythm;
  }

  syncEventSection(name, states, listStates = true) {
    const { section, summary, output } = this.eventSections.get(name);
    const active = states.length > 0;
    const value = String(active);
    if (section.dataset.active !== value) {
      section.dataset.active = value;
      summary.setAttribute("aria-disabled", value);
      summary.tabIndex = active ? -1 : 0;
    }
    if (active && !section.open) section.open = true;
    output.hidden = !active || !listStates;
    setText(output, states.join(", "));
  }

  // --- Sync from snapshots ------------------------------------------

  showClinicalFeedback(feedback) {
    const panel = $("clinical-feedback");
    panel.hidden = !feedback;
    panel.dataset.level = feedback?.level ?? "";
    setText($("clinical-feedback-text"), feedback?.message ?? "");
  }

  sync(c, time) {
    this.state = c;
    const observations = [];
    if (c.convulsions) observations.push("Generalized convulsions");
    if (c.muscle_rigidity) observations.push("Generalized muscle rigidity");
    $("patient-observation").hidden = observations.length === 0;
    setText($("patient-observation"), observations.join("; "));
    setText($("c-dantrolene-total"), c.dantrolene_total_mg > 0
      ? `${round(c.dantrolene_total_mg, 0)} mg given (${round(c.dantrolene_total_mg / this.weight, 1)} mg/kg)` : "");
    setPressed($("c-cooling"), c.surface_cooling, c.surface_cooling ? "Stop surface cooling" : "Start surface cooling");
    setValue($("c-lipid-rate"), round(c.lipid.rate));
    $("c-lipid-stop").hidden = c.lipid.rate <= 0;
    setText($("c-lipid-total"), c.lipid.rate <= 0 && c.lipid.total_ml <= 0 ? ""
      : (c.lipid.rate > 0 ? `${round(c.lipid.rate)} mL/min, ` : "") + `${round(c.lipid.total_ml)} mL given`
        + (c.lipid.total_ml > this.lipidMaximum ? "; exceeds 12 mL/kg" : ""));
    this.time = time;
    this.showBloodGases();
    this.showClinicalFeedback(c.clinical_feedback);

    for (const button of $("c-airway").querySelectorAll("button")) {
      setPressed(button, button.dataset.value === c.airway);
    }
    if (c.airway !== "Mask") this.maskSealEditor.cancel();
    $("c-mask-seal-row").hidden = c.airway !== "Mask";
    setValue($("c-mask-seal"), round(c.mask_seal, 0));
    setValue($("c-o2"), round(c.fgf.o2));
    setValue($("c-air"), round(c.fgf.air));
    setValue($("c-n2o"), round(c.fgf.n2o));
    const fio2 = $("c-fio2");
    setText(fio2, `${Math.round(c.circuit_fio2 * 100)}%`);
    fio2.className = "fio2" + (!c.o2_connected || c.circuit_fio2 < 0.21 ? " bad" : c.circuit_fio2 < 0.3 ? " warn" : "");
    setPressed($("c-o2-supply"), !c.o2_connected, c.o2_connected ? "Disconnect O₂ supply" : "Connect backup O₂");
    setValue($("c-vap"), round(c.vaporizer));

    const compressions = c.compressions;
    setPressed($("c-compressions"), compressions.active, compressions.active ? "Stop compressions" : "Start compressions");
    $("c-compressions-status").hidden = !compressions.active;
    if (compressions.active) setText($("c-compressions-status"), `Compressing for ${clock(time - compressions.since)}`);
    this.syncDefibrillator(c.defibrillator);

    setPressed($("c-bag"), c.bag_mask, c.bag_mask ? "Stop bag ventilation" : "Start bag ventilation");
    $("c-bag-settings").hidden = !c.bag_mask;
    setValue($("c-bag-rr"), c.bag.rr);
    setValue($("c-bag-tv"), c.bag.tv);
    setValue($("c-bag-pmax"), c.bag.p_max);
    const vent = c.vent;
    setPressed($("c-vent-power"), vent.on, vent.on ? "Stop ventilator" : "Start ventilator");
    setValue($("c-vent-mode"), vent.mode);
    this.applyVentMode($("c-vent-mode").value);
    for (const [name, id] of Object.entries(ventInputs)) setValue($(id), vent[name]);
    setValue($("c-ie"), vent.ie);

    let running = false;
    for (const [key, w] of Object.entries(this.drugs)) {
      const d = c.drugs[key];
      if (w.mode) {
        setValue(w.mode, d.is_tci ? "tci" : "rate");
        setValue(w.target, round(d.target));
        w.target.disabled = w.editor.pending || w.mode.value !== "tci";
      }
      if (w.rate) {
        w.rate.disabled = w.editor.pending || w.mode?.value === "tci";
        setValue(w.rate, d.is_tci ? shortRate(d.rate) : rateText(d.rate));
        const active = d.rate > 0 || (d.is_tci && d.target > 0);
        if (!active && w.runningRow.contains(document.activeElement)) $("drug-search").focus();
        if (!active) this.closeInfusionEditor(w);
        w.runningRow.hidden = !active;
        if (active) {
          setText(w.runningValue, d.is_tci
            ? `${w.spec.target_label}: ${round(d.target)} ${w.spec.tci_unit}, rate ${shortRate(d.rate)} ${w.spec.rate_unit}`
            : `${shortRate(d.rate)} ${w.spec.rate_unit}`);
        }
        running ||= active;
      }
    }
    $("running-infusions").hidden = !running;

    setValue($("c-bair"), String(c.bair_hugger));
    setValue($("c-maint"), round(c.maintenance_ml_hr, 0));
    const profile = c.disturbance.profile ?? "";
    setValue($("c-disturbance"), profile);
    $("c-disturbance").disabled = c.disturbance.active;
    $("c-disturb").disabled = !profile;
    setPressed($("c-disturb"), c.disturbance.active, c.disturbance.active ? "Stop stimulation" : "Start stimulation");
    setValue($("c-obstruction"), round(c.obstruction, 0));
    setValue($("c-bronchospasm"), round(c.bronchospasm, 0));
    setText($("c-laryngo"), `Laryngospasm: ${c.laryngospasm}`);
    $("c-auto-laryngo").checked = c.auto_laryngospasm;

    setPressed($("c-hemorrhage"), c.hemorrhage.active, c.hemorrhage.active ? "Stop bleeding" : "Start bleeding");
    $("c-hem-rate").disabled = c.hemorrhage.active;
    if (c.hemorrhage.active) setValue($("c-hem-rate"), String(c.hemorrhage.rate));
    this.syncRhythm(c.ecg_rhythm);
    setValue($("c-position"), c.position);
    setPressed($("c-anaphylaxis"), c.anaphylaxis, c.anaphylaxis ? "Stop anaphylaxis" : "Start anaphylaxis");
    setPressed($("c-sepsis"), c.sepsis, c.sepsis ? "Stop sepsis" : "Start sepsis");
    setPressed($("c-mh"), c.malignant_hyperthermia,
      c.malignant_hyperthermia ? "Stop malignant hyperthermia" : "Start malignant hyperthermia");

    const circulation = [];
    if (c.hemorrhage.active) circulation.push(`Bleeding ${c.hemorrhage.rate} mL/min`);
    if (c.anaphylaxis) circulation.push("Anaphylaxis");
    if (c.sepsis) circulation.push("Sepsis");
    if (c.ecg_rhythm !== "Sinus rhythm") circulation.push(c.ecg_rhythm);
    this.syncEventSection("circulation", circulation);
    this.syncEventSection("metabolism", c.malignant_hyperthermia ? ["Malignant hyperthermia"] : []);
    const airway = [];
    if (c.obstruction > 0) airway.push(`Obstruction ${round(c.obstruction, 0)}%`);
    if (c.bronchospasm > 0) airway.push(`Bronchospasm ${round(c.bronchospasm, 0)}%`);
    if (c.laryngospasm !== "none") airway.push(`Laryngospasm ${c.laryngospasm}`);
    this.syncEventSection("airway", airway);
    // The disabled profile select already names the active stimulation.
    this.syncEventSection("stimulation", c.disturbance.active ? [$("c-disturbance").selectedOptions[0].text] : [], false);
    this.syncEventSection("equipment", c.o2_connected ? [] : ["O₂ supply disconnected"]);
  }
}
