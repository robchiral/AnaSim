// Machine, medication, circulation, and event controls.
// Numeric settings remain drafts until explicitly applied.
import { $, minutesSeconds, setText } from "./display.js";
import { SettingsEditor } from "./settings.js";

const editors = new WeakMap();

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
const signed = (value) => {
  const text = value.toFixed(1);
  return Number(text) > 0 ? `+${text}` : Number(text) === 0 ? "0.0" : text;
};
// Blood gas report rows: label, unit, field, formatter, adult reference range, [low, high].
const bloodGasRows = [
  ["pH", "", "ph", (value) => value.toFixed(2), "7.35–7.45", 7.35, 7.45],
  ["PaCO₂", "mmHg", "pa_co2", (value) => value.toFixed(0), "35–45", 35, 45],
  ["PaO₂", "mmHg", "pao2", (value) => value.toFixed(0), "≥80", 80, null],
  ["HCO₃⁻", "mmol/L", "hco3", (value) => value.toFixed(1), "22–26", 22, 26],
  ["Base excess", "mmol/L", "base_excess", signed, "−2 to +2", -2, 2],
  ["Lactate", "mmol/L", "lactate", (value) => value.toFixed(1), "<2.0", null, 2.0],
  ["Na⁺", "mmol/L", "sodium", (value) => value.toFixed(0), "135–145", 135, 145],
  ["K⁺", "mmol/L", "potassium", (value) => value.toFixed(1), "3.5–5.0", 3.5, 5.0],
  ["Cl⁻", "mmol/L", "chloride", (value) => value.toFixed(0), "98–107", 98, 107],
  ["Ionized Ca²⁺", "mmol/L", "ionized_calcium", (value) => value.toFixed(2), "1.15–1.30", 1.15, 1.30],
  ["Glucose", "mg/dL", "glucose", (value) => value.toFixed(0), "70–140", 70, 140],
  ["SaO₂", "%", "sao2", (value) => value.toFixed(1), "≥95", 95, null],
  ["Hb", "g/dL", "hb", (value) => value.toFixed(1), "12.0–17.5", 12.0, 17.5],
  ["FiO₂", "%", "fio2", (value) => (value * 100).toFixed(0), "—", null, null],
];

// ISO 26825 / ASTM D4774 syringe-label drug color categories.
const ISO_DRUG_CLASS = {
  propofol: "induction",
  etomidate: "induction",
  ketamine: "induction",
  midazolam: "benzo",
  fentanyl: "opioid",
  remi: "opioid",
  hydromorphone: "opioid",
  morphine: "opioid",
  roc: "nmba",
  succinylcholine: "nmba",
  neostigmine: "reversal",
  sugammadex: "reversal",
  phenyl: "vasopressor",
  ephedrine: "vasopressor",
  nore: "vasopressor",
  vaso: "vasopressor",
  dobu: "vasopressor",
  milri: "vasopressor",
  esmolol: "vasopressor",
  labetalol: "vasopressor",
  epi: "vasopressor",
  atropine: "anticholinergic",
  glyco: "anticholinergic",
  lidocaine: "local",
  amiodarone: "local",
  dantrolene: "rescue",
  lipid: "rescue",
  calcium_chloride: "rescue",
  sodium_bicarbonate: "rescue",
  heparin: "local",
  protamine: "reversal",
  insulin: "rescue",
  d50w: "rescue",
  d10w: "rescue",
};
// Four significant figures keep small rates visible without long decimals.
const rateText = (value) => String(Number(value.toPrecision(4)));
// Displayed rates: whole numbers from 10 up, two significant figures below.
const shortRate = (value) => (value >= 10 ? Math.round(value) : Number(value.toPrecision(2)));

// Settings each ventilator mode uses (GE Carestation mode layout).
// PSV uses RR, Pinsp, and Tinsp for apnea backup breaths.
const ventFields = {
  "VCV": ["rr", "tv", "ie", "pause", "p_max"],
  "PCV": ["rr", "p_insp", "ie"],
  "PCV-VG": ["rr", "tv", "ie", "p_max"],
  "SIMV-VC": ["rr", "tv", "t_insp", "pause", "p_max", "p_support", "trigger"],
  "SIMV-PC": ["rr", "p_insp", "t_insp", "p_support", "trigger"],
  "SIMV-VG": ["rr", "tv", "t_insp", "p_max", "p_support", "trigger"],
  "PSV": ["p_support", "trigger", "rr", "p_insp", "t_insp"],
  "CPAP": [],
};
const ventInputs = {
  rr: "c-rr", tv: "c-tv", p_insp: "c-pinsp", p_support: "c-psupport", peep: "c-peep",
  t_insp: "c-tinsp", pause: "c-pause", p_max: "c-pmax", trigger: "c-trigger",
};

// Presentation order stays fixed while medications are administered.
// Routine anesthesia groups open by default; specialty and rescue groups start collapsed.
const medicationGroups = [
  { name: "Anesthesia and sedation", keys: ["propofol", "midazolam", "ketamine", "etomidate"], open: true },
  { name: "Opioids", keys: ["fentanyl", "remi", "hydromorphone", "morphine"], open: true },
  { name: "Neuromuscular blockade and reversal", keys: ["roc", "succinylcholine", "neostigmine", "sugammadex"], open: true },
  { name: "Vasopressors and inotropes", keys: ["phenyl", "ephedrine", "nore", "epi", "vaso", "dobu", "milri"], open: true },
  { name: "Heart rate and blood pressure", keys: ["atropine", "glyco", "esmolol", "labetalol"], open: true },
  { name: "Antiarrhythmics", keys: ["lidocaine", "amiodarone"], open: false },
  { name: "Anticoagulation and reversal", keys: ["heparin", "protamine"], open: false },
  { name: "Endocrine and metabolic", keys: ["insulin", "d50w", "d10w"], open: false },
  { name: "Rescue and antidotes", keys: ["dantrolene", "lipid", "calcium_chloride", "sodium_bicarbonate"], open: false },
];
const neuraxialSearchText = "neuraxial anesthesia spinal epidural bupivacaine lidocaine block level";
const medicationSearchTerms = {
  neostigmine: "reversal anticholinesterase prostigmin",
  dantrolene: "malignant hyperthermia MH",
  lipid: "local anesthetic systemic toxicity LAST",
  calcium_chloride: "hypocalcemia hyperkalemia CaCl2",
  sodium_bicarbonate: "NaHCO3 bicarb 8.4% acidosis hyperkalemia LAST toxicity alkalinization",
  heparin: "UFH anticoagulation ACT CPB ECMO",
  protamine: "heparin reversal ACT CPB",
  insulin: "regular hyperglycemia hyperkalemia glucose diabetes",
  d50w: "dextrose 50% D50 D50W glucose hypoglycemia hyperkalemia 25 g",
  d10w: "dextrose 10% D10 D10W glucose hypoglycemia 10 g",
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
    this.acts = [...(info.acts ?? [])];
    this.shownActs = null;
    this.glucoses = [...(info.glucoses ?? [])];
    this.shownGlucoses = null;
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
    this.bindExtracorporeal();
    this.bindAirwayTools();
    this.bindNeuraxial();
    this.bindCrisisProcedures();
    this.bindLabs();
    this.bindEvents(info);
    this.pregnant = info.patient.gestational_age_weeks > 0;
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
    clearTimeout(this.ecmoFluidFeedbackTimer);
    clearTimeout(this.shockFeedbackTimer);
    clearTimeout(this.crisisProcFeedbackTimer);
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
      if (!this.ended) {
        button.disabled = false;
        if (this.state?.defibrillator && (button.id === "c-shock" || button.id === "c-charge")) {
          this.syncDefibrillator(this.state.defibrillator);
        } else if (this.state && button.id === "c-bag-breath") {
          button.disabled = this.state.airway === "None" || this.state.vent.on || this.state.bag_mask;
        } else if (this.state && button.id === "c-o2-flush") {
          button.disabled = !this.state.o2_connected;
        }
      }
    }
  }

  showMedicationHistory() {
    const list = $("medication-history-list");
    list.replaceChildren();
    $("medication-history").hidden = this.history.length === 0;
    for (const entry of this.history) {
      const row = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = minutesSeconds(entry.time);
      row.append(time, entry.text);
      list.prepend(row);
    }
  }

  showMedicationFeedback(entry) {
    const feedback = this.medicationRows.get(entry.key)?.feedback ?? (entry.key === "neuraxial" ? $("c-neuraxial-readout") : null);
    if (!feedback) return;
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
    this.tabList = document.querySelector(".tabs");
    this.tabs = [...this.tabList.querySelectorAll("button")];
    for (const tab of this.tabs) tab.onclick = () => this.openTab(tab.dataset.tab);
    this.tabList.onkeydown = (event) => {
      const visible = this.tabs.filter((tab) => !tab.hidden);
      const index = visible.indexOf(event.target);
      if (index === -1) return;
      let next;
      switch (event.key) {
        case "ArrowLeft": next = visible.at(index - 1); break;
        case "ArrowRight": next = visible[(index + 1) % visible.length]; break;
        case "Home": next = visible[0]; break;
        case "End": next = visible.at(-1); break;
        default: return;
      }
      event.preventDefault();
      this.openTab(next.dataset.tab);
      next.focus();
    };
    this.openTab("Machine");
  }

  openTab(name) {
    const tabChanged = this.currentTab !== name;
    for (const tab of this.tabs) {
      const selected = tab.dataset.tab === name;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    }
    for (const panel of document.querySelectorAll(".tab-panel")) panel.hidden = panel.dataset.panel !== name;
    this.currentTab = name;
    if (name === "Medications") this.onMedicationsShown?.();
    this.onViewChanged?.(tabChanged);
  }

  // --- Machine ------------------------------------------------------

  bindMachine() {
    $("c-bair").onchange = () => this.send("bair_hugger", { target_c: Number($("c-bair").value) });
    $("c-fluid-warmer").onclick = () => this.send("fluid_warmer", { enabled: !this.state.fluid_warmer });
    $("c-cooling").onclick = () => this.send("surface_cooling", { on: !this.state.surface_cooling });
    this.airwayButtons = [...$("c-airway").querySelectorAll("button")];
    for (const button of this.airwayButtons) {
      button.onclick = () => {
        const mode = button.dataset.value;
        if (mode === "ETT" && $("c-airway-technique")) {
          return this.send("airway", { mode, technique: $("c-airway-technique").value });
        }
        return this.send("airway", { mode });
      };
    }
    this.maskSealEditor = this.editSettings(
      [$("c-mask-seal")],
      () => this.send("mask_seal", { percent: readNumber($("c-mask-seal")) }),
      $("c-mask-seal").closest(".airway-extras"),
    );
    const fgf = () => this.send("fgf", { o2: readNumber($("c-o2")), air: readNumber($("c-air")), n2o: readNumber($("c-n2o")) });
    this.editSettings([$("c-o2"), $("c-air"), $("c-n2o")], fgf, $("c-o2").closest(".fresh-gas-settings"));
    this.editSettings([$("c-vap")], () => this.send("vaporizer", { percent: readNumber($("c-vap")) }), $("c-vap-actions"));
    const vapRefill = $("c-vap-refill");
    if (vapRefill) vapRefill.onclick = () => this.performAction(vapRefill, "refill_vaporizer");
    $("c-bag").onclick = () => this.send("bag_mask", { active: !this.state.bag_mask });
    $("c-bag-breath").onclick = () => this.performAction($("c-bag-breath"), "bag_breath");
    $("c-o2-flush").onclick = () => this.performAction($("c-o2-flush"), "o2_flush");
    this.editSettings([$("c-apl")], () => this.send("apl", { cmh2o: readNumber($("c-apl")) }), $("c-apl").closest(".manual-ventilation"));
    this.editSettings(
      [$("c-bag-rr"), $("c-bag-tv")],
      () => this.send("bag_settings", {
        rr: readNumber($("c-bag-rr")), tv: readNumber($("c-bag-tv")),
      }),
      $("c-bag-settings"),
    );
    $("c-vent-power").onclick = () => this.send("vent_power", { on: !this.state.vent.on });
    this.ventEditor = this.editSettings(
      ["c-vent-mode", "c-ie", ...Object.values(ventInputs)].map($),
      () => {
        const mode = $("c-vent-mode").value;
        const used = ventFields[mode];
        return this.send("vent", {
          mode,
          ...(used.includes("ie") ? { ie: $("c-ie").value } : {}),
          peep: readNumber($("c-peep")),
          ...Object.fromEntries(used.filter((n) => n !== "ie").map((n) => [n, readNumber($(ventInputs[n]))])),
        });
      },
      $("c-vent-settings"),
      () => this.applyVentMode($("c-vent-mode").value),
    );
  }

  applyVentMode(mode) {
    const pending = Boolean(this.ventEditor?.pending);
    if (this.displayedVentMode === mode && this.displayedVentPending === pending) return;
    const used = ventFields[mode];
    for (const el of document.querySelectorAll("#c-vent-settings [data-vent]")) {
      const unused = !used.includes(el.dataset.vent);
      el.hidden = unused;
      if (!pending) {
        for (const input of el.querySelectorAll?.("input, select") ?? []) {
          input.disabled = unused;
        }
      }
    }
    this.displayedVentPending = pending;
    if (this.displayedVentMode === mode) return;
    this.displayedVentMode = mode;
    $("c-rr-label").textContent = mode === "PSV" ? "Backup RR" : "RR";
    $("c-pinsp-label").textContent = mode === "PSV" ? "Backup Pinsp" : "Pinsp";
  }

  // --- Medications --------------------------------------------------

  buildDrugs(drugs) {
    const container = $("drug-cards");
    const lipid = $("lipid-controls");
    lipid.open = false;
    container.replaceChildren();
    $("running-infusion-list").replaceChildren();
    $("running-infusions").hidden = true;
    $("drug-search").value = "";
    this.searchGroupState = null;
    this.medicationRows = new Map();
    this.medicationGroups = medicationGroups.map(({ name, keys, open }) => {
      const section = document.createElement("details");
      section.className = "medication-group";
      section.open = open;
      const heading = document.createElement("summary");
      const label = document.createElement("span");
      label.textContent = name;
      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = String(keys.length);
      heading.append(label, count);
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
      card.dataset.iso = ISO_DRUG_CLASS[spec.key] ?? "local";
      const idPrefix = `drug-${spec.key}`;
      card.innerHTML = `
        <summary><span class="drug-name"></span><span class="drug-strength"></span></summary>
        <div class="drug-controls">
        ${tci ? `<div class="control-setting">
          <div class="control-setting-head"><label for="${idPrefix}-mode">Infusion mode</label></div>
          <select id="${idPrefix}-mode" class="infusion-mode"><option value="rate">Manual rate</option><option value="tci">TCI</option></select>
        </div>` : ""}
        ${infusion ? `<div class="inputs${tci ? "" : " manual-rate"}">
          <div class="control-setting">
            <div class="control-setting-head"><label for="${idPrefix}-rate">Infusion rate</label><span class="rate-unit"></span></div>
            <input id="${idPrefix}-rate" class="rate" type="number" min="0" max="2000" step="any">
          </div>
          ${tci ? `<div class="control-setting">
            <div class="control-setting-head"><label for="${idPrefix}-target" class="target-label"></label><span class="target-unit"></span></div>
            <input id="${idPrefix}-target" class="target" type="number" step="0.1">
          </div>` : ""}
        </div>` : ""}
        ${infusion ? '<div class="infusion-actions"></div>' : ""}
        <div class="bolus">
          <div class="bolus-field">
            <div class="control-setting-head"><label for="${idPrefix}-bolus">Bolus</label><span class="bolus-unit"></span></div>
            <input id="${idPrefix}-bolus" class="bolus-amount" type="number" min="0" max="1000" step="any">
          </div>
          <button type="button" class="btn">Give bolus</button>
        </div>
        <p class="medication-feedback" role="status" hidden></p>
        <p class="csht" hidden></p>
        </div>`;
      card.querySelector(".drug-name").textContent = spec.generic_name;
      card.querySelector(".drug-strength").textContent = spec.name.startsWith(spec.generic_name)
        ? spec.name.slice(spec.generic_name.length).trim()
        : spec.name;
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
      w.bolus.max = String(spec.max_bolus ?? 1000);
      if (spec.key === "dantrolene") {
        const total = document.createElement("span");
        total.id = "c-dantrolene-total";
        total.className = "drug-strength drug-total";
        card.querySelector("summary").append(total);
      }
      w.bolus.setAttribute("aria-label", `${spec.name} bolus (${spec.bolus_unit})`);

      const key = spec.key;
      if (infusion) {
        card.querySelector(".rate-unit").textContent = spec.rate_unit;
        w.rate.value = "0";
        w.rate.max = String(spec.max_rate ?? 2000);
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
          card.querySelector(".infusion-actions"), () => this.syncDrugInputs(w));
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
    reversal.dataset.iso = ISO_DRUG_CLASS.sugammadex;
    reversal.innerHTML = `
      <summary><span class="drug-name">Sugammadex</span></summary>
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
    calcium.dataset.iso = ISO_DRUG_CLASS.calcium_chloride;
    calcium.innerHTML = `
      <summary><span class="drug-name">Calcium chloride</span><span class="drug-strength">10%, 100 mg/mL</span></summary>
      <div class="drug-controls">
        <button type="button" class="btn outlined full" data-calcium="1000">Give 1 g</button>
        <p class="medication-feedback" role="status" hidden></p>
      </div>`;
    this.medicationRows.set("calcium_chloride", {
      card: calcium, name: `Calcium chloride ${medicationSearchTerms.calcium_chloride}`, feedback: calcium.querySelector(".medication-feedback"),
    });
    const addBolusCard = ({ key, label, strength, unit, defaultValue, max, onGive }) => {
      const card = document.createElement("details");
      card.className = "drug-card";
      card.dataset.iso = ISO_DRUG_CLASS[key] ?? "rescue";
      const inputId = `drug-${key}-bolus`;
      card.innerHTML = `
        <summary><span class="drug-name"></span><span class="drug-strength"></span></summary>
        <div class="drug-controls">
          <div class="bolus">
            <div class="bolus-field">
              <div class="control-setting-head"><label for="${inputId}">Bolus</label><span class="bolus-unit"></span></div>
              <input id="${inputId}" class="bolus-amount" type="number" min="0.001" max="${max}" step="any" required>
            </div>
            <button type="button" class="btn">Give bolus</button>
          </div>
          <p class="medication-feedback" role="status" hidden></p>
        </div>`;
      card.querySelector(".drug-name").textContent = label;
      card.querySelector(".drug-strength").textContent = strength;
      card.querySelector(".bolus-unit").textContent = unit;
      const input = card.querySelector(".bolus-amount");
      const give = card.querySelector(".bolus .btn");
      input.value = String(defaultValue);
      input.setAttribute("aria-label", `${label} bolus (${unit})`);
      give.onclick = () => {
        if (input.reportValidity && !input.reportValidity()) return;
        return onGive(give, readNumber(input));
      };
      this.medicationRows.set(key, {
        card,
        name: `${label} ${strength} ${medicationSearchTerms[key] ?? ""}`,
        feedback: card.querySelector(".medication-feedback"),
      });
    };
    addBolusCard({
      key: "d50w",
      label: "Dextrose 50%",
      strength: "D50W · 0.5 g/mL",
      unit: "mL",
      defaultValue: 50,
      max: 500,
      onGive: (button, volume_ml) => this.performAction(button, "fluid", { kind: "d50w", volume_ml }),
    });
    addBolusCard({
      key: "d10w",
      label: "Dextrose 10%",
      strength: "D10W · 0.1 g/mL",
      unit: "mL",
      defaultValue: 100,
      max: 2000,
      onGive: (button, volume_ml) => this.performAction(button, "fluid", { kind: "d10w", volume_ml }),
    });
    addBolusCard({
      key: "sodium_bicarbonate",
      label: "Sodium bicarbonate",
      strength: "8.4%, 1 mEq/mL",
      unit: "mEq",
      defaultValue: 50,
      max: 500,
      onGive: (button, meq) => this.performAction(button, "sodium_bicarbonate", { meq }),
    });
    this.medicationRows.set("lipid", {
      card: lipid, name: `Lipid emulsion 20% ${medicationSearchTerms.lipid}`, feedback: $("c-lipid-feedback"),
    });
    for (const group of this.medicationGroups) {
      const groupLower = group.name.toLowerCase();
      for (const key of group.keys) {
        const row = this.medicationRows.get(key);
        row.searchText = `${row.name.toLowerCase()} ${groupLower}`;
        group.section.append(row.card);
      }
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
    const neuraxial = $("neuraxial-module");
    if (query && this.searchGroupState === null) {
      this.searchGroupState = this.medicationGroups.map(({ section }) => section.open);
      this.searchNeuraxialOpen = neuraxial?.open ?? false;
    }
    let foundDrugs = false;
    this.medicationGroups.forEach((group, index) => {
      let groupMatches = false;
      for (const key of group.keys) {
        const row = this.medicationRows.get(key);
        const matches = !query || row.searchText.includes(query);
        row.card.hidden = !matches;
        groupMatches ||= matches;
      }
      group.section.hidden = !groupMatches;
      if (query && groupMatches) group.section.open = true;
      if (!query && this.searchGroupState !== null) group.section.open = this.searchGroupState[index];
      foundDrugs ||= groupMatches;
    });
    let found = foundDrugs;
    if (neuraxial) {
      const neuraxialMatches = !query || neuraxialSearchText.includes(query);
      neuraxial.hidden = !neuraxialMatches;
      if (query && neuraxialMatches && !foundDrugs) neuraxial.open = true;
      if (!query && this.searchGroupState !== null) neuraxial.open = Boolean(this.searchNeuraxialOpen);
      found ||= neuraxialMatches;
    }
    if (!query) this.searchGroupState = null;
    $("drug-search-clear").hidden = !$("drug-search").value;
    $("drug-search-empty").hidden = found;
  }

  buildRunningInfusion(key, w) {
    const row = document.createElement("div");
    row.className = "running-infusion";
    if (row.dataset) row.dataset.iso = ISO_DRUG_CLASS[key] ?? "local";
    row.hidden = true;
    row.innerHTML = `
      <div class="pump-info">
        <span class="infusion-name"></span>
        <output></output>
      </div>
      <button type="button" class="btn outlined infusion-adjust">Adjust</button>
      <button type="button" class="btn outlined infusion-stop">Stop</button>
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
    // Stop TCI via target reset so receipts and mode reflect TCI rather than manual rate.
    stop.onclick = async () => {
      const [action, payload] = this.state?.drugs?.[key]?.is_tci
        ? ["drug_target", { key, target: null }]
        : ["drug_rate", { key, rate: 0 }];
      if (await this.performAction(stop, action, payload) !== undefined) w.editor.cancel();
    };
    w.runningRow = row;
    w.runningValue = row.querySelector("output");
    $("running-infusion-list").append(row);
  }

  syncDrugInputs(w) {
    const tciMode = w.mode?.value === "tci";
    w.rate.disabled = w.editor.pending || tciMode;
    if (w.target) w.target.disabled = w.editor.pending || !tciMode;
  }

  closeInfusionEditor(w) {
    if (w.runningEditor.hidden) return;
    w.card.append(w.form);
    w.runningEditor.hidden = true;
    w.runningAdjust.setAttribute("aria-expanded", "false");
  }

  needsCsht() {
    return this.currentTab === "Medications" && halfTimeDrugs.some(
      (key) => this.drugs[key].card.open || !this.drugs[key].runningEditor?.hidden,
    );
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
      if ($("c-lipid-bolus").reportValidity && !$("c-lipid-bolus").reportValidity()) return;
      const result = await this.performAction(lipidGive, "fluid", {
        kind: "lipid", volume_ml: readNumber($("c-lipid-bolus")),
      });
      if (this.destroyed || result === undefined || result.medication) return;
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
    this.lastShockTime = null;
    this.lastDefibStatus = "idle";
    this.syncButtons = [...$("c-sync").querySelectorAll("button")];
    for (const button of this.syncButtons) {
      setPressed(button, button.dataset.value === "false");
      button.onclick = () => {
        this.synchronized = button.dataset.value === "true";
        const syncBadge = $("defib-sync-badge");
        if (syncBadge) syncBadge.hidden = !this.synchronized;
        for (const other of this.syncButtons) setPressed(other, other === button);
      };
    }
    const syncBadge = $("defib-sync-badge");
    if (syncBadge) syncBadge.hidden = true;
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
    const lastShock = defibrillator.last_shock;
    const shockDelivered = Boolean(lastShock && lastShock.time !== this.lastShockTime);
    const defibModule = $("defib-module");
    if (defibModule?.dataset) defibModule.dataset.status = status;
    const syncButtons = (this.syncButtons ??= [...$("c-sync").querySelectorAll("button")]);
    if (defibrillator.synchronized || status === "waiting_for_r") {
      this.synchronized = true;
      for (const button of syncButtons) setPressed(button, button.dataset.value === "true");
    } else if (shockDelivered || (this.lastDefibStatus === "waiting_for_r" && status === "idle")) {
      // Auto-reset synchronized cardioversion after shock delivery or disarm so
      // a subsequent shock in ventricular fibrillation cannot wait for an R wave.
      this.synchronized = false;
      for (const button of syncButtons) setPressed(button, button.dataset.value === "false");
    }
    this.lastDefibStatus = status;
    const syncBadge = $("defib-sync-badge");
    if (syncBadge) syncBadge.hidden = !this.synchronized;
    const chargeTrack = $("defib-charge-track");
    if (chargeTrack) chargeTrack.hidden = status === "idle";
    const chargeBar = $("defib-charge-bar");
    if (chargeBar?.style) {
      const fill = status === "idle" ? 0
        : status === "charging" ? Math.min(100, Math.max(8, (1 - defibrillator.charge_remaining / 5) * 100))
        : 100;
      chargeBar.style.setProperty("--fill", `${fill.toFixed(0)}%`);
    }
    setText($("c-defibrillator-status"), status === "charging"
      ? `Charging, ${Math.ceil(Math.max(0, defibrillator.charge_remaining - 1e-6))} s remaining`
      : { idle: "Not charged", ready: "Ready, 200 J", waiting_for_r: "Waiting for an R wave" }[status]);
    setText($("c-charge"), status === "idle" ? "Charge" : "Disarm");
    $("c-shock").disabled = status !== "ready";
    for (const button of syncButtons) {
      button.disabled = status === "waiting_for_r";
    }
    if (shockDelivered) {
      this.lastShockTime = lastShock.time;
      this.showShockReceipt(lastShock);
    }
  }

  showShockReceipt(receipt) {
    const feedback = $("c-shock-feedback");
    setText(feedback, `${minutesSeconds(receipt.time)} ${receipt.text}`);
    feedback.hidden = false;
    clearTimeout(this.shockFeedbackTimer);
    this.shockFeedbackTimer = setTimeout(() => { feedback.hidden = true; }, 6000);
  }

  // --- Extracorporeal support ---------------------------------------

  bindExtracorporeal() {
    const module = $("ecmo-module");
    if (!module) return;
    module.open = false;
    module.ontoggle = () => this.onViewChanged?.();
    this.ecmoWasActive = false;
    const ecmoFluidFeedback = $("ecmo-fluid-feedback");
    if (ecmoFluidFeedback) {
      ecmoFluidFeedback.hidden = true;
      setText(ecmoFluidFeedback, "");
    }
    for (const button of module.querySelectorAll("[data-ecmo-start]")) {
      button.onclick = () => this.performAction(button, "extracorporeal_start", {
        mode: button.dataset.ecmoStart,
        ...(button.dataset.cannulation ? { cannulation: button.dataset.cannulation } : {}),
      });
    }
    const settingsBox = $("c-ecmo-settings");
    const fields = [
      "c-ecmo-flow", "c-ecmo-sweep", "c-ecmo-fio2", "c-ecmo-water",
      "c-ecmo-lv-vent", "c-ecmo-sevo", "c-ecmo-venous-clamp", "c-ecmo-bg-strategy",
    ].map($).filter(Boolean);
    if (settingsBox && fields.length) {
      this.ecmoEditor = this.editSettings(fields, () => {
        const ec = this.state?.extracorporeal;
        const payload = {};
        const addNum = (id, key, current, digits) => {
          const el = $(id);
          if (!el || el.disabled) return;
          const val = readNumber(el);
          if (!ec || val !== round(current, digits)) payload[key] = val;
        };
        addNum("c-ecmo-flow", "blood_flow", ec?.target_flow, 2);
        addNum("c-ecmo-sweep", "sweep_gas", ec?.sweep, 2);
        addNum("c-ecmo-fio2", "fio2_percent", ec?.fio2, 0);
        addNum("c-ecmo-water", "water_target_c", ec?.water_target_c, 1);
        if (ec?.mode === "va_ecmo") addNum("c-ecmo-lv-vent", "lv_vent", ec.lv_vent, 2);
        if (ec?.mode === "cpb") {
          addNum("c-ecmo-sevo", "sweep_sevo", ec.sevo, 1);
          addNum("c-ecmo-venous-clamp", "venous_clamp", ec.venous_clamp, 0);
          const bgEl = $("c-ecmo-bg-strategy");
          if (bgEl && !bgEl.disabled && bgEl.value !== ec.blood_gas_strategy) {
            payload.blood_gas_strategy = bgEl.value;
          }
        }
        return Object.keys(payload).length ? this.send("extracorporeal_settings", payload) : true;
      }, settingsBox);
    }
    const hc = $("c-ecmo-hc");
    if (hc) {
      hc.onclick = () => this.send("extracorporeal_settings", {
        heater_cooler_on: !this.state?.extracorporeal?.heater_cooler_on,
      });
    }
    const crossClamp = $("c-cpb-cross-clamp");
    if (crossClamp) {
      crossClamp.onclick = () => this.send("extracorporeal_settings", {
        cross_clamp: !this.state?.extracorporeal?.cross_clamp,
      });
    }
    const cardioplegia = $("c-cpb-cardioplegia");
    if (cardioplegia) {
      cardioplegia.onclick = () => this.send("extracorporeal_settings", {
        cardioplegia: !this.state?.extracorporeal?.cardioplegia,
      });
    }
    const alarmMode = $("c-cpb-alarm-mode");
    if (alarmMode) {
      alarmMode.onclick = () => this.send("extracorporeal_settings", {
        bypass_alarm_mode: !this.state?.extracorporeal?.bypass_alarm_mode,
      });
    }
    for (const button of module.querySelectorAll("[data-reservoir-fluid]")) {
      button.onclick = async () => {
        const result = await this.performAction(button, "cpb_reservoir_fluid", {
          kind: button.dataset.reservoirFluid,
          volume_ml: Number(button.dataset.volume),
        });
        if (this.destroyed || result === undefined) return;
        const feedback = $("ecmo-fluid-feedback") ?? $("fluid-feedback");
        if (feedback && result.fluid?.text) {
          setText(feedback, result.fluid.text);
          feedback.hidden = false;
          clearTimeout(this.ecmoFluidFeedbackTimer);
          this.ecmoFluidFeedbackTimer = setTimeout(() => { feedback.hidden = true; }, 6000);
        }
      };
    }
    const stop = $("c-ecmo-stop");
    if (stop) {
      stop.onclick = () => this.performAction(stop, "extracorporeal_stop", { reinfuse: false });
    }
  }

  // --- Difficult airway tools, neuraxial block, crisis procedures ---

  bindAirwayTools() {
    const module = $("airway-tools-module");
    if (!module) return;
    module.open = false;
    module.ontoggle = () => this.onViewChanged?.();
    this.lastAirwayAttempts = 0;
    for (const id of ["c-airway-technique", "c-opa", "c-two-hand", "c-airway-help", "c-fona"]) {
      const el = $(id);
      if (el) el.disabled = false;
    }
    const view = $("c-airway-view");
    if (view) {
      view.hidden = true;
      setText(view, "");
    }
    const summary = $("airway-tools-summary");
    if (summary) {
      summary.hidden = true;
      setText(summary, "");
    }
    const tech = $("c-airway-technique");
    if (tech) tech.onchange = () => this.send("airway_technique", { technique: tech.value });
    const opa = $("c-opa");
    if (opa) opa.onclick = () => this.send("oral_airway", { placed: !this.state?.airway_crisis?.oral_airway });
    const twoHand = $("c-two-hand");
    if (twoHand) twoHand.onclick = () => this.send("two_hand_mask", { active: !this.state?.airway_crisis?.two_hand_mask });
    const help = $("c-airway-help");
    if (help) help.onclick = () => this.performAction(help, "call_airway_help");
    const fona = $("c-fona");
    if (fona) fona.onclick = () => this.performAction(fona, "perform_fona");
  }

  syncAirwayTools(ac, airwayMode) {
    const module = $("airway-tools-module");
    if (!module || !ac) return;
    if (ac.attempts > (this.lastAirwayAttempts ?? 0) && airwayMode !== "ETT" && airwayMode !== "FONA") {
      module.open = true;
    }
    this.lastAirwayAttempts = ac.attempts ?? 0;
    const tech = $("c-airway-technique");
    if (tech && ac.technique) setValue(tech, ac.technique);
    if ($("c-opa")) setPressed($("c-opa"), Boolean(ac.oral_airway));
    if ($("c-two-hand")) setPressed($("c-two-hand"), Boolean(ac.two_hand_mask));
    if ($("c-airway-help")) {
      setPressed($("c-airway-help"), Boolean(ac.help_called), ac.help_called ? "Airway help called" : "Call for airway help");
    }
    if ($("c-fona")) {
      const fonaActive = airwayMode === "FONA";
      setPressed($("c-fona"), fonaActive, fonaActive ? "eFONA airway active" : "Cricothyrotomy (eFONA)");
    }
    const view = $("c-airway-view");
    if (view) {
      const parts = [];
      if (ac.attempts > 0) parts.push(`Attempts: ${ac.attempts}`);
      if (ac.last_view) parts.push(`Last view: ${ac.last_view}`);
      view.hidden = parts.length === 0;
      setText(view, parts.join(" · "));
    }
    const summary = $("airway-tools-summary");
    if (summary) {
      const tags = [];
      if (airwayMode === "FONA") tags.push("eFONA");
      if (ac.technique && ac.technique !== "dl") tags.push(ac.technique.toUpperCase().replace("_", "+"));
      if (ac.oral_airway) tags.push("OPA");
      if (ac.two_hand_mask) tags.push("2-hand");
      if (ac.help_called) tags.push("Help");
      summary.hidden = tags.length === 0;
      setText(summary, tags.join(" · "));
    }
  }

  bindNeuraxial() {
    const module = $("neuraxial-module");
    if (!module) return;
    module.open = false;
    module.ontoggle = () => this.onViewChanged?.();
    this.lastNeuraxialCheckTime = null;
    this.epiduralRateDraft = false;
    clearTimeout(this.feedbackTimers?.get("neuraxial"));
    this.feedbackTimers?.delete("neuraxial");
    for (const id of [
      "c-spinal-drug", "c-spinal-dose", "c-spinal-baricity", "c-spinal-give",
      "c-epidural-drug", "c-epidural-vol", "c-epidural-conc", "c-epidural-epi",
      "c-epidural-bolus", "c-epidural-rate", "c-epidural-rate-apply",
      "c-epidural-stop", "c-neuraxial-check",
    ]) {
      const el = $(id);
      if (el) el.disabled = false;
    }
    const spinalDrug = $("c-spinal-drug");
    const spinalDose = $("c-spinal-dose");
    const spinalBaricity = $("c-spinal-baricity");
    const epiDrug = $("c-epidural-drug");
    const epiVol = $("c-epidural-vol");
    const epiConc = $("c-epidural-conc");
    const epiCheck = $("c-epidural-epi");
    const epiRate = $("c-epidural-rate");
    const readout = $("c-neuraxial-readout");
    const summary = $("neuraxial-summary");
    if (spinalDrug) spinalDrug.value = "bupivacaine";
    if (spinalDose) spinalDose.value = "12.5";
    if (spinalBaricity) spinalBaricity.value = "hyperbaric";
    if (epiDrug) epiDrug.value = "bupivacaine";
    if (epiVol) epiVol.value = "10";
    if (epiConc) epiConc.value = "0.5";
    if (epiCheck) epiCheck.checked = false;
    if (epiRate) {
      epiRate.value = "0";
      epiRate.oninput = () => { this.epiduralRateDraft = true; };
    }
    if (readout) {
      readout.hidden = true;
      setText(readout, "");
    }
    if (summary) {
      summary.hidden = true;
      setText(summary, "");
    }
    if (spinalDrug && spinalDose) {
      spinalDrug.onchange = () => {
        spinalDose.value = spinalDrug.value === "lidocaine" ? "50" : "12.5";
      };
    }
    if (epiDrug && epiConc) {
      const syncEpiDrugConc = () => {
        const isLido = epiDrug.value === "lidocaine";
        epiConc.value = isLido ? "2" : "0.5";
        const rateLabel = $("c-epidural-rate-label");
        if (rateLabel) setText(rateLabel, `Epidural infusion (${isLido ? "0.5" : "0.125"}%)`);
      };
      epiDrug.onchange = syncEpiDrugConc;
      syncEpiDrugConc();
    }
    const spinalGive = $("c-spinal-give");
    if (spinalGive) {
      spinalGive.onclick = () => {
        if (spinalDose?.reportValidity && !spinalDose.reportValidity()) return;
        return this.performAction(spinalGive, "neuraxial_spinal", {
          drug: spinalDrug?.value ?? "bupivacaine",
          dose_mg: readNumber(spinalDose),
          baricity: spinalBaricity?.value ?? "hyperbaric",
        });
      };
    }
    const epiBolus = $("c-epidural-bolus");
    if (epiBolus) {
      epiBolus.onclick = () => {
        if (epiVol?.reportValidity && !epiVol.reportValidity()) return;
        return this.performAction(epiBolus, "neuraxial_epidural_bolus", {
          drug: epiDrug?.value ?? "bupivacaine",
          volume_ml: readNumber(epiVol),
          concentration_pct: Number(epiConc?.value ?? 0.5),
          with_epinephrine: Boolean(epiCheck?.checked),
        });
      };
    }
    const epiRateApply = $("c-epidural-rate-apply");
    if (epiRateApply) {
      epiRateApply.onclick = async () => {
        if (epiRate?.reportValidity && !epiRate.reportValidity()) return;
        const drug = epiDrug?.value ?? "bupivacaine";
        const result = await this.performAction(epiRateApply, "neuraxial_epidural_rate", {
          ml_hr: readNumber(epiRate),
          drug,
          concentration_pct: drug === "lidocaine" ? 0.5 : 0.125,
        });
        if (!this.destroyed && result !== undefined) this.epiduralRateDraft = false;
      };
    }
    const epiStop = $("c-epidural-stop");
    if (epiStop) {
      epiStop.onclick = async () => {
        const result = await this.performAction(epiStop, "neuraxial_epidural_rate", { ml_hr: 0 });
        if (!this.destroyed && result !== undefined) this.epiduralRateDraft = false;
      };
    }
    const checkBtn = $("c-neuraxial-check");
    if (checkBtn) {
      checkBtn.onclick = async () => {
        const result = await this.performAction(checkBtn, "neuraxial_check_level");
        if (this.destroyed || !result?.neuraxial_check) return;
        this.lastNeuraxialCheckTime = result.neuraxial_check.time;
        clearTimeout(this.feedbackTimers?.get("neuraxial"));
        this.feedbackTimers?.delete("neuraxial");
        if (readout) {
          setText(readout, `${minutesSeconds(result.neuraxial_check.time)} ${result.neuraxial_check.note}`);
          readout.hidden = false;
        }
      };
    }
  }

  syncNeuraxial(nx) {
    const module = $("neuraxial-module");
    if (!module || !nx) return;
    const summary = $("neuraxial-summary");
    if (summary) {
      const parts = [];
      if (nx.active) parts.push(`Sensory ${nx.sensory_label}`);
      if (nx.epidural_rate > 0) parts.push(`${nx.epidural_rate} mL/hr`);
      else if (!nx.active && nx.epidural_catheter) parts.push("Catheter in place");
      summary.hidden = parts.length === 0;
      setText(summary, parts.join(" · "));
    }
    const epiRate = $("c-epidural-rate");
    if (epiRate && !this.epiduralRateDraft) setValue(epiRate, round(nx.epidural_rate ?? 0, 1));
    if ($("c-epidural-stop")) $("c-epidural-stop").hidden = !(nx.epidural_rate > 0);
    if (nx.last_check && nx.last_check.time !== this.lastNeuraxialCheckTime) {
      this.lastNeuraxialCheckTime = nx.last_check.time;
      const readout = $("c-neuraxial-readout");
      if (readout) {
        clearTimeout(this.feedbackTimers?.get("neuraxial"));
        this.feedbackTimers?.delete("neuraxial");
        setText(readout, `${minutesSeconds(nx.last_check.time)} ${nx.last_check.note}`);
        readout.hidden = false;
      }
    } else if (!nx.last_check && this.lastNeuraxialCheckTime !== null) {
      this.lastNeuraxialCheckTime = null;
      const readout = $("c-neuraxial-readout");
      if (readout) {
        clearTimeout(this.feedbackTimers?.get("neuraxial"));
        this.feedbackTimers?.delete("neuraxial");
        readout.hidden = true;
        setText(readout, "");
      }
    }
    if (!nx.active && nx.epidural_rate === 0 && !nx.epidural_catheter) {
      this.epiduralRateDraft = false;
    }
  }

  bindCrisisProcedures() {
    const module = $("crisis-procedures-module");
    if (!module) return;
    module.open = false;
    module.ontoggle = () => this.onViewChanged?.();
    clearTimeout(this.crisisProcFeedbackTimer);
    for (const id of ["c-chest-decompress", "c-vae-aspirate", "c-pe-thrombolysis"]) {
      const el = $(id);
      if (el) el.disabled = false;
    }
    const feedback = $("crisis-proc-feedback");
    if (feedback) {
      feedback.hidden = true;
      setText(feedback, "");
    }
    const summary = $("crisis-proc-summary");
    if (summary) {
      summary.hidden = true;
      setText(summary, "");
    }
    const showResult = (result) => {
      if (this.destroyed || !result?.procedure?.text) return;
      const el = $("crisis-proc-feedback");
      if (!el) return;
      setText(el, `${minutesSeconds(result.procedure.time)} ${result.procedure.text}`);
      el.hidden = false;
      clearTimeout(this.crisisProcFeedbackTimer);
      this.crisisProcFeedbackTimer = setTimeout(() => { el.hidden = true; }, 8000);
    };
    const ptxBtn = $("c-chest-decompress");
    if (ptxBtn) ptxBtn.onclick = async () => showResult(await this.performAction(ptxBtn, "chest_decompress"));
    const vaeBtn = $("c-vae-aspirate");
    if (vaeBtn) vaeBtn.onclick = async () => showResult(await this.performAction(vaeBtn, "vae_aspirate"));
    const peBtn = $("c-pe-thrombolysis");
    if (peBtn) peBtn.onclick = async () => showResult(await this.performAction(peBtn, "pe_thrombolysis"));
  }

  syncCrisisProcedures(c) {
    const summary = $("crisis-proc-summary");
    if (!summary) return;
    const tags = [];
    if (c.pneumothorax?.decompressed) tags.push("Chest vented");
    if (c.embolism?.thrombolysis) tags.push("Lysis running");
    summary.hidden = tags.length === 0;
    setText(summary, tags.join(" · "));
  }

  syncExtracorporeal(ec) {
    const module = $("ecmo-module");
    if (!module || !ec) return;
    const active = Boolean(ec.active);
    const modeLabels = { vv_ecmo: "VV ECMO", va_ecmo: "VA ECMO", cpb: "CPB" };
    const summary = $("ecmo-summary");
    if (summary) {
      summary.hidden = !active;
      if (active) setText(summary, `${modeLabels[ec.mode] ?? ec.mode} · ${ec.flow.toFixed(1)} L/min`);
    }
    if (active && !this.ecmoWasActive) module.open = true;
    this.ecmoWasActive = active;
    const startPanel = $("ecmo-start-panel");
    const activePanel = $("ecmo-active-panel");
    if (startPanel) startPanel.hidden = active;
    if (activePanel) activePanel.hidden = !active;
    const abgSiteRow = $("c-abg-site-row");
    if (abgSiteRow) abgSiteRow.hidden = !active;
    if (!active) {
      const abgSite = $("c-abg-site");
      if (abgSite) abgSite.value = "arterial";
      this.ecmoEditor?.cancel();
      return;
    }
    const isVV = ec.mode === "vv_ecmo";
    const isVA = ec.mode === "va_ecmo";
    const isCPB = ec.mode === "cpb";
    const setRow = (rowId, inputId, visible) => {
      const row = $(rowId);
      if (row) row.hidden = !visible;
      const input = $(inputId);
      if (input && !this.ecmoEditor?.pending) input.disabled = !visible;
    };
    setRow("ecmo-lv-vent-row", "c-ecmo-lv-vent", isVA);
    setRow("ecmo-sevo-row", "c-ecmo-sevo", isCPB);
    setRow("ecmo-venous-row", "c-ecmo-venous-clamp", isCPB);
    setRow("ecmo-bg-row", "c-ecmo-bg-strategy", isCPB);
    if ($("ecmo-cpb-controls")) $("ecmo-cpb-controls").hidden = !isCPB;
    if ($("c-cpb-alarm-mode")) {
      $("c-cpb-alarm-mode").hidden = !(isCPB || isVA);
      if (isCPB || isVA) {
        setPressed($("c-cpb-alarm-mode"), ec.bypass_alarm_mode,
          ec.bypass_alarm_mode ? "Bypass alarm mode: On" : "Bypass alarm mode: Off");
      }
    }
    setValue($("c-ecmo-flow"), round(ec.target_flow, 2));
    setValue($("c-ecmo-sweep"), round(ec.sweep, 2));
    setValue($("c-ecmo-fio2"), round(ec.fio2, 0));
    setValue($("c-ecmo-water"), round(ec.water_target_c, 1));
    if ($("c-ecmo-lv-vent") && ec.lv_vent !== undefined) setValue($("c-ecmo-lv-vent"), round(ec.lv_vent, 2));
    if ($("c-ecmo-sevo") && ec.sevo !== undefined) setValue($("c-ecmo-sevo"), round(ec.sevo, 1));
    if ($("c-ecmo-venous-clamp") && ec.venous_clamp !== undefined) setValue($("c-ecmo-venous-clamp"), round(ec.venous_clamp, 0));
    if ($("c-ecmo-bg-strategy") && ec.blood_gas_strategy !== undefined) setValue($("c-ecmo-bg-strategy"), ec.blood_gas_strategy);
    if ($("c-ecmo-hc")) {
      setPressed($("c-ecmo-hc"), ec.heater_cooler_on,
        ec.heater_cooler_on ? `Heater-cooler: On (${ec.water_target_c.toFixed(1)} °C)` : "Heater-cooler: Off");
    }
    if (isCPB) {
      if ($("c-cpb-cross-clamp")) {
        setPressed($("c-cpb-cross-clamp"), ec.cross_clamp,
          ec.cross_clamp ? "Release aortic cross-clamp" : "Apply aortic cross-clamp");
      }
      if ($("c-cpb-cardioplegia")) {
        setPressed($("c-cpb-cardioplegia"), ec.cardioplegia,
          ec.cardioplegia ? "Stop cardioplegia" : "Start cardioplegia");
      }
    }
    const readout = $("ecmo-readout");
    if (readout) {
      const parts = [
        `${modeLabels[ec.mode] ?? ec.mode} ${ec.flow.toFixed(2)} L/min`,
        `Sweep ${ec.sweep.toFixed(1)} L/min`,
        `FiO₂ ${Math.round(ec.fio2)}%`,
      ];
      if ((isVV || isVA) && ec.pre_so2 !== undefined) parts.push(`Pre-O₂ SvO₂ ${Math.round(ec.pre_so2)}%`);
      if (isCPB) parts.push(`Reservoir ${Math.round(ec.reservoir_ml)} mL${ec.reservoir_low ? " (LOW)" : ""}`);
      setText(readout, parts.join(" · "));
      readout.className = "ecmo-readout" + (ec.reservoir_low ? " bad" : "");
    }
  }

  // --- Labs ---------------------------------------------------------

  bindLabs() {
    const abgSite = $("c-abg-site");
    if (abgSite) abgSite.value = "arterial";
    const abgSiteRow = $("c-abg-site-row");
    if (abgSiteRow) abgSiteRow.hidden = true;
    const button = $("c-abg");
    button.onclick = async () => {
      const site = (!abgSiteRow?.hidden && abgSite?.value) ? abgSite.value : "arterial";
      const args = site !== "arterial" ? { site } : undefined;
      const result = await this.performAction(button, "blood_gas", args);
      if (this.destroyed || result === undefined) return;
      this.bloodGases.push(result.blood_gas);
      this.showBloodGases();
    };
    const actBtn = $("c-act");
    if (actBtn) {
      actBtn.onclick = async () => {
        const result = await this.performAction(actBtn, "act");
        if (this.destroyed || result === undefined) return;
        (this.acts ??= []).push(result.act);
        this.showActs();
      };
    }
    const glucoseBtn = $("c-glucose");
    if (glucoseBtn) {
      glucoseBtn.onclick = async () => {
        const result = await this.performAction(glucoseBtn, "glucose");
        if (this.destroyed || result === undefined) return;
        (this.glucoses ??= []).push(result.glucose);
        this.showGlucoses();
      };
    }
    this.showBloodGases();
    this.showActs();
    this.showGlucoses();
  }

  showGlucoses() {
    if (!this.glucoses) return;
    const status = $("glucose-pending");
    const container = $("glucose-results");
    if (!status || !container) return;
    const ready = this.glucoses.filter((result) => result.ready_at <= this.time);
    const pending = this.glucoses.length - ready.length;
    status.hidden = pending === 0;
    if (pending > 0) {
      const next = minutesSeconds(Math.ceil(this.glucoses[ready.length].ready_at - this.time));
      setText(status, pending === 1 ? `Result in ${next}` : `${pending} samples running; next result in ${next}`);
    }
    if (ready.length === this.shownGlucoses) return;
    this.shownGlucoses = ready.length;
    container.hidden = ready.length === 0;
    const table = container.querySelector?.("table");
    if (!table) return;
    const newest = [...ready].reverse();
    table.tBodies[0].replaceChildren(...newest.map((result) => {
      const row = document.createElement("tr");
      const timeCell = document.createElement("th");
      timeCell.scope = "row";
      timeCell.textContent = minutesSeconds(result.drawn_at);
      const valCell = document.createElement("td");
      if (result.glucose === null || result.glucose === undefined) {
        valCell.textContent = "—";
      } else {
        const rounded = Math.round(result.glucose);
        valCell.textContent = String(rounded);
        const flag = rounded < 70 ? "L" : rounded > 180 ? "H" : "";
        if (flag) {
          valCell.className = flag === "H" ? "abg-high" : "abg-low";
          const badge = document.createElement("span");
          badge.className = "abg-flag";
          badge.textContent = flag;
          valCell.append(badge);
        }
      }
      row.append(timeCell, valCell);
      return row;
    }));
  }

  showActs() {
    if (!this.acts) return;
    const status = $("act-pending");
    const container = $("act-results");
    if (!status || !container) return;
    const ready = this.acts.filter((result) => result.ready_at <= this.time);
    const pending = this.acts.length - ready.length;
    status.hidden = pending === 0;
    if (pending > 0) {
      const next = minutesSeconds(Math.ceil(this.acts[ready.length].ready_at - this.time));
      setText(status, pending === 1 ? `Result in ${next}` : `${pending} samples running; next result in ${next}`);
    }
    if (ready.length === this.shownActs) return;
    this.shownActs = ready.length;
    container.hidden = ready.length === 0;
    const table = container.querySelector?.("table");
    if (!table) return;
    const newest = [...ready].reverse();
    table.tBodies[0].replaceChildren(...newest.map((result) => {
      const row = document.createElement("tr");
      const timeCell = document.createElement("th");
      timeCell.scope = "row";
      timeCell.textContent = minutesSeconds(result.drawn_at);
      const valCell = document.createElement("td");
      if (result.act_s === null) {
        valCell.textContent = "—";
      } else {
        const roundedAct = Math.round(result.act_s);
        valCell.textContent = String(roundedAct);
        if (roundedAct > 150) {
          valCell.className = "abg-high";
        }
      }
      row.append(timeCell, valCell);
      return row;
    }));
  }

  // Results post at their ready time on the simulation clock.
  showBloodGases() {
    const ready = this.bloodGases.filter((result) => result.ready_at <= this.time);
    const pending = this.bloodGases.length - ready.length;
    const status = $("abg-pending");
    status.hidden = pending === 0;
    if (pending > 0) {
      const next = minutesSeconds(Math.ceil(this.bloodGases[ready.length].ready_at - this.time));
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
    const siteTags = {
      right_radial: "R rad",
      femoral: "Fem",
      pre_oxy: "Pre",
      post_oxy: "Post",
    };
    const header = document.createElement("tr");
    header.append(
      cell("th", "Analyte", "col"),
      cell("th", "Ref", "col"),
      ...newest.map((result) => {
        const tag = result.site && siteTags[result.site] ? ` (${siteTags[result.site]})` : "";
        return cell("th", `${minutesSeconds(result.drawn_at)}${tag}`, "col");
      }),
    );
    table.tHead.replaceChildren(header);
    table.tBodies[0].replaceChildren(...bloodGasRows.map(([label, unit, key, format, refLabel, low, high]) => {
      const row = document.createElement("tr");
      const name = cell("th", label, "row");
      if (unit) {
        const span = cell("span", unit);
        span.className = "unit";
        name.append(span);
      }
      const refCell = cell("td", refLabel);
      refCell.className = "abg-ref";
      const strictHigh = refLabel.startsWith("<");
      row.append(name, refCell, ...newest.map((result) => {
        const raw = result[key];
        if (raw === null || raw === undefined) return cell("td", "—");
        const shown = format(raw);
        const value = Number(shown);
        const td = cell("td", shown);
        const flag = low !== null && value < low ? "L"
          : high !== null && (strictHigh ? value >= high : value > high) ? "H" : "";
        if (flag) {
          td.className = flag === "H" ? "abg-high" : "abg-low";
          const badge = cell("span", flag);
          badge.className = "abg-flag";
          td.append(badge);
        }
        return td;
      }));
      return row;
    }));
  }

  // --- Events -------------------------------------------------------

  bindEvents(info) {
    this.eventSections = new Map([...document.querySelectorAll(".event-section[data-event]")].map((section) => {
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

    const obstruction = $("c-obstruction"), bronchospasm = $("c-bronchospasm"), ardsShunt = $("c-ards-shunt");
    const airwayFields = ardsShunt ? [obstruction, bronchospasm, ardsShunt] : [obstruction, bronchospasm];
    this.editSettings(airwayFields, async () => {
      const updates = [];
      if (readNumber(obstruction) !== round(this.state.obstruction, 0)) {
        updates.push(["obstruction", readNumber(obstruction)]);
      }
      if (readNumber(bronchospasm) !== round(this.state.bronchospasm, 0)) {
        updates.push(["bronchospasm", readNumber(bronchospasm)]);
      }
      if (ardsShunt && readNumber(ardsShunt) !== round(this.state.ards_shunt ?? 0, 0)) {
        updates.push(["ards_shunt", readNumber(ardsShunt)]);
      }
      for (const [command, percent] of updates) {
        if (await this.send(command, { percent }) === undefined) return undefined;
      }
      return true;
    }, obstruction.closest(".vent-settings"));
    $("c-auto-laryngo").onchange = () => this.send("auto_laryngospasm", { enabled: $("c-auto-laryngo").checked });

    $("c-hem-rate").value = "500";
    $("c-hemorrhage").onclick = () => this.send("hemorrhage", {
      active: !this.state.hemorrhage.active,
      rate_ml_min: Number($("c-hem-rate").value) || 500,
    });
    const rhythms = $("c-rhythm");
    rhythms.replaceChildren(...info.rhythms.map((name) => new Option(name, name)));
    this.observedRhythm = null;
    rhythms.onchange = () => this.send("rhythm", { name: rhythms.value });
    $("c-position-row").hidden = info.patient.gestational_age_weeks === 0;
    $("c-position").onchange = () => this.send("position", { position: $("c-position").value });
    $("c-anaphylaxis").onclick = () => this.send("anaphylaxis", { active: !this.state.anaphylaxis });
    $("c-sepsis").onclick = () => this.send("sepsis", { active: !this.state.sepsis });
    const protReaction = $("c-protamine-reaction");
    if (protReaction) {
      protReaction.onclick = () => this.send("protamine_reaction", { active: !this.state.protamine_reaction });
    }
    const diffAirway = $("c-difficult-airway");
    if (diffAirway) {
      diffAirway.onchange = () => this.send("difficult_airway", { profile: diffAirway.value });
    }
    const armEsoph = $("c-arm-esophageal");
    if (armEsoph) {
      armEsoph.onclick = () => this.send("esophageal_intubation", {
        armed: !this.state?.airway_crisis?.esophageal_armed,
      });
    }
    const esophNow = $("c-esophageal-now");
    if (esophNow) {
      esophNow.onclick = () => this.send("esophageal_intubation", {
        active: !this.state?.airway_crisis?.esophageal_active,
      });
    }
    const ptx = $("c-pneumothorax");
    if (ptx) {
      ptx.onclick = () => this.send("pneumothorax", { active: !this.state?.pneumothorax?.active });
    }
    const pneumo = $("c-pneumoperitoneum");
    const pneumoIap = $("c-pneumo-iap");
    if (pneumo) {
      pneumo.onclick = () => this.send("pneumoperitoneum", {
        active: !this.state?.pneumoperitoneum?.active,
        iap_mmhg: Number(pneumoIap?.value || 14),
      });
    }
    if (pneumoIap) {
      pneumoIap.onchange = () => {
        if (this.state?.pneumoperitoneum?.active) {
          this.send("pneumoperitoneum", { active: true, iap_mmhg: Number(pneumoIap.value) });
        }
      };
    }
    const pe = $("c-pe");
    if (pe) {
      pe.onclick = () => this.send("pulmonary_embolism", { active: !this.state?.embolism?.pe_active });
    }
    const vae = $("c-vae");
    if (vae) {
      vae.onclick = () => this.send("venous_air_embolism", { active: !this.state?.embolism?.vae_active });
    }
    const highSpinal = $("c-high-spinal");
    if (highSpinal) {
      highSpinal.onclick = () => this.send("high_spinal", {
        active: !(this.state?.neuraxial?.active && this.state?.neuraxial?.mode === "high_spinal"),
      });
    }
    $("c-mh").onclick = () => this.send("malignant_hyperthermia", { active: !this.state.malignant_hyperthermia });
  }

  syncHemorrhageRate(rate) {
    const select = $("c-hem-rate");
    const text = String(rate);
    if (this.observedHemorrhageRate && this.observedHemorrhageRate.value !== text) {
      this.observedHemorrhageRate.remove?.();
      this.observedHemorrhageRate = null;
    }
    if (select.value === text) return;
    if (select.options) {
      const observed = select.querySelector?.("option[data-observed]");
      if (observed && observed.value !== text) observed.remove();
      if (!Array.from(select.options).some(option => option.value === text)) {
        const option = new Option(`${rate} mL/min`, text);
        option.disabled = true;
        option.dataset.observed = "true";
        select.add(option);
        this.observedHemorrhageRate = option;
      }
    }
    setValue(select, text);
  }

  syncRhythm(rhythm) {
    const select = $("c-rhythm");
    if (this.observedRhythm && this.observedRhythm.value !== rhythm) {
      this.observedRhythm.remove();
      this.observedRhythm = null;
    }
    if (select.value === rhythm) return;
    if (!Array.from(select.options).some(option => option.value === rhythm)) {
      const option = new Option(rhythm, rhythm);
      option.disabled = true;
      option.dataset.observed = "true";
      select.add(option);
      this.observedRhythm = option;
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

  sync(c, time, ended = false) {
    this.state = c;
    this.ended = Boolean(ended);
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
    this.showActs();
    this.showGlucoses();
    this.showClinicalFeedback(c.clinical_feedback);

    for (const button of (this.airwayButtons ??= [...$("c-airway").querySelectorAll("button")])) {
      setPressed(button, button.dataset.value === c.airway);
    }
    if (c.airway !== "Mask") this.maskSealEditor.cancel();
    $("c-mask-seal-row").hidden = c.airway !== "Mask";
    setValue($("c-mask-seal"), round(c.mask_seal, 0));
    setValue($("c-o2"), round(c.fgf.o2));
    setValue($("c-air"), round(c.fgf.air));
    setValue($("c-n2o"), round(c.fgf.n2o));
    const deliveredO2 = c.o2_connected ? c.fgf.o2 : 0;
    const deliveredN2o = c.o2_connected ? Math.min(c.fgf.n2o, 3 * deliveredO2) : 0;
    const totalFgf = deliveredO2 + c.fgf.air + deliveredN2o;
    const barO2 = $("fgf-bar-o2");
    if (barO2?.style) barO2.style.setProperty("--fill", `${Math.min(100, (deliveredO2 / 15) * 100).toFixed(0)}%`);
    if (barO2?.dataset) barO2.dataset.fault = String(!c.o2_connected);
    const barAir = $("fgf-bar-air");
    if (barAir?.style) barAir.style.setProperty("--fill", `${Math.min(100, (c.fgf.air / 15) * 100).toFixed(0)}%`);
    const barN2o = $("fgf-bar-n2o");
    if (barN2o?.style) barN2o.style.setProperty("--fill", `${Math.min(100, (deliveredN2o / 15) * 100).toFixed(0)}%`);
    const totalFgfEl = $("c-total-fgf");
    if (totalFgfEl) setText(totalFgfEl, `${totalFgf.toFixed(1)} L/min`);
    const o2Alert = $("c-o2-supply-alert");
    if (o2Alert) o2Alert.hidden = c.o2_connected;
    const fio2 = $("c-fio2");
    setText(fio2, `${Math.round(c.circuit_fio2 * 100)}%`);
    fio2.className = "fio2" + (!c.o2_connected || c.circuit_fio2 < 0.21 ? " bad" : c.circuit_fio2 < 0.3 ? " warn" : "");
    setPressed($("c-o2-supply"), !c.o2_connected, c.o2_connected ? "Disconnect O₂ supply" : "Connect backup O₂");
    setValue($("c-vap"), round(c.vaporizer));
    const vapBar = $("vap-bar");
    if (vapBar?.style) vapBar.style.setProperty("--fill", `${Math.min(100, Math.max(0, (c.vaporizer / 8) * 100)).toFixed(0)}%`);
    const vapLevel = $("c-vap-level");
    if (vapLevel && c.vaporizer_level !== undefined) {
      setText(vapLevel, c.vaporizer_level <= 0 ? "Empty" : `${Math.round(c.vaporizer_level)} mL`);
      vapLevel.className = c.vaporizer_level <= 0 ? "bad" : c.vaporizer_level < 50 ? "warn" : "";
    }
    const vapRefill = $("c-vap-refill");
    if (vapRefill && c.vaporizer_level !== undefined) {
      vapRefill.hidden = c.vaporizer_level >= 200;
      vapRefill.disabled = this.ended;
    }

    const compressions = c.compressions;
    setPressed($("c-compressions"), compressions.active, compressions.active ? "Stop compressions" : "Start compressions");
    $("c-compressions-status").hidden = !compressions.active;
    if (compressions.active) setText($("c-compressions-status"), `Compressing for ${minutesSeconds(time - compressions.since)}`);
    this.syncDefibrillator(c.defibrillator);
    if (c.extracorporeal) this.syncExtracorporeal(c.extracorporeal);
    if (c.airway_crisis) this.syncAirwayTools(c.airway_crisis, c.airway);
    if (c.neuraxial) this.syncNeuraxial(c.neuraxial);
    this.syncCrisisProcedures(c);
    if ($("c-position-row")) {
      const showPosition = Boolean(
        this.pregnant
        || c.position === "left_lateral"
        || c.neuraxial?.active
        || c.neuraxial?.epidural_catheter
        || c.embolism?.vae_active
        || (c.embolism?.vae_air_ml ?? 0) > 1
      );
      $("c-position-row").hidden = !showPosition;
      const posLabel = $("c-position-label");
      if (posLabel) setText(posLabel, this.pregnant ? "Maternal position" : "Patient position");
    }

    setPressed($("c-bag"), c.bag_mask, c.bag_mask ? "Stop bag ventilation" : "Start bag ventilation");
    const aplStandby = c.airway === "None" || c.vent.on;
    const aplOpenNoPressure = !aplStandby && c.bag_mask && c.apl === 0;
    const aplState = $("c-apl-state");
    setText(aplState, aplStandby ? "Standby" : c.apl === 0 ? "Open" : "");
    aplState.className = aplOpenNoPressure ? "warn" : "";
    setValue($("c-apl"), c.apl);
    const reservoir = c.reservoir ?? { fill: 1, pressure: 0, state: "normal" };
    const bagFill = $("c-bag-fill");
    if (bagFill?.style) bagFill.style.setProperty("--fill", `${Math.round(reservoir.fill * 100)}%`);
    if (bagFill?.dataset) bagFill.dataset.state = reservoir.state;
    const bagState = $("c-bag-state");
    let bagLabel = "Normal";
    let bagClass = "";
    if (reservoir.state === "empty") {
      bagLabel = "Empty";
      bagClass = "bad";
    } else if (reservoir.state === "low") {
      bagLabel = "Low";
      bagClass = "warn";
    } else if (reservoir.state === "high") {
      bagLabel = `High · ${Math.round(reservoir.pressure)} cmH₂O`;
      bagClass = "bad";
    } else if (reservoir.state === "tense") {
      bagLabel = `Tense · ${Math.round(reservoir.pressure)} cmH₂O`;
      bagClass = "warn";
    } else if (aplOpenNoPressure) {
      bagLabel = "No pressure";
      bagClass = "warn";
    }
    setText(bagState, bagLabel);
    bagState.className = bagClass;
    const bagBreath = $("c-bag-breath");
    if (bagBreath) {
      bagBreath.disabled = this.ended || aplStandby || c.bag_mask;
      setText(bagBreath, `Give breath (${c.bag.tv} mL)`);
    }
    const o2Flush = $("c-o2-flush");
    if (o2Flush) {
      setPressed(o2Flush, Boolean(c.o2_flush), c.o2_flush ? "O₂ flushing…" : "O₂ flush");
      o2Flush.disabled = this.ended || !c.o2_connected;
    }
    $("c-bag-settings").hidden = !c.bag_mask;
    setValue($("c-bag-rr"), c.bag.rr);
    setValue($("c-bag-tv"), c.bag.tv);
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
        const idleMode = w.editor.values?.get(w.mode) ?? w.mode.value;
        setValue(w.mode, d.is_tci ? "tci" : d.rate > 0 ? "rate" : idleMode);
        setValue(w.target, round(d.target, 2));
      }
      if (w.rate) {
        this.syncDrugInputs(w);
        setValue(w.rate, d.is_tci ? shortRate(d.rate) : rateText(d.rate));
        const active = d.rate > 0 || (d.is_tci && d.target > 0);
        if (!active && w.runningRow.contains(document.activeElement)) $("drug-search").focus();
        if (!active) this.closeInfusionEditor(w);
        w.runningRow.hidden = !active;
        if (active) {
          setText(w.runningValue, d.is_tci
            ? `${w.spec.target_label}: ${round(d.target, 2)} ${w.spec.tci_unit}, rate ${shortRate(d.rate)} ${w.spec.rate_unit}`
            : `${shortRate(d.rate)} ${w.spec.rate_unit}`);
        }
        running ||= active;
      }
    }
    $("running-infusions").hidden = !running;

    setValue($("c-bair"), String(c.bair_hugger));
    setPressed($("c-fluid-warmer"), c.fluid_warmer, c.fluid_warmer ? "Fluid warmer: On (37 °C)" : "Fluid warmer: Off");
    setValue($("c-maint"), round(c.maintenance_ml_hr, 0));
    const profile = c.disturbance.profile ?? "";
    setValue($("c-disturbance"), profile);
    $("c-disturbance").disabled = c.disturbance.active;
    $("c-disturb").disabled = !profile;
    setPressed($("c-disturb"), c.disturbance.active, c.disturbance.active ? "Stop stimulation" : "Start stimulation");
    setValue($("c-obstruction"), round(c.obstruction, 0));
    setValue($("c-bronchospasm"), round(c.bronchospasm, 0));
    if ($("c-ards-shunt") && c.ards_shunt !== undefined) setValue($("c-ards-shunt"), round(c.ards_shunt, 0));
    setText($("c-laryngo"), `Laryngospasm: ${c.laryngospasm}`);
    $("c-auto-laryngo").checked = c.auto_laryngospasm;

    setPressed($("c-hemorrhage"), c.hemorrhage.active, c.hemorrhage.active ? "Stop bleeding" : "Start bleeding");
    $("c-hem-rate").disabled = c.hemorrhage.active;
    if (c.hemorrhage.active) this.syncHemorrhageRate(c.hemorrhage.rate);
    this.syncRhythm(c.ecg_rhythm);
    setValue($("c-position"), c.position);
    setPressed($("c-anaphylaxis"), c.anaphylaxis, c.anaphylaxis ? "Stop anaphylaxis" : "Start anaphylaxis");
    setPressed($("c-sepsis"), c.sepsis, c.sepsis ? "Stop sepsis" : "Start sepsis");
    if ($("c-protamine-reaction") && c.protamine_reaction !== undefined) {
      setPressed($("c-protamine-reaction"), c.protamine_reaction,
        c.protamine_reaction ? "Stop protamine reaction" : "Start protamine reaction");
    }
    if ($("c-difficult-airway") && c.airway_crisis) {
      setValue($("c-difficult-airway"), c.airway_crisis.difficult_profile);
    }
    if ($("c-arm-esophageal") && c.airway_crisis) {
      setPressed($("c-arm-esophageal"), Boolean(c.airway_crisis.esophageal_armed),
        c.airway_crisis.esophageal_armed ? "Disarm esophageal intubation" : "Arm esophageal intubation");
    }
    if ($("c-esophageal-now") && c.airway_crisis) {
      setPressed($("c-esophageal-now"), Boolean(c.airway_crisis.esophageal_active),
        c.airway_crisis.esophageal_active ? "Clear esophageal intubation" : "Place tube in esophagus");
    }
    if ($("c-pneumothorax") && c.pneumothorax) {
      setPressed($("c-pneumothorax"), Boolean(c.pneumothorax.active),
        c.pneumothorax.active ? "Stop pleural leak" : "Start tension pneumothorax");
    }
    if ($("c-pneumoperitoneum") && c.pneumoperitoneum) {
      const pActive = Boolean(c.pneumoperitoneum.active);
      setPressed($("c-pneumoperitoneum"), pActive, pActive ? "Desufflate abdomen" : "Start CO₂ insufflation");
    }
    if ($("c-pe") && c.embolism) {
      setPressed($("c-pe"), Boolean(c.embolism.pe_active),
        c.embolism.pe_active ? "Clear massive PE" : "Start massive PE");
    }
    if ($("c-vae") && c.embolism) {
      setPressed($("c-vae"), Boolean(c.embolism.vae_active),
        c.embolism.vae_active ? "Stop air entrainment" : "Start venous air embolism");
    }
    if ($("c-high-spinal") && c.neuraxial) {
      const hsActive = Boolean(c.neuraxial.active && c.neuraxial.mode === "high_spinal");
      setPressed($("c-high-spinal"), hsActive, hsActive ? "Clear high spinal" : "Trigger high / total spinal");
    }
    setPressed($("c-mh"), c.malignant_hyperthermia,
      c.malignant_hyperthermia ? "Stop malignant hyperthermia" : "Start malignant hyperthermia");

    const circulation = [];
    if (c.hemorrhage.active) circulation.push(`Bleeding ${c.hemorrhage.rate} mL/min`);
    if (c.anaphylaxis) circulation.push("Anaphylaxis");
    if (c.sepsis) circulation.push("Sepsis");
    if (c.protamine_reaction) circulation.push("Protamine reaction");
    if (c.embolism?.pe_active) circulation.push("Massive PE");
    if (c.embolism?.vae_active) circulation.push("Venous air embolism");
    if (c.neuraxial?.active && c.neuraxial?.mode === "high_spinal") circulation.push("High spinal");
    if (c.ecg_rhythm !== "Sinus rhythm") circulation.push(c.ecg_rhythm);
    this.syncEventSection("circulation", circulation);
    this.syncEventSection("metabolism", c.malignant_hyperthermia ? ["Malignant hyperthermia"] : []);
    const airway = [];
    if (c.obstruction > 0) airway.push(`Obstruction ${round(c.obstruction, 0)}%`);
    if (c.bronchospasm > 0) airway.push(`Bronchospasm ${round(c.bronchospasm, 0)}%`);
    if ((c.ards_shunt ?? 0) > 0) airway.push(`ARDS shunt ${round(c.ards_shunt, 0)}%`);
    if (c.laryngospasm !== "none") airway.push(`Laryngospasm ${c.laryngospasm}`);
    if (c.airway_crisis?.difficult_profile && c.airway_crisis.difficult_profile !== "none") {
      airway.push(c.airway_crisis.difficult_profile === "severe_cico" ? "CICO airway" : "Difficult DL");
    }
    if (c.airway_crisis?.esophageal_active) airway.push("Esophageal ETT");
    if (c.pneumothorax?.active) airway.push("Tension PTX");
    this.syncEventSection("airway", airway);
    // The disabled profile select already names the active stimulation.
    const stim = [];
    if (c.disturbance.active) stim.push($("c-disturbance").selectedOptions[0].text);
    if (c.pneumoperitoneum?.active) stim.push(`IAP ${round(c.pneumoperitoneum.iap, 0)} mmHg`);
    this.syncEventSection("stimulation", stim, false);
    this.syncEventSection("equipment", c.o2_connected ? [] : ["O₂ supply disconnected"]);
  }
}
