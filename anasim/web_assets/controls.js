// Machine, medication, fluid, and event controls.
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

const round = (value, digits = 2) => Number(value.toFixed(digits));
// Four significant figures keep small rates visible without long decimals.
const rateText = (value) => String(Number(value.toPrecision(4)));

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
  { name: "Anesthesia and sedation", keys: ["propofol", "midazolam", "ketamine", "etomidate", "lidocaine"] },
  { name: "Opioids", keys: ["fentanyl", "remi", "hydromorphone", "morphine"] },
  { name: "Vasopressors and inotropes", keys: ["phenyl", "ephedrine", "nore", "epi", "vaso", "dobu", "milri"] },
  { name: "Heart rate and blood pressure", keys: ["atropine", "glyco", "esmolol", "labetalol"] },
  { name: "Neuromuscular blockade and reversal", keys: ["roc", "succinylcholine", "sugammadex"] },
];
const halfTimeDrugs = ["propofol", "remi", "fentanyl"];

export class Controls {
  constructor(info, send) {
    this.settingEditors = [];
    this.destroyed = false;
    this.feedbackTimers = new Map();
    this.history = [...info.medication_history];
    this.send = async (name, args) => {
      const result = await send(name, args);
      if (!this.destroyed && result?.medication) {
        this.history.push(result.medication);
        this.history = this.history.slice(-50);
        this.showMedicationHistory();
        this.showMedicationFeedback(result.medication);
      }
      return result;
    };
    this.state = null;
    this.bindTabs();
    this.bindMachine();
    this.buildDrugs(info.drugs);
    this.bindFluids();
    this.bindEvents(info);
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
      const seconds = Math.floor(entry.time);
      const stamp = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
      const row = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = stamp;
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
    for (const button of $("c-airway").querySelectorAll("button")) {
      button.onclick = () => this.send("airway", { mode: button.dataset.value });
    }
    const fgf = () => this.send("fgf", { o2: readNumber($("c-o2")), air: readNumber($("c-air")), n2o: readNumber($("c-n2o")) });
    this.editSettings([$("c-o2"), $("c-air"), $("c-n2o")], fgf, $("c-o2").closest(".fresh-gas-settings"));
    this.editSettings([$("c-vap")], () => this.send("vaporizer", { percent: readNumber($("c-vap")) }));
    $("c-bag").onclick = () => this.send("bag_mask", { active: !this.state.bag_mask });
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
      card.querySelector(".drug-strength").textContent = ` · ${spec.name.slice(spec.generic_name.length).trim()}`;
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
      this.medicationRows.set(key, { card, name: spec.name, feedback: card.querySelector(".medication-feedback") });
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
    for (const group of this.medicationGroups) {
      for (const key of group.keys) group.section.append(this.medicationRows.get(key).card);
    }
    for (const button of document.querySelectorAll("[data-sugammadex]")) {
      button.onclick = () => this.performAction(button, "sugammadex", { mg_per_kg: Number(button.dataset.sugammadex) });
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
        label.title = "Estimated PK effect-site half-time from the current model state; not a guaranteed wake-up time.";
      }
    }
  }

  // --- Fluids -------------------------------------------------------

  bindFluids() {
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
    $("c-anaphylaxis").onclick = () => this.send("anaphylaxis", { active: !this.state.anaphylaxis });
    $("c-sepsis").onclick = () => this.send("sepsis", { active: !this.state.sepsis });
  }

  openEventSection(name) {
    this.eventSections.get(name).section.open = true;
  }

  syncEventSection(name, states) {
    const { section, summary, output } = this.eventSections.get(name);
    const active = states.length > 0;
    const value = String(active);
    if (section.dataset.active !== value) {
      section.dataset.active = value;
      summary.setAttribute("aria-disabled", value);
      summary.tabIndex = active ? -1 : 0;
    }
    if (active && !section.open) section.open = true;
    output.hidden = !active;
    setText(output, states.join(" · "));
  }

  // --- Sync from snapshots ------------------------------------------

  sync(c) {
    this.state = c;

    for (const button of $("c-airway").querySelectorAll("button")) {
      setPressed(button, button.dataset.value === c.airway);
    }
    setValue($("c-o2"), round(c.fgf.o2));
    setValue($("c-air"), round(c.fgf.air));
    setValue($("c-n2o"), round(c.fgf.n2o));
    const fio2 = $("c-fio2");
    setText(fio2, `${Math.round(c.circuit_fio2 * 100)}%`);
    fio2.className = "fio2" + (!c.o2_connected || c.circuit_fio2 < 0.21 ? " bad" : c.circuit_fio2 < 0.3 ? " warn" : "");
    setPressed($("c-o2-supply"), !c.o2_connected, c.o2_connected ? "Disconnect O₂ supply" : "Connect backup O₂");
    setValue($("c-vap"), round(c.vaporizer, 1));

    setPressed($("c-bag"), c.bag_mask, c.bag_mask ? "Stop bag ventilation" : "Start bag ventilation");
    $("c-bag-settings").hidden = !c.bag_mask;
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
        setValue(w.rate, rateText(d.rate));
        const active = d.rate > 0 || (d.is_tci && d.target > 0);
        if (!active && w.runningRow.contains(document.activeElement)) $("drug-search").focus();
        if (!active) this.closeInfusionEditor(w);
        w.runningRow.hidden = !active;
        if (active) {
          setText(w.runningValue, d.is_tci
            ? `${w.spec.target_label}: ${round(d.target)} ${w.spec.tci_unit} · ${rateText(d.rate)} ${w.spec.rate_unit}`
            : `${rateText(d.rate)} ${w.spec.rate_unit}`);
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
    setValue($("c-rhythm"), c.rhythm);
    setPressed($("c-anaphylaxis"), c.anaphylaxis, c.anaphylaxis ? "Stop anaphylaxis" : "Start anaphylaxis");
    setPressed($("c-sepsis"), c.sepsis, c.sepsis ? "Stop sepsis" : "Start sepsis");

    const circulation = [];
    if (c.hemorrhage.active) circulation.push(`Bleeding ${c.hemorrhage.rate} mL/min`);
    if (c.anaphylaxis) circulation.push("Anaphylaxis");
    if (c.sepsis) circulation.push("Sepsis");
    if (c.rhythm !== "Sinus rhythm") circulation.push(c.rhythm);
    this.syncEventSection("circulation", circulation);
    const airway = [];
    if (c.obstruction > 0) airway.push(`Obstruction ${round(c.obstruction, 0)}%`);
    if (c.bronchospasm > 0) airway.push(`Bronchospasm ${round(c.bronchospasm, 0)}%`);
    if (c.laryngospasm !== "none") airway.push(`Laryngospasm ${c.laryngospasm}`);
    this.syncEventSection("airway", airway);
    this.syncEventSection("stimulation", c.disturbance.active ? [$("c-disturbance").selectedOptions[0].text] : []);
    this.syncEventSection("equipment", c.o2_connected ? [] : ["O₂ supply disconnected"]);
  }
}
