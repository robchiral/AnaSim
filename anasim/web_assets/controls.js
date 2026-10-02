// Machine, medication, and event controls.
// Numeric settings remain drafts until explicitly applied.
import { SettingsEditor } from "./settings.js";

const $ = (id) => document.getElementById(id);
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
  button.setAttribute("aria-pressed", String(pressed));
  if (text !== undefined && button.textContent !== text) button.textContent = text;
}

function readNumber(input) {
  return Number(input.value);
}

const round = (value, digits = 2) => Number(value.toFixed(digits));

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
  { name: "Opioids", keys: ["fentanyl", "remi"] },
  { name: "Vasopressors and inotropes", keys: ["phenyl", "nore", "epi", "vaso", "dobu", "milri"] },
  { name: "Heart rate and blood pressure", keys: ["glyco", "esmolol", "labetalol"] },
  { name: "Neuromuscular blockade and reversal", keys: ["roc", "sugammadex"] },
];

export class Controls {
  constructor(info, send) {
    this.settingEditors = [];
    this.destroyed = false;
    this.history = [...info.medication_history];
    this.send = async (name, args) => {
      const result = await send(name, args);
      if (!this.destroyed && result?.medication) {
        this.history.push(result.medication);
        this.history = this.history.slice(-50);
        this.showMedicationHistory();
      }
      return result;
    };
    this.info = info;
    this.state = null;
    this.bindTabs();
    this.bindMachine();
    this.buildDrugs(info.drugs);
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
    for (const editor of this.settingEditors) {
      for (const field of editor.fields) editors.delete(field);
      editor.destroy();
    }
  }

  async medicationAction(button, name, args) {
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
    $("medication-history-empty").hidden = this.history.length > 0;
    const latest = new Map();
    for (const entry of this.history) {
      const seconds = Math.floor(entry.time);
      const stamp = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
      const row = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = stamp;
      row.append(time, entry.text);
      list.prepend(row);
      latest.set(entry.key, `${entry.text} · ${stamp}`);
    }
    for (const [key, text] of latest) {
      const feedback = this.medicationRows.get(key).card.querySelector(".medication-feedback");
      feedback.hidden = false;
      if (feedback.textContent !== text) feedback.textContent = text;
    }
  }

  // --- Tabs ---------------------------------------------------------

  bindTabs() {
    this.tabs = [...document.querySelectorAll(".tabs button")];
    for (const tab of this.tabs) tab.onclick = () => this.openTab(tab.dataset.tab);
    this.openTab("Machine");
  }

  openTab(name) {
    for (const tab of this.tabs) tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
    for (const panel of document.querySelectorAll(".tab-panel")) panel.hidden = panel.dataset.panel !== name;
    this.currentTab = name;
    if (name === "Medications") this.onMedicationsShown?.();
  }

  // --- Machine ------------------------------------------------------

  bindMachine() {
    for (const button of $("c-airway").querySelectorAll("button")) {
      button.onclick = () => this.send("airway", { mode: button.dataset.value });
    }
    const fgf = () => this.send("fgf", { o2: readNumber($("c-o2")), air: readNumber($("c-air")), n2o: readNumber($("c-n2o")) });
    this.editSettings([$("c-o2"), $("c-air"), $("c-n2o")], fgf);
    $("c-o2-supply").onclick = () => this.send("oxygen_supply", { connected: !this.state.o2_connected });
    this.editSettings([$("c-vap")], () => this.send("vaporizer", { percent: readNumber($("c-vap")) }));
    $("c-bag").onclick = () => this.send("bag_mask", { active: !this.state.bag_mask });
    $("c-vent-power").onclick = () => this.send("vent_power", { on: !this.state.vent.on });
    this.editSettings(
      ["c-vent-mode", "c-ie", ...Object.values(ventInputs)].map($),
      () => this.send("vent", {
        mode: $("c-vent-mode").value, ie: $("c-ie").value,
        ...Object.fromEntries(Object.entries(ventInputs).map(([name, id]) => [name, readNumber($(id))])),
      }),
      $("c-vent-mode").closest("fieldset"),
      () => this.applyVentMode($("c-vent-mode").value),
    );
  }

  applyVentMode(mode) {
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
        <summary></summary>
        <div class="drug-controls">
        ${tci ? `<label class="inline">Infusion mode
          <select class="infusion-mode"><option value="rate">Manual rate</option><option value="tci">TCI</option></select>
        </label>` : ""}
        ${infusion ? `<div class="inputs">
          <label>Infusion rate</label>
          ${tci ? '<label class="target-label"></label>' : "<span></span>"}
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
      card.querySelector("summary").textContent = spec.name;
      card.querySelector(".bolus-unit").textContent = spec.bolus_unit;

      const w = {
        card,
        spec,
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
        return this.medicationAction(w.give, "drug_bolus", { key, amount: readNumber(w.bolus) });
      };
      this.drugs[key] = w;
      this.medicationRows.set(key, { card, name: spec.name });
      if (infusion) this.buildRunningInfusion(key, w);
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
    this.medicationRows.set("sugammadex", { card: reversal, name: "Sugammadex" });
    for (const group of this.medicationGroups) {
      for (const key of group.keys) group.section.append(this.medicationRows.get(key).card);
    }
    for (const button of document.querySelectorAll("[data-sugammadex]")) {
      button.onclick = () => this.medicationAction(button, "sugammadex", { mg_per_kg: Number(button.dataset.sugammadex) });
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
      <button type="button" class="btn small outlined infusion-stop">Stop</button>`;
    row.querySelector(".infusion-name").textContent = w.spec.name;
    const adjust = row.querySelector(".infusion-adjust");
    adjust.setAttribute("aria-label", `Adjust ${w.spec.name}`);
    adjust.onclick = () => {
      $("drug-search").value = "";
      this.filterMedications();
      w.card.parentElement.open = true;
      w.card.open = true;
      const input = w.mode?.value === "tci" ? w.target : w.rate;
      input.focus({ preventScroll: true });
      w.card.scrollIntoView({ block: "nearest" });
    };
    const stop = row.querySelector(".infusion-stop");
    stop.setAttribute("aria-label", `Stop ${w.spec.name}`);
    // A manual rate command also disables TCI in the shared engine.
    stop.onclick = async () => {
      const result = await this.medicationAction(stop, "drug_rate", { key, rate: 0 });
      if (result !== undefined) w.editor.cancel();
    };
    w.runningRow = row;
    w.runningValue = row.querySelector("output");
    $("running-infusion-list").append(row);
  }

  showCsht(values) {
    for (const key of ["propofol", "remi", "fentanyl"]) {
      const label = this.drugs[key]?.csht;
      if (!label) continue;
      const minutes = values[key];
      label.hidden = !(minutes > 0);
      if (minutes > 0) {
        label.textContent = `Effect-site half-time ~${minutes.toFixed(0)} min`;
        label.title = "Estimated PK effect-site half-time from the current model state; not a guaranteed wake-up time.";
      }
    }
  }

  // --- Events and fluids --------------------------------------------

  bindEvents(info) {
    $("c-bair").onchange = () => this.send("bair_hugger", { target_c: Number($("c-bair").value) });
    for (const button of document.querySelectorAll("[data-fluid]")) {
      button.onclick = () => this.send("fluid", { kind: button.dataset.fluid, volume_ml: Number(button.dataset.volume) });
    }
    this.editSettings([$("c-maint")], () => this.send("maintenance_fluid", { ml_hr: readNumber($("c-maint")) }));

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
    $("c-stop-events").onclick = () => this.send("stop_events");
  }

  // --- Sync from snapshots ------------------------------------------

  sync(c) {
    this.state = c;

    for (const button of $("c-airway").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.value === c.airway));
    }
    setValue($("c-o2"), round(c.fgf.o2));
    setValue($("c-air"), round(c.fgf.air));
    setValue($("c-n2o"), round(c.fgf.n2o));
    const fio2 = $("c-fio2");
    fio2.textContent = `${Math.round(c.circuit_fio2 * 100)}%`;
    fio2.className = "fio2" + (!c.o2_connected || c.circuit_fio2 < 0.21 ? " bad" : c.circuit_fio2 < 0.3 ? " warn" : "");
    setPressed($("c-o2-supply"), !c.o2_connected, c.o2_connected ? "Disconnect O₂ supply" : "Connect backup O₂");
    setValue($("c-vap"), round(c.vaporizer, 1));

    setPressed($("c-bag"), c.bag_mask, c.bag_mask ? "Stop bag-mask ventilation" : "Start bag-mask ventilation");
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
        setValue(w.rate, round(d.rate));
        const active = d.rate > 0 || (d.is_tci && d.target > 0);
        if (!active && w.runningRow.contains(document.activeElement)) $("drug-search").focus();
        w.runningRow.hidden = !active;
        if (active) {
          w.runningValue.textContent = d.is_tci
            ? `${w.spec.target_label}: ${round(d.target)} ${w.spec.tci_unit} · ${round(d.rate)} ${w.spec.rate_unit}`
            : `${round(d.rate)} ${w.spec.rate_unit}`;
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
    $("c-laryngo").textContent = `Laryngospasm: ${c.laryngospasm}`;
    $("c-auto-laryngo").checked = c.auto_laryngospasm;

    setPressed($("c-hemorrhage"), c.hemorrhage.active, c.hemorrhage.active ? "Stop bleeding" : "Start bleeding");
    $("c-hem-rate").disabled = c.hemorrhage.active;
    if (c.hemorrhage.active) setValue($("c-hem-rate"), String(c.hemorrhage.rate));
    setValue($("c-rhythm"), c.rhythm);
    setPressed($("c-anaphylaxis"), c.anaphylaxis, c.anaphylaxis ? "Stop anaphylaxis" : "Start anaphylaxis");
    setPressed($("c-sepsis"), c.sepsis, c.sepsis ? "Stop sepsis" : "Start sepsis");
  }
}
