// Machine, medication, and event controls.
// Inputs send commands on change; snapshots update any input not being edited.

const $ = (id) => document.getElementById(id);

function setValue(input, value) {
  if (document.activeElement === input) return;
  const text = String(value);
  if (input.value !== text) input.value = text;
}

function setPressed(button, pressed, text) {
  button.setAttribute("aria-pressed", String(pressed));
  if (text !== undefined && button.textContent !== text) button.textContent = text;
}

function readNumber(input) {
  const min = input.min === "" ? -Infinity : Number(input.min);
  const max = input.max === "" ? Infinity : Number(input.max);
  let value = Number(input.value);
  if (!Number.isFinite(value)) value = Math.max(0, min);
  value = Math.min(Math.max(value, min), max);
  input.value = String(value);
  return value;
}

const round = (value, digits = 2) => Number(value.toFixed(digits));

export class Controls {
  constructor(info, send) {
    this.send = send;
    this.info = info;
    this.state = null;
    this.bindTabs();
    this.bindMachine();
    this.buildDrugs(info.drugs);
    this.bindEvents(info);
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
    for (const id of ["c-o2", "c-air", "c-n2o"]) $(id).onchange = fgf;
    $("c-o2-supply").onclick = () => this.send("oxygen_supply", { connected: !this.state.o2_connected });
    $("c-vap").onchange = () => this.send("vaporizer", { percent: readNumber($("c-vap")) });
    $("c-bag").onclick = () => this.send("bag_mask", { active: !this.state.bag_mask });
    $("c-vent-power").onclick = () => this.send("vent_power", { on: !this.state.vent.on });
    $("c-vent-mode").onchange = () => {
      this.applyVentMode($("c-vent-mode").value);
      this.send("vent", { mode: $("c-vent-mode").value });
    };
    for (const [id, field] of [["c-rr", "rr"], ["c-tv", "tv"], ["c-pinsp", "p_insp"], ["c-peep", "peep"]]) {
      $(id).onchange = () => this.send("vent", { [field]: readNumber($(id)) });
    }
    $("c-ie").onchange = () => this.send("vent", { ie: $("c-ie").value });
  }

  applyVentMode(mode) {
    const volume = mode === "VCV";
    $("c-tv-label").hidden = !volume;
    $("c-tv-wrap").hidden = !volume;
    const pressure = mode === "PCV" || mode === "PSV";
    $("c-pinsp-label").hidden = !pressure;
    $("c-pinsp-wrap").hidden = !pressure;
  }

  // --- Medications --------------------------------------------------

  buildDrugs(drugs) {
    const container = $("drug-cards");
    container.replaceChildren();
    this.drugs = {};
    for (const spec of drugs) {
      const infusion = spec.rate_unit !== null;
      const tci = spec.tci_unit !== null;
      const card = document.createElement("fieldset");
      card.className = "group drug-card";
      card.innerHTML = `
        <legend></legend>
        ${tci ? `<div class="segmented compact">
          <button type="button" data-mode="rate">Rate</button>
          <button type="button" data-mode="tci">TCI</button>
        </div>` : ""}
        ${infusion ? `<div class="inputs">
          <label>Infusion rate</label>
          ${tci ? '<label class="target-label"></label>' : "<span></span>"}
          <span class="unit-input"><input class="rate" type="number" min="0" max="2000" step="any"><span class="rate-unit"></span></span>
          ${tci ? '<span class="unit-input"><input class="target" type="number" step="0.1"><span class="target-unit"></span></span>' : ""}
        </div>` : ""}
        <div class="bolus">
          <span class="unit-input"><input class="bolus-amount" type="number" min="0" max="1000" step="any"><span class="bolus-unit"></span></span>
          <button type="button" class="btn outlined">Give bolus</button>
        </div>
        <p class="csht" hidden></p>`;
      card.querySelector("legend").textContent = spec.name;
      card.querySelector(".bolus-unit").textContent = spec.bolus_unit;

      const w = {
        rateButton: card.querySelector('[data-mode="rate"]'),
        tciButton: card.querySelector('[data-mode="tci"]'),
        rate: card.querySelector(".rate"),
        target: card.querySelector(".target"),
        bolus: card.querySelector(".bolus-amount"),
        give: card.querySelector(".bolus .btn"),
        csht: card.querySelector(".csht"),
      };
      w.bolus.value = String(spec.default_bolus);

      const key = spec.key;
      if (infusion) {
        card.querySelector(".rate-unit").textContent = spec.rate_unit;
        w.rate.value = "0";
        w.rate.onchange = () => {
          if (!this.state.drugs[key].is_tci) this.send("drug_rate", { key, rate: readNumber(w.rate) });
        };
      }
      if (tci) {
        card.querySelector(".target-label").textContent = spec.target_label;
        card.querySelector(".target-unit").textContent = spec.tci_unit;
        w.target.min = String(spec.tci_range[0]);
        w.target.max = String(spec.tci_range[1]);
        w.target.value = "0";
        w.tciButton.onclick = () => {
          if (!this.state.drugs[key].is_tci) this.send("drug_target", { key, target: readNumber(w.target) });
        };
        w.rateButton.onclick = () => {
          if (!this.state.drugs[key].is_tci) return;
          this.send("drug_target", { key, target: null });
          this.send("drug_rate", { key, rate: readNumber(w.rate) });
        };
        w.target.onchange = () => {
          if (this.state.drugs[key].is_tci) this.send("drug_target", { key, target: readNumber(w.target) });
        };
      }
      w.give.onclick = () => this.send("drug_bolus", { key, amount: readNumber(w.bolus) });
      this.drugs[key] = w;
      container.append(card);
    }
    for (const button of document.querySelectorAll("[data-sugammadex]")) {
      button.onclick = () => this.send("sugammadex", { mg_per_kg: Number(button.dataset.sugammadex) });
    }
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
    $("c-maint").onchange = () => this.send("maintenance_fluid", { ml_hr: readNumber($("c-maint")) });

    const profiles = $("c-disturbance");
    profiles.replaceChildren(new Option("Off", ""));
    for (const { key, label } of info.disturbances) profiles.append(new Option(label, key));
    profiles.onchange = () => this.send("disturbance_profile", { profile: profiles.value || null });
    $("c-disturb").onclick = () => this.send("disturbance", { active: !this.state.disturbance.active });

    $("c-obstruction").onchange = () => this.send("obstruction", { percent: readNumber($("c-obstruction")) });
    $("c-bronchospasm").onchange = () => this.send("bronchospasm", { percent: readNumber($("c-bronchospasm")) });
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
    this.applyVentMode(vent.mode);
    setValue($("c-rr"), vent.rr);
    setValue($("c-tv"), vent.tv);
    setValue($("c-pinsp"), vent.p_insp);
    setValue($("c-peep"), vent.peep);
    setValue($("c-ie"), vent.ie);

    for (const [key, w] of Object.entries(this.drugs)) {
      const d = c.drugs[key];
      if (w.tciButton) {
        w.rateButton.setAttribute("aria-pressed", String(!d.is_tci));
        w.tciButton.setAttribute("aria-pressed", String(d.is_tci));
        w.target.disabled = !d.is_tci;
        if (d.is_tci) setValue(w.target, round(d.target));
      }
      if (w.rate) {
        w.rate.disabled = d.is_tci;
        setValue(w.rate, round(d.rate));
      }
    }

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
