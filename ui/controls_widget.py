
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
    QPushButton,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QTabWidget,
    QGridLayout,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
)
from PySide6.QtCore import Qt
from core.state import AirwayType

from .styles import (
    COLORS,
    STYLE_GROUPBOX,
    STYLE_SPINBOX,
    STYLE_COMBOBOX,
    STYLE_LABEL,
    STYLE_TAB_WIDGET,
    STYLE_SCROLLAREA,
    get_groupbox_style,
    get_button_style,
    get_toggle_button_style,
    get_radiobutton_style,
    get_base_widget_style,
)
from physiology.disturbances import list_disturbance_profiles

class ControlPanelWidget(QWidget):
    """
    Control Panel for Anesthesia Delivery.
    Tabs: Machine, Drugs & Infusions, Events & Fluids.
    """
    def __init__(self, engine, tutorial_mode=False):
        super().__init__()
        self.engine = engine
        self.tutorial_mode = tutorial_mode
        self._disturbance_profiles = [("Off", None)]
        self._disturbance_profiles.extend(list_disturbance_profiles())
        self.setStyleSheet(f"""
            {get_base_widget_style()}
            {STYLE_SPINBOX}
            {STYLE_COMBOBOX}
            {STYLE_LABEL}
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(STYLE_TAB_WIDGET)
        layout.addWidget(self.tabs)
        
        # 1. Anesthesia Machine (Airway, Gases, Vent)
        self.tab_machine = QWidget()
        self.setup_machine_tab()
        self.tabs.addTab(self.tab_machine, "Machine")
        
        # 2. Infusions (TIVA/TCI)
        self.tab_drugs = QWidget()
        self.drug_widgets = {} 
        self.setup_drugs_tab()
        self.tabs.addTab(self.tab_drugs, "Drugs")

        # 4. Events / Fluids
        self.tab_events = QWidget()
        self.setup_events_tab()
        self.tabs.addTab(self.tab_events, "Events")

    def add_tutorial_label(self, layout, text):
        """Helper to add explanatory text if in tutorial mode."""
        if self.tutorial_mode:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 11px; margin-bottom: 6px; padding: 4px; background: transparent;"
            )
            layout.addWidget(lbl)

    def _create_scroll_area(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(STYLE_SCROLLAREA)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        scroll.setWidget(content)
        return scroll, layout
        
    def sync_with_engine(self):
        """Update UI controls to match Engine state."""
        # 1. Gases
        if self.engine.circuit and hasattr(self.engine.circuit, 'vaporizer_setting'):
            idx = self.cb_agent.findText(self.engine.active_agent)
            if idx >= 0:
                self.cb_agent.setCurrentIndex(idx)
            
            self.sb_vap.blockSignals(True)
            self.sb_vap.setValue(self.engine.circuit.vaporizer_setting)
            self.sb_vap.blockSignals(False)
            
        # 1.5 Airway Sync
        if hasattr(self.engine.state, 'airway_mode'):
            mode = self.engine.state.airway_mode
            self.abg_air.blockSignals(True)
            if mode == AirwayType.NONE:
                self.rb_none.setChecked(True)
            elif mode == AirwayType.MASK:
                self.rb_mask.setChecked(True)
            elif mode == AirwayType.ETT:
                self.rb_ett.setChecked(True)
            self.abg_air.blockSignals(False)

        # 2. Drugs (Refactored)
        for key, w in self.drug_widgets.items():
            if hasattr(self.engine, 'get_drug_state'):
                dstate = self.engine.get_drug_state(key)
                
                w['rb_tci'].blockSignals(True)
                w['rb_man'].blockSignals(True)
                w['rate'].blockSignals(True)
                w['target'].blockSignals(True)
                
                if dstate['is_tci']:
                    w['rb_tci'].setChecked(True)
                    w['target'].setValue(dstate['target'])
                    w['target'].setEnabled(True)
                    w['rate'].setEnabled(False)
                else:
                    w['rb_man'].setChecked(True)
                    w['rate'].setValue(dstate['rate'])
                    w['rate'].setEnabled(True)
                    w['target'].setEnabled(False)
                    
                w['rb_tci'].blockSignals(False)
                w['rb_man'].blockSignals(False)
                w['rate'].blockSignals(False)
                w['target'].blockSignals(False)
        
        # 3. Ventilator Sync
        if hasattr(self.engine, 'vent'):
            self.btn_vent_power.blockSignals(True)
            self.sb_rr.blockSignals(True)
            self.sb_tv.blockSignals(True)
            self.sb_peep.blockSignals(True)
            
            is_on = self.engine.vent.is_on
            self.btn_vent_power.setChecked(is_on)
            
            if is_on:
                self.btn_vent_power.setText("Ventilator ON")
                self.sb_rr.setEnabled(True)
                self.sb_tv.setEnabled(True)
                self.sb_peep.setEnabled(True)
                self.cb_ie.setEnabled(True)
            else:
                self.btn_vent_power.setText("Spontaneous / Manual")
                self.sb_rr.setEnabled(False)
                self.sb_tv.setEnabled(False)
                self.sb_peep.setEnabled(False)
                self.cb_ie.setEnabled(False)
            
            if is_on and hasattr(self.engine.vent, 'settings'):
                s = self.engine.vent.settings
                self.sb_rr.setValue(int(s.rr))
                self.sb_tv.setValue(int(s.tv))
                self.sb_peep.setValue(int(s.peep))
                
            self.btn_vent_power.blockSignals(False)
            self.sb_rr.blockSignals(False)
            self.sb_tv.blockSignals(False)
            self.sb_peep.blockSignals(False)

        # 4. Disturbances
        if hasattr(self, "cb_disturbance"):
            profile = getattr(self.engine, "disturbance_profile", None)
            idx = 0
            for i, (_, key) in enumerate(self._disturbance_profiles):
                if key == profile:
                    idx = i
                    break
            self.cb_disturbance.blockSignals(True)
            self.cb_disturbance.setCurrentIndex(idx)
            self.cb_disturbance.blockSignals(False)

            active = bool(getattr(self.engine, "disturbance_active", False) and profile)
            self.b_disturb.blockSignals(True)
            self.b_disturb.setChecked(active)
            self.b_disturb.setText("Stop stimulation" if active else "Start stimulation")
            self.b_disturb.setEnabled(profile is not None)
            self.b_disturb.blockSignals(False)
            self.cb_disturbance.setEnabled(not active)
        
    def update_fgf(self):
        o2 = self.sb_o2.value()
        air = self.sb_air.value()
        total = o2 + air
        if total > 0:
            fio2 = (o2 + 0.21 * air) / total
        else:
            fio2 = 0.21
        self.lbl_fio2.setText(f"{int(fio2*100)}%")
        
        if hasattr(self.engine, 'set_fgf'):
            self.engine.set_fgf(o2, air)
            
    def update_vaporizer(self):
        val = self.sb_vap.value()
        agent = self.cb_agent.currentText()
        if hasattr(self.engine, 'set_vaporizer'):
            self.engine.set_vaporizer(agent, val)

    def update_airway(self, btn):
        text = btn.text()
        mode = "None"
        if "Mask" in text: mode = "Mask"
        elif "ETT" in text: mode = "ETT"
        
        if mode == "ETT" and hasattr(self, 'btn_bag_mask') and self.btn_bag_mask.isChecked():
            self.btn_bag_mask.setChecked(False)
        
        if hasattr(self.engine, 'set_airway_mode'):
            self.engine.set_airway_mode(mode)
        
    def setup_machine_tab(self):
        """Combined Airway, Gases, and Ventilator controls."""
        main_layout = QVBoxLayout(self.tab_machine)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        scroll, layout = self._create_scroll_area()
        main_layout.addWidget(scroll)

        self.add_tutorial_label(layout, "The Anesthesia Machine integrates gas delivery, vaporizer, and mechanical ventilation.")

        # --- Section 1: Airway ---
        gp_air = QGroupBox("Airway management")
        gp_air.setStyleSheet(STYLE_GROUPBOX)
        l_air = QHBoxLayout(gp_air)
        l_air.setSpacing(8)
        
        self.abg_air = QButtonGroup(self)
        
        self.rb_none = self._create_radio_button("None", COLORS['text_dim'])
        self.rb_mask = self._create_radio_button("Mask", COLORS['info'])
        self.rb_ett = self._create_radio_button("ETT (Intubated)", COLORS['success'])
        
        self.abg_air.addButton(self.rb_none)
        self.abg_air.addButton(self.rb_mask)
        self.abg_air.addButton(self.rb_ett)
        self.rb_none.setChecked(True)
        self.abg_air.buttonClicked.connect(self.update_airway)
        
        l_air.addWidget(self.rb_none)
        l_air.addWidget(self.rb_mask)
        l_air.addWidget(self.rb_ett)
        l_air.addStretch()
        layout.addWidget(gp_air)
        
        # --- Section 2: Gas Flow & Vaporizer ---
        h_gases = QHBoxLayout()
        h_gases.setSpacing(12)
        
        # FGF
        gp_fgf = QGroupBox("Fresh gas flow")
        gp_fgf.setStyleSheet(STYLE_GROUPBOX)
        l_fgf = QGridLayout(gp_fgf)
        l_fgf.setSpacing(8)
        
        lbl_o2 = QLabel("O₂ (L/min)")
        lbl_o2.setStyleSheet(f"color: {COLORS['info']}; background: transparent;")
        self.sb_o2 = QDoubleSpinBox()
        self.sb_o2.setRange(0, 15)
        self.sb_o2.setValue(2.0)
        self.sb_o2.setSingleStep(0.5)
        self.sb_o2.valueChanged.connect(self.update_fgf)
        l_fgf.addWidget(lbl_o2, 0, 0)
        l_fgf.addWidget(self.sb_o2, 0, 1)
        
        lbl_air = QLabel("Air (L/min)")
        self.sb_air = QDoubleSpinBox()
        self.sb_air.setRange(0, 15)
        self.sb_air.setValue(0.0)
        self.sb_air.setSingleStep(0.5)
        self.sb_air.valueChanged.connect(self.update_fgf)
        l_fgf.addWidget(lbl_air, 1, 0)
        l_fgf.addWidget(self.sb_air, 1, 1)
        
        lbl_fio2_title = QLabel("FiO₂:")
        lbl_fio2_title.setStyleSheet(f"color: {COLORS['text_dim']}; background: transparent;")
        self.lbl_fio2 = QLabel("100%")
        self.lbl_fio2.setStyleSheet(f"font-weight: bold; color: {COLORS['success']}; font-size: 16px; background: transparent;")
        l_fgf.addWidget(lbl_fio2_title, 2, 0)
        l_fgf.addWidget(self.lbl_fio2, 2, 1)
        
        h_gases.addWidget(gp_fgf)
        
        # Vaporizer
        gp_vap = QGroupBox("Vaporizer")
        gp_vap.setStyleSheet(STYLE_GROUPBOX)
        l_vap = QFormLayout(gp_vap)
        l_vap.setSpacing(8)
        
        self.cb_agent = QComboBox()
        self.cb_agent.addItems(["Sevoflurane"])
        self.cb_agent.currentTextChanged.connect(self.update_vaporizer)
        l_vap.addRow("Agent:", self.cb_agent)
        
        self.sb_vap = QDoubleSpinBox()
        self.sb_vap.setRange(0, 8)
        self.sb_vap.setSuffix(" %")
        self.sb_vap.setSingleStep(0.2)
        self.sb_vap.valueChanged.connect(self.update_vaporizer)
        l_vap.addRow("Dial setting:", self.sb_vap)
        
        h_gases.addWidget(gp_vap)
        layout.addLayout(h_gases)
        
        # --- Section 3: Bag-Mask Ventilation ---
        gp_bag = QGroupBox("Manual ventilation")
        gp_bag.setStyleSheet(STYLE_GROUPBOX)
        l_bag = QHBoxLayout(gp_bag)
        
        self.btn_bag_mask = QPushButton("Bag-mask: OFF")
        self.btn_bag_mask.setCheckable(True)
        self.btn_bag_mask.setStyleSheet(get_toggle_button_style(COLORS['success']))
        self.btn_bag_mask.toggled.connect(self.toggle_bag_mask)
        l_bag.addWidget(self.btn_bag_mask)
        lbl_bag_info = QLabel("Simulates ~12 bpm, 500mL")
        lbl_bag_info.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;")
        l_bag.addWidget(lbl_bag_info)
        l_bag.addStretch()
        layout.addWidget(gp_bag)
        
        # --- Section 4: Mechanical Ventilator ---
        gp_vent = QGroupBox("Mechanical ventilation")
        gp_vent.setStyleSheet(STYLE_GROUPBOX)
        l_vent = QVBoxLayout(gp_vent)
        l_vent.setSpacing(10)
        
        # Header: Mode/Power
        h_vent_top = QHBoxLayout()
        self.btn_vent_power = QPushButton("Spontaneous / manual")
        self.btn_vent_power.setCheckable(True)
        self.btn_vent_power.setStyleSheet(get_toggle_button_style(COLORS['primary']))
        self.btn_vent_power.toggled.connect(self.toggle_vent_power)
        h_vent_top.addWidget(self.btn_vent_power)
        h_vent_top.addStretch()
        l_vent.addLayout(h_vent_top)
        
        # Vent Mode Selection (VCV / PCV)
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Mode:"))
        self.cb_vent_mode = QComboBox()
        self.cb_vent_mode.addItems(["VCV (Volume)", "PCV (Pressure)"])
        self.cb_vent_mode.currentIndexChanged.connect(self.on_vent_mode_changed)
        h_mode.addWidget(self.cb_vent_mode)
        h_mode.addStretch()
        l_vent.addLayout(h_mode)
        
        # Settings Grid
        g_vent = QGridLayout()
        g_vent.setSpacing(8)
        
        lbl_rr = QLabel("RR:")
        self.sb_rr = QSpinBox()
        self.sb_rr.setRange(0, 60)
        self.sb_rr.setValue(12)
        self.sb_rr.setSuffix(" /min")
        self.sb_rr.valueChanged.connect(self.update_vent)
        g_vent.addWidget(lbl_rr, 0, 0)
        g_vent.addWidget(self.sb_rr, 0, 1)
        
        # Tidal Volume (VCV mode)
        self.lbl_tv = QLabel("Vt:")
        self.sb_tv = QSpinBox()
        self.sb_tv.setRange(0, 1500)
        self.sb_tv.setValue(500)
        self.sb_tv.setSuffix(" mL")
        self.sb_tv.setSingleStep(50)
        self.sb_tv.valueChanged.connect(self.update_vent)
        g_vent.addWidget(self.lbl_tv, 0, 2)
        g_vent.addWidget(self.sb_tv, 0, 3)
        
        # Inspiratory Pressure (PCV mode) - initially hidden
        self.lbl_pinsp = QLabel("Pinsp:")
        self.sb_pinsp = QSpinBox()
        self.sb_pinsp.setRange(5, 40)
        self.sb_pinsp.setValue(15)
        self.sb_pinsp.setSuffix(" cmH₂O")
        self.sb_pinsp.setSingleStep(1)
        self.sb_pinsp.valueChanged.connect(self.update_vent)
        g_vent.addWidget(self.lbl_pinsp, 0, 4)
        g_vent.addWidget(self.sb_pinsp, 0, 5)
        self.lbl_pinsp.hide()
        self.sb_pinsp.hide()
        
        lbl_peep = QLabel("PEEP:")
        self.sb_peep = QSpinBox()
        self.sb_peep.setRange(0, 20)
        self.sb_peep.setValue(5)
        self.sb_peep.setSuffix(" cmH₂O")
        self.sb_peep.valueChanged.connect(self.update_vent)
        g_vent.addWidget(lbl_peep, 1, 0)
        g_vent.addWidget(self.sb_peep, 1, 1)
        
        lbl_ie = QLabel("I:E:")
        self.cb_ie = QComboBox()
        self.cb_ie.addItems(["1:2", "1:1", "1:3", "1:4"])
        self.cb_ie.currentIndexChanged.connect(self.update_vent)
        g_vent.addWidget(lbl_ie, 1, 2)
        g_vent.addWidget(self.cb_ie, 1, 3)
        
        l_vent.addLayout(g_vent)
        layout.addWidget(gp_vent)
        
        layout.addStretch()
        
        # Initial State
        self.sb_rr.setEnabled(False)
        self.sb_tv.setEnabled(False)
        self.sb_peep.setEnabled(False)
        self.cb_ie.setEnabled(False)
        self.cb_vent_mode.setEnabled(False)
        self.sb_pinsp.setEnabled(False)

    def _create_radio_button(self, text, color):
        """Create styled radio button."""
        rb = QRadioButton(text)
        rb.setStyleSheet(get_radiobutton_style(color))
        return rb

    def setup_drugs_tab(self):
        scroll, layout = self._create_scroll_area()
        
        self.add_tutorial_label(layout, "TCI (Target Controlled Infusion) uses pharmacokinetic models to target a specific concentration in the blood or brain. Manual mode sets a fixed rate.")
         
        def create_drug_box(name, unit_rate, tci_unit, set_rate_cb, set_tci_cb, key_name, def_bolus, color):
            gb = QGroupBox(name)
            gb.setStyleSheet(get_groupbox_style(color))
            l = QVBoxLayout(gb)
            l.setSpacing(8)
            
            # Mode Switch
            h_mode = QHBoxLayout()
            rb_man = self._create_radio_button("Manual", COLORS['text'])
            rb_tci = self._create_radio_button("TCI", COLORS['primary'])
            rb_man.setChecked(True)
            grp = QButtonGroup(gb)
            grp.addButton(rb_man)
            grp.addButton(rb_tci)
            h_mode.addWidget(rb_man)
            h_mode.addWidget(rb_tci)
            h_mode.addStretch()
            l.addLayout(h_mode)
            
            # Controls
            sb_rate = QDoubleSpinBox()
            sb_rate.setRange(0, 2000)
            sb_rate.setSuffix(f" {unit_rate}")
            lbl_rate = QLabel("Infusion rate:")
            lbl_rate.setStyleSheet("background: transparent;")
            l.addWidget(lbl_rate)
            l.addWidget(sb_rate)
            
            sb_target = QDoubleSpinBox()
            sb_target.setRange(0, 20)
            sb_target.setSuffix(f" {tci_unit}")
            sb_target.setEnabled(False)
            lbl_target = QLabel("Target concentration:")
            lbl_target.setStyleSheet("background: transparent;")
            l.addWidget(lbl_target)
            l.addWidget(sb_target)
            
            def mode_changed():
                is_tci = rb_tci.isChecked()
                sb_rate.setEnabled(not is_tci)
                sb_target.setEnabled(is_tci)
                if not is_tci:
                    set_tci_cb(None)
                    set_rate_cb(sb_rate.value())
                else:
                    set_tci_cb(sb_target.value())
                    
            rb_man.toggled.connect(mode_changed)
            rb_tci.toggled.connect(mode_changed)
            
            sb_rate.valueChanged.connect(lambda v: set_rate_cb(v) if rb_man.isChecked() else None)
            sb_target.valueChanged.connect(lambda v: set_tci_cb(v) if rb_tci.isChecked() else None)
            
            # Bolus Controls
            h_bolus = QHBoxLayout()
            sb_bolus = QDoubleSpinBox()
            sb_bolus.setRange(0, 1000)
            
            if "mg" in unit_rate: 
                sb_bolus.setSuffix(" mg")
            else: 
                sb_bolus.setSuffix(" mcg")
                
            sb_bolus.setValue(def_bolus)
                
            btn_give = QPushButton("Bolus")
            btn_give.setToolTip("Administer immediate bolus dose")
            btn_give.setStyleSheet(get_button_style(bg_color=color, padding="6px 14px"))
            btn_give.clicked.connect(lambda: self.engine.give_drug_bolus(name, sb_bolus.value()))
            
            h_bolus.addWidget(sb_bolus)
            h_bolus.addWidget(btn_give)
            
            l.addLayout(h_bolus)
            
            self.drug_widgets[key_name] = {
                'rb_man': rb_man,
                'rb_tci': rb_tci,
                'rate': sb_rate,
                'target': sb_target,
                'bolus': sb_bolus
            }
            
            return gb

        if not hasattr(self.engine, 'get_controllable_drugs'):
            layout.addWidget(QLabel("Error: Engine does not support drug interface."))
            layout.addStretch()
            return

        drugs = self.engine.get_controllable_drugs()
        
        # Color assignments for different drug types
        drug_colors = {
            'propofol': COLORS['drug_hypnotic'],
            'remi': COLORS['drug_opioid'],
            'roc': COLORS['drug_nmb'],
            'nore': COLORS['drug_pressor'],
            'epi': COLORS['drug_inotrope'],
            'phenyl': COLORS['drug_vasoconstrictor'],
        }
        
        for d in drugs:
            key = d['key']
            color = drug_colors.get(key, COLORS['primary'])
            
            def make_set_rate(k):
                return lambda v: self.engine.set_drug_rate(k, v)
                
            def make_set_tci(k):
                return lambda v: self.engine.set_drug_target(k, v)

            gb = create_drug_box(
                d['name'], 
                d['rate_unit'], 
                d['tci_unit'],
                make_set_rate(key),
                make_set_tci(key),
                key,
                d['default_bolus'],
                color
            )
            layout.addWidget(gb)

        # --- Reversal Agents (Bolus-only) ---
        gp_reversal = QGroupBox("Reversal agents")
        gp_reversal.setStyleSheet(get_groupbox_style(COLORS['info']))
        l_rev = QVBoxLayout(gp_reversal)
        l_rev.setSpacing(8)
        
        lbl_sug = QLabel("Sugammadex (NMB reversal)")
        lbl_sug.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold; background: transparent;")
        l_rev.addWidget(lbl_sug)
        
        lbl_sug_info = QLabel("2 mg/kg: moderate block  |  4 mg/kg: deep block  |  16 mg/kg: immediate")
        lbl_sug_info.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px; background: transparent;")
        l_rev.addWidget(lbl_sug_info)
        
        h_sug = QHBoxLayout()
        h_sug.setSpacing(8)
        
        sug_doses = [
            ("2 mg/kg", 2.0),
            ("4 mg/kg", 4.0),
            ("16 mg/kg", 16.0)
        ]
        
        for label, dose_mg_kg in sug_doses:
            btn = QPushButton(label)
            btn.setStyleSheet(get_button_style(bg_color=COLORS['info'], padding="6px 14px"))
            dose_mg = dose_mg_kg * self.engine.patient.weight
            btn.setToolTip(f"Administer {int(dose_mg)} mg Sugammadex")
            btn.clicked.connect(lambda checked, d=dose_mg: self.engine.give_drug_bolus("sugammadex", d))
            h_sug.addWidget(btn)
        
        h_sug.addStretch()
        l_rev.addLayout(h_sug)
        
        layout.addWidget(gp_reversal)

        layout.addStretch()
        
        main_l = QVBoxLayout(self.tab_drugs)
        main_l.setContentsMargins(6, 6, 6, 6)
        main_l.addWidget(scroll)

    def toggle_vent_power(self, checked):
        if checked:
            mode_text = "VCV" if self.cb_vent_mode.currentIndex() == 0 else "PCV"
            self.btn_vent_power.setText("Ventilator ON")
            # Turn off bag-mask when switching to mechanical vent (mutually exclusive)
            if hasattr(self, 'btn_bag_mask') and self.btn_bag_mask.isChecked():
                self.btn_bag_mask.setChecked(False)
            self.sb_rr.setEnabled(True)
            self.sb_peep.setEnabled(True)
            self.cb_ie.setEnabled(True)
            self.cb_vent_mode.setEnabled(True)
            self.on_vent_mode_changed(self.cb_vent_mode.currentIndex())
            self.update_vent()
        else:
            self.btn_vent_power.setText("Spontaneous / manual")
            self.sb_rr.setEnabled(False)
            self.sb_tv.setEnabled(False)
            self.sb_peep.setEnabled(False)
            self.cb_ie.setEnabled(False)
            self.cb_vent_mode.setEnabled(False)
            self.sb_pinsp.setEnabled(False)
            self.engine.set_vent_settings(0, 0, self.sb_peep.value(), "1:2")
    
    def on_vent_mode_changed(self, index):
        """Handle VCV/PCV mode switch."""
        is_vcv = (index == 0)
        
        if is_vcv:
            self.lbl_tv.show()
            self.sb_tv.show()
            self.lbl_pinsp.hide()
            self.sb_pinsp.hide()
            if self.btn_vent_power.isChecked():
                self.sb_tv.setEnabled(True)
                self.sb_pinsp.setEnabled(False)
        else:
            self.lbl_tv.hide()
            self.sb_tv.hide()
            self.lbl_pinsp.show()
            self.sb_pinsp.show()
            if self.btn_vent_power.isChecked():
                self.sb_tv.setEnabled(False)
                self.sb_pinsp.setEnabled(True)
        
        if self.btn_vent_power.isChecked():
            self.btn_vent_power.setText("Ventilator ON")
            self.update_vent()

    def update_vent(self):
        if self.btn_vent_power.isChecked():
            mode = "VCV" if self.cb_vent_mode.currentIndex() == 0 else "PCV"
            
            self.engine.set_vent_settings(
                self.sb_rr.value(),
                self.sb_tv.value() / 1000.0,
                self.sb_peep.value(),
                self.cb_ie.currentText(),
                mode=mode,
                p_insp=self.sb_pinsp.value()
            )
            
    def toggle_bag_mask(self, checked):
        """Toggle manual bag-mask ventilation (separate from mechanical vent)."""
        if checked:
            self.btn_bag_mask.setText("Bag-mask: ON")
            # Turn off mechanical vent if it's on (mutually exclusive in practice)
            if self.btn_vent_power.isChecked():
                self.btn_vent_power.setChecked(False)
            # Use dedicated bag-mask method (does NOT turn on mechanical vent)
            self.engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.5)
        else:
            self.btn_bag_mask.setText("Bag-mask: OFF")
            self.engine.set_bag_mask_ventilation(False)

    def setup_events_tab(self):
        main_layout = QVBoxLayout(self.tab_events)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Scroll area wrapper for consistent padding with other tabs
        scroll, layout = self._create_scroll_area()
        main_layout.addWidget(scroll)
        
        # Temperature Management
        gp_temp = QGroupBox("Patient warming")
        gp_temp.setStyleSheet(STYLE_GROUPBOX)
        l_temp = QHBoxLayout(gp_temp)
        
        self.combo_bair = QComboBox()
        self.combo_bair.addItems(["Off", "Low (32°C)", "Medium (38°C)", "High (43°C)"])
        self.combo_bair.currentIndexChanged.connect(self.change_bair_hugger)
        
        l_temp.addWidget(QLabel("Bair Hugger:"))
        l_temp.addWidget(self.combo_bair)
        l_temp.addStretch()
        
        layout.addWidget(gp_temp)
        
        # Fluids
        gp_fluids = QGroupBox("Fluid administration")
        gp_fluids.setStyleSheet(STYLE_GROUPBOX)
        l_fl = QHBoxLayout(gp_fluids)
        
        b_250 = QPushButton("250 mL")
        b_250.setStyleSheet(get_button_style(variant="info"))
        b_250.clicked.connect(lambda: self.engine.give_fluid(250))
        
        b_500 = QPushButton("500 mL")
        b_500.setStyleSheet(get_button_style(variant="info"))
        b_500.clicked.connect(lambda: self.engine.give_fluid(500))
        
        b_prbc = QPushButton("PRBC 300 mL")
        b_prbc.setStyleSheet(get_button_style(variant="primary"))
        b_prbc.clicked.connect(lambda: self.engine.give_blood(300))
        
        l_fl.addWidget(b_250)
        l_fl.addWidget(b_500)
        l_fl.addWidget(b_prbc)
        l_fl.addStretch()
        layout.addWidget(gp_fluids)

        # Scripted stimulation / disturbances
        gp_stim = QGroupBox("Surgical stimulation")
        gp_stim.setStyleSheet(STYLE_GROUPBOX)
        l_stim = QHBoxLayout(gp_stim)

        self.cb_disturbance = QComboBox()
        for label, _ in self._disturbance_profiles:
            self.cb_disturbance.addItem(label)
        self.cb_disturbance.currentIndexChanged.connect(self.on_disturbance_profile_changed)

        self.b_disturb = QPushButton("Start stimulation")
        self.b_disturb.setCheckable(True)
        self.b_disturb.setStyleSheet(get_toggle_button_style(COLORS['warning'], text_color=COLORS['warning']))
        self.b_disturb.toggled.connect(self.toggle_disturbance)
        self.b_disturb.setEnabled(False)

        l_stim.addWidget(self.cb_disturbance)
        l_stim.addWidget(self.b_disturb)
        l_stim.addStretch()
        layout.addWidget(gp_stim)
        
        # Crisis Events
        gp_crisis = QGroupBox("Critical events")
        gp_crisis.setStyleSheet(get_groupbox_style(COLORS['danger']))
        l_cr = QVBoxLayout(gp_crisis)
        l_cr.setSpacing(10)
        
        # Hemorrhage Controls
        h_hemo = QHBoxLayout()
        self.cb_hemo_severity = QComboBox()
        self.cb_hemo_severity.addItems([
            "Mild (500 mL/min)",
            "Moderate (1000 mL/min)",
            "Severe (2000 mL/min)",
            "Massive (4000 mL/min)"
        ])
        h_hemo.addWidget(self.cb_hemo_severity)
        
        self.b_hem = QPushButton("Start bleeding")
        self.b_hem.setToolTip("Simulate massive hemorrhage event")
        self.b_hem.setCheckable(True)
        self.b_hem.setStyleSheet(get_toggle_button_style(COLORS['danger'], text_color=COLORS['danger']))
        h_hemo.addWidget(self.b_hem)
        l_cr.addLayout(h_hemo)
        
        self.b_anaph = QPushButton("Start anaphylaxis")
        self.b_anaph.setToolTip("Simulate anaphylactic shock")
        self.b_anaph.setCheckable(True)
        self.b_anaph.setStyleSheet(get_toggle_button_style(COLORS['warning'], text_color=COLORS['warning']))
        l_cr.addWidget(self.b_anaph)
        
        b_stop = QPushButton("Stop all events")
        b_stop.setStyleSheet(get_button_style(outlined=True, variant="neutral"))
        l_cr.addWidget(b_stop)
        
        self.b_hem.toggled.connect(self.toggle_hemorrhage)
        self.b_anaph.toggled.connect(self.toggle_anaphylaxis)
        b_stop.clicked.connect(self.stop_all_events)
        
        layout.addWidget(gp_crisis)
        layout.addStretch()

    def on_disturbance_profile_changed(self, index):
        profile = self._disturbance_profiles[index][1]
        if profile is None:
            self.b_disturb.setEnabled(False)
            if self.b_disturb.isChecked():
                self.b_disturb.setChecked(False)
            if hasattr(self.engine, "stop_disturbance"):
                self.engine.stop_disturbance(clear_profile=True)
            return

        self.b_disturb.setEnabled(True)
        if hasattr(self.engine, "set_disturbance_profile"):
            self.engine.set_disturbance_profile(profile)
        if self.b_disturb.isChecked() and hasattr(self.engine, "start_disturbance"):
            self.engine.start_disturbance(profile)

    def toggle_disturbance(self, checked):
        profile = self._disturbance_profiles[self.cb_disturbance.currentIndex()][1]
        if checked:
            if not profile:
                self.b_disturb.setChecked(False)
                return
            self.b_disturb.setText("Stop Stimulation")
            self.cb_disturbance.setEnabled(False)
            if hasattr(self.engine, "start_disturbance"):
                self.engine.start_disturbance(profile)
        else:
            self.b_disturb.setText("Start Stimulation")
            self.cb_disturbance.setEnabled(True)
            if hasattr(self.engine, "stop_disturbance"):
                self.engine.stop_disturbance()

    def toggle_hemorrhage(self, checked):
        if checked:
            txt = self.cb_hemo_severity.currentText()
            rate = 500.0
            if "1000" in txt: rate = 1000.0
            elif "2000" in txt: rate = 2000.0
            elif "4000" in txt: rate = 4000.0
            
            self.b_hem.setText("Stop Bleeding")
            self.engine.start_hemorrhage(rate)
            self.cb_hemo_severity.setEnabled(False)
        else:
            self.b_hem.setText("Start Bleeding")
            self.engine.stop_hemorrhage()
            self.cb_hemo_severity.setEnabled(True)
            
    def toggle_anaphylaxis(self, checked):
        if checked:
            self.b_anaph.setText("Stop Anaphylaxis")
            self.engine.start_anaphylaxis()
        else:
            self.b_anaph.setText("Start Anaphylaxis")
            self.engine.stop_anaphylaxis()
            
    def stop_all_events(self):
        self.b_hem.setChecked(False)
        self.b_anaph.setChecked(False)
        if hasattr(self, "b_disturb"):
            self.b_disturb.setChecked(False)
        if hasattr(self.engine, 'stop_events'):
            self.engine.stop_events()
            
    def change_bair_hugger(self, index):
        """Handle Bair Hugger setting change."""
        target = 0.0
        if index == 1: target = 32.0
        elif index == 2: target = 38.0
        elif index == 3: target = 43.0
        
        if hasattr(self.engine, 'set_bair_hugger'):
            self.engine.set_bair_hugger(target)
