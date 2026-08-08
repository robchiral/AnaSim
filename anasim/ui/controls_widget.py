import time

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anasim.core.enums import RhythmType
from anasim.core.state import AirwayType
from anasim.physiology.disturbances import list_disturbance_profiles

from .styles import (
    COLORS,
    STYLE_COMBOBOX,
    STYLE_LABEL,
    STYLE_SCROLLAREA,
    STYLE_SPINBOX,
    STYLE_TAB_WIDGET,
    get_base_widget_style,
    get_button_style,
    get_drug_card_style,
    get_section_group_style,
    get_segment_button_style,
    get_toggle_button_style,
)


class ControlPanelWidget(QWidget):
    """Controls for anesthesia delivery, interventions, and simulated events."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._disturbance_profiles = [("Off", None)]
        self._disturbance_profiles.extend(list_disturbance_profiles())
        self._last_sync_state = None
        self._last_drug_state = None
        self._last_circuit_readout = None
        self._next_csht_update = 0.0
        self.setStyleSheet(f"""
            {get_base_widget_style()}
            {STYLE_SPINBOX}
            {STYLE_COMBOBOX}
            {STYLE_LABEL}
        """)
        self.setMinimumWidth(480)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(STYLE_TAB_WIDGET)
        layout.addWidget(self.tabs)

        self.tab_machine = QWidget()
        self.setup_machine_tab()
        self.tabs.addTab(self.tab_machine, "Machine")

        self.tab_drugs = QWidget()
        self.drug_widgets = {}
        self.setup_drugs_tab()
        self.tabs.addTab(self.tab_drugs, "Medications")

        self.tab_events = QWidget()
        self.setup_events_tab()
        self.tabs.addTab(self.tab_events, "Events and fluids")

    def open_tab(self, name: str):
        """Open a named control area requested by a tutorial objective."""
        tabs = {
            "Machine": self.tab_machine,
            "Medications": self.tab_drugs,
            "Events": self.tab_events,
        }
        self.tabs.setCurrentWidget(tabs[name])

    def _create_scroll_area(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(STYLE_SCROLLAREA)
        content = QWidget()
        content.setObjectName("controlScrollContent")
        content.setStyleSheet(
            f"QWidget#controlScrollContent {{ background-color: {COLORS['panel']}; }}"
        )
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 8, 16)
        layout.setSpacing(8)
        scroll.setWidget(content)
        return scroll, layout

    def _silent_update(self, widget, setter_name, value):
        """Silently update a widget without triggering its signals."""
        signals_were_blocked = widget.blockSignals(True)
        try:
            getattr(widget, setter_name)(value)
        finally:
            widget.blockSignals(signals_were_blocked)

    def sync_with_engine(self):
        """Update UI controls to match Engine state."""
        self._update_circuit_readout()

        settings = self.engine.vent.settings
        drug_state = []
        for spec in self.engine.get_controllable_drugs():
            controller = getattr(self.engine, spec.tci_attr)
            drug_state.append(
                (
                    spec.key,
                    getattr(self.engine, spec.rate_attr),
                    controller.target if controller is not None else None,
                )
            )
        drug_state = tuple(drug_state)
        if drug_state != self._last_drug_state:
            self._last_drug_state = drug_state
            self._sync_drug_controls()

        now = time.monotonic()
        if (
            self.tabs.currentWidget() is self.tab_drugs
            and now >= self._next_csht_update
        ):
            self._sync_csht()
            self._next_csht_update = now + 5.0

        laryngospasm_level = sum(
            self.engine.laryngospasm_severity >= threshold
            for threshold in (0.05, 0.3, 0.6)
        )
        sync_state = (
            self.engine.circuit.vaporizer_setting,
            self.engine.circuit.fgf_o2,
            self.engine.circuit.fgf_air,
            self.engine.circuit.fgf_n2o,
            self.engine.circuit.oxygen_supply_connected,
            self.engine.vent.is_on,
            settings.mode,
            settings.rr,
            settings.tv,
            settings.peep,
            settings.ie_ratio,
            settings.p_insp,
            self.engine.bag_mask_active,
            self.engine.maintenance_fluid_rate_ml_min,
            self.engine.disturbance_profile,
            self.engine.disturbance_active,
            self.engine.airway_obstruction_manual,
            self.engine.bronchospasm_manual,
            laryngospasm_level,
            self.engine.auto_laryngospasm_enabled,
            self.engine.state.airway_mode,
        )
        if sync_state == self._last_sync_state:
            return
        self._last_sync_state = sync_state

        self._sync_gases_and_airway()
        self._sync_ventilator()
        self._sync_fluids()
        self._sync_disturbances()
        self._sync_airway_complications()

    def _sync_gases_and_airway(self):
        """Sync gases, vaporizer, and airway choices."""
        circuit = self.engine.circuit
        self._silent_update(self.sb_vap, "setValue", circuit.vaporizer_setting)
        self._silent_update(self.sb_o2, "setValue", circuit.fgf_o2)
        self._silent_update(self.sb_air, "setValue", circuit.fgf_air)
        self._silent_update(self.sb_n2o, "setValue", circuit.fgf_n2o)
        disconnected = not circuit.oxygen_supply_connected
        self._silent_update(self.btn_o2_supply, "setChecked", disconnected)
        self.btn_o2_supply.setText(
            "Connect backup O₂" if disconnected else "Disconnect O₂ supply"
        )

        mode = self.engine.state.airway_mode
        signals_were_blocked = self.abg_air.blockSignals(True)
        try:
            if mode == AirwayType.NONE:
                self.rb_none.setChecked(True)
            elif mode == AirwayType.MASK:
                self.rb_mask.setChecked(True)
            elif mode == AirwayType.ETT:
                self.rb_ett.setChecked(True)
        finally:
            self.abg_air.blockSignals(signals_were_blocked)

    def _sync_drug_controls(self):
        """Sync infusion mode, rate, and target controls."""
        for key, w in self.drug_widgets.items():
            dstate = self.engine.get_drug_state(key)

            self._silent_update(w["rb_tci"], "setChecked", dstate["is_tci"])
            self._silent_update(w["rb_man"], "setChecked", not dstate["is_tci"])
            self._silent_update(w["target"], "setValue", dstate["target"])
            self._silent_update(w["rate"], "setValue", dstate["rate"])

            if dstate["is_tci"]:
                w["target"].setEnabled(True)
                w["rate"].setEnabled(False)
            else:
                w["rate"].setEnabled(True)
                w["target"].setEnabled(False)

    def _sync_csht(self):
        """Refresh expensive context-sensitive half-time estimates."""
        for key, w in self.drug_widgets.items():
            if w["csht_label"] is not None:
                csht = self.engine.get_predicted_csht(key)
                if csht > 0:
                    w["csht_label"].setText(
                        f"Estimated context-sensitive half-time: {csht:.0f} min"
                    )
                    w["csht_label"].setToolTip(
                        "Estimated PK effect-site half-time from the current model state; not a guaranteed wake-up time."
                    )
                    w["csht_label"].show()
                else:
                    w["csht_label"].hide()

    def _sync_ventilator(self):
        """Sync ventilator settings."""
        is_on = self.engine.vent.is_on
        self._silent_update(self.btn_vent_power, "setChecked", is_on)
        settings = self.engine.vent.settings
        mode_index = self.cb_vent_mode.findData(settings.mode)
        self._silent_update(self.cb_vent_mode, "setCurrentIndex", mode_index)
        self._silent_update(self.sb_rr, "setValue", int(settings.rr))
        self._silent_update(self.sb_tv, "setValue", int(settings.tv))
        self._silent_update(self.sb_peep, "setValue", int(settings.peep))
        self._silent_update(self.sb_pinsp, "setValue", int(settings.p_insp))
        self._silent_update(
            self.btn_bag_mask, "setChecked", self.engine.bag_mask_active
        )
        self.btn_bag_mask.setText(
            "Stop bag-mask ventilation"
            if self.engine.bag_mask_active
            else "Start bag-mask ventilation"
        )

        if is_on:
            self.btn_vent_power.setText("Stop ventilator")
            self.sb_rr.setEnabled(True)
            self.sb_tv.setEnabled(True)
            self.sb_peep.setEnabled(True)
            self.cb_ie.setEnabled(True)
        else:
            self.btn_vent_power.setText("Start ventilator")
            self.sb_rr.setEnabled(False)
            self.sb_tv.setEnabled(False)
            self.sb_peep.setEnabled(False)
            self.cb_ie.setEnabled(False)
        self._apply_vent_mode_controls(settings.mode)

    def _sync_fluids(self):
        """Sync continuous fluid rate."""
        self._silent_update(
            self.sb_cont_fluid, "setValue", self.engine.get_continuous_fluid_rate()
        )

    def _sync_disturbances(self):
        """Sync programmed disturbance states."""
        profile = self.engine.disturbance_profile
        idx = next(
            (
                i
                for i, (_, key) in enumerate(self._disturbance_profiles)
                if key == profile
            ),
            0,
        )
        self._silent_update(self.cb_disturbance, "setCurrentIndex", idx)

        active = bool(self.engine.disturbance_active and profile)
        self._silent_update(self.b_disturb, "setChecked", active)
        self.b_disturb.setText("Stop stimulation" if active else "Start stimulation")
        self.b_disturb.setEnabled(profile is not None)
        self.cb_disturbance.setEnabled(not active)

    def _sync_airway_complications(self):
        """Sync airway obstruction and laryngospasm."""
        self._silent_update(
            self.sb_obstruction,
            "setValue",
            self.engine.airway_obstruction_manual * 100.0,
        )
        self._silent_update(
            self.sb_bronchospasm,
            "setValue",
            self.engine.bronchospasm_manual * 100.0,
        )

        laryng = self.engine.laryngospasm_severity
        if laryng < 0.05:
            level = "none"
        elif laryng < 0.3:
            level = "mild"
        elif laryng < 0.6:
            level = "moderate"
        else:
            level = "severe"
        self.lbl_laryngo_status.setText(f"Laryngospasm: {level}")

        auto_on = self.engine.auto_laryngospasm_enabled
        self._silent_update(self.btn_auto_laryngo, "setChecked", auto_on)
        self.btn_auto_laryngo.setText(
            "Automatic laryngospasm on" if auto_on else "Automatic laryngospasm off"
        )

    def update_fgf(self):
        o2 = self.sb_o2.value()
        air = self.sb_air.value()
        n2o = self.sb_n2o.value()
        self.engine.set_fgf(o2, air, n2o)
        self._update_circuit_readout()

    def toggle_oxygen_supply(self, disconnected):
        """Simulate loss or restoration of the oxygen source."""
        self.engine.set_oxygen_supply_connected(not disconnected)
        self.btn_o2_supply.setText(
            "Connect backup O₂" if disconnected else "Disconnect O₂ supply"
        )
        self._last_sync_state = None
        self._update_circuit_readout()

    def _update_circuit_readout(self):
        """Refresh the measured oxygen concentration in the breathing circuit."""
        fio2 = self.engine.circuit.composition.fio2
        if not self.engine.circuit.oxygen_supply_connected or fio2 < 0.21:
            color = COLORS["danger"]
        elif fio2 < 0.30:
            color = COLORS["warning"]
        else:
            color = COLORS["success"]
        percent = round(fio2 * 100)
        readout = (percent, color)
        if readout == self._last_circuit_readout:
            return
        self._last_circuit_readout = readout
        self.lbl_fio2.setText(f"{percent}%")
        self.lbl_fio2.setStyleSheet(
            f"font-weight: 700; color: {color}; font-size: 16px;"
        )

    def update_vaporizer(self):
        val = self.sb_vap.value()
        self.engine.set_vaporizer(self.engine.active_agent, val)

    def update_airway(self, btn):
        mode = btn.property("airway_mode")

        if mode == "ETT" and self.btn_bag_mask.isChecked():
            self.btn_bag_mask.setChecked(False)

        self.engine.set_airway_mode(mode)

    def setup_machine_tab(self):
        """Combined Airway, Gases, and Ventilator controls."""
        main_layout = QVBoxLayout(self.tab_machine)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll, layout = self._create_scroll_area()
        main_layout.addWidget(scroll)

        # --- Section 1: Airway ---
        gp_air = QGroupBox("Airway management")
        gp_air.setStyleSheet(get_section_group_style())
        l_air = QVBoxLayout(gp_air)
        l_air.setSpacing(8)

        self.abg_air = QButtonGroup(self)

        self.rb_none = self._create_segment_button("Disconnected", COLORS["text_dim"])
        self.rb_mask = self._create_segment_button("Facemask", COLORS["info"])
        self.rb_ett = self._create_segment_button("Tracheal tube", COLORS["success"])
        self.rb_none.setProperty("airway_mode", "None")
        self.rb_mask.setProperty("airway_mode", "Mask")
        self.rb_ett.setProperty("airway_mode", "ETT")

        self.abg_air.addButton(self.rb_none)
        self.abg_air.addButton(self.rb_mask)
        self.abg_air.addButton(self.rb_ett)
        self.rb_none.setChecked(True)
        self.abg_air.buttonClicked.connect(self.update_airway)

        airway_buttons = QHBoxLayout()
        airway_buttons.setSpacing(6)
        airway_buttons.addWidget(self.rb_none)
        airway_buttons.addWidget(self.rb_mask)
        airway_buttons.addWidget(self.rb_ett)
        l_air.addLayout(airway_buttons)
        layout.addWidget(gp_air)

        # --- Section 2: Gas Flow & Vaporizer ---
        h_gases = QHBoxLayout()
        h_gases.setSpacing(12)

        # FGF
        gp_fgf = QGroupBox("Fresh gas flow")
        gp_fgf.setStyleSheet(get_section_group_style())
        l_fgf = QGridLayout(gp_fgf)
        l_fgf.setSpacing(8)

        lbl_o2 = QLabel("O₂")
        lbl_o2.setStyleSheet(f"color: {COLORS['info']};")
        self.sb_o2 = QDoubleSpinBox()
        self.sb_o2.setRange(0, 15)
        self.sb_o2.setValue(2.0)
        self.sb_o2.setSingleStep(0.5)
        self.sb_o2.setSuffix(" L/min")
        self.sb_o2.valueChanged.connect(self.update_fgf)
        l_fgf.addWidget(lbl_o2, 0, 0)
        l_fgf.addWidget(self.sb_o2, 0, 1)

        lbl_air = QLabel("Air")
        self.sb_air = QDoubleSpinBox()
        self.sb_air.setRange(0, 15)
        self.sb_air.setValue(0.0)
        self.sb_air.setSingleStep(0.5)
        self.sb_air.setSuffix(" L/min")
        self.sb_air.valueChanged.connect(self.update_fgf)
        l_fgf.addWidget(lbl_air, 1, 0)
        l_fgf.addWidget(self.sb_air, 1, 1)

        lbl_n2o = QLabel("N₂O")
        lbl_n2o.setStyleSheet(f"color: {COLORS['warning']};")
        self.sb_n2o = QDoubleSpinBox()
        self.sb_n2o.setRange(0, 15)
        self.sb_n2o.setValue(0.0)
        self.sb_n2o.setSingleStep(0.5)
        self.sb_n2o.setSuffix(" L/min")
        self.sb_n2o.valueChanged.connect(self.update_fgf)
        l_fgf.addWidget(lbl_n2o, 2, 0)
        l_fgf.addWidget(self.sb_n2o, 2, 1)

        lbl_fio2_title = QLabel("Circuit FiO₂")
        lbl_fio2_title.setStyleSheet(f"color: {COLORS['text_dim']};")
        self.lbl_fio2 = QLabel("21%")
        self.lbl_fio2.setStyleSheet(
            f"font-weight: bold; color: {COLORS['success']}; font-size: 16px;"
        )
        l_fgf.addWidget(lbl_fio2_title, 3, 0)
        l_fgf.addWidget(self.lbl_fio2, 3, 1)

        self.btn_o2_supply = QPushButton("Disconnect O₂ supply")
        self.btn_o2_supply.setCheckable(True)
        self.btn_o2_supply.setStyleSheet(
            get_toggle_button_style(COLORS["danger"], text_color=COLORS["danger"])
        )
        self.btn_o2_supply.toggled.connect(self.toggle_oxygen_supply)
        l_fgf.addWidget(self.btn_o2_supply, 4, 0, 1, 2)

        h_gases.addWidget(gp_fgf)

        # Vaporizer
        gp_vap = QGroupBox("Sevoflurane vaporizer")
        gp_vap.setStyleSheet(get_section_group_style())
        l_vap = QFormLayout(gp_vap)
        l_vap.setSpacing(8)

        self.sb_vap = QDoubleSpinBox()
        self.sb_vap.setRange(0, 8)
        self.sb_vap.setSuffix(" %")
        self.sb_vap.setSingleStep(0.2)
        self.sb_vap.valueChanged.connect(self.update_vaporizer)
        l_vap.addRow("Setting", self.sb_vap)

        h_gases.addWidget(gp_vap)
        layout.addLayout(h_gases)

        # --- Section 3: Bag-Mask Ventilation ---
        gp_bag = QGroupBox("Manual ventilation")
        gp_bag.setStyleSheet(get_section_group_style())
        l_bag = QHBoxLayout(gp_bag)

        self.btn_bag_mask = QPushButton("Start bag-mask ventilation")
        self.btn_bag_mask.setCheckable(True)
        self.btn_bag_mask.setStyleSheet(get_toggle_button_style(COLORS["success"]))
        self.btn_bag_mask.toggled.connect(self.toggle_bag_mask)
        l_bag.addWidget(self.btn_bag_mask)
        lbl_bag_info = QLabel("12 breaths/min · 500 mL")
        lbl_bag_info.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        l_bag.addWidget(lbl_bag_info)
        l_bag.addStretch()
        layout.addWidget(gp_bag)

        # --- Section 4: Mechanical Ventilator ---
        gp_vent = QGroupBox("Mechanical ventilation")
        gp_vent.setStyleSheet(get_section_group_style())
        l_vent = QVBoxLayout(gp_vent)
        l_vent.setSpacing(10)

        # Header: Mode/Power
        h_vent_top = QHBoxLayout()
        self.btn_vent_power = QPushButton("Start ventilator")
        self.btn_vent_power.setCheckable(True)
        self.btn_vent_power.setStyleSheet(get_toggle_button_style(COLORS["primary"]))
        self.btn_vent_power.toggled.connect(self.toggle_vent_power)
        h_vent_top.addWidget(self.btn_vent_power)
        h_vent_top.addStretch()
        l_vent.addLayout(h_vent_top)

        # Vent Mode Selection (VCV / PCV)
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Mode:"))
        self.cb_vent_mode = QComboBox()
        for label, mode in (
            ("VCV (Volume)", "VCV"),
            ("PCV (Pressure)", "PCV"),
            ("PSV (Support)", "PSV"),
            ("CPAP", "CPAP"),
        ):
            self.cb_vent_mode.addItem(label, mode)
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
        self.sb_pinsp.setRange(0, 40)
        self.sb_pinsp.setValue(15)
        self.sb_pinsp.setSuffix(" cmH₂O")
        self.sb_pinsp.setSingleStep(1)
        self.sb_pinsp.valueChanged.connect(self.update_vent)
        # They swap with Tidal Volume, so they can occupy the same cells
        g_vent.addWidget(self.lbl_pinsp, 0, 2)
        g_vent.addWidget(self.sb_pinsp, 0, 3)
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

    def _create_segment_button(self, text, color=COLORS["primary"], compact=False):
        """Create an exclusive, checkable state button."""
        button = QPushButton(text)
        button.setCheckable(True)
        button.setStyleSheet(get_segment_button_style(color, compact=compact))
        return button

    def setup_drugs_tab(self):
        scroll, layout = self._create_scroll_area()

        def create_drug_box(spec, set_rate_cb, set_tci_cb):
            gb = QGroupBox(spec.name)
            gb.setStyleSheet(get_drug_card_style())
            drug_layout = QVBoxLayout(gb)
            drug_layout.setSpacing(8)

            # Mode Switch
            h_mode = QHBoxLayout()
            rb_man = self._create_segment_button(
                "Rate", COLORS["text_secondary"], compact=True
            )
            rb_tci = self._create_segment_button("TCI", COLORS["primary"], compact=True)
            rb_man.setChecked(True)
            grp = QButtonGroup(gb)
            grp.addButton(rb_man)
            grp.addButton(rb_tci)
            h_mode.addWidget(rb_man)
            h_mode.addWidget(rb_tci)
            h_mode.addStretch()
            drug_layout.addLayout(h_mode)

            # Controls
            sb_rate = QDoubleSpinBox()
            sb_rate.setRange(0, 2000)
            sb_rate.setSuffix(f" {spec.rate_unit}")
            sb_rate.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            sb_target = QDoubleSpinBox()
            sb_target.setRange(*spec.tci_range)
            sb_target.setSuffix(f" {spec.tci_unit}")
            sb_target.setEnabled(False)
            sb_target.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            controls_grid = QGridLayout()
            controls_grid.setHorizontalSpacing(10)
            controls_grid.setVerticalSpacing(4)

            lbl_rate = QLabel("Infusion rate")
            lbl_rate.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 10px;"
            )
            target_compartment = (
                spec.fixed_tci_mode.value.replace("_", " ")
                if spec.fixed_tci_mode
                else "effect site"
            )
            lbl_target = QLabel(f"{target_compartment.capitalize()} target")
            lbl_target.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 10px;"
            )

            controls_grid.addWidget(lbl_rate, 0, 0)
            controls_grid.addWidget(lbl_target, 0, 1)
            controls_grid.addWidget(sb_rate, 1, 0)
            controls_grid.addWidget(sb_target, 1, 1)
            controls_grid.setColumnStretch(0, 1)
            controls_grid.setColumnStretch(1, 1)
            drug_layout.addLayout(controls_grid)

            def mode_changed():
                is_tci = rb_tci.isChecked()
                sb_rate.setEnabled(not is_tci)
                sb_target.setEnabled(is_tci)
                if not is_tci:
                    set_tci_cb(None)
                    set_rate_cb(sb_rate.value())
                else:
                    set_tci_cb(sb_target.value())

            rb_tci.toggled.connect(mode_changed)

            sb_rate.valueChanged.connect(
                lambda v: set_rate_cb(v) if rb_man.isChecked() else None
            )
            sb_target.valueChanged.connect(
                lambda v: set_tci_cb(v) if rb_tci.isChecked() else None
            )

            # Bolus Controls
            h_bolus = QHBoxLayout()
            sb_bolus = QDoubleSpinBox()
            sb_bolus.setRange(0, 1000)

            sb_bolus.setSuffix(f" {spec.bolus_unit}")
            sb_bolus.setValue(spec.default_bolus)

            btn_give = QPushButton("Give bolus")
            btn_give.setStyleSheet(
                get_button_style(variant="neutral", outlined=True, padding="6px 14px")
            )
            btn_give.clicked.connect(
                lambda: self.engine.give_drug_bolus(spec.key, sb_bolus.value())
            )

            h_bolus.addWidget(sb_bolus)
            h_bolus.addWidget(btn_give)

            drug_layout.addLayout(h_bolus)

            # PK effect-site half-time estimate for propofol and remi.
            lbl_csht = None
            if spec.key in ("propofol", "remi"):
                lbl_csht = QLabel()
                lbl_csht.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 10px; font-style: italic;"
                )
                lbl_csht.hide()  # Hidden until drug active
                drug_layout.addWidget(lbl_csht)

            self.drug_widgets[spec.key] = {
                "rb_man": rb_man,
                "rb_tci": rb_tci,
                "rate": sb_rate,
                "target": sb_target,
                "bolus": sb_bolus,
                "csht_label": lbl_csht,
            }

            return gb

        drugs = self.engine.get_controllable_drugs()

        for spec in drugs:
            key = spec.key

            def make_set_rate(k):
                return lambda v: self.engine.set_drug_rate(k, v)

            def make_set_tci(k):
                return lambda v: self.engine.set_drug_target(k, v)

            gb = create_drug_box(
                spec,
                make_set_rate(key),
                make_set_tci(key),
            )
            layout.addWidget(gb)

        gp_reversal = QGroupBox("Sugammadex")
        gp_reversal.setStyleSheet(get_drug_card_style())
        l_rev = QVBoxLayout(gp_reversal)
        l_rev.setSpacing(8)

        lbl_sug_info = QLabel(
            "Moderate block 2 mg/kg  •  Deep block 4 mg/kg  •  Immediate reversal 16 mg/kg"
        )
        lbl_sug_info.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px;"
        )
        l_rev.addWidget(lbl_sug_info)

        h_sug = QHBoxLayout()
        h_sug.setSpacing(8)

        sug_doses = [("2 mg/kg", 2.0), ("4 mg/kg", 4.0), ("16 mg/kg", 16.0)]

        for label, dose_mg_kg in sug_doses:
            btn = QPushButton(label)
            btn.setStyleSheet(
                get_button_style(variant="neutral", outlined=True, padding="6px 14px")
            )
            dose_mg = dose_mg_kg * self.engine.patient.weight
            btn.clicked.connect(
                lambda checked, d=dose_mg: self.engine.give_drug_bolus("sugammadex", d)
            )
            h_sug.addWidget(btn)

        h_sug.addStretch()
        l_rev.addLayout(h_sug)

        layout.addWidget(gp_reversal)

        layout.addStretch()

        main_l = QVBoxLayout(self.tab_drugs)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.addWidget(scroll)

    def toggle_vent_power(self, checked):
        if checked:
            self.btn_vent_power.setText("Stop ventilator")
            # Turn off bag-mask when switching to mechanical vent (mutually exclusive)
            if self.btn_bag_mask.isChecked():
                self.btn_bag_mask.setChecked(False)
            self.sb_rr.setEnabled(True)
            self.sb_peep.setEnabled(True)
            self.cb_ie.setEnabled(True)
            self.cb_vent_mode.setEnabled(True)
            self.on_vent_mode_changed(self.cb_vent_mode.currentIndex())
            self.update_vent()
        else:
            self.btn_vent_power.setText("Start ventilator")
            self.sb_rr.setEnabled(False)
            self.sb_tv.setEnabled(False)
            self.sb_peep.setEnabled(False)
            self.cb_ie.setEnabled(False)
            self.cb_vent_mode.setEnabled(False)
            self.sb_pinsp.setEnabled(False)
            mode = self.cb_vent_mode.currentData()
            self.engine.set_vent_settings(0, 0, 0.0, "1:2", mode=mode, p_insp=0.0)

    def on_vent_mode_changed(self, index):
        """Handle ventilator mode switch."""
        self._apply_vent_mode_controls(self.cb_vent_mode.itemData(index))

        if self.btn_vent_power.isChecked():
            self.btn_vent_power.setText("Stop ventilator")
            self.update_vent()

    def _apply_vent_mode_controls(self, mode):
        """Show and enable only the settings relevant to a ventilator mode."""
        is_vcv = mode == "VCV"
        is_pcv = mode == "PCV"
        is_psv = mode == "PSV"
        is_cpap = mode == "CPAP"

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
                self.sb_pinsp.setEnabled(is_pcv or is_psv)
                if is_cpap:
                    self.sb_pinsp.setValue(0)
                    self.sb_pinsp.setEnabled(False)

    def update_vent(self):
        if self.btn_vent_power.isChecked():
            mode = self.cb_vent_mode.currentData()
            p_insp = self.sb_pinsp.value() if mode in ("PCV", "PSV") else 0.0

            self.engine.set_vent_settings(
                self.sb_rr.value(),
                self.sb_tv.value() / 1000.0,
                self.sb_peep.value(),
                self.cb_ie.currentText(),
                mode=mode,
                p_insp=p_insp,
            )

    def toggle_bag_mask(self, checked):
        """Toggle manual bag-mask ventilation (separate from mechanical vent)."""
        if checked:
            self.btn_bag_mask.setText("Stop bag-mask ventilation")
            # Turn off mechanical vent if it's on (mutually exclusive in practice)
            if self.btn_vent_power.isChecked():
                self.btn_vent_power.setChecked(False)
            # Use dedicated bag-mask method (does NOT turn on mechanical vent)
            self.engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.5)
        else:
            self.btn_bag_mask.setText("Start bag-mask ventilation")
            self.engine.set_bag_mask_ventilation(False)

    def setup_events_tab(self):
        main_layout = QVBoxLayout(self.tab_events)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area wrapper for consistent padding with other tabs
        scroll, layout = self._create_scroll_area()
        main_layout.addWidget(scroll)

        # Temperature Management
        gp_temp = QGroupBox("Temperature management")
        gp_temp.setStyleSheet(get_section_group_style())
        l_temp = QHBoxLayout(gp_temp)

        self.combo_bair = QComboBox()
        for label, target in (
            ("Off", 0.0),
            ("Low (32°C)", 32.0),
            ("Medium (38°C)", 38.0),
            ("High (43°C)", 43.0),
        ):
            self.combo_bair.addItem(label, target)
        self.combo_bair.currentIndexChanged.connect(self.change_bair_hugger)

        l_temp.addWidget(QLabel("Forced-air warmer"))
        l_temp.addWidget(self.combo_bair)
        l_temp.addStretch()

        layout.addWidget(gp_temp)

        # Fluids
        gp_fluids = QGroupBox("Fluid administration")
        gp_fluids.setStyleSheet(get_section_group_style())
        l_fl = QVBoxLayout(gp_fluids)
        l_fl.setSpacing(6)

        b_250 = QPushButton("Crystalloid 250 mL")
        b_250.setStyleSheet(
            get_button_style(variant="info", padding="6px 10px", min_width=90)
        )
        b_250.clicked.connect(lambda: self.engine.give_fluid(250))

        b_500 = QPushButton("Crystalloid 500 mL")
        b_500.setStyleSheet(
            get_button_style(variant="info", padding="6px 10px", min_width=90)
        )
        b_500.clicked.connect(lambda: self.engine.give_fluid(500))

        b_albumin = QPushButton("Albumin 250 mL")
        b_albumin.setStyleSheet(
            get_button_style(variant="success", padding="6px 10px", min_width=120)
        )
        b_albumin.clicked.connect(lambda: self.engine.give_albumin(250))

        b_prbc = QPushButton("PRBC 300 mL")
        b_prbc.setStyleSheet(
            get_button_style(variant="primary", padding="6px 10px", min_width=120)
        )
        b_prbc.clicked.connect(lambda: self.engine.give_blood(300))

        l_fl_grid = QGridLayout()
        l_fl_grid.setHorizontalSpacing(8)
        l_fl_grid.setVerticalSpacing(6)
        l_fl_grid.addWidget(b_250, 0, 0)
        l_fl_grid.addWidget(b_500, 0, 1)
        l_fl_grid.addWidget(b_albumin, 1, 0)
        l_fl_grid.addWidget(b_prbc, 1, 1)

        # Continuous fluids
        l_fl_cont = QHBoxLayout()
        l_fl_cont.addWidget(QLabel("Maintenance fluid"))
        self.sb_cont_fluid = QDoubleSpinBox()
        self.sb_cont_fluid.setRange(0, 5000)
        self.sb_cont_fluid.setSingleStep(25)
        self.sb_cont_fluid.setSuffix(" mL/hr")
        self.sb_cont_fluid.setToolTip("Continuous IV fluids (mL/hr)")
        self.sb_cont_fluid.setMaximumWidth(140)
        self.sb_cont_fluid.setValue(self.engine.get_continuous_fluid_rate())
        self.sb_cont_fluid.valueChanged.connect(
            lambda v: self.engine.set_continuous_fluid_rate(v)
        )
        l_fl_cont.addWidget(self.sb_cont_fluid)
        l_fl_cont.addStretch()

        l_fl.addLayout(l_fl_grid)
        l_fl.addLayout(l_fl_cont)
        layout.addWidget(gp_fluids)

        # Scripted stimulation / disturbances
        gp_stim = QGroupBox("Surgical stimulation")
        gp_stim.setStyleSheet(get_section_group_style())
        l_stim = QHBoxLayout(gp_stim)

        self.cb_disturbance = QComboBox()
        for label, _ in self._disturbance_profiles:
            self.cb_disturbance.addItem(label)
        self.cb_disturbance.currentIndexChanged.connect(
            self.on_disturbance_profile_changed
        )

        self.b_disturb = QPushButton("Start stimulation")
        self.b_disturb.setCheckable(True)
        self.b_disturb.setStyleSheet(
            get_toggle_button_style(COLORS["warning"], text_color=COLORS["warning"])
        )
        self.b_disturb.toggled.connect(self.toggle_disturbance)
        self.b_disturb.setEnabled(False)

        l_stim.addWidget(self.cb_disturbance)
        l_stim.addWidget(self.b_disturb)
        l_stim.addStretch()
        layout.addWidget(gp_stim)

        # Airway complications
        gp_airway_events = QGroupBox("Airway complications")
        gp_airway_events.setStyleSheet(get_section_group_style())
        l_airway = QGridLayout(gp_airway_events)
        l_airway.setSpacing(8)

        lbl_obstruction = QLabel("Upper airway obstruction")
        lbl_obstruction.setStyleSheet("")
        self.sb_obstruction = QDoubleSpinBox()
        self.sb_obstruction.setRange(0, 100)
        self.sb_obstruction.setSuffix(" %")
        self.sb_obstruction.setSingleStep(5)
        self.sb_obstruction.valueChanged.connect(self.update_airway_obstruction)
        l_airway.addWidget(lbl_obstruction, 0, 0)
        l_airway.addWidget(self.sb_obstruction, 0, 1)

        lbl_bronch = QLabel("Bronchospasm")
        lbl_bronch.setStyleSheet("")
        self.sb_bronchospasm = QDoubleSpinBox()
        self.sb_bronchospasm.setRange(0, 100)
        self.sb_bronchospasm.setSuffix(" %")
        self.sb_bronchospasm.setSingleStep(5)
        self.sb_bronchospasm.valueChanged.connect(self.update_bronchospasm)
        l_airway.addWidget(lbl_bronch, 1, 0)
        l_airway.addWidget(self.sb_bronchospasm, 1, 1)

        self.lbl_laryngo_status = QLabel("Laryngospasm: none")
        self.lbl_laryngo_status.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        l_airway.addWidget(self.lbl_laryngo_status, 2, 0, 1, 2)

        self.btn_auto_laryngo = QPushButton("Automatic laryngospasm on")
        self.btn_auto_laryngo.setCheckable(True)
        self.btn_auto_laryngo.setChecked(True)
        self.btn_auto_laryngo.setStyleSheet(get_segment_button_style(COLORS["warning"]))
        self.btn_auto_laryngo.toggled.connect(self.update_auto_laryngospasm)
        l_airway.addWidget(self.btn_auto_laryngo, 3, 0, 1, 2)
        layout.addWidget(gp_airway_events)

        # Crisis Events
        gp_crisis = QGroupBox("Critical events")
        gp_crisis.setStyleSheet(get_section_group_style())
        l_cr = QVBoxLayout(gp_crisis)
        l_cr.setSpacing(10)

        # Hemorrhage Controls
        h_hemo = QHBoxLayout()
        self.cb_hemo_severity = QComboBox()
        self.cb_hemo_severity.addItem("Mild (500 mL/min)", 500.0)
        self.cb_hemo_severity.addItem("Moderate (1000 mL/min)", 1000.0)
        self.cb_hemo_severity.addItem("Severe (2000 mL/min)", 2000.0)
        self.cb_hemo_severity.addItem("Massive (4000 mL/min)", 4000.0)
        h_hemo.addWidget(self.cb_hemo_severity)

        self.b_hem = QPushButton("Start bleeding")
        self.b_hem.setCheckable(True)
        self.b_hem.setStyleSheet(
            get_toggle_button_style(COLORS["danger"], text_color=COLORS["danger"])
        )
        h_hemo.addWidget(self.b_hem)
        l_cr.addLayout(h_hemo)

        # Arrhythmia Controls
        h_arr = QHBoxLayout()
        lbl_arr = QLabel("Cardiac rhythm:")
        lbl_arr.setStyleSheet("")
        h_arr.addWidget(lbl_arr)

        self.cb_rhythm = QComboBox()
        self.cb_rhythm.addItems([r.value for r in RhythmType])
        # Set default to Sinus
        idx = self.cb_rhythm.findText(RhythmType.SINUS.value)
        self.cb_rhythm.setCurrentIndex(idx)

        self.cb_rhythm.currentTextChanged.connect(lambda t: self.engine.set_rhythm(t))
        h_arr.addWidget(self.cb_rhythm)
        l_cr.addLayout(h_arr)

        self.b_anaph = QPushButton("Start anaphylaxis")
        self.b_anaph.setCheckable(True)
        self.b_anaph.setStyleSheet(
            get_toggle_button_style(COLORS["warning"], text_color=COLORS["warning"])
        )
        l_cr.addWidget(self.b_anaph)

        self.b_sepsis = QPushButton("Start sepsis")
        self.b_sepsis.setCheckable(True)
        self.b_sepsis.setStyleSheet(
            get_toggle_button_style(COLORS["danger"], text_color=COLORS["danger"])
        )
        l_cr.addWidget(self.b_sepsis)

        b_stop = QPushButton("Stop all events")
        b_stop.setStyleSheet(get_button_style(outlined=True, variant="neutral"))
        l_cr.addWidget(b_stop)

        self.b_hem.toggled.connect(self.toggle_hemorrhage)
        self.b_anaph.toggled.connect(self.toggle_anaphylaxis)
        self.b_sepsis.toggled.connect(self.toggle_sepsis)
        b_stop.clicked.connect(self.stop_all_events)

        layout.addWidget(gp_crisis)
        layout.addStretch()

    def on_disturbance_profile_changed(self, index):
        profile = self._disturbance_profiles[index][1]
        if profile is None:
            self.b_disturb.setEnabled(False)
            if self.b_disturb.isChecked():
                self.b_disturb.setChecked(False)
            self.engine.stop_disturbance(clear_profile=True)
            return

        self.b_disturb.setEnabled(True)
        self.engine.set_disturbance_profile(profile)
        if self.b_disturb.isChecked():
            self.engine.start_disturbance(profile)

    def toggle_disturbance(self, checked):
        profile = self._disturbance_profiles[self.cb_disturbance.currentIndex()][1]
        if checked:
            if not profile:
                self.b_disturb.setChecked(False)
                return
            self.b_disturb.setText("Stop stimulation")
            self.cb_disturbance.setEnabled(False)
            self.engine.start_disturbance(profile)
        else:
            self.b_disturb.setText("Start stimulation")
            self.cb_disturbance.setEnabled(True)
            self.engine.stop_disturbance()

    def _toggle_simple_event(self, button, checked, start_fn, stop_fn, label):
        """Shared toggle behavior for simple on/off events."""
        if checked:
            button.setText(f"Stop {label}")
            start_fn()
        else:
            button.setText(f"Start {label}")
            stop_fn()

    def _hemorrhage_rate_from_ui(self) -> float:
        return float(self.cb_hemo_severity.currentData())

    def toggle_hemorrhage(self, checked):
        if checked:
            rate = self._hemorrhage_rate_from_ui()
            self.b_hem.setText("Stop bleeding")
            self.engine.start_hemorrhage(rate)
            self.cb_hemo_severity.setEnabled(False)
        else:
            self.b_hem.setText("Start bleeding")
            self.engine.stop_hemorrhage()
            self.cb_hemo_severity.setEnabled(True)

    def toggle_anaphylaxis(self, checked):
        self._toggle_simple_event(
            self.b_anaph,
            checked,
            self.engine.start_anaphylaxis,
            self.engine.stop_anaphylaxis,
            "anaphylaxis",
        )

    def toggle_sepsis(self, checked):
        self._toggle_simple_event(
            self.b_sepsis,
            checked,
            self.engine.start_sepsis,
            self.engine.stop_sepsis,
            "sepsis",
        )

    def update_airway_obstruction(self, value):
        """Manual upper airway obstruction (0-100%)."""
        self.engine.set_airway_obstruction(value / 100.0)

    def update_bronchospasm(self, value):
        """Manual bronchospasm severity (0-100%)."""
        self.engine.set_bronchospasm(value / 100.0)

    def update_auto_laryngospasm(self, checked):
        """Toggle auto-triggered laryngospasm."""
        self.btn_auto_laryngo.setText(
            "Automatic laryngospasm on" if checked else "Automatic laryngospasm off"
        )
        self.engine.set_auto_laryngospasm(checked)

    def stop_all_events(self):
        for button in (self.b_hem, self.b_anaph, self.b_sepsis):
            button.setChecked(False)
        self.b_disturb.setChecked(False)
        self.engine.stop_events()

    def change_bair_hugger(self, index):
        """Handle Bair Hugger setting change."""
        self.engine.set_bair_hugger(self.combo_bair.itemData(index))
