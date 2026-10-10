"""One integrated adult path per clinical workflow; comments cite each bound's source."""

import pytest

from anasim.core.state import SimulationConfig
from anasim.scenarios.oxygen_supply import create_oxygen_supply_failure


def _stop_tiva(engine) -> None:
    engine.set_drug_rate("propofol", 0.0)
    engine.set_drug_rate("remi", 0.0)


def _first_time(engine, seconds, predicate):
    for elapsed in range(seconds + 1):
        if predicate(engine):
            return elapsed
        engine.step(1.0)
    return None


class TestClinicalAcceptance:
    def test_induction_reaches_hypnosis_and_neuromuscular_block(
        self, awake_engine, advance_time
    ):
        """Label doses should reach BIS 40-60 and intubating block on time."""
        engine = awake_engine
        engine.set_airway_mode("Mask")
        advance_time(engine, 5.0, dt=0.1)

        # DailyMed: 2-2.5 mg/kg propofol for induction in healthy adults.
        engine.give_drug_bolus("Propofol", 2.0 * engine.patient.weight)
        hypnosis_time = _first_time(engine, 120, lambda e: e.state.bis <= 60.0)

        assert hypnosis_time is not None
        assert engine.state.map >= 60.0

        # Label: 0.6 mg/kg reaches 80% block in a median 1.0 min (range 0.4-6).
        engine.give_drug_bolus("Rocuronium", 0.6 * engine.patient.weight)
        block_time = _first_time(engine, 90, lambda e: e.state.tof < 5.0)

        assert block_time is not None
        assert 24 <= block_time <= 90

    @pytest.mark.parametrize("tci_enabled", [False, True], ids=["manual", "tci"])
    def test_maintenance_requires_explicit_vasopressor_treatment(self, engine_factory, tci_enabled):
        """Maintenance holds depth without automatically treating hypotension."""
        for maint_type in ("tiva", "balanced"):
            engine = engine_factory(
                config=SimulationConfig(
                    mode="steady_state",
                    maint_type=maint_type,
                    tci_enabled=tci_enabled,
                    rng_seed=123,
                ),
                start=True,
            )
            # Hidden settling must not consume visible case time or totals.
            assert engine.state.time == 0.0
            assert engine.state.fluid_in_ml == engine.state.urine_out_ml == 0.0
            assert engine.state.temp_c == pytest.approx(37.0)
            assert engine.state.nibp_map == pytest.approx(engine.state.map, abs=1e-3)
            assert engine.get_drug_state("nore") == {"rate": 0.0, "target": 0.0, "is_tci": False}
            if maint_type == "balanced":
                assert engine.state.fi_sevo > 0.0
                assert 0.8 <= engine.state.mac_sevo <= 1.05
            bis_values = []
            map_values = []

            for _ in range(901):
                bis_values.append(engine.state.bis)
                map_values.append(engine.state.map)
                engine.step(1.0)

            # NICE gives BIS 40-60 as the target range during general anesthesia.
            assert min(bis_values) >= 40.0
            assert max(bis_values) <= 60.0
            # Compare whole 5 s ventilator cycles, preserving respiratory MAP variation.
            cycle_maps = [sum(map_values[i:i + 5]) / 5.0 for i in range(0, 900, 5)]
            assert max(cycle_maps) - min(cycle_maps) < 5.0
            assert engine.get_drug_state("nore")["rate"] == 0.0
            if maint_type == "tiva":
                # Su predicts hypotension during unstimulated TIVA; the user
                # treats it.
                assert 50.0 <= min(map_values) <= max(map_values) < 65.0
                engine.set_drug_rate("nore", 5.0)
                for _ in range(300):
                    engine.step(1.0)
                assert engine.state.map >= 65.0
                assert engine.get_drug_state("nore")["rate"] == pytest.approx(5.0)
            else:
                assert min(map_values) >= 65.0

    def test_tiva_emergence_recovers_ventilation_and_wakefulness(
        self, anesthetized_engine
    ):
        """TIVA washout should recover breathing and wakefulness within 20 minutes."""
        engine = anesthetized_engine
        assert 40.0 <= engine.state.bis <= 60.0

        _stop_tiva(engine)
        ventilation_time = None
        bis_70_time = None
        bis_80_time = None

        for elapsed in range(1201):
            if engine.resp.state.mv > 3.0 and ventilation_time is None:
                ventilation_time = elapsed
            if engine.state.bis > 70.0 and bis_70_time is None:
                bis_70_time = elapsed
            if engine.state.bis > 80.0 and bis_80_time is None:
                bis_80_time = elapsed
            engine.step(1.0)

        # Published propofol-remifentanil recovery studies report spontaneous
        # respiration and eye opening in roughly 4-15 minutes, depending on dose.
        # Ventilation continues, holding PaCO2 below the patient's own level.
        assert ventilation_time is not None and 180 <= ventilation_time <= 900
        assert bis_70_time is not None and 180 <= bis_70_time <= 900
        assert bis_80_time is not None and bis_80_time <= 1200

    def test_class_iii_hemorrhage_and_blood_rescue(
        self, anesthetized_engine, advance_time
    ):
        """A 30-40% loss should cause shock; hemostasis and blood should restore pressure."""
        engine = anesthetized_engine
        initial_volume = engine.hemo.blood_volume
        initial_v1 = engine.pk["propofol"].v1
        baseline_hr = engine.state.hr

        engine.start_hemorrhage(500.0)
        advance_time(engine, 180.0)
        engine.stop_hemorrhage()

        loss_fraction = (initial_volume - engine.hemo.blood_volume) / initial_volume
        shock_map = engine.state.map
        assert 0.30 <= loss_fraction <= 0.40
        # Propofol and remifentanil blunt the tachycardia.
        assert engine.state.hr > baseline_hr + 5.0
        assert engine.state.sbp < 90.0
        assert shock_map >= 20.0
        assert engine.pk["propofol"].v1 / initial_v1 == pytest.approx(
            engine.hemo.blood_volume / initial_volume, rel=0.05
        )

        engine.give_blood(600.0)
        engine.give_fluid(500.0)
        advance_time(engine, 600.0)

        # The European major-bleeding guideline uses SBP 80-90 mmHg as the
        # restricted-resuscitation target until bleeding is controlled.
        assert engine.state.map >= 60.0
        assert engine.state.map > shock_map + 30.0
        # The same guideline adds norepinephrine if volume replacement leaves
        # SBP below target.
        if engine.state.sbp < 80.0:
            engine.set_drug_rate("nore", 8.0)
            advance_time(engine, 300.0)
        assert engine.state.sbp >= 80.0

    def test_awake_hemorrhage_follows_atls_classes(self, awake_engine):
        """ATLS: SBP stays normal in class II (15-30% loss) and falls with HR > 100 in class IV (> 40%)."""
        engine = awake_engine
        baseline_map = engine.state.map
        initial_volume = engine.hemo.blood_volume
        engine.start_hemorrhage(100.0)

        def bleed_to(loss_fraction):
            while engine.hemo.blood_volume > (1.0 - loss_fraction) * initial_volume:
                engine.step(1.0)
            return engine.state.sbp, engine.state.map, engine.state.hr

        sbp, map_val, _ = bleed_to(0.20)
        assert sbp >= 100.0
        assert map_val >= 0.85 * baseline_map
        sbp, _, hr = bleed_to(0.45)
        assert sbp < 90.0
        assert hr > 100.0

    def test_septic_shock_and_guideline_resuscitation(
        self, anesthetized_engine, advance_time
    ):
        """Sepsis should produce warm shock that fluid and norepinephrine reverse."""
        engine = anesthetized_engine
        baseline_svr = engine.state.svr

        engine.start_sepsis()
        advance_time(engine, 600.0)

        assert engine.state.hr > 90.0
        assert engine.state.map < 65.0
        assert engine.state.svr < baseline_svr * 0.75

        # Surviving Sepsis Campaign: 30 mL/kg crystalloid, norepinephrine first,
        # and an initial MAP target of 65 mmHg.
        engine.give_fluid(30.0 * engine.patient.weight)
        engine.set_drug_rate("nore", 15.0)
        advance_time(engine, 900.0)

        assert engine.state.map >= 65.0

    def test_awake_crystalloid_load_is_partly_excreted(self, engine_factory, advance_time):
        """Reid 2003: 2 L over 1 h gave median 6 h urine of 450 mL (saline) to 1000 mL (Hartmann's)."""
        engine = engine_factory(start=True, age=25, weight=75, height=178)
        engine.set_continuous_fluid_rate(2000.0)
        advance_time(engine, 3600.0)
        engine.set_continuous_fluid_rate(0.0)
        advance_time(engine, 5 * 3600.0)

        assert 450.0 <= engine.state.urine_out_ml <= 1000.0
        assert engine.state.lung_water == 0.0

    def test_fluid_overload_floods_lungs_and_peep_reaerates(self, anesthetized_engine, advance_time):
        """Massive crystalloid raises LAP into the edema range; PEEP helps oxygenation."""
        engine = anesthetized_engine
        base_plat = engine.state.paw_plat

        # A healthy heart tolerates 3 L (43 mL/kg).
        engine.give_fluid(3000.0)
        advance_time(engine, 1800.0)
        assert engine.state.lung_water == 0.0
        well_filled_pf = engine.state.pao2 / engine.state.fio2

        engine.give_fluid(4000.0)
        advance_time(engine, 3600.0)

        # Pulmonary edema: EVLWI above 10 mL/kg (normal about 7).
        assert engine.state.lap > 20.0
        assert engine.state.lung_water > 3.0
        edema_pao2 = engine.state.pao2
        assert edema_pao2 / engine.state.fio2 < 0.75 * well_filled_pf
        assert engine.state.paw_plat > base_plat + 1.0

        # Malo 1984: PEEP re-aerates flooded alveoli without removing lung water.
        engine.vent.update_settings(peep=13.0)
        advance_time(engine, 300.0)
        assert engine.state.pao2 > 1.2 * edema_pao2
        assert engine.state.lung_water > 3.0

    def test_anaphylaxis_and_epinephrine_rescue(
        self, anesthetized_engine, advance_time
    ):
        """Anaphylaxis should combine airway and circulatory signs and respond to epinephrine."""
        engine = anesthetized_engine
        baseline_sbp = engine.state.sbp
        baseline_gap = engine.state.paw_peak - engine.state.paw_plat
        baseline_mv = engine.state.mv

        engine.start_anaphylaxis()
        advance_time(engine, 120.0)

        assert engine.state.sbp <= baseline_sbp * 0.70
        assert engine.state.map < 65.0
        assert engine.state.bronchospasm >= 0.9
        # Bronchospasm widens the resistive Ppeak - Pplat gap and cuts alveolar
        # ventilation, while volume-controlled breaths keep exhaled MV.
        gap = engine.state.paw_peak - engine.state.paw_plat
        assert gap > 2.0 * baseline_gap
        assert engine.state.mv == pytest.approx(baseline_mv, rel=0.05)
        obstructed_va = engine.state.va

        # ANZCA/ANZAAG: 50-100 mcg IV epinephrine boluses for severe
        # perioperative anaphylaxis and an initial 1000 mL crystalloid bolus.
        engine.give_drug_bolus("epi", 100.0)
        engine.give_fluid(1000.0)
        # Start near 0.1 mcg/kg/min; repeat boluses and titrate persistent hypotension.
        epi_rate = 6.0
        engine.set_drug_rate("epi", epi_rate)
        peak_map = 0.0
        for _ in range(6):
            for _ in range(60):
                engine.step(1.0)
                peak_map = max(peak_map, engine.state.map)
            if engine.state.map < 65.0:
                engine.give_drug_bolus("epi", 50.0)
                epi_rate += 2.0
                engine.set_drug_rate("epi", epi_rate)

        # Guideline doses restore pressure without a hypertensive crisis.
        assert peak_map < 130.0
        assert engine.state.map >= 65.0
        assert engine.state.bronchospasm < 0.5
        assert engine.state.paw_peak - engine.state.paw_plat < gap - 2.0
        assert engine.state.va > obstructed_va + 1.0

    def test_oxygen_analyzer_warns_before_desaturation_and_backup_recovers(
        self, anesthetized_engine
    ):
        """Inspired oxygen should warn first, then recover on high-flow backup oxygen."""
        engine = anesthetized_engine
        scenario = create_oxygen_supply_failure()
        scenario.prepare(engine)

        assert 0.34 <= engine.circuit.composition.fio2 <= 0.40
        assert engine.state.spo2 >= 94.0

        engine.set_oxygen_supply_connected(False)
        warning_time = _first_time(
            engine,
            60,
            lambda e: e.circuit.composition.fio2 < 0.30,
        )

        # Association of Anaesthetists guidance requires continuous inspired
        # oxygen analysis with a low-concentration alarm during anesthesia.
        assert warning_time is not None
        assert engine.state.spo2 >= 94.0
        alarm_time = _first_time(engine, 180, lambda e: "FiO2" in e.state.alarms)
        assert alarm_time is not None
        assert engine.state.spo2 >= 94.0

        engine.set_oxygen_supply_connected(True)
        engine.set_fgf(10.0, 0.0, 0.0)
        recovery_time = _first_time(
            engine,
            120,
            lambda e: e.circuit.composition.fio2 >= 0.85,
        )

        assert recovery_time is not None
        assert engine.state.spo2 >= 94.0

    def test_propofol_remifentanil_respiratory_depression_and_bag_mask_rescue(
        self, awake_engine
    ):
        """Bag-mask ventilation should reverse opioid-hypnotic gas-exchange failure."""
        engine = awake_engine
        engine.set_airway_mode("Mask")
        engine.set_fgf(0.0, 6.0)
        for _ in range(30):
            engine.step(1.0)

        baseline_mv = engine.state.mv
        baseline_spo2 = engine.state.sao2
        engine.give_drug_bolus("Propofol", 2.0 * engine.patient.weight)
        engine.give_drug_bolus("Remifentanil", 100.0)

        min_mv = baseline_mv
        for _ in range(120):
            engine.step(1.0)
            min_mv = min(min_mv, engine.state.mv)

        rescue_co2 = engine.state.pa_co2
        rescue_o2 = engine.state.pao2
        rescue_spo2 = engine.state.sao2
        assert min_mv < 1.0
        assert rescue_co2 > 50.0
        assert rescue_spo2 < baseline_spo2 - 10.0

        engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.55)
        for _ in range(120):
            engine.step(1.0)

        assert engine.state.mv > 5.0
        assert engine.state.pa_co2 < rescue_co2 - 5.0
        assert engine.state.pao2 > rescue_o2 + 5.0
        assert engine.state.sao2 > rescue_spo2 + 5.0


def test_remifentanil_blunts_laryngoscopy_response(engine_factory):
    """Opioid blunts the pressor response to laryngoscopy (Bouillon 2004 TOL surface)."""

    def peak_map_rise(remi_target: float) -> float:
        engine = engine_factory(config=SimulationConfig(mode="steady_state", tci_enabled=True, rng_seed=5), start=True)
        engine.enable_tci("remi", remi_target)
        for _ in range(600):
            engine.step(1.0)
        baseline = engine.state.map
        engine.set_auto_laryngospasm(False)
        engine.start_disturbance("stim_intubation_pulse")
        peak = 0.0
        for _ in range(600):
            engine.step(0.1)
            peak = max(peak, engine.state.map - baseline)
        return peak

    propofol_only = peak_map_rise(0.0)
    with_remifentanil = peak_map_rise(4.0)
    assert propofol_only > 15.0
    assert with_remifentanil < 0.25 * propofol_only


def test_atrial_fibrillation_then_sinus_bradycardia(engine_factory):
    """AF: beats stay irregular, but the HR and ART numerics average recent beats.

    An irregular R-R sequence alone lowers CO about 15% at the same rate (Clark 1997),
    on top of the lost atrial kick, so AF at about 110 bpm should not raise CO or MAP.
    When AF ends in sinus bradycardia, SV stays about the same (Hogue 1996), so CO
    falls; awake, reflex vasoconstriction limits the fall in MAP.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=7, arterial_line_enabled=True), start=True
    )
    for _ in range(300):
        engine.step(0.1)
    baseline_map, baseline_co, baseline_svr = engine.state.map, engine.hemo.state.co, engine.hemo.state.svr

    engine.set_rhythm("AFIB")
    for _ in range(300):
        engine.step(0.1)
    shown = []
    art = []
    for _ in range(600):
        engine.step(0.1)
        shown.append(engine.state.display_hr)
        art.append((engine.state.art_sbp, engine.state.art_map - engine.state.map))

    rate = engine.hemo.state.hr
    each_second = shown[::10]
    assert max(abs(value - rate) for value in shown) < 0.2 * rate
    assert max(abs(b - a) for a, b in zip(each_second, each_second[1:])) < 15.0
    art_sbp, art_map_error = zip(*art)
    assert max(art_sbp) - min(art_sbp) < 8.0
    assert max(abs(error) for error in art_map_error) < 5.0
    assert engine.hemo.state.co < baseline_co
    assert engine.state.map < baseline_map

    engine.set_rhythm("SINUS_BRADY")
    for _ in range(600):
        engine.step(0.1)
    assert engine.state.display_hr == pytest.approx(50.0, abs=1.0)
    assert engine.hemo.state.svr > 1.05 * baseline_svr
    assert 0.75 * baseline_map < engine.state.map < 0.9 * baseline_map
