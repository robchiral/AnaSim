"""Reference-backed, integrated clinical acceptance tests.

These tests cross subsystem boundaries. Unit tests own equations and component
details; this file owns one representative adult path for each clinical
workflow. Comments identify the source used for each acceptance bound.
"""

from anasim.core.state import SimulationConfig
from anasim.ui.scenarios.oxygen_supply import create_oxygen_supply_failure


def _stop_tiva(engine) -> None:
    engine.disable_tci("propofol")
    engine.disable_tci("remi")
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
        engine.give_drug_bolus("Propofol", 2.5 * engine.patient.weight)
        hypnosis_time = _first_time(engine, 120, lambda e: e.state.bis <= 60.0)

        assert hypnosis_time is not None
        assert engine.state.map >= 60.0

        # FDA label: 0.6 mg/kg rocuronium, with intubation at 60-90 seconds.
        engine.give_drug_bolus("Rocuronium", 0.6 * engine.patient.weight)
        block_time = _first_time(engine, 90, lambda e: e.state.tof < 5.0)

        assert block_time is not None
        assert 60 <= block_time <= 90

    def test_maintenance_stays_within_depth_and_pressure_ranges(self, engine_factory):
        """TIVA and balanced maintenance should remain stable for 15 minutes."""
        for maint_type in ("tiva", "balanced"):
            engine = engine_factory(
                config=SimulationConfig(
                    mode="steady_state",
                    maint_type=maint_type,
                    rng_seed=123,
                ),
                start=True,
            )
            bis_values = []
            map_values = []

            for _ in range(901):
                bis_values.append(engine.state.bis)
                map_values.append(engine.state.map)
                engine.step(1.0)

            # NICE gives BIS 40-60 as the target range during general anesthesia.
            assert min(bis_values) >= 40.0
            assert max(bis_values) <= 60.0
            # POQI recommends maintaining intraoperative MAP at or above 60 mmHg.
            assert min(map_values) >= 60.0

            if engine.state.nore_ce > 0.5:
                assert engine.tci_nore is not None
                assert engine.get_drug_state("nore")["is_tci"]

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
        assert ventilation_time is not None and 180 <= ventilation_time <= 720
        assert bis_70_time is not None and 180 <= bis_70_time <= 900
        assert bis_80_time is not None and bis_80_time <= 1200

    def test_class_iii_hemorrhage_and_blood_rescue(
        self, anesthetized_engine, advance_time
    ):
        """A 30-40% loss should cause shock; hemostasis and blood should restore pressure."""
        engine = anesthetized_engine
        initial_volume = engine.hemo.blood_volume

        engine.start_hemorrhage(500.0)
        advance_time(engine, 180.0)
        engine.stop_hemorrhage()

        loss_fraction = (initial_volume - engine.hemo.blood_volume) / initial_volume
        shock_map = engine.state.map
        assert 0.30 <= loss_fraction <= 0.40
        assert engine.state.hr > 100.0
        assert engine.state.sbp < 90.0

        engine.give_blood(600.0)
        engine.give_fluid(500.0)
        advance_time(engine, 600.0)

        # The European major-bleeding guideline uses SBP 80-90 mmHg as the
        # restricted-resuscitation target until bleeding is controlled.
        assert engine.state.map >= 60.0
        assert engine.state.sbp >= 80.0
        assert engine.state.map > shock_map + 30.0

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
        engine.set_drug_target("nore", 3.0)
        advance_time(engine, 900.0)

        assert engine.state.map >= 65.0

    def test_anaphylaxis_and_epinephrine_rescue(
        self, anesthetized_engine, advance_time
    ):
        """Anaphylaxis should combine airway and circulatory signs and respond to epinephrine."""
        engine = anesthetized_engine
        baseline_sbp = engine.state.sbp

        engine.start_anaphylaxis()
        advance_time(engine, 120.0)

        assert engine.state.sbp <= baseline_sbp * 0.70
        assert engine.state.map < 65.0
        assert engine.state.bronchospasm >= 0.9

        # ANZCA/ANZAAG: 50-100 mcg IV epinephrine boluses for severe
        # perioperative anaphylaxis and an initial 1000 mL crystalloid bolus.
        engine.give_drug_bolus("epi", 100.0)
        engine.give_fluid(1000.0)
        engine.set_drug_target("epi", 3.0)
        advance_time(engine, 300.0)

        assert engine.state.map >= 65.0

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
