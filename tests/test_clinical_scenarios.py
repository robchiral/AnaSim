"""
Acceptance-level clinical scenario checks.

This file intentionally keeps only one representative path per major workflow:
induction, emergence, and hemorrhage/rescue.
"""


from anasim.core.state import AirwayType, SimulationConfig


def _stop_tiva(engine) -> None:
    engine.disable_tci("propofol")
    engine.disable_tci("remi")
    engine.set_drug_rate("propofol", 0.0)
    engine.set_drug_rate("remi", 0.0)


class TestAcceptanceScenarios:
    def test_induction_path(self, awake_engine, advance_time):
        """A standard IV induction should produce hypnosis, paralysis, and tolerable hemodynamics."""
        engine = awake_engine
        engine.set_airway_mode("Mask")
        advance_time(engine, 5, dt=0.1)

        baseline_bis = engine.state.bis
        baseline_map = engine.state.map
        engine.give_drug_bolus("Propofol", 2.0 * engine.patient.weight)

        loc_time = None
        min_bis = baseline_bis
        min_map = engine.state.map
        min_map_early = engine.state.map
        for t in range(180):
            engine.step(1.0)
            min_bis = min(min_bis, engine.state.bis)
            min_map = min(min_map, engine.state.map)
            if t < 120:
                min_map_early = min(min_map_early, engine.state.map)
            if engine.state.bis < 70.0 and loc_time is None:
                loc_time = t

        assert loc_time is not None and loc_time < 120
        assert baseline_bis - min_bis > 25.0
        assert baseline_map - min_map_early > 8.0
        assert min_map > 65.0

        engine.give_drug_bolus("Rocuronium", 0.6 * engine.patient.weight)
        paralysis_time = None
        for t in range(180):
            engine.step(1.0)
            if engine.state.tof < 5.0 and paralysis_time is None:
                paralysis_time = t

        assert paralysis_time is not None and paralysis_time < 150

    def test_emergence_path(self, anesthetized_engine):
        """Stopping a maintained TIVA should produce clinically plausible emergence milestones."""
        engine = anesthetized_engine
        assert engine.state.bis < 60.0
        assert engine.state.propofol_ce > 2.0

        _stop_tiva(engine)

        mv_3_time = None
        bis_70_time = None
        bis_80_time = None
        prev_bis = engine.state.bis
        max_bis_swing = 0.0

        for t in range(1800):
            engine.step(1.0)
            max_bis_swing = max(max_bis_swing, abs(engine.state.bis - prev_bis))
            prev_bis = engine.state.bis

            if engine.resp.state.mv > 3.0 and mv_3_time is None:
                mv_3_time = t
            if engine.state.bis > 70.0 and bis_70_time is None:
                bis_70_time = t
            if engine.state.bis > 80.0 and bis_80_time is None:
                bis_80_time = t

        assert mv_3_time is not None and mv_3_time < 720
        assert bis_70_time is not None and 390 <= bis_70_time <= 840
        assert bis_80_time is not None and bis_80_time < 1200
        assert max_bis_swing < 10.0

    def test_hemorrhage_and_rescue_path(self, anesthetized_engine, advance_time):
        """Major hemorrhage should cause shock physiology, and fluid rescue should partially recover MAP."""
        engine = anesthetized_engine
        baseline_hr = engine.state.hr
        baseline_map = engine.state.map

        engine.start_hemorrhage(500.0)
        advance_time(engine, 180)
        engine.stop_hemorrhage()

        bleed_hr = engine.state.hr
        bleed_map = engine.state.map
        assert bleed_hr - baseline_hr > 15.0
        assert baseline_map - bleed_map > 15.0

        engine.give_fluid(2000.0)
        advance_time(engine, 900)
        rescued_map = engine.state.map
        assert rescued_map > bleed_map + 5.0

        engine.state.airway_mode = AirwayType.ETT
        engine.set_drug_rate("nore", 14.0)
        advance_time(engine, 300)
        assert engine.state.nore_ce > 1.0

    def test_maintenance_startup_exposes_any_required_pressor(self, engine_factory):
        """Managed maintenance must avoid severe hypotension and expose support."""
        for maint_type in ("tiva", "balanced"):
            engine = engine_factory(
                config=SimulationConfig(mode="steady_state", maint_type=maint_type),
                start=True,
            )

            assert engine.state.map >= 65.0
            if engine.state.nore_ce > 0.5:
                assert engine.tci_nore is not None
                assert engine.get_drug_state("nore")["is_tci"]

            min_map = engine.state.map
            for _ in range(300):
                engine.step(1.0)
                min_map = min(min_map, engine.state.map)
            assert min_map >= 64.0

    def test_propofol_remifentanil_respiratory_depression_and_bag_mask_rescue(self, awake_engine):
        """Propofol plus remifentanil should depress ventilation; bag-mask ventilation should reverse gas trends."""
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
