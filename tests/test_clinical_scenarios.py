"""One integrated adult path per clinical workflow; comments cite each bound's source."""

import copy
import itertools
from dataclasses import replace

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.disturbances import PROFILE_SPECS
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.lactate import LactateModel
from anasim.physiology.respiration import RespiratoryModel
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

        # Propofol prescribing information: 2-2.5 mg/kg for induction in healthy adults.
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
            spontaneous_effort = []

            for _ in range(301):
                bis_values.append(engine.state.bis)
                map_values.append(engine.state.map)
                spontaneous_effort.append(engine.resp.state.effort_rr)
                engine.step(1.0)

            # Ventilation to PaCO2 about 41 mmHg is below the apneic threshold,
            # 4-5 mmHg under the anesthetized resting PaCO2 (Hickey 1971).
            assert max(spontaneous_effort) == 0.0

            # NICE gives BIS 40-60 as the target range during general anesthesia.
            assert min(bis_values) >= 40.0
            assert max(bis_values) <= 60.0
            # Compare whole 5 s ventilator cycles, preserving respiratory MAP variation.
            cycle_maps = [sum(map_values[i:i + 5]) / 5.0 for i in range(0, 300, 5)]
            assert max(cycle_maps) - min(cycle_maps) < 5.0
            assert engine.get_drug_state("nore")["rate"] == 0.0
            # Unstimulated maintenance settles at least 20% below baseline, as
            # before incision (VitalDB TIVA median 27%; Ebert 1995: 0.83 MAC
            # sevoflurane lowered MAP 29%); the learner treats it.
            assert 50.0 <= min(map_values) <= max(map_values) < 0.8 * engine.patient.baseline_map
            engine.set_drug_rate("nore", 5.0)
            for _ in range(100):
                engine.step(2.0)
            assert engine.state.map >= max(65.0, max(map_values))
            assert engine.get_drug_state("nore")["rate"] == pytest.approx(5.0)

    def test_tiva_emergence_recovers_ventilation_and_wakefulness(
        self, anesthetized_engine
    ):
        """TIVA washout should recover breathing and wakefulness within 20 minutes."""
        engine = anesthetized_engine
        assert 40.0 <= engine.state.bis <= 60.0
        shunt = engine.aeration.shunt_fraction

        _stop_tiva(engine)
        ventilation_time = None
        bis_70_time = None
        bis_80_time = None

        for step_idx in range(601):
            elapsed = step_idx * 2
            # Withdraw controlled breaths once hypnosis clears and effort
            # returns, then assess unassisted ventilation through the ETT.
            if engine.state.bis > 70.0 and engine.resp.state.mv > 1.0:
                engine.set_vent_power(False)
            if engine.resp.state.mv > 3.0 and ventilation_time is None:
                ventilation_time = elapsed
            if engine.state.bis > 70.0 and bis_70_time is None:
                bis_70_time = elapsed
            if engine.state.bis > 80.0 and bis_80_time is None:
                bis_80_time = elapsed
            if ventilation_time is not None and bis_70_time is not None and bis_80_time is not None:
                break
            engine.step(2.0)

        # Published propofol-remifentanil recovery studies report spontaneous
        # respiration and eye opening in roughly 4-15 minutes, depending on dose.
        assert ventilation_time is not None and 180 <= ventilation_time <= 900
        assert bis_70_time is not None and 180 <= bis_70_time <= 900
        assert bis_80_time is not None and bis_80_time <= 1200
        # Atelectasis from induction persists after emergence (Benoît 2002).
        assert engine.aeration.shunt_fraction >= shunt

    def test_class_iii_hemorrhage_and_blood_rescue(
        self, anesthetized_engine, advance_time
    ):
        """A 30-40% loss should cause shock; hemostasis and blood should restore pressure."""
        engine = anesthetized_engine
        initial_volume = engine.hemo.blood_volume
        initial_v1 = engine.pk["propofol"].v1
        initial_cp = engine.state.propofol_cp
        baseline_hr = engine.state.hr

        engine.start_hemorrhage(500.0)
        advance_time(engine, 180.0, dt=2.0)
        engine.stop_hemorrhage()

        loss_fraction = (initial_volume - engine.hemo.blood_volume) / initial_volume
        shock_map = engine.state.map
        assert 0.30 <= loss_fraction <= 0.40
        # Propofol and remifentanil blunt the tachycardia.
        assert engine.state.hr > baseline_hr + 5.0
        assert engine.state.sbp < 90.0
        assert shock_map >= 20.0
        # Propofol's central volume exceeds blood volume, so it loses the shed volume.
        assert initial_v1 - engine.pk["propofol"].v1 == pytest.approx(
            (initial_volume - engine.hemo.blood_volume) / 1000.0, rel=0.05
        )
        # Lower clearance raises exposure (Johnson 2003); shed blood also removes drug.
        assert engine.state.propofol_cp > initial_cp

        engine.give_blood(600.0)
        engine.give_fluid(500.0)
        advance_time(engine, 600.0, dt=2.0)

        # The European major-bleeding guideline uses SBP 80-90 mmHg as the
        # restricted-resuscitation target until bleeding is controlled.
        assert engine.state.map >= 60.0
        assert engine.state.map > shock_map + 30.0
        # The same guideline adds norepinephrine if volume replacement leaves
        # SBP below target.
        if engine.state.sbp < 80.0:
            engine.set_drug_rate("nore", 8.0)
            advance_time(engine, 300.0, dt=2.0)
        assert engine.state.sbp >= 80.0

    def test_stepwise_blood_loss_matches_awake_and_anesthetized_responses(self, engine_factory, advance_time):
        """Johnson 2014: awake men bled 1 L (17%) in 333 mL steps at about 41
        mL/min kept MAP within 2%, with HR +14%, SV -20%, and TPR +8%. Under
        propofol-remifentanil, 10% steps every 30 min raised HR 26% at 30% loss
        (Kurita 2011, swine); propofol depresses the cardiac and sympathetic
        baroreflex limbs about 70% (Ebert 1992; Sellgren 1994).
        """
        engine = engine_factory(start=True, age=32, height=185, weight=91, baseline_hr=59, baseline_map=91)
        engine.set_continuous_fluid_rate(0.0)
        advance_time(engine, 60.0, dt=2.0)
        before = engine.state.map, engine.state.hr, engine.state.sv, engine.state.svr
        for _ in range(3):
            engine.start_hemorrhage(41.0)
            advance_time(engine, 333.0 / 41.0 * 60.0, dt=2.0)
            engine.stop_hemorrhage()
            advance_time(engine, 300.0, dt=2.0)
        map_change, hr_change, sv_change, svr_change = (
            now / then - 1.0 for now, then in zip(
                (engine.state.map, engine.state.hr, engine.state.sv, engine.state.svr), before, strict=True,
            )
        )
        assert abs(map_change) < 0.05
        assert 0.08 < hr_change < 0.22
        assert -0.26 < sv_change < -0.14
        assert 0.03 < svr_change < 0.13

        engine = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
        engine.set_continuous_fluid_rate(0.0)
        advance_time(engine, 30.0, dt=2.0)
        baseline_hr, baseline_map = engine.state.hr, engine.state.map
        step_ml = 0.1 * engine.hemo.blood_volume
        for _ in range(3):
            engine.start_hemorrhage(step_ml / 4.0)
            advance_time(engine, 240.0, dt=2.0)
            engine.stop_hemorrhage()
            advance_time(engine, 1560.0, dt=2.0)
        assert 1.15 * baseline_hr < engine.state.hr < 1.35 * baseline_hr
        assert 0.65 * baseline_map < engine.state.map < 0.9 * baseline_map

    def test_age_blunts_reflex_tachycardia(self, engine_factory, advance_time):
        """Older adults have a smaller compensatory HR rise (Laitinen 2004)."""
        rises = {}
        for age in (30, 68):
            engine = engine_factory(start=True, age=age)
            engine.set_continuous_fluid_rate(0.0)
            advance_time(engine, 30.0, dt=2.0)
            baseline_hr, initial_volume = engine.state.hr, engine.hemo.blood_volume
            engine.start_hemorrhage(100.0)
            while engine.hemo.blood_volume > 0.8 * initial_volume:
                engine.step(2.0)
            engine.stop_hemorrhage()
            advance_time(engine, 300.0, dt=2.0)
            rises[age] = engine.state.hr - baseline_hr

        assert 0.6 < rises[68] / rises[30] < 0.9

    def test_awake_hemorrhage_follows_atls_classes(self, awake_engine):
        """ATLS: SBP stays normal in class II (15-30% loss) and falls with HR > 100 in class IV (> 40%)."""
        engine = awake_engine
        baseline_map = engine.state.map
        initial_volume = engine.hemo.blood_volume
        engine.start_hemorrhage(100.0)

        def bleed_to(loss_fraction):
            while engine.hemo.blood_volume > (1.0 - loss_fraction) * initial_volume:
                engine.step(2.0)
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
        baseline_milrinone_clearance = engine.pk["milri"].cl1
        baseline_fentanyl_clearance = engine.pk["fentanyl"].cl1

        engine.start_sepsis()
        advance_time(engine, 600.0, dt=2.0)

        assert engine.state.hr > 90.0
        assert engine.state.map < 65.0
        assert engine.state.svr < baseline_svr * 0.75
        # Hypotension slows renal milrinone clearance while flow-limited
        # hepatic fentanyl clearance follows the raised cardiac output.
        assert engine.pk["milri"].cl1 < 0.9 * baseline_milrinone_clearance
        assert engine.pk["fentanyl"].cl1 > baseline_fentanyl_clearance

        # Surviving Sepsis Campaign: 30 mL/kg crystalloid, norepinephrine first,
        # and an initial MAP target of 65 mmHg.
        engine.give_fluid(30.0 * engine.patient.weight)
        engine.set_drug_rate("nore", 15.0)
        advance_time(engine, 900.0, dt=2.0)

        assert engine.state.map >= 65.0

    def test_anaphylaxis_plasma_escape_hemoconcentration_and_colloid_recovery(
        self, engine_factory, advance_time
    ):
        """Fisher 1986 (n = 22, Table 3): plasma escape during anaphylaxis causes
        up to ~35% blood-volume loss within 10 min without red-cell loss,
        raising Hb sharply until source control and volume replacement restore it.
        """
        untreated = engine_factory(
            config=SimulationConfig(
                mode="steady_state", maint_type="tiva", rng_seed=1, maintenance_fluid_ml_hr=0.0
            ),
            start=True,
            age=45,
            weight=70,
            height=170,
            baseline_hb=14.0,
        )
        treated = copy.deepcopy(untreated)
        advance_time(untreated, 30.0, dt=2.0)
        model = untreated.hemo
        bv0 = model.blood_volume
        rbc0 = model.red_cell_volume_ml
        uop0 = model.total_urine_out_ml
        untreated.start_anaphylaxis()
        advance_time(untreated, 600.0, dt=2.0)

        bv_loss_frac = 1.0 - model.blood_volume / bv0
        assert 0.18 <= bv_loss_frac <= 0.35
        assert model.red_cell_volume_ml == pytest.approx(rbc0, rel=1e-6)
        assert untreated.state.hb_g_dl > 17.0
        assert untreated.state.map < 55.0
        assert (
            model.blood_volume + model.total_third_space_ml + model.lung_water_ml
            + model.crystalloid.interstitial_ml + (model.total_urine_out_ml - uop0)
        ) == pytest.approx(bv0, abs=1e-3)

        treated.fluid_infusion_rate_ml_min = 250.0
        treated.start_anaphylaxis()
        advance_time(treated, 180.0, dt=2.0)
        hb_hemoconcentrated = treated.state.hb_g_dl
        assert hb_hemoconcentrated > 14.5

        treated.give_drug_bolus("epi", 100.0)
        treated.set_drug_rate("epi", 8.0)
        treated.give_fluid(1000.0)
        advance_time(treated, 300.0, dt=2.0)
        treated.stop_anaphylaxis()
        treated.give_albumin(1500.0)
        advance_time(treated, 900.0, dt=2.0)

        assert treated.state.map >= 65.0
        assert 13.5 <= treated.state.hb_g_dl < hb_hemoconcentrated - 0.3

    @pytest.mark.parametrize("heart_failure", ["none", "hfref"])
    def test_septic_shock_norepinephrine_preload_output_and_weaning(
        self, engine_factory, advance_time, heart_failure
    ):
        """Hamzaoui 2010 (n = 105), Carelli 2026 (n = 30), and Persichini 2012
        (n = 16): norepinephrine restores septic MAP while raising MCFP, RAP,
        SV, and CI without adding tachycardia, and weaning reverses each change.
        """
        engine = engine_factory(
            config=SimulationConfig(
                mode="steady_state",
                maint_type="tiva",
                rng_seed=1,
                maintenance_fluid_ml_hr=300.0,
            ),
            start=True,
            age=65,
            weight=75,
            height=172,
            sex="male",
            heart_failure=heart_failure,
        )
        engine.set_vent_settings(mode="VCV", vt=0.45, rr=20, peep=6, ie="1:2")
        engine.start_sepsis()
        advance_time(engine, 900.0, dt=1.0)

        bsa = engine.patient.bsa
        pre_map = engine.state.map
        pre_hr = engine.state.hr
        pre_ci = engine.state.co / bsa
        pre_sv = engine.state.sv
        pre_mcfp = engine.hemo.state.mcfp
        pre_rap = engine.hemo.state.rap
        assert pre_map < 62.0

        engine.set_drug_rate("nore", 0.24 * 75.0)
        advance_time(engine, 600.0, dt=1.0)
        mid_map = engine.state.map
        mid_hr = engine.state.hr
        mid_ci = engine.state.co / bsa
        mid_sv = engine.state.sv
        mid_mcfp = engine.hemo.state.mcfp
        mid_rap = engine.hemo.state.rap

        assert mid_map >= 65.0
        assert mid_mcfp > pre_mcfp + 0.8
        assert mid_rap > pre_rap + 0.2
        assert mid_sv > pre_sv + 4.0
        assert mid_ci > pre_ci * 1.08
        assert mid_hr <= pre_hr + 2.0

        # Carelli 2026: doubling NE raises MAP, CVP, and SV even in reduced LVEF.
        engine.set_drug_rate("nore", 0.50 * 75.0)
        advance_time(engine, 600.0, dt=1.0)
        hi_map = engine.state.map
        hi_ci = engine.state.co / bsa
        hi_sv = engine.state.sv
        hi_mcfp = engine.hemo.state.mcfp
        hi_rap = engine.hemo.state.rap
        assert hi_map > mid_map + 8.0
        assert hi_mcfp > mid_mcfp + 0.3
        assert hi_rap > mid_rap + 0.1
        assert hi_sv > mid_sv
        assert hi_ci >= mid_ci

        # Persichini 2012: weaning NE reduces MAP, MCFP, RAP, and CI together.
        engine.set_drug_rate("nore", 0.19 * 75.0)
        advance_time(engine, 600.0, dt=1.0)
        assert engine.state.map < hi_map - 10.0
        assert engine.hemo.state.mcfp < hi_mcfp - 0.5
        assert engine.hemo.state.rap < hi_rap - 0.15
        assert engine.state.co / bsa < hi_ci

    def test_septic_shock_preserves_partial_catecholamine_chronotropy(
        self, engine_factory, advance_time
    ):
        """Hernández 2013 (n = 20): dobutamine 5 mcg/kg/min still increases HR and
        CI in septic shock despite beta-adrenoceptor hyporesponsiveness. Also verifies
        that epinephrine relieves sinus bradycardia during full sepsis.
        """
        dobu = engine_factory(
            config=SimulationConfig(
                mode="steady_state", maint_type="tiva", rng_seed=1, maintenance_fluid_ml_hr=300.0
            ),
            start=True,
            age=65,
            weight=75,
            height=172,
            sex="male",
        )
        dobu.start_sepsis()
        dobu.set_drug_rate("nore", 0.20 * 75.0)
        advance_time(dobu, 600.0, dt=2.0)
        bsa = dobu.patient.bsa
        pre_hr, pre_ci = dobu.state.hr, dobu.state.co / bsa

        dobu.set_drug_rate("dobu", 5.0)
        advance_time(dobu, 900.0, dt=2.0)
        assert dobu.state.hr > pre_hr + 4.0
        assert dobu.state.co / bsa > pre_ci * 1.12

        brady = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1),
            start=True,
        )
        brady.start_sepsis()
        advance_time(brady, 180.0, dt=2.0)
        brady.set_rhythm("SINUS_BRADY")
        advance_time(brady, 60.0, dt=2.0)
        assert brady.state.hr == pytest.approx(50.0, abs=1.5)

        brady.give_drug_bolus("epi", 50.0)
        brady.set_drug_rate("epi", 10.0)
        peak_hr = brady.state.hr
        for _ in range(120):
            brady.step(1.0)
            peak_hr = max(peak_hr, brady.state.hr)
        assert peak_hr > 85.0

    @pytest.mark.parametrize(("fluid", "urine_ml", "tolerance_ml"), [("crystalloid", 1000.0, 150.0), ("saline", 450.0, 100.0)])
    def test_awake_saline_is_excreted_more_slowly_than_ringers(self, fluid, urine_ml, tolerance_ml):
        """Reid 2003: 2 L over 1 h gave median 6 h urine of 1000 mL after Hartmann's
        and 450 mL after saline, which kept chloride above 105 mmol/L for over 6 h.
        """
        patient = Patient(age=25, weight=75, height=178)
        hemo = HemodynamicModel(patient)
        resp = RespiratoryModel(patient)
        resp.attach_fluid_balance(hemo.fluids, hemo.config.donor_mchc_g_dl)
        resp.baseline_co_l_min = hemo.base_co_l_min
        lactate = LactateModel(patient.weight, patient.hepatic_function)
        dt = 2.0
        for step_idx in range(round(6 * 3600 / dt)):
            if step_idx * dt < 3600:
                hemo.add_volume(2000.0 / 3600.0 * dt, label=fluid)
            resp_state = resp.step(
                dt, 0.0, 0.0, cardiac_output=hemo.state.co,
                hb_g_dl=hemo.hb_conc, blood_volume_ml=hemo.blood_volume,
            )
            lactate.update_fluid_volume(hemo.fluids.lactate_volume_l, hemo.fluids.consume_lactate_loss_l())
            base_removed = lactate.step(
                dt, resp.oxygen_deficit_l, resp.oxygen_supply_fraction,
                infused_lactate_mmol=hemo.fluids.infused_l_lactate_mmol,
            )
            resp.oxygen_deficit_l = 0.0
            if base_removed != 0.0:
                resp.remove_buffer_base(base_removed)
            hemo.step(dt, 0.0, 0.0, 0.0, -2.0, resp_state.pa_co2, resp_state.p_arterial_o2, ph=resp_state.ph)
            hemo.fluids.step(
                dt, ph=resp_state.ph, paco2=resp_state.pa_co2, resting_paco2=resp.paco2_setpoint,
                renal_function=1.0, renal_perfusion=hemo.renal_perfusion, hepatic_function=1.0,
                oxygen_supply=resp.oxygen_supply_fraction, epinephrine=0.0,
            )
        assert hemo.total_urine_out_ml == pytest.approx(urine_ml, abs=tolerance_ml)
        assert hemo.lung_water_ml_kg == 0.0
        if fluid == "saline":
            assert hemo.fluids.concentration("chloride") > 105.0

    def test_fluid_overload_floods_lungs_and_peep_reaerates(self, anesthetized_engine, advance_time):
        """Massive crystalloid raises LAP into the edema range; PEEP helps oxygenation."""
        engine = anesthetized_engine
        base_plat = engine.state.paw_plat
        # FiO2 about 0.55, where added shunt shows in P/F.
        engine.set_fgf(1.0, 1.0)

        # A healthy heart tolerates 3 L (43 mL/kg) without alveolar flooding.
        engine.give_fluid(3000.0)
        advance_time(engine, 1800.0, dt=2.0)
        assert engine.state.lung_water < 3.0  # Below the modeled flooding threshold.
        assert engine.aeration.flooded == 0.0
        well_filled_pf = engine.state.pao2 / engine.state.fio2

        engine.give_fluid(5000.0)
        advance_time(engine, 3600.0, dt=2.0)

        # Pulmonary edema: EVLWI above 10 mL/kg (normal about 7).
        assert engine.state.lap > 20.0
        assert engine.state.lung_water > 3.0
        edema_pao2 = engine.state.pao2
        assert edema_pao2 / engine.state.fio2 < 0.75 * well_filled_pf
        assert engine.state.paw_plat > base_plat + 0.5  # Flooded alveoli stiffen the lung.

        # Malo 1984: PEEP re-aerates flooded alveoli without removing lung water.
        engine.vent.update_settings(peep=13.0)
        advance_time(engine, 300.0, dt=2.0)
        assert engine.state.pao2 > 1.2 * edema_pao2
        assert engine.state.lung_water > 3.0

    @pytest.mark.parametrize("dt", [0.2, 1.0])
    def test_anaphylaxis_and_epinephrine_rescue(
        self, anesthetized_engine, advance_time, dt
    ):
        """ANZAAG/ANZCA 2022 guides epinephrine and rapid crystalloid dosing.

        Brown 2004, p. 151: SBP exceeded 90 mmHg within five minutes in all
        hypotensive sting reactions except one with cannula extravasation.
        """
        engine = anesthetized_engine
        baseline_sbp = engine.state.sbp
        baseline_gap = engine.state.paw_peak - engine.state.paw_plat
        baseline_mv = engine.state.mv

        engine.start_anaphylaxis()
        advance_time(engine, 120.0, dt)

        assert engine.state.sbp <= baseline_sbp * 0.70
        assert engine.state.map < 65.0
        assert engine.state.bronchospasm >= 0.9
        # Bronchospasm widens the resistive Ppeak - Pplat gap and cuts alveolar
        # ventilation, while volume-controlled breaths keep exhaled MV.
        gap = engine.state.paw_peak - engine.state.paw_plat
        assert gap > 2.0 * baseline_gap
        assert engine.state.mv == pytest.approx(baseline_mv, rel=0.05)
        obstructed_va = engine.state.va

        engine.fluid_infusion_rate_ml_min = 500.0
        engine.give_drug_bolus("epi", 100.0)
        engine.give_fluid(1000.0)
        epi_rate = 6.0
        engine.set_drug_rate("epi", epi_rate)
        peak_map = 0.0
        sbp_recovery_s = None
        for minute in range(1, 7):
            for sample in range(round(60 / dt)):
                engine.step(dt)
                peak_map = max(peak_map, engine.state.map)
                if sbp_recovery_s is None and engine.state.sbp > 90.0:
                    sbp_recovery_s = (minute - 1) * 60.0 + (sample + 1) * dt
            if engine.state.map < 65.0:
                engine.give_drug_bolus("epi", 50.0)
                epi_rate += 2.0
                engine.set_drug_rate("epi", epi_rate)
                if minute % 2 == 0:
                    engine.give_fluid(1000.0)
        advance_time(engine, 60.0, dt)

        assert sbp_recovery_s is not None and sbp_recovery_s <= 300.0
        assert peak_map < 130.0
        assert engine.state.map >= 65.0
        assert engine.state.bronchospasm < 0.5
        assert engine.state.paw_peak - engine.state.paw_plat < gap - 2.0
        assert engine.state.va > obstructed_va + 1.0

    def test_oxygen_analyzer_warns_before_desaturation_and_backup_recovers(
        self, anesthetized_engine
    ):
        """Inspired oxygen should warn first on low-flow carrier gas, then recover on backup O2."""
        engine = anesthetized_engine
        scenario = create_oxygen_supply_failure()
        scenario.prepare(engine)

        assert engine.state.fio2 == pytest.approx(engine.circuit.composition.fio2)
        assert 0.34 <= engine.circuit.composition.fio2 <= 0.42
        assert engine.state.et_o2 < 40.0
        assert engine.state.pao2 < 230.0
        assert engine.state.spo2 >= 94.0

        engine.set_oxygen_supply_connected(False)
        engine.step(0.5)
        assert engine.state.alarms.get("O2Supply", {}).get("low") is True
        warning_time = _first_time(
            engine,
            240,
            lambda e: e.circuit.composition.fio2 < 0.30,
        )

        # Association of Anaesthetists guidance requires continuous inspired
        # oxygen analysis with a low-concentration alarm during anesthesia.
        assert warning_time is not None
        assert engine.state.spo2 >= 94.0
        alarm_time = _first_time(engine, 300, lambda e: "FiO2" in e.state.alarms)
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
        assert "O2Supply" not in engine.state.alarms
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
        baseline_co2 = engine.state.pa_co2
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
        assert rescue_co2 > baseline_co2 + 5.0
        assert rescue_spo2 < baseline_spo2 - 10.0

        engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.55, p_max=20.0)
        for _ in range(120):
            engine.step(1.0)

        assert engine.state.mv > 5.0
        assert engine.state.pa_co2 < baseline_co2 + 0.5 * (rescue_co2 - baseline_co2)
        assert engine.state.pao2 > rescue_o2 + 5.0
        assert engine.state.sao2 > rescue_spo2 + 5.0


def test_ongoing_surgery_keeps_tiva_pressure_raised(engine_factory, advance_time):
    """VitalDB TIVA (validation/vitaldb/results/vitaldb_tiva.md): in untreated
    patients, MAP rose 11.0 mmHg (95% CI 9.4-12.8, n=182) from the 10 minutes
    before incision to surgery and stayed raised. Typical values: age 55,
    pre-induction MAP 101 and HR 74, propofol and remifentanil near 4.
    """
    base = engine_factory(
        config=SimulationConfig(mode="steady_state", tci_enabled=True, rng_seed=1), start=True,
        age=55, weight=62, height=162, sex="female", baseline_map=101, baseline_hr=74,
    )
    base.enable_tci("propofol", 4.0, mode="plasma")
    base.enable_tci("remi", 4.0, mode="plasma")
    advance_time(base, 900.0, dt=2.0)
    before = base.state.map
    changes: dict[bool, list[float]] = {}
    for surgery in (False, True):
        engine = copy.deepcopy(base) if not surgery else base
        if surgery:
            engine.start_disturbance("stim_sustained_surgery")
        changes[surgery] = []
        for _ in range(2):
            advance_time(engine, 900.0, dt=2.0)
            changes[surgery].append(engine.state.map - before)
    assert all(abs(change) < 2.0 for change in changes[False])
    assert all(7.0 < change < 16.5 for change in changes[True])


@pytest.mark.parametrize(("profile", "ce50", "seconds"), [
    ("stim_intubation_pulse", 5.0, 300.0),
    ("stim_sustained_surgery", 2.1, 300.0),
])
def test_remifentanil_blunts_the_pressor_response(engine_factory, monkeypatch, profile, ce50, seconds):
    """Albertin 2005: at propofol 3.4 mcg/mL (BIS 40-50), remifentanil blocked a
    15% HR or MAP rise in half of patients at 5.0 ng/mL for intubation and
    2.1 ng/mL for skin incision. Blocking movement takes far less (Bouillon 2004a).
    Surgery's unblocked tone is calibrated separately and removed here.
    """
    monkeypatch.setitem(PROFILE_SPECS, profile, replace(PROFILE_SPECS[profile], tonic_svr_fraction=0.0))

    def peak_map_rise(remi_target: float) -> float:
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", tci_enabled=True, rng_seed=5), start=True,
            age=53, weight=67, height=166, sex="female",
        )
        engine.enable_tci("propofol", 3.4)
        engine.enable_tci("remi", remi_target)
        for _ in range(450):
            engine.step(2.0)
        baseline = engine.state.map
        engine.set_auto_laryngospasm(False)
        engine.start_disturbance(profile)
        peak = 0.0
        for _ in range(round(seconds / 0.5)):
            engine.step(0.5)
            peak = max(peak, engine.state.map / baseline - 1.0)
        return peak

    propofol_only = peak_map_rise(0.0)
    at_ce50 = peak_map_rise(ce50)
    assert propofol_only > 0.15
    assert 0.3 * propofol_only < at_ce50 < 0.7 * propofol_only


@pytest.mark.parametrize(("sevo", "fentanyl", "low", "high"), [
    (0.71, 0.0, 25.0, None),  # Far below MAC-BAR: all patients responded
    (4.15, 0.0, 10.0, 20.0),  # MAC-BAR
    (1.78, 1.0, 10.0, 20.0),
    (0.71, 3.0, 10.0, 20.0),
    (6.26, 0.0, None, 15.0),  # MAC-BAR95
])
def test_sevoflurane_fentanyl_incision_response(engine_factory, sevo, fentanyl, low, high):
    """Katoh 1999: in patients aged 20-50, end-tidal sevoflurane blocked a
    >15% HR or MAP rise within 5 minutes of incision in half of patients at
    4.15%, 1.78% with fentanyl 1 ng/mL, and 0.71% with 3 ng/mL, and in 95%
    at 6.26% alone.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="balanced", tci_enabled=True, rng_seed=1),
        start=True, age=42, weight=57, height=160, sex="female",
    )
    engine.disable_tci("remi")
    if fentanyl:
        engine.enable_tci("fentanyl", fentanyl, mode="plasma")
    vaporizer = sevo
    for _ in range(40):
        engine.set_vaporizer(vaporizer)
        for _ in range(12):
            engine.step(5.0)
        vaporizer = min(8.0, max(0.0, vaporizer + 0.8 * (sevo - engine.state.et_sevo)))
    assert engine.state.et_sevo == pytest.approx(sevo, abs=0.05)
    hr, pressure = engine.state.hr, engine.state.map
    engine.start_disturbance("stim_sustained_surgery")
    rise = 0.0
    for _ in range(150):
        engine.step(2.0)
        rise = max(rise, 100.0 * (engine.state.hr / hr - 1.0), 100.0 * (engine.state.map / pressure - 1.0))
    assert low is None or rise > low
    assert high is None or rise < high


def test_atrial_fibrillation_then_sinus_bradycardia(engine_factory):
    """AF: beats stay irregular, but the HR and ART numerics average recent beats.

    An irregular R-R sequence alone lowers CO about 15% at the same rate (Clark 1997),
    on top of the lost atrial kick, so AF at about 110 bpm should not raise CO or MAP.
    In sinus bradycardia, SV varies little with atrial rate (Hogue 1996), so switching
    from AF to sinus bradycardia lowers CO; awake, reflex vasoconstriction limits the fall in MAP.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=7, arterial_line_enabled=True), start=True
    )
    for _ in range(60):
        engine.step(0.5)
    baseline_map, baseline_co, baseline_svr = engine.state.map, engine.hemo.state.co, engine.hemo.state.svr

    engine.set_rhythm("AFIB")
    for _ in range(60):
        engine.step(0.5)
    shown = []
    art = []
    for _ in range(600):
        engine.step(0.1)
        shown.append(engine.state.display_hr)
        art.append((engine.state.art_sbp, engine.state.art_map - engine.state.map))

    rate = engine.hemo.state.hr
    each_second = shown[::10]
    assert max(abs(value - rate) for value in shown) < 0.2 * rate
    assert max(abs(b - a) for a, b in itertools.pairwise(each_second)) < 15.0
    art_sbp, art_map_error = zip(*art, strict=True)
    assert max(art_sbp) - min(art_sbp) < 8.0
    assert max(abs(error) for error in art_map_error) < 5.0
    assert engine.hemo.state.co < baseline_co
    assert engine.state.map < baseline_map

    engine.set_rhythm("SINUS_BRADY")
    for _ in range(120):
        engine.step(0.5)
    assert engine.state.display_hr == pytest.approx(50.0, abs=1.0)
    assert engine.hemo.state.svr > 1.05 * baseline_svr
    assert 0.75 * baseline_map < engine.state.map < 0.9 * baseline_map
