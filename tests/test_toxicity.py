import copy
import math

import numpy as np
import pytest
from scipy.linalg import expm

from anasim.core import runtime as runtime_core
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.monitors.ecg import ECGMonitor, displayed_rhythm
from anasim.patient.pk_models import LidocainePK
from anasim.physiology.arrhythmia import antiarrhythmic_effects
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.toxicity import LocalAnestheticToxicity


def _step(tox, seconds=60, **changes):
    inputs = dict(neural_ce=0.0, cardiac_ce=0.0, ph=7.4, propofol_ce=0.0,
                  midazolam_ce=0.0, mac_sevo=0.0, motor_strength=1.0, cortical_function=1.0)
    inputs.update(changes)
    tox.step(seconds, **inputs)


def test_therapeutic_lidocaine_does_not_cause_toxicity_but_high_exposure_does():
    tox = LocalAnestheticToxicity(70)
    _step(tox, neural_ce=3, cardiac_ce=3)
    assert tox.seizure_activity == tox.conduction_block == tox.contractility_loss == tox.heart_rate_increase == 0
    _step(tox, neural_ce=8, cardiac_ce=8)
    assert tox.heart_rate_increase > 10 and tox.svr_increase > 1.2
    _step(tox, neural_ce=10, cardiac_ce=10)
    assert tox.seizure_activity > 0.95 and tox.unresponsiveness > 0.95
    assert tox.conduction_block == tox.contractility_loss == 0
    _step(tox, neural_ce=50, cardiac_ce=50)
    assert tox.neural_depression == tox.conduction_block == tox.contractility_loss == 1
    assert tox.respiratory_drive == 0 and tox.heart_rate_increase < 0.001


@pytest.mark.parametrize(
    "suppression",
    [{"midazolam_ce": 100}, {"propofol_ce": 4}, {"mac_sevo": 1.5}, {"cortical_function": 0}],
)
def test_anesthetics_and_cortical_failure_suppress_seizures_and_sympathetic_rise(suppression):
    awake = LocalAnestheticToxicity(70)
    _step(awake, neural_ce=8, cardiac_ce=8)
    blunted = LocalAnestheticToxicity(70)
    _step(blunted, neural_ce=8, cardiac_ce=8, **suppression)
    assert blunted.seizure_activity < awake.seizure_activity * 0.25
    assert blunted.heart_rate_increase < awake.heart_rate_increase * 0.25
    assert blunted.svr_increase < awake.svr_increase * 0.25


def test_lipid_needs_circulation_and_does_not_directly_switch_off_seizures():
    tox = LocalAnestheticToxicity(70)
    tox.add_lipid(100)
    for _ in range(150):
        tox.step_lipid(1, 0)
    assert tox.lipid_effect == 0 and tox.lipid_ml == 100
    for _ in range(150):
        tox.step_lipid(1, 1)
    assert 0.75 < tox.effect_input_fraction < 0.95
    _step(tox, neural_ce=10, cardiac_ce=20)
    assert tox.seizure_activity > 0.95
    assert tox.total_lipid_ml == 100


def test_lipid_effect_preserves_drug_mass_and_matches_the_pk_matrix(patient):
    model = LidocainePK(patient)
    model.state.c1 = 20
    model.effect_input_fraction = 0.8
    initial = model.state_vector()
    a, _ = model.get_ss_matrices()
    expected = expm(a * 0.1 / 60) @ initial
    model.step(0.1, 0)
    assert model.state_vector() == pytest.approx(expected, abs=0.002)
    untreated = LidocainePK(patient)
    untreated.state.c1 = 20
    untreated.step(0.1, 0)
    assert model.state.c1 == untreated.state.c1
    assert model.state.ce_cv < untreated.state.ce_cv


@pytest.mark.parametrize("sex", ["male", "female"])
def test_lidocaine_dose_exposure_scales_with_published_fat_free_mass(patient_factory, sex):
    """Foong 2025 Table IV: V1 follows FFM; excess fat does not dilute a bolus proportionally."""
    concentrations = []
    for weight in (70, 120):
        patient = patient_factory(weight=weight, sex=sex)
        model = LidocainePK(patient)
        denominator = 6680 + 216 * patient.bmi if sex == "male" else 8780 + 244 * patient.bmi
        ffm = 9270 * weight / denominator
        assert model.v1 == pytest.approx(25.2 * ffm / 56.1)
        # Return clearances to their measured anesthetized-output reference.
        model.update_hemodynamics(1, 0.82, 1)
        flow = (ffm / 56.1) ** 0.75
        assert model.cl1 == pytest.approx(45.9 / 60 * flow)
        assert model.cl2 == pytest.approx(142 / 60 * flow)
        assert model.cl3 == pytest.approx(5.81 / 60 * flow)
        model.state.c1 = 100 / model.v1
        concentrations.append(model.state.c1)
    assert concentrations[1] > concentrations[0] * 70 / 120


def test_toxic_lidocaine_does_not_keep_increasing_its_antiarrhythmic_benefit():
    therapeutic, _ = antiarrhythmic_effects(3, 0)
    toxic, _ = antiarrhythmic_effects(20, 0)
    assert therapeutic > 0.5 and toxic == 0


def test_sodium_block_widens_qrs_without_implying_ischemic_p_wave_loss():
    waveforms = []
    for block in (0, 0.8):
        clock, ecg = CardiacCycle(), ECGMonitor(np.random.default_rng(1))
        sample = clock.seed(70, RhythmType.SINUS)
        ecg.step(0.01, sample, sodium_block=block)
        qrs = []
        for _ in range(10):
            sample = clock.step(0.005, 70, RhythmType.SINUS)
            qrs.append(ecg.step(0.005, sample, sodium_block=block))
        waveforms.append(qrs)
    assert waveforms[1][-1] > waveforms[0][-1] + 0.1
    assert displayed_rhythm(RhythmType.SINUS, 0, 0.8) == "Wide-complex rhythm"
    assert displayed_rhythm(RhythmType.SINUS, 0, 0.995) == "Asystole"
    assert displayed_rhythm(RhythmType.SINUS, 0.8, 0.8) == "Idioventricular rhythm"
    assert displayed_rhythm(RhythmType.SINUS, 0.98, 0.8) == "Agonal rhythm"


def _ventilated(engine_factory):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=7), start=True)
    engine.set_airway_mode("ETT")
    engine.set_fgf(10, 0)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    engine.set_vent_power(True)
    return engine


def test_overdose_causes_seizures_despite_oxygenation_and_midazolam_treats_them(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.sao2 > 98 and engine.state.seizure_activity > 0.5
    assert engine.state.convulsions and engine.state.loc > 0.9
    engine.give_drug_bolus("midazolam", 5)
    for _ in range(90):
        engine.step(1)
    assert engine.state.seizure_activity < 0.2 and not engine.state.convulsions


def test_motor_block_hides_overdose_convulsions_without_treating_the_seizure(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.convulsions
    engine.give_drug_bolus("roc", 30)
    for _ in range(150):
        engine.step(1)
    assert engine.state.twitch < 10 and engine.state.seizure_activity > 0.5
    assert not engine.state.convulsions
    assert engine.toxicity.metabolic_factor > 1  # Cerebral activity persists.


def test_large_overdose_depresses_circulation_before_myocardial_energy_is_exhausted(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 2000)
    for _ in range(60):
        engine.step(1)
    assert engine.hemo.state.perfusion_map < 10 and engine.state.intrinsic_co < 0.3
    assert engine.state.dbp <= engine.state.map <= engine.state.sbp
    assert engine.hemo.coronary.energy_reserve > 0.5
    assert engine.toxicity.conduction_block > 0.5
    hypoxia, energy_failure, _ = runtime_core.myocardial_substrate(engine)
    assert hypoxia > 0.65 and energy_failure < 0.3


def test_cerebral_oxygen_failure_suppresses_toxic_convulsions_despite_retained_drug(engine_factory):
    engine = _ventilated(engine_factory)
    engine.give_drug_bolus("lidocaine", 500)
    for _ in range(60):
        engine.step(1)
    assert engine.state.convulsions
    engine.set_rhythm("VFIB")
    for _ in range(60):
        engine.step(1)
    assert engine.pk["lidocaine"].state.ce_neuro > 6
    assert engine.cerebral.cortical_function < 0.01
    assert engine.state.seizure_activity < 0.02 and not engine.state.convulsions


def test_lipid_rescue_gradually_reduces_cardiac_toxicity_during_supported_ventilation(engine_factory):
    outcomes = []
    for rescue in (False, True):
        engine = _ventilated(engine_factory)
        engine.give_drug_bolus("lidocaine", 1100)
        for i in range(240):
            if i == 30:
                engine.give_drug_bolus("midazolam", 5)
                if rescue:
                    engine.give_lipid_emulsion(100)
                    engine.set_lipid_infusion(17.5)
            engine.step(1)
            if rescue and i == 31:
                assert engine.toxicity.lipid_effect < 0.01
        outcomes.append((
            engine.hemo.state.perfusion_map,
            engine.state.map,
            engine.toxicity.conduction_block,
            engine.toxicity.contractility_loss,
        ))
    assert outcomes[1][0] > outcomes[0][0] + 10
    assert outcomes[1][1] > outcomes[0][1]
    assert 0.3 < outcomes[1][2] < outcomes[0][2]  # Rescue leaves substantial toxicity.
    assert outcomes[1][3] < outcomes[0][3] - 0.15


def test_lidocaine_post_distribution_kinetics_and_subconvulsive_infusions(patient_factory):
    """Tucker and Boas 1971 (3 mg/kg over 3 min [n=5] or 19 min [n=1]) and Rowland 1971 (110 mg bolus + 0.75 mg/min)."""
    patient = patient_factory(age=30, weight=69, height=175, sex="male")
    pk = LidocainePK(patient)
    # Foong's model relates HCl dose to base plasma concentration; Tucker's mean was 0.93 L/min.
    assert 0.85 < pk.cl1 < 1.05
    pk.step(60.0, 110.0 / 60.0)
    for minute in range(2, 241):
        pk.step(60.0, (0.75 / 60.0) if 31 <= minute <= 150 else 0.0)
        if minute == 150:
            plateau_hcl = pk.state.c1 / 0.8654  # Rowland reported concentrations as the HCl salt.
    assert 0.80 < plateau_hcl < 1.20 and pk.state.c1 / 0.8654 < 0.40

    for minutes, weight in ((3, 69), (19, 75)):
        pt = patient_factory(age=30, weight=weight, height=175, sex="male")
        lido = LidocainePK(pt)
        tox = LocalAnestheticToxicity(weight)
        duration = minutes * 60
        rate_mg_s = 3.0 * weight / duration
        c1_10m = c1_30m = peak_c1 = 0.0
        convulsed = False
        for second in range(1, max(duration + 60, 1800) + 1):
            lido.step(1.0, rate_mg_s if second <= duration else 0.0)
            _step(tox, seconds=1.0, neural_ce=lido.state.ce_neuro, cardiac_ce=lido.state.ce_cv)
            c1 = lido.state.c1
            peak_c1 = max(peak_c1, c1)
            convulsed = convulsed or (tox.convulsion_intensity > 0.2)
            if second == 600:
                c1_10m = c1
            elif second == 1800:
                c1_30m = c1
        assert not convulsed
        if minutes == 3:
            assert 2.4 < c1_10m < 3.6 and 1.1 < c1_30m < 1.9
        else:
            assert 2.8 < peak_c1 < 4.0


def test_awake_convulsive_infusion_raises_rate_and_pressure_then_recovers(engine_factory):
    """Foldes 1960 (0.5 mg/kg/min for 12.8 min) and Jorfeldt 1968: HR and pressure rise before convulsions."""
    engine = engine_factory(
        age=28, weight=70, height=175, sex="male",
        config=SimulationConfig(mode="awake", rng_seed=1), start=True,
    )
    base_hr, base_sbp = engine.state.hr, engine.state.sbp
    engine.set_drug_rate("lidocaine", 30.0)
    pre_hr = pre_sbp = peak_hr = 0.0
    convulsed = False
    for second in range(2, 768 + 1140 + 1, 2):
        if second == 770:
            engine.set_drug_rate("lidocaine", 0.0)
        engine.step(2.0)
        if not convulsed and not engine.state.convulsions:
            pre_hr = max(pre_hr, engine.state.hr)
            pre_sbp = max(pre_sbp, engine.state.sbp)
        peak_hr = max(peak_hr, engine.state.hr)
        convulsed = convulsed or engine.state.convulsions
    assert convulsed
    assert pre_hr > base_hr * 1.10 and pre_sbp > base_sbp * 1.10
    assert peak_hr > base_hr * 1.15
    assert not engine.state.convulsions and engine.state.loc < 0.05


def test_one_hour_overdose_in_hfref_stays_in_sinus_rhythm_and_recovers(engine_factory):
    """Armstrong 1988: 2 g over 1 h causes hypotension and coma in sinus rhythm.

    With linear kinetics and moderate HFrEF, the model clears lidocaine faster
    than in that case report (3.8 h total half-life at EF 15%).
    """
    engine = engine_factory(
        age=68, weight=74, height=172, sex="male", heart_failure="hfref",
        config=SimulationConfig(mode="awake", rng_seed=1), start=True,
    )
    engine.set_airway_mode("ETT")
    engine.set_fgf(2, 4)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.50, peep=5, ie="1:2")
    engine.set_vent_power(True)
    engine.set_drug_rate("lidocaine", 2000.0 / 74.0)
    map_60m = c1_75m = loc_75m = 0.0
    for minute in range(1, 241):
        if minute == 61:
            engine.set_drug_rate("lidocaine", 0.0)
        for _ in range(30):
            engine.step(2.0)
        rhythm = displayed_rhythm(
            engine.hemo.rhythm_type, engine.hemo.coronary.energy_failure, engine.toxicity.conduction_block
        )
        assert rhythm == "Sinus rhythm"
        assert not (minute > 75 and engine.state.convulsions)
        if minute == 60:
            map_60m = engine.state.map
        elif minute == 75:
            c1_75m = engine.pk["lidocaine"].state.c1
            loc_75m = engine.state.loc
            engine.give_drug_bolus("midazolam", 5)
        elif minute == 120:
            engine.give_drug_bolus("midazolam", 5)

    assert 45 < map_60m < 75
    assert 14.5 < c1_75m < 20.5 and loc_75m > 0.7
    # Recovery follows the model's faster washout, not Armstrong's 3.8 h total half-life.
    assert engine.pk["lidocaine"].state.c1 < 5.0
    assert engine.state.loc < 0.05 and engine.toxicity.respiratory_drive > 0.95


def test_volunteer_lipid_protocol_reduces_effective_lidocaine_auc(patient_factory):
    """Heinonen 2015: lipid reduces 30-min unentrapped lidocaine AUC by 23%
    (from larger tissue distribution plus ~6% plasma entrapment), modeled here
    as reduced effect-site input.

    Both arms receive 1.5 mL/kg over 1 min before 1 mg/kg lidocaine,
    then 0.25 mL/kg/min of the same study fluid for 30 min.
    """
    patient = patient_factory(age=25, weight=75, height=180, sex="male")
    bolus_ml = 1.5 * 75
    infusion_ml_min = 0.25 * 75
    aucs = []
    for use_lipid in (False, True):
        hemo = HemodynamicModel(patient)
        lido = LidocainePK(patient)
        tox = LocalAnestheticToxicity(75.0)
        base_bv = hemo.blood_volume_0
        base_co = hemo.base_co_l_min / hemo.failure.output_factor
        for _ in range(60):
            if use_lipid:
                hemo.add_volume(bolus_ml / 60.0, retention_fraction=0.25, label="lipid")
                tox.add_lipid(bolus_ml / 60.0)
            else:
                hemo.add_volume(bolus_ml / 60.0, label="crystalloid")
            state = hemo.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
            hemo.fluids.step(
                1.0, ph=7.4, paco2=40.0, resting_paco2=40.0, renal_function=1.0,
                renal_perfusion=hemo.renal_perfusion, hepatic_function=1.0,
                oxygen_supply=1.0, epinephrine=0.0,
            )
            lido.update_hemodynamics(
                hemo.blood_volume / base_bv, state.co / base_co, hemo.renal_perfusion,
                blood_volume_l=base_bv / 1000.0,
            )
            tox.step_lipid(1.0, lido.circulation)
            lido.effect_input_fraction = tox.effect_input_fraction
            lido.step(1.0, 0.0)
        if use_lipid:
            assert tox.total_lipid_ml == pytest.approx(bolus_ml)
        else:
            assert hemo.total_crystalloid_in_ml == pytest.approx(bolus_ml)

        lido.state.c1 += 75.0 / lido.v1
        auc = 0.0
        for _ in range(1800):
            if use_lipid:
                hemo.add_volume(infusion_ml_min / 60.0, retention_fraction=0.25, label="lipid")
                tox.add_lipid(infusion_ml_min / 60.0)
            else:
                hemo.add_volume(infusion_ml_min / 60.0, label="crystalloid")
            state = hemo.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
            hemo.fluids.step(
                1.0, ph=7.4, paco2=40.0, resting_paco2=40.0, renal_function=1.0,
                renal_perfusion=hemo.renal_perfusion, hepatic_function=1.0,
                oxygen_supply=1.0, epinephrine=0.0,
            )
            lido.update_hemodynamics(
                hemo.blood_volume / base_bv, state.co / base_co, hemo.renal_perfusion,
                blood_volume_l=base_bv / 1000.0,
            )
            tox.step_lipid(1.0, lido.circulation)
            lido.effect_input_fraction = tox.effect_input_fraction
            lido.step(1.0, 0.0)
            auc += lido.state.c1 * tox.effect_input_fraction / 60.0
        total_ml = tox.total_lipid_ml if use_lipid else hemo.total_crystalloid_in_ml
        assert total_ml == pytest.approx(bolus_ml + infusion_ml_min * 30)
        aucs.append(auc)
    reduction = 1.0 - aucs[1] / aucs[0]
    assert 0.19 < reduction < 0.26


def test_lipid_is_delivered_over_time_accounted_as_fluid_and_sheds_proportionally(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=7), start=True,
    )
    engine.set_airway_mode("ETT")
    engine.set_fgf(10, 0)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=5, ie="1:2")
    engine.set_vent_power(True)

    bolus_arm = copy.deepcopy(engine)
    infusion_arm = copy.deepcopy(engine)
    bolus_arm.give_lipid_emulsion(100.0)
    infusion_arm.set_lipid_infusion(40.0)
    for _ in range(150):
        bolus_arm.step(1.0)
        infusion_arm.step(1.0)
    infusion_arm.set_lipid_infusion(0.0)
    assert bolus_arm.hemo.blood_volume == pytest.approx(infusion_arm.hemo.blood_volume, abs=1e-6)
    assert bolus_arm.hemo.fluids.ecf_l == pytest.approx(infusion_arm.hemo.fluids.ecf_l, abs=1e-6)
    lipid_before = bolus_arm.toxicity.lipid_ml
    bv_before = bolus_arm.hemo.blood_volume
    bolus_arm.hemo.add_volume(-1000.0)
    bolus_arm.step(0.1)
    assert bolus_arm.toxicity.lipid_ml / lipid_before == pytest.approx(
        (bv_before - 1000.0) / bv_before, rel=1e-3
    )

    fluids = engine.hemo.fluids
    sodium, body_water = engine.state.sodium, fluids.ecf_l + fluids.icf_l
    engine.give_lipid_emulsion(100)
    engine.set_lipid_infusion(17.5)
    engine.step(1)
    assert 0 < engine.toxicity.total_lipid_ml < 2
    for _ in range(149):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == pytest.approx(143.75)
    assert engine.hemo.total_colloid_in_ml == pytest.approx(143.75)
    # Sodium-free lipid water spreads through total body water, not only extracellular fluid.
    assert sodium - engine.state.sodium == pytest.approx(sodium * 0.14375 / body_water, rel=0.1)
    assert engine.toxicity.lipid_effect > 0
    engine.set_lipid_infusion(0)
    given = engine.toxicity.total_lipid_ml
    for _ in range(10):
        engine.step(1)
    assert engine.toxicity.total_lipid_ml == given
    for invalid in (math.nan, math.inf, -1):
        with pytest.raises(ValueError):
            engine.set_lipid_infusion(invalid)


def test_last_seizures_generate_lactic_acidosis(engine_factory, advance_time):
    """Sustained LAST convulsions should produce substantial skeletal-muscle lactate."""
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    lac0 = engine.state.lactate
    engine.give_drug_bolus("lidocaine", 500.0)
    advance_time(engine, 300.0, dt=1.0)
    assert engine.state.lactate > lac0 + 1.0


def test_bupivacaine_ropivacaine_pk_epidural_absorption_and_cc_cns_margins(patient_factory):
    """Bupivacaine and ropivacaine arterial 3-cpt PK, biphasic epidural absorption, and comparative CC/CNS margins."""
    from anasim.patient.pk_models import BupivacainePK, RopivacainePK

    pt = patient_factory(age=50, weight=70, height=170, sex="male")
    bupi = BupivacainePK(pt)
    ropi = RopivacainePK(pt)
    assert bupi.v1 == pytest.approx(6.85, rel=0.05)
    assert ropi.v1 == pytest.approx(6.85, rel=0.05)
    assert bupi.v3 > ropi.v3 * 1.5  # Olofsen 2008: Vdss ~67 vs ~45 L

    # Epinephrine slows fast epidural systemic absorption.
    b_plain, b_epi = BupivacainePK(pt), BupivacainePK(pt)
    b_plain.add_epidural_dose(100.0, with_epinephrine=False)
    b_epi.add_epidural_dose(100.0, with_epinephrine=True)
    peak_plain = peak_epi = 0.0
    for _ in range(1800):
        b_plain.step(1.0, 0.0)
        b_epi.step(1.0, 0.0)
        peak_plain = max(peak_plain, b_plain.state.c1)
        peak_epi = max(peak_epi, b_epi.state.c1)
    assert 0.5 < peak_plain < 2.2
    assert peak_epi < peak_plain * 0.85

    # Comparative toxicity: at 4.5 mcg/mL, bupivacaine causes severe conduction block + seizures,
    # ropivacaine causes milder conduction block, and lidocaine causes neither.
    t_lido, t_ropi, t_bupi = LocalAnestheticToxicity(70), LocalAnestheticToxicity(70), LocalAnestheticToxicity(70)
    for _ in range(120):
        _step(t_lido, seconds=1.0, neural_ce=4.5, cardiac_ce=4.5)
        _step(t_ropi, seconds=1.0, ropivacaine_neuro_ce=4.5, ropivacaine_cv_ce=4.5)
        _step(t_bupi, seconds=1.0, bupivacaine_neuro_ce=4.5, bupivacaine_cv_ce=4.5)
    assert t_lido.seizure_activity == 0.0 and t_lido.conduction_block == 0.0
    assert t_bupi.seizure_activity > 0.7 and t_bupi.conduction_block > 0.25
    assert t_bupi.conduction_block > t_ropi.conduction_block + 0.20

    # Clarkson 1985 rate-dependent bupivacaine block: tachycardia worsens conduction block.
    slow, fast = LocalAnestheticToxicity(70), LocalAnestheticToxicity(70)
    for _ in range(60):
        _step(slow, seconds=1.0, bupivacaine_cv_ce=4.5, hr=55.0)
        _step(fast, seconds=1.0, bupivacaine_cv_ce=4.5, hr=145.0)
    assert fast.conduction_block > slow.conduction_block + 0.15

    # Pregnancy increases bupivacaine cardiotoxicity sensitivity (Morishima 1985; Neal 2018).
    nonpreg, preg = LocalAnestheticToxicity(70, pregnant=False), LocalAnestheticToxicity(70, pregnant=True)
    for _ in range(60):
        _step(nonpreg, seconds=1.0, bupivacaine_cv_ce=4.0)
        _step(preg, seconds=1.0, bupivacaine_cv_ce=4.0)
    assert preg.conduction_block > nonpreg.conduction_block + 0.08


def test_lipophilicity_lipid_rescue_acidosis_attenuation_and_asra_pressors(engine_factory, advance_time):
    """Lipid partitioning is strongest for bupivacaine, attenuated by acidemia, and impaired by high-dose epi / vasopressin."""
    tox = LocalAnestheticToxicity(70)
    tox.add_lipid(200.0)
    for _ in range(180):
        tox.step_lipid(1.0, 1.0)
    # Lipophilicity ordering: bupivacaine > ropivacaine > lidocaine (Dureau 2016; Mazoit 2009).
    f_bupi = tox.effect_input_fraction_for("bupivacaine", ph=7.40)
    f_ropi = tox.effect_input_fraction_for("ropivacaine", ph=7.40)
    f_lido = tox.effect_input_fraction_for("lidocaine", ph=7.40)
    assert f_bupi < f_ropi < f_lido < 1.0
    # Acidemia (pH 7.0) weakens bupivacaine lipid binding (Mazoit 2009).
    assert tox.effect_input_fraction_for("bupivacaine", ph=7.00) > f_bupi + 0.10

    # ASRA vasopressin harm during bupivacaine LAST CPR: pulmonary edema + lactic acidosis + worse recovery.
    ctrl = _ventilated(engine_factory)
    ctrl.trigger_last_crisis("bupivacaine", 75.0)
    advance_time(ctrl, 30.0, dt=1.0)
    vaso = copy.deepcopy(ctrl)
    epi_hi = copy.deepcopy(ctrl)
    for eng in (ctrl, vaso, epi_hi):
        eng.set_chest_compressions(True)
        eng.give_drug_bolus("midazolam", 5.0)
        eng.give_lipid_emulsion(100.0)
        eng.set_lipid_infusion(17.5)
    vaso.give_drug_bolus("vaso", 40.0)
    epi_hi.give_drug_bolus("epi", 1000.0)
    for eng in (ctrl, vaso, epi_hi):
        advance_time(eng, 180.0, dt=1.0)
    assert vaso.toxicity.vaso_last_harm > 0.3
    assert vaso.state.lung_water > ctrl.state.lung_water + 0.4
    assert vaso.state.lactate > ctrl.state.lactate + 0.5
    assert epi_hi.toxicity.epi_last_harm > 0.3
    assert epi_hi.toxicity.conduction_block > ctrl.toxicity.conduction_block

    # Clinical safety checks flag all ASRA contraindicated drugs during active LAST.
    from anasim.core.clinical_actions import check_drug_administration

    assert ctrl.toxicity.active or vaso.toxicity.active
    for key, amt, code in (
        ("vaso", 40.0, "vasopressin_in_last"),
        ("esmolol", 30.0, "beta_blocker_in_last"),
        ("labetalol", 10.0, "beta_blocker_in_last"),
        ("lidocaine", 100.0, "lidocaine_in_last"),
        ("flumazenil", 0.2, "flumazenil_in_last"),
        ("epi", 1000.0, "high_dose_epi_in_last"),
        ("propofol", 100.0, "propofol_in_last_cv"),
    ):
        fb = check_drug_administration(vaso, key, amount=amt)
        assert fb is not None and fb.code == code
    # Small <= 1 mcg/kg epinephrine bolus is permitted without high-dose warning.
    assert check_drug_administration(vaso, "epi", amount=50.0) is None


def test_epidural_intravascular_injection_and_epinephrine_test_dose(engine_factory, advance_time):
    """Accidental intravascular epidural bolus causes bupivacaine LAST and immediate epi test-dose tachycardia."""
    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=3), start=True)
    hr0 = eng.state.hr
    eng.give_epidural_bolus(
        drug="bupivacaine", volume_ml=3.0, concentration_pct=0.5,
        with_epinephrine=True, accidental_intravascular=True,
    )
    advance_time(eng, 45.0, dt=1.0)
    assert eng.state.hr > hr0 + 12.0  # 15 mcg IV epi positive test dose
    assert eng.neuraxial.sensory_level == 0.0  # No neuraxial block from IV injection
    assert eng.state.bupivacaine_ce > 0.5


def test_naloxone_reversal_renarcotization_and_opioid_rigidity(engine_factory, advance_time):
    """Naloxone reverses opioid apnea with withdrawal surge and renarcotization; high-dose fentanyl causes rigidity."""
    from anasim.core.clinical_actions import check_intubation

    # 1. Morphine respiratory depression -> naloxone reverses -> renarcotization as short-lived naloxone clears.
    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=2), start=True)
    eng.set_airway_mode("Mask")
    eng.set_fgf(10.0, 0.0, 0.0)
    eng.give_drug_bolus("morphine", 40.0)
    advance_time(eng, 900.0, dt=1.0)
    rr_depressed = eng.state.rr
    paco2_depressed = eng.state.pa_co2
    assert rr_depressed < 6.0 and paco2_depressed > 52.0

    hr_pre = eng.state.hr
    eng.give_drug_bolus("naloxone", 0.4)
    advance_time(eng, 180.0, dt=1.0)
    assert eng.state.rr > rr_depressed + 4.0
    assert eng.state.hr > hr_pre + 5.0  # Acute sympathetic reversal surge
    rr_reversed = eng.state.rr

    # Wait 75 min: naloxone clears faster than morphine -> renarcotization.
    advance_time(eng, 4500.0, dt=2.0)
    assert eng.state.naloxone_ce < 0.45
    assert eng.state.rr < rr_reversed - 2.0

    # 2. Rapid high-dose fentanyl bolus triggers chest-wall & glottic rigidity; rocuronium or naloxone relieves it.
    rig = engine_factory(config=SimulationConfig(mode="awake", rng_seed=5), start=True)
    rig.set_airway_mode("Mask")
    rig.set_fgf(10.0, 0.0, 0.0)
    rig.give_drug_bolus("fentanyl", 750.0)
    advance_time(rig, 120.0, dt=1.0)
    assert rig.state.opioid_rigidity > 0.55
    assert rig.state.muscle_rigidity is True
    assert rig.opioid_toxicity.glottic_obstruction > 0.50
    fb = check_intubation(rig)
    assert fb is not None and fb.blocked and fb.code == "opioid_rigidity_blocked"

    rig_nal = copy.deepcopy(rig)
    rig.give_drug_bolus("roc", 70.0)
    advance_time(rig, 90.0, dt=1.0)
    assert rig.state.opioid_rigidity < 0.15
    assert rig.opioid_toxicity.glottic_obstruction < 0.15

    rig_nal.give_drug_bolus("naloxone", 0.8)
    advance_time(rig_nal, 120.0, dt=1.0)
    assert rig_nal.state.opioid_rigidity < 0.15


def test_flumazenil_reverses_midazolam_resedates_and_unmasks_last_seizures(engine_factory, advance_time):
    """Flumazenil rapidly reverses midazolam hypnosis, permits resedation, and unmasks LAST convulsions."""
    eng = engine_factory(config=SimulationConfig(mode="awake", rng_seed=4), start=True)
    eng.set_airway_mode("Mask")
    eng.set_fgf(10.0, 0.0, 0.0)
    # Sustained midazolam loading (bolus + 60-min infusion) saturates peripheral compartments,
    # creating clinically realistic post-flumazenil resedation risk (Klotz and Kanto 1988).
    eng.give_drug_bolus("midazolam", 15.0)
    eng.set_drug_rate("midazolam", 60.0)
    advance_time(eng, 3600.0, dt=2.0)
    eng.set_drug_rate("midazolam", 0.0)
    loc_sedated = eng.state.loc
    bis_sedated = eng.state.bis
    assert loc_sedated > 0.65 and bis_sedated < 75.0

    eng.give_drug_bolus("flumazenil", 0.2)
    advance_time(eng, 120.0, dt=1.0)
    loc_reversed = eng.state.loc
    bis_reversed = eng.state.bis
    assert loc_reversed < 0.25 and bis_reversed > bis_sedated + 12.0

    # Resedation: short-lived flumazenil (t1/2 ~53 min) clears faster than loaded midazolam over 90 min.
    advance_time(eng, 5400.0, dt=2.0)
    assert eng.state.flumazenil_ce < 1.0
    assert eng.state.loc > loc_reversed + 0.08
    assert eng.state.bis < bis_reversed - 1.5

    # Flumazenil unmasks midazolam-suppressed LAST convulsions.
    last_eng = _ventilated(engine_factory)
    last_eng.give_drug_bolus("lidocaine", 300.0)
    advance_time(last_eng, 45.0, dt=1.0)
    assert last_eng.state.convulsions
    last_eng.give_drug_bolus("midazolam", 8.0)
    advance_time(last_eng, 60.0, dt=1.0)
    assert not last_eng.state.convulsions
    last_eng.give_drug_bolus("flumazenil", 1.0)
    advance_time(last_eng, 45.0, dt=1.0)
    assert last_eng.state.convulsions and last_eng.state.seizure_activity > 0.5


def test_hypermagnesemia_nmba_potentiation_paralysis_ecg_calcium_and_renal_accumulation(engine_factory, advance_time):
    """MgSO4 potentiates rocuronium, causes presynaptic paralysis & conduction block antagonized by CaCl2, and accumulates in renal failure."""
    from anasim.core.clinical_actions import check_drug_administration

    # 1. Renal failure slows MgSO4 clearance and triggers clinical warning.
    norm = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, renal_function=1.0)
    ckd = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True, renal_function=0.2)
    fb = check_drug_administration(ckd, "magnesium_sulfate", amount=4.0)
    assert fb is not None and fb.code == "magnesium_in_renal_impairment"
    ph0 = norm.state.ph
    for eng in (norm, ckd):
        eng.give_magnesium_sulfate(4.0)
        advance_time(eng, 3600.0, dt=2.0)
    assert ckd.state.magnesium > norm.state.magnesium + 0.15
    assert abs(norm.state.ph - ph0) < 0.02  # Electroneutral MgSO4 does not perturb SID/pH

    # 2. Therapeutic Mg2+ potentiates rocuronium blockade.
    no_mg = _ventilated(engine_factory)
    with_mg = _ventilated(engine_factory)
    with_mg.give_magnesium_sulfate(4.0)
    advance_time(with_mg, 180.0, dt=1.0)
    advance_time(no_mg, 180.0, dt=1.0)
    assert with_mg.state.magnesium > 1.5
    no_mg.give_drug_bolus("roc", 20.0)
    with_mg.give_drug_bolus("roc", 20.0)
    advance_time(no_mg, 180.0, dt=1.0)
    advance_time(with_mg, 180.0, dt=1.0)
    assert with_mg.state.tof < no_mg.state.tof - 10.0

    # 3. Severe hypermagnesemia causes respiratory paralysis, hypotension, and wide-complex block relieved by CaCl2.
    tox = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    tox.give_magnesium_sulfate(24.0)
    advance_time(tox, 240.0, dt=1.0)
    assert tox.state.magnesium > 6.5
    assert tox.state.twitch < 40.0
    assert tox.hemo.magnesium_block > 0.15
    abg = tox.draw_blood_gas()
    assert abg.magnesium == pytest.approx(tox.state.magnesium, abs=0.15)
    block_pre = tox.hemo.magnesium_block
    twitch_pre = tox.state.twitch
    tox.give_calcium_chloride(2000.0)
    advance_time(tox, 90.0, dt=1.0)
    assert tox.hemo.magnesium_block < block_pre
    assert tox.state.twitch > twitch_pre


def test_methemoglobinemia_spo2_gap_and_methylene_blue_rescue_and_ceiling(engine_factory, advance_time):
    """Benzocaine MetHb lowers ABG fractional SaO2 while SpO2 plateaus near 85%; methylene blue rescues up to 7 mg/kg."""
    from anasim.core.clinical_actions import check_drug_administration

    eng = _ventilated(engine_factory)
    advance_time(eng, 180.0, dt=1.0)
    assert eng.state.pao2 > 400.0 and eng.state.spo2 > 98.0

    eng.trigger_methemoglobinemia("benzocaine", severity=1.1)
    advance_time(eng, 1200.0, dt=2.0)
    assert 25.0 < eng.state.methb_pct < 50.0
    assert eng.state.pao2 > 350.0  # PaO2 remains high on 100% O2
    # Fractional SaO2 drops roughly linearly with MetHb while 2-wavelength SpO2 plateaus near ~85%.
    assert eng.state.sao2 < 75.0
    assert 82.0 <= eng.state.spo2 <= 89.0
    abg = eng.draw_blood_gas()
    assert abg.methb_pct == pytest.approx(eng.state.methb_pct, abs=0.5)
    assert abg.sao2 < 75.0 and abg.pao2 > 350.0

    # Methylene blue 1.5 mg/kg causes brief optical SpO2 dip, then clears MetHb substantially over 60 min.
    spo2_pre = eng.state.spo2
    methb_pre = eng.state.methb_pct
    eng.give_drug_bolus("methylene_blue", 1.5 * eng.patient.weight)
    advance_time(eng, 15.0, dt=1.0)
    assert eng.state.spo2 < spo2_pre - 1.5
    advance_time(eng, 3600.0, dt=2.0)
    assert eng.state.methb_pct < methb_pre * 0.45
    assert eng.state.sao2 > 85.0

    # Exceeding 7 mg/kg cumulative methylene blue triggers safety warning.
    fb = check_drug_administration(eng, "methylene_blue", amount=6.0 * eng.patient.weight)
    assert fb is not None and fb.code == "methylene_blue_max_dose"
