"""Quantitative physiology tests for general neuraxial anesthesia and high/total spinal block."""

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import AirwayType, SimulationConfig
from anasim.patient.patient import Patient


def _advance(engine: SimulationEngine, seconds: float, dt: float = 0.5) -> None:
    engine.running = True
    for _ in range(round(seconds / dt)):
        engine.step(dt)


@pytest.mark.clinical
def test_spinal_age_spread_and_hemodynamics_veering_1988_rooke_1997():
    """15 mg hyperbaric bupivacaine spreads higher in elderly patients (Veering 1988) and
    causes larger SVR and MAP drops in elderly adults than young adults without ischemic coma (Ward 1965; Rooke 1997)."""
    young = SimulationEngine(Patient(age=33), SimulationConfig(mode="awake", rng_seed=1))
    old = SimulationEngine(Patient(age=73), SimulationConfig(mode="awake", rng_seed=1))
    very_old = SimulationEngine(Patient(age=85), SimulationConfig(mode="awake", rng_seed=1))
    m_y0, svr_y0 = young.state.map, young.state.svr
    m_o0, svr_o0 = old.state.map, old.state.svr

    young.give_spinal("bupivacaine", 15.0, "hyperbaric")
    old.give_spinal("bupivacaine", 15.0, "hyperbaric")
    very_old.give_spinal("bupivacaine", 15.0, "hyperbaric")
    _advance(young, 720.0, dt=1.0)
    _advance(old, 720.0, dt=1.0)
    _advance(very_old, 720.0, dt=1.0)

    # Elderly spread ~2+ segments higher (Veering 1988) and drop SVR/MAP more steeply than young (Rooke 1997).
    assert old.neuraxial.sensory_level > young.neuraxial.sensory_level + 1.5
    assert young.neuraxial.sensory_level - young.neuraxial.motor_level == pytest.approx(2.8, abs=0.2)
    young_map_drop = young.state.map / m_y0 - 1.0
    old_map_drop = old.state.map / m_o0 - 1.0
    young_svr_drop = young.state.svr / svr_y0 - 1.0
    old_svr_drop = old.state.svr / svr_o0 - 1.0
    assert young.state.map < m_y0 - 8.0
    assert old_map_drop < young_map_drop - 0.10
    assert old_svr_drop < young_svr_drop - 0.10
    assert -0.45 < old_map_drop < -0.25
    assert -0.35 < old_svr_drop < -0.18
    assert old.state.neuraxial_bromage == 3
    # Routine spinal in elderly 73-85 yr stays perfusing without myocardial ischemia or coma (Rooke 1997).
    for elderly in (old, very_old):
        assert elderly.state.map >= 48.0
        assert elderly.state.st_depression_mv < 0.05
        assert elderly.state.loc == 0.0
        assert not elderly.state.cardiac_arrest

    check = old.check_neuraxial_level()
    assert check.sensory_label == old.state.neuraxial_sensory_label
    assert check.bromage == 3


@pytest.mark.clinical
def test_epidural_plain_vs_epinephrine_matches_ward_1965_and_blocks_surgical_stimulus():
    """Plain epidural produces gentler MAP drop than spinal and wider sensory-motor gap (Freund 1967);
    adding 1:200,000 epinephrine raises CO and HR while lowering SVR (Ward 1965); bolus and continuous
    infusion block surgical nociception and obey drug-potency regression."""
    spinal = SimulationEngine(Patient(age=33), SimulationConfig(mode="awake", rng_seed=1))
    plain = SimulationEngine(Patient(age=33), SimulationConfig(mode="awake", rng_seed=1))
    epi = SimulationEngine(Patient(age=33), SimulationConfig(mode="awake", rng_seed=1))
    co0, svr0, hr0 = epi.state.co, epi.state.svr, epi.state.hr

    spinal.give_spinal("bupivacaine", 15.0, "hyperbaric")
    plain.give_epidural_bolus("bupivacaine", 15.0, 0.5, with_epinephrine=False)
    epi.give_epidural_bolus("bupivacaine", 15.0, 0.5, with_epinephrine=True)
    _advance(spinal, 840.0, dt=1.0)
    _advance(plain, 840.0, dt=1.0)
    _advance(epi, 840.0, dt=1.0)

    assert plain.neuraxial.sensory_level - plain.neuraxial.motor_level == pytest.approx(4.6, abs=0.2)
    assert plain.state.map > spinal.state.map + 3.0  # Gentler plain epidural hypotension than spinal (Ward 1965)
    assert plain.state.map > 72.0
    # Absorbed epidural epinephrine raises CO and HR while dropping SVR (Ward 1965).
    assert epi.state.co > co0 * 1.10
    assert epi.state.hr > hr0 + 4.0
    assert epi.state.svr < svr0 * 0.85

    # Surgical stimulation under a continuous epidural infusion causes far less MAP surge than unblocked.
    unblocked = SimulationEngine(Patient(age=40, surgical_site="lower_abdominal"), SimulationConfig(mode="steady_state", rng_seed=1))
    blocked = SimulationEngine(Patient(age=40, surgical_site="lower_abdominal"), SimulationConfig(mode="steady_state", rng_seed=1))
    blocked.give_epidural_bolus("bupivacaine", 15.0, 0.5)
    blocked.set_epidural_infusion(10.0, drug="bupivacaine", concentration_pct=0.125)
    _advance(unblocked, 600.0, dt=1.0)
    _advance(blocked, 600.0, dt=1.0)
    m_u0, m_b0 = unblocked.state.map, blocked.state.map
    unblocked.start_disturbance("stim_sustained_surgery")
    blocked.start_disturbance("stim_sustained_surgery")
    _advance(unblocked, 90.0, dt=1.0)
    _advance(blocked, 90.0, dt=1.0)
    assert (blocked.state.map - m_b0) < (unblocked.state.map - m_u0) - 5.0

    # Prior surgical bolus regresses down toward a low maintenance infusion; lidocaine infusion is less potent than bupivacaine.
    bupi_inf = SimulationEngine(Patient(age=40), SimulationConfig(mode="awake", rng_seed=1))
    lido_inf = SimulationEngine(Patient(age=40), SimulationConfig(mode="awake", rng_seed=1))
    bupi_inf.set_epidural_infusion(10.0, drug="bupivacaine", concentration_pct=0.125)
    lido_inf.set_epidural_infusion(10.0, drug="lidocaine", concentration_pct=0.125)
    _advance(bupi_inf, 1200.0, dt=2.0)
    _advance(lido_inf, 1200.0, dt=2.0)
    assert bupi_inf.neuraxial.sensory_level > lido_inf.neuraxial.sensory_level + 2.0


@pytest.mark.clinical
def test_high_and_total_spinal_apnea_loc_bezold_jarisch_and_resuscitation():
    """High thoracic spinal spares the diaphragm (Freund 1967); total spinal ascends past C4 -> phrenic apnea,
    brainstem LOC allowing intubation without extra propofol, and preload/vagal bradycardia reversible by
    atropine + fluids + pressors (Pollard 2001; Caplan 1988)."""
    # High thoracic spinal spares the diaphragm (Freund 1967: no apnea).
    thoracic = SimulationEngine(Patient(age=40), SimulationConfig(mode="awake", rng_seed=1))
    thoracic.give_spinal("bupivacaine", 18.0)
    _advance(thoracic, 600.0, dt=1.0)
    assert thoracic.neuraxial.sensory_level >= 18.5
    assert not thoracic.state.apnea and thoracic.state.vt > 250.0

    # Accidental intrathecal epidural bolus -> high/total spinal.
    hs = SimulationEngine(
        Patient(age=31, weight=78, height=166, sex="female", gestational_age_weeks=39.0),
        SimulationConfig(mode="awake", rng_seed=1),
    )
    hs.trigger_high_spinal()
    _advance(hs, 220.0, dt=0.5)
    assert hs.neuraxial.sensory_level >= 28.0
    assert hs.state.apnea is True
    assert hs.state.loc >= 0.85
    assert hs.state.hr < 50.0
    assert hs.state.map < 50.0

    # Total spinal unconsciousness permits intubation without extra hypnotic and without awareness warning.
    fb = hs.attempt_intubation("dl")
    assert fb is None
    assert hs.state.airway_mode == AirwayType.ETT
    hs.set_fgf(10.0, 0.0, 0.0)
    hs.set_vent_power(True)

    # Resuscitation: left lateral tilt, crystalloid preload, atropine for Bezold-Jarisch bradycardia, pressor.
    hr_nadir, map_nadir = hs.state.hr, hs.state.map
    hs.set_position("left_lateral")
    hs.give_fluid(1000.0)
    hs.give_drug_bolus("atropine", 1.0)
    hs.give_drug_bolus("ephedrine", 25.0)
    hs.set_drug_rate("nore", 10.0)
    _advance(hs, 120.0, dt=0.5)
    assert hs.state.hr > hr_nadir + 15.0
    assert hs.state.map > map_nadir + 25.0 and hs.state.map >= 65.0
    assert hs.state.sao2 >= 96.0


@pytest.mark.invariant
def test_neuraxial_step_size_invariance():
    """Spinal and high-spinal trajectories and event_active state match across dt=0.05..2.0 s."""
    for high in (False, True):
        res = []
        for dt in (0.1, 0.5):
            e = SimulationEngine(Patient(age=50), SimulationConfig(mode="awake", dt=dt, rng_seed=1))
            if high:
                e.trigger_high_spinal()
            else:
                e.give_spinal("bupivacaine", 15.0)
            _advance(e, 180.0, dt=dt)
            if high:
                assert e.event_active("high_spinal")
                assert e.neuraxial.last_mode == "high_spinal"
            res.append((e.neuraxial.sensory_level, e.state.map, e.state.hr, e.state.co))
        assert res[0][0] == pytest.approx(res[1][0], abs=0.1)
        assert res[0][1] == pytest.approx(res[1][1], abs=1.5)
        assert res[0][2] == pytest.approx(res[1][2], abs=1.5)
        assert res[0][3] == pytest.approx(res[1][3], abs=0.15)

    # High spinal last_mode and event_active survive step 1 at fine dt=0.05 s and coarse dt=2.0 s.
    for dt in (0.05, 2.0):
        e = SimulationEngine(Patient(age=40), SimulationConfig(mode="awake", dt=dt, rng_seed=1))
        e.running = True
        e.trigger_high_spinal()
        e.step(dt)
        assert e.neuraxial.last_mode == "high_spinal"
        assert e.event_active("high_spinal")
