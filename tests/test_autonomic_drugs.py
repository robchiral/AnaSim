"""Esmolol, labetalol, and glycopyrrolate through the engine; docstrings cite each bound."""

import pytest

from anasim.core.state import SimulationConfig


def _advance(engine, seconds, dt=1.0):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _extreme_hr(engine, seconds, pick):
    values = []
    for _ in range(round(seconds / 0.1)):
        engine.step(0.1)
        values.append(engine.state.hr)
    return pick(values)


def _tiva(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=3), start=True
    )
    engine.disable_tci("nore")
    _advance(engine, 300)
    return engine


def test_labetalol_lowers_heart_rate_longer_than_blood_pressure(engine_factory):
    """Abernethy 1987, 50 mg over 10 min in young hypertensive adults (32 y, 95 kg):
    HR fell up to 17 bpm and SBP up to 25 mmHg; BP recovered within 30 min, and
    HR stayed low for 3 h. In AnaSim a smaller BP fall lasts as long as the beta
    block, because in sinus rhythm the baroreflex adjusts HR only.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="awake", rng_seed=3), start=True, age=32, weight=95, height=180
    )
    _advance(engine, 60)
    base_hr, base_sbp = engine.state.hr, engine.state.sbp

    changes = []
    for minute in range(180):
        if minute < 10:
            engine.give_drug_bolus("labetalol", 5.0)
        _advance(engine, 60)
        changes.append((engine.state.hr - base_hr, engine.state.sbp - base_sbp))

    max_hr_fall = min(hr for hr, _ in changes)
    max_sbp_fall = min(sbp for _, sbp in changes)
    assert -20.0 < max_hr_fall < -8.0
    assert -35.0 < max_sbp_fall < -15.0
    assert changes[59][1] > 0.5 * max_sbp_fall
    assert changes[179][0] < 0.5 * max_hr_fall


def test_esmolol_onset_offset_and_atrial_fibrillation_rate(engine_factory):
    """Wiest 2012: a 500 mcg/kg load gives 90% of beta block by 5 min, recovery
    takes 18-30 min after stopping, and esmolol controls the AF ventricular rate.
    Sum 1983: steady-state blood level 0.569 mcg/mL at 150 mcg/kg/min.
    """
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=3), start=True)
    _advance(engine, 60)
    base_hr = engine.state.hr

    engine.set_drug_rate("esmolol", 0.5 * engine.patient.weight)
    _advance(engine, 60)
    engine.set_drug_rate("esmolol", 0.15 * engine.patient.weight)
    _advance(engine, 240)
    fall_at_5_min = engine.state.hr - base_hr
    _advance(engine, 1500)
    steady_fall = engine.state.hr - base_hr
    assert steady_fall < -6.0
    assert fall_at_5_min < 0.8 * steady_fall
    assert engine.pk_esmolol.state.c1 == pytest.approx(0.569, rel=0.1)

    engine.set_drug_rate("esmolol", 0.0)
    _advance(engine, 18 * 60)
    assert engine.state.hr - base_hr > 0.35 * steady_fall
    _advance(engine, 12 * 60)
    assert engine.state.hr - base_hr > -1.5

    engine.set_rhythm("AFIB")
    _advance(engine, 300)
    assert engine.state.hr == pytest.approx(110.0, abs=2.0)
    engine.set_drug_rate("esmolol", 0.15 * engine.patient.weight)
    _advance(engine, 600)
    assert engine.state.hr < 95.0


def test_esmolol_reduces_intubation_and_epinephrine_tachycardia(engine_factory):
    """Wiest 2012: esmolol 1-2 mg/kg attenuates the HR response to laryngoscopy.
    Competitive beta-1 block shifts the epinephrine chronotropy curve right.
    """
    responses = {}
    for esmolol_mg in (0.0, 70.0):
        engine = _tiva(engine_factory)
        engine.give_drug_bolus("esmolol", esmolol_mg)
        _advance(engine, 120)
        base_hr = engine.state.hr
        engine.start_disturbance("stim_intubation_pulse")
        stimulation = _extreme_hr(engine, 60, max) - base_hr
        _advance(engine, 60)
        base_hr = engine.state.hr
        engine.give_drug_bolus("epi", 10.0)
        responses[esmolol_mg] = stimulation, _extreme_hr(engine, 120, max) - base_hr

    (stim_off, epi_off), (stim_on, epi_on) = responses[0.0], responses[70.0]
    assert stim_on < 0.6 * stim_off
    assert epi_on < 0.6 * epi_off


def test_glycopyrrolate_raises_rate_and_blocks_vagal_bradycardia(engine_factory):
    """Ali-Melkkilä 1993: 4 mcg/kg after induction gave 11.8 ng/mL about 5 min
    later and prevented reflex bradycardia during eye-muscle traction.
    """
    engine = _tiva(engine_factory)
    base_hr = engine.state.hr
    engine.give_drug_bolus("phenyl", 100.0)
    reflex_before = _extreme_hr(engine, 180, min) - base_hr
    _advance(engine, 300)

    base_hr = engine.state.hr
    engine.give_drug_bolus("glyco", 0.004 * engine.patient.weight)
    _advance(engine, 60)
    assert engine.state.hr - base_hr > 8.0
    _advance(engine, 240)
    assert engine.pk_glyco.state.c1 == pytest.approx(11.8, rel=0.25)
    assert engine.state.hr - base_hr < 30.0

    base_hr = engine.state.hr
    engine.give_drug_bolus("phenyl", 100.0)
    reflex_after = _extreme_hr(engine, 180, min) - base_hr
    assert reflex_before < -6.0
    assert reflex_after > 0.5 * reflex_before

    engine.set_rhythm("SINUS_BRADY")
    _advance(engine, 120)
    assert engine.state.hr > 60.0
