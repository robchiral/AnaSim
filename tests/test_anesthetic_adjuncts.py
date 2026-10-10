"""Fentanyl, midazolam, etomidate, ketamine, and lidocaine through the engine; docstrings cite each bound."""

from anasim.core.state import SimulationConfig


def _advance(engine, seconds, dt=1.0):
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _awake(engine_factory, **patient):
    engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=3), start=True, **patient)
    engine.set_airway_mode("Mask")
    engine.set_fgf(8.0, 0.0)
    _advance(engine, 30)
    return engine


def _propofol_induction(engine_factory, midazolam_mg=0.0, propofol_mg=140.0):
    engine = _awake(engine_factory)
    engine.give_drug_bolus("midazolam", midazolam_mg)
    _advance(engine, 180)
    engine.give_drug_bolus("propofol", propofol_mg)
    _advance(engine, 90)
    return engine


def _intubation_response(engine_factory, fentanyl_mcg=0.0, lidocaine_mg=0.0, lidocaine_lead_s=210):
    """Peak HR and MAP rises, and the tolerance of laryngoscopy, at intubation 210 s after propofol."""
    engine = _awake(engine_factory)
    engine.give_drug_bolus("fentanyl", fentanyl_mcg)
    _advance(engine, 180)
    for second in range(210):
        if second == 0:
            engine.give_drug_bolus("propofol", 140.0)
        if second == 90:
            engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)
        if second == 180:
            engine.set_airway_mode("ETT")
            engine.set_vent_power(True)
        if second == 210 - lidocaine_lead_s:
            engine.give_drug_bolus("lidocaine", lidocaine_mg)
        engine.step(1.0)
    base_hr, base_map, tolerance = engine.state.hr, engine.state.map, engine.state.tol
    engine.start_disturbance("stim_intubation_pulse")
    peak_hr, peak_map = base_hr, base_map
    for _ in range(600):
        engine.step(0.1)
        peak_hr = max(peak_hr, engine.state.hr)
        peak_map = max(peak_map, engine.state.map)
    return peak_hr - base_hr, peak_map - base_map, tolerance


def test_ketamine_induction_keeps_breathing_and_raises_pressure(engine_factory):
    """Idvall 1979: arterial pressure, HR, and CO rose 15-30%, and patients woke
    at 0.64 mcg/mL. Bourke 1987: 3 mg/kg shifted the CO2 response 2 mmHg.
    In AnaSim, ketamine preserves pharyngeal tone and BIS.
    """
    engine = _awake(engine_factory)
    base_map, base_co, base_mv = engine.state.map, engine.state.co, engine.state.mv

    engine.give_drug_bolus("ketamine", 2.0 * engine.patient.weight)
    _advance(engine, 60)
    assert engine.state.loc > 0.9
    _advance(engine, 120)
    assert 1.12 * base_map < engine.state.map < 1.35 * base_map
    assert engine.state.co > 1.05 * base_co
    assert engine.state.mv > 0.85 * base_mv
    assert engine.state.airway_obstruction < 0.05
    assert engine.state.bis > 90.0

    for _ in range(30 * 60):
        engine.step(1.0)
        if engine.state.loc < 0.5:
            break
    assert 0.5 < engine.state.ketamine_ce < 0.8


def test_midazolam_sedates_more_with_age_and_potentiates_propofol(engine_factory):
    """Albrecht 1999: elderly volunteers needed half the young EC50. Short 1992:
    midazolam lowered the propofol ED50 for anesthesia by 52%.
    """
    engine = _awake(engine_factory)
    engine.give_drug_bolus("midazolam", 2.0)
    _advance(engine, 600)
    assert engine.state.loc < 0.05
    assert engine.state.bis > 85.0

    peak_loc = {}
    for age in (26, 70):
        engine = _awake(engine_factory, age=age)
        engine.give_drug_bolus("midazolam", 0.1 * engine.patient.weight)
        peaks = []
        for _ in range(900):
            engine.step(1.0)
            peaks.append(engine.state.loc)
        peak_loc[age] = max(peaks)
    assert peak_loc[70] > 2.0 * peak_loc[26]

    # Probe near the LOC transition; 70 and 140 mg both saturate the clinical
    # response and cannot distinguish the interaction.
    half_dose = _propofol_induction(engine_factory, propofol_mg=25.0)
    with_midazolam = _propofol_induction(engine_factory, midazolam_mg=2.0, propofol_mg=25.0)
    full_dose = _propofol_induction(engine_factory, propofol_mg=50.0)
    assert half_dose.state.loc < 0.4
    assert with_midazolam.state.loc > full_dose.state.loc


def test_fentanyl_depresses_breathing_and_accumulates(engine_factory):
    """Fentanyl acts as 0.82 times its concentration of remifentanil (isoflurane
    MAC reduction: McEwan 1993; Lang 1996). Hughes 1992: the half-time after an
    infusion depends on its duration.
    """
    engine = _awake(engine_factory)
    engine.set_fgf(0.0, 8.0)
    base_paco2 = engine.state.pa_co2
    base_rr = engine.resp.state.rr
    engine.give_drug_bolus("fentanyl", 100.0)
    _advance(engine, 300)
    assert engine.state.pa_co2 > base_paco2 + 3.0
    assert engine.resp.state.rr < 0.85 * base_rr

    half_times = []
    for minutes in (15, 180):
        engine = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True, rng_seed=3), start=True)
        engine.enable_tci("fentanyl", 2.0)
        _advance(engine, minutes * 60, dt=2.0)
        half_times.append(engine.get_predicted_csht("fentanyl"))
    assert half_times[1] > 3.0 * half_times[0]
    assert half_times[1] > 60.0


def test_fentanyl_blunts_the_intubation_response_more_than_lidocaine(engine_factory):
    """Qin 2025 (18 trials): lidocaine 1-2 mg/kg lowered the MAP rise at
    intubation by 3.85 mmHg (95% CI 1.09-6.61), with no significant HR effect in
    Western trials; Wilson 1991 found HR rose 21-26% with or without it. Okuda
    1990: intubation 30 s after lidocaine raised MAP and HR, unlike at 1-3 min.
    """
    hr_alone, map_alone, tol_alone = _intubation_response(engine_factory)
    hr_fentanyl, _, tol_fentanyl = _intubation_response(engine_factory, fentanyl_mcg=140.0)
    hr_lidocaine, map_lidocaine, _ = _intubation_response(engine_factory, lidocaine_mg=105.0)
    _, map_rushed, _ = _intubation_response(engine_factory, lidocaine_mg=105.0, lidocaine_lead_s=30)

    assert tol_fentanyl > tol_alone + 0.2
    assert hr_fentanyl < 0.7 * hr_alone
    assert 1.09 < map_alone - map_lidocaine < 6.61
    assert hr_lidocaine > 0.8 * hr_alone
    assert map_alone - map_rushed < 0.7 * (map_alone - map_lidocaine)


def test_etomidate_induction_is_brief_and_hemodynamically_stable(engine_factory):
    """Valk 2021: 0.3 mg/kg gives 5-10 min of hypnosis with minimal change in
    blood pressure and less ventilatory depression than other hypnotics.
    Arden 1986: a smaller initial volume deepens the effect in the elderly.
    """
    inductions = {}
    for drug, mg_per_kg in (("etomidate", 0.3), ("propofol", 2.0)):
        engine = _awake(engine_factory)
        base_map = engine.state.map
        engine.give_drug_bolus(drug, mg_per_kg * engine.patient.weight)
        trace = []
        for _ in range(900):
            engine.step(1.0)
            trace.append((engine.state.loc, engine.state.map, engine.state.mv))
        inductions[drug] = base_map, trace

    base_map, trace = inductions["etomidate"]
    assert trace[119][0] > 0.85
    awake_at = next(second for second, (loc, _, _) in enumerate(trace) if second > 60 and loc < 0.5)
    assert 4 * 60 < awake_at < 10 * 60
    assert min(map_ for _, map_, _ in trace) > 0.95 * base_map
    propofol_base, propofol_trace = inductions["propofol"]
    assert min(map_ for _, map_, _ in propofol_trace) / propofol_base < min(map_ for _, map_, _ in trace) / base_map
    assert min(mv for _, _, mv in trace) > min(mv for _, _, mv in propofol_trace)

    peak_loc = {}
    for age in (25, 70):
        engine = _awake(engine_factory, age=age)
        engine.give_drug_bolus("etomidate", 0.2 * engine.patient.weight)
        peaks = []
        for _ in range(600):
            engine.step(1.0)
            peaks.append(engine.state.loc)
        peak_loc[age] = max(peaks)
    assert peak_loc[70] > peak_loc[25] + 0.1
