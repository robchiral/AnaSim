"""Oxygen-limited metabolism, lactate kinetics, and buffer-base accounting."""

import pytest
from scipy.integrate import solve_ivp

from anasim.core.projection import step_respiration
from anasim.core.runtime import step_lactate
from anasim.core.state import SimulationConfig
from anasim.physiology.lactate import LACTATED_RINGERS_MMOL_L, LactateModel
from anasim.physiology.respiration import RespiratoryModel

# Dunham 1991, Tables 1-2: O2 debt at 30 and 60 min (mL/kg) and lactate rise
# at 60 and 140 min (mmol/L). ANAEROBIC_PER_O2 was fitted to these groups.
DUNHAM = (
    (31.7, 58.6, 4.3, 0.3),
    (40.5, 79.8, 6.3, 1.2),
    (44.4, 98.5, 7.6, 2.6),
    (46.9, 108.4, 7.8, 1.9),
    (51.0, 120.5, 8.0, 2.2),
)


def _intervals(duration, dt):
    elapsed = 0.0
    while elapsed < duration:
        seconds = min(dt, duration - elapsed)
        yield elapsed, seconds
        elapsed += seconds


def _replay_debt(debt_30, debt_60, dt=1.0):
    """Give each 30-minute debt at a constant rate, then 80 minutes of recovery.

    A canine VO2 of 5 mL/kg/min sets the supply fraction for hepatic clearance.
    Recovery restores O2 at once and omits the study's dilution and reinfusion.
    """
    model = LactateModel(70.0)
    baseline = model.concentration
    for debt in (debt_30, debt_60 - debt_30):
        deficit_per_min = debt / 30.0
        for _, seconds in _intervals(1800.0, dt):
            model.step(seconds, deficit_per_min * 70.0 / 1000.0 * seconds / 60.0,
                       1.0 - deficit_per_min / 5.0)
    rise = model.concentration - baseline
    for _, seconds in _intervals(4800.0, dt):
        model.step(seconds, 0.0, 1.0)
    return rise, model.concentration - baseline


def _replay_ringers(dt=1.0):
    """Zitek 2018: 30 mL/kg racemic LR over 47 minutes."""
    model = LactateModel(70.0)
    baseline = model.concentration
    duration = 47.0 * 60.0
    for elapsed, seconds in _intervals(duration, dt):
        model.step(seconds, 0.0, 1.0, crystalloid_in_ml=30.0 * 70.0 * (elapsed + seconds) / duration)
    return model.concentration - baseline


def _replay_epinephrine(dt=1.0):
    """Grip 2015: a 60-minute ramp then 120 minutes at measured plasma levels."""
    model = LactateModel(70.0)
    plasma = (10.6 - 0.41) * 0.1832  # Exogenous nmol/L to ng/mL.
    for elapsed, seconds in _intervals(180.0 * 60.0, dt):
        ramp = min(1.0, (elapsed + seconds / 2.0) / 3600.0)
        model.step(seconds, 0.0, 1.0, epinephrine_ng_ml=plasma * ramp)
    return model.concentration


@pytest.mark.parametrize("hepatic_function", [0.5, 1.0])
def test_resting_turnover_is_at_equilibrium(hepatic_function):
    model = LactateModel(70.0, hepatic_function)
    initial = model.concentration
    acid = sum(model.step(1.0, 0.0, 1.0) for _ in range(3600))
    assert model.concentration == pytest.approx(initial, abs=1e-10)
    assert acid == pytest.approx(0.0, abs=1e-9)
    assert initial == pytest.approx(0.9 if hepatic_function == 1.0 else 1.475409836)


def test_reference_studies_bound_lactate_responses():
    errors = []
    for d30, d60, rise, observed_recovery in DUNHAM:
        predicted_rise, predicted_recovery = _replay_debt(d30, d60)
        errors.append(predicted_rise - rise)
        assert predicted_recovery == pytest.approx(observed_recovery, abs=1.5)
    # One coefficient cannot reproduce all five canine group means.
    assert (sum(error ** 2 for error in errors) / len(errors)) ** 0.5 < 1.0
    assert max(abs(error) for error in errors) < 1.5
    assert 0.42 < _replay_ringers() < 1.44
    assert _replay_epinephrine() == pytest.approx(3.9, abs=2.2)


def test_elevated_lactate_clears_more_slowly_than_resting_turnover():
    model = LactateModel(70.0)
    model.endogenous_mmol = 8.6 * model.volume_l
    target = 0.9 + (8.6 - 0.9) / 2.0
    for seconds in range(1, 3601):
        model.step(1.0, 0.0, 1.0)
        if model.concentration <= target:
            assert 40.0 < seconds / 60.0 < 60.0
            break
    else:
        pytest.fail("Excess lactate did not halve within 60 minutes")


def test_ringers_first_step_and_zero_duration_preserve_the_dose():
    model = LactateModel(70.0)
    assert model.step(0.0, 0.0, 1.0, crystalloid_in_ml=500.0) == 0.0
    assert model.concentration == pytest.approx(0.9)
    acid = model.step(1.0, 0.0, 1.0, crystalloid_in_ml=500.0)
    assert model.infused_mmol + model.cleared_infused_mmol == pytest.approx(
        0.5 * LACTATED_RINGERS_MMOL_L, abs=1e-10,
    )
    # Sodium lactate adds no proton; its metabolism returns buffer base.
    assert acid < 0.0
    for _ in range(3600):
        acid += model.step(1.0, 0.0, 1.0, crystalloid_in_ml=500.0)
    assert acid == pytest.approx(model.base_removed_mmol, abs=1e-9)


def test_septic_plateau_matches_reported_lactate_and_clearance():
    """Septic shock raised lactate production 2.4-fold, with lactate
    3.2 +/- 2.6 mmol/L (Revelly 2005). Hyperlactatemic septic patients cleared lactate at
    7.9 +/- 1.7 mL/kg/min (Levraut 1998)."""
    normal, septic, hepatic = LactateModel(70.0), LactateModel(70.0), LactateModel(70.0, 0.5)
    for _ in range(6 * 360):
        normal.step(10.0, 0.0, 1.0)
        septic.step(10.0, 0.0, 1.0, sepsis_severity=1.0)
        hepatic.step(10.0, 0.0, 1.0, sepsis_severity=1.0)
    assert septic.concentration == pytest.approx(3.2, abs=2.6)
    # After 6 hours, elimination matches the septic production rate.
    clearance_ml_kg_min = 1000.0 * 2.4 * septic.basal_mmol_min / septic.concentration / 70.0
    assert clearance_ml_kg_min == pytest.approx(7.9, abs=1.7)
    assert hepatic.concentration > septic.concentration > normal.concentration


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_oxygen_deficit_and_buffer_base_balance_in_engine(engine_factory, dt):
    engine = engine_factory(config=SimulationConfig(mode="steady_state", end_on_cardiac_arrest=False),
                            start=True)
    engine.maintenance_fluid_rate_ml_min = 0.0
    # Severe blood loss crosses the critical delivery without removing all flow.
    engine.hemo.add_volume(-0.4 * engine.hemo.blood_volume)
    step_respiration(engine, 0.0, engine.state.co)
    for _ in range(round(120.0 / dt)):
        engine.step(dt)
    assert engine.resp.oxygen_supply_fraction < 1.0
    assert engine.state.lactate > 1.0
    assert engine.state.base_excess < -0.1
    reference = engine.resp.chemistry.solve(
        engine.state.pa_co2, engine.state.sao2 / 100.0, engine.resp._reference_buffer_base,
    )
    # Falling aerobic CO2 output can raise arterial pH during shock. At the
    # same PaCO2, the metabolic acid still lowers pH relative to reference base.
    assert engine.state.ph < reference.ph
    assert engine.state.lactate == engine.lactate.concentration
    assert engine.state.base_excess == engine.resp.state.base_excess
    assert engine.resp.oxygen_deficit_l == 0.0


@pytest.mark.parametrize("cardiac_output", [0.0, 0.5, 5.0])
def test_oxygen_supply_caps_consumption_and_conserves_gas(patient, cardiac_output):
    model = RespiratoryModel(patient)
    transport = model.gas_transport
    initial_o2 = transport.oxygen_l + model.gas_volume_l * model._gas_capacity_per_l * model.state.p_alveolar_o2
    initial_base = transport.base_meq
    for _ in range(120):
        model.step(0.5, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                   mech_rr=12.0, mech_vt_l=0.5, cardiac_output=cardiac_output)
    final_o2 = transport.oxygen_l + model.gas_volume_l * model._gas_capacity_per_l * model.state.p_alveolar_o2
    demand = model.vco2 / model.rq / 1000.0
    assert final_o2 - initial_o2 == pytest.approx(transport.oxygen_from_air_l - transport.consumed_o2_l, abs=1e-8)
    assert model.oxygen_deficit_l + transport.consumed_o2_l == pytest.approx(demand, abs=1e-8)
    assert transport.base_meq == pytest.approx(initial_base, abs=1e-7)
    if cardiac_output == 5.0:
        assert model.oxygen_supply_fraction == 1.0
    else:
        assert model.oxygen_supply_fraction < 1.0


def test_lactate_replays_do_not_depend_on_step_size():
    for replay in (_replay_ringers, _replay_epinephrine):
        reference = replay(dt=0.1)
        for dt in (0.7, 1.0):
            assert replay(dt=dt) == pytest.approx(reference, abs=1e-5)
    reference_debt = _replay_debt(44.4, 98.5, dt=0.1)
    for dt in (0.7, 1.0):
        assert _replay_debt(44.4, 98.5, dt=dt) == pytest.approx(reference_debt, abs=1e-5)


def test_mixed_lactate_sources_match_an_independent_ode_solution():
    model = LactateModel(70.0)
    # Hold sources and organ factors constant while saturation changes. The
    # reference integrates both pools together with adaptive RK45 steps.
    endogenous_source = model.basal_mmol_min + 0.56 * 0.1 / 0.022414
    infused_source = 0.5 * LACTATED_RINGERS_MMOL_L / 60.0

    def balance(_time, amounts):
        rate = model.vmax_mmol_min / (model.km_mmol + sum(amounts))
        return [endogenous_source - rate * amounts[0], infused_source - rate * amounts[1]]

    reference = solve_ivp(balance, (0.0, 60.0), [model.resting_mmol, 0.0], rtol=1e-10, atol=1e-12)
    assert reference.success
    base_removed = 0.0
    for seconds in range(1, 3601):
        base_removed += model.step(1.0, 0.1 / 60.0, 1.0, crystalloid_in_ml=500.0 * seconds / 3600.0)
    assert (model.endogenous_mmol, model.infused_mmol) == pytest.approx(reference.y[:, -1], abs=1e-5)
    assert model.infused_mmol + model.cleared_infused_mmol == pytest.approx(7.0, abs=1e-9)
    assert base_removed == pytest.approx(model.base_removed_mmol, abs=1e-9)


def test_transport_and_lactate_conserve_buffer_base_during_shock_and_recovery(awake_engine):
    engine = awake_engine
    model = engine.resp
    transport = model.gas_transport
    initial_base = transport.base_meq
    initial_base_excess = model.state.base_excess
    for flow in (0.5, 5.0):
        for _ in range(300):
            model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                       mech_rr=12.0, mech_vt_l=0.5, cardiac_output=flow)
            step_lactate(engine, 1.0)
        removed = engine.lactate.base_removed_mmol
        assert transport.base_meq + removed == pytest.approx(initial_base, abs=1e-7)
        assert model.state.base_excess == pytest.approx(
            initial_base_excess - removed / (sum(transport.volumes) + transport.extracellular_volume), abs=1e-7,
        )
        if flow == 0.5:
            peak = engine.lactate.concentration
        else:
            assert 0.9 < engine.lactate.concentration < peak


def test_compensated_blood_loss_preserves_lactate_and_base_excess(engine_factory):
    """Ward 2010: progressive central hypovolemia lowered O2 delivery without
    changing O2 uptake, lactate, or base excess."""
    engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True)
    engine.maintenance_fluid_rate_ml_min = 0.0
    baseline_co = engine.state.co
    baseline_be = engine.state.base_excess
    engine.start_hemorrhage(100.0)  # 1 L over 10 minutes.
    for _ in range(600):
        engine.step(1.0)
        assert engine.resp.oxygen_supply_fraction == 1.0
        assert engine.state.lactate == pytest.approx(0.9, abs=0.05)
        assert engine.state.base_excess == pytest.approx(baseline_be, abs=0.1)
    assert engine.state.co < 0.85 * baseline_co
