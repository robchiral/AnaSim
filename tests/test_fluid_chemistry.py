"""Fluid-loading protocols, transfusion, and renal compensation."""

import math

import pytest

from anasim.core.state import SimulationConfig
from anasim.physiology.fluid_balance import FluidBalance
from anasim.physiology.hemodynamics import HemodynamicModel
from anasim.physiology.plasma_buffers import PlasmaBuffers, buffer_coefficients


def test_figge_albumin_phosphate_charge_and_analytic_derivatives():
    """Figge 2018 Tables A1-A2 agree with the clinical linearization near pH 7.4."""
    for ph in (6.9, 7.1, 7.4, 7.7, 7.9):
        albumin, albumin_ph, phosphate, phosphate_ph = buffer_coefficients(ph)
        before = buffer_coefficients(ph - 1e-5)
        after = buffer_coefficients(ph + 1e-5)
        assert albumin_ph == pytest.approx((after[0] - before[0]) / 2e-5, rel=1e-7)
        assert phosphate_ph == pytest.approx((after[2] - before[2]) / 2e-5, rel=1e-7)
        assert albumin == pytest.approx(0.123 * ph - 0.631, abs=0.012)
        assert phosphate == pytest.approx(0.309 * ph - 0.469, abs=0.06)
    assert PlasmaBuffers(42.0, 1.0).contents(7.4)[0] == pytest.approx(13.55, abs=0.1)


def test_saline_and_ringers_follow_scheingraber_loading_protocol(engine_factory):
    """Scheingraber 1999, with ventilation adjusted to keep PaCO2 near 40.

    The study gave 71 mL/kg by 120 min with about 900 mL of blood loss. Here
    60 mL/kg without blood loss gives the study's dilution (chloride 104 to 115,
    SID 9 mmol/L lower, Hb 12 to 9 g/dL), as model urine is below the 717 mL
    reported. Table 2: Stewart bicarbonate (SID - Prot-) fell from 23.9 to 19.3
    with saline and from 23.8 to 23.2 with Ringer's. Measured bicarbonate fell
    to 18.4 (pH 7.28) because unmeasured anions rose by 1 mmol/L in both groups
    during surgery, which the model omits.
    """
    def strong_ion_difference(state):
        return state.sodium + state.potassium - state.chloride - state.lactate

    changes = {}
    results = {}
    for fluid in ("saline", "crystalloid"):
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0),
            sex="female", weight=65.0, height=165.0, age=58.0, start=True,
        )
        state = engine.state
        initial = state.albumin_g_l, state.hco3, strong_ion_difference(state), engine.hemo.hb_conc
        for second in range(7200):
            engine.hemo.add_volume(30.0 * 65.0 / 3600.0, label=fluid)
            if second % 30 == 0:
                settings = engine.vent.settings
                deadspace = engine.resp.vd_deadspace
                vt = deadspace + (settings.tv / 1000.0 - deadspace) * state.pa_co2 / 40.0
                engine.set_vent_settings(rr=settings.rr, vt=vt, peep=settings.peep, ie="1:2", mode="VCV")
            engine.step(1.0)
        albumin, hco3, sid, hb = initial
        assert state.pa_co2 == pytest.approx(40.0, abs=0.5)
        assert 0.55 * albumin < state.albumin_g_l < 0.8 * albumin
        assert engine.hemo.hb_conc / hb == pytest.approx(0.75, abs=0.04)
        changes[fluid] = state.hco3 - hco3, strong_ion_difference(state) - sid
        results[fluid] = state
    saline, ringers = results["saline"], results["crystalloid"]
    assert saline.chloride == pytest.approx(115.0, abs=2.0)
    assert changes["saline"][1] == pytest.approx(-9.0, abs=1.0)
    assert changes["saline"][0] == pytest.approx(19.3 - 23.9, abs=1.0)
    assert changes["crystalloid"][0] == pytest.approx(23.2 - 23.8, abs=1.0)
    assert saline.sodium - saline.chloride - saline.hco3 < 9.0
    assert ringers.lactate == pytest.approx(2.0, abs=0.6)
    assert ringers.lactate > saline.lactate + 0.5
    assert ringers.chloride < saline.chloride - 7.0


def test_protein_and_water_follow_bleeding_leak_and_albumin_replacement(engine_factory):
    engine = engine_factory(start=True)
    hemo = engine.hemo
    fluids = hemo.fluids
    initial_protein = fluids.plasma_albumin_g + fluids.interstitial_albumin_g
    initial_water = fluids.ecf_l + fluids.icf_l + fluids.red_cells_l
    plasma_lost = 0.5 * (1.0 - hemo.get_hematocrit())
    hemo.add_volume(-500.0)
    assert fluids.plasma_albumin_g + fluids.interstitial_albumin_g == pytest.approx(
        initial_protein - 42.0 * plasma_lost,
    )
    remaining_protein = fluids.plasma_albumin_g + fluids.interstitial_albumin_g
    fluids.shift_water(-250.0, protein_fraction=1.0)
    fluids.shift_water(100.0)
    assert fluids.plasma_albumin_g + fluids.interstitial_albumin_g == pytest.approx(remaining_protein)
    fluids.infuse(250.0, 200.0, 0.0, "colloid")
    assert fluids.plasma_albumin_g + fluids.interstitial_albumin_g == pytest.approx(remaining_protein + 12.5)
    assert fluids.ecf_l + fluids.icf_l + fluids.red_cells_l == pytest.approx(initial_water - 0.5 + 0.25)


def test_rapid_transfusion_of_stored_blood_and_calcium_replacement(engine_factory):
    """Ten 3-week-old units in 5 min carry potassium, citrate, acid, and CO2
    (Rabcuka 2025: pH 6.5 and CO2 13.0 kPa, supernatant potassium 38 mmol/L).
    """
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0), start=True,
    )
    initial = engine.draw_blood_gas()
    for _ in range(300):
        engine.hemo.add_volume(10.0, hematocrit=0.55, label="blood")
        engine.step(1.0)
    transfused = engine.draw_blood_gas()
    assert transfused.potassium > initial.potassium + 0.5
    assert transfused.ionized_calcium < initial.ionized_calcium - 0.15
    assert transfused.base_excess < initial.base_excess - 2.0
    assert transfused.pa_co2 > initial.pa_co2 + 3.0
    engine.give_calcium_chloride(1000.0)
    treated = engine.draw_blood_gas()
    assert treated.ionized_calcium > transfused.ionized_calcium + 0.2
    for dose in (0.0, -1.0, math.nan, math.inf, 10001.0):
        with pytest.raises(ValueError):
            engine.give_calcium_chloride(dose)


def test_renal_response_is_slow_and_requires_function_and_urine():
    outcomes = []
    for renal_function in (0.0, 1.0):
        fluids = FluidBalance(70.0, 4900.0, 1984.5)
        fluids.seed_base(700.0, 72.0, 50.0)
        for minute in range(360):
            if renal_function:
                volume = 70.0 * 0.5 / 60.0
                fluids.lose_urine(volume)
                fluids.shift_water(volume)
            fluids.step(
                60.0, ph=7.26, paco2=60.0, resting_paco2=40.0,
                renal_function=renal_function, renal_perfusion=1.0, hepatic_function=1.0,
                oxygen_supply=1.0, epinephrine=0.0,
            )
            if minute == 14:
                assert fluids.renal_base_meq / fluids.ecf_l < 0.1
        assert fluids.base_meq - 700.0 == pytest.approx(
            fluids.net_acid_excretion_mmol - fluids.fixed_acid_mmol, abs=1e-8,
        )
        outcomes.append(fluids)
    anuric, normal = outcomes
    assert anuric.net_acid_excretion_mmol == 0.0
    assert anuric.base_meq < 700.0
    assert 1.0 < normal.renal_base_meq / normal.ecf_l < 3.0
    assert normal.base_meq > 700.0
    # Compensation for hypercapnia lowers chloride as bicarbonate rises; urine alone does not.
    assert anuric.concentration("chloride") == pytest.approx(104.0)
    assert 104.0 - normal.concentration("chloride") == pytest.approx(
        normal.renal_base_meq / normal.ecf_l, rel=0.15,
    )


@pytest.mark.parametrize("target", [20.0, 80.0])
def test_engine_respiratory_buffering_keeps_base_excess_within_human_bands(engine_factory, target):
    """Brackett 1965 and Arbus 1969 H+ bands, with the fluid and renal models active."""
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0), start=True,
    )
    state = engine.state
    initial_base_excess = state.base_excess
    settings = engine.vent.settings
    deadspace = engine.resp.vd_deadspace
    vt = deadspace + (settings.tv / 1000.0 - deadspace) * state.pa_co2 / target
    engine.set_vent_settings(rr=settings.rr, vt=vt, peep=settings.peep, ie="1:2", mode="VCV")
    for _ in range(3600):
        engine.step(1.0)
    expected_hydrogen = 0.77 * state.pa_co2 + 8.0 if target > 40.0 else 0.74 * state.pa_co2 + 10.4
    assert 10.0**(9.0 - state.ph) == pytest.approx(expected_hydrogen, abs=3.5 if target > 40.0 else 3.0)
    assert state.base_excess == pytest.approx(initial_base_excess, abs=0.7)


def test_calcium_binding_follows_ph_and_citrate_and_deficits_persist():
    fluids = FluidBalance(70.0, 4900.0, 1984.5)
    baseline = fluids.ionized_calcium(7.4)
    alkalotic = fluids.ionized_calcium(7.6)
    assert baseline == pytest.approx(1.2, abs=1e-9)
    # Thode's in vivo and in vitro log10 slopes were -0.17 and -0.22 per pH.
    slope = math.log10(alkalotic / baseline) / 0.2
    assert -0.26 < slope < -0.12
    assert fluids.ionized_calcium(7.2) > baseline > alkalotic
    # Bolan 2001 Fig. 2A: ionized calcium 1.272 - 0.251 x serum citrate during
    # plateletpheresis, as total calcium fell 3% by the highest citrate (1.9 mmol/L).
    total = fluids.amounts["calcium"]
    for citrate in (0.5, 1.0, 1.5, 2.0):
        fluids.citrate_mmol = citrate * fluids.citrate_volume_l
        fluids.amounts["calcium"] = total * (1.0 - 0.03 * citrate / 1.9)
        assert fluids.ionized_calcium(7.4) / baseline == pytest.approx(
            (1.272 - 0.251 * citrate) / 1.272, abs=0.03,
        )
    # Raised PTH did not restore total calcium within 90 min, so deficits are not replaced.
    depleted = fluids.amounts["calcium"]
    fluids.step(5400.0, ph=7.4, paco2=40.0, resting_paco2=40.0, renal_function=0.0, renal_perfusion=1.0,
                hepatic_function=1.0, oxygen_supply=1.0, epinephrine=0.0)
    assert fluids.amounts["calcium"] == depleted
    assert fluids.ionized_calcium(7.4) < baseline


def test_calcium_chloride_peaks_and_halves_as_in_parturients(engine_factory):
    """Ansari 2025: 1 g CaCl2 over 10 min after cesarean delivery raised ionized
    calcium by 0.39 mmol/L (0.38-0.42) at the end of infusion. Half the rise
    remained 29 min (23-40) after the start and 40% at 60 min in their model.
    """
    engine = engine_factory(sex="female", weight=80.0, height=165.0, age=34.0,
                            gestational_age_weeks=39.0, start=True)
    fluids = engine.hemo.fluids
    baseline = fluids.ionized_calcium(engine.state.ph)
    rise = []
    for second in range(3600):
        if second < 600 and second % 10 == 0:
            engine.give_calcium_chloride(1000.0 / 60.0)
        engine.step(1.0)
        rise.append(fluids.ionized_calcium(engine.state.ph) - baseline)
    peak = max(rise)
    assert 585 <= rise.index(peak) < 600
    assert 0.38 <= peak <= 0.42
    half_minutes = next(t for t in range(600, 3600) if rise[t] <= peak / 2) / 60.0
    assert 23.0 < half_minutes < 40.0
    assert rise[-1] / peak == pytest.approx(0.40, abs=0.05)


def test_prolonged_capillary_leak_and_bleeding_preserve_plasma_volume(patient):
    model = HemodynamicModel(patient)
    model.anaphylaxis_severity = 1.0
    for _ in range(3600):
        model.step(1.0, 0.0, 0.0, 0.0, pit=-2.0, paco2=40.0, pao2=95.0)
    assert model.red_cell_volume_ml / model.blood_volume <= 0.7
    model.add_volume(-200.0)
    assert model.fluids.plasma_l > 0.0
    assert model.fluids.plasma_l == pytest.approx(
        (model.blood_volume - model.red_cell_volume_ml) / 1000.0, abs=1e-11,
    )


def test_same_packed_cell_unit_carries_same_base_into_different_recipients(engine_factory):
    gains = []
    for hb, hct in ((13.5, 0.405), (10.0, 0.4)):
        engine = engine_factory(baseline_hb=hb, baseline_hct=hct)
        before = engine.hemo.fluids.base_meq
        engine.hemo.add_volume(300.0, hematocrit=0.55, label="blood")
        gains.append(engine.hemo.fluids.base_meq - before)
    assert gains[0] == pytest.approx(gains[1], abs=1e-10)


def test_saline_loading_and_bleeding_preserve_carbon_and_charge(engine_factory):
    engine = engine_factory()
    model = engine.resp
    transport = model.gas_transport
    carbon, base = transport.co2_l, transport.base_meq
    engine.hemo.add_volume(1000.0, label="saline")
    model.update_fluid_balance(engine.hemo.hb_conc, engine.hemo.get_hematocrit(), engine.hemo.blood_volume)
    assert transport.co2_l == pytest.approx(carbon, abs=1e-10)
    assert transport.base_meq == pytest.approx(base, abs=1e-10)
    blood_carbon = sum(v * c for v, c in zip(transport.volumes, transport.co2, strict=True))
    lost_fraction = 0.4 / sum(transport.volumes)
    engine.hemo.add_volume(-400.0)
    model.update_fluid_balance(engine.hemo.hb_conc, engine.hemo.get_hematocrit(), engine.hemo.blood_volume)
    assert transport.co2_l == pytest.approx(carbon - lost_fraction * blood_carbon * 0.022414, abs=1e-10)
    assert transport.base_meq == pytest.approx(engine.hemo.fluids.base_meq, abs=1e-10)
