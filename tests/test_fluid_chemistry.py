"""Fluid-loading protocols, transfusion, and renal compensation."""

import copy
import math
from dataclasses import replace

import pytest

from anasim.core.state import SimulationConfig
from anasim.physiology.fluid_balance import FluidBalance
from anasim.physiology.fluids import CrystalloidKinetics
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
    acidemic = FluidBalance(70.0, 4900.0, 1984.5)
    ca0 = acidemic.amounts["calcium"]
    acidemic.step(
        1800.0, ph=7.10, paco2=80.0, resting_paco2=40.0, renal_function=1.0,
        renal_perfusion=1.0, hepatic_function=1.0, oxygen_supply=1.0, epinephrine=0.0,
    )
    assert acidemic.amounts["calcium"] == pytest.approx(ca0, abs=1e-9)
    baseline = fluids.ionized_calcium(7.4)
    alkalotic = fluids.ionized_calcium(7.6)
    assert baseline == pytest.approx(1.2, abs=1e-9)
    # Thode's in vivo and in vitro log10 slopes were -0.17 and -0.22 per pH.
    slope = math.log10(alkalotic / baseline) / 0.2
    assert -0.26 < slope < -0.12
    assert fluids.ionized_calcium(7.2) > baseline > alkalotic
    # Bolan 2001 Fig. 2A: ionized calcium 1.272 - 0.251 x serum citrate during
    # plateletpheresis, while 90-min total calcium fell 3% (highest group mean citrate 1.9 mmol/L).
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

    batched = engine_factory()
    flushed = copy.deepcopy(batched)
    batched.hemo.add_volume(2000.0, label="saline")
    batched.hemo.add_volume(-1500.0)
    batched.resp.update_fluid_balance(batched.hemo.hb_conc, batched.hemo.get_hematocrit(), batched.hemo.blood_volume)
    flushed.hemo.add_volume(2000.0, label="saline")
    flushed.resp.update_fluid_balance(flushed.hemo.hb_conc, flushed.hemo.get_hematocrit(), flushed.hemo.blood_volume)
    flushed.hemo.add_volume(-1500.0)
    flushed.resp.update_fluid_balance(flushed.hemo.hb_conc, flushed.hemo.get_hematocrit(), flushed.hemo.blood_volume)
    assert batched.resp.gas_transport.oxygen_l == pytest.approx(flushed.resp.gas_transport.oxygen_l, abs=1e-10)
    assert batched.resp.gas_transport.co2_l == pytest.approx(flushed.resp.gas_transport.co2_l, abs=1e-10)
    assert batched.resp.gas_transport.base_meq == pytest.approx(flushed.resp.gas_transport.base_meq, abs=1e-10)

    lactate = engine.lactate
    v0 = lactate.volume_l
    lactate.endogenous_mmol = 5.0 * v0
    lactate._base_removed_mmol = initial_removed = 4.1 * v0
    lactate.update_fluid_volume(v0, lost_plasma_l=2.0)
    assert lactate._base_removed_mmol == pytest.approx(initial_removed * (1.0 - 2.0 / v0), rel=1e-9)


def test_calcium_inotropy_citrate_hypotension_hyperkalemia_and_starling_refill(engine_factory, advance_time):
    """Denlinger 1975 & Lang 1988: CaCl2 raises SV and CO without vasoconstriction,
    reverses citrate hypocalcemic hypotension, and antagonizes hyperkalemic conduction block;
    unresuscitated hemorrhage recruits transcapillary interstitial refill with hemodilution.
    """
    ca = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    citrate = copy.deepcopy(ca)
    hyperk = copy.deepcopy(ca)
    co0, sv0, svr0 = ca.state.co, ca.state.sv, ca.state.svr
    ca.give_calcium_chloride(500.0)
    advance_time(ca, 60.0, dt=1.0)
    assert ca.state.co > 1.09 * co0 and ca.state.sv > 1.12 * sv0
    assert ca.state.svr <= 1.02 * svr0

    base_map = citrate.state.map
    citrate.hemo.fluids.citrate_mmol = 3.5 * citrate.hemo.fluids.citrate_volume_l
    advance_time(citrate, 60.0, dt=1.0)
    hypoca_map = citrate.state.map
    assert citrate.state.ionized_calcium < 0.75 and hypoca_map < 0.75 * base_map
    citrate.give_calcium_chloride(1500.0)
    advance_time(citrate, 60.0, dt=1.0)
    assert citrate.state.ionized_calcium > 1.10 and citrate.state.map > hypoca_map + 20.0

    hyperk.hemo.fluids.amounts["potassium"] = 7.4 * hyperk.hemo.fluids.ecf_l
    hyperk.step(1.0)
    slowed_hr = hyperk.state.hr
    assert hyperk.hemo.hyperkalemia_block > 0.25
    hyperk.give_calcium_chloride(2000.0)
    advance_time(hyperk, 30.0, dt=1.0)
    assert hyperk.hemo.hyperkalemia_block < 0.05 and hyperk.state.hr > slowed_hr + 8.0

    bleed = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1), start=True
    )
    bleed.hemo.config = replace(bleed.hemo.config, uop_ml_kg_hr=0.0)
    bleed.hemo.add_volume(-1000.0)
    bv_after, hb_after = bleed.hemo.blood_volume, bleed.hemo.hb_conc
    advance_time(bleed, 1800.0, dt=2.0)
    assert bleed.hemo.blood_volume > bv_after + 60.0
    assert bleed.hemo.hb_conc < hb_after - 0.15


def test_reversible_protein_free_water_shift_and_calcium_chloride_acid_load(
    engine_factory, advance_time
):
    """Reversing a protein-free water shift restores baseline plasma albumin,
    and exogenous CaCl2 clearance lowers strong-ion base_meq.
    """
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    fluids = engine.hemo.fluids
    albumin_0 = fluids.plasma_buffers.albumin_g_l
    fluids.shift_water(-1000.0, protein_fraction=0.0)
    fluids.shift_water(1000.0, protein_fraction=0.0)
    assert fluids.plasma_buffers.albumin_g_l == pytest.approx(albumin_0, abs=1e-9)

    base_0 = fluids.base_meq
    engine.give_calcium_chloride(2000.0)
    unmixed_0 = fluids.calcium_unmixed_mmol
    exo_0 = fluids.exogenous_calcium_mmol
    fluids.bleed(1000.0, hematocrit=0.40)
    assert fluids.calcium_unmixed_mmol < unmixed_0 * 0.92
    assert fluids.exogenous_calcium_mmol < exo_0
    advance_time(engine, 600.0, dt=2.0)
    assert fluids.base_meq < base_0 - 1.0


def test_rapid_saline_resuscitation_lowers_ph_monotonically(engine_factory, advance_time):
    """Rapid 0.9% saline lowers pH monotonically without transient alkalemia."""
    saline = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    bleed = copy.deepcopy(saline)
    ph_0 = saline.state.ph
    saline.hemo.add_volume(2000.0, label="saline")
    saline.step(1.0)
    ph_step1 = saline.state.ph
    advance_time(saline, 60.0, dt=1.0)
    assert saline.state.ph <= ph_step1 <= ph_0

    bleed.hemo.add_volume(-2200.0)
    bleed.hemo.add_volume(2500.0, label="saline")
    advance_time(bleed, 60.0, dt=1.0)
    assert math.isfinite(bleed.state.ph) and 7.05 < bleed.state.ph < ph_0


def test_lipid_partition_donor_prbc_oxygen_and_zero_flow_crystalloid_kinetics(
    engine_factory, advance_time
):
    """20% lipid emulsion stays extracellular without hyponatremia or excess third-spacing;
    stored pRBC transfusion adds venous-saturated donor O2; crystalloid third-space refill
    remains protein-free; crystalloid k21 return halts at zero circulation.
    """
    lipid = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        start=True,
    )
    cryst = copy.deepcopy(lipid)
    na_0 = lipid.state.sodium
    lipid.give_lipid_emulsion(105.0)
    lipid.set_lipid_infusion(17.5)
    advance_time(lipid, 180.0, dt=1.0)
    assert lipid.state.sodium == pytest.approx(na_0, abs=0.7)
    assert lipid.hemo.total_third_space_ml < 30.0

    prbc = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    o2_before = prbc.resp.gas_transport.oxygen_l
    hb_before = prbc.hemo.hb_mass
    prbc.hemo.add_volume(600.0, hematocrit=0.60, label="blood")
    prbc.resp.update_fluid_balance(prbc.hemo.hb_conc, prbc.hemo.get_hematocrit(), prbc.hemo.blood_volume)
    added_hb_g = prbc.hemo.hb_mass - hb_before
    assert prbc.resp.gas_transport.oxygen_l - o2_before == pytest.approx(
        added_hb_g * 1.34 * 0.60 / 1000.0, rel=0.02
    )

    alb_total_0 = cryst.hemo.fluids.plasma_albumin_g + cryst.hemo.fluids.interstitial_albumin_g
    cryst.give_fluid(2000.0)
    advance_time(cryst, 600.0, dt=2.0)
    assert cryst.hemo.fluids.plasma_albumin_g + cryst.hemo.fluids.interstitial_albumin_g == pytest.approx(
        alb_total_0, abs=1e-6
    )

    ck = CrystalloidKinetics(central_ml=0.0, interstitial_ml=500.0)
    ck.step(60.0, 0.0, 40.0, 1.0, 1000.0, circulation=0.0)
    assert ck.central_ml == 0.0 and ck.interstitial_ml == pytest.approx(500.0)


def test_citrate_clearance_and_massive_transfusion_calcium_and_acid_base_rebound(engine_factory):
    """Kramer 2003 & Kahn 1979: hepatic impairment slows citrate clearance (~48% of control),
    and during normovolemic massive transfusion ionized calcium reaches its nadir at peak
    transfusion rate and rebounds partially as transfusion slows and citrate yields bicarbonate.
    """
    outcomes = {}
    for label, hep in (("control", 1.0), ("cirrhotic", 0.31)):
        fine = FluidBalance(72.0, 5040.0, 2000.0)
        coarse = FluidBalance(72.0, 5040.0, 2000.0)
        infused_mmol = 0.5 * 72.0 * 2.0
        for fb, dt_s, n_steps in ((fine, 1.0, 7200), (coarse, 60.0, 120)):
            cit_per_step = (0.5 * 72.0 / 3600.0) * dt_s
            for _ in range(n_steps):
                fb.citrate_mmol += cit_per_step
                fb.step(
                    dt_s, ph=7.42, paco2=40.0, resting_paco2=40.0, renal_function=1.0,
                    renal_perfusion=1.0, hepatic_function=hep, oxygen_supply=1.0, epinephrine=0.0,
                )
        cmax_fine = fine.citrate_mmol / fine.citrate_volume_l
        cmax_coarse = coarse.citrate_mmol / coarse.citrate_volume_l
        assert cmax_fine == pytest.approx(cmax_coarse, abs=0.015)
        outcomes[label] = (cmax_fine, fine.citrate_mmol / infused_mmol, fine.base_meq)
    assert 0.70 < outcomes["control"][0] < 1.05
    assert outcomes["cirrhotic"][0] > 1.5 * outcomes["control"][0]
    assert outcomes["cirrhotic"][1] > 1.5 * outcomes["control"][1]
    assert outcomes["control"][2] > outcomes["cirrhotic"][2] + 30.0

    # Kahn 1979 pattern: rapid normovolemic replacement (60 mL/kg/h for 45 min, ~3.15 L)
    # followed by 45 min after transfusion ends at controlled normocapnia without CaCl2.
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", maintenance_fluid_ml_hr=0.0, rng_seed=1),
        age=52.0, weight=70.0, height=170.0, sex="male", baseline_hb=12.5, start=True,
    )
    base_ica = engine.state.ionized_calcium
    rate_ml_s = 60.0 * 70.0 / 3600.0
    for phase in ("transfusion", "post_transfusion"):
        for sec in range(2700):
            if phase == "transfusion":
                engine.hemo.add_volume(-rate_ml_s)
                engine.hemo.add_volume(rate_ml_s, hematocrit=0.55, label="blood")
            if sec % 30 == 0:
                settings = engine.vent.settings
                deadspace = engine.resp.vd_deadspace
                vt = deadspace + (settings.tv / 1000.0 - deadspace) * engine.state.pa_co2 / 40.0
                engine.set_vent_settings(rr=settings.rr, vt=vt, peep=settings.peep, ie="1:2", mode="VCV")
            engine.step(1.0)
        if phase == "transfusion":
            peak_cit = engine.hemo.fluids.citrate_mmol / engine.hemo.fluids.citrate_volume_l
            peak_ica = engine.state.ionized_calcium
            peak_be = engine.state.base_excess
    end_cit = engine.hemo.fluids.citrate_mmol / engine.hemo.fluids.citrate_volume_l
    assert 0.25 < peak_cit < 0.55 and end_cit < 0.30 * peak_cit
    assert peak_ica < base_ica - 0.15
    assert peak_ica + 0.02 < engine.state.ionized_calcium < base_ica - 0.05
    assert engine.state.base_excess > peak_be + 0.6


def test_intraoperative_crystalloid_loading_matches_ewaldsson_hahn_dilution_and_urine(engine_factory):
    """Ewaldsson and Hahn 2005: 25 mL/kg Ringer's over 30 min under volatile or propofol
    anesthesia expands plasma ~30% at 30 min and ~15-20% at 150 min, with ~7-11% urinary
    excretion and similar extravascular retention across anesthetics.
    """
    results = {}
    for mode, age, weight in (("volatile", 50.0, 68.5), ("propofol", 57.0, 68.7)):
        engine = engine_factory(
            config=SimulationConfig(mode="awake", maintenance_fluid_ml_hr=0.0, rng_seed=1),
            sex="female", age=age, weight=weight, height=167.4, baseline_hb=12.8, start=True,
        )
        engine.give_drug_bolus("propofol", 2.0 * weight)
        engine.give_drug_bolus("fentanyl", 150.0)
        engine.give_drug_bolus("roc", 0.55 * weight)
        if mode == "volatile":
            engine.set_vaporizer(2.0)
        else:
            engine.set_drug_rate("propofol", 9.5 * 1000.0 / 60.0)
        engine.set_airway_mode("ETT")
        engine.set_fgf(2.0, 2.0)
        engine.set_vent_settings(mode="VCV", rr=12, vt=0.45, peep=0, ie="1:2")
        engine.set_vent_power(True)
        for _ in range(600):
            engine.step(1.0)
        engine.set_fgf(0.4, 0.4)
        uop_0 = engine.state.urine_out_ml
        pv_0 = engine.hemo.fluids.plasma_l
        seq_0 = engine.hemo.total_third_space_ml
        na_0 = engine.state.sodium
        infused = 25.0 * weight
        dil_30 = 0.0
        for sec in range(9000):
            if sec in (1800, 4500):
                engine.give_drug_bolus("fentanyl", 125.0)
            if sec < 1800:
                engine.hemo.add_volume(infused / 1800.0, label="crystalloid")
            engine.hemo.add_volume(-100.0 / 9000.0)
            engine.step(1.0)
            if sec == 1799:
                dil_30 = (engine.hemo.fluids.plasma_l - pv_0) / pv_0
        dil_150 = (engine.hemo.fluids.plasma_l - pv_0) / pv_0
        uop_150 = engine.state.urine_out_ml - uop_0
        seq_150 = engine.hemo.total_third_space_ml - seq_0
        assert 0.25 < dil_30 < 0.38
        assert 0.12 < dil_150 < 0.22
        assert 0.06 * infused < uop_150 < 0.13 * infused
        assert engine.state.sodium == pytest.approx(na_0, abs=1.5)
        results[mode] = (uop_150, seq_150)
    assert results["volatile"][0] == pytest.approx(results["propofol"][0], abs=40.0)
    assert results["volatile"][1] == pytest.approx(results["propofol"][1], abs=45.0)
