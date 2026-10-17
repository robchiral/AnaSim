"""Carbon conservation and acute respiratory acid-base behavior."""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from anasim.physiology.respiration import RespiratoryModel


def _carbon_l(model):
    gas = model.frc * model._gas_capacity_per_l * model.state.p_alveolar_co2
    return gas + model.gas_transport.co2_l


def test_apnea_and_ventilation_conserve_carbon_and_change_buffering(patient):
    model = RespiratoryModel(patient)
    model.step(0.0, ce_prop=0.0, ce_remi=0.0, airway_patency=0.0)
    initial_carbon = _carbon_l(model)
    initial_ph, initial_bicarbonate = model.state.ph, model.state.hco3
    initial_base = model.gas_transport.base_meq
    initial_oxygen = (model.frc * model._gas_capacity_per_l * model.state.p_alveolar_o2
                      + model.gas_transport.oxygen_l)
    for _ in range(120):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                   airway_patency=0.0, unconscious=1.0)
    acidotic_ph = model.state.ph
    assert acidotic_ph < initial_ph - 0.03
    assert model.state.pa_co2 > 45.0
    assert model.state.hco3 > initial_bicarbonate
    # Closed airway and falling oxygenation exercise Haldane redistribution.
    assert model.gas_transport.exhaled_l == 0.0
    assert _carbon_l(model) - initial_carbon == pytest.approx(model.gas_transport.produced_l, abs=1e-7)

    for _ in range(300):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                   fio2=1.0, mech_rr=20.0, mech_vt_l=0.6, unconscious=1.0)
    assert model.state.ph > acidotic_ph
    assert model.gas_transport.exhaled_l > 0.0
    assert _carbon_l(model) - initial_carbon == pytest.approx(
        model.gas_transport.produced_l - model.gas_transport.exhaled_l, abs=1e-7,
    )
    equilibrium = model.chemistry.equilibrate(
        ph=model.state.ph, pco2_mmhg=model.state.pa_co2, oxygen_saturation=model.state.sao2 / 100.0,
    )
    assert equilibrium.buffer_base_meq_l == pytest.approx(model.buffer_base, abs=1e-7)
    assert model.gas_transport.base_meq == pytest.approx(initial_base, abs=1e-7)
    final_oxygen = (model.frc * model._gas_capacity_per_l * model.state.p_alveolar_o2
                    + model.gas_transport.oxygen_l)
    assert final_oxygen - initial_oxygen == pytest.approx(
        model.gas_transport.oxygen_from_air_l - model.gas_transport.consumed_o2_l, abs=1e-8,
    )


def _controlled(patient, cardiac_output=3.0):
    model = RespiratoryModel(patient)
    model.frc = 2.2
    model.equilibrate_oxygen(1.0)
    model.equilibrate_blood_gases(cardiac_output, model.vco2 * 0.8 / 1000.0)
    settings = dict(
        ce_prop=0.0, ce_remi=0.0, ce_roc=10.0, unconscious=1.0,
        fio2=1.0, cardiac_output=cardiac_output, metabolic_factor=0.8,
        measured_breaths=True, mech_rr=12.0,
        mech_vt_l=model.vd_deadspace + model.co2_constant * model.vco2 * 0.8 / 40.0 / 12.0,
    )
    for _ in range(300):
        model.step(1.0, **settings)
    return model, settings


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_ventilated_arrest_washes_out_lung_co2_and_isolates_arterial_blood(patient, dt):
    model, settings = _controlled(patient, cardiac_output=5.0)
    initial_pressure, initial_ph = model.state.pa_co2, model.state.ph
    initial_tissue = model.gas_transport.tissue_co2_l
    initial_carbon = _carbon_l(model)
    produced = model.gas_transport.produced_l
    exhaled = model.gas_transport.exhaled_l
    settings["cardiac_output"] = 0.0
    for _ in range(round(60.0 / dt)):
        model.step(dt, **settings)
    assert model.state.etco2 < 5.0
    for _ in range(840):
        model.step(1.0, **settings)
    assert model.state.pa_co2 == pytest.approx(initial_pressure, abs=1e-7)
    assert model.state.ph == pytest.approx(initial_ph, abs=1e-9)
    assert model.gas_transport.tissue_co2_l - initial_tissue == pytest.approx(15.0 * model.vco2 * 0.8 / 1000.0, abs=1e-9)
    assert _carbon_l(model) - initial_carbon == pytest.approx(
        model.gas_transport.produced_l - produced - model.gas_transport.exhaled_l + exhaled, abs=1e-7,
    )
    retained = model.gas_transport.tissue_co2_l
    settings["cardiac_output"] = 5.0
    for _ in range(60):
        model.step(1.0, **settings)
    assert model.state.etco2 > 20.0
    assert model.state.pa_co2 > initial_pressure + 5.0
    assert model.gas_transport.tissue_co2_l < retained


def test_low_flow_separates_venous_and_arterial_acid_base_status(patient):
    """Weil 1986: during CPR, mixed venous blood is more hypercarbic and acidotic than arterial blood."""
    model, settings = _controlled(patient, cardiac_output=5.0)
    settings["cardiac_output"] = 0.5
    for _ in range(300):
        model.step(1.0, **settings)
    assert model.state.pv_co2 > model.state.pa_co2 + 15.0
    assert model.state.venous_ph < model.state.ph - 0.1
    assert model.state.etco2 < 20.0


@pytest.mark.parametrize("target", [20.0, 80.0])
def test_whole_body_respiratory_buffering_matches_human_significance_bands(patient, target):
    """Brackett 1965 and Arbus 1969 report H+ bands about 7 and 6 nmol/L wide.

    Compare at the attained PaCO2, since slow tissue storage delays the
    ventilation target. Both bands describe acute extrarenal responses.
    """
    model, settings = _controlled(patient, cardiac_output=5.0)
    initial_carbon, initial_base = _carbon_l(model), model.gas_transport.base_meq
    produced, exhaled = model.gas_transport.produced_l, model.gas_transport.exhaled_l
    settings["mech_vt_l"] = model.vd_deadspace + model.co2_constant * model.vco2 * 0.8 / target / 12.0
    for _ in range(7200):
        model.step(1.0, **settings)
    state = model.state
    expected_hydrogen = 0.77 * state.pa_co2 + 8.0 if target > 40.0 else 0.74 * state.pa_co2 + 10.4
    assert 10.0**(9.0 - state.ph) == pytest.approx(expected_hydrogen, abs=3.5 if target > 40.0 else 3.0)
    if target > 40.0:
        assert 26.0 < state.hco3 < 29.5
    else:
        assert 17.7 < state.hco3 < 22.8
    assert model.gas_transport.base_meq == pytest.approx(initial_base, abs=1e-7)
    assert _carbon_l(model) - initial_carbon == pytest.approx(
        model.gas_transport.produced_l - produced - model.gas_transport.exhaled_l + exhaled, abs=1e-7,
    )


def test_preceding_hypocapnia_and_prolonged_apnea_depend_on_stored_co2(patient):
    """Gentz 1998 and Eger 1961 used brief and sustained preceding hyperventilation."""
    first_minutes = []
    for preparation in (0, 600, 4200):
        model, settings = _controlled(patient)
        if preparation:
            settings["mech_vt_l"] = model.vd_deadspace + 1.6 * (settings["mech_vt_l"] - model.vd_deadspace)
            for _ in range(preparation):
                model.step(1.0, **settings)
        starting_co2 = model.state.pa_co2
        settings.update(mech_rr=0.0, mech_vt_l=0.0)
        for _ in range(60):
            model.step(1.0, **settings)
        first_minutes.append(model.state.pa_co2 - starting_co2)
        if preparation == 0:
            for _ in range(2640):
                model.step(1.0, **settings)
            # Frumin 1959 Table 1 reports average rises of 2.7-4.9 mmHg/min
            # over whole prolonged apneas.
            assert 2.7 < (model.state.pa_co2 - starting_co2) / 45.0 < 4.9
    assert first_minutes[1] > first_minutes[0] + 0.5
    assert first_minutes[2] < first_minutes[0] - 0.5


def test_ventilation_change_retains_a_slow_asymmetric_co2_elimination_tail(patient):
    """Taskar 1995: longer fits have longer tau; decreasing VT has the slower tail.

    The study's cardiac output and CO2 production are unknown, so only relative
    time constants are compared.
    """
    constants = []
    for change in (0.9, 1.1):
        model, settings = _controlled(patient)
        settings["mech_vt_l"] *= change
        times, elimination = [], []
        for second in range(1, 1251):
            before = model.gas_transport.exhaled_l
            model.step(1.0, **settings)
            times.append(second / 60.0)
            elimination.append((model.gas_transport.exhaled_l - before) * 60.0 - model.vco2 * 0.8 / 1000.0)
        times_array, elimination_array = np.array(times), np.array(elimination)
        fitted = []
        # Exclude the first ten breaths, as Taskar did.
        start = 10.0 / settings["mech_rr"]
        for duration in (5.0, 20.0):
            selected = (times_array >= start) & (times_array <= start + duration)

            def exponential(time, amplitude, tau):
                return amplitude * np.exp(-time / tau)

            parameters, _ = curve_fit(
                exponential, times_array[selected] - start, elimination_array[selected],
                p0=(elimination_array[selected][0], 15.0), bounds=([-1.0, 0.01], [1.0, 1000.0]),
            )
            fitted.append(parameters[1])
        assert 5.0 < fitted[0] < fitted[1] < 60.0
        constants.append(fitted[1])
    assert constants[0] > constants[1]


def test_zero_flow_retains_metabolic_co2_in_tissues(patient):
    model = RespiratoryModel(patient)
    initial_tissue = model.gas_transport.tissue_co2_l
    for _ in range(60):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, cardiac_output=0.0,
                   ce_roc=10.0, airway_patency=0.0, unconscious=1.0)
    assert model.gas_transport.tissue_co2_l - initial_tissue == pytest.approx(model.vco2 / 1000.0, abs=1e-10)


@pytest.mark.parametrize("shunt", [0.0, 0.95])
def test_fast_blood_turnover_preserves_nonnegative_stores_and_gas_balance(patient, shunt):
    model = RespiratoryModel(patient)
    settings = dict(ce_prop=0.0, ce_remi=0.0, ce_roc=10.0, unconscious=1.0,
                    cardiac_output=30.0, blood_volume_ml=1000.0, shunt_fraction=shunt,
                    fio2=0.0, mech_rr=12.0, mech_vt_l=0.5)
    # Deliberately stress transport with very short vascular transit and FiO2 0.
    # Resize before taking the conservation reference.
    model.step(0.0, **settings)
    transport = model.gas_transport
    carbon, base = _carbon_l(model), transport.base_meq
    oxygen = model.frc * model._gas_capacity_per_l * model.state.p_alveolar_o2 + transport.oxygen_l
    for _ in range(90):
        model.step(1.0, **settings)
        assert min(transport.co2 + transport.oxygen + transport.tissue_co2) >= 0.0
    assert _carbon_l(model) - carbon == pytest.approx(transport.produced_l - transport.exhaled_l, abs=1e-7)
    assert transport.base_meq == pytest.approx(base, abs=1e-7)
    final_oxygen = model.frc * model._gas_capacity_per_l * model.state.p_alveolar_o2 + transport.oxygen_l
    assert final_oxygen - oxygen == pytest.approx(
        transport.oxygen_from_air_l - transport.consumed_o2_l, abs=1e-8,
    )


def test_apneic_rise_matches_lyons_short_term_data(patient):
    """Lyons 2024 no-oxygen arm: 1.39 +/- 0.39 kPa, then 0.34 +/- 0.08 kPa/min.

    FRC 2.2 L follows Magor-Elliott 2021's comparison. Q 3 L/min and 80%
    basal metabolism are simulator anesthesia assumptions; Lyons reports neither.
    """
    model = RespiratoryModel(patient)
    model.frc = 2.2
    model.equilibrate_oxygen(1.0)
    model.equilibrate_blood_gases(3.0, model.vco2 * 0.8 / 1000.0)
    first_minute = 0.0
    for elapsed in range(240):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, cardiac_output=3.0,
                   ce_roc=10.0, fio2=1.0, metabolic_factor=0.8, unconscious=1.0)
        if elapsed == 59:
            first_minute = model.state.pa_co2
    assert first_minute - 40.0 == pytest.approx(1.39 * 7.5006, abs=0.39 * 7.5006)
    assert (model.state.pa_co2 - first_minute) / 3.0 == pytest.approx(0.34 * 7.5006, abs=0.08 * 7.5006)


def test_public_blood_gas_state_tracks_live_chemistry(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.give_drug_bolus("roc", engine.patient.weight)
    for _ in range(60):
        engine.step(1.0)
    assert engine.state.ph < 7.4
    assert engine.state.hco3 > 24.5
    assert engine.state.ph == engine.resp.state.ph
    assert engine.state.hco3 == engine.resp.state.hco3
