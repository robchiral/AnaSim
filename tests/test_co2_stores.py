"""Carbon conservation and acute respiratory acid-base behavior."""

import pytest

from anasim.physiology.respiration import RespiratoryModel


def _carbon_l(model):
    state = model.state
    chemical = model.chemistry.equilibrate(
        ph=state.ph, pco2_mmhg=state.pa_co2, oxygen_saturation=state.sao2 / 100.0,
    )
    gas = model.frc * model._gas_capacity_per_l * state.p_alveolar_co2
    blood = 0.2 * model.baseline_blood_volume_ml / 1000.0 * 0.022414 * chemical.total_co2_mmol_l
    return gas + blood + model.co2_stores.tissue_co2_l


def test_apnea_and_ventilation_conserve_carbon_and_change_buffering(patient):
    model = RespiratoryModel(patient)
    model.step(0.0, ce_prop=0.0, ce_remi=0.0, airway_patency=0.0)
    initial_carbon = _carbon_l(model)
    initial_ph, initial_bicarbonate = model.state.ph, model.state.hco3
    for _ in range(120):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                   airway_patency=0.0, unconscious=1.0)
    acidotic_ph = model.state.ph
    assert acidotic_ph < initial_ph - 0.05
    assert model.state.hco3 > initial_bicarbonate
    # Closed airway and falling oxygenation exercise Haldane redistribution.
    assert model.co2_stores.exhaled_l == 0.0
    assert _carbon_l(model) - initial_carbon == pytest.approx(model.co2_stores.produced_l, abs=1e-7)

    for _ in range(300):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                   fio2=1.0, mech_rr=20.0, mech_vt_l=0.6, unconscious=1.0)
    assert model.state.ph > acidotic_ph
    assert model.co2_stores.exhaled_l > 0.0
    assert _carbon_l(model) - initial_carbon == pytest.approx(
        model.co2_stores.produced_l - model.co2_stores.exhaled_l, abs=1e-7,
    )
    equilibrium = model.chemistry.equilibrate(
        ph=model.state.ph, pco2_mmhg=model.state.pa_co2, oxygen_saturation=model.state.sao2 / 100.0,
    )
    assert equilibrium.buffer_base_meq_l == pytest.approx(model.buffer_base, abs=1e-7)


def test_zero_flow_retains_metabolic_co2_in_tissues(patient):
    model = RespiratoryModel(patient)
    initial_tissue = model.co2_stores.tissue_co2_l
    for _ in range(60):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, cardiac_output=0.0,
                   ce_roc=10.0, airway_patency=0.0, unconscious=1.0)
    assert model.co2_stores.tissue_co2_l - initial_tissue == pytest.approx(model.vco2 / 1000.0, abs=1e-10)


def test_apneic_rise_matches_short_term_calibration(patient):
    """Stock 1989: first minute about 12 mmHg, then about 3.4 mmHg/min.

    FRC 2.2 L follows Magor-Elliott 2021's comparison. Q 3 L/min and 80%
    basal metabolism are simulator anesthesia assumptions. This is a
    calibration check, not independent validation of the reduced model.
    """
    model = RespiratoryModel(patient)
    model.frc = 2.2
    model.equilibrate_oxygen(1.0)
    model.co2_stores.equilibrate(40.0, 3.0, model.vco2 * 0.8 / 1000.0)
    first_minute = 0.0
    for elapsed in range(300):
        model.step(1.0, ce_prop=0.0, ce_remi=0.0, cardiac_output=3.0,
                   ce_roc=10.0, fio2=1.0, metabolic_factor=0.8, unconscious=1.0)
        if elapsed == 59:
            first_minute = model.state.pa_co2
    assert first_minute - 40.0 == pytest.approx(12.0, abs=2.0)
    assert (model.state.pa_co2 - first_minute) / 4.0 == pytest.approx(3.4, abs=0.6)


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
