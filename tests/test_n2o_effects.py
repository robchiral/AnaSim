import pytest

from anasim.core.state import SUPPORTED_MODEL_OPTIONS, SimulationConfig
from anasim.patient.pd.anesthesia import SEVO_BIS_AT_1_MAC, BISModel


def test_n2o_washin_deepens_hypnosis_and_neuromuscular_block(engine_factory, advance_time):
    """Fiset 1991: N2O potentiates a partial rocuronium block."""
    endpoints = []
    for n2o_flow in (0.0, 8.0):
        engine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=123), start=True)
        engine.set_airway_mode("Mask")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        engine.set_vent_power(True)
        advance_time(engine, 30.0, dt=0.5)
        engine.give_drug_bolus("roc", 0.15 * engine.patient.weight)
        engine.set_fgf(2.0, 0.0, n2o_l_min=n2o_flow)
        advance_time(engine, 300.0, dt=0.5)
        endpoints.append((engine.state.mac_n2o, engine.state.loc, engine.state.tof))

    control, with_n2o = endpoints
    assert with_n2o[0] > 0.2
    assert with_n2o[1] > control[1] + 0.05
    assert 10.0 < control[2] < 90.0
    assert with_n2o[2] < control[2] - 10.0


def test_sevoflurane_deepens_bis_continuously_for_every_bis_model(patient):
    """Adding volatile to a propofol-remifentanil state must never raise BIS."""
    for model_name in SUPPORTED_MODEL_OPTIONS["bis_model"]:
        bis = BISModel(patient, model_name=model_name)
        values = [bis.compute_bis(3.0, 3.0, mac_sevo=mac) for mac in (0.0, 1e-4, 0.01, 0.02, 0.5, 1.0)]
        assert values[1] == pytest.approx(values[0], abs=0.05)
        assert all(later <= earlier for earlier, later in zip(values, values[1:]))
        assert bis.compute_bis(0.0, 0.0, mac_sevo=1.0) == pytest.approx(SEVO_BIS_AT_1_MAC, abs=0.5)
