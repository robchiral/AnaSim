import pytest

from anasim.core.state import SUPPORTED_MODEL_OPTIONS, SimulationConfig
from anasim.patient.pd.anesthesia import SEVO_BIS_AT_1_MAC, BISModel, LOCModel
from anasim.patient.pd.nmba import TOFModel


def test_loc_model_n2o_increases_probability():
    loc = LOCModel()
    baseline = loc.compute_probability(0.0, 0.0)
    n2o_only = loc.compute_probability(0.0, 0.0, mac_n2o=1.0)
    assert n2o_only > baseline + 0.3

    sevo_only = loc.compute_probability(0.0, 0.0, mac_sevo=0.5)
    sevo_n2o = loc.compute_probability(0.0, 0.0, mac_sevo=0.5, mac_n2o=0.5)
    assert sevo_n2o > sevo_only


def test_tof_model_n2o_potentiation(patient):
    tof = TOFModel(patient)
    base = tof.compute_tof_from_ce(1.0, mac_sevo=0.0, mac_n2o=0.0)
    with_n2o = tof.compute_tof_from_ce(1.0, mac_sevo=0.0, mac_n2o=0.7)
    assert with_n2o < base


def test_engine_n2o_increases_loc(engine_factory, advance_time):
    config = SimulationConfig(mode="awake", dt=0.5)
    engine = engine_factory(config=config, start=True)
    engine.set_airway_mode("Mask")
    engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")

    engine.set_fgf(2.0, 0.0, n2o_l_min=0.0)
    advance_time(engine, 60.0, dt=0.5)
    loc_base = engine.state.loc

    engine.set_fgf(2.0, 0.0, n2o_l_min=8.0)
    advance_time(engine, 300.0, dt=0.5)

    assert engine.state.mac_n2o > 0.2
    assert engine.state.loc > loc_base + 0.05


def test_sevoflurane_deepens_bis_continuously_for_every_bis_model(patient):
    """Adding volatile to a propofol-remifentanil state must never raise BIS."""
    for model_name in SUPPORTED_MODEL_OPTIONS["bis_model"]:
        bis = BISModel(patient, model_name=model_name)
        values = [bis.compute_bis(3.0, 3.0, mac_sevo=mac) for mac in (0.0, 1e-4, 0.01, 0.02, 0.5, 1.0)]
        assert values[1] == pytest.approx(values[0], abs=0.05)
        assert all(later <= earlier for earlier, later in zip(values, values[1:]))
        assert bis.compute_bis(0.0, 0.0, mac_sevo=1.0) == pytest.approx(SEVO_BIS_AT_1_MAC, abs=0.5)


def test_bis_processing_delay_is_independent_of_step_size(patient):
    outputs = []
    for dt in (0.01, 0.1):
        bis = BISModel(patient, model_name="Eleveld")
        bis.initialize(93.0)
        output = None
        for _ in range(round(10.0 / dt)):
            output = bis.step(dt, 6.0)
        outputs.append(output)
    assert outputs[0] == pytest.approx(93.0)
    assert outputs[1] == pytest.approx(93.0)
