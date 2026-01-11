from anasim.core.state import SimulationConfig
from anasim.patient.pd_models import LOCModel, TOFModel


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
    base = tof.compute_tof_from_ce(ce_roc=1.0, mac_sevo=0.0, mac_n2o=0.0)
    with_n2o = tof.compute_tof_from_ce(ce_roc=1.0, mac_sevo=0.0, mac_n2o=0.7)
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
