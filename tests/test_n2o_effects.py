import pytest

from anasim.core.state import SUPPORTED_MODEL_OPTIONS, SimulationConfig
from anasim.patient.pd.anesthesia import SEVO_BIS_AT_1_MAC, BISModel
from anasim.patient.pd.nmba import TOFModel


def test_n2o_potentiates_neuromuscular_block(patient):
    """Fiset 1991: N2O lowers the relaxant dose needed for block."""
    tof = TOFModel(patient)
    assert tof.compute_tof_from_ce(1.0, mac_n2o=0.7) < tof.compute_tof_from_ce(1.0)


def test_n2o_washin_raises_probability_of_unconsciousness(engine_factory, advance_time):
    engine = engine_factory(config=SimulationConfig(mode="awake", dt=0.5), start=True)
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
