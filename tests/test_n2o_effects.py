import itertools

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.pd.anesthesia import (
    SEVO_BIS_AT_1_MAC,
    BISModel,
    ClinicalResponseModel,
)


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


def test_sevoflurane_deepens_bis_continuously(patient):
    bis = BISModel(patient)
    values = [bis.compute_bis(3.0, mac_sevo=mac) for mac in (0.0, 1e-4, 0.01, 0.02, 0.5, 1.0)]
    assert values[1] == pytest.approx(values[0], abs=0.05)
    assert all(later <= earlier for earlier, later in itertools.pairwise(values))
    assert bis.compute_bis(0.0, mac_sevo=1.0) == pytest.approx(SEVO_BIS_AT_1_MAC, abs=0.5)


def test_n2o_and_sevoflurane_interaction_by_endpoint():
    """Katoh 1997: 45% N2O halves sevoflurane MAC-awake, less than additively.
    For noxious stimulation, MAC fractions of the two agents add.
    """
    model = ClinicalResponseModel()
    sevo_awake = 0.90 / model.mac_reference  # Sevoflurane MAC-awake in MAC

    def awake_loss(sevo_fraction, n2o_percent):
        return model.loss_of_response(0.0, 0.0, mac_sevo=sevo_fraction * sevo_awake, mac_n2o=n2o_percent / 104.0)

    assert awake_loss(0.5, 45.0) == pytest.approx(0.5, abs=0.05)
    # Half of each agent's MAC-awake falls short of the additive 50%.
    assert awake_loss(0.5, 0.61 * 104.0 / 2.0) < 0.4
    # The interaction fades with residual sevoflurane instead of switching off.
    assert awake_loss(1e-6, 60.0) == pytest.approx(awake_loss(0.0, 60.0), abs=1e-4)

    for n2o_mac in (0.3, 0.6):
        assert model.tolerance(0.0, 1.0, mac_sevo=0.8, mac_n2o=n2o_mac) == pytest.approx(
            model.tolerance(0.0, 1.0, mac_sevo=0.8 + n2o_mac)
        )
