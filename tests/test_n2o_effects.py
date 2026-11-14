import copy
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
    model = ClinicalResponseModel(age=40.0)
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


def test_n2o_second_gas_concentrating_effect_and_fink_diffusion_hypoxia(engine_factory, advance_time):
    """Epstein 1964 & Fink 1955: high-flow N2O accelerates early sevoflurane wash-in,
    and switching from 75% N2O to room air transiently dilutes alveolar O2 unless O2 is given.
    """
    sevo_air = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    sevo_n2o = copy.deepcopy(sevo_air)
    ctrl_air = copy.deepcopy(sevo_air)
    wash_air = copy.deepcopy(sevo_air)
    for model, fgf in ((sevo_air, (2.0, 6.0, 0.0)), (sevo_n2o, (2.0, 0.0, 6.0))):
        model.set_airway_mode("ETT")
        model.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
        model.set_vent_power(True)
        model.set_fgf(fgf[0], fgf[1], n2o_l_min=fgf[2])
        model.set_vaporizer(2.0)
        advance_time(model, 90.0, dt=1.0)
    assert sevo_n2o.state.et_sevo > sevo_air.state.et_sevo + 0.08

    for model, fgf in ((wash_air, (2.0, 0.0, 6.0)), (ctrl_air, (0.5, 7.5, 0.0))):
        model.set_airway_mode("ETT")
        model.set_vent_settings(rr=10, vt=0.45, peep=5.0, ie="1:2", mode="VCV")
        model.set_vent_power(True)
        model.set_fgf(fgf[0], fgf[1], n2o_l_min=fgf[2])
        advance_time(model, 600.0, dt=2.0)
    wash_o2 = copy.deepcopy(wash_air)
    for model in (wash_air, ctrl_air):
        model.set_auto_laryngospasm(False)
        model.set_airway_mode("None")
    wash_o2.set_fgf(10.0, 0.0, n2o_l_min=0.0)
    for model in (wash_air, ctrl_air, wash_o2):
        advance_time(model, 75.0, dt=1.0)
    assert wash_air.state.pao2 < ctrl_air.state.pao2 - 3.0 and wash_air.state.pao2 < 90.0
    assert wash_o2.state.pao2 > 220.0
