"""Results should not depend on the outer simulation step."""

import copy

import numpy as np
import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.pd.anesthesia import BISModel


def test_bis_processing_delay_runs_on_simulation_time(patient):
    outputs = []
    for dt in (0.01, 0.1):
        bis = BISModel(patient)
        bis.initialize(93.0)
        trace = []
        for seconds in (10.0, 35.0, 75.0):
            for _ in range(round(seconds / dt)):
                output = bis.step(dt, 6.0)
            trace.append(output)
        outputs.append(trace)
    # The initial reading is delayed, then responds to sustained hypnosis.
    assert outputs[0][0] == outputs[1][0] == pytest.approx(93.0)
    assert 20.0 < outputs[0][-1] < 40.0
    assert outputs[1] == pytest.approx(outputs[0], abs=0.3)


def test_pressure_control_gas_exchange_does_not_depend_on_step_size(engine_factory):
    outputs = []
    for dt in (0.1, 1.0):
        engine = engine_factory(start=True)
        engine.set_airway_mode("ETT")
        engine.set_vent_settings(rr=12, vt=0.5, peep=5.0, ie="1:2", mode="PCV", p_insp=15.0)
        engine.set_vent_power(True)
        engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
        for _ in range(round(180.0 / dt)):
            engine.step(dt)
        outputs.append(engine.state)
    fine, coarse = outputs
    assert coarse.vt == pytest.approx(fine.vt, abs=1e-6)
    assert coarse.va == pytest.approx(fine.va, abs=1e-8)
    assert coarse.pa_co2 == pytest.approx(fine.pa_co2, abs=0.1)
    assert coarse.pao2 == pytest.approx(fine.pao2, abs=1.0)


def test_tci_maintenance_concentrations_match_across_step_sizes(engine_factory):
    """Effect-site TCI maintenance holds identical concentrations across dt=0.05, 0.1, and 1.0 s."""
    traces = []
    for dt in (0.05, 0.1, 1.0):
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", tci_enabled=True, dt=dt, rng_seed=1),
            start=True,
        )
        for _ in range(round(120.0 / dt)):
            engine.step(dt)
        traces.append((engine.state.propofol_ce, engine.state.remi_ce))
    for trace in traces[1:]:
        assert trace == pytest.approx(traces[0], rel=0.01)


def test_tci_bolus_sugammadex_coupling_and_bis_display_noise_are_step_size_invariant(engine_factory):
    """TCI induction, sugammadex reversal coupling, and BIS display noise variance remain consistent across dt."""
    # TCI induction mass/time conservation across dt=0.1 vs 1.0 s.
    tci_fine = engine_factory(
        config=SimulationConfig(mode="awake", tci_enabled=True, dt=0.1, rng_seed=1), start=True
    )
    tci_fine.enable_tci("propofol", 3.0)
    tci_coarse = copy.deepcopy(tci_fine)
    for _ in range(100):
        tci_fine.step(0.1)
    for _ in range(10):
        tci_coarse.step(1.0)
    assert tci_coarse.state.propofol_ce == pytest.approx(tci_fine.state.propofol_ce, abs=0.05)

    # Sugammadex-rocuronium binding substeps keep TOF recovery invariant across dt=0.05 vs 1.0 s.
    sug_fine = engine_factory(config=SimulationConfig(mode="awake", rng_seed=1), start=True)
    sug_fine.give_drug_bolus("roc", 70.0)
    for _ in range(60):
        sug_fine.step(1.0)
    sug_coarse = copy.deepcopy(sug_fine)
    sug_fine.give_drug_bolus("sugammadex", 140.0)
    sug_coarse.give_drug_bolus("sugammadex", 140.0)
    for _ in range(600):
        sug_fine.step(0.05)
    for _ in range(30):
        sug_coarse.step(1.0)
    assert sug_coarse.state.tof == pytest.approx(sug_fine.state.tof, abs=2.0)

    # OU BIS display noise keeps stationary jitter SD consistent across dt=0.05 vs 1.0 s.
    sds = []
    for dt in (0.05, 1.0):
        eng = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=42), start=True)
        diffs = []
        for _ in range(round(300.0 / dt)):
            eng.step(dt)
            if eng.state.time > 60.0:
                diffs.append(eng.smooth_bis - eng.state.bis)
        sds.append(float(np.std(diffs)))
    assert 0.5 < sds[0] / sds[1] < 2.0
