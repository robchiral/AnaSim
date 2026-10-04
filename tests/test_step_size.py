"""Results should not depend on the outer simulation step."""

import pytest

from anasim.patient.pd.anesthesia import BISModel


def test_bis_processing_delay_runs_on_simulation_time(patient):
    outputs = []
    for dt in (0.01, 0.1):
        bis = BISModel(patient, model_name="Eleveld")
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
