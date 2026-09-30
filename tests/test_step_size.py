"""Results should not depend on the outer simulation step."""

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.pd.anesthesia import BISModel


@pytest.mark.parametrize("dt", [0.3, 1.0])
def test_tci_induction_does_not_depend_on_step_size(engine_factory, dt):
    def effect_site_after_one_minute(step):
        engine = engine_factory(config=SimulationConfig(mode="awake", dt=step), start=True)
        engine.set_airway_mode("Mask")
        engine.enable_tci("propofol", 4.0)
        for _ in range(round(60 / step)):
            engine.step(step)
        return engine.state.propofol_ce

    assert effect_site_after_one_minute(dt) == pytest.approx(
        effect_site_after_one_minute(0.1), rel=0.02
    )


def test_bis_processing_delay_runs_on_simulation_time(patient):
    outputs = []
    for dt in (0.01, 0.1):
        bis = BISModel(patient, model_name="Eleveld")
        bis.initialize(93.0)
        for _ in range(round(10.0 / dt)):
            output = bis.step(dt, 6.0)
        outputs.append(output)
    assert outputs == pytest.approx([93.0, 93.0])


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
