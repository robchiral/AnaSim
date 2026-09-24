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
