import pytest

from anasim.core.state import SimulationConfig
from anasim.physiology.hemodynamics import HemodynamicModel


def test_awake_metabolic_acidosis_develops_respiratory_compensation(awake_engine):
    engine = awake_engine
    for step in range(1200):
        # Deliver the acid over two minutes so blood can distribute it.
        if step < 240:
            engine.resp.remove_buffer_base(250.0 / 240.0)
        engine.step(0.5)
    state = engine.state
    assert state.hco3 < 12
    # Winter's rule; compensation does not normalize pH.
    assert state.pa_co2 == pytest.approx(1.5 * state.hco3 + 8, abs=3)
    assert 7.1 < state.ph < 7.3
    assert state.rr > 15


def test_severe_acidemia_reduces_output_and_adrenergic_rescue(patient):
    outputs: dict[float, tuple[float, ...]] = {}
    for ph in (7.4, 7.0):
        model = HemodynamicModel(patient)
        for _ in range(600):
            state = model.step(0.1, 0, 0, 0, pit=-2, paco2=40, pao2=95, ph=ph)
        outputs[ph] = state.co, state.map
        for _ in range(600):
            state = model.step(0.1, 0, 0, 25, pit=-2, paco2=40, pao2=95, ph=ph)
        outputs[ph] += (state.map - outputs[ph][1],)
    assert outputs[7.0][0] < 0.8 * outputs[7.4][0]
    assert outputs[7.0][2] < outputs[7.4][2]


def test_acidemia_blunts_catecholamine_emax_while_vasopressin_remains_potent(
    engine_factory, advance_time, patient
):
    """Severe acidemia depresses catecholamine vascular Emax while vasopressin remains potent."""
    def svr_increment(ph: float, *, ce_nore: float = 0.0, ce_vaso: float = 0.0) -> float:
        base = HemodynamicModel(patient)
        treated = HemodynamicModel(patient)
        for _ in range(30):
            s_base = base.step(1.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0, ph=ph)
            s_treated = treated.step(1.0, 0.0, 0.0, ce_nore, -2.0, 40.0, 95.0, ph=ph, ce_vaso=ce_vaso)
        return s_treated.svr - s_base.svr

    assert svr_increment(6.85, ce_nore=200.0) < 0.70 * svr_increment(7.40, ce_nore=200.0)
    assert svr_increment(6.85, ce_vaso=40.0) > 0.85 * svr_increment(7.40, ce_vaso=40.0)

    vaso_eng = engine_factory(config=SimulationConfig(mode="steady_state", rng_seed=1), start=True)
    vaso_eng.start_sepsis()
    vaso_eng.resp.remove_buffer_base(220.0)
    advance_time(vaso_eng, 300.0, dt=1.0)
    assert vaso_eng.state.ph < 7.15
    map_pre = vaso_eng.state.map
    vaso_eng.set_drug_rate("vaso", 0.04)
    advance_time(vaso_eng, 600.0, dt=1.0)
    assert vaso_eng.state.map > map_pre + 10.0
