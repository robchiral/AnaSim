import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.hemodynamics import HemodynamicModel


def _model(sepsis_severity: float = 0.0) -> HemodynamicModel:
    model = HemodynamicModel(Patient(age=40, weight=70, sex="male"))
    model.tde_hr = 0.0
    model.tde_sv = 0.0
    model.sepsis_severity = sepsis_severity
    return model


def _steady_exposure(seconds: int = 60, **ce):
    model = _model()
    base = model.step(1.0, 0, 0, 0, -2, 40, 95)
    for _ in range(seconds):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95, **ce)
    return base, state


def test_propofol_plasma_concentration_lowers_map():
    model = _model()
    base = model.step(1.0, 0, 0, 0, -2, 40, 95)
    for _ in range(600):
        state = model.step(1.0, 4.0, 0, 0, -2, 40, 95)
    drop = (base.map - state.map) / base.map
    assert 0.18 < drop < 0.45


def test_pressors_raise_svr_without_beta_effects():
    base, phenyl = _steady_exposure(ce_phenyl=100.0)
    assert phenyl.map > base.map + 10.0
    assert phenyl.svr > base.svr + 4.0
    assert phenyl.sv <= base.sv * 1.1

    base, vaso = _steady_exposure(ce_vaso=40.0)
    assert vaso.map > base.map + 5.0
    assert vaso.svr > base.svr * 1.1
    assert vaso.hr <= base.hr + 5.0


@pytest.mark.parametrize(("drug", "min_co_ratio"), [("ce_dobu", 1.1), ("ce_mil", 1.05)])
def test_inodilators_raise_output_and_lower_svr(drug, min_co_ratio):
    """Magnani 1977 (dobutamine); Baim 1983 (milrinone)."""
    base, state = _steady_exposure(**{drug: 200.0})
    assert state.co > base.co * min_co_ratio
    assert state.svr < base.svr * 0.9


def test_hypoxia_and_peep_raise_pvr():
    """Carlsson 1985 (hypoxic vasoconstriction); Koganov 1997 (PEEP)."""
    for pao2, peep in ((50, 5.0), (95, 15.0)):
        model = _model()
        base = model.step(1.0, 0, 0, 0, -2, 40, 95, peep_cmH2O=5.0)
        for _ in range(30):
            state = model.step(1.0, 0, 0, 0, -2, 40, pao2, peep_cmH2O=peep)
        assert state.pvr > base.pvr * 1.2


def test_pulmonary_transit_delays_lv_inflow():
    model = _model()
    base = model.step(1.0, 0, 0, 0, -2, 40, 95, peep_cmH2O=5.0)
    # Acute hypoxia and high PEEP raise PVR and drop RV output immediately.
    state = model.step(1.0, 0, 0, 0, -2, 40, 30, peep_cmH2O=20.0)
    assert state.lv_inflow < base.lv_inflow
    assert state.lv_inflow > state.rv_co


class TestSepsis:
    def test_capillary_leak_reduces_volume(self):
        model = _model(sepsis_severity=1.0)
        base_volume = model.blood_volume
        for _ in range(360):
            model.step(10.0, 0, 0, 0, -2, 40, 95)
        assert model.blood_volume < base_volume - 120.0

    def test_sepsis_blunts_norepinephrine_vasoconstriction(self):
        """Compare SVR rather than MAP, which also includes reflex HR (Bellissant 2000)."""

        def svr_response(sepsis_severity: float) -> float:
            model = _model(sepsis_severity)
            for _ in range(300):
                base = model.step(1.0, 0, 0, 0, -2, 40, 95)
            for _ in range(60):
                treated = model.step(1.0, 0, 0, 10.0, -2, 40, 95)
            return treated.svr - base.svr

        control = svr_response(0.0)
        septic = svr_response(1.0)
        assert control > 5.0
        assert septic < control * 0.8


class TestReflexesAndHypoxia:
    """Fast baroreflex and myocardial hypoxia in the integrated engine."""

    @pytest.fixture
    def tiva_without_pressor(self, engine_factory):
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.1, rng_seed=3),
            start=True,
        )
        engine.disable_tci("nore")
        for _ in range(3000):
            engine.step(0.1)
        return engine

    @staticmethod
    def _peak_response(engine, drug, amount, seconds=300):
        base_map, base_hr = engine.state.map, engine.state.hr
        engine.give_drug_bolus(drug, amount)
        peak_map, hr_at_peak = 0.0, 0.0
        for _ in range(int(seconds / 0.1)):
            engine.step(0.1)
            if engine.state.map - base_map > peak_map:
                peak_map, hr_at_peak = engine.state.map - base_map, engine.state.hr - base_hr
        return peak_map, hr_at_peak

    @staticmethod
    def _step_until_sao2_below(engine, sao2, max_seconds=600.0):
        for _ in range(int(max_seconds / 0.1)):
            if engine.state.sao2 <= sao2:
                return True
            engine.step(0.1)
        return False

    def test_phenylephrine_bolus_raises_map_with_reflex_bradycardia(self, tiva_without_pressor):
        """100-200 mcg under propofol-remifentanil: MAP +29.5 mmHg, HR -17 bpm (Meng 2011)."""
        peak_map, hr_change = self._peak_response(tiva_without_pressor, "phenyl", 100.0)
        assert 15.0 < peak_map < 35.0
        assert hr_change < -5.0

    def test_epinephrine_bolus_is_chronotropic(self, tiva_without_pressor):
        # Delivery check under TIVA; test_epinephrine.py covers peak sizes and timing.
        peak_map, hr_change = self._peak_response(tiva_without_pressor, "epi", 10.0)
        assert 5.0 < peak_map < 35.0
        assert 15.0 < hr_change < 40.0

    def test_severe_hypoxemia_slows_the_heart_and_oxygen_rescues(self, engine_factory):
        engine = engine_factory(config=SimulationConfig(dt=0.1, rng_seed=1), start=True)
        engine.give_drug_bolus("propofol", 140.0)
        engine.give_drug_bolus("roc", 42.0)
        assert self._step_until_sao2_below(engine, 60.0)
        hr_at_60 = engine.state.hr
        assert self._step_until_sao2_below(engine, 35.0)
        assert engine.state.hr < hr_at_60 - 10.0

        engine.set_airway_mode("Mask")
        engine.set_fgf(10.0, 0.0)
        engine.set_bag_mask_ventilation(True, 12.0, 0.5)
        depressed_map = engine.state.map
        for _ in range(1800):
            engine.step(0.1)
        assert engine.state.sao2 > 97.0
        assert engine.hemo.myocardial_hypoxia < 0.1
        assert engine.state.map > depressed_map + 15.0
