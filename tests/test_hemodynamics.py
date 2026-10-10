from dataclasses import replace

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.hemo_config import HemodynamicConfig
from anasim.physiology.hemodynamics import HemodynamicModel


def _model(sepsis_severity: float = 0.0) -> HemodynamicModel:
    model = HemodynamicModel(Patient(age=40, weight=70, sex="male"))
    model.sepsis_severity = sepsis_severity
    return model


def _steady_exposure(**ce: float):
    model = _model()
    base = model.step(1.0, 0, 0, 0, -2, 40, 95)
    for _ in range(60):
        state = model.step(1.0, 0, 0, 0, -2, 40, 95, **ce)
    return base, state


@pytest.mark.parametrize("renal_function", [1.0, 0.4])
def test_urine_output_tapers_with_pressure_and_preserves_fluid_balance(renal_function):
    patient = Patient(renal_function=renal_function)
    rates = []
    for pressure in (20.0, 40.0, 60.0, 65.0, 90.0):
        model = HemodynamicModel(patient)
        model._prev_map = pressure
        before = model.blood_volume
        model.step(1.0, 0, 0, 0, -2, 40, 95)
        urine = model.total_urine_out_ml
        assert before - model.blood_volume == pytest.approx(urine)
        rates.append(urine * 3600 / patient.weight)
    assert rates[0] == 0.0
    assert rates[0] < rates[1] < rates[2] < rates[3]
    assert 0.4 * renal_function < rates[2] < 0.5 * renal_function
    assert rates[3] == rates[4] == pytest.approx(0.5 * renal_function)


@pytest.mark.parametrize(
    ("propofol", "remifentanil", "expected"),
    [
        (4.0, 0.0, (
            (70.741, 56.054, 81.472), (63.226, 57.048, 79.708), (57.165, 61.043, 79.080),
        )),
        (3.0, 2.0, (
            (68.868, 55.476, 80.394), (58.095, 55.062, 75.639), (52.184, 57.608, 71.768),
        )),
        (4.0, 8.0, (
            (63.130, 53.739, 76.923), (44.359, 49.885, 64.213), (38.709, 50.714, 56.188),
        )),
    ],
)
def test_su_concentration_step_time_course(propofol, remifentanil, expected):
    """Su Eqs 6-17/Table 2, integrated independently with solve_ivp, at 1, 5, and 15 min.

    The baroreflex and fluid clearance are off. The baseline is the paper's
    35-year-old (HR 56, SV 82.2, TPR 0.016) without the anxiety transient.
    """
    patient = Patient(age=35, baseline_hr=56, baseline_map=56 * 82.2 * 0.016)
    config = replace(
        HemodynamicConfig(), ci_adult=56 * 82.2 / 1000 / patient.bsa,
        vol_clearance=0.0, baro_gain_brady=0.0, baro_gain_tachy=0.0,
    )
    model = HemodynamicModel(patient, config)
    observations = []
    for index in range(9000):
        state = model.step(0.1, propofol, remifentanil, 0.0, -2.0, 40.0, 85.0, peep_cmH2O=5.0)
        if index in (599, 2999, 8999):
            observations.append((state.map, state.hr, state.sv))
    for observed, reference in zip(observations, expected, strict=True):
        assert observed == pytest.approx(reference, abs=0.02)


def test_pressors_raise_svr_without_beta_effects():
    base, phenyl = _steady_exposure(ce_phenyl=100.0)
    assert phenyl.map > base.map + 10.0
    assert phenyl.svr > base.svr + 4.0
    # Reflex slowing can raise output SV through filling, without beta inotropy.
    assert phenyl.sv_star < base.sv_star
    assert phenyl.hr < base.hr
    assert phenyl.co < base.co

    base, vaso = _steady_exposure(ce_vaso=40.0)
    assert vaso.map > base.map + 5.0
    assert vaso.svr > base.svr * 1.1
    assert vaso.hr <= base.hr + 5.0


def test_steady_state_initialization_keeps_fast_and_slow_drug_effects(patient):
    model = HemodynamicModel(patient, replace(HemodynamicConfig(), vol_clearance=0.0))
    initialized = model.initialize_steady_state(3.53, 3.6, 10.0)
    assert initialized.map > 100.0
    initial_values = (initialized.map, initialized.hr, initialized.co)
    assert (model.state.map, model.state.hr, model.state.co) == pytest.approx(initial_values)
    for _ in range(60):
        state = model.step(1.0, 3.53, 3.6, 10.0, -2.0, 40.0, 95.0)
    assert (state.map, state.hr, state.co) == pytest.approx(initial_values, rel=0.01)


def test_thoracic_pressure_tracks_breathing_effort_and_applied_cpap(awake_engine):
    engine = awake_engine
    pressure = []
    for _ in range(150):
        engine.step(0.1)
        pressure.append(engine.state.pit)
    assert max(pressure) == pytest.approx(engine.hemo.config.pit_0, abs=0.01)
    assert min(pressure) < engine.hemo.config.pit_0 - 2.0

    engine.set_airway_mode("ETT")
    engine.set_fgf(6.0, 0.0)
    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    for _ in range(1200):
        engine.step(0.1)
    assert engine.resp_mech.effort.amplitude == 0.0
    assert engine.state.pit == pytest.approx(engine.hemo.config.pit_0, abs=0.1)

    engine.set_vent_settings(rr=0, vt=0, peep=5, ie="1:2", mode="CPAP")
    engine.set_vent_power(True)
    for _ in range(200):
        engine.step(0.1)
    assert engine.state.pit > engine.hemo.config.pit_0 + 1.8


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
    def tiva_engine(self, engine_factory):
        return engine_factory(
            config=SimulationConfig(mode="steady_state", maint_type="tiva", dt=0.1, rng_seed=3),
            start=True,
        )

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

    def test_phenylephrine_bolus_raises_map_with_reflex_bradycardia(self, tiva_engine):
        """100-200 mcg under propofol-remifentanil: MAP +29.5 mmHg, HR -17 bpm (Meng 2011)."""
        peak_map, hr_change = self._peak_response(tiva_engine, "phenyl", 100.0)
        assert 15.0 < peak_map < 35.0
        assert hr_change < -5.0

    def test_epinephrine_bolus_is_chronotropic(self, tiva_engine):
        # Delivery check under TIVA; test_epinephrine.py covers peak sizes and timing.
        peak_map, hr_change = self._peak_response(tiva_engine, "epi", 10.0)
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
        for _ in range(150):
            engine.step(0.1)
        assert engine.state.spo2_signal_valid
        assert engine.state.display_spo2 < 40.0
        assert engine.state.alarms["SpO2"]["low"]

        engine.set_airway_mode("Mask")
        engine.set_fgf(10.0, 0.0)
        engine.set_bag_mask_ventilation(True, 12.0, 0.5)
        depressed_map = engine.state.map
        for _ in range(1800):
            engine.step(0.1)
        assert engine.state.sao2 > 97.0
        assert engine.state.display_spo2 > 94.0
        assert "SpO2" not in engine.state.alarms
        assert engine.state.map > depressed_map + 15.0
