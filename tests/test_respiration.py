from dataclasses import replace
from typing import Any

import pytest

from anasim.core.state import SimulationConfig
from anasim.physiology.respiration import RespiratoryModel


def _breath(patient, paco2=None, **drugs):
    model = RespiratoryModel(patient)
    if paco2 is not None:
        model.state.p_alveolar_co2 = paco2
    drugs.setdefault("ce_prop", 0.0)
    drugs.setdefault("ce_remi", 0.0)
    # Concentration-response checks hold CO2 fixed while the controller settles.
    for _ in range(900):
        model.state.p_alveolar_co2 = 40.0 if paco2 is None else paco2
        state = model.step(1.0, **drugs)
    return state


def test_opioid_slows_rate_and_propofol_reduces_depth(patient):
    baseline = _breath(patient)

    remi = _breath(patient, ce_remi=1.0)
    remi_rr = remi.rr / baseline.rr
    assert 0.6 < remi_rr < 0.8
    assert remi.vt / baseline.vt >= remi_rr - 0.1
    assert _breath(patient, ce_remi=5.0).rr < 6

    propofol = _breath(patient, ce_prop=3.5)
    propofol_vt = propofol.vt / baseline.vt
    assert propofol_vt < 0.7
    assert propofol.rr / baseline.rr >= propofol_vt - 0.1


@pytest.mark.parametrize("ce_prop", [2.0, 3.0, 6.0])
def test_bouillon_spontaneous_steady_state_co2_and_alveolar_ventilation(patient, ce_prop):
    """Bouillon 2004 Eq. 18, with and without its CO2-production correction."""
    keep = 1.0 / (1.0 + (ce_prop / 1.33)**1.68)
    for production in (1.0, 0.7 + 0.3 * keep):
        model = RespiratoryModel(patient)
        for _ in range(3600):
            state = model.step(
                1.0, ce_prop=ce_prop, ce_remi=0.0, fio2=1.0,
                unconscious=1.0, metabolic_factor=production,
            )
        expected_co2 = 40.0 * (production / keep)**(1.0 / 5.37)
        expected_va = model.va_baseline * production**(4.37 / 5.37) * keep**(1.0 / 5.37)
        assert state.p_alveolar_co2 == pytest.approx(expected_co2, abs=0.5)
        assert state.va == pytest.approx(expected_va, rel=0.02)
        assert not state.apnea


def test_propofol_induction_causes_transient_hypoventilation_with_patent_airway(awake_engine):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.give_drug_bolus("propofol", 2.0 * engine.patient.weight)
    baseline_va = engine.resp.va_baseline
    low_ventilation_seconds = 0.0
    min_va = baseline_va
    min_bis = 100.0
    for _ in range(3600):
        engine.step(0.2)
        min_va = min(min_va, engine.resp.state.va)
        low_ventilation_seconds += 0.2 * (engine.resp.state.va < 0.5 * baseline_va)
        min_bis = min(min_bis, engine.state.bis)
    assert min_va < 0.3 * baseline_va
    assert 60.0 < low_ventilation_seconds < 360.0
    assert min_bis < 60.0
    assert engine.resp.state.rr > 8.0


def test_low_dose_propofol_promotes_transient_remifentanil_apnea(patient):
    """Olofsen 2010: low-dose propofol promotes apnea; accumulating CO2
    restores ventilation even while remifentanil remains present.
    """
    apnea = {}
    for propofol in (0.0, 1.0):
        model = RespiratoryModel(patient)
        apnea[propofol] = 0
        peak_co2 = 40.0
        for second in range(1500):
            remi = 4.0 if second < 600 else 0.0
            state = model.step(1.0, ce_prop=propofol, ce_remi=remi, fio2=1.0)
            apnea[propofol] += state.apnea
            peak_co2 = max(peak_co2, state.pa_co2)
            if second == 599:
                assert not state.apnea  # CO2 feedback restores breathing before drug withdrawal.
        assert not state.apnea and state.rr > 8.0
        assert peak_co2 > 50.0
    assert apnea[0.0] == 0
    assert 10 < apnea[1.0] < 300


def test_apneic_co2_accumulates_independently_of_baseline_breathing(patient_factory):
    outcomes = []
    for baseline_rr in (0.0, 12.0):
        model = RespiratoryModel(patient_factory(baseline_rr=baseline_rr))
        initial = model.state.p_alveolar_co2
        for _ in range(300):
            state = model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                               fio2=1.0, metabolic_factor=0.8)
        assert state.apnea and state.va == 0.0
        assert state.p_alveolar_co2 > initial + 15.0
        outcomes.append(state.p_alveolar_co2)
        for _ in range(2400):
            state = model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                               fio2=1.0, metabolic_factor=0.8)
        assert state.p_alveolar_co2 > 170.0  # Production continues throughout prolonged apnea.
    assert outcomes[0] == pytest.approx(outcomes[1], abs=0.1)


def _ventilated(patient, seconds, **inputs):
    model = RespiratoryModel(patient)
    for _ in range(round(seconds / 0.1)):
        state = model.step(
            0.1, ce_prop=0.0, ce_remi=0.0, mech_vent_mv=6.0, mech_rr=12.0, mech_vt_l=0.5,
            **inputs,
        )
    return state


def test_anemia_and_low_cardiac_output_do_not_lower_pao2(patient):
    """They reduce O2 content and delivery, not arterial O2 tension."""
    reference = _ventilated(patient, 60.0)
    for inputs in ({"hb_g_dl": 6.0}, {"cardiac_output": 0.1}):
        state = _ventilated(patient, 60.0, **inputs)
        assert state.p_arterial_o2 == pytest.approx(reference.p_arterial_o2, abs=2.0)
        assert state.sao2 == pytest.approx(reference.sao2, abs=0.5)


def test_low_cardiac_output_widens_the_paco2_etco2_gap(patient):
    normal = _ventilated(patient, 30.0, fio2=0.5)
    low_flow = _ventilated(patient, 30.0, fio2=0.5, cardiac_output=0.5)
    assert low_flow.etco2 < normal.etco2
    assert low_flow.pa_co2 - low_flow.etco2 > normal.pa_co2 - normal.etco2 + 3.0


def test_measured_shallow_breaths_widen_the_co2_gap(patient):
    """Measured breaths set the dead-space fraction, not the unassisted pattern."""
    state = RespiratoryModel(patient).step(
        0.0, ce_prop=0.0, ce_remi=0.0, measured_breaths=True,
        mech_rr=12.0, mech_vt_l=0.2, mech_vent_mv=2.4,
    )
    assert state.pa_co2 - state.etco2 > 10.0


def test_hemorrhage_with_atelectasis_retains_oxygen_from_ventilated_lung(engine_factory):
    engine = engine_factory(
        config=SimulationConfig(mode="steady_state", end_on_cardiac_arrest=False), start=True,
    )
    engine.set_fgf(10.0, 0.0)
    engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=0, ie="1:2")
    engine.hemo.add_volume(-0.4 * engine.hemo.blood_volume)
    for _ in range(240):
        engine.step(1.0)
    state, resp = engine.state, engine.resp
    assert state.co < 2.0
    assert resp.shunt_fraction > 0.25
    assert state.fio2 > 0.95
    # Shunt causes hypoxemia, while ventilated lung still oxygenates blood.
    assert 60.0 < state.sao2 < 90.0


def test_co2_drives_breathing_and_opioids_blunt_it(patient):
    """Babenco 2000: opioids shift the CO2 response right and flatten it."""
    assert _breath(patient, paco2=50.0).drive_central > _breath(patient, paco2=40.0).drive_central
    assert (
        _breath(patient, paco2=45.0, ce_remi=3.0).drive_central
        < _breath(patient, paco2=45.0).drive_central
    )
    baseline, mild = _breath(patient, paco2=40.0), _breath(patient, paco2=40.5)
    assert mild.rr == baseline.rr
    assert mild.va > baseline.va
    low = _breath(patient, paco2=25.0)
    assert low.rr == baseline.rr and not low.apnea
    assert low.vt < baseline.vt  # Awake hypocapnia weakens effort without imposing apnea.


def test_hyperventilation_stops_breathing_under_anesthesia_until_co2_recovers(patient):
    """Hickey 1971 motivates a 4-5 mmHg hypocapnic gap. AnaSim applies it
    below the nominal drug-free PaCO2 set point. Hold drug concentrations and consciousness
    fixed to isolate the CO2 threshold from PK and induction kinetics.
    """
    model = RespiratoryModel(patient)
    inputs: dict[str, Any] = dict(ce_prop=2.0, ce_remi=0.0, unconscious=1.0, fio2=1.0)
    for _ in range(9000):
        model.step(0.1, **inputs)
    assert model.state.rr > 8.0

    for _ in range(6000):
        model.step(0.1, **inputs, mech_rr=20.0, mech_vt_l=0.5, mech_vent_mv=10.0)
    assert model.state.apnea

    for _ in range(6000):
        model.step(0.1, **inputs)
        if not model.state.apnea:
            break
    assert not model.state.apnea
    assert 35.0 <= model.state.p_alveolar_co2 <= 36.5


def test_opioids_reduce_the_equilibrated_co2_response(patient):
    normocapnic = _breath(patient, paco2=40.0, ce_prop=3.5, ce_remi=4.0)
    hypercapnic = _breath(patient, paco2=70.0, ce_prop=3.5, ce_remi=4.0)

    assert normocapnic.apnea
    assert hypercapnic.apnea  # At these high concentrations, 70 mmHg is insufficient.
    # CO2 stimulates ventilation; opioids lower it at the same PaCO2.
    moderate = _breath(patient, paco2=70.0, ce_remi=1.0)
    assert moderate.mv > _breath(patient, paco2=40.0, ce_remi=1.0).mv
    assert moderate.mv < _breath(patient, paco2=70.0).mv


class TestOxygenStores:
    """Apneic desaturation follows lung and blood O2 stores (Benumof 1997)."""

    def test_blood_loss_shortens_apneic_desaturation(self, engine_factory):
        """Blood loss lowers the Hb mass that stores O2 before Hb concentration falls."""
        times = []
        for loss_fraction in (0.0, 0.3):
            engine = engine_factory(start=True)
            engine.hemo.config = replace(engine.hemo.config, vol_clearance=0.0)
            engine.hemo.add_volume(-loss_fraction * engine.hemo.blood_volume)
            engine.set_airway_obstruction(1.0)
            while engine.state.sao2 >= 80.0 and engine.state.time < 120.0:
                engine.step(1.0)
            times.append(engine.state.time)
        assert times[1] < times[0] - 5.0

    @staticmethod
    def _minutes_to_sao2_below_90(engine, max_seconds=900.0, dt=0.1):
        start = engine.state.time
        for _ in range(int(max_seconds / dt)):
            engine.step(dt)
            if engine.state.sao2 < 90.0:
                return (engine.state.time - start) / 60.0
        return None

    @staticmethod
    def _induce_apnea(engine):
        engine.give_drug_bolus("propofol", 2.0 * engine.patient.weight)
        engine.give_drug_bolus("roc", 0.6 * engine.patient.weight)

    def test_preoxygenation_extends_safe_apnea_time(self, awake_engine):
        awake_engine.set_airway_mode("Mask")
        awake_engine.set_fgf(10.0, 0.0)
        for _ in range(1800):
            awake_engine.step(0.1)
        assert awake_engine.state.pao2 > 450.0, "Three minutes of tidal breathing should denitrogenate"
        # Near the EtO2 90% preoxygenation endpoint, limited by circuit FiO2 (Nimmagadda 2017).
        assert 85.0 < awake_engine.state.et_o2 < 100.0 * awake_engine.state.fio2

        self._induce_apnea(awake_engine)
        awake_engine.set_airway_obstruction(1.0)
        minutes = self._minutes_to_sao2_below_90(awake_engine)
        assert minutes is not None and 5.0 < minutes < 11.0

    def test_room_air_apnea_desaturates_within_two_minutes(self, awake_engine):
        self._induce_apnea(awake_engine)
        minutes = self._minutes_to_sao2_below_90(awake_engine)
        assert minutes is not None and minutes < 2.0

    def test_apneic_oxygenation_through_patent_airway(self, engine_factory):
        endpoints = []
        for obstruction in (0.0, 1.0):
            engine = engine_factory(start=True)
            engine.set_airway_mode("Mask")
            engine.set_fgf(10.0, 0.0)
            for _ in range(1800):
                engine.step(0.1)
            self._induce_apnea(engine)
            # A tracheal tube keeps the patent arm open during induction;
            # the obstructed arm models an occluded tube.
            engine.set_airway_mode("ETT")
            engine.set_airway_obstruction(obstruction)
            for _ in range(6000):
                engine.step(0.1)
            endpoints.append((engine.resp.state.p_alveolar_o2, engine.state.sao2, engine.state.pa_co2))
        patent, blocked = endpoints
        # Absorbed O2 is replenished through the airway; atelectasis still
        # leaves an arterial-alveolar difference despite high alveolar oxygen.
        assert patent[0] > blocked[0] + 150.0
        assert patent[1] > 95.0 and blocked[1] < 90.0
        # Apneic PaCO2 rises about 3-6 mmHg/min from 40 mmHg (Stock 1989).
        assert 70.0 < patent[2] < 100.0
