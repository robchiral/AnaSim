import pytest

from anasim.physiology.respiration import RespiratoryModel


def _breath(patient, paco2=None, **drugs):
    model = RespiratoryModel(patient)
    if paco2 is not None:
        model.state.p_alveolar_co2 = paco2
    drugs.setdefault("ce_prop", 0.0)
    drugs.setdefault("ce_remi", 0.0)
    return model.step(1.0, **drugs)


def test_opioid_slows_rate_and_propofol_reduces_depth(patient):
    baseline = _breath(patient)

    remi = _breath(patient, ce_remi=1.0)
    remi_rr = remi.rr / baseline.rr
    assert 0.4 < remi_rr < 0.65
    assert remi.vt / baseline.vt >= remi_rr - 0.1
    assert _breath(patient, ce_remi=5.0).rr < 6

    propofol = _breath(patient, ce_prop=3.5)
    propofol_vt = propofol.vt / baseline.vt
    assert propofol_vt < 0.7
    assert propofol.rr / baseline.rr >= propofol_vt - 0.1


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


def test_hyperventilation_stops_breathing_under_anesthesia_until_co2_recovers(awake_engine):
    """Hickey 1971: under anesthesia the apneic threshold lies 4-5 mmHg below the
    resting PaCO2 at any depth. Ventilation above need stops the patient's efforts;
    after the ventilator stops, breathing resumes once CO2 rises to the threshold."""
    engine = awake_engine
    engine.enable_tci("propofol", 3.0)
    engine.enable_tci("remi", 1.0)
    engine.set_airway_mode("ETT")
    for _ in range(3600):
        engine.step(0.1)
    resting = engine.resp.state.p_alveolar_co2
    assert engine.resp.state.rr > 8.0

    engine.set_vent_settings(rr=14, vt=0.5, peep=5, ie="1:2", mode="PCV", p_insp=14)
    engine.set_vent_power(True)
    for _ in range(2400):
        engine.step(0.1)
    assert engine.resp.state.apnea and engine.resp_mech.effort.amplitude == 0.0

    engine.set_vent_power(False)
    for _ in range(6000):
        engine.step(0.1)
        if not engine.resp.state.apnea:
            break
    assert resting - 5.0 <= engine.resp.state.p_alveolar_co2 <= resting - 4.0


def test_hypercapnia_does_not_overcome_deep_drug_depression(patient):
    baseline = _breath(patient, paco2=40.0)
    normocapnic = _breath(patient, paco2=40.0, ce_prop=3.5, ce_remi=4.0)
    hypercapnic = _breath(patient, paco2=70.0, ce_prop=3.5, ce_remi=4.0)

    assert hypercapnic.mv > normocapnic.mv
    assert hypercapnic.mv < baseline.mv * 0.6
    assert hypercapnic.mv < 4.0


class TestOxygenStores:
    """Apneic desaturation follows lung and blood O2 stores (Benumof 1997)."""

    def test_blood_loss_shortens_apneic_desaturation(self, engine_factory):
        """Blood loss lowers the Hb mass that stores O2 before Hb concentration falls."""
        times = []
        for loss_fraction in (0.0, 0.3):
            engine = engine_factory(start=True)
            engine.hemo.vol_clearance = 0.0
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
