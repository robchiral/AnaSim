from anasim.core.constants import SHIVER_MAX_MULTIPLIER
from anasim.patient.patient import Patient
from anasim.physiology.respiration import RespiratoryModel


def _breath(patient, paco2=None, **drugs):
    model = RespiratoryModel(patient)
    if paco2 is not None:
        model.state.p_alveolar_co2 = paco2
    drugs.setdefault("ce_prop", 0.0)
    drugs.setdefault("ce_remi", 0.0)
    return model.step(1.0, **drugs)


def test_nmba_abolishes_breathing_but_spares_central_drive(patient):
    state = _breath(patient, ce_roc=10.0)
    assert state.drive_central > 0.95
    assert state.muscle_factor < 0.1
    assert state.rr < 1.0
    assert state.vt < 10.0


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


def test_anemia_does_not_lower_pao2():
    patient = Patient(age=40, weight=70, height=170, sex="Male", baseline_hb=13.5)
    pao2 = []
    for hb in (13.5, 6.0):
        model = RespiratoryModel(patient)
        for _ in range(30):
            state = model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=0, hb_g_dl=hb)
        pao2.append(state.p_arterial_o2)
    assert abs(pao2[1] - pao2[0]) < 2.0


def test_co2_drives_breathing_and_opioids_blunt_it(patient):
    """Babenco 2000: opioids shift the CO2 response right and flatten it."""
    assert _breath(patient, paco2=50.0).drive_central > _breath(patient, paco2=40.0).drive_central
    assert (
        _breath(patient, paco2=45.0, ce_remi=3.0).drive_central
        < _breath(patient, paco2=45.0).drive_central
    )


def test_hypercapnia_does_not_overcome_deep_drug_depression(patient):
    baseline = _breath(patient, paco2=40.0)
    normocapnic = _breath(patient, paco2=40.0, ce_prop=3.5, ce_remi=4.0)
    hypercapnic = _breath(patient, paco2=70.0, ce_prop=3.5, ce_remi=4.0)

    assert hypercapnic.mv > normocapnic.mv
    assert hypercapnic.mv < baseline.mv * 0.6
    assert hypercapnic.mv < 4.0


def test_shivering_raises_paco2_at_fixed_ventilation(patient):
    paco2 = []
    for metabolic_factor in (1.0, 1.0 + SHIVER_MAX_MULTIPLIER):
        model = RespiratoryModel(patient)
        for _ in range(600):
            model.step(
                1.0,
                ce_prop=0.0,
                ce_remi=0.0,
                ce_roc=0.0,
                mech_vent_mv=6.0,
                mech_rr=12.0,
                mech_vt_l=0.5,
                metabolic_factor=metabolic_factor,
            )
        paco2.append(model.state.p_alveolar_co2)
    assert paco2[1] > paco2[0] + 10.0


class TestOxygenStores:
    """Apneic desaturation follows lung and blood O2 stores (Benumof 1997)."""

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

        self._induce_apnea(awake_engine)
        awake_engine.set_airway_obstruction(1.0)
        minutes = self._minutes_to_sao2_below_90(awake_engine)
        assert minutes is not None and 5.0 < minutes < 11.0

    def test_room_air_apnea_desaturates_within_two_minutes(self, awake_engine):
        self._induce_apnea(awake_engine)
        minutes = self._minutes_to_sao2_below_90(awake_engine)
        assert minutes is not None and minutes < 2.0

    def test_apneic_oxygenation_through_patent_airway(self, awake_engine):
        awake_engine.set_airway_mode("Mask")
        awake_engine.set_fgf(10.0, 0.0)
        for _ in range(1800):
            awake_engine.step(0.1)
        self._induce_apnea(awake_engine)
        for _ in range(6000):
            awake_engine.step(0.1)
        assert awake_engine.state.sao2 > 97.0
        # Apneic PaCO2 rises about 3-6 mmHg/min from 40 mmHg (Stock 1989).
        assert 70.0 < awake_engine.state.pa_co2 < 100.0
