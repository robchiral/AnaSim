import copy
from typing import Any

import pytest

from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.respiration import RespiratoryModel


def _breath(patient, paco2=None, **drugs):
    model = RespiratoryModel(patient)
    if paco2 is not None:
        model.state.p_alveolar_co2 = paco2
    drugs.setdefault("ce_prop", 0.0)
    drugs.setdefault("ce_remi", 0.0)
    drugs.setdefault("fio2", 1.0)  # Isolate CO2/drug effects from hypoxic feedback.
    # Concentration-response checks hold CO2 fixed while the controller settles.
    for _ in range(900):
        model.state.p_alveolar_co2 = 40.0 if paco2 is None else paco2
        model.state.pa_co2 = model.state.p_alveolar_co2
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


def test_bradypnea_preserves_controller_ventilation_until_apnea(patient):
    """A smooth opioid response must not halve ventilation again at RR 4.

    Hold CO2 at baseline and seed the settled controller, as in a controlled
    ventilatory-response protocol. The apnea cutoff still suppresses breaths.
    """
    for rate in (1.99, 2.01, 3.99, 4.01):
        model = RespiratoryModel(patient)
        remi = 2.0 * model.c50_remi * (1.0 - rate / model.rr_0)
        model.initialize_drug_effects(0.0, remi)
        state = model.step(0.0, ce_prop=0.0, ce_remi=remi, fio2=1.0)
        if rate < 2.0:
            assert state.apnea and state.rr == state.va == 0.0
        else:
            assert not state.apnea
            assert state.rr == pytest.approx(rate)
            assert state.va == pytest.approx(model.va_baseline * state.drive_central)


@pytest.mark.parametrize("ce_prop", [2.0, 3.0, 6.0])
def test_bouillon_spontaneous_steady_state_co2_and_alveolar_ventilation(patient, ce_prop):
    """Bouillon 2004 Eq. 18, with and without its CO2-production correction."""
    keep = 1.0 / (1.0 + (ce_prop / 1.33)**1.68)
    for production in (1.0, 0.7 + 0.3 * keep):
        model = RespiratoryModel(patient)
        # The slow tissue CO2 store takes longer than one hour to settle at deep anesthesia.
        for _ in range(7200):
            state = model.step(
                1.0, ce_prop=ce_prop, ce_remi=0.0, fio2=1.0,
                unconscious=1.0, metabolic_factor=production,
            )
        expected_co2 = 40.0 * (production / keep)**(1.0 / 5.37)
        expected_va = model.va_baseline * production**(4.37 / 5.37) * keep**(1.0 / 5.37)
        assert state.p_alveolar_co2 == pytest.approx(expected_co2, abs=0.5)
        assert state.va == pytest.approx(expected_va, rel=0.02)
        assert not state.apnea


def test_oxygen_saturation_gradient_matches_pressure_perturbation(patient):
    model = RespiratoryModel(patient)
    for temperature in (32.0, 37.0, 40.0):
        p50 = model._temperature_p50(temperature)
        for pressure in (0.0, 1.0, 20.0, 40.0, 100.0, 500.0):
            saturation, derivative = model._saturation_gradient(pressure, p50)
            assert saturation == model._saturation_at_p50(pressure, p50)
            if pressure == 0.0:
                assert derivative == 0.0
                continue
            epsilon = 1e-4
            expected = (model._saturation_at_p50(pressure + epsilon, p50)
                        - model._saturation_at_p50(pressure - epsilon, p50)) / (2 * epsilon)
            assert derivative == pytest.approx(expected, rel=1e-5, abs=1e-9)


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


def test_hypoxic_response_declines_with_sustained_exposure_and_propofol():
    """Nieuwenhuijs 2000: fractional responses to pulsed and sustained hypoxia."""
    traces = []
    for dt in (0.1, 1.0):
        responses = {}
        for propofol in (0.0, 0.6):
            model = RespiratoryModel(Patient(age=24))

            def expose(pressure, seconds, model=model, propofol=propofol, dt=dt):
                for _ in range(round(seconds / dt)):
                    model.state.p_alveolar_co2 = 46.6
                    model.state.pa_co2 = 46.6
                    model.state.p_arterial_o2 = pressure
                    model.state.p_alveolar_o2 = pressure + model.aa_grad_base
                    state = model.step(dt, ce_prop=propofol, ce_remi=0.0, fio2=0.5)
                return state.va

            baseline = expose(110.0, 900.0)
            acute = expose(55.0, 180.0) - baseline
            sustained = expose(55.0, 720.0) - baseline
            assert acute > 0.5 * baseline
            decline = 1.0 - sustained / acute
            assert decline == pytest.approx(0.35 if propofol == 0 else 0.54, abs=0.08)
            # Two minutes of normoxia do not erase the sustained-hypoxia history.
            expose(110.0, 120.0)
            rechallenge = expose(55.0, 180.0) - baseline
            assert 0.4 < rechallenge / acute < 0.8
            responses[propofol] = acute, sustained, rechallenge
        assert responses[0.6][0] / responses[0.0][0] == pytest.approx(0.5, abs=0.1)
        traces.append((*responses[0.0], *responses[0.6]))
    assert traces[0] == pytest.approx(traces[1], rel=0.005)

    # Repeated 3-min challenges generate no sustained ventilatory decline.
    model = RespiratoryModel(Patient(age=24))
    pulses = []
    for _ in range(6):
        for pressure, seconds in ((110.0, 300), (55.0, 180)):
            for _ in range(seconds):
                model.state.p_alveolar_co2 = 46.6
                model.state.pa_co2 = 46.6
                model.state.p_arterial_o2 = pressure
                model.state.p_alveolar_o2 = pressure + model.aa_grad_base
                state = model.step(1.0, ce_prop=0.6, ce_remi=0.0, fio2=0.5)
        pulses.append(state.va)
    assert pulses[-1] == pytest.approx(pulses[0], rel=0.01)


def test_apneic_co2_accumulation_and_ventilated_recovery(patient_factory):
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
        assert state.p_alveolar_co2 > outcomes[-1] + 20.0
        recovered = []
        for dt in (1.0, 300.0):
            ventilated = copy.deepcopy(model)
            for _ in range(round(600.0 / dt)):
                state = ventilated.step(
                    dt, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                    fio2=1.0, metabolic_factor=0.8, mech_rr=12.0,
                    mech_vt_l=0.5, measured_breaths=True,
                )
            # Slow stores prolong CO2 washout after 45 minutes of apnea.
            assert 25.0 < state.etco2 < state.pa_co2 < 0.7 * model.state.pa_co2
            recovered.append((state.pa_co2, state.etco2))
        assert recovered[1] == pytest.approx(recovered[0], abs=1e-8)
    assert outcomes[0] == pytest.approx(outcomes[1], abs=0.1)


def _ventilated(patient, seconds, **inputs):
    model = RespiratoryModel(patient)
    for _ in range(round(seconds / 0.1)):
        state = model.step(
            0.1, ce_prop=0.0, ce_remi=0.0, mech_vent_mv=6.0, mech_rr=12.0, mech_vt_l=0.5,
            measured_breaths=True,
            **inputs,
        )
    return state


def test_anemia_and_low_cardiac_output_do_not_lower_pao2(patient):
    """They reduce O2 content and delivery, not arterial O2 tension."""
    reference = _ventilated(patient, 60.0)
    for inputs in ({"hb_g_dl": 6.0}, {"cardiac_output": 0.1}):
        state = _ventilated(patient, 60.0, **inputs)
        assert state.p_arterial_o2 >= reference.p_arterial_o2 - 2.0
        if "hb_g_dl" in inputs:
            assert state.sao2 > 95.0
            oxygen_content = 1.34 * inputs["hb_g_dl"] * state.sao2 / 100.0 + 0.0031 * state.p_arterial_o2
            reference_content = 1.34 * patient.baseline_hb * reference.sao2 / 100.0 + 0.0031 * reference.p_arterial_o2
            assert oxygen_content < 0.6 * reference_content
        else:
            # Low pulmonary flow washes CO2 out of the lung. Arterial blood
            # follows with a transit delay and gains affinity as pH rises.
            assert state.pa_co2 < reference.pa_co2
            assert state.ph > reference.ph
            assert state.sao2 > reference.sao2


def test_blood_temperature_shifts_oxygen_affinity_without_creating_oxygen(patient):
    """Dash 2016: cooling increases affinity, with O2 supplied by existing stores."""
    model = RespiratoryModel(patient)
    model.equilibrate_oxygen(0.14)
    model.equilibrate_blood_gases(5.0, model.vco2 / 1000.0)
    model.step(0.0, ce_prop=0, ce_remi=0, airway_patency=0.0)

    def oxygen_l():
        state = model.state
        gas = model.gas_volume_l * model._gas_capacity_per_l * state.p_alveolar_o2
        return gas + model.gas_transport.oxygen_l

    stored = oxygen_l()
    initial_pressure = model.state.p_arterial_o2
    for temperature in (34.0, 40.0, 37.0):
        model.step(0.0, ce_prop=0, ce_remi=0, temp_c=temperature, airway_patency=0.0)
        assert oxygen_l() == pytest.approx(stored, abs=1e-9)
        if temperature == 34:
            assert model.state.p_arterial_o2 < initial_pressure
        elif temperature == 40:
            assert model.state.p_arterial_o2 > initial_pressure
    assert model.state.p_arterial_o2 == pytest.approx(initial_pressure, abs=1e-7)
    endpoints = [_ventilated(patient, 900, fio2=0.14, temp_c=temperature) for temperature in (34, 37, 40)]
    assert endpoints[0].sao2 > endpoints[1].sao2 > endpoints[2].sao2
    assert endpoints[0].p_arterial_o2 == pytest.approx(endpoints[2].p_arterial_o2, abs=0.5)


def test_changing_shunt_redistributes_existing_oxygen(patient):
    model = RespiratoryModel(patient)
    model.equilibrate_oxygen(1.0)
    model.equilibrate_blood_gases(5.0, model.vco2 / 1000.0)
    model.step(0.0, ce_prop=0, ce_remi=0, airway_patency=0.0)

    def stored_oxygen():
        state = model.state
        return (
            model.gas_volume_l * model._gas_capacity_per_l * state.p_alveolar_o2
            + model.gas_transport.oxygen_l
        )

    initial = stored_oxygen()
    for shunt in (0.1, 0.35, 0.0):
        for _ in range(100):
            model.step(0.1, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                       airway_patency=0.0, shunt_fraction=shunt)
        assert stored_oxygen() + model.gas_transport.consumed_o2_l == pytest.approx(initial, abs=1e-8)


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


def test_low_cardiac_output_amplifies_atelectatic_shunt(engine_factory):
    """ZEEP atelectasis shunts 1-12% of cardiac output (Tokics 1996). Blood loss
    lowers mixed venous O2, so the same shunt lowers PaO2 further."""
    endpoints = []
    for loss in (0.0, 0.4):
        engine = engine_factory(
            config=SimulationConfig(mode="steady_state", end_on_cardiac_arrest=False), start=True,
        )
        engine.set_fgf(10.0, 0.0)
        engine.set_vent_settings(mode="VCV", rr=12, vt=0.5, peep=0, ie="1:2")
        engine.hemo.add_volume(-loss * engine.hemo.blood_volume)
        for _ in range(240):
            engine.step(1.0)
        assert engine.state.fio2 > 0.95
        assert 0.05 < engine.resp.shunt_fraction < 0.12
        endpoints.append((engine.state.co, engine.state.pao2, engine.state.sao2))
    normal, bled = endpoints
    assert bled[0] < 2.0
    assert bled[1] < 0.5 * normal[1]
    # Ventilated lung still saturates most of the blood.
    assert bled[2] > 95.0


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
    """Hickey 1971: under anesthesia, breathing stops 4-5 mmHg below the
    resting PaCO2, whatever the agent and depth. Hold drug concentrations and
    consciousness fixed to isolate the CO2 threshold from PK and induction kinetics.
    """
    model = RespiratoryModel(patient)
    inputs: dict[str, Any] = dict(ce_prop=2.0, ce_remi=0.0, unconscious=1.0, fio2=1.0)
    for _ in range(9000):
        model.step(0.1, **inputs)
    assert model.state.rr > 8.0
    resting = model.state.pa_co2
    assert resting > 45.0  # Propofol raises the resting PaCO2.

    for _ in range(6000):
        model.step(0.1, **inputs, mech_rr=20.0, mech_vt_l=0.5, mech_vent_mv=10.0)
    assert model.state.apnea

    for _ in range(6000):
        model.step(0.1, **inputs)
        if not model.state.apnea:
            break
    assert not model.state.apnea
    assert 3.75 <= resting - model.state.pa_co2 <= 5.0


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

    def test_blood_loss_shortens_apneic_desaturation_at_the_same_flow(self, patient):
        """Blood loss reduces O2 reserve; fixed flow isolates it from arterial transit delay."""
        times = []
        stores = []
        for loss_fraction in (0.0, 0.3):
            model = RespiratoryModel(patient)
            blood_volume = model.baseline_blood_volume_ml * (1.0 - loss_fraction)
            model.step(0.0, ce_prop=0.0, ce_remi=0.0, blood_volume_ml=blood_volume)
            stores.append(model.gas_transport.oxygen_l)
            for _step in range(1200):
                model.step(0.1, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                           airway_patency=0.0, blood_volume_ml=blood_volume, cardiac_output=5.0)
                if model.state.sao2 < 80.0:
                    break
            times.append((_step + 1) * 0.1)
        assert stores[1] < 0.75 * stores[0]
        assert times[1] < times[0] - 3.0

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

    @pytest.mark.parametrize(("fio2", "air_l_min", "eto2", "seconds"), [
        (1.0, 0.0, 93.0, (327.0, 495.0)),
        (0.8, 1.52, 75.0, (244.0, 362.0)),
        (0.6, 3.04, 53.0, (144.0, 282.0)),
    ])
    def test_preoxygenation_sets_open_airway_apnea_time(self, engine_factory, fio2, air_l_min, eto2, seconds):
        """Edmark 2003: women aged 44 given 100, 80, or 60% O2 at 6 L/min during
        induction reached EtO2 93, 75, and 53%; with the tube open to air,
        SpO2 fell to 90% after 411 ± 84, 303 ± 59, and 213 ± 69 s.
        """
        dt = 0.1
        engine = engine_factory(
            config=SimulationConfig(tci_enabled=True, dt=dt, rng_seed=1), start=True,
            age=44, height=166, weight=67, sex="female", baseline_hb=12.1,
        )

        def advance(seconds_):
            for _ in range(round(seconds_ / dt)):
                engine.step(dt)

        engine.set_airway_mode("Mask")
        engine.set_fgf(6.0 - air_l_min, air_l_min)
        advance(60.0)
        engine.give_drug_bolus("fentanyl", 300)  # Fentanyl 0.2 mg and alfentanil 1 mg
        advance(60.0)
        engine.enable_tci("propofol", 8.0, mode="plasma")
        advance(30.0)
        engine.give_drug_bolus("roc", 30)
        engine.set_bag_mask_ventilation(True, rr=9, vt=0.55)
        advance(180.0)
        assert engine.state.et_o2 == pytest.approx(eto2, abs=5.0)

        engine.set_bag_mask_ventilation(False)
        engine.set_airway_mode("ETT")
        engine.set_fgf(0.0, 10.0)
        start = engine.state.time
        while engine.state.display_spo2 >= 90.0 and engine.state.time - start < 900.0:
            engine.step(dt)
        assert seconds[0] < engine.state.time - start < seconds[1]

    def test_clamped_tube_after_oxygen_matches_stock_blood_gases(self, engine_factory):
        """Stock 1989 Table 1: after 10 min of O2 ventilation the tracheal tube was clamped.

        Changes are applied to the PaCO2 39 and PaO2 412 mmHg baselines; bounds
        are 2 SEM. ZEEP is assumed, and Stock used enflurane rather than propofol.
        The sealed lung shrinks as it absorbs O2, so PaO2 falls slowly.
        """
        engine = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1),
                                start=True, age=36)
        settings = engine.vent.settings
        engine.set_vent_settings(rr=settings.rr, vt=settings.tv / 1000.0, peep=0.0, ie="1:2", mode="VCV")
        engine.set_fgf(6.0, 0.0)
        for _ in range(600):
            engine.step(1.0)
        engine.set_airway_obstruction(1.0)
        resp = engine.resp.state
        pa_co2, pa_o2 = [resp.pa_co2], [resp.p_arterial_o2]
        for _ in range(300):
            engine.step(1.0)
            pa_co2.append(resp.pa_co2)
            pa_o2.append(resp.p_arterial_o2)
        for time, mean, sd, n in ((20, 44, 7, 13), (40, 48, 8, 14), (60, 50, 7, 14), (120, 53, 6, 13),
                                  (180, 56, 6, 11), (240, 59, 7, 9), (300, 63, 9, 7)):
            assert 39.0 + pa_co2[time] - pa_co2[0] == pytest.approx(mean, abs=2.0 * sd / n**0.5)
        assert 412.0 + pa_o2[300] - pa_o2[0] == pytest.approx(314.0, abs=2.0 * 87.0 / 7**0.5)

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
        # Prolonged apnea retains CO2 even when oxygenation is sustained.
        assert 60.0 < patent[2] < 100.0


def test_airway_rescue_reoxygenates_during_ongoing_apnea(patient):
    """Laviola 2020: opening an obstructed airway causes oxygen-dependent passive refill.

    This checks the reported direction and mechanism, rather than numerical
    agreement with that paper's different cardiopulmonary model.
    """
    peaks, endpoints = [], []
    for fio2 in (0.21, 0.6, 1.0):
        model = RespiratoryModel(patient)
        model.equilibrate_oxygen(1.0)
        for _ in range(900):
            state = model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                               fio2=1.0, airway_patency=0.0, unconscious=1.0)
            if state.sao2 <= 60.0:
                break
        assert 55.0 < state.sao2 <= 60.0
        assert model.oxygen_exchange_l_min(1.0, 0.0, 0.0, 0.0) == 0.0
        # The circuit uses this same transport function before the lung step.
        # Opening the airway draws more O2 than maintenance apneic uptake.
        transport = model.oxygen_exchange_l_min(fio2, 0.0, 1.0, 0.0)
        assert transport > fio2 * model.vco2 / model.rq / 1000.0
        peak = state.sao2
        for _ in range(300):
            state = model.step(1.0, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                               fio2=fio2, airway_patency=1.0, unconscious=1.0)
            assert state.apnea and state.va == 0.0
            peak = max(peak, state.sao2)
        peaks.append(peak)
        endpoints.append(state.sao2)
    assert 70.0 < peaks[0] < peaks[1] < peaks[2]
    assert endpoints[0] < 90.0 < endpoints[1] < endpoints[2]

    # Rapid refill must conserve O2 through the nonlinear blood store, so
    # subdividing the same interval cannot change the rescued oxygen tension.
    fine = RespiratoryModel(patient)
    fine.equilibrate_oxygen(1.0)
    for _ in range(9000):
        state = fine.step(0.1, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                          fio2=1.0, airway_patency=0.0, unconscious=1.0)
        if state.sao2 <= 60.0:
            break
    for _ in range(3000):
        state = fine.step(0.1, ce_prop=0.0, ce_remi=0.0, ce_roc=10.0,
                          fio2=1.0, airway_patency=1.0, unconscious=1.0)
    assert state.p_arterial_o2 == pytest.approx(model.state.p_arterial_o2, rel=0.01)


@pytest.mark.parametrize("preoxygenate_first", [False, True])
def test_preoxygenation_survives_a_change_in_blood_composition(patient, preoxygenate_first):
    model = RespiratoryModel(patient)
    hb = 0.99 * model.baseline_hb
    if not preoxygenate_first:
        model.step(0.0, ce_prop=0.0, ce_remi=0.0, hb_g_dl=hb)
    model.equilibrate_oxygen(1.0)
    initial_po2 = model.state.p_arterial_o2
    assert model.state.sao2 > 99.0
    # A small dilution keeps the equilibrated gas state.
    model.step(0.0, ce_prop=0.0, ce_remi=0.0, hb_g_dl=hb)
    assert model.state.p_arterial_o2 == pytest.approx(initial_po2, abs=0.1)
    assert model.state.sao2 > 99.0


def test_ventilation_requirement_scales_with_body_size(engine_factory):
    """Resting breathing matches CO2 production; fixed ventilator settings do not.

    VO2 scales with weight^0.75 and dead space with predicted body weight, so
    awake room-air oxygenation is the same across the supported domain, while
    VCV 500 mL x 12 overventilates a small patient and underventilates a large
    one (PaCO2 = 0.863 VCO2 / VA).
    """
    sizes: tuple[dict[str, Any], ...] = (
        dict(weight=50, height=165, sex="female"), {}, dict(weight=100, height=177),
    )
    ventilated_co2 = []
    for size in sizes:
        engine = engine_factory(start=True, **size)
        for _ in range(900):
            engine.step(1.0)
        assert engine.state.sao2 > 95.0
        assert engine.state.pa_co2 == pytest.approx(40.0, abs=1.5)

        engine = engine_factory(config=SimulationConfig(mode="steady_state"), start=True, **size)
        engine.set_vent_settings(rr=12, vt=0.5, peep=5, ie="1:2", mode="VCV")
        for _ in range(1200):
            engine.step(1.0)
        ventilated_co2.append(engine.state.pa_co2)
    small, typical, large = ventilated_co2
    assert small < 32.0 < typical < 40.0 < 42.0 < large
