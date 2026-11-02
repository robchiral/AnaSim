"""Recruitment, oxygen reserve, and conservation across changing mechanics."""

from itertools import pairwise

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.patient.patient import Patient
from anasim.physiology.lung import LungAeration
from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.physiology.respiration import RespiratoryModel


def _run(engine, dt, seconds):
    for _ in range(round(seconds / dt)):
        engine.step(dt)
    aeration = engine.aeration
    return (aeration.recruited, aeration.frc, engine.resp_mech.compliance,
            engine.state.vt, engine.state.pao2)


def test_recruitment_changes_mechanics_and_oxygenation_across_step_sizes():
    traces = []
    for dt in (0.01, 0.1, 1.0):
        engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", dt=dt, rng_seed=123))
        engine.start()
        engine.set_fgf(10.0, 0.0)

        engine.set_vent_settings(mode="VCV", vt=0.5, rr=12, peep=0, ie="1:2")
        collapsed = _run(engine, dt, 240)
        engine.set_vent_settings(mode="PCV", vt=0.5, rr=3, peep=5, ie="1:1", p_insp=35)
        recruited = _run(engine, dt, 40)
        engine.set_vent_settings(mode="VCV", vt=0.5, rr=12, peep=5, ie="1:2")
        maintained = _run(engine, dt, 120)
        engine.set_vent_settings(mode="VCV", vt=0.5, rr=12, peep=0, ie="1:2")
        withdrawn = _run(engine, dt, 240)

        assert collapsed[0] < 0.75 and recruited[0] > 0.95
        assert maintained[0] > 0.95 and withdrawn[0] < 0.75
        assert maintained[1] > collapsed[1] + 0.5
        assert maintained[2] > 1.25 * collapsed[2]
        # Atelectasis shunts about 5-7% at ZEEP (Gunnarsson 1991; Tokics 1996;
        # Rothen 1995); recruitment raises PaO2.
        assert maintained[4] > 1.15 * collapsed[4]
        assert withdrawn[4] < 0.9 * maintained[4]
        assert maintained[3] == pytest.approx(500.0, abs=2.0)
        traces.append((collapsed, recruited, maintained, withdrawn))
    for trace in traces[1:]:
        for actual, reference in zip(trace, traces[0], strict=True):
            assert actual[:3] == pytest.approx(reference[:3], rel=0.005, abs=0.001)
            assert actual[3:] == pytest.approx(reference[3:], rel=0.01, abs=0.2)


def test_age_and_peep_set_anesthetized_oxygenation():
    """Awake on room air, the A-a gradient is 12.6 mmHg at 65 (Crapo 1999).
    Under inhaled anesthesia at FiO2 0.4 without PEEP it is 52 mmHg at 25 and
    rises 1.35 mmHg/yr (Gunnarsson 1991). In VitalDB intraoperative blood gases
    at FiO2 0.36, PEEP 4-8 raised PaO2 by about 5 mmHg at 30 (95% CI -24 to
    +30) and 35 at 70 (+15 to +51)."""
    awake = SimulationEngine(Patient(age=65), SimulationConfig(mode="awake", rng_seed=123))
    awake.start()
    for _ in range(120):
        awake.step(1.0)
    assert awake.resp.state.p_alveolar_o2 - awake.state.pao2 == pytest.approx(12.6, abs=2.0)

    def anesthetized(age, maint_type, peep, o2_l_min):
        engine = SimulationEngine(Patient(age=age), SimulationConfig(
            mode="steady_state", maint_type=maint_type, rng_seed=123))
        engine.start()
        engine.set_fgf(o2_l_min, 10.0 - o2_l_min)
        engine.set_vent_settings(mode="VCV", vt=0.5, rr=12, peep=peep, ie="1:2")
        for _ in range(1500):
            engine.step(1.0)
        return engine

    gradient = []
    for age in (25, 65):
        engine = anesthetized(age, "balanced", 0, 2.5)
        assert engine.state.fio2 == pytest.approx(0.39, abs=0.01)
        gradient.append(engine.resp.state.p_alveolar_o2 - engine.state.pao2)
    assert gradient[0] == pytest.approx(52.0, abs=10.0)
    assert 1.0 < (gradient[1] - gradient[0]) / 40.0 < 1.8

    for age, low, high in ((30, -10.0, 15.0), (70, 20.0, 50.0)):
        pao2 = [anesthetized(age, "tiva", peep, 2.0).state.pao2 for peep in (0, 5)]
        assert low < pao2[1] - pao2[0] < high


def test_copd_and_airway_closure_have_separate_oxygen_responses_across_step_sizes():
    """Similar room-air hypoxemia can have different oxygen responses.

    COPD units have a V/Q near 0.1 (Wagner 1977, Figs. 3-4); units from
    airway closure under anesthesia have a V/Q of 0.05.
    """
    traces = []
    for dt in (0.1, 1.0):
        outcomes = []
        for copd, unconscious in ((0.75, 0.0), (0.0, 1.0)):
            model = RespiratoryModel(Patient(age=70, copd_severity=copd))
            values = []
            for fio2, pressure in ((0.21, 0), (0.28, 0), (0.4, 0), (1.0, 0), (0.4, 8)):
                for _ in range(round(600.0 / dt)):
                    state = model.step(
                        dt, ce_prop=0, ce_remi=0, ce_roc=10, measured_breaths=True,
                        mech_rr=12, mech_vt_l=model.vt_0 / 1000.0, fio2=fio2,
                        unconscious=unconscious, airway_pressure=pressure,
                        cardiac_output=model.baseline_co_l_min,
                    )
                values.append(state.p_arterial_o2)
            assert all(right > left for left, right in pairwise(values[:4]))
            assert values[3] > 450.0
            outcomes.append(values)
        copd_oxygen, airway_oxygen = outcomes
        assert copd_oxygen[0] == pytest.approx(airway_oxygen[0], abs=3.0)
        assert copd_oxygen[2] > airway_oxygen[2] + 15.0
        # PEEP opens anesthesia-related closure; the chronic COPD component persists.
        assert copd_oxygen[4] == pytest.approx(copd_oxygen[2], abs=1.0)
        assert airway_oxygen[4] > airway_oxygen[2] + 30.0
        traces.append(outcomes)
    for actual, reference in zip(traces[1], traces[0], strict=True):
        assert actual == pytest.approx(reference, abs=0.3)


@pytest.mark.parametrize(("fio2", "flow", "vco2", "vo2", "paco2", "shunt", "low_flow", "pao2", "sem"), [
    (0.21, 6.68, 224.0, 256.0, 36.8, 0.008, 0.116, 77.0, 8.1),
    (1.0, 6.44, 239.0, 274.0, 39.8, 0.021, 0.235, 487.0, 40.0),
])
def test_acute_asthma_oxygen_exchange_matches_ballester_1989(
    fio2, flow, vco2, vo2, paco2, shunt, low_flow, pao2, sem,
):
    """Table 5: five patients on air and after 20 min of 100% oxygen.

    Use the measured flow, metabolism, shunt, and low V/Q perfusion, with
    ventilation set for the measured PaCO2. Hb was not reported, so the
    default is used. Bands are 2 SEM.
    """
    model = RespiratoryModel(Patient(age=40, weight=62, height=157, sex="female"))
    model.vco2, model.rq = vco2, vco2 / vo2
    model.baseline_co_l_min = flow
    model.low_vq_awake = low_flow
    model.state.p_alveolar_co2 = paco2
    model.equilibrate_oxygen(fio2)
    model.equilibrate_blood_gases(flow, vco2 / 1000.0)
    tidal_volume = model.vd_deadspace + model.co2_constant * vco2 / paco2 / 12.0
    for _ in range(1200):
        state = model.step(
            1.0, ce_prop=0, ce_remi=0, ce_roc=10, measured_breaths=True,
            mech_rr=12, mech_vt_l=tidal_volume, fio2=fio2,
            cardiac_output=flow, shunt_fraction=shunt,
        )
    assert state.pa_co2 == pytest.approx(paco2, abs=0.5)
    assert state.p_arterial_o2 == pytest.approx(pao2, abs=2.0 * sem)


def test_low_vq_oxygen_gradient_matches_pressure_perturbation():
    model = RespiratoryModel(Patient(age=65, copd_severity=1))
    for rq in (0.7, 0.875, 1.0):
        model.rq = rq
        for pressure in (40.0, 100.0, 250.0, 700.0):
            low = model._low_vq_oxygen(pressure, 0.15, 14.0, 26.8)
            epsilon = 1e-3
            above = model._low_vq_oxygen(pressure + epsilon, 0.15, 14.0, 26.8)
            below = model._low_vq_oxygen(pressure - epsilon, 0.15, 14.0, 26.8)
            assert low is not None and above is not None and below is not None
            assert low[3:] == pytest.approx(
                ((above[1] - below[1]) / (2 * epsilon), (above[2] - below[2]) / (2 * epsilon)),
                rel=1e-5, abs=1e-8,
            )


def test_sustained_pressure_has_a_recruitment_ceiling_across_step_sizes():
    traces = []
    # Rothen 1993: normalized reduction in CT atelectatic area at 20, 30, 40.
    # CT area constrains pressure response, not absolute gas volume or shunt.
    for dt in (0.01, 0.1, 1.0):
        aeration = LungAeration(Patient(), recruited=0.65)
        aeration.unconscious = 1.0
        lung = RespiratoryMechanics(aeration=aeration)
        vent = AnesthesiaVentilator()
        vent.update_settings(mode="PCV", peep=5, p_insp=15, rr=3, ie="1:1")
        vent.step(0.1, lung, "vent")  # Establish the absolute-volume reference.
        recruited = []
        for pressure, seconds in ((20, 120), (30, 60), (40, 60), (5, 40)):
            vent.update_settings(mode="PCV", peep=5, p_insp=pressure - 5, rr=3, ie="1:1")
            for _ in range(round(seconds / dt)):
                before = vent.volume
                vent.step(dt, lung, "vent", collect_samples=True)
                assert vent.volume - before == pytest.approx(
                    sum(sample[1] for sample in vent.samples), abs=1e-11)
            recruited.append(aeration.recruited)
        opened = [(value - 0.65) / 0.35 for value in recruited[:3]]
        assert opened == pytest.approx((0.078, 0.453, 0.875), abs=0.04)
        assert recruited[3] == pytest.approx(recruited[2], abs=0.001)
        traces.append(recruited)
    for trace in traces[1:]:
        assert trace == pytest.approx(traces[0], abs=1e-10)


def test_inert_gas_slows_recollapse_without_changing_bulk_compliance():
    """Rothen 1995: nitrogen delays CT collapse recurrence after recruitment."""
    traces = []
    for dt in (0.01, 0.1, 1.0):
        outcomes = []
        for nitrogen in (0.0, 0.6):
            aeration = LungAeration(Patient(), recruited=0.65)
            aeration.unconscious = 1.0
            aeration.inert_fraction = nitrogen
            values = []
            for pressure, seconds in ((45.0, 40), (0.0, 300), (0.0, 2100)):
                for _ in range(round(seconds / dt)):
                    aeration.advance(dt, pressure * dt)
                values.append(aeration.atelectatic / 0.35)
            outcomes.append(values)
            if nitrogen == 0.0:
                oxygen_compliance = aeration.static_mechanics(aeration.relaxed_volume)[1]
            else:
                assert aeration.static_mechanics(aeration.relaxed_volume)[1] == pytest.approx(oxygen_compliance)
        oxygen, mixed = outcomes
        assert oxygen[0] < 0.05 and mixed[0] < 0.05
        assert oxygen[1] > 0.9
        assert 0.1 < mixed[2] < 0.3
        assert mixed[2] < 0.3 * oxygen[2]
        traces.append((*oxygen, *mixed))
    for trace in traces[1:]:
        assert trace == pytest.approx(traces[0], abs=1e-9)


def test_changing_aeration_and_peep_conserves_volume_and_stiffens_at_high_inflation():
    aeration = LungAeration(Patient(), recruited=0.7)
    aeration.unconscious = 1.0
    lung = RespiratoryMechanics(aeration=aeration)
    vent = AnesthesiaVentilator()
    vent.update_settings(mode="PCV", peep=5, p_insp=35, rr=6, ie="1:1")
    vent.step(0.1, lung, "vent", collect_samples=True)
    recruited_compliances = []
    for i in range(300):
        if i == 150:
            vent.update_settings(peep=10)
        before = vent.volume
        vent.step(0.1, lung, "vent", collect_samples=True)
        assert vent.volume - before == pytest.approx(sum(sample[1] for sample in vent.samples), abs=1e-12)
        if aeration.recruited > 0.95:
            recruited_compliances.append(lung.compliance)
    assert aeration.recruited > 0.95
    assert min(recruited_compliances) < 0.9 * max(recruited_compliances)


def test_patient_lung_volume_sets_apnea_reserve_and_shunt_limits_oxygen_rescue():
    def seconds_to_desaturation(weight):
        patient = Patient(weight=weight, height=170)
        model = RespiratoryModel(patient)
        model.equilibrate_oxygen(1.0)
        seconds = 0
        while model.state.sao2 >= 90.0:
            model.step(1.0, ce_prop=0, ce_remi=0, ce_roc=10, airway_patency=0,
                       blood_volume_ml=5000, metabolic_factor=70.0 / weight)
            seconds += 1
        return model.frc, seconds

    (lean_frc, lean_seconds), (heavier_frc, heavier_seconds) = (
        seconds_to_desaturation(55), seconds_to_desaturation(90))
    assert heavier_frc < lean_frc
    assert heavier_seconds < lean_seconds - 60

    saturation = []
    for shunt in (0.0, 0.4):
        model = RespiratoryModel(Patient())
        for _ in range(3000):
            state = model.step(0.1, ce_prop=0, ce_remi=0, ce_roc=10, fio2=1.0,
                               mech_rr=12, mech_vt_l=0.5, measured_breaths=True, shunt_fraction=shunt)
        saturation.append(state.sao2)
    assert saturation[0] > 99.0 and saturation[1] < 95.0
