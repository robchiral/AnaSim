"""Flow limitation and trapped-volume responses to ventilator settings."""

import pytest

from anasim.core.state import SimulationConfig
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.patient.patient import Patient
from anasim.physiology.lung import LungAeration
from anasim.physiology.resp_mech import ExpiratorySegment, RespiratoryMechanics


def test_obstructed_expiration_is_scooped_and_flow_limited():
    # Negative expiratory pressure increases flow through a fixed resistance,
    # but cannot increase flow once the expiratory ceiling is reached.
    normal, obstructed = [], []
    for pressure in (0.0, -5.0, -10.0):
        lung = RespiratoryMechanics(compliance=0.05, resistance=30.0)
        lung.volume = 0.5
        normal.append(lung.pressure_segment((pressure, 0.0)).volume(0.05))
        lung.bronchospasm, lung.bronch_resistance = 1.0, 20.0
        parameters = lung.expiration_parameters((pressure, 0.0), (0.0, 0.0), 0.0)
        segment = ExpiratorySegment(lung, (pressure, 0.0), (0.0, 0.0), 0.0, parameters)
        obstructed.append(segment.volume(0.05))
        assert segment.paw(0.05) == pressure
    assert normal[2] < normal[1] < normal[0]
    assert obstructed == pytest.approx([obstructed[0]] * 3, abs=1e-12)

    # An effort drawing gas before the ventilator triggers uses inspiratory
    # resistance, rather than the additional narrowing of the expiratory limb.
    drive = (20.0, 0.0)
    parameters = lung.expiration_parameters((0.0, 0.0), drive, 1.0)
    incoming = ExpiratorySegment(lung, (0.0, 0.0), drive, 1.0, parameters)
    assert incoming.volume(0.05) == pytest.approx(lung.pressure_segment((0.0, 0.0), drive, 1.0).volume(0.05))

    lung = RespiratoryMechanics(compliance=0.05, resistance=30.0)
    lung.volume, lung.viscoelastic_ratio = 0.5, 0.0
    lung.bronchospasm, lung.bronch_resistance = 1.0, 20.0
    vent = AnesthesiaVentilator()
    samples = []
    for _ in range(1000):
        before = lung.volume
        vent.step(0.01, lung, "spontaneous", collect_samples=True)
        assert lung.volume - before == pytest.approx(sum(row[1] for row in vent.samples), abs=1e-12)
        samples.extend(vent.samples)
    flows = [-min(samples, key=lambda row: abs(row[4] - volume))[3] for volume in (0.1, 0.2, 0.4)]
    chord = flows[0] + (flows[2] - flows[0]) / 3.0
    assert flows[1] < 0.98 * chord


def test_longer_expiration_and_bronchospasm_relief_reduce_intrinsic_peep_across_step_sizes(engine_factory):
    traces = []
    patient = Patient()
    for dt in (0.01, 0.1, 1.0):
        aeration = LungAeration(patient, recruited=0.9, atelectatic=LungAeration.INDUCTION_ATELECTASIS)
        aeration.unconscious = 1.0
        lung = RespiratoryMechanics(resistance=patient.airway_resistance(), aeration=aeration)
        vent = AnesthesiaVentilator()
        trace = []
        for rr, ie, bronch in ((24, "1:1", 1.0), (24, "1:4", 1.0), (8, "1:4", 1.0), (24, "1:1", 0.0)):
            lung.bronchospasm = bronch
            lung.bronch_resistance = 20.0 * bronch
            lung.resistance = patient.airway_resistance() + 20.0 * bronch
            vent.update_settings(mode="VCV", rr=rr, tv=500, ie=ie, peep=5, p_max=60)
            for _ in range(round(60 / dt)):
                vent.step(dt, lung, "vent")
            trace.append((vent.monitors.auto_peep, aeration.frc, vent.monitors.tv_exp))
        traces.append(trace)
    for trace in traces[1:]:
        for actual, reference in zip(trace, traces[0], strict=True):
            assert actual == pytest.approx(reference, abs=1e-8)

    engine = engine_factory(config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=123), start=True)
    engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
    eng_trace = []
    for rr, ie, bronch in ((24, "1:1", 1.0), (24, "1:4", 1.0), (8, "1:4", 1.0), (24, "1:1", 0.0)):
        engine.set_bronchospasm(bronch)
        engine.set_vent_settings(mode="VCV", rr=rr, vt=0.5, ie=ie, peep=5, p_max=60)
        for _ in range(60):
            engine.step(1.0)
        eng_trace.append((engine.vent.monitors.auto_peep, engine.aeration.frc, engine.state.vt))
    short, longer, slow, relieved = eng_trace
    assert short[0] > 5.0
    assert longer[0] < 0.75 * short[0]
    assert longer[1] < short[1] - 0.2
    assert slow[0] < 0.3 * longer[0]
    assert relieved[0] < 0.5 * short[0]
    assert relieved[1] < short[1] - 0.3
    assert all(endpoint[2] == pytest.approx(500.0, abs=2.0) for endpoint in eng_trace)


def test_copd_mechanics_match_lourens_2000_time_constants_and_passive_flow_limitation(engine_factory):
    """Lourens 2000 & Jubran 1995: COPD keeps inspiratory RC moderate while dynamic
    expiratory collapse extends RCexp to ~1.1-1.6 s in moderate and >2.4 s in severe COPD,
    producing passive ZEEP expiratory flow limitation in severe disease.
    """
    results = {}
    for sev in (0.0, 0.5, 1.0):
        eng = engine_factory(
            config=SimulationConfig(mode="steady_state", dt=1.0, rng_seed=1),
            start=True, age=65, weight=70.0, height=170.0, sex="male", copd_severity=sev,
        )
        eng.give_drug_bolus("roc", 70.0)
        eng.set_vent_settings(mode="VCV", rr=12.0, vt=0.5, peep=0.0, ie="1:2", p_max=60.0)
        for _ in range(120):
            eng.step(1.0)
        lung = eng.resp_mech
        lung.volume = 0.5
        _, _, limited_high = lung.expiration_parameters((0.0, 0.0), (0.0, 0.0), 1.0)
        lung.volume = 0.1
        r_exp_low, r_ceil_low, _ = lung.expiration_parameters((0.0, 0.0), (0.0, 0.0), 1.0)
        results[sev] = (
            lung.compliance,
            lung.resistance * lung.compliance,
            r_exp_low * lung.compliance,
            r_ceil_low * lung.compliance,
            limited_high,
        )
    # Compliance rises modestly (+15% nominal) rather than inflating by 60%.
    assert 1.10 < results[1.0][0] / results[0.0][0] < 1.45
    assert results[0.5][1] < 0.90 and results[1.0][1] < 1.25
    assert 1.05 <= results[0.5][2] <= 1.75
    assert results[1.0][2] >= 2.40 and results[1.0][3] >= 3.0
    assert not results[0.0][4] and results[1.0][4]
