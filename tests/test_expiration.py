"""Flow limitation and trapped-volume responses to ventilator settings."""

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.machine.ventilator import AnesthesiaVentilator
from anasim.patient.patient import Patient
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


def test_longer_expiration_and_bronchospasm_relief_reduce_intrinsic_peep_across_step_sizes():
    traces = []
    for dt in (0.01, 0.1, 1.0):
        engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", dt=dt, rng_seed=123))
        engine.start()
        engine.give_drug_bolus("roc", 0.8 * engine.patient.weight)
        trace = []
        for rr, ie, bronch in ((24, "1:1", 1.0), (24, "1:4", 1.0), (8, "1:4", 1.0), (24, "1:1", 0.0)):
            engine.set_bronchospasm(bronch)
            engine.set_vent_settings(mode="VCV", rr=rr, vt=0.5, ie=ie, peep=5, p_max=60)
            for _ in range(round(90 / dt)):
                engine.step(dt)
            trace.append((engine.vent.monitors.auto_peep, engine.resp_mech.aeration.frc, engine.state.vt))
        short, longer, slow, relieved = trace
        assert short[0] > 5.0
        assert longer[0] < 0.75 * short[0]
        assert longer[1] < short[1] - 0.2
        assert slow[0] < 0.3 * longer[0]
        assert relieved[0] < 0.5 * short[0]
        assert relieved[1] < short[1] - 0.3
        assert all(endpoint[2] == pytest.approx(500.0, abs=2.0) for endpoint in trace)
        traces.append(trace)
    for trace in traces[1:]:
        for actual, reference in zip(trace, traces[0]):
            assert actual == pytest.approx(reference, abs=1e-8)
