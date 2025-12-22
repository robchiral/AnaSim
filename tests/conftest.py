from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from patient.patient import Patient
from core.engine import SimulationEngine
from core.state import SimulationConfig


DEFAULT_PATIENT = dict(age=40, weight=70, height=170, sex="male")


@pytest.fixture
def patient():
    """Standard adult patient used across most tests."""
    return Patient(**DEFAULT_PATIENT)


@pytest.fixture
def awake_engine(patient):
    """Engine initialized in the awake state."""
    config = SimulationConfig(mode="awake")
    engine = SimulationEngine(patient, config)
    engine.start()
    return engine


@pytest.fixture
def anesthetized_engine(patient):
    """Engine initialized in a steady-state anesthetized configuration."""
    config = SimulationConfig(mode="steady_state", maint_type="tiva")
    engine = SimulationEngine(patient, config)
    engine.start()
    return engine


@pytest.fixture
def advance_time():
    """Helper to advance simulations using consistent step handling."""
    def _advance(engine, seconds, dt=1.0):
        if seconds <= 0:
            return
        steps = int(seconds / dt)
        for _ in range(steps):
            engine.step(dt)
        remainder = seconds - steps * dt
        if remainder > 1e-9:
            engine.step(remainder)

    return _advance
