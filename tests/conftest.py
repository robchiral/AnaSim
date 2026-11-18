import copy
import sys
from dataclasses import astuple
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient
from anasim.physiology.disturbances import PROFILE_SPECS

DEFAULT_PATIENT: dict[str, Any] = dict(age=40, weight=70, height=170, sex="male")
_STEADY_STATE_CACHE: dict[tuple[Any, ...], SimulationEngine] = {}


def _build_engine(patient: Patient, config: SimulationConfig) -> SimulationEngine:
    """Build an engine, reusing cached steady-state baselines when deterministic."""
    if (
        config.mode != "steady_state"
        or config.patient_variability
        or (patient.baseline_temp < 29.5 and config.rng_seed is None)
    ):
        return SimulationEngine(patient, config)
    key = (
        astuple(patient),
        repr(astuple(config)),
        repr(PROFILE_SPECS.get(config.disturbance_profile)) if config.disturbance_profile else None,
        repr(PROFILE_SPECS["stim_intubation_pulse"]),
    )
    template = _STEADY_STATE_CACHE.get(key)
    if template is None:
        template = SimulationEngine(copy.deepcopy(patient), copy.deepcopy(config))
        _STEADY_STATE_CACHE[key] = template
    clone = copy.deepcopy(template)
    if config.rng_seed is None:
        clone.rng = np.random.default_rng(None)
        clone._ecg_rng = clone.ecg.rng = np.random.default_rng(clone.rng.integers(0, 2**32 - 1))
        clone._nibp_rng = clone.nibp.rng = np.random.default_rng(clone.rng.integers(0, 2**32 - 1))
        clone._cardiac_rng = clone.cardiac_cycle.rng = np.random.default_rng(clone.rng.integers(0, 2**32 - 1))
        clone._shock_rng = np.random.default_rng(None)
        clone.arrhythmias.rng = np.random.default_rng(None)
        clone.arrhythmias._remaining.clear()
    return clone


@pytest.fixture
def patient():
    """Standard adult patient used across most tests."""
    return Patient(**DEFAULT_PATIENT)


@pytest.fixture
def patient_factory():
    """Factory for creating patients with default overrides."""
    def _factory(**overrides):
        params = DEFAULT_PATIENT.copy()
        params.update(overrides)
        return Patient(**params)

    return _factory


@pytest.fixture
def engine_factory(patient_factory):
    """Factory for creating engines with optional patient/config overrides."""
    def _factory(config=None, start=False, **patient_overrides):
        if config is None:
            config = SimulationConfig(mode="awake")
        engine = _build_engine(patient_factory(**patient_overrides), config)
        if start:
            engine.start()
        return engine

    return _factory


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
    engine = _build_engine(patient, config)
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
