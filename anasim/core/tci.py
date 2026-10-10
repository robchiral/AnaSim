"""Target-controlled infusion (Shafer and Gregg. J Pharmacokinet Biopharm. 1992)."""

import math

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp

PREDICTION_HORIZON_S = 600.0
PREDICTION_GRID_S = 1.0


def _discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization of dx/dt = A x + B u."""
    n = A.shape[0]
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = A
    augmented[:n, n:] = B
    transition = expm(augmented * dt)
    return transition[:n, :n], transition[:n, n:]


def _pk_signature(pk_model) -> tuple[float, ...]:
    return tuple(
        float(getattr(pk_model, name))
        for name in ("v1", "v2", "v3", "elimination_clearance", "cl2", "cl3", "ke0", "basal_input_rate")
    )


class TCIController:
    """Plasma- or effect-site-targeted TCI with a pump rate limit.

    Each update selects the largest rate that keeps the predicted concentration
    at or below target over the horizon. Above target, the infusion pauses.
    """

    def __init__(
        self,
        pk_model,
        target_compartment: str = "effect_site",
        max_rate: float = 1200.0,
        sampling_time: float = 1.0,
        control_time: float = 10.0,
    ):
        """Build a controller for pk_model.

        Args:
            pk_model: PK model exposing get_ss_matrices() and state_vector().
            target_compartment: "plasma" or "effect_site".
            max_rate: Pump limit in model units per second.
            sampling_time: Interval between controller ticks (s).
            control_time: Interval between rate updates (s).
        """
        if sampling_time > control_time:
            raise ValueError("Sampling time cannot be larger than control time")
        self.target_compartment = target_compartment
        self.pk_model = pk_model
        self.max_rate = max_rate
        self.sampling_time = sampling_time
        # Rate changes fall on sampling instants; prediction uses that same duration.
        self.control_time = math.ceil(control_time / sampling_time) * sampling_time
        self.target = 0.0
        self.infusion_rate = 0.0
        self._time = 0.0
        self._next_update = 0.0
        self._load_pk_model(pk_model)
        self.sync_state_estimate(pk_model)

    def _load_pk_model(self, pk_model) -> None:
        A_min, B = pk_model.get_ss_matrices()
        A = A_min / 60.0  # B already maps a per-second input to concentration per second.
        self.n_state = A.shape[0]
        self.target_id = 0 if self.target_compartment == "plasma" else self.n_state - 1
        A_grid, B_grid = _discretize(A, B, PREDICTION_GRID_S)
        steps = int(PREDICTION_HORIZON_S / PREDICTION_GRID_S)
        pulse_steps = math.floor(self.control_time / PREDICTION_GRID_S)
        partial_time = self.control_time - pulse_steps * PREDICTION_GRID_S
        partial_input = np.zeros_like(B_grid)
        if partial_time > 0.0:
            _, B_partial = _discretize(A, B, partial_time)
            partial_input = expm(A * (PREDICTION_GRID_S - partial_time)) @ B_partial
        predictions = steps + int(partial_time > 0.0)
        self._free_response = np.empty((predictions, self.n_state))
        self._unit_response = np.empty(predictions)
        self._basal_response = np.zeros(predictions)
        self.basal_input_rate = pk_model.basal_input_rate
        row = np.eye(self.n_state)[self.target_id]
        unit_state = np.zeros((self.n_state, 1))
        basal_state = np.zeros((self.n_state, 1))
        for k in range(steps):
            row = row @ A_grid
            unit_state = A_grid @ unit_state
            if k < pulse_steps:
                unit_state += B_grid
            elif k == pulse_steps:
                unit_state += partial_input
            self._free_response[k] = row
            self._unit_response[k] = unit_state[self.target_id, 0]
            if self.basal_input_rate:
                basal_state = A_grid @ basal_state + B_grid * self.basal_input_rate
                self._basal_response[k] = basal_state[self.target_id, 0]
        if partial_time > 0.0:
            # A plasma peak at pump stop can lie between the one-second grid points.
            A_stop, B_stop = _discretize(A, B, self.control_time)
            self._free_response[-1] = A_stop[self.target_id]
            self._unit_response[-1] = B_stop[self.target_id, 0]
            self._basal_response[-1] = self._unit_response[-1] * self.basal_input_rate
        self._responsive = self._unit_response > 1e-6 * self._unit_response.max()
        self._signature = _pk_signature(pk_model)

    def sync_from_pk_model(self, pk_model, rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
        """Rebuild the prediction model after material PK drift and reseed the state.

        Returns True when the prediction model was rebuilt.
        """
        self.pk_model = pk_model
        signature = _pk_signature(pk_model)
        rebuilt = any(
            abs(curr - prev) > max(abs(prev) * rel_tol, abs_tol)
            for prev, curr in zip(self._signature, signature)
        )
        if rebuilt:
            self._load_pk_model(pk_model)
        self.sync_state_estimate(pk_model)
        return rebuilt

    def sync_state_estimate(self, pk_model) -> None:
        """Seed the internal state estimate from the live PK compartments."""
        self.x = pk_model.state_vector().reshape(-1, 1)

    def set_target(self, target: float) -> None:
        """Change the target and recompute the rate at the next step."""
        self.target = max(0.0, float(target))
        self._next_update = self._time

    def _control_rate(self) -> float:
        if self.target <= 0.0:
            return 0.0
        free = self._free_response[self._responsive] @ self.x[:, 0] + self._basal_response[self._responsive]
        unit = self._unit_response[self._responsive]
        return clamp(float(np.min((self.target - free) / unit)), 0.0, self.max_rate)

    def step(self, target: float | None = None, sim_time: float | None = None) -> float:
        """Advance one sampling interval and return the infusion rate (model units/s)."""
        if sim_time is not None:
            self._time = sim_time
        if target is not None and target != self.target:
            self.set_target(target)
        if self._time + 1e-9 >= self._next_update:
            self.sync_from_pk_model(self.pk_model, rel_tol=0.0, abs_tol=0.0)
            self.infusion_rate = self._control_rate()
            self._next_update = self._time + self.control_time
        if sim_time is None:
            self._time += self.sampling_time
        return self.infusion_rate
