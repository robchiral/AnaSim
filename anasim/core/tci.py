"""Target-controlled infusion (Shafer and Gregg. J Pharmacokinet Biopharm. 1992)."""

import math

import numpy as np
from scipy.linalg import expm

from anasim.core.utils import clamp, fill_response_rows
from anasim.patient.domain import finite_number

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
        for name in (
            "v1", "v2", "v3", "elimination_clearance", "cl2", "cl3", "ke0",
            "basal_input_rate", "circulation", "effect_input_fraction",
        )
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
        predictions = steps + int(partial_time > 0.0)
        self._free_response = np.empty((predictions, self.n_state))
        self._unit_response = np.zeros(predictions)
        self._basal_response = np.zeros(predictions)
        self.basal_input_rate = pk_model.effective_basal_input_rate
        response = self._free_response[:steps]
        fill_response_rows(response, A_grid, self.target_id)

        # The pump pulse sums the impulse response C A^k B over its active steps.
        impulse = np.empty(steps)
        impulse[0] = B_grid[self.target_id, 0]
        impulse[1:] = response[:-1] @ B_grid[:, 0]
        unit = self._unit_response[:steps]
        if pulse_steps:
            unit[:] = np.convolve(impulse, np.ones(min(pulse_steps, steps)))[:steps]
        if partial_time > 0.0 and pulse_steps < steps:
            _, B_partial = _discretize(A, B, partial_time)
            partial_input = expm(A * (PREDICTION_GRID_S - partial_time)) @ B_partial
            unit[pulse_steps] += partial_input[self.target_id, 0]
            unit[pulse_steps + 1:] += response[:steps - pulse_steps - 1] @ partial_input[:, 0]
        if self.basal_input_rate:
            self._basal_response[:steps] = np.cumsum(impulse) * self.basal_input_rate
        A_stop, B_stop = _discretize(A, B, self.control_time)

        def stop_terms(row: int) -> tuple[np.ndarray, float, float]:
            unit_stop = float(B_stop[row, 0])
            return A_stop[row], unit_stop, unit_stop * self.basal_input_rate

        self._stop_c1_free, self._stop_c1_unit, self._stop_c1_basal = stop_terms(0)
        self._stop_target_free, self._stop_target_unit, self._stop_target_basal = stop_terms(self.target_id)
        if partial_time > 0.0:
            # A plasma peak at pump stop can lie between the one-second grid points.
            self._free_response[-1] = self._stop_target_free
            self._unit_response[-1] = self._stop_target_unit
            self._basal_response[-1] = self._stop_target_basal
        self._responsive = self._unit_response > 1e-6 * self._unit_response.max()
        sub_pulse_steps = pulse_steps if partial_time > 0.0 else pulse_steps - 1
        if self.target_compartment == "effect_site" and sub_pulse_steps > 0:
            # Effect-site peaks occur at or after pump stop; ignoring sub-pulse
            # decay steps prevents 1e-4 concentration noise from zeroing a 10 s step.
            post_pulse = self._responsive.copy()
            post_pulse[:min(sub_pulse_steps, steps)] = False
            if post_pulse.any():
                self._responsive = post_pulse
        self._signature = _pk_signature(pk_model)

    def sync_from_pk_model(self, pk_model, rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
        """Reseed from live PK; return True if parameter drift required a model rebuild."""
        self.pk_model = pk_model
        signature = _pk_signature(pk_model)
        rebuilt = any(
            abs(curr - prev) > max(abs(prev) * rel_tol, abs_tol)
            for prev, curr in zip(self._signature, signature, strict=True)
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
        self.target = max(0.0, finite_number("target", target))
        self._next_update = self._time

    def _control_rate(self) -> float:
        if self.target <= 0.0 or not self._responsive.any():
            return 0.0
        x = self.x[:, 0]
        free = self._free_response[self._responsive] @ x + self._basal_response[self._responsive]
        unit = self._unit_response[self._responsive]
        u_pulse = max(0.0, float(np.min((self.target - free) / unit)))
        if self.target_compartment == "effect_site" and self._stop_c1_unit > 0.0:
            effect_frac = max(1e-6, float(self.pk_model.effect_input_fraction))
            c1_target = self.target / effect_frac
            u_c1 = max(0.0, (c1_target - float(self._stop_c1_free @ x) - self._stop_c1_basal) / self._stop_c1_unit)
            ce_at_stop = float(self._stop_target_free @ x) + self._stop_target_basal + self._stop_target_unit * u_c1
            blend = clamp((ce_at_stop / self.target - 0.98) / 0.018, 0.0, 1.0)
            rate = (1.0 - blend) * u_pulse + blend * min(u_pulse, u_c1)
        else:
            rate = u_pulse
        return clamp(rate, 0.0, self.max_rate)

    def recompute(self, sim_time: float) -> float:
        """Force an immediate control-rate update at sim_time."""
        self._time = sim_time
        self._next_update = sim_time
        return self.step(sim_time=sim_time)

    def step(self, target: float | None = None, sim_time: float | None = None) -> float:
        """Advance one sampling interval and return the infusion rate (model units/s)."""
        if sim_time is not None:
            self._time = sim_time
        if target is not None and target != self.target:
            self.set_target(target)
        time_eps = 1e-9 * max(1.0, abs(self._time))
        if self._time + time_eps >= self._next_update:
            self.sync_from_pk_model(self.pk_model, rel_tol=0.0, abs_tol=0.0)
            self.infusion_rate = self._control_rate()
            self._next_update = self._time + self.control_time
        if sim_time is None:
            self._time += self.sampling_time
        return self.infusion_rate
