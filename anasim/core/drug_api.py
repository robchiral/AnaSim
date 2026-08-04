"""
Drug Controller API Mixin for SimulationEngine.

This module contains drug control interface methods extracted from engine.py
for better maintainability while preserving the SimulationEngine API.
"""

from typing import TYPE_CHECKING, Optional

from .action_log import ACTION_INFUSION_RATE, ACTION_TCI_TARGET
from .drug_registry import DRUG_REGISTRY, DrugSpec, TCIMode, get_drug_spec
from .units import convert_rate

if TYPE_CHECKING:
    from .engine import SimulationEngine
    from .tci import TCIController


class DrugControllerMixin:
    """
    Mixin providing drug control interface for SimulationEngine.
    
    Provides methods for:
    - Setting drug infusion rates
    - Enabling/disabling TCI (Target Controlled Infusion)
    - Getting drug metadata and state for UI
    """

    def enable_tci(
        self: "SimulationEngine",
        drug: str,
        target: float,
        mode: str = TCIMode.EFFECT_SITE.value,
    ):
        """
        Enable TCI for a drug.
        
        Args:
            drug: Registry key for the controlled drug.
            target: Target concentration.
            mode: Requested target compartment; fixed-mode drugs use their registry mode.
        """
        from .tci import TCIController

        # TCI control does not benefit from waveform-rate updates. A 100 ms
        # floor avoids excessive controller iterations and coarse-step
        # oscillation while remaining much faster than infusion-pump updates.
        sampling_time = max(self.config.dt, 0.1)
        control_time = max(10.0, sampling_time)
        
        spec = get_drug_spec(drug)

        max_rate = self._tci_max_rate(spec)
        tci_attr = spec.tci_attr
        pk_attr = spec.pk_attr
        controller = getattr(self, tci_attr)
        tci_mode = spec.fixed_tci_mode.value if spec.fixed_tci_mode else TCIMode(mode).value

        if not controller:
            controller = TCIController(
                getattr(self, pk_attr),
                spec.tci_name,
                tci_mode,
                max_rate=max_rate,
                sampling_time=sampling_time,
                control_time=control_time,
            )
            setattr(self, tci_attr, controller)

        controller.max_rate = max_rate
        controller.sync_state_estimate(getattr(self, pk_attr))
        controller.target = target
        self.actions.record(
            self.state.time, ACTION_TCI_TARGET, label=spec.key, amount=target
        )

    def sync_active_tci_from_pk(self: "SimulationEngine", *drug_keys: str):
        """Resynchronize active TCI controllers with the live PK model state."""
        keys = drug_keys or tuple(spec.key for spec in DRUG_REGISTRY)
        for key in keys:
            spec = get_drug_spec(key)
            controller = getattr(self, spec.tci_attr)
            pk_model = getattr(self, spec.pk_attr)
            if not controller:
                continue
            controller.sync_from_pk_model(pk_model)

    def _tci_max_rate(self: "SimulationEngine", spec: DrugSpec) -> float:
        """Return a clinically realistic max infusion rate for a drug."""
        return spec.max_rate.internal_rate(self.patient.weight)

    def disable_tci(self: "SimulationEngine", drug: str):
        """Disable TCI for a drug and reset infusion rate to zero."""
        spec = get_drug_spec(drug)
        setattr(self, spec.tci_attr, None)
        setattr(self, spec.rate_attr, 0.0)
        self.actions.record(self.state.time, ACTION_TCI_TARGET, label=spec.key)

    def get_controllable_drugs(self: "SimulationEngine") -> tuple[DrugSpec, ...]:
        """Return the typed registry in UI display order."""
        return DRUG_REGISTRY

    def set_drug_rate(self: "SimulationEngine", key: str, rate_user_unit: float):
        """Generic setter for drug infusion rate."""
        self._set_rate_from_user(key, rate_user_unit)

    def set_drug_target(self: "SimulationEngine", key: str, target: Optional[float]):
        """Generic setter for TCI target. None/Negative disables TCI."""
        if target is None or target < 0:
            self.disable_tci(key)
        else:
            self.enable_tci(key, target)

    def get_drug_state(self: "SimulationEngine", key: str):
        """Return current state generic dict."""
        state = {'rate': 0.0, 'target': 0.0, 'is_tci': False}
        
        spec = get_drug_spec(key)

        state["rate"] = self._get_rate_to_user(key)
        controller = getattr(self, spec.tci_attr)
        if controller:
            state["is_tci"] = True
            state["target"] = controller.target
                 
        return state

    def _set_rate_from_user(self: "SimulationEngine", key: str, rate_user_unit: float):
        spec = get_drug_spec(key)
        rate_internal = convert_rate(rate_user_unit, spec.rate_unit, spec.internal_rate_unit)
        rate_internal = max(0.0, rate_internal)
        setattr(self, spec.rate_attr, rate_internal)
        self.actions.record(
            self.state.time,
            ACTION_INFUSION_RATE,
            label=spec.key,
            amount=max(0.0, rate_user_unit),
        )

    def _get_rate_to_user(self: "SimulationEngine", key: str) -> float:
        spec = get_drug_spec(key)
        rate_internal = getattr(self, spec.rate_attr)
        return convert_rate(rate_internal, spec.internal_rate_unit, spec.rate_unit)
