"""Drug infusion and TCI controls for SimulationEngine."""

from typing import TYPE_CHECKING, Optional

from .action_log import ACTION_INFUSION_RATE, ACTION_TCI_TARGET
from .drug_registry import DRUG_REGISTRY, DrugSpec, TCIMode, get_drug_spec
from .tci import TCIController
from .units import convert_rate

if TYPE_CHECKING:
    from .engine import SimulationEngine


class DrugControllerMixin:
    """Infusion-rate and TCI methods shared by SimulationEngine."""

    def enable_tci(
        self: "SimulationEngine",
        drug: str,
        target: float,
        mode: str = TCIMode.EFFECT_SITE.value,
    ):
        """Start or retarget TCI; fixed-mode drugs ignore the requested compartment."""
        spec = get_drug_spec(drug)
        _require(spec, spec.has_tci, "target-controlled infusion")
        pk_model = getattr(self, spec.pk_attr)
        controller = getattr(self, spec.tci_attr)
        target_compartment = (spec.fixed_tci_mode or TCIMode(mode)).value
        if controller is None or controller.target_compartment != target_compartment:
            # Waveform-rate controller updates add cost without improving control.
            sampling_time = max(self.config.dt, 0.1)
            controller = TCIController(
                pk_model,
                spec.generic_name,
                target_compartment,
                sampling_time=sampling_time,
                control_time=max(10.0, sampling_time),
            )
            setattr(self, spec.tci_attr, controller)
            self._tci_accumulators.pop(spec.tci_attr, None)

        controller.max_rate = spec.max_rate.internal_rate(self.patient.weight)
        controller.sync_state_estimate(pk_model)
        controller.set_target(target)
        self.actions.record(self.state.time, ACTION_TCI_TARGET, label=spec.key, amount=target)

    def sync_active_tci_from_pk(self: "SimulationEngine", *drug_keys: str):
        """Resynchronize active TCI controllers with the live PK model state."""
        specs = [get_drug_spec(key) for key in drug_keys] if drug_keys else DRUG_REGISTRY
        for spec in specs:
            controller = getattr(self, spec.tci_attr) if spec.has_tci else None
            if controller:
                controller.sync_from_pk_model(getattr(self, spec.pk_attr))

    def disable_tci(self: "SimulationEngine", drug: str):
        """Disable TCI for a drug and stop its infusion."""
        spec = get_drug_spec(drug)
        _require(spec, spec.has_tci, "target-controlled infusion")
        setattr(self, spec.tci_attr, None)
        self._tci_accumulators.pop(spec.tci_attr, None)
        setattr(self, spec.rate_attr, 0.0)
        self.actions.record(self.state.time, ACTION_TCI_TARGET, label=spec.key)

    def get_controllable_drugs(self: "SimulationEngine") -> tuple[DrugSpec, ...]:
        """Return the typed registry in UI display order."""
        return DRUG_REGISTRY

    def set_drug_rate(self: "SimulationEngine", key: str, rate_user_unit: float):
        """Switch to manual infusion at the rate in the registry's user unit."""
        spec = get_drug_spec(key)
        _require(spec, spec.has_infusion, "infusion")
        rate = max(0.0, rate_user_unit)
        if spec.has_tci and getattr(self, spec.tci_attr) is not None:
            self.disable_tci(spec.key)
        setattr(self, spec.rate_attr, convert_rate(rate, spec.rate_unit, spec.internal_rate_unit))
        self.actions.record(self.state.time, ACTION_INFUSION_RATE, label=spec.key, amount=rate)

    def set_drug_target(self: "SimulationEngine", key: str, target: Optional[float]):
        """Set a TCI target; None or a negative target disables TCI."""
        if target is None or target < 0:
            self.disable_tci(key)
        else:
            self.enable_tci(key, target)

    def get_drug_state(self: "SimulationEngine", key: str) -> dict:
        """Return the current user-unit rate and TCI target; zero for bolus-only drugs."""
        spec = get_drug_spec(key)
        controller = getattr(self, spec.tci_attr) if spec.has_tci else None
        rate = 0.0
        if spec.has_infusion:
            rate = convert_rate(getattr(self, spec.rate_attr), spec.internal_rate_unit, spec.rate_unit)
        return {
            "rate": rate,
            "target": controller.target if controller else 0.0,
            "is_tci": controller is not None,
        }


def _require(spec: DrugSpec, available: bool, control: str) -> None:
    if not available:
        raise ValueError(f"{spec.generic_name} has no {control}")
