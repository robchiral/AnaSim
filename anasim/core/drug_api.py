"""Drug infusion and TCI controls for SimulationEngine."""

from anasim.patient.patient import Patient

from .action_log import ACTION_INFUSION_RATE, ACTION_TCI_TARGET, ActionLog
from .drug_registry import (
    DRUG_REGISTRY,
    DrugSpec,
    MaxRatePolicy,
    TCIMode,
    get_drug_spec,
)
from .state import SimulationConfig, SimulationState
from .tci import TCIController
from .units import convert_rate


class DrugControllerMixin:
    """Infusion-rate and TCI methods shared by SimulationEngine."""

    # Provided by SimulationEngine.
    config: SimulationConfig
    patient: Patient
    state: SimulationState
    actions: ActionLog
    infusion_rates: dict[str, float]
    tci: dict[str, TCIController]
    _tci_accumulators: dict[str, float]

    def enable_tci(
        self,
        drug: str,
        target: float,
        mode: str = TCIMode.EFFECT_SITE.value,
    ):
        """Start or retarget TCI; fixed-mode drugs ignore the requested compartment."""
        spec = get_drug_spec(drug)
        max_rate = _tci_max_rate(spec)
        if not self.config.tci_enabled:
            raise ValueError("TCI is disabled for this session")
        pk_model = getattr(self, spec.pk_attr)
        controller = self.tci.get(spec.key)
        target_compartment = (spec.fixed_tci_mode or TCIMode(mode)).value
        if controller is None or controller.target_compartment != target_compartment:
            # Waveform-rate controller updates add cost without improving control.
            sampling_time = max(self.config.dt, 0.1)
            controller = TCIController(
                pk_model,
                target_compartment=target_compartment,
                sampling_time=sampling_time,
                control_time=max(10.0, sampling_time),
            )
            self.tci[spec.key] = controller
            self._tci_accumulators.pop(spec.key, None)

        controller.max_rate = max_rate.internal_rate(self.patient.weight)
        controller.sync_state_estimate(pk_model)
        controller.set_target(target)
        self.actions.record(self.state.time, ACTION_TCI_TARGET, label=spec.key, amount=target)

    def sync_active_tci_from_pk(self, *drug_keys: str):
        """Resynchronize active TCI controllers with the live PK model state."""
        specs = [get_drug_spec(key) for key in drug_keys] if drug_keys else DRUG_REGISTRY
        for spec in specs:
            controller = self.tci.get(spec.key)
            if controller:
                controller.sync_from_pk_model(getattr(self, spec.pk_attr))

    def disable_tci(self, drug: str):
        """Disable TCI for a drug and stop its infusion."""
        spec = get_drug_spec(drug)
        _require(spec, spec.has_tci, "target-controlled infusion")
        self.tci.pop(spec.key, None)
        self._tci_accumulators.pop(spec.key, None)
        self.infusion_rates[spec.key] = 0.0
        self.actions.record(self.state.time, ACTION_TCI_TARGET, label=spec.key)

    def get_controllable_drugs(self) -> tuple[DrugSpec, ...]:
        """Return the typed registry in UI display order."""
        return DRUG_REGISTRY

    def set_drug_rate(self, key: str, rate_user_unit: float):
        """Switch to manual infusion at the rate in the registry's user unit."""
        spec = get_drug_spec(key)
        user_unit, model_unit = _rate_units(spec)
        rate = max(0.0, rate_user_unit)
        if spec.key in self.tci:
            self.disable_tci(spec.key)
        self.infusion_rates[spec.key] = convert_rate(rate, user_unit, model_unit, weight_kg=self.patient.weight)
        self.actions.record(self.state.time, ACTION_INFUSION_RATE, label=spec.key, amount=rate)

    def set_drug_target(self, key: str, target: float | None):
        """Set a TCI target; None or a negative target disables TCI."""
        if target is None or target < 0:
            self.disable_tci(key)
        else:
            self.enable_tci(key, target)

    def get_drug_state(self, key: str) -> dict:
        """Return the current user-unit rate and TCI target; zero for bolus-only drugs."""
        spec = get_drug_spec(key)
        controller = self.tci.get(spec.key)
        rate = 0.0
        if spec.has_infusion:
            user_unit, model_unit = _rate_units(spec)
            rate = convert_rate(self.infusion_rates[spec.key], model_unit, user_unit, weight_kg=self.patient.weight)
        return {
            "rate": rate,
            "target": controller.target if controller else 0.0,
            "is_tci": controller is not None,
        }


def _require(spec: DrugSpec, available: bool, control: str) -> None:
    if not available:
        raise ValueError(f"{spec.generic_name} has no {control}")


def _rate_units(spec: DrugSpec) -> tuple[str, str]:
    """Return the (user, model) infusion rate units."""
    if spec.rate_unit is None or spec.internal_rate_unit is None:
        raise ValueError(f"{spec.generic_name} has no infusion")
    return spec.rate_unit, spec.internal_rate_unit


def _tci_max_rate(spec: DrugSpec) -> MaxRatePolicy:
    if not spec.has_tci or spec.max_rate is None:
        raise ValueError(f"{spec.generic_name} has no target-controlled infusion")
    return spec.max_rate
