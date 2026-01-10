"""
Drug Controller API Mixin for SimulationEngine.

This module contains drug control interface methods extracted from engine.py
for better maintainability while preserving the SimulationEngine API.
"""

from typing import TYPE_CHECKING

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

    _DRUG_ORDER = (
        "propofol",
        "remi",
        "nore",
        "vaso",
        "phenyl",
        "epi",
        "dobu",
        "milri",
        "roc",
    )
    _DRUG_SPECS = {
        "propofol": {
            "name": "Propofol 1%",
            "rate_attr": "propofol_rate_mg_sec",
            "rate_unit": "mg/hr",
            "rate_scale": 3600.0,
            "tci_attr": "tci_prop",
            "tci_name": "Propofol",
            "tci_unit": "ug/mL",
            "tci_mode": None,
            "tci_range": (0.0, 10.0),
            "pk_attr": "pk_prop",
            "max_rate": ("mg_kg_min", 0.3),
            "default_bolus": 150,
            "bolus_unit": "mg",
        },
        "remi": {
            "name": "Remifentanil 50mcg/mL",
            "rate_attr": "remi_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_remi",
            "tci_name": "Remifentanil",
            "tci_unit": "ng/mL",
            "tci_mode": None,
            "tci_range": (0.0, 10.0),
            "pk_attr": "pk_remi",
            "max_rate": ("ug_kg_min", 0.5),
            "default_bolus": 10,
            "bolus_unit": "mcg",
        },
        "nore": {
            "name": "Norepinephrine 16mcg/mL",
            "rate_attr": "nore_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_nore",
            "tci_name": "Norepinephrine",
            "tci_unit": "ng/mL",
            "tci_mode": "plasma",
            "tci_range": (0.0, 30.0),
            "pk_attr": "pk_nore",
            "max_rate": ("ug_kg_min", 1.0),
            "default_bolus": 10,
            "bolus_unit": "mcg",
        },
        "vaso": {
            "name": "Vasopressin 20U/mL",
            "rate_attr": "vaso_rate_mu_sec",
            "rate_unit": "U/min",
            "rate_scale": 0.06,  # U/min -> mU/sec (x1000/60)
            "tci_attr": "tci_vaso",
            "tci_name": "Vasopressin",
            "tci_unit": "mU/L",
            "tci_mode": "plasma",
            "tci_range": (0.0, 80.0),
            "pk_attr": "pk_vaso",
            "max_rate": ("u_min", 0.1),
            "default_bolus": 1.0,
            "bolus_unit": "U",
        },
        "epi": {
            "name": "Epinephrine 100mcg/mL",
            "rate_attr": "epi_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_epi",
            "tci_name": "Epinephrine",
            "tci_unit": "ng/mL",
            "tci_mode": "plasma",
            "tci_range": (0.0, 20.0),
            "pk_attr": "pk_epi",
            "max_rate": ("ug_kg_min", 0.5),
            "default_bolus": 10,
            "bolus_unit": "mcg",
        },
        "phenyl": {
            "name": "Phenylephrine 100mcg/mL",
            "rate_attr": "phenyl_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_phenyl",
            "tci_name": "Phenylephrine",
            "tci_unit": "ng/mL",
            "tci_mode": "plasma",
            "tci_range": (0.0, 120.0),
            "pk_attr": "pk_phenyl",
            "max_rate": ("ug_kg_min", 2.0),
            "default_bolus": 100,
            "bolus_unit": "mcg",
        },
        "dobu": {
            "name": "Dobutamine 1mg/mL",
            "rate_attr": "dobu_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_dobu",
            "tci_name": "Dobutamine",
            "tci_unit": "ng/mL",
            "tci_mode": "plasma",
            "tci_range": (0.0, 500.0),
            "pk_attr": "pk_dobu",
            "max_rate": ("ug_kg_min", 20.0),
            "default_bolus": 0.0,
            "bolus_unit": "mcg",
        },
        "milri": {
            "name": "Milrinone 200mcg/mL",
            "rate_attr": "mil_rate_ug_sec",
            "rate_unit": "ug/min",
            "rate_scale": 60.0,
            "tci_attr": "tci_mil",
            "tci_name": "Milrinone",
            "tci_unit": "ng/mL",
            "tci_mode": "plasma",
            "tci_range": (0.0, 500.0),
            "pk_attr": "pk_mil",
            "max_rate": ("ug_kg_min", 0.75),
            "default_bolus": 0.0,
            "bolus_unit": "mcg",
        },
        "roc": {
            "name": "Rocuronium 10mg/mL",
            "rate_attr": "roc_rate_mg_sec",
            "rate_unit": "mg/hr",
            "rate_scale": 3600.0,
            "tci_attr": "tci_roc",
            "tci_name": "Rocuronium",
            "tci_unit": "ug/mL",
            "tci_mode": None,
            "tci_range": (0.0, 10.0),
            "pk_attr": "pk_roc",
            "max_rate": ("mg_kg_hr", 1.0),
            "default_bolus": 50,
            "bolus_unit": "mg",
        },
    }
    
    def set_propofol_rate(self: "SimulationEngine", mg_per_hr: float):
        self._set_rate_from_user("propofol", mg_per_hr)

    def set_remi_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("remi", ug_per_min)

    def set_nore_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("nore", ug_per_min)
    
    def set_epi_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("epi", ug_per_min)
    
    def set_phenyl_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("phenyl", ug_per_min)

    def set_vaso_rate(self: "SimulationEngine", u_per_min: float):
        self._set_rate_from_user("vaso", u_per_min)

    def set_dobu_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("dobu", ug_per_min)

    def set_milri_rate(self: "SimulationEngine", ug_per_min: float):
        self._set_rate_from_user("milri", ug_per_min)
        
    def set_rocuronium_rate(self: "SimulationEngine", mg_per_hr: float):
        self._set_rate_from_user("roc", mg_per_hr)

    def enable_tci(self: "SimulationEngine", drug: str, target: float, mode: str = 'effect_site'):
        """
        Enable TCI for a drug.
        
        Args:
            drug: 'propofol', 'remi', 'nore', 'vaso', 'epi', 'dobu', 'milri', 'phenyl', 'roc'
            target: Target concentration
            mode: 'plasma' or 'effect_site' (ignored for nore/epi/phenyl which use plasma)
        """
        from .tci import TCIController

        sampling_time = 1.0
        if hasattr(self, "config") and getattr(self.config, "dt", None):
            sampling_time = self.config.dt
        control_time = max(10.0, sampling_time)
        
        drug = drug.lower()
        spec = self._DRUG_SPECS.get(drug)
        if not spec:
            return

        max_rate = self._tci_max_rate(spec)
        tci_attr = spec["tci_attr"]
        pk_attr = spec["pk_attr"]
        controller = getattr(self, tci_attr, None)
        tci_mode = spec["tci_mode"] if spec["tci_mode"] else mode

        if not controller:
            controller = TCIController(
                getattr(self, pk_attr),
                spec["tci_name"],
                tci_mode,
                max_rate=max_rate,
                sampling_time=sampling_time,
                control_time=control_time,
            )
            setattr(self, tci_attr, controller)

        controller.max_rate = max_rate
        self._seed_tci_state(controller, getattr(self, pk_attr))
        controller.target = target

    def _seed_tci_state(self: "SimulationEngine", controller, pk_model):
        """Seed TCI state from current PK concentrations to avoid bolus overshoot."""
        if not controller or not pk_model or not hasattr(pk_model, "state"):
            return
        state = pk_model.state
        c1 = getattr(state, "c1", 0.0)
        c2 = getattr(state, "c2", 0.0)
        c3 = getattr(state, "c3", 0.0)
        ce = getattr(state, "ce", 0.0)
        if controller.n_state == 1:
            controller.set_state(c1)
        elif controller.n_state == 2:
            controller.set_state(c1, c2)
        elif controller.n_state == 3:
            controller.set_state(c1, c2, ce)
        else:
            controller.set_state(c1, c2, c3, ce)

    def _tci_max_rate(self: "SimulationEngine", spec: dict) -> float:
        """Return a clinically realistic max infusion rate for a drug."""
        weight = max(0.0, getattr(self.patient, "weight", 0.0))
        if weight <= 0.0:
            return 0.0
        unit, value = spec["max_rate"]
        if unit == "mg_kg_min" or unit == "ug_kg_min":
            return weight * value / 60.0
        if unit == "mg_kg_hr":
            return weight * value / 3600.0
        if unit == "u_min":
            # Vasopressin: return mU/sec (model units)
            return (value * 1000.0) / 60.0
        return 0.0

    def disable_tci(self: "SimulationEngine", drug: str):
        """Disable TCI for a drug and reset infusion rate to zero."""
        drug = drug.lower()
        spec = self._DRUG_SPECS.get(drug)
        if not spec:
            return
        setattr(self, spec["tci_attr"], None)
        setattr(self, spec["rate_attr"], 0.0)

    def get_controllable_drugs(self: "SimulationEngine"):
        """
        Return metadata for all controllable drugs.
        Used by UI to build controls dynamically.
        """
        drugs = []
        for key in self._DRUG_ORDER:
            spec = self._DRUG_SPECS[key]
            drugs.append({
                "key": key,
                "name": spec["name"],
                "rate_unit": spec["rate_unit"],
                "tci_unit": spec["tci_unit"],
                "can_tci": True,
                "default_bolus": spec["default_bolus"],
                "bolus_unit": spec.get("bolus_unit"),
                "tci_range": spec.get("tci_range"),
            })
        return drugs

    def set_drug_rate(self: "SimulationEngine", key: str, rate_user_unit: float):
        """Generic setter for drug infusion rate."""
        self._set_rate_from_user(key, rate_user_unit)

    def set_drug_target(self: "SimulationEngine", key: str, target: float):
        """Generic setter for TCI target. None/Negative disables TCI."""
        if target is None or target < 0:
            self.disable_tci(key)
        else:
            self.enable_tci(key, target)

    def get_drug_state(self: "SimulationEngine", key: str):
        """Return current state generic dict."""
        state = {'rate': 0.0, 'target': 0.0, 'is_tci': False}
        
        spec = self._DRUG_SPECS.get(key)
        if not spec:
            return state

        state["rate"] = self._get_rate_to_user(key)
        controller = getattr(self, spec["tci_attr"], None)
        if controller:
            state["is_tci"] = True
            state["target"] = controller.target
                 
        return state

    def _set_rate_from_user(self: "SimulationEngine", key: str, rate_user_unit: float):
        spec = self._DRUG_SPECS.get(key)
        if not spec:
            return
        setattr(self, spec["rate_attr"], rate_user_unit / spec["rate_scale"])

    def _get_rate_to_user(self: "SimulationEngine", key: str) -> float:
        spec = self._DRUG_SPECS.get(key)
        if not spec:
            return 0.0
        return getattr(self, spec["rate_attr"], 0.0) * spec["rate_scale"]
