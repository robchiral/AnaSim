"""Typed metadata registry for controllable intravenous drugs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional


class TCIMode(str, Enum):
    """Supported TCI target compartments."""

    PLASMA = "plasma"
    EFFECT_SITE = "effect_site"


class MaxRateBasis(Enum):
    """Units used to express a clinical maximum infusion rate."""

    PER_KG_MINUTE = "per_kg_minute"
    PER_KG_HOUR = "per_kg_hour"
    ABSOLUTE_PER_MINUTE = "absolute_per_minute"


@dataclass(frozen=True, slots=True)
class MaxRatePolicy:
    """Convert a clinical infusion limit to the PK model's per-second units."""

    basis: MaxRateBasis
    value: float
    model_unit_scale: float = 1.0

    def internal_rate(self, weight_kg: float) -> float:
        weight_kg = max(0.0, weight_kg)
        if self.basis is MaxRateBasis.PER_KG_MINUTE:
            return weight_kg * self.value * self.model_unit_scale / 60.0
        if self.basis is MaxRateBasis.PER_KG_HOUR:
            return weight_kg * self.value * self.model_unit_scale / 3600.0
        if self.basis is MaxRateBasis.ABSOLUTE_PER_MINUTE:
            return self.value * self.model_unit_scale / 60.0
        raise ValueError(f"Unsupported max-rate basis: {self.basis}")


@dataclass(frozen=True, slots=True)
class DrugSpec:
    """Complete controller, PK, bolus, and UI metadata for one drug."""

    key: str
    name: str
    rate_attr: str
    rate_unit: str
    internal_rate_unit: str
    bolus_unit: str
    default_bolus: float
    bolus_model_scale: float
    pk_attr: str
    tci_attr: str
    tci_name: str
    tci_unit: str
    tci_range: tuple[float, float]
    fixed_tci_mode: Optional[TCIMode]
    max_rate: MaxRatePolicy


DRUG_REGISTRY = (
    DrugSpec(
        key="propofol",
        name="Propofol 10 mg/mL",
        rate_attr="propofol_rate_mg_sec",
        rate_unit="mg/hr",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=150.0,
        bolus_model_scale=1.0,
        pk_attr="pk_prop",
        tci_attr="tci_prop",
        tci_name="Propofol",
        tci_unit="mcg/mL",
        tci_range=(0.0, 10.0),
        fixed_tci_mode=None,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.3),
    ),
    DrugSpec(
        key="remi",
        name="Remifentanil 50 mcg/mL",
        rate_attr="remi_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_attr="pk_remi",
        tci_attr="tci_remi",
        tci_name="Remifentanil",
        tci_unit="ng/mL",
        tci_range=(0.0, 10.0),
        fixed_tci_mode=None,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.5),
    ),
    DrugSpec(
        key="nore",
        name="Norepinephrine 16 mcg/mL",
        rate_attr="nore_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_attr="pk_nore",
        tci_attr="tci_nore",
        tci_name="Norepinephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 30.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 1.0),
    ),
    DrugSpec(
        key="vaso",
        name="Vasopressin 20 U/mL",
        rate_attr="vaso_rate_mu_sec",
        rate_unit="U/min",
        internal_rate_unit="mU/sec",
        bolus_unit="U",
        default_bolus=1.0,
        bolus_model_scale=1000.0,
        pk_attr="pk_vaso",
        tci_attr="tci_vaso",
        tci_name="Vasopressin",
        tci_unit="mU/L",
        tci_range=(0.0, 80.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(
            MaxRateBasis.ABSOLUTE_PER_MINUTE,
            0.1,
            model_unit_scale=1000.0,
        ),
    ),
    DrugSpec(
        key="phenyl",
        name="Phenylephrine 100 mcg/mL",
        rate_attr="phenyl_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=100.0,
        bolus_model_scale=1.0,
        pk_attr="pk_phenyl",
        tci_attr="tci_phenyl",
        tci_name="Phenylephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 120.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 2.0),
    ),
    DrugSpec(
        key="epi",
        name="Epinephrine 100 mcg/mL",
        rate_attr="epi_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_attr="pk_epi",
        tci_attr="tci_epi",
        tci_name="Epinephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 20.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.5),
    ),
    DrugSpec(
        key="dobu",
        name="Dobutamine 1 mg/mL",
        rate_attr="dobu_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=0.0,
        bolus_model_scale=1.0,
        pk_attr="pk_dobu",
        tci_attr="tci_dobu",
        tci_name="Dobutamine",
        tci_unit="ng/mL",
        tci_range=(0.0, 500.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 20.0),
    ),
    DrugSpec(
        key="milri",
        name="Milrinone 200 mcg/mL",
        rate_attr="mil_rate_ug_sec",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=0.0,
        bolus_model_scale=1.0,
        pk_attr="pk_mil",
        tci_attr="tci_mil",
        tci_name="Milrinone",
        tci_unit="ng/mL",
        tci_range=(0.0, 500.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.75),
    ),
    DrugSpec(
        key="roc",
        name="Rocuronium 10 mg/mL",
        rate_attr="roc_rate_mg_sec",
        rate_unit="mg/hr",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=50.0,
        bolus_model_scale=1.0,
        pk_attr="pk_roc",
        tci_attr="tci_roc",
        tci_name="Rocuronium",
        tci_unit="mcg/mL",
        tci_range=(0.0, 10.0),
        fixed_tci_mode=None,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_HOUR, 1.0),
    ),
)


def _ensure_unique_attribute(attribute: str) -> None:
    seen = set()
    for spec in DRUG_REGISTRY:
        value = getattr(spec, attribute)
        if value in seen:
            raise ValueError(f"Drug registry contains duplicate {attribute}: {value!r}")
        seen.add(value)


def _bolus_index() -> dict[str, DrugSpec]:
    index: dict[str, DrugSpec] = {}
    for spec in DRUG_REGISTRY:
        for alias in (spec.key, spec.name, spec.tci_name):
            normalized = alias.strip().casefold()
            existing = index.get(normalized)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"Drug registry contains duplicate bolus alias: {alias!r}"
                )
            index[normalized] = spec
    return index


for _attribute in ("key", "rate_attr", "pk_attr", "tci_attr"):
    _ensure_unique_attribute(_attribute)
for _spec in DRUG_REGISTRY:
    if _spec.key != _spec.key.strip().casefold():
        raise ValueError(f"Drug key must be normalized: {_spec.key!r}")
    if _spec.tci_range[0] < 0.0 or _spec.tci_range[0] >= _spec.tci_range[1]:
        raise ValueError(f"Invalid TCI range for {_spec.key}: {_spec.tci_range!r}")
    if _spec.default_bolus < 0.0 or _spec.bolus_model_scale <= 0.0:
        raise ValueError(f"Invalid bolus metadata for {_spec.key}")

DRUGS_BY_KEY = MappingProxyType({spec.key: spec for spec in DRUG_REGISTRY})
DRUGS_BY_BOLUS_NAME = MappingProxyType(_bolus_index())

PK_HEMODYNAMIC_TARGETS = tuple((spec.key, spec.pk_attr) for spec in DRUG_REGISTRY)
TCI_TARGET_CONFIG = tuple((spec.tci_attr, spec.rate_attr) for spec in DRUG_REGISTRY)


def get_drug_spec(key: str) -> DrugSpec:
    """Return the canonical spec for a controller key."""
    try:
        return DRUGS_BY_KEY[key.strip().casefold()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unknown controllable drug: {key!r}") from exc


def resolve_bolus_drug(name: str) -> DrugSpec:
    """Resolve a canonical key, UI name, or clinical drug name for bolus delivery."""
    try:
        return DRUGS_BY_BOLUS_NAME[name.strip().casefold()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unknown bolus drug: {name!r}") from exc
