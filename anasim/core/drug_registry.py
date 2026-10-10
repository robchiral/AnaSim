"""Typed metadata registry for controllable intravenous drugs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType

from anasim.patient.patient import Patient
from anasim.patient.pk_models import (
    DobutaminePK,
    EpinephrinePK,
    EsmololPK,
    EtomidatePK,
    FentanylPK,
    GlycopyrrolatePK,
    KetaminePK,
    LabetalolPK,
    LidocainePK,
    MammillaryPK,
    MidazolamPK,
    MilrinonePK,
    NorepinephrinePK,
    PhenylephrinePK,
    PropofolPKEleveld,
    RemifentanilPKEleveld,
    RocuroniumPK,
    VasopressinPK,
)

from .state import SimulationConfig, SimulationState

PKFactory = Callable[[Patient, SimulationConfig], MammillaryPK]


def _from_patient(model: Callable[[Patient], MammillaryPK]) -> PKFactory:
    return lambda patient, _config: model(patient)


def _propofol(patient: Patient, config: SimulationConfig) -> MammillaryPK:
    return PropofolPKEleveld(patient, concomitant_opioids=config.concomitant_opioids)


class TCIMode(str, Enum):
    """Supported TCI target compartments."""

    PLASMA = "plasma"
    EFFECT_SITE = "effect_site"


class MaxRateBasis(Enum):
    """Units used to express a TCI pump rate limit."""

    PER_KG_MINUTE = "per_kg_minute"
    PER_KG_HOUR = "per_kg_hour"
    ABSOLUTE_PER_MINUTE = "absolute_per_minute"


@dataclass(frozen=True, slots=True)
class MaxRatePolicy:
    """Convert a TCI rate limit to the PK model's per-second units."""

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
    """Complete PK, bolus, infusion, TCI, and UI metadata for one drug.

    Infusion fields are None for bolus-only drugs, and TCI fields are None for
    drugs given by bolus or manual rate only. ce_field and cp_field name the
    SimulationState fields that receive the effect-site and plasma
    concentrations.
    """

    key: str
    name: str
    generic_name: str
    pk_model: PKFactory
    bolus_unit: str
    default_bolus: float
    bolus_model_scale: float
    ce_field: str | None = None
    cp_field: str | None = None
    rate_unit: str | None = None
    internal_rate_unit: str | None = None
    tci_unit: str | None = None
    tci_range: tuple[float, float] | None = None
    fixed_tci_mode: TCIMode | None = None
    max_rate: MaxRatePolicy | None = None

    @property
    def has_infusion(self) -> bool:
        return self.rate_unit is not None

    @property
    def has_tci(self) -> bool:
        return self.tci_unit is not None


DRUG_REGISTRY = (
    DrugSpec(
        key="propofol",
        name="Propofol 10 mg/mL",
        rate_unit="mcg/kg/min",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=150.0,
        bolus_model_scale=1.0,
        pk_model=_propofol,
        ce_field="propofol_ce",
        cp_field="propofol_cp",
        generic_name="Propofol",
        tci_unit="mcg/mL",
        tci_range=(0.0, 10.0),
        fixed_tci_mode=None,
        # Syringe-pump limit: 1200 mL/h of 10 mg/mL.
        max_rate=MaxRatePolicy(MaxRateBasis.ABSOLUTE_PER_MINUTE, 200.0),
    ),
    DrugSpec(
        key="remi",
        name="Remifentanil 50 mcg/mL",
        rate_unit="mcg/kg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(RemifentanilPKEleveld),
        ce_field="remi_ce",
        cp_field="remi_cp",
        generic_name="Remifentanil",
        tci_unit="ng/mL",
        tci_range=(0.0, 10.0),
        fixed_tci_mode=None,
        # Syringe-pump limit: 1200 mL/h of 50 mcg/mL.
        max_rate=MaxRatePolicy(MaxRateBasis.ABSOLUTE_PER_MINUTE, 1000.0),
    ),
    DrugSpec(
        key="fentanyl",
        name="Fentanyl 50 mcg/mL",
        rate_unit="mcg/hr",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=50.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(FentanylPK),
        ce_field="fentanyl_ce",
        cp_field="fentanyl_cp",
        generic_name="Fentanyl",
        tci_unit="ng/mL",
        tci_range=(0.0, 10.0),
        # Syringe-pump limit: 1200 mL/h of 50 mcg/mL.
        max_rate=MaxRatePolicy(MaxRateBasis.ABSOLUTE_PER_MINUTE, 1000.0),
    ),
    DrugSpec(
        key="midazolam",
        name="Midazolam 1 mg/mL",
        rate_unit="mg/hr",
        internal_rate_unit="ug/sec",
        bolus_unit="mg",
        default_bolus=2.0,
        bolus_model_scale=1000.0,
        pk_model=_from_patient(MidazolamPK),
        ce_field="midazolam_ce",
        generic_name="Midazolam",
    ),
    DrugSpec(
        key="etomidate",
        name="Etomidate 2 mg/mL",
        bolus_unit="mg",
        default_bolus=20.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(EtomidatePK),
        ce_field="etomidate_ce",
        generic_name="Etomidate",
    ),
    DrugSpec(
        key="ketamine",
        name="Ketamine 10 mg/mL",
        rate_unit="mg/hr",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=50.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(KetaminePK),
        ce_field="ketamine_ce",
        generic_name="Ketamine",
    ),
    DrugSpec(
        key="lidocaine",
        name="Lidocaine 20 mg/mL",
        rate_unit="mg/hr",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=100.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(LidocainePK),
        ce_field="lidocaine_ce",
        generic_name="Lidocaine",
    ),
    DrugSpec(
        key="nore",
        name="Norepinephrine 16 mcg/mL",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(NorepinephrinePK),
        ce_field="nore_ce",
        generic_name="Norepinephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 30.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 1.0),
    ),
    DrugSpec(
        key="vaso",
        name="Vasopressin 20 U/mL",
        rate_unit="U/min",
        internal_rate_unit="mU/sec",
        bolus_unit="U",
        default_bolus=1.0,
        bolus_model_scale=1000.0,
        pk_model=_from_patient(VasopressinPK),
        ce_field="vaso_ce",
        generic_name="Vasopressin",
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
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=100.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(PhenylephrinePK),
        ce_field="phenyl_ce",
        generic_name="Phenylephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 120.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 2.0),
    ),
    DrugSpec(
        key="epi",
        name="Epinephrine 100 mcg/mL",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=10.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(EpinephrinePK),
        ce_field="epi_ce",
        generic_name="Epinephrine",
        tci_unit="ng/mL",
        tci_range=(0.0, 20.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.5),
    ),
    DrugSpec(
        key="dobu",
        name="Dobutamine 1 mg/mL",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=0.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(DobutaminePK),
        ce_field="dobu_ce",
        generic_name="Dobutamine",
        tci_unit="ng/mL",
        tci_range=(0.0, 500.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 20.0),
    ),
    DrugSpec(
        key="milri",
        name="Milrinone 200 mcg/mL",
        rate_unit="mcg/min",
        internal_rate_unit="ug/sec",
        bolus_unit="mcg",
        default_bolus=0.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(MilrinonePK),
        ce_field="mil_ce",
        generic_name="Milrinone",
        tci_unit="ng/mL",
        tci_range=(0.0, 500.0),
        fixed_tci_mode=TCIMode.PLASMA,
        max_rate=MaxRatePolicy(MaxRateBasis.PER_KG_MINUTE, 0.75),
    ),
    DrugSpec(
        key="esmolol",
        name="Esmolol 10 mg/mL",
        rate_unit="mg/min",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=30.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(EsmololPK),
        ce_field="esmolol_ce",
        generic_name="Esmolol",
    ),
    DrugSpec(
        key="labetalol",
        name="Labetalol 5 mg/mL",
        bolus_unit="mg",
        default_bolus=10.0,
        bolus_model_scale=1000.0,
        pk_model=_from_patient(LabetalolPK),
        ce_field="labetalol_ce",
        generic_name="Labetalol",
    ),
    DrugSpec(
        key="glyco",
        name="Glycopyrrolate 0.2 mg/mL",
        bolus_unit="mg",
        default_bolus=0.2,
        bolus_model_scale=1000.0,
        pk_model=_from_patient(GlycopyrrolatePK),
        ce_field="glyco_ce",
        generic_name="Glycopyrrolate",
    ),
    DrugSpec(
        key="roc",
        name="Rocuronium 10 mg/mL",
        rate_unit="mg/hr",
        internal_rate_unit="mg/sec",
        bolus_unit="mg",
        default_bolus=50.0,
        bolus_model_scale=1.0,
        pk_model=_from_patient(RocuroniumPK),
        cp_field="roc_cp",
        generic_name="Rocuronium",
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
        if value is None:
            continue
        if value in seen:
            raise ValueError(f"Drug registry contains duplicate {attribute}: {value!r}")
        seen.add(value)


def _bolus_index() -> dict[str, DrugSpec]:
    index: dict[str, DrugSpec] = {}
    for spec in DRUG_REGISTRY:
        for alias in (spec.key, spec.name, spec.generic_name):
            normalized = alias.strip().casefold()
            existing = index.get(normalized)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"Drug registry contains duplicate bolus alias: {alias!r}"
                )
            index[normalized] = spec
    return index


_STATE_FIELDS = {field.name for field in fields(SimulationState)}
for _attribute in ("key", "ce_field", "cp_field"):
    _ensure_unique_attribute(_attribute)
for _spec in DRUG_REGISTRY:
    if _spec.key != _spec.key.strip().casefold():
        raise ValueError(f"Drug key must be normalized: {_spec.key!r}")
    infusion_fields = (_spec.rate_unit, _spec.internal_rate_unit)
    if any(field is None for field in infusion_fields) != all(field is None for field in infusion_fields):
        raise ValueError(f"Incomplete infusion metadata for {_spec.key}")
    if _spec.has_tci:
        if not _spec.has_infusion or _spec.max_rate is None:
            raise ValueError(f"Incomplete TCI metadata for {_spec.key}")
        if _spec.tci_range is None or _spec.tci_range[0] < 0.0 or _spec.tci_range[0] >= _spec.tci_range[1]:
            raise ValueError(f"Invalid TCI range for {_spec.key}: {_spec.tci_range!r}")
    if _spec.default_bolus < 0.0 or _spec.bolus_model_scale <= 0.0:
        raise ValueError(f"Invalid bolus metadata for {_spec.key}")
    for _field in (_spec.ce_field, _spec.cp_field):
        if _field is not None and _field not in _STATE_FIELDS:
            raise ValueError(f"{_spec.key} projects to unknown state field {_field!r}")

DRUGS_BY_KEY = MappingProxyType({spec.key: spec for spec in DRUG_REGISTRY})
DRUGS_BY_BOLUS_NAME = MappingProxyType(_bolus_index())


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
