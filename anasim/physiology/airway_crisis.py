"""Difficult airway anatomy, laryngoscopy trauma, mask adjuncts, and esophageal intubation.

Pathway structure follows the 2022 ASA Practice Guidelines for Management of
the Difficult Airway (Apfelbaum 2022), DAS 2025 adult guidelines (Ahmad 2026),
and the PUMA consensus guideline on preventing unrecognised oesophageal
intubation (Chrimes 2022). Attempt-dependent airway trauma and hypoxemia
escalation reflect Mort 2004b; esophageal capnography washout and returned
expired volume follow Linko 1983 and Sum-Ping 1989; two-handed jaw-thrust
mask ventilation follows Joffe 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01
from anasim.physiology.resp_mech import RespiratoryMechanics

DIFFICULT_AIRWAY_PROFILES = ("none", "anterior_dl", "severe_cico")
INTUBATION_TECHNIQUES = ("dl", "vl", "vl_bougie", "fiberoptic")

INTUBATION_TECHNIQUE_LABELS = {
    "dl": "Direct laryngoscopy (DL)",
    "vl": "Video laryngoscopy (VL)",
    "vl_bougie": "VL + bougie",
    "fiberoptic": "Flexible bronchoscopy",
}

# Sum-Ping 1989 (n=13 volume arm): esophageal expired tidal volume averaged
# 35-46 mL (range 15-80 mL, ~6-7.5% mean, up to 13% range) on 615 +/- 64 mL set
# VT; return fraction, compliance, and resistance below are model calibrations.
ESOPHAGEAL_RETURN_FRACTION = 0.14
ESOPHAGEAL_COMPLIANCE_L_CMH2O = 0.028
ESOPHAGEAL_RESISTANCE_CMH2O_L_S = 14.0
# Linko 1983 and Chrimes 2022 (PUMA): prior mask ventilation can insufflate
# exhaled CO2 into the stomach (0-1.5 vol% after routine mask ventilation,
# up to 4.4-4.9 vol% after simulated esophageal insufflation) that decays
# rapidly across breaths and fails the PUMA >=7 breaths >7.5 mmHg criterion.
ESOPHAGEAL_DEFAULT_CO2_MMHG = 18.0
ESOPHAGEAL_CO2_BREATH_DECAY = 0.50
ESOPHAGEAL_CO2_FLOOR_MMHG = 0.6

TRAUMA_DECAY_TAU_S = 600.0
TRAUMA_MAX = 0.85
# Mort 2004b: outside-OR complications climb steeply after >2 conventional attempts.
_TRAUMA_STEPS = (0.08, 0.12, 0.18, 0.22)


@dataclass(frozen=True, slots=True)
class IntubationAttemptResult:
    placed: bool
    esophageal: bool
    view: str
    code: str | None = None
    message: str = ""
    trauma_added: float = 0.0


class AirwayCrisisModel:
    """State for difficult airway profiles, mask adjuncts, trauma, and esophageal intubation."""

    def __init__(self) -> None:
        self.difficult_profile: str = "none"
        self.laryngoscopy_attempts: int = 0
        self.last_view: str = ""
        self.trauma_obstruction: float = 0.0
        self.help_called: bool = False
        self.oral_airway: bool = False
        self.two_hand_mask: bool = False
        self.esophageal_armed: bool = False
        self.esophageal_intubation: bool = False
        self.esophageal_co2_mmhg: float = 0.0
        self.fona_performed: bool = False
        self._last_vent_breath_count: int = 0
        self.esophageal_mech = RespiratoryMechanics(
            compliance=ESOPHAGEAL_COMPLIANCE_L_CMH2O,
            resistance=ESOPHAGEAL_RESISTANCE_CMH2O_L_S,
            aeration=None,
        )

    @property
    def base_pathology_obstruction(self) -> float:
        """Intrinsic upper-airway narrowing before mask adjuncts and laryngoscopy trauma."""
        if self.difficult_profile == "severe_cico":
            return 0.78
        if self.difficult_profile == "anterior_dl":
            return 0.18
        return 0.0

    def set_difficult_profile(self, profile: str) -> None:
        if not isinstance(profile, str):
            choices = ", ".join(DIFFICULT_AIRWAY_PROFILES)
            raise ValueError(f"Unsupported difficult airway profile {profile!r}; choose one of: {choices}")
        norm = profile.strip().lower()
        if norm not in DIFFICULT_AIRWAY_PROFILES:
            choices = ", ".join(DIFFICULT_AIRWAY_PROFILES)
            raise ValueError(f"Unsupported difficult airway profile {profile!r}; choose one of: {choices}")
        self.difficult_profile = norm
        if norm == "none":
            self.trauma_obstruction = 0.0
            self.laryngoscopy_attempts = 0
            self.last_view = ""

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        if self.trauma_obstruction > 0.0:
            self.trauma_obstruction *= math.exp(-dt / TRAUMA_DECAY_TAU_S)
            if self.trauma_obstruction < 1e-4:
                self.trauma_obstruction = 0.0

    def _add_attempt_trauma(self) -> float:
        idx = min(len(_TRAUMA_STEPS) - 1, max(0, self.laryngoscopy_attempts - 1))
        delta = _TRAUMA_STEPS[idx]
        before = self.trauma_obstruction
        self.trauma_obstruction = clamp(before + delta, 0.0, TRAUMA_MAX)
        return self.trauma_obstruction - before

    def evaluate_intubation(
        self,
        technique: str = "dl",
        *,
        prior_mask_ventilation: bool = False,
        vent_breath_count: int = 0,
    ) -> IntubationAttemptResult:
        """Evaluate anatomical/device outcome of a laryngoscopy or fiberoptic attempt."""
        if not isinstance(technique, str):
            choices = ", ".join(INTUBATION_TECHNIQUES)
            raise ValueError(f"Unsupported intubation technique {technique!r}; choose one of: {choices}")
        tech = technique.strip().lower()
        if tech not in INTUBATION_TECHNIQUES:
            choices = ", ".join(INTUBATION_TECHNIQUES)
            raise ValueError(f"Unsupported intubation technique {technique!r}; choose one of: {choices}")

        self.laryngoscopy_attempts += 1

        if self.difficult_profile == "severe_cico":
            trauma = self._add_attempt_trauma()
            view = "Grade IV (severe edema)" if tech != "fiberoptic" else "Obscured glottis (FOI)"
            self.last_view = view
            return IntubationAttemptResult(
                placed=False,
                esophageal=False,
                view=view,
                code="failed_laryngoscopy",
                message=(
                    f"{INTUBATION_TECHNIQUE_LABELS[tech]}: {view}; intubation failed. "
                    "Oxygenation is critical; prepare emergency front-of-neck airway (eFONA)."
                ),
                trauma_added=trauma,
            )

        if self.difficult_profile == "anterior_dl":
            if tech == "dl":
                trauma = self._add_attempt_trauma()
                view = "Grade IV" if self.trauma_obstruction >= 0.35 else "Grade III"
                self.last_view = view
                return IntubationAttemptResult(
                    placed=False,
                    esophageal=False,
                    view=view,
                    code="failed_laryngoscopy",
                    message=(
                        f"Direct laryngoscopy: Cormack-Lehane {view} (anterior glottis not seen); "
                        "tube not placed. Reoxygenate by mask, call for help, and switch to VL + bougie, "
                        "flexible bronchoscopy, or SGA."
                    ),
                    trauma_added=trauma,
                )
            if tech == "vl":
                if self.trauma_obstruction >= 0.35:
                    trauma = self._add_attempt_trauma()
                    view = "Grade IIb (anterior + edema)"
                    self.last_view = view
                    return IntubationAttemptResult(
                        placed=False,
                        esophageal=False,
                        view=view,
                        code="failed_laryngoscopy",
                        message=(
                            "Video laryngoscopy: Grade IIb anterior view with airway edema; "
                            "tube could not pass anteriorly. Use VL + bougie, SGA, or fiberoptic."
                        ),
                        trauma_added=trauma,
                    )
                self.clear_esophageal()
                self.esophageal_armed = False
                self.last_view = "Grade IIa"
                return IntubationAttemptResult(
                    placed=True,
                    esophageal=False,
                    view="Grade IIa",
                    message="Video laryngoscopy: Grade IIa view; tracheal tube placed.",
                )
            if tech == "vl_bougie":
                if self.trauma_obstruction >= 0.65:
                    trauma = self._add_attempt_trauma()
                    view = "Grade III (severe trauma)"
                    self.last_view = view
                    return IntubationAttemptResult(
                        placed=False,
                        esophageal=False,
                        view=view,
                        code="failed_laryngoscopy",
                        message=(
                            "VL + bougie: severe airway edema/bleeding from repeated attempts; "
                            "rescue with SGA or emergency cricothyrotomy (eFONA)."
                        ),
                        trauma_added=trauma,
                    )
                self.clear_esophageal()
                self.esophageal_armed = False
                self.last_view = "Grade IIa"
                return IntubationAttemptResult(
                    placed=True,
                    esophageal=False,
                    view="Grade IIa",
                    message="VL + bougie: Grade IIa anterior view; bougie guided tracheal tube placement.",
                )
            # fiberoptic
            if self.trauma_obstruction >= 0.55:
                trauma = self._add_attempt_trauma()
                view = "Soiled view (FOI)"
                self.last_view = view
                return IntubationAttemptResult(
                    placed=False,
                    esophageal=False,
                    view=view,
                    code="failed_laryngoscopy",
                    message=(
                        "Flexible bronchoscopy: bleeding and edema from prior attempts obscure view; "
                        "rescue with SGA or eFONA."
                    ),
                    trauma_added=trauma,
                )
            self.clear_esophageal()
            self.esophageal_armed = False
            self.last_view = "Glottis visualized (FOI)"
            return IntubationAttemptResult(
                placed=True,
                esophageal=False,
                view="Glottis visualized (FOI)",
                message="Flexible bronchoscopy: vocal cords and tracheal rings confirmed; tube placed.",
            )

        # Normal airway ("none")
        if self.esophageal_armed and tech == "dl":
            self.esophageal_armed = False
            gastric_co2 = ESOPHAGEAL_DEFAULT_CO2_MMHG if prior_mask_ventilation else 0.0
            self.activate_esophageal(gastric_co2_mmhg=gastric_co2, vent_breath_count=vent_breath_count)
            self.last_view = "Grade III"
            return IntubationAttemptResult(
                placed=True,
                esophageal=True,
                view="Grade III",
                message="Direct laryngoscopy: Grade III view; tube passed blindly; confirm with capnography.",
            )

        if self.esophageal_armed and tech != "dl":
            # Video laryngoscopy or bronchoscopy directly visualizes the glottis and prevents esophageal placement.
            self.esophageal_armed = False

        self.clear_esophageal()
        view = "Glottis visualized (FOI)" if tech == "fiberoptic" else "Grade I"
        self.last_view = view
        return IntubationAttemptResult(
            placed=True,
            esophageal=False,
            view=view,
            message="",
        )

    def evaluate_sga(self) -> tuple[bool, str]:
        """Evaluate supraglottic airway seating against current airway pathology/trauma."""
        if self.difficult_profile == "severe_cico" or self.trauma_obstruction >= 0.70:
            return (
                False,
                "SGA failed to seat against severe glottic/supraglottic obstruction (CICO); "
                "proceed to emergency cricothyrotomy (eFONA).",
            )
        return True, ""

    def activate_esophageal(
        self,
        *,
        gastric_co2_mmhg: float = ESOPHAGEAL_DEFAULT_CO2_MMHG,
        vent_breath_count: int = 0,
    ) -> None:
        self.esophageal_intubation = True
        self.esophageal_co2_mmhg = clamp(float(gastric_co2_mmhg), 0.0, 40.0)
        self._last_vent_breath_count = int(vent_breath_count)
        self.esophageal_mech.volume = 0.0
        self.esophageal_mech.p2 = 0.0
        self.esophageal_mech.peep = 0.0
        self.esophageal_mech.effort.set_drive(0.0, 0.0)

    def clear_esophageal(self) -> None:
        self.esophageal_intubation = False
        self.esophageal_co2_mmhg = 0.0
        self.esophageal_mech.volume = 0.0
        self.esophageal_mech.p2 = 0.0
        self.esophageal_mech.peep = 0.0

    def sync_esophageal_breaths(self, vent_breath_count: int) -> None:
        """Advance esophageal CO2 washout on completed ventilator or bag breaths."""
        if not self.esophageal_intubation:
            self._last_vent_breath_count = int(vent_breath_count)
            return
        delta = int(vent_breath_count) - self._last_vent_breath_count
        if delta <= 0:
            return
        self._last_vent_breath_count = int(vent_breath_count)
        if self.esophageal_co2_mmhg > 0.0:
            self.esophageal_co2_mmhg *= ESOPHAGEAL_CO2_BREATH_DECAY ** delta
            if self.esophageal_co2_mmhg < ESOPHAGEAL_CO2_FLOOR_MMHG:
                self.esophageal_co2_mmhg = 0.0

    def mask_collapse_multiplier(self) -> float:
        """Fractional reduction of sedated pharyngeal collapse from OPA and two-handed jaw thrust."""
        mult = 1.0
        if self.oral_airway:
            mult *= 0.40
        if self.two_hand_mask:
            mult *= 0.35
        return mult

    def mask_pathology_obstruction(self) -> float:
        """Effective supraglottic narrowing during mask ventilation after adjuncts (Joffe 2010)."""
        raw = clamp01(max(self.base_pathology_obstruction, self.trauma_obstruction))
        if raw <= 0.0:
            return 0.0
        relief = 0.0
        if self.oral_airway:
            relief += 0.25
        if self.two_hand_mask:
            relief += 0.30
        if self.difficult_profile == "severe_cico":
            # Fixed glottic/subglottic CICO pathology responds minimally to jaw thrust / OPA.
            relief *= 0.25
        return clamp01(raw * (1.0 - clamp01(relief)))

    def sga_pathology_obstruction(self) -> float:
        """Residual supraglottic obstruction under an SGA (bypasses pharyngeal/tongue base collapse)."""
        if self.difficult_profile == "severe_cico":
            return clamp01(max(self.base_pathology_obstruction, self.trauma_obstruction))
        return clamp01(0.30 * self.trauma_obstruction)
