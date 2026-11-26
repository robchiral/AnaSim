"""Spinal and epidural neuraxial anesthesia model, including high and total spinal block.

Age-related cephalad spread of 0.5% hyperbaric bupivacaine follows Veering 1988.
Sensory-motor dissociation (2.8 segments spinal, 4.6 epidural) and diaphragmatic
sparing up to T1 motor block follow Freund 1967 (lidocaine extended to
bupivacaine). Hemodynamic responses for spinal, plain epidural, and
epinephrine-containing epidural follow Ward 1965 and Rooke 1997 (lidocaine
extended to bupivacaine). High-spinal cardioaccelerator blockade, preload-
sensitive Bezold-Jarisch vagal bradycardia, and arrest follow Pollard 2001 and
Caplan 1988.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01

NEURAXIAL_DRUGS = ("bupivacaine", "lidocaine")
SPINAL_BARICITIES = ("hyperbaric", "isobaric")

# Continuous dermatome scale: 0 = None, 1..5 = S5..S1, 6..10 = L5..L1,
# 11..22 = T12..T1, 23..29 = C8..C2, 30 = Brainstem (total spinal).
_DERMATOME_NAMES = (
    "None",
    "S5", "S4", "S3", "S2", "S1",
    "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
    "C8", "C7", "C6", "C5", "C4", "C3", "C2",
    "Brainstem",
)

_SITE_REQUIRED_LEVEL = {
    "other": 13.0,            # T10 (lower extremity / pelvic)
    "lower_abdominal": 17.0,  # T6 (cesarean / lower abdominal)
    "upper_abdominal": 19.0,  # T4 (upper abdominal)
}


def dermatome_label(level: float) -> str:
    """Return clinical dermatome label for a continuous block height (0-30)."""
    if level < 0.5:
        return "None"
    idx = int(clamp(round(level), 1, len(_DERMATOME_NAMES) - 1))
    return _DERMATOME_NAMES[idx]


@dataclass(frozen=True, slots=True)
class NeuraxialLevelCheck:
    """Bedside sensory (cold/pinprick) and modified Bromage motor assessment."""

    time: float
    sensory_level: float
    sensory_label: str
    motor_level: float
    motor_label: str
    bromage: int
    mode: str
    note: str


class NeuraxialModel:
    """Spinal bolus, epidural catheter bolus/infusion, and high/total spinal physiology."""

    def __init__(self, age: float = 40.0, *, pregnant: bool = False) -> None:
        self.age = float(age)
        self.pregnant = bool(pregnant)
        # Epidural catheter, infusion settings, and recent delivery mode metadata.
        self.epidural_catheter: bool = False
        self.epidural_infusion_ml_hr: float = 0.0
        self.epidural_infusion_conc_pct: float = 0.125
        self.epidural_infusion_drug: str = "bupivacaine"
        self.last_mode: str = "none"  # "none", "spinal", "epidural", "high_spinal"
        self.last_drug: str = ""

        # Continuous target and actual sensory block levels (0-30) for spinal and epidural compartments.
        self._spinal_depot: float = 0.0       # Residual subarachnoid drug pool (level units)
        self._spinal_level: float = 0.0       # Established spinal sensory height (0-30)
        self._spinal_tau_on: float = 240.0    # s (Veering 1988: ~4 min onset, 11-14 min peak)
        self._spinal_tau_off: float = 5400.0  # s (~90-105 min 2-seg regression for bupivacaine)

        self._epidural_depot: float = 0.0     # Epidural space drug pool (level units)
        self._epidural_level: float = 0.0     # Established epidural sensory height (0-30)
        self._epidural_density: float = 0.55  # Block density (0.25-0.65) relative to spinal (1.0)
        self._epidural_tau_on: float = 420.0  # s (slower onset than spinal; Ward 1965)
        self._epidural_tau_off: float = 4800.0

        # Systemically absorbed epinephrine from 1:200,000 epidural solutions (Ward 1965).
        self._epi_depot_ng_ml: float = 0.0
        self.absorbed_epi_ng_ml: float = 0.0

        self.last_check: NeuraxialLevelCheck | None = None

    @property
    def active(self) -> bool:
        return (
            self.sensory_level >= 0.5
            or self._spinal_depot > 1e-4
            or self._epidural_depot > 1e-4
        )

    @property
    def sensory_level(self) -> float:
        return clamp(max(self._spinal_level, self._epidural_level), 0.0, 30.0)

    @property
    def is_spinal_dominant(self) -> bool:
        return self._spinal_level >= self._epidural_level and self._spinal_level >= 0.5

    @property
    def density(self) -> float:
        if self.sensory_level < 0.5:
            return 0.0
        if self.is_spinal_dominant:
            return 1.0
        return clamp(self._epidural_density, 0.20, 0.70)

    @property
    def sympathetic_level(self) -> float:
        """Calibrated sympathetic level extending ~2 dermatomes cephalad to sensory height."""
        s = self.sensory_level
        return min(30.0, s + 2.0) if s >= 0.5 else 0.0

    @property
    def motor_level(self) -> float:
        """Motor block lags sensory by 2.8 segments for spinal and 4.6 for epidural (Freund 1967)."""
        s = self.sensory_level
        if s < 0.5:
            return 0.0
        lag = 2.8 if self.is_spinal_dominant else 4.6
        return max(0.0, s - lag)

    @property
    def bromage_score(self) -> int:
        m = self.motor_level
        if m < 6.0:
            return 0
        if m < 8.5:
            return 1
        if m < 10.5:
            return 2
        return 3

    @property
    def sensory_label(self) -> str:
        return dermatome_label(self.sensory_level)

    @property
    def motor_label(self) -> str:
        return dermatome_label(self.motor_level)

    @property
    def vasomotor_symp_fraction(self) -> float:
        """Fraction of thoracolumbar vasomotor/venous capacitance outflow blocked (L3-T4, 0-1)."""
        symp = self.sympathetic_level
        if symp <= 8.0:
            return 0.0
        return clamp01((symp - 8.0) / 11.0)

    @property
    def cardioaccelerator_block(self) -> float:
        """Fraction of T1-T4 cardiac sympathetic accelerator fibers blocked (T4-C7, 0-1)."""
        symp = self.sympathetic_level
        if symp <= 19.0:
            return 0.0
        return clamp01((symp - 19.0) / 5.0) * self.density

    @property
    def venous_pool_fraction(self) -> float:
        """Blood volume fraction pooled into unstressed capacitance (Ward 1965; Pollard 2001)."""
        return 0.115 * self.vasomotor_symp_fraction * self.density

    @property
    def svr_factor(self) -> float:
        """Direct arteriolar resistance multiplier from sympathectomy (Ward 1965; Rooke 1997)."""
        vaso = self.vasomotor_symp_fraction
        if vaso <= 0.0:
            return 1.0
        # Young healthy adults largely preserve resting SVR via upper-body vasoconstriction
        # (Ward 1965: TPR -5% at T5), whereas elderly men lose ~26% SVR (Rooke 1997).
        age_drop = 0.08 + 0.20 * clamp01((self.age - 30.0) / 45.0)
        # High/total spinal above T1 (index 22) blocks upper-body compensatory vasoconstriction too.
        high_symp = clamp01((self.sympathetic_level - 22.0) / 5.0)
        drop = (age_drop * vaso + 0.26 * high_symp) * self.density
        return clamp(1.0 - drop, 0.42, 1.0)

    @property
    def symp_vasc_block(self) -> float:
        """Attenuation of endogenous reflex vasoconstriction (0-0.85); exogenous pressors preserved."""
        return clamp(0.80 * self.vasomotor_symp_fraction * self.density, 0.0, 0.85)

    @property
    def respiratory_muscle_strength(self) -> float:
        """Inspiratory muscle strength fraction (1.0 normal; >=0.86 at T1; 0.0 by C5-C4 phrenic block)."""
        m = self.motor_level
        if m <= 11.0:
            return 1.0
        # Freund 1967: thoracic motor block spares the diaphragm (mean IC -8% spinal, -19% at T1).
        thoracic_loss = 0.14 * clamp01((m - 11.0) / 11.0)
        base = 1.0 - thoracic_loss
        if m <= 24.5:
            return base
        # Calibrated cervical phrenic paralysis across C6-C4 motor roots (24.5 to 26.7 -> apnea by C4).
        phrenic_loss = clamp01((m - 24.5) / 2.2)
        return clamp(base * (1.0 - phrenic_loss), 0.0, 1.0)

    @property
    def unconsciousness(self) -> float:
        """Calibrated high/total spinal loss of consciousness fraction (0-1 across C4-C2/brainstem)."""
        s = self.sensory_level
        if s <= 26.5:
            return 0.0
        return clamp01((s - 26.5) / 2.0)

    @property
    def thermal_redistribution_depth(self) -> float:
        """Effective lower-body vasodilation depth for core-to-peripheral heat redistribution."""
        return 0.35 * self.vasomotor_symp_fraction * self.density

    def surgical_analgesia(self, surgical_site: str = "other") -> float:
        """Fraction of surgical nociception blocked by the neuraxial sensory level (0-0.95)."""
        req = _SITE_REQUIRED_LEVEL.get(surgical_site, 13.0)
        s = self.sensory_level
        if s < req - 2.0:
            return 0.0
        coverage = clamp01((s - (req - 2.0)) / 2.5)
        return clamp(coverage * (0.35 + 0.60 * self.density), 0.0, 0.95)

    def give_spinal(
        self,
        drug: str = "bupivacaine",
        dose_mg: float = 12.5,
        baricity: str = "hyperbaric",
        *,
        high_spinal: bool = False,
    ) -> None:
        """Administer a subarachnoid (spinal) local anesthetic bolus."""
        drug_norm = str(drug).strip().lower()
        if drug_norm not in NEURAXIAL_DRUGS:
            raise ValueError(f"Unsupported neuraxial drug {drug!r}; choose bupivacaine or lidocaine")
        bar_norm = str(baricity).strip().lower()
        if bar_norm not in SPINAL_BARICITIES:
            raise ValueError(f"Unsupported baricity {baricity!r}; choose hyperbaric or isobaric")
        dose = clamp(float(dose_mg), 1.0, 150.0)
        # Convert lidocaine dose to bupivacaine-equivalent mass (~4.5:1 potency ratio).
        bupi_eq = dose if drug_norm == "bupivacaine" else dose / 4.5
        age_shift = 0.06 * (self.age - 40.0)  # Veering 1988: ~2.4 segments higher at 73 vs 33 yr
        preg_shift = 3.0 if self.pregnant else 0.0
        bar_shift = 0.0 if bar_norm == "hyperbaric" else -1.5
        if high_spinal:
            target = 32.5
            tau_on = 65.0
        else:
            # 12.5 mg hyperbaric bupivacaine -> ~T6 (17.0) in 40 yo, ~T4 in 73 yo or pregnant;
            # 15 mg -> ~T5; >30 mg intrathecal overdose -> high/total spinal (C4-Brainstem).
            # Multiply depot slightly (~1.12) so the bi-exponential onset/regression peak lands on target.
            raw_target = clamp(17.0 + 0.40 * (bupi_eq - 12.5) + age_shift + preg_shift + bar_shift, 8.0, 30.0)
            target = min(31.5, raw_target * 1.12)
            tau_on = (200.0 if drug_norm == "bupivacaine" else 140.0) * (1.0 if bar_norm == "hyperbaric" else 1.2)
        age_dur = 1.0 + 0.20 * clamp01((self.age - 40.0) / 35.0)
        tau_off = (5400.0 if drug_norm == "bupivacaine" else 2700.0) * age_dur
        self._spinal_depot = max(self._spinal_depot, target)
        self._spinal_tau_on = tau_on
        self._spinal_tau_off = tau_off
        self.last_drug = drug_norm
        self.last_mode = "high_spinal" if (high_spinal or target >= 26.0) else "spinal"

    def place_epidural(self, placed: bool = True) -> None:
        self.epidural_catheter = bool(placed)
        if not self.epidural_catheter:
            self.epidural_infusion_ml_hr = 0.0

    def give_epidural_bolus(
        self,
        drug: str = "bupivacaine",
        volume_ml: float = 10.0,
        concentration_pct: float = 0.5,
        *,
        with_epinephrine: bool = False,
        accidental_intrathecal: bool = False,
    ) -> None:
        """Administer an epidural bolus (or accidental intrathecal injection causing total spinal)."""
        drug_norm = str(drug).strip().lower()
        if drug_norm not in NEURAXIAL_DRUGS:
            raise ValueError(f"Unsupported neuraxial drug {drug!r}; choose bupivacaine or lidocaine")
        vol = clamp(float(volume_ml), 1.0, 30.0)
        conc = clamp(float(concentration_pct), 0.05, 2.0)
        self.epidural_catheter = True
        dose_mg = vol * conc * 10.0  # 1% = 10 mg/mL
        if accidental_intrathecal:
            self.give_spinal(drug=drug_norm, dose_mg=max(dose_mg, 40.0), high_spinal=True)
            self.last_mode = "high_spinal"
            return
        age_shift = 0.05 * (self.age - 40.0)
        preg_shift = 2.5 if self.pregnant else 0.0
        # Calibrated volume-spread curve (10 mL -> ~T8, 15 mL -> ~T5, 20 mL -> ~T3).
        raw_target = clamp(10.0 + 0.55 * vol + age_shift + preg_shift, 8.0, 23.0)
        target = min(26.5, raw_target * 1.20)
        bupi_conc_eq = conc if drug_norm == "bupivacaine" else conc / 4.0
        density = clamp(0.25 + 0.70 * bupi_conc_eq, 0.25, 0.65)
        self._epidural_depot = max(self._epidural_depot, target)
        self._epidural_density = max(self._epidural_density, density)
        self._epidural_tau_on = 360.0 if drug_norm == "bupivacaine" else 240.0
        self._epidural_tau_off = 4800.0 if drug_norm == "bupivacaine" else 2400.0
        self.last_drug = drug_norm
        if self.last_mode != "high_spinal":
            self.last_mode = "epidural"
        if with_epinephrine:
            # 1:200,000 = 5 mcg/mL -> systemic beta-2/beta-1 absorption (Ward 1965).
            self._epi_depot_ng_ml = clamp(self._epi_depot_ng_ml + 0.055 * vol, 0.0, 1.35)

    def set_epidural_infusion(
        self,
        ml_hr: float,
        *,
        drug: str = "bupivacaine",
        concentration_pct: float = 0.125,
    ) -> None:
        """Start or adjust continuous epidural infusion (mL/hr)."""
        drug_norm = str(drug).strip().lower()
        if drug_norm not in NEURAXIAL_DRUGS:
            raise ValueError(f"Unsupported neuraxial drug {drug!r}; choose bupivacaine or lidocaine")
        rate = clamp(float(ml_hr), 0.0, 25.0)
        if rate > 0.0:
            self.epidural_catheter = True
        self.epidural_infusion_ml_hr = rate
        self.epidural_infusion_drug = drug_norm
        self.epidural_infusion_conc_pct = clamp(float(concentration_pct), 0.05, 0.5)
        if rate > 0.0 and self.last_mode == "none":
            self.last_mode = "epidural"
            self.last_drug = drug_norm

    def check_level(self, sim_time: float) -> NeuraxialLevelCheck:
        """Perform a bedside cold/pinprick dermatome & Bromage motor assessment."""
        s_lvl = self.sensory_level
        m_lvl = self.motor_level
        s_lbl = dermatome_label(s_lvl)
        m_lbl = dermatome_label(m_lvl)
        bromage = self.bromage_score
        if s_lvl < 0.5:
            note = "No neuraxial block detected (normal sensation and motor strength)."
        elif s_lvl >= 27.5:
            note = f"Total spinal block ({s_lbl}); apnea and loss of consciousness; secure airway and support circulation."
        elif s_lvl >= 23.0:
            note = f"High spinal block to {s_lbl} (motor {m_lbl}, Bromage {bromage}); impending phrenic/respiratory compromise."
        elif s_lvl >= 19.0:
            note = f"High thoracic block to {s_lbl} (motor {m_lbl}, Bromage {bromage}); cardioaccelerator fibers (T1-T4) involved."
        else:
            note = f"Sensory block to {s_lbl} (motor {m_lbl}, Bromage {bromage})."
        check = NeuraxialLevelCheck(
            time=float(sim_time),
            sensory_level=s_lvl,
            sensory_label=s_lbl,
            motor_level=m_lvl,
            motor_label=m_lbl,
            bromage=bromage,
            mode=self.last_mode,
            note=note,
        )
        self.last_check = check
        return check

    def clear(self) -> None:
        self.epidural_infusion_ml_hr = 0.0
        self._spinal_depot = 0.0
        self._spinal_level = 0.0
        self._epidural_depot = 0.0
        self._epidural_level = 0.0
        self._epi_depot_ng_ml = 0.0
        self.absorbed_epi_ng_ml = 0.0
        self.last_mode = "none"
        self.last_drug = ""

    def step(self, dt: float, *, circulation: float = 1.0) -> None:
        """Advance spinal/epidural onset, regression, infusion maintenance, and epi absorption."""
        if dt <= 0.0:
            return
        # Spinal compartment: first-order onset toward decaying subarachnoid depot.
        if self._spinal_depot > 0.0 or self._spinal_level > 0.0:
            if self._spinal_depot > self._spinal_level:
                self._spinal_level += (self._spinal_depot - self._spinal_level) * (
                    -math.expm1(-dt / max(10.0, self._spinal_tau_on))
                )
            # Slow CSF washout / regression once established
            decay_off = math.exp(-dt / max(300.0, self._spinal_tau_off))
            self._spinal_depot *= decay_off
            if self._spinal_level >= self._spinal_depot:
                self._spinal_level = self._spinal_depot
            if self._spinal_level < 0.05 and self._spinal_depot < 0.05:
                self._spinal_level = self._spinal_depot = 0.0

        # Epidural compartment: infusion sustains its equilibrium target while prior higher
        # boluses gradually regress toward that infusion floor and density.
        inf_target = 0.0
        inf_density = 0.0
        if self.epidural_infusion_ml_hr > 0.0:
            bupi_inf_conc = (
                self.epidural_infusion_conc_pct
                if self.epidural_infusion_drug == "bupivacaine"
                else self.epidural_infusion_conc_pct / 4.0
            )
            potency_scale = clamp(0.65 + 2.8 * bupi_inf_conc, 0.70, 1.05)
            inf_target = clamp(
                (9.5 + 0.55 * self.epidural_infusion_ml_hr + 0.04 * (self.age - 40.0) + (2.0 if self.pregnant else 0.0))
                * potency_scale,
                8.0,
                18.5,
            )
            inf_density = clamp(0.20 + 0.70 * bupi_inf_conc, 0.20, 0.55)
            if self._epidural_depot < inf_target:
                self._epidural_depot = inf_target
                self._epidural_density = max(self._epidural_density, inf_density)
        if self._epidural_depot > 0.0 or self._epidural_level > 0.0:
            if self._epidural_depot > self._epidural_level:
                self._epidural_level += (self._epidural_depot - self._epidural_level) * (
                    -math.expm1(-dt / max(20.0, self._epidural_tau_on))
                )
            decay_e = math.exp(-dt / max(300.0, self._epidural_tau_off))
            if self._epidural_depot > inf_target:
                self._epidural_depot = inf_target + (self._epidural_depot - inf_target) * decay_e
            if inf_density > 0.0 and self._epidural_density > inf_density:
                self._epidural_density = inf_density + (self._epidural_density - inf_density) * decay_e
            if self._epidural_level >= self._epidural_depot:
                self._epidural_level = self._epidural_depot
            if self._epidural_level < 0.05 and self._epidural_depot < 0.05:
                self._epidural_level = self._epidural_depot = 0.0

        # Systemic absorption of epidural epinephrine (1:200,000; Ward 1965); requires perfusion.
        circ = clamp01(circulation)
        if circ > 0.0 and (self._epi_depot_ng_ml > 0.0 or self.absorbed_epi_ng_ml > 0.0):
            alpha_in = -math.expm1(-dt * circ / 220.0)
            self.absorbed_epi_ng_ml += (self._epi_depot_ng_ml - self.absorbed_epi_ng_ml) * alpha_in
            self._epi_depot_ng_ml *= math.exp(-dt * circ / 1400.0)
            if self._epi_depot_ng_ml < 1e-4 and self.absorbed_epi_ng_ml < 1e-4:
                self._epi_depot_ng_ml = self.absorbed_epi_ng_ml = 0.0

        if (
            not self.active
            and self._spinal_depot <= 0.0
            and self._epidural_depot <= 0.0
            and self.epidural_infusion_ml_hr <= 0.0
        ):
            self.last_mode = "none"
