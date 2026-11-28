"""Local-anesthetic methemoglobinemia and methylene blue reduction kinetics.

Grounded in:
- Guay 2009 (Anesth Analg 108:837-845; 242 clinical episodes): benzocaine
  accounts for 65.7% of LA-related episodes (single 20% topical spray can reach
  29-40% MetHb; rebound can occur 2.5-18 h post-methylene blue), prilocaine
  28.1% (peak up to 42%), and lidocaine 5.0%. Recommended methylene blue dosing
  is 1-2 mg/kg IV (maximum cumulative dose 7 mg/kg; excess paradoxically
  oxidizes hemoglobin; transient false SpO2 dip noted in 5 patients), clearing
  MetHb to <= 2% in 0.33-36.2 h vs 15-36 h untreated.
- Barker et al. 1989 (Anesthesiology 70:112-117; n=5 dogs): two-wavelength
  pulse oximetry optical bias plateauing near 85% (82-86%) at MetHb >= 30-35%
  despite normal PaO2, with linearly falling fractional SaO2 = FSaO2 * (1 - MetHb).
"""

from __future__ import annotations

import math

from anasim.core.utils import clamp, clamp01, hill_function

METHEMOGLOBIN_AGENTS = ("benzocaine", "prilocaine", "lidocaine")

# Agent oxidative parameters: (default_load_units, depot_tau_s, conversion_rate_per_min)
# Calibrated so severity=1.0 produces ~34% peak MetHb in ~15-20 min for benzocaine
# and ~26% peak MetHb over ~30-40 min for prilocaine (Guay 2009).
_AGENT_PARAMS: dict[str, tuple[float, float, float]] = {
    "benzocaine": (0.46, 1800.0, 0.090),
    "prilocaine": (0.34, 3600.0, 0.045),
    "lidocaine": (0.14, 2400.0, 0.040),
}

# Endogenous NADH-cytochrome b5 reductase reduction rate (t1/2 ~9.5 h -> resolves
# over 15-36 h untreated per Guay 2009).
_ENDOGENOUS_REDUCTASE_PER_MIN = 0.0012
# Maximum methylene-blue-driven NADPH-methemoglobin reductase rate (calibrated
# t1/2 ~16.5 min at therapeutic 1-2 mg/kg so severe MetHb clears substantially
# over 1-2 h, consistent with Guay 2009 clinical reversal timelines).
_MB_REDUCTASE_MAX_PER_MIN = 0.042
_MB_EC50_UG_ML = 0.25
MAX_METHYLENE_BLUE_MG_KG = 7.0


class MethemoglobinModel:
    """Tracks arterial and systemic methemoglobin fraction and methylene blue rescue."""

    def __init__(self, weight_kg: float = 70.0) -> None:
        self.weight_kg = max(1.0, float(weight_kg))
        self.methb_fraction = 0.0  # 0.0 at rest (zero perturbation unless triggered); up to 0.85
        self._arterial_methb = 0.0
        self._circulation = 1.0
        self.oxidizer_load = 0.0
        self._depot_tau_s = 1800.0
        self._conv_rate_per_min = 0.090
        self.trigger_agent = "none"
        self.cumulative_mb_mg = 0.0
        self.mb_optical_dip = 0.0  # Transient post-IV methylene blue pulse-ox dip (%)

    @property
    def active(self) -> bool:
        return self.methb_fraction >= 0.05 or self.oxidizer_load > 0.02

    @property
    def arterial_methb_fraction(self) -> float:
        """Arterial-sample MetHb fraction (0-0.85); frozen during zero-flow arrest."""
        if self._circulation >= 0.99:
            return self.methb_fraction
        return self._arterial_methb

    @property
    def cumulative_mb_mg_kg(self) -> float:
        return self.cumulative_mb_mg / self.weight_kg

    @property
    def p50_scale(self) -> float:
        """Leftward ODC shift factor from Fe3+ heme allostery (1.0 at 0% MetHb)."""
        if self.methb_fraction <= 0.0:
            return 1.0
        return 1.0 - 0.55 * clamp(self.methb_fraction, 0.0, 0.70)

    @property
    def arterial_p50_scale(self) -> float:
        m = self.arterial_methb_fraction
        if m <= 0.0:
            return 1.0
        return 1.0 - 0.55 * clamp(m, 0.0, 0.70)

    def reset(self) -> None:
        self.methb_fraction = 0.0
        self._arterial_methb = 0.0
        self._circulation = 1.0
        self.oxidizer_load = 0.0
        self.trigger_agent = "none"
        self.cumulative_mb_mg = 0.0
        self.mb_optical_dip = 0.0

    def trigger(
        self,
        agent: str = "benzocaine",
        *,
        severity: float = 1.0,
        initial_fraction: float | None = None,
    ) -> None:
        """Expose patient to a methemoglobinemia-inducing local anesthetic."""
        agent_norm = str(agent).strip().lower()
        if agent_norm not in _AGENT_PARAMS:
            raise ValueError(
                f"Unsupported methemoglobinemia agent {agent!r}; choose benzocaine, prilocaine, or lidocaine"
            )
        sev = clamp(float(severity), 0.1, 2.5)
        base_load, tau_s, conv_rate = _AGENT_PARAMS[agent_norm]
        self.trigger_agent = agent_norm
        self._depot_tau_s = tau_s
        self._conv_rate_per_min = conv_rate
        if initial_fraction is not None:
            frac = clamp(float(initial_fraction), 0.0, 0.80)
            self.methb_fraction = max(self.methb_fraction, frac)
            if self._circulation >= 0.99:
                self._arterial_methb = self.methb_fraction
        else:
            # Add mucosal/systemic oxidizer depot; onset builds rapidly over 10-25 min.
            self.oxidizer_load = clamp(self.oxidizer_load + base_load * sev, 0.0, 1.8)
            # Immediate small mucosal absorption seed for topical benzocaine spray.
            if agent_norm == "benzocaine":
                self.methb_fraction = clamp(self.methb_fraction + 0.04 * sev, 0.0, 0.85)
                if self._circulation >= 0.99:
                    self._arterial_methb = self.methb_fraction

    def record_methylene_blue_dose(self, dose_mg: float) -> None:
        """Record IV methylene blue dose for cumulative toxicity tracking and optical dip."""
        if dose_mg <= 0.0:
            return
        self.cumulative_mb_mg += dose_mg
        # Guay 2009: IV methylene blue dye causes a brief (~2-4 min) false SpO2 dip.
        dose_mg_kg = dose_mg / self.weight_kg
        self.mb_optical_dip = clamp(self.mb_optical_dip + 5.0 * clamp(dose_mg_kg / 1.5, 0.2, 2.0), 0.0, 12.0)

    def sync_circulation(self, circulation: float) -> None:
        new_circ = clamp01(circulation)
        if self._circulation >= 0.99 and new_circ < 0.99:
            self._arterial_methb = self.methb_fraction
        self._circulation = new_circ

    def step(
        self,
        dt: float,
        *,
        methylene_blue_ce: float = 0.0,
        circulation: float = 1.0,
    ) -> None:
        """Advance methemoglobin oxidation, endogenous reduction, and methylene blue rescue."""
        self.sync_circulation(circulation)
        if (
            dt <= 0.0
            or (
                self.methb_fraction <= 0.0
                and self.oxidizer_load <= 0.0
                and self.mb_optical_dip <= 0.0
                and self.cumulative_mb_mg_kg <= MAX_METHYLENE_BLUE_MG_KG
            )
        ):
            return
        circ = self._circulation
        dt_min = dt / 60.0
        # Optical dye bolus clears rapidly from central arterial pass (~90 s tau).
        if self.mb_optical_dip > 0.0 and circ > 0.0:
            self.mb_optical_dip *= math.exp(-dt * circ / 90.0)
            if self.mb_optical_dip < 0.02:
                self.mb_optical_dip = 0.0

        # Oxidizer depot absorption/conversion requires systemic perfusion.
        oxidation_rate = 0.0
        if self.oxidizer_load > 0.0 and circ > 0.0:
            oxidation_rate = self._conv_rate_per_min * self.oxidizer_load * circ
            self.oxidizer_load *= math.exp(-dt * circ / self._depot_tau_s)
            if self.oxidizer_load < 1e-4:
                self.oxidizer_load = 0.0
        # Paradoxical oxidation when cumulative methylene blue exceeds 7 mg/kg (Guay 2009).
        excess_mb_mg_kg = max(0.0, self.cumulative_mb_mg_kg - MAX_METHYLENE_BLUE_MG_KG)
        if excess_mb_mg_kg > 0.0 and methylene_blue_ce > 0.2:
            oxidation_rate += 0.004 * clamp(excess_mb_mg_kg / 3.0, 0.0, 2.0) * hill_function(
                methylene_blue_ce, 1.0, 1.5,
            ) * circ

        mb_red = (
            _MB_REDUCTASE_MAX_PER_MIN * hill_function(max(0.0, methylene_blue_ce), _MB_EC50_UG_ML, 1.4)
            if methylene_blue_ce > 0.0 and excess_mb_mg_kg <= 2.0
            else 0.0
        )
        reduction_rate = _ENDOGENOUS_REDUCTASE_PER_MIN + mb_red * max(0.15, circ)

        # Exact linear step for dm/dt = oxidation_rate * (0.85 - m) - reduction_rate * m.
        max_m = 0.85
        total_rate = oxidation_rate + reduction_rate
        if total_rate > 0.0:
            steady = (oxidation_rate * max_m) / total_rate
            self.methb_fraction = clamp(
                steady + (self.methb_fraction - steady) * math.exp(-total_rate * dt_min),
                0.0,
                max_m,
            )
        if self.methb_fraction < 1e-5 and self.oxidizer_load <= 0.0:
            self.methb_fraction = 0.0
            self.trigger_agent = "none"

        if circ >= 0.99:
            self._arterial_methb = self.methb_fraction
        elif circ > 0.0:
            self._arterial_methb += (self.methb_fraction - self._arterial_methb) * -math.expm1(
                -dt * circ / 20.0,
            )
