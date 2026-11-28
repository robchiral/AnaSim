"""Temporary cardiac pacing (transcutaneous and epicardial) physiology.

Capture thresholds and conscious transcutaneous tolerance follow Zoll et al.
(Circulation 1985; 71:937-944). Atrial-kick stroke-volume loss during VVI/VOO
versus AAI/DDD pacing across filling pressures follows Greenberg et al.
(Circulation 1979; 60:1194-1201). Paced cardiac-output flattening and high-rate
decline follow Ross et al. (Circ Res 1965; 16:332-342), Tweed et al.
(Anesthesiology 1972; 37:613-619), and Hogue et al. (Anesth Analg 1996;
83:271-278).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.enums import RhythmType
from anasim.core.utils import clamp01
from anasim.patient.domain import bounded_number
from anasim.physiology.toxicity import SILENT_BLOCK

PACING_MODES = ("off", "vvi", "voo", "aai", "ddd", "doo")
PACING_ROUTES = ("transcutaneous", "epicardial")

DEMAND_MODES = frozenset({"vvi", "aai", "ddd"})
ASYNC_MODES = frozenset({"voo", "doo"})
VENTRICULAR_MODES = frozenset({"vvi", "voo"})
DUAL_CHAMBER_MODES = frozenset({"ddd", "doo"})

PACING_RATE_RANGE = (30.0, 180.0)  # bpm (Zoll 1985; Ross 1965)
TRANSCUTANEOUS_OUTPUT_RANGE_MA = (0.0, 200.0)
EPICARDIAL_OUTPUT_RANGE_MA = (0.0, 25.0)

# Base capture thresholds (Zoll 1985: transcutaneous 40-70 mA typical; epicardial 1-5 mA).
TRANSCUTANEOUS_BASE_THRESHOLD_MA = 55.0
EPICARDIAL_BASE_THRESHOLD_MA = 3.0


@dataclass(slots=True)
class PacingStatus:
    """Instantaneous pacing capture, sensing, and hemodynamic modifier summary."""

    mode: str = "off"
    route: str = "transcutaneous"
    rate_bpm: float = 70.0
    output_ma: float = 0.0
    threshold_ma: float = TRANSCUTANEOUS_BASE_THRESHOLD_MA
    delivering_spikes: bool = False
    inhibited: bool = False
    captured: bool = False
    paced_hr: float = 0.0
    av_synchrony: bool = True
    wide_qrs: bool = False
    sv_multiplier: float = 1.0
    async_competition: bool = False
    discomfort_stress: float = 0.0  # 0-1 awake transcutaneous stimulation stress


class CardiacPacemaker:
    """Temporary external pacemaker supporting transcutaneous and epicardial modes."""

    def __init__(self, bmi: float = 24.0) -> None:
        self.bmi = max(12.0, float(bmi))
        self.mode: str = "off"
        self.route: str = "transcutaneous"
        self.rate_bpm: float = 70.0
        self.output_ma: float = 0.0
        self.status = PacingStatus()
        self._spike_elapsed_s: float = 0.0

    @property
    def is_on(self) -> bool:
        return self.mode != "off"

    def max_output_ma(self, route: str | None = None) -> float:
        r = self.route if route is None else route
        return EPICARDIAL_OUTPUT_RANGE_MA[1] if r == "epicardial" else TRANSCUTANEOUS_OUTPUT_RANGE_MA[1]

    def configure(
        self,
        *,
        mode: str | None = None,
        route: str | None = None,
        rate_bpm: float | None = None,
        output_ma: float | None = None,
    ) -> None:
        """Validate and update pacemaker mode, route, rate, and output current."""
        new_route = self.route
        if route is not None:
            norm_route = str(route).strip().lower()
            if norm_route not in PACING_ROUTES:
                choices = ", ".join(PACING_ROUTES)
                raise ValueError(f"Unknown pacing route {route!r}; choose one of: {choices}")
            new_route = norm_route
        new_mode = self.mode
        if mode is not None:
            norm_mode = str(mode).strip().lower()
            if norm_mode not in PACING_MODES:
                choices = ", ".join(PACING_MODES)
                raise ValueError(f"Unknown pacing mode {mode!r}; choose one of: {choices}")
            new_mode = norm_mode
        new_rate = self.rate_bpm
        if rate_bpm is not None:
            new_rate = bounded_number("pacing rate", rate_bpm, *PACING_RATE_RANGE, unit="bpm")
        max_ma = self.max_output_ma(new_route)
        new_output = self.output_ma
        if output_ma is not None:
            new_output = bounded_number("pacing output", output_ma, 0.0, max_ma, unit="mA")
        elif new_route != self.route:
            # Clamp or set sensible route default when switching routes.
            new_output = min(new_output, max_ma)
        self.route = new_route
        self.mode = new_mode
        self.rate_bpm = new_rate
        self.output_ma = new_output
        if self.mode == "off":
            self.status = PacingStatus(
                mode="off",
                route=self.route,
                rate_bpm=self.rate_bpm,
                output_ma=self.output_ma,
                threshold_ma=self.capture_threshold_ma(),
            )

    def capture_threshold_ma(
        self,
        *,
        ischemia: float = 0.0,
        hypoxia: float = 0.0,
        energy_failure: float = 0.0,
        hyperkalemia_block: float = 0.0,
        sodium_block: float = 0.0,
        magnesium_block: float = 0.0,
        temp_c: float = 37.0,
    ) -> float:
        """Myocardial capture threshold (mA); rises with ischemia, hypoxia, hyperK, and hypothermia."""
        if self.route == "epicardial":
            base = EPICARDIAL_BASE_THRESHOLD_MA
        else:
            base = TRANSCUTANEOUS_BASE_THRESHOLD_MA + 1.2 * max(0.0, self.bmi - 25.0)
        mult = (
            1.0
            + 0.70 * clamp01(ischemia)
            + 0.90 * clamp01(hypoxia)
            + 1.50 * clamp01(energy_failure)
            + 1.80 * clamp01(hyperkalemia_block)
            + 1.50 * clamp01(sodium_block)
            + 0.80 * clamp01(magnesium_block)
            + 0.60 * clamp01((34.0 - temp_c) / 6.0)
        )
        return base * mult

    def evaluate(
        self,
        *,
        intrinsic_rhythm: RhythmType,
        intrinsic_hr: float,
        lap: float = 9.0,
        ischemia: float = 0.0,
        hypoxia: float = 0.0,
        energy_failure: float = 0.0,
        hyperkalemia_block: float = 0.0,
        sodium_block: float = 0.0,
        magnesium_block: float = 0.0,
        temp_c: float = 37.0,
        cardioplegia_arrest: bool = False,
        stimulus_tolerance: float = 1.0,
    ) -> PacingStatus:
        """Evaluate sensing, capture, AV synchrony SV modifier, and transcutaneous discomfort."""
        threshold = self.capture_threshold_ma(
            ischemia=ischemia,
            hypoxia=hypoxia,
            energy_failure=energy_failure,
            hyperkalemia_block=hyperkalemia_block,
            sodium_block=sodium_block,
            magnesium_block=magnesium_block,
            temp_c=temp_c,
        )
        if self.mode == "off":
            self.status = PacingStatus(
                mode="off",
                route=self.route,
                rate_bpm=self.rate_bpm,
                output_ma=self.output_ma,
                threshold_ma=threshold,
            )
            return self.status

        mode = self.mode
        electrically_silent = (
            cardioplegia_arrest
            or energy_failure >= 0.99
            or hyperkalemia_block >= SILENT_BLOCK
            or sodium_block >= SILENT_BLOCK
            or magnesium_block >= SILENT_BLOCK
        )
        eff_intrinsic_hr = 0.0 if (
            electrically_silent or intrinsic_rhythm in (RhythmType.ASYSTOLE, RhythmType.VFIB)
        ) else max(0.0, float(intrinsic_hr))

        inhibited = mode in DEMAND_MODES and eff_intrinsic_hr >= self.rate_bpm - 1e-6
        delivering_spikes = (not inhibited) and self.output_ma > 0.0

        can_capture_chamber = (
            not electrically_silent
            and intrinsic_rhythm != RhythmType.VFIB
            and not (mode == "aai" and intrinsic_rhythm == RhythmType.AFIB)
        )
        suprathreshold = delivering_spikes and self.output_ma >= threshold
        # In asynchronous modes, overdrive capture occurs when paced rate exceeds intrinsic rate
        # (or during asystole); slower asynchronous spikes compete rather than raising HR.
        overrides_rate = self.rate_bpm > eff_intrinsic_hr - 1e-6
        captured = bool(suprathreshold and can_capture_chamber and overrides_rate)
        async_competition = bool(
            mode in ASYNC_MODES
            and suprathreshold
            and can_capture_chamber
            and eff_intrinsic_hr > 0.0
            and intrinsic_rhythm not in (RhythmType.ASYSTOLE, RhythmType.VFIB)
        )

        # Greenberg 1979: losing atrial kick in VVI/VOO drops SV ~24% at normal PCWP (<20 mmHg)
        # and ~10% at high PCWP (>=20 mmHg); AAI and DDD/DOO preserve AV synchrony.
        av_synchrony = (
            mode not in VENTRICULAR_MODES
            and intrinsic_rhythm != RhythmType.AFIB
        )
        wide_qrs = captured and mode != "aai"
        sv_mult = 1.0
        if captured:
            if mode in VENTRICULAR_MODES and intrinsic_rhythm != RhythmType.AFIB:
                kick_loss = 0.24 - 0.14 * clamp01((lap - 14.0) / 10.0)
                sv_mult *= 1.0 - kick_loss
            # Ross 1965: rapid atrial/dual-chamber pacing flattens CI across 80-120 bpm
            # and reduces CI ~12-15% (SVI -53%) by ~148 bpm.
            if self.rate_bpm > 105.0:
                excess = (self.rate_bpm - 105.0) / 45.0
                sv_mult /= 1.0 + 0.55 * (excess ** 1.7)

        # Zoll 1985: conscious transcutaneous pacing produces skeletal-muscle discomfort
        # proportional to current output when not sedated/analgesed.
        discomfort = 0.0
        if self.route == "transcutaneous" and delivering_spikes and self.output_ma >= 20.0:
            intensity = clamp01((self.output_ma - 20.0) / 70.0)
            discomfort = intensity * clamp01(1.0 - stimulus_tolerance)

        self.status = PacingStatus(
            mode=mode,
            route=self.route,
            rate_bpm=self.rate_bpm,
            output_ma=self.output_ma,
            threshold_ma=threshold,
            delivering_spikes=delivering_spikes,
            inhibited=inhibited,
            captured=captured,
            paced_hr=self.rate_bpm if captured else 0.0,
            av_synchrony=av_synchrony,
            wide_qrs=wide_qrs,
            sv_multiplier=sv_mult,
            async_competition=async_competition,
            discomfort_stress=discomfort,
        )
        return self.status

    def step_unpaced_spike(self, dt: float) -> bool:
        """Advance free-running pacer clock for non-captured spikes on the ECG; returns True on spike."""
        if not self.status.delivering_spikes or self.status.captured or self.rate_bpm <= 0.0:
            self._spike_elapsed_s = 0.0
            return False
        period = 60.0 / self.rate_bpm
        self._spike_elapsed_s += dt
        if self._spike_elapsed_s >= period:
            self._spike_elapsed_s %= period
            return True
        return False

    def spike_offsets_s(self) -> tuple[float, ...]:
        """Spike timings relative to R-peak (s) when actively capturing beats."""
        if not self.status.captured:
            return ()
        if self.mode == "aai":
            return (-0.185,)
        if self.mode in DUAL_CHAMBER_MODES:
            return (-0.185, -0.045)
        return (-0.045,)

    @staticmethod
    def competition_vf_probability(dt: float, async_competition: bool) -> float:
        """Probability of R-on-T VF induction during dt of competing asynchronous pacing."""
        if not async_competition or dt <= 0.0:
            return 0.0
        return -math.expm1(-0.012 * dt)
