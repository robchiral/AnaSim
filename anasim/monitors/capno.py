import numpy as np
from dataclasses import dataclass

@dataclass
class CapnoState:
    co2: float = 0.0 # Instantaneous CO2 (mmHg)
    phase: int = 4   # 1=Deadspace, 2=Rise, 3=Plateau, 4=Insp

@dataclass
class CapnoContext:
    """Pre-computed parameters describing the next breath for waveform synthesis."""
    exp_duration: float
    is_spontaneous: bool
    curare_active: bool
    effort_scale: float
    spontaneous_weight: float
    effective_rr: float

class Capnograph:
    """
    Generates realistic Capnography waveforms based on respiratory state.
    """
    def __init__(self, rng: np.random.Generator = None):
        self.state = CapnoState()
        self.last_phase = "EXP"
        self.time_in_phase = 0.0
        # Use provided RNG for reproducibility, or create default
        self.rng = rng if rng is not None else np.random.default_rng()
        
        self.val_at_change = 0.0
        # Waveform Parameters
        self.deadspace_fraction = 0.15 # Phase I fraction of exp time
        self.rise_time = 0.1 # Phase II duration (s)
        self.insp_drop_time = 0.1 # Phase IV duration (s)

    @staticmethod
    def build_context(resp_state, vent_rr: float, insp_fraction: float, vent_active: bool) -> CapnoContext:
        """
        Normalize all capnogram meta-data in one place so cleft logic is not duplicated.
        """
        insp_fraction = float(np.clip(insp_fraction, 0.05, 0.8))
        if vent_active and vent_rr > 0.1:
            drive = max(0.0, getattr(resp_state, 'drive_central', 0.0))
            muscle = max(0.0, getattr(resp_state, 'muscle_factor', 0.0))
            spont_rr = max(0.0, getattr(resp_state, 'rr', 0.0))
            delta_rr = max(0.0, spont_rr - vent_rr)
            effort_signal = drive * muscle
            
            # Curare cleft: Patient "fights the vent" with spontaneous efforts.
            # Conditions:
            # 1. Drive > 0.3 (emerging)
            # 2. Muscle 0.1-0.9 (partial block)
            # 3. Effort > 0.15 or significant Rate Mismatch
            curare_active = (
                drive > 0.3 and
                0.1 < muscle < 0.9 and
                (effort_signal > 0.15 or delta_rr > 2.0)
            )
            
            effort_scale = 0.0
            if curare_active:
                # Scale cleft depth with both central drive and rate mismatch.
                # Gradual onset: effort_scale increases with effort_signal
                effort_scale = min(3.0,
                                   0.2 * delta_rr +
                                   1.5 * effort_signal +
                                   0.3 * (drive - 0.3))  # Bonus for higher drive

            # Blend between controlled and spontaneous timing to avoid abrupt switches.
            rr_ratio = 0.0
            if vent_rr > 0.1:
                rr_ratio = max(0.0, (spont_rr - vent_rr) / max(vent_rr, 1.0))
            rr_weight = np.clip(rr_ratio / 0.5, 0.0, 1.0)
            effort_weight = np.clip((effort_signal - 0.2) / 0.6, 0.0, 1.0)
            spontaneous_weight = float(np.clip(0.6 * rr_weight + 0.4 * effort_weight, 0.0, 1.0))

            effective_rr = (1.0 - spontaneous_weight) * vent_rr + spontaneous_weight * spont_rr
            exp_fraction = (1.0 - spontaneous_weight) * max(0.1, 1.0 - insp_fraction) + spontaneous_weight * 0.65
            cycle_time = 60.0 / max(effective_rr, 0.1)
            exp_duration = cycle_time * exp_fraction

            is_spontaneous = spontaneous_weight >= 0.6
            if spontaneous_weight >= 0.5:
                curare_active = False
                effort_scale = 0.0

            return CapnoContext(exp_duration, is_spontaneous, curare_active, effort_scale, spontaneous_weight, effective_rr)
        else:
            spont_rr = getattr(resp_state, 'rr', 0.0)
            if spont_rr <= 0.5:
                spont_rr = 12.0
            cycle_time = 60.0 / spont_rr
            exp_duration = cycle_time * 0.65
            return CapnoContext(exp_duration, True, False, 0.0, 1.0, spont_rr)
        
    def step(self, dt: float, phase: str, p_alv: float, is_spontaneous: bool = False, curare_cleft: bool = False,
             exp_duration: float = 3.0, effort_scale: float = 1.0, airway_obstruction: float = 0.0) -> float:
        """
        Compute instantaneous CO2.
        phase: "INSP" or "EXP" (from mechanics)
        p_alv: Alveolar PCO2 (mmHg)
        is_spontaneous: True if purely spontaneous breathing pattern
        curare_cleft: True if spontaneous efforts occurring during mechanical ventilation
        exp_duration: Duration of expiratory phase (s) for relative timing
        effort_scale: Multiplier for cleft depth (0.0 - 2.0+)
        airway_obstruction: 0.0=Normal, 1.0=Severe (controls shark fin shape)
        """
        
        if phase != self.last_phase:
            self.val_at_change = self.state.co2 # Capture last value
            self.time_in_phase = 0.0
            self.last_phase = phase
        else:
            self.time_in_phase += dt
            
        co2 = 0.0
        
        if phase == "INSP":
            self.state.phase = 4
            # Phase IV: Rapid Exponential Drop
            k = 10.0 
            co2 = self.val_at_change * np.exp(-k * self.time_in_phase)
                
        else: # EXPIRATION
            
            # Physiologic Parameters
            # Base values for normal lungs
            base_tau = 0.08
            base_slope = 0.1
            
            # Adjust for obstruction (Shark Fin effect)
            # obstruction ranges 0.0 to 1.0+
            tau = base_tau * (1 + 4 * airway_obstruction)
            slope_scale = 2.0 / max(exp_duration, 0.5)
            plateau_slope = base_slope * slope_scale * (1 + 5 * airway_obstruction)
            
            # Variability for spontaneous breaths
            if is_spontaneous:
                # Spontaneous breaths have natural flow variability from diaphragm effort
                pass

            # Deadspace as fraction of expiratory time
            deadspace_time = self.deadspace_fraction * exp_duration
            
            # Safety clamp for very short expirations
            if deadspace_time < 0.05: deadspace_time = 0.05
                
            if self.time_in_phase < deadspace_time:
                 self.state.phase = 1
                 co2 = 0.0
            else:
                 # Effective Expiration Time
                 t_exp = self.time_in_phase - deadspace_time
                 
                 # Alpha angle (rise)
                 rise_component = 1.0 - np.exp(-t_exp / tau)
                 
                 # Slope component (Phase III)
                 slope_component = plateau_slope * t_exp
                 
                 # Combine:
                 co2 = (p_alv * rise_component) + (slope_component * rise_component)
                 
                 # Add noise for spontaneous breaths to simulate flow irregularities
                 if is_spontaneous:
                     noise = self.rng.normal(0, 0.2) * rise_component # little noise on plateau
                     co2 += noise
                 
                 # Curare Cleft Logic (partial NMBA with diaphragmatic efforts).
                 # Render mid-late plateau to match typical clinical appearance
                 # (Bissinger et al. 1993).
                 if curare_cleft and not is_spontaneous:
                     
                     exp_effective = max(0.2, exp_duration - deadspace_time)
                     rel_cleft = exp_effective * (0.55 + 0.1 * min(1.0, effort_scale / 2.0))
                     t_cleft = min(exp_effective * 0.85, max(exp_effective * 0.30, rel_cleft))
                     
                     # Render only after reaching the cleft point.
                     # Dip parameters
                     depth = min(p_alv * 0.6, 4.0 + 8.0 * effort_scale)
                     width = max(0.08, 0.12 * (exp_effective / 2.0))
                     
                     dist = abs(t_exp - t_cleft)
                     
                     # Only apply if close enough to matter (optimization)
                     if dist < 0.5:
                         dynamic_depth = min(p_alv * 0.7, depth + 12.0 * effort_scale)
                         dip = dynamic_depth * np.exp(-(dist**2) / (2 * width**2))
                         co2 -= dip

                 # Avoid unphysiologically high plateaus.
                 plateau_cap = p_alv + 5.0 + 10.0 * airway_obstruction
                 co2 = min(co2, plateau_cap)
                
                 if co2 < 0: co2 = 0
                 
                 if rise_component < 0.95:
                     self.state.phase = 2
                 else:
                     self.state.phase = 3
                
        if co2 < 0: co2 = 0
            
        self.state.co2 = co2
        return co2
