"""
Respiratory Mechanics Model with VCV/PCV Mode Support.

Implements single-compartment lung model with realistic ventilator dynamics:
- Volume Control Ventilation (VCV): Constant flow, variable pressure
- Pressure Control Ventilation (PCV): Constant pressure, variable flow/volume
- Auto-PEEP calculation for incomplete exhalation
- PEEP effects on lung mechanics
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class VentMode(Enum):
    """Ventilator mode enumeration."""
    VCV = "VCV"  # Volume Control Ventilation
    PCV = "PCV"  # Pressure Control Ventilation
    # PSV = "PSV"  # Pressure Support (future)


@dataclass
class MechState:
    """
    Respiratory mechanics state snapshot.
    
    Contains instantaneous values and breath-cycle monitoring statistics.
    """
    # Instantaneous values
    paw: float = 0.0          # Airway Pressure (cmH2O)
    flow: float = 0.0         # Flow (L/min, positive = inspiration)
    volume: float = 0.0       # Volume above FRC (L)
    phase: str = "EXP"        # "INSP" or "EXP"
    
    # Breath-cycle monitors
    paw_peak: float = 0.0     # Peak Paw this breath (cmH2O)
    paw_plat: float = 0.0     # Plateau Paw (cmH2O) - end-inspiratory
    paw_mean: float = 0.0     # Mean Paw (cmH2O) - rolling average
    auto_peep: float = 0.0    # Intrinsic PEEP (cmH2O)
    delivered_vt: float = 0.0  # Actual delivered Vt (mL) - important for PCV
    
    # End-expiratory values
    eelv: float = 0.0         # End-Expiratory Lung Volume above FRC (L)


class RespiratoryMechanics:
    """
    Single Compartment Lung Model with VCV/PCV Mode Support.
    
    Equation of Motion: Paw = Volume/Compliance + Flow*Resistance + PEEP
    
    VCV Mode: Flow is controlled (square wave), Paw is calculated
    PCV Mode: Paw is controlled (set P_insp above PEEP), Flow/Vt are calculated
    
    Physiological interactions:
    - Higher PEEP increases mean Paw → impairs venous return
    - Auto-PEEP develops with short expiratory times → hemodynamic effects
    - PCV: Decreased compliance → lower delivered Vt
    - VCV: Decreased compliance → higher peak Paw
    """
    
    def __init__(self, compliance: float = 0.05, resistance: float = 10.0):
        """
        Initialize respiratory mechanics model.
        
        Args:
            compliance: Lung compliance in L/cmH2O (normal ~0.05-0.1)
            resistance: Airway resistance in cmH2O/(L/s) (normal ~5-15 for ETT)
        """
        # Lung Parameters
        # Compliance: L/cmH2O (Normal ~50-100 mL/cmH2O -> 0.05-0.1)
        self.compliance = compliance
        
        # Airway Resistance: cmH2O/(L/s)
        # Natural airway: 1-3 cmH2O/(L/s)
        # Endotracheal tube (7.0-8.0mm): 5-10 cmH2O/(L/s)
        # Small ETT (6.0mm): ~15 cmH2O/(L/s)
        # Bronchospasm: 15-30+ cmH2O/(L/s)
        self.resistance = resistance
        
        # Ventilator Settings
        self.mode = VentMode.VCV
        self.set_rr = 12.0              # Respiratory rate (bpm)
        self.set_vt = 0.5               # Set tidal volume (L) - VCV
        self.set_peep = 5.0             # Set PEEP (cmH2O)
        self.set_p_insp = 15.0          # Inspiratory pressure above PEEP (cmH2O) - PCV
        self.insp_time_fraction = 1.0 / 3.0  # I:E = 1:2 default
        
        # State
        self.state = MechState()
        self.cycle_time = 0.0           # Time within breath cycle
        
        # Monitoring accumulators
        self._paw_accumulator = 0.0     # For mean Paw calculation
        self._paw_samples = 0
        self._breath_peak = 0.0         # Peak Paw this breath
        self._breath_peak_volume = 0.0  # Peak absolute lung volume this breath
        self._last_insp_volume = 0.0    # Volume at end of inspiration
        
    def set_mode(self, mode: str):
        """Set ventilator mode (VCV or PCV)."""
        if mode.upper() == "VCV":
            self.mode = VentMode.VCV
        elif mode.upper() == "PCV":
            self.mode = VentMode.PCV
        
    def set_settings(self, rr: float, vt: float, peep: float, ie: str = "1:2", 
                     mode: str = None, p_insp: float = None):
        """
        Update ventilator settings.
        
        Args:
            rr: Respiratory rate (bpm)
            vt: Tidal volume (L) - used in VCV mode
            peep: PEEP (cmH2O)
            ie: I:E ratio string (e.g., "1:2")
            mode: Optional mode string ("VCV" or "PCV")
            p_insp: Inspiratory pressure above PEEP (cmH2O) - used in PCV mode
        """
        self.set_rr = rr
        self.set_vt = vt  # Liters
        self.set_peep = peep
        
        if mode is not None:
            self.set_mode(mode)
            
        if p_insp is not None:
            self.set_p_insp = p_insp
        
        # Parse I:E ratio
        try:
            i, e = map(float, ie.split(':'))
            self.insp_time_fraction = i / (i + e)
        except (ValueError, AttributeError):
            # ValueError: bad number format; AttributeError: ie not a string
            self.insp_time_fraction = 1.0 / 3.0

    def step(self, dt: float) -> MechState:
        """
        Advance respiratory mechanics by dt seconds.
        
        Returns updated MechState with current Paw, flow, volume, and monitors.
        """
        # Handle vent off case
        if self.set_rr <= 0:
            self.state.phase = "EXP"
            self.state.flow = 0.0
            self.state.paw = self.set_peep
            # Passive exhalation if any volume present
            if self.state.volume > 0.001:
                time_constant = self.resistance * self.compliance
                if time_constant > 0:
                    decay_flow = -self.state.volume / time_constant
                    self.state.volume += decay_flow * dt
                    self.state.volume = max(0, self.state.volume)
                    self.state.flow = decay_flow * 60.0
            return self.state
            
        # 1. Determine Cycle Phase
        breath_period = 60.0 / self.set_rr
        insp_duration = breath_period * self.insp_time_fraction
        exp_duration = breath_period - insp_duration
        
        self.cycle_time += dt
        
        # Check for new breath
        new_breath = False
        if self.cycle_time >= breath_period:
            self.cycle_time -= breath_period
            new_breath = True
            
            # Calculate auto-PEEP from end-expiratory volume
            # Auto-PEEP = EELV / Compliance
            if self.state.volume > 0.005:  # > 5mL residual
                self.state.auto_peep = self.state.volume / self.compliance
            else:
                self.state.auto_peep = 0.0
                
            self.state.eelv = self.state.volume
            
            # Update breath-cycle monitors
            self.state.paw_peak = self._breath_peak
            vt_l = max(0.0, self._breath_peak_volume - self.state.eelv)
            self.state.delivered_vt = vt_l * 1000.0  # Convert to mL
            
            if self._paw_samples > 0:
                self.state.paw_mean = self._paw_accumulator / self._paw_samples
            
            # Reset accumulators
            self._breath_peak = 0.0
            self._breath_peak_volume = self.state.volume
            self._paw_accumulator = 0.0
            self._paw_samples = 0
            
        in_insp = self.cycle_time < insp_duration
        
        # Phase transition detection
        phase_changed = (in_insp and self.state.phase == "EXP") or \
                       (not in_insp and self.state.phase == "INSP")
                       
        if phase_changed and self.state.phase == "INSP":
            # End of inspiration - record plateau pressure and volume
            self.state.paw_plat = self.state.volume / self.compliance + self.set_peep
            self._last_insp_volume = self.state.volume
        
        # 2. Calculate Flow and Pressure based on Mode
        target_flow = 0.0
        
        if in_insp:
            self.state.phase = "INSP"
            
            if self.mode == VentMode.VCV:
                target_flow = self._step_vcv_insp(insp_duration)
            else:  # PCV
                target_flow = self._step_pcv_insp(dt)
        else:
            self.state.phase = "EXP"
            target_flow = self._step_expiration()
            
        # 3. Integrate Volume
        self.state.volume += target_flow * dt
        
        # Clamp Volume (cannot go below FRC/0)
        if self.state.volume < 0:
            self.state.volume = 0
            if self.state.phase == "EXP":
                target_flow = 0
                
        # Track breath tidal volume
        if in_insp and target_flow > 0:
            self._breath_peak_volume = max(self._breath_peak_volume, self.state.volume)
                
        # 4. Calculate Airway Pressure
        paw = self._calculate_paw(target_flow)
        self.state.paw = paw
        self.state.flow = target_flow * 60.0  # Convert L/s to L/min
        
        # 5. Update monitors
        self._breath_peak = max(self._breath_peak, paw)
        self._paw_accumulator += paw
        self._paw_samples += 1
        
        return self.state
    
    def _step_vcv_insp(self, insp_duration: float) -> float:
        """
        VCV inspiration: Constant flow to deliver set Vt.
        
        Returns flow in L/s.
        """
        if insp_duration > 0:
            # Square wave flow = Vt / Ti
            return self.set_vt / insp_duration
        return 0.0
    
    def _step_pcv_insp(self, dt: float) -> float:
        """
        PCV inspiration: Pressure-driven decelerating flow.
        
        Flow = (Paw_set - Palv) / R
        where Palv = V/C + PEEP + auto_PEEP
        
        Returns flow in L/s.
        """
        # Total driving pressure
        total_peep = self.set_peep + self.state.auto_peep
        p_insp_total = self.set_p_insp + self.set_peep
        
        # Alveolar pressure (elastic recoil + PEEP)
        p_alv = self.state.volume / self.compliance + total_peep
        
        # Pressure gradient drives flow
        pressure_gradient = p_insp_total - p_alv
        
        if pressure_gradient > 0 and self.resistance > 0:
            flow = pressure_gradient / self.resistance
            # Limit maximum flow (realistic airways limit ~1.5 L/s inspiratory)
            flow = min(flow, 1.5)
            return flow
        return 0.0
    
    def _step_expiration(self) -> float:
        """
        Passive expiration: Exponential decay driven by elastic recoil.
        
        Flow = -V / (R * C)
        
        Returns flow in L/s (negative for expiration).
        """
        time_constant = self.resistance * self.compliance
        
        if time_constant > 0 and self.state.volume > 0:
            # Expiratory flow driven by elastic recoil
            flow = -self.state.volume / time_constant
            return flow
        return 0.0
    
    def _calculate_paw(self, flow: float) -> float:
        """
        Calculate airway pressure using equation of motion.
        
        Paw = V/C + R*Flow + PEEP
        
        Args:
            flow: Current flow in L/s
            
        Returns:
            Airway pressure in cmH2O
        """
        total_peep = self.set_peep + self.state.auto_peep
        
        if self.state.phase == "EXP":
            # During expiration, Paw at airway opening = PEEP
            # (assuming no expiratory resistance issues)
            return total_peep
        else:
            # During inspiration
            if self.mode == VentMode.PCV:
                # In PCV, Paw is controlled (set value)
                return self.set_p_insp + self.set_peep
            else:
                # In VCV, Paw is calculated from equation of motion
                elastic_p = self.state.volume / self.compliance
                resistive_p = self.resistance * flow
                return elastic_p + resistive_p + total_peep
    
    def get_mean_paw(self) -> float:
        """Return current mean airway pressure."""
        return self.state.paw_mean
    
    def get_auto_peep(self) -> float:
        """Return current auto-PEEP."""
        return self.state.auto_peep
    
    def get_total_peep(self) -> float:
        """Return total PEEP (set + auto)."""
        return self.set_peep + self.state.auto_peep
