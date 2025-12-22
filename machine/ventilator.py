"""
Anesthesia Ventilator Model.

Handles ventilator settings and monitoring for both VCV and PCV modes.
"""

from dataclasses import dataclass
from physiology.resp_mech import VentMode


@dataclass
class VentSettings:
    """Ventilator settings storage."""
    mode: str = "VCV"         # VCV, PCV
    tv: float = 500.0         # Target tidal volume (mL) - VCV
    rr: float = 12.0          # Respiratory rate (bpm)
    peep: float = 5.0         # PEEP (cmH2O)
    ie_ratio: float = 0.5     # I:E ratio as I/E (1:2 = 0.5)
    fio2: float = 0.21        # Fraction inspired O2
    p_insp: float = 15.0      # Inspiratory pressure above PEEP (cmH2O) - PCV
    

@dataclass
class VentMonitors:
    """Ventilator monitoring values."""
    paw_peak: float = 0.0     # Peak airway pressure (cmH2O)
    paw_plat: float = 0.0     # Plateau pressure (cmH2O)
    paw_mean: float = 0.0     # Mean airway pressure (cmH2O)
    auto_peep: float = 0.0    # Intrinsic PEEP (cmH2O)
    mv_exp: float = 0.0       # Expired minute ventilation (L/min)
    tv_exp: float = 0.0       # Expired tidal volume (mL)
    rr_total: float = 0.0     # Total respiratory rate (bpm)
    compliance: float = 50.0  # Dynamic compliance (mL/cmH2O)
    

class AnesthesiaVentilator:
    """
    Anesthesia Ventilator Model.
    
    Controls settings and outputs flow/pressure targets for Respiratory Mechanics.
    Tracks monitoring values including peak, plateau, and mean Paw.
    
    Supports:
    - VCV (Volume Control Ventilation): Set Vt, Paw varies
    - PCV (Pressure Control Ventilation): Set P_insp, Vt varies
    """
    
    def __init__(self):
        self.settings = VentSettings()
        self.monitors = VentMonitors()
        self.is_on = True
        
        # Internal tracking for rolling averages
        self._paw_peak_history = []
        self._tv_exp_history = []
        self._breath_count = 0
        
    def set_mode(self, mode: str):
        """Set ventilator mode (VCV or PCV)."""
        mode_upper = mode.upper()
        if mode_upper in ["VCV", "PCV"]:
            self.settings.mode = mode_upper
        
    def update_settings(self, rr=None, tv=None, peep=None, fio2=None, 
                       ie=None, p_insp=None, mode=None):
        """
        Update ventilator settings.
        
        Args:
            rr: Respiratory rate (bpm)
            tv: Tidal volume (mL)
            peep: PEEP (cmH2O)
            fio2: Fraction inspired O2
            ie: I:E ratio as string (e.g., "1:2") or float
            p_insp: Inspiratory pressure above PEEP (cmH2O) - PCV
            mode: Ventilator mode ("VCV" or "PCV")
        """
        if rr is not None: 
            self.settings.rr = rr
        if tv is not None: 
            self.settings.tv = tv
        if peep is not None: 
            self.settings.peep = peep
        if fio2 is not None: 
            self.settings.fio2 = fio2
        if p_insp is not None: 
            self.settings.p_insp = p_insp
        if mode is not None:
            self.set_mode(mode)
        
        if ie is not None:
            # Parse "1:2" format
            try:
                if isinstance(ie, str) and ':' in ie:
                    parts = ie.split(':')
                    i, e = float(parts[0]), float(parts[1])
                    self.settings.ie_ratio = i / e
                else:
                    self.settings.ie_ratio = float(ie)
            except:
                pass

    def step(self, dt: float, mech_state):
        """
        Update monitors based on respiratory mechanics state.
        
        Args:
            dt: Time step (seconds)
            mech_state: MechState from respiratory mechanics model
        """
        # Update from mechanics state
        self.monitors.paw_peak = mech_state.paw_peak
        self.monitors.paw_plat = mech_state.paw_plat
        self.monitors.paw_mean = mech_state.paw_mean
        self.monitors.auto_peep = mech_state.auto_peep
        self.monitors.tv_exp = mech_state.delivered_vt
        
        # Calculate minute ventilation
        if self.settings.rr > 0:
            self.monitors.mv_exp = (self.monitors.tv_exp / 1000.0) * self.settings.rr
        else:
            self.monitors.mv_exp = 0.0
            
        # Calculate dynamic compliance (only meaningful in VCV)
        # Cdyn = Vt / (Pplat - PEEP)
        if self.settings.mode == "VCV":
            delta_p = mech_state.paw_plat - self.settings.peep - mech_state.auto_peep
            if delta_p > 0.5:  # Avoid divide by zero
                self.monitors.compliance = self.monitors.tv_exp / delta_p
            else:
                self.monitors.compliance = 50.0  # Default
        else:
            # In PCV, we can infer compliance from delivered Vt
            if self.settings.p_insp > 0:
                self.monitors.compliance = self.monitors.tv_exp / self.settings.p_insp
                
        self.monitors.rr_total = self.settings.rr
        
    def get_mode_display(self) -> str:
        """Return human-readable mode string."""
        if self.settings.mode == "VCV":
            return f"VCV - Vt {int(self.settings.tv)} mL"
        else:
            return f"PCV - P {int(self.settings.p_insp)} cmH2O"
            
    def get_alarm_status(self) -> dict:
        """
        Check for ventilator alarms.
        
        Returns dict with alarm conditions.
        """
        alarms = {}
        
        # High peak pressure alarm (> 40 cmH2O)
        if self.monitors.paw_peak > 40:
            alarms['high_paw'] = self.monitors.paw_peak
            
        # Low tidal volume alarm (< 80% of set in VCV, or < 200mL in PCV)
        if self.settings.mode == "VCV":
            if self.monitors.tv_exp < 0.8 * self.settings.tv:
                alarms['low_tv'] = self.monitors.tv_exp
        else:
            if self.monitors.tv_exp < 200:
                alarms['low_tv'] = self.monitors.tv_exp
                
        # Auto-PEEP warning (> 5 cmH2O)
        if self.monitors.auto_peep > 5:
            alarms['auto_peep'] = self.monitors.auto_peep
            
        return alarms
