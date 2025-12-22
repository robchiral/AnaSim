
import numpy as np
from scipy.signal import cont2discrete
from typing import Optional
from core.constants import TCI_MIN_TARGET_CHANGE_INTERVAL, TCI_PEAK_TIME_MAX
from core.utils import clamp

class TCIController:
    """
    TCI Controller implementing the Shafer/Gregg algorithm.
    """
    def __init__(
        self,
        pk_model,
        drug_name: str,
        target_compartment: str = 'effect_site',
        max_rate: float = 1200.0, # mg/hr or similarly high
        sampling_time: float = 1.0, # sec
        control_time: float = 10.0, # sec
    ):
        """
        Args:
            pk_model: Instance of PK model (ThreeCompartmentPK, etc.)
            drug_name: 'Propofol', 'Remifentanil', etc.
            target_compartment: 'plasma' or 'effect_site'
            max_rate: Maximum infusion rate in model units (mg/s or ug/s).
                      Propofol example: 1200 mL/h at 20 mg/mL -> 24,000 mg/h (~6.6 mg/s).
                      Remifentanil example: 50 ug/mL.
            sampling_time: Internal calculation step (s)
            control_time: Control update interval (s)
        """
        self.pk_model = pk_model
        self.drug_name = drug_name
        self.sampling_time = sampling_time
        self.control_time = control_time
        self.max_rate = max_rate
        
        if sampling_time > control_time:
            raise ValueError("Sampling time cannot be larger than control time")
            
        A_min, B_min = pk_model.get_ss_matrices()
        
        # Convert to 1/sec
        A_sec = A_min / 60.0
        B_sec = B_min # B matrix (1/V1) is invariant to time unit if input is Mass/Time
        
        n_states = A_sec.shape[0]
        C_dummy = np.eye(n_states)
        D_dummy = np.zeros((n_states, 1))
        
        # Discretize for simulation step
        sys_sim = cont2discrete((A_sec, B_sec, C_dummy, D_dummy), sampling_time, method='bilinear')
        self.Ad = sys_sim[0]
        self.Bd = sys_sim[1]
        
        # Discretize for control step
        sys_ctrl = cont2discrete((A_sec, B_sec, C_dummy, D_dummy), control_time, method='bilinear')
        self.Ad_control = sys_ctrl[0]
        self.Bd_control = sys_ctrl[1]
        
        # Identify Target Index
        # Generic: [c1, c2, c3, ce] or [c1, c2, 0, 0] or [c1, c2, ce1, ce2]
        # Plasma is always index 0.
        # Effect site uses the last index for current PK models.
        # ThreeCompartmentPK: [c1, c2, c3, ce] -> index 3
        # RocuroniumPK: [c1, c2, c3, ce] -> index 3
        # NorepinephrinePK: [c1] or [c1, c2]. 
        # For Norepi, usually Plasma target.
        
        if target_compartment == 'plasma':
            self.target_id = 0
        else:
            # Effect site
            if drug_name == 'Norepinephrine':
                 self.target_id = 0 # No effect site model usually
            else:
                 self.target_id = n_states - 1 # Assume last state is effect site
                 
        self.n_state = n_states
        
        # Precompute Peak Time (Ttpe)
        self._compute_peak_time()
        
        # State variables
        self.x = np.zeros((n_states, 1)) # Estimated patient state
        self.infusion_rate = 0.0
        self.target = 0.0
        
        # Peak finding helpers
        self.tpeak_0 = 0.0
        self.tpeak_1 = 0.0
        self.time = 0.0  # Deprecated: use sim_time parameter instead
        self._last_target_change_time = -999.0  # For rate-limiting target changes
        self._min_target_change_interval = TCI_MIN_TARGET_CHANGE_INTERVAL
        
        # Endogenous input (for Norepi support if needed, but usually 0 for TCI)
        self.u_endo = 0.0 

    def _compute_peak_time(self):
        """Simulate a bolus to find time to peak at effect site."""
        x = np.zeros((self.n_state, 1))
        x_prev = np.zeros((self.n_state, 1))
        
        self.Ce_response = []
        t = self.sampling_time
        found_peak = False
        
        # Simulate impulse/short infusion
        # Burst infusion of 1 unit for control_time duration.
        
        # We want the response to a single control interval infusion
        infusion = 1.0
        
        # Simulate until peak found or reasonable timeout
        max_t = 600.0 # 10 min
        self.t_peak = 0.0
         
        while not found_peak and t < max_t:
            u = infusion if t <= self.control_time else 0.0
            
            x = self.Ad @ x + self.Bd * u
            
            ce = float(x[self.target_id, 0])
            self.Ce_response.append(ce)
            
            if t > self.control_time and ce < x_prev[self.target_id, 0] and self.t_peak == 0.0:
                self.t_peak = t - self.sampling_time
                found_peak = True
                
            x_prev = x.copy()
            t += self.sampling_time
            
        self.Ce_response = np.array(self.Ce_response)
        
        if self.t_peak == 0.0:
            # If logic fails (e.g. plasma target), peak is immediate or at end of infusion
            if self.target_id == 0:
                self.t_peak = self.control_time
            else:
                self.t_peak = self.control_time # Fallback

    def step(self, target: float, sim_time: float = None) -> float:
        """
        Perform one control step.
        Returns infusion rate (mass/sec).
        Call this every `sampling_time`.
        
        Args:
            target: Target concentration
            sim_time: Current simulation time (seconds). If provided, syncs
                      controller timing with simulation clock to prevent drift.
        """
        # Use provided sim_time or fall back to internal tracking
        if sim_time is not None:
            self.time = sim_time
        
        # Check if it's time to update control (every control_time)
        # We use modulo arithmetic with tolerance
        update_control = False
        if abs(self.time % self.control_time) < (self.sampling_time / 2.0):
            update_control = True
        
        # Rate-limit target changes to prevent control instability
        # Only react to target change if enough time has passed
        # Exception: zero target always bypasses rate-limit (safety: stop infusion immediately)
        target_changed = (target != self.target)
        if target_changed:
            if target == 0.0:
                # Always allow immediate stop
                update_control = True
                self._last_target_change_time = self.time
            else:
                time_since_last_change = self.time - self._last_target_change_time
                if time_since_last_change >= self._min_target_change_interval:
                    update_control = True
                    self._last_target_change_time = self.time
                # If too soon, ignore the change this step (will be picked up later)
             
        if update_control:
            if target != self.target:
                self.tpeak_0 = self.t_peak
                self.target = target
                
            # Prediction from current state (Zero input response)
            x_pred = self.x.copy()
            
            # Predict ahead to t_peak or next control step
            # We need Ce values curve for optimization
            # Horizon: max(NextControl, Tpeak)
            horizon = max(self.control_time + self.sampling_time, self.t_peak)
            steps = int(horizon / self.sampling_time)
            
            Ce_pred = np.zeros(steps)
            for i in range(steps):
                x_pred = self.Ad @ x_pred # + Bd * u_endo
                Ce_pred[i] = x_pred[self.target_id, 0]
                
            # Logic:
            # 1. If we can reach target in the next control step
            ce_next = Ce_pred[0] # Next-step estimate
            
            # PAS Logic:
            # if Ce_0[0] close to target (within 5%), solve for exact infusion to hit target at next control step.
            # else solve to hit target at peak time.
            
            tol = 0.05 * target if target > 0 else 0.001
            
            if abs(ce_next - target) < tol:
                # Solve: Target = (Ad_control @ x + Bd_control * rate)[target_id]
                # rate = (Target - (Ad_control @ x)[target_id]) / Bd_control[target_id]
                num = target - (self.Ad_control @ self.x)[self.target_id, 0]
                den = self.Bd_control[self.target_id, 0]
                if den != 0:
                    calc_rate = num / den
                else:
                    calc_rate = 0.0
                self.infusion_rate = max(0.0, calc_rate)
            else:
                # Reach target at Tpeak
                if target == 0:
                    self.infusion_rate = 0.0
                elif Ce_pred[int(self.control_time / self.sampling_time)] > self.target:
                    # Overshoot -> stop infusion
                    self.infusion_rate = 0.0
                else:
                    # Iterative search for rate
                    # Initial guess
                    idx_peak = int(self.tpeak_0 / self.sampling_time) - 1
                    if idx_peak < 0:
                        idx_peak = 0
                    if idx_peak >= len(Ce_pred):
                        idx_peak = len(Ce_pred) - 1

                    ce_peak_base = Ce_pred[idx_peak]
                    ce_resp_peak = self.Ce_response[idx_peak]

                    if ce_resp_peak != 0:
                        rate_guess = (target - ce_peak_base) / ce_resp_peak
                    else:
                        rate_guess = 0.0

                    # Refine peak time
                    self.infusion_rate = clamp(rate_guess, 0.0, self.max_rate)

                    # Re-evaluate peak time with this infusion
                    # Total Ce = Ce_pred + rate * Ce_response
                    # We want argmax(Total Ce)
                    # PAS optimization loop omitted; use the initial estimate.
        
        # Update State Estimate
        self.x = self.Ad @ self.x + self.Bd * self.infusion_rate
        
        # Advance internal time (used if sim_time not provided)
        if sim_time is None:
            self.time += self.sampling_time
        
        # Clamp rate
        self.infusion_rate = clamp(self.infusion_rate, 0.0, self.max_rate)
        
        if isinstance(self.infusion_rate, np.ndarray):
            return float(self.infusion_rate)
            
        return self.infusion_rate
    def set_state(self, c1: float, c2: float = 0.0, c3: float = 0.0, ce: float = 0.0):
        """
        Manually set the internal state estimate.
        """
        if self.n_state >= 1: self.x[0, 0] = c1
        if self.n_state >= 2: self.x[1, 0] = c2
        if self.n_state >= 3: self.x[2, 0] = c3
        if self.n_state >= 4: self.x[3, 0] = ce
