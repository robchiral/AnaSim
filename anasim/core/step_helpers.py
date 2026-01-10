"""
Step Helper Methods Mixin for SimulationEngine.

This module contains the private _step_* methods that implement the core
simulation update logic. They are extracted here for maintainability while
preserving the SimulationEngine API.
"""

from typing import TYPE_CHECKING

from .state import AirwayType
from anasim.core.constants import (
    TEMP_METABOLIC_COEFFICIENT,
    RR_APNEA_THRESHOLD,
    SHIVER_BASE_THRESHOLD,
    SHIVER_DEPTH_DROP_MAX,
    SHIVER_REMI_DROP_MAX,
    SHIVER_DELTA_FULL,
    SHIVER_BIS_ON,
    SHIVER_BIS_FULL,
    SHIVER_MAX_MULTIPLIER,
    SHIVER_TAU_ON,
    SHIVER_TAU_OFF,
)
from anasim.core.utils import pao2_to_sao2, clamp, clamp01, hill_function
from anasim.monitors.capno import Capnograph
from anasim.physiology.resp_mech import VentMode

if TYPE_CHECKING:
    from .engine import SimulationEngine

TCI_TARGETS = (
    ("tci_prop", "propofol_rate_mg_sec"),
    ("tci_remi", "remi_rate_ug_sec"),
    ("tci_nore", "nore_rate_ug_sec"),
    ("tci_epi", "epi_rate_ug_sec"),
    ("tci_phenyl", "phenyl_rate_ug_sec"),
    ("tci_roc", "roc_rate_mg_sec"),
)

ZERO_DIST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

class StepHelpersMixin:
    """
    Mixin providing step helper methods for SimulationEngine.
    
    These methods implement the per-step update logic for various subsystems:
    - Temperature model
    - Disturbances and events
    - TCI controllers
    - Machine state
    - Pharmacokinetics
    - Physiology (hemodynamics, respiration)
    - Monitors
    - Patient viability checks
    """
    
    def _phase_from_rr(self: "SimulationEngine", rr: float) -> str:
        """Calculate respiratory phase from rate."""
        if rr <= 0:
            return "EXP"
        cycle_time = 60.0 / rr
        insp_time = cycle_time / 3.0
        t_cycle = self.state.time % cycle_time
        return "INSP" if t_cycle < insp_time else "EXP"

    def _step_mechanics(self: "SimulationEngine", dt: float, vent_active: bool, bag_mask_active: bool):
        """Advance respiratory mechanics and return (mech_state, total_peep_effect, mech_rr_for_resp)."""
        resp_mech = self.resp_mech
        def estimate_effort(vt_l: float) -> float:
            """Estimate patient inspiratory effort (cmH2O) from tidal volume and compliance."""
            if resp_mech.compliance <= 0:
                return 0.0
            return clamp(vt_l / resp_mech.compliance, 0.0, 20.0)

        if vent_active:
            mech_rr_for_resp = resp_mech.set_rr
            if resp_mech.mode in (VentMode.PSV, VentMode.CPAP):
                # Patient-triggered support: use spontaneous RR/VT to derive timing and effort.
                saved_settings = resp_mech.snapshot_settings()
                spont_rr = max(0.0, self.resp.state.rr)
                spont_vt_l = max(0.0, self.resp.state.vt / 1000.0)
                is_apneic = spont_rr < RR_APNEA_THRESHOLD or self.resp.state.apnea
                if resp_mech.mode == VentMode.PSV:
                    if is_apneic:
                        self._psv_apnea_timer += dt
                    else:
                        self._psv_apnea_timer = 0.0
                    backup_rr = resp_mech.set_rr if resp_mech.set_rr > 0.0 else 0.0
                    use_backup = (
                        is_apneic
                        and backup_rr > 0.0
                        and self._psv_apnea_timer >= self.psv_apnea_backup_delay
                    )
                    if use_backup:
                        spont_rr = backup_rr
                else:
                    self._psv_apnea_timer = 0.0
                    use_backup = False
                if is_apneic:
                    spont_vt_l = 0.0
                mech_rr_for_resp = spont_rr

                effort_cm_h2o = estimate_effort(spont_vt_l)
                if use_backup:
                    effort_cm_h2o = 0.0
                self._last_patient_effort_cmH2O = effort_cm_h2o

                support_cm_h2o = resp_mech.set_p_insp if resp_mech.mode == VentMode.PSV else 0.0
                resp_mech.set_rr = mech_rr_for_resp
                resp_mech.set_p_insp = clamp(support_cm_h2o, 0.0, 40.0)
                resp_mech.patient_effort_cmH2O = effort_cm_h2o

                mech_state = resp_mech.step(dt)
                total_peep_effect = resp_mech.get_total_peep()
                resp_mech.restore_settings(saved_settings)
                resp_mech.patient_effort_cmH2O = 0.0
            else:
                self._psv_apnea_timer = 0.0
                self._last_patient_effort_cmH2O = 0.0
                resp_mech.patient_effort_cmH2O = 0.0
                mech_state = resp_mech.step(dt)
                total_peep_effect = resp_mech.get_total_peep()
        elif bag_mask_active:
            self._psv_apnea_timer = 0.0
            self._last_patient_effort_cmH2O = 0.0
            saved_settings = resp_mech.snapshot_settings()
            resp_mech.set_settings(self.bag_mask_rr, self.bag_mask_vt, 0.0, ie="1:2", mode="VCV")
            resp_mech.patient_effort_cmH2O = 0.0
            mech_state = resp_mech.step(dt)
            total_peep_effect = resp_mech.get_total_peep()
            resp_mech.restore_settings(saved_settings)
            mech_rr_for_resp = self.bag_mask_rr
        else:
            self._psv_apnea_timer = 0.0
            spont_rr = max(0.0, self.resp.state.rr)
            spont_vt_l = max(0.0, self.resp.state.vt / 1000.0)
            is_apneic = spont_rr < RR_APNEA_THRESHOLD or self.resp.state.apnea
            if is_apneic:
                spont_vt_l = 0.0
            self._last_patient_effort_cmH2O = estimate_effort(spont_vt_l)
            saved_settings = resp_mech.snapshot_settings()
            resp_mech.set_rr = 0.0
            resp_mech.set_peep = 0.0
            resp_mech.patient_effort_cmH2O = 0.0
            mech_state = resp_mech.step(dt)
            total_peep_effect = 0.0
            mech_state.paw_mean = 0.0
            mech_state.auto_peep = 0.0
            resp_mech.restore_settings(saved_settings)
            mech_rr_for_resp = 0.0
        return mech_state, total_peep_effect, mech_rr_for_resp

    def _update_nibp(self: "SimulationEngine", dt: float, hemo_state) -> None:
        """Update NIBP state and trigger cycles when appropriate."""
        state = self.state
        if state.time >= self._next_nibp_time and not self.nibp.is_cycling:
            self.nibp.trigger()
            self._next_nibp_time = state.time + self.nibp.interval

        prev_ts = state.nibp_timestamp
        cuff_p = self.nibp.step(dt, state.time, hemo_state.map, true_sys=getattr(hemo_state, "sbp", None))

        state.nibp_is_cycling = self.nibp.is_cycling
        state.nibp_cuff_pressure = cuff_p

        latest = self.nibp.latest_reading
        if latest.timestamp > 0.0 and latest.timestamp != prev_ts:
            state.nibp_sys = latest.systolic
            state.nibp_dia = latest.diastolic
            state.nibp_map = latest.map
            state.nibp_timestamp = latest.timestamp

    def _update_shivering(self: "SimulationEngine", dt: float) -> float:
        """
        Update shivering intensity based on temperature and anesthetic state.
        Returns shivering level (0-1).
        """
        state = self.state

        depth_metric = state.mac + (state.propofol_ce / 4.0)
        depth_factor = clamp01(depth_metric)

        remi_effect = 0.0
        if self.resp:
            remi_effect = hill_function(state.remi_ce, self.resp.c50_remi, self.resp.gamma_remi)

        threshold = (
            SHIVER_BASE_THRESHOLD
            - SHIVER_DEPTH_DROP_MAX * depth_factor
            - SHIVER_REMI_DROP_MAX * remi_effect
        )

        temp_deficit = max(0.0, threshold - state.temp_c)
        cold_drive = clamp01(temp_deficit / SHIVER_DELTA_FULL)

        if SHIVER_BIS_FULL <= SHIVER_BIS_ON:
            emergence = 1.0 if state.bis >= SHIVER_BIS_ON else 0.0
        else:
            emergence = clamp01((state.bis - SHIVER_BIS_ON) / (SHIVER_BIS_FULL - SHIVER_BIS_ON))

        muscle_factor = 1.0
        if self.resp:
            nmba_effect = hill_function(state.roc_ce, self.resp.c50_nmba, self.resp.gamma_nmba)
            muscle_factor = clamp01(1.0 - nmba_effect)

        target = cold_drive * emergence * muscle_factor

        tau = SHIVER_TAU_ON if target > self._shiver_level else SHIVER_TAU_OFF
        if tau > 0:
            self._shiver_level += (target - self._shiver_level) * (dt / tau)
        else:
            self._shiver_level = target
        self._shiver_level = clamp01(self._shiver_level)
        state.shivering = self._shiver_level
        return self._shiver_level
    
    def _step_temperature(self: "SimulationEngine", dt: float):
        """
        Update patient temperature based on metabolic heat production and heat loss.
        """
        state = self.state
        temp_c = state.temp_c
        # Metabolic heat production (reduced by anesthetic depth).
        depth_index = state.mac + (state.propofol_ce / 4.0)
        depth_factor = min(1.0, depth_index)
        
        metabolic_reduction = 0.3 * depth_factor
        current_production = self.heat_production_basal * (1.0 - metabolic_reduction)
        shiver_heat = self.heat_production_basal * SHIVER_MAX_MULTIPLIER * self._shiver_level
        current_production += shiver_heat
        
        # Heat loss: linear transfer vs ambient.
        t_ambient = 20.0
        
        # Base conductance tuned for ~80-100W loss at steady state (37C vs 20C).
        base_conductance = 3.0 
        
        # Vasodilation under anesthesia increases conductance (up to +50%).
        anest_conductance_boost = 0.5 * depth_factor
        total_conductance = base_conductance * (1.0 + anest_conductance_boost) * (self.surface_area / 1.9)
        
        heat_loss = total_conductance * (temp_c - t_ambient)
        
        # Redistribution: rapid deepening adds transient heat loss.
        d_depth = (depth_index - self._last_depth_index) / dt
        self._last_depth_index = depth_index
        
        if d_depth > 0:
            k_redist = self.K_REDIST
            redistribution_flux = k_redist * d_depth
            heat_loss += redistribution_flux
        
        # Active warming (Bair Hugger) via convection.
        warming_input = 0.0
        if state.bair_hugger_target > 0:
             k_bair = 7.0 
             dt_warming = max(0.0, state.bair_hugger_target - temp_c)
             warming_input = k_bair * dt_warming
        
        net_heat_flux = current_production + warming_input - heat_loss
        
        # dT = Q * dt / (mass * specific_heat).
        heat_capacity = self.patient.weight * self.specific_heat
        
        d_temp = (net_heat_flux * dt) / heat_capacity
        
        temp_c += d_temp
        # Clamp to physiologic bounds.
        state.temp_c = clamp(temp_c, 25.0, 42.0)

    def _step_disturbances(self: "SimulationEngine", dt: float) -> tuple:
        """Calculate disturbances and handle events."""
        state = self.state
        disturbance_active = self.disturbance_active
        disturbances = self.disturbances
        if not disturbance_active or not disturbances:
            dist_vec = ZERO_DIST
        else:
            t_rel = max(0.0, state.time - self.disturbance_start_time)
            dist_vec = disturbances.compute_dist(t_rel)
        if hasattr(dist_vec, "as_tuple"):
            dist_vec = dist_vec.as_tuple()
        d_bis, d_map, d_co, d_svr, d_sv, d_hr = dist_vec
        hemo = self.hemo

        # Attenuate stimulus with anesthetic depth (deeper = less response).
        # Use cached depth_index from step()
        depth_factor = min(1.0, self._depth_index)
        stim_gain = max(0.3, 1.0 - 0.6 * depth_factor)
        d_bis *= stim_gain
        d_svr *= stim_gain
        d_sv *= stim_gain
        d_hr *= stim_gain
        
        if self.active_hemorrhage and hemo:
            # Bleed rate (mL/s). Default 500 mL/min = 8.33 mL/s
            rate_sec = self.hemorrhage_rate_ml_min / 60.0
            hemo.add_volume(-rate_sec * dt)
        
        # Infusions (fluids / blood) with realistic rates.
        if self.pending_infusions:
            remaining = []
            for infusion in self.pending_infusions:
                rate_sec = infusion['rate'] / 60.0
                amount_this_step = min(infusion['remaining'], rate_sec * dt)
                infusion['remaining'] -= amount_this_step
                if amount_this_step > 0:
                    if hemo:
                        hemo.add_volume(amount_this_step, hematocrit=infusion['hematocrit'])
                if infusion['remaining'] > 1e-3:
                    remaining.append(infusion)
            self.pending_infusions[:] = remaining
            
        # Anaphylaxis: ramp on/off for realistic hemodynamic changes.
        if self.active_anaphylaxis:
            # Ramp up severity
            self.anaphylaxis_severity = min(1.0, self.anaphylaxis_severity + self.anaphylaxis_onset_rate * dt)
        else:
            # Decay severity when not active (simulates treatment effect)
            self.anaphylaxis_severity = max(0.0, self.anaphylaxis_severity - self.anaphylaxis_decay_rate * dt)
        
        if self.anaphylaxis_severity > 0.01:
            # Bronchospasm handled in airway model (see _update_airway_complications)
            pass

        # Sepsis / distributive shock: gradual onset/offset.
        if self.active_sepsis:
            self.sepsis_severity = min(1.0, self.sepsis_severity + self.sepsis_onset_rate * dt)
        else:
            self.sepsis_severity = max(0.0, self.sepsis_severity - self.sepsis_decay_rate * dt)
        if hemo:
            self.hemo.anaphylaxis_severity = self.anaphylaxis_severity
            self.hemo.sepsis_severity = self.sepsis_severity
            if hasattr(hemo, "invalidate_state_cache"):
                hemo.invalidate_state_cache()
        
        dist_vec = (d_bis, d_map, d_co, d_svr, d_sv, d_hr)
        return dist_vec

    def _step_tci(self: "SimulationEngine", dt: float):
        """Update TCI controllers."""
        sim_time = self.state.time
        for tci_attr, rate_attr in TCI_TARGETS:
            controller = getattr(self, tci_attr, None)
            if controller:
                rate = controller.step(controller.target, sim_time=sim_time)
                setattr(self, rate_attr, rate)

    def _step_machine(self: "SimulationEngine", dt: float) -> float:
        """
        Update Machine State (Circuit -> Composition).
        Returns: fi_sevo
        """
        state = self.state
        pk_sevo = self.pk_sevo
        circuit = self.circuit
        composition = circuit.composition
        # Use prior-step VA from state (spontaneous + mechanical).
        connected = (state.airway_mode != AirwayType.NONE)
        total_va = state.va if (connected and state.va > 0) else 0.0
        
        if not connected:
            fi_sevo = 0.0
            state.fio2 = 0.21
        else:
            fi_vapor_circuit = composition.fi_agent 
            fi_sevo = fi_vapor_circuit if self.active_agent == "Sevoflurane" else 0.0
            # Update state FiO2 from circuit
            state.fio2 = composition.fio2

        state.fi_sevo = fi_sevo * 100.0
        
        pk_sevo_state = pk_sevo.state
        p_alv_prev = pk_sevo_state.p_alv
        uptake_sevo = (fi_sevo - p_alv_prev) * total_va

        # O2 uptake tied to metabolic rate and temperature
        uptake_o2 = 0.25
        if self.resp:
            if abs(state.temp_c - self._cached_temp_metabolic) > 0.01:
                self._cached_temp_metabolic = state.temp_c
                self._cached_temp_metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - state.temp_c)
            metabolic_factor = self._cached_temp_metabolic_factor
            shiver_mult = 1.0 + SHIVER_MAX_MULTIPLIER * self._shiver_level
            vco2_ml_min = self.resp.vco2 * metabolic_factor * shiver_mult
            uptake_o2 = (vco2_ml_min / 1000.0) / max(self.resp.rq, 0.1)
        
        circuit.step(dt, uptake_o2, uptake_sevo)
        
        return fi_sevo

    def _step_pk(self: "SimulationEngine", dt: float, fi_sevo: float, co_curr: float):
        """Update Pharmacokinetics."""
        state = self.state
        pk_sevo = self.pk_sevo
        pk_prop = self.pk_prop
        pk_remi = self.pk_remi
        pk_nore = self.pk_nore
        pk_roc = self.pk_roc
        pk_epi = self.pk_epi
        pk_phenyl = self.pk_phenyl
        self._update_pk_hemodynamics(co_curr)

        pk_sevo.step(dt, fi_sevo, state.va, co_curr, temp_c=state.temp_c)
        pk_sevo_state = pk_sevo.state
        state.et_sevo = pk_sevo_state.p_alv * 100.0
        state.mac = pk_sevo_state.mac
        
        pk_prop.step(dt, self.propofol_rate_mg_sec)
        pk_remi.step(dt, self.remi_rate_ug_sec) 
        pk_nore.step(dt, self.nore_rate_ug_sec, propofol_conc_ug_ml=pk_prop.state.c1)
        pk_roc.step(dt, self.roc_rate_mg_sec)
        pk_epi.step(dt, self.epi_rate_ug_sec)
        pk_phenyl.step(dt, self.phenyl_rate_ug_sec)
        
        # Single point of truth for PK sync to public state.
        self._sync_pk_state()

    def _sync_pk_state(self: "SimulationEngine"):
        """Synchronize PK concentrations from subsystem states to public state.
        
        This is the only place where PK subsystem → engine state copying occurs.
        Called at the end of _step_pk after all PK models have been updated.
        
        All vasopressors now use effect-site concentration (.ce) for hemodynamic
        effects, providing realistic 1-2 minute delays after bolus administration.
        """
        state = self.state
        state.propofol_ce = self.pk_prop.state.ce
        state.propofol_cp = self.pk_prop.state.c1
        state.remi_ce = self.pk_remi.state.ce
        state.remi_cp = self.pk_remi.state.c1
        state.nore_ce = self.pk_nore.state.ce  # Effect-site (ke0 modeled)
        state.roc_ce = self.pk_roc.state.ce
        state.roc_cp = self.pk_roc.state.c1
        state.epi_ce = self.pk_epi.state.ce    # Effect-site (ke0 modeled)
        state.phenyl_ce = self.pk_phenyl.state.ce  # Effect-site (ke0 modeled)

    def _update_pk_hemodynamics(self: "SimulationEngine", co_curr: float):
        """Scale PK parameters based on current blood volume and CO."""
        if not self.hemo:
            return
        base_bv = getattr(self.hemo, "blood_volume_0", 0.0)
        base_co = getattr(self.hemo, "base_co_l_min", 0.0)
        if base_bv <= 0.0 or base_co <= 0.0:
            return
        v_ratio = self.hemo.blood_volume / base_bv
        co_ratio = co_curr / base_co
        v_ratio = clamp(v_ratio, 0.1, 2.0)
        co_ratio = clamp(co_ratio, 0.1, 2.0)
        for model in (
            self.pk_prop,
            self.pk_remi,
            self.pk_nore,
            self.pk_roc,
            self.pk_epi,
            self.pk_phenyl,
        ):
            if model and hasattr(model, "update_hemodynamics"):
                model.update_hemodynamics(v_ratio, co_ratio)

    def _update_airway_complications(self: "SimulationEngine", dt: float):
        """
        Update airway obstruction/bronchospasm and apply to mechanics.

        Integrates manual controls, anaphylaxis bronchospasm, and
        auto-triggered laryngospasm from light anesthesia + stimulation.
        """
        # Tolerance of airway stimulation (0-1). Higher = more tolerant.
        state = self.state
        if self.tol_pd:
            tol = self.tol_pd.compute_probability(
                state.propofol_ce, state.remi_ce, mac=state.mac
            )
        else:
            tol = getattr(state, "tol", 0.0)
        tol = clamp01(tol)

        # Airway stimulation signal (use intubation profile as proxy)
        stim_profile = self.disturbance_profile or ""
        stim_active = bool(
            self.auto_laryngospasm_enabled and
            self.disturbance_active and
            ("intubation" in stim_profile)
        )
        stim_scale = 1.0

        # NMBA reduces laryngospasm reflexes
        nmba_effect = 0.0
        if self.resp:
            nmba_effect = hill_function(state.roc_ce, self.resp.c50_nmba, self.resp.gamma_nmba)
        muscle_factor = clamp01(1.0 - nmba_effect)

        # Auto laryngospasm target (upper airway) only if not intubated
        laryng_target = 0.0
        if state.airway_mode != AirwayType.ETT and stim_active:
            light_factor = clamp01(1.0 - tol)
            laryng_target = clamp01(light_factor * muscle_factor * stim_scale)

        # First-order approach to target with separate on/off time constants
        tau = self.laryngospasm_tau_on if laryng_target > self.laryngospasm_severity else self.laryngospasm_tau_off
        if tau > 0:
            self.laryngospasm_severity += (laryng_target - self.laryngospasm_severity) * (dt / tau)
        self.laryngospasm_severity = clamp01(self.laryngospasm_severity)

        # Upper airway obstruction (manual + auto)
        upper_obstruction = self.airway_obstruction_manual
        if state.airway_mode != AirwayType.ETT:
            upper_obstruction = max(upper_obstruction, self.laryngospasm_severity)
        upper_obstruction = clamp01(upper_obstruction)

        # Bronchospasm (manual + anaphylaxis)
        bronch = 1.0 - (1.0 - self.bronchospasm_manual) * (1.0 - self.anaphylaxis_severity)
        bronch = clamp01(bronch)

        # Apply to mechanics (resistance in cmH2O/(L/s))
        base_r = getattr(self, "_base_airway_resistance", 10.0)
        r_upper = 40.0 * upper_obstruction
        r_bronch = 20.0 * bronch
        if self.resp_mech:
            self.resp_mech.resistance = base_r + r_upper + r_bronch

        # Patency: upper airway affects delivered volume; bronchospasm affects efficiency
        self._airway_patency = clamp(1.0 - upper_obstruction, 0.0, 1.0)
        self._ventilation_efficiency = clamp(1.0 - 0.5 * bronch - 0.2 * upper_obstruction, 0.1, 1.0)
        self._capno_obstruction = clamp01(upper_obstruction + 0.7 * bronch)
        self._vq_mismatch = clamp01(0.85 * bronch + 0.25 * upper_obstruction)

        # Publish to public state
        self._tol_current = tol
        state.airway_obstruction = upper_obstruction
        state.bronchospasm = bronch
        state.laryngospasm = self.laryngospasm_severity

    def _step_physiology(self: "SimulationEngine", dt: float, dist_vec: tuple):
        """Update Hemo and Respiration with enhanced ventilator dynamics."""
        _d_bis, _d_map, _d_co, d_svr, d_sv, d_hr = dist_vec
        state = self.state
        resp_mech = self.resp_mech
        resp = self.resp
        hemo = self.hemo
        
        self._update_airway_complications(dt)

        connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
        vent_active = connected and self.vent.is_on
        bag_mask_active = self.bag_mask_active and connected and not vent_active
        assisted_active = vent_active or bag_mask_active

        # Mechanical lung step (assisted ventilation only).
        mech_state, total_peep_effect, mech_rr_for_resp = self._step_mechanics(dt, vent_active, bag_mask_active)
        
        # Intrathoracic pressure from smoothed mean Paw.
        alpha_paw = 0.05  # Faster tracking of mean Paw changes
        if mech_state.paw_mean > 0:
            self.current_mean_paw = (1 - alpha_paw) * self.current_mean_paw + alpha_paw * mech_state.paw_mean
        else:
            self.current_mean_paw = (1 - alpha_paw) * self.current_mean_paw + alpha_paw * mech_state.paw
        
        # Higher mean Paw increases Pit, reducing venous return.
        # Spontaneous inspiratory effort makes Pit more negative (improves venous return).
        pit_base = getattr(hemo, "pit_0", -2.0)
        pit_estimate = pit_base + 0.4 * (self.current_mean_paw - 5.0)
        effort_mmHg = self._last_patient_effort_cmH2O * 0.74
        pit_estimate -= 0.3 * effort_mmHg

        # Mechanical ventilator MV calculation.
        mech_rr = mech_rr_for_resp if vent_active else 0.0
        delivered_vt_raw_l = resp_mech.set_vt if vent_active else 0.0
        delivered_vt_display_l = 0.0
        if vent_active:
            if mech_state.delivered_vt > 0:
                delivered_vt_raw_l = mech_state.delivered_vt / 1000.0
            delivered_vt_display_l = delivered_vt_raw_l * self._airway_patency
            mech_state.delivered_vt = delivered_vt_display_l * 1000.0
        mech_vent_mv = mech_rr * delivered_vt_raw_l if vent_active else 0.0
        
        # Bag-mask ventilation (separate from mechanical vent).
        bag_mask_mv = 0.0
        assisted_rr_for_resp = mech_rr
        assisted_vt_for_resp = delivered_vt_raw_l
        assisted_vt_display = delivered_vt_display_l
        if bag_mask_active:
            bag_mask_mv = self.bag_mask_rr * self.bag_mask_vt
            assisted_rr_for_resp = self.bag_mask_rr
            assisted_vt_for_resp = self.bag_mask_vt
            assisted_vt_display = self.bag_mask_vt * self._airway_patency
        
        # Total assisted MV = mechanical vent + bag-mask (mutually exclusive in practice).
        total_assisted_mv = mech_vent_mv + bag_mask_mv
        
        # Respiration step with PEEP for oxygenation.
        resp_state = resp.step(
            dt, 
            ce_prop=state.propofol_ce, 
            ce_remi=state.remi_ce, 
            mech_vent_mv=total_assisted_mv,  # Use combined assisted MV
            fio2=state.fio2,
            ce_roc=state.roc_ce, 
            et_sevo=state.et_sevo,
            mac_sevo=state.mac,
            peep=total_peep_effect,  # Pass total PEEP for oxygenation
            mean_paw=self.current_mean_paw,
            temp_c=state.temp_c,
            mech_rr=assisted_rr_for_resp,
            mech_vt_l=assisted_vt_for_resp,
            airway_patency=self._airway_patency,
            ventilation_efficiency=self._ventilation_efficiency,
            vq_mismatch=self._vq_mismatch,
            hb_g_dl=hemo.hb_conc,
            oxygen_delivery_ratio=self._do2_ratio,
            shiver_level=self._shiver_level,
            cardiac_output=state.co
        )
        
        spont_rr = resp_state.rr
        spont_vt_l = resp_state.vt / 1000.0
        
        # Calculate synchronized MV (prevent double counting).
        if assisted_active:
             eff_rr = max(assisted_rr_for_resp, spont_rr)
             eff_vt = max(assisted_vt_display, spont_vt_l)
             total_patient_mv = eff_rr * eff_vt
        else:
             total_patient_mv = spont_rr * spont_vt_l
        
        assisted_rr = self.bag_mask_rr if bag_mask_active else mech_rr

        phase = mech_state.phase
        if not assisted_active:
            phase = self._phase_from_rr(spont_rr)
        elif bag_mask_active and not vent_active:
            phase = self._phase_from_rr(self.bag_mask_rr)
        
        # Displayed RR matches clinical capnography/flow-based detection.
        if assisted_active:
            state.rr = max(assisted_rr, spont_rr)
        else:
            state.rr = spont_rr

        mac_sevo = state.mac
        hemo_state = hemo.step(
            dt,
            state.propofol_ce,
            state.remi_ce,
            state.nore_ce,
            pit=pit_estimate,
            paco2=resp_state.p_alveolar_co2,
            pao2=resp_state.p_arterial_o2,
            dist_hr=d_hr,
            dist_sv=d_sv,
            dist_svr=d_svr,
            mac=state.mac,
            mac_sevo=mac_sevo,
            ce_epi=state.epi_ce,
            ce_phenyl=state.phenyl_ce,
            temp_c=state.temp_c,
            peep_cmH2O=total_peep_effect,
        )
        
        self.vent.step(dt, mech_state, rr_total=mech_rr)
                                    
        state.map = hemo_state.map # Raw
        state.hr = hemo_state.hr   # Raw
        state.sv = hemo_state.sv
        state.svr = hemo_state.svr 
        state.co = hemo_state.co
        
        # New hemodynamic fields
        state.sbp = getattr(hemo_state, 'sbp', state.map + 20)
        state.dbp = getattr(hemo_state, 'dbp', state.map - 20)
        state.blood_volume = getattr(hemo, 'blood_volume', 5000.0)
        state.hb_g_dl = getattr(hemo, 'hb_conc', state.hb_g_dl)
        if hasattr(hemo, 'get_hematocrit'):
            state.hct = hemo.get_hematocrit()

        state.pit = pit_estimate
        
        state.paco2 = resp_state.p_alveolar_co2
        state.pao2 = resp_state.p_arterial_o2
        
        # Use delivered Vt from mechanics (important for PCV mode).
        if vent_active and mech_state.delivered_vt > 0:
            state.vt = mech_state.delivered_vt
        elif vent_active:
            state.vt = resp_mech.set_vt * 1000.0
        else:
            state.vt = resp_state.vt
            
        state.mv = total_patient_mv
        state.va = resp_state.va
        state.apnea = resp_state.apnea
        state.paw = mech_state.paw
        state.flow = mech_state.flow
        state.volume = mech_state.volume
        self._vent_active = vent_active
        pao2_est = max(0.0, resp_state.p_arterial_o2)
        sao2_est = pao2_to_sao2(pao2_est)
        co_for_do2 = max(0.1, state.co)
        self._do2_ratio = hemo.compute_do2_ratio(sao2_est / 100.0, pao2_est, co_for_do2)
        state.oxygen_delivery_ratio = self._do2_ratio
        
        return hemo_state, resp_state, phase

    def _compute_capno_value(self: "SimulationEngine", dt: float, phase: str, resp_state):
        """Compute capnography waveform value for the current step."""
        state = self.state
        if state.airway_mode == AirwayType.NONE:
            return 0.0
        if self._airway_patency < 0.05 or state.rr == 0:
            # Near-complete obstruction or apnea: no measurable waveform.
            self.capno.state.co2 = 0.0
            return 0.0

        resp_mech = self.resp_mech
        bag_mask_active = self.bag_mask_active and not self._vent_active
        vent_rr = resp_mech.set_rr if self._vent_active else (self.bag_mask_rr if bag_mask_active else 0.0)
        insp_fraction = resp_mech.insp_time_fraction
        if bag_mask_active and not self._vent_active:
            insp_fraction = 1.0 / 3.0

        capno_context = Capnograph.build_context(
            resp_state,
            vent_rr=vent_rr,
            insp_fraction=insp_fraction,
            vent_active=(self._vent_active or bag_mask_active),
        )
        capno_phase = phase
        capno_exp_duration = capno_context.exp_duration
        capno_is_spontaneous = capno_context.is_spontaneous
        capno_curare = capno_context.curare_active

        if self._vent_active or bag_mask_active:
            support_mode = self._vent_active and resp_mech.mode in (VentMode.PSV, VentMode.CPAP)
            if support_mode:
                # PSV/CPAP are patient-driven; always render spontaneous timing.
                capno_is_spontaneous = True
                capno_curare = False
                capno_phase = self._phase_from_rr(resp_state.rr)
                if resp_state.rr > 0:
                    capno_exp_duration = (60.0 / resp_state.rr) * 0.65
            else:
                # If spontaneous effort dominates, treat capno timing as patient-driven (AC-like).
                # Otherwise lock to driven ventilation timing.
                if capno_context.spontaneous_weight >= 0.6:
                    capno_phase = self._phase_from_rr(capno_context.effective_rr)
                    capno_exp_duration = capno_context.exp_duration
                else:
                    driven_rr = vent_rr if vent_rr > 0.1 else capno_context.effective_rr
                    cycle_time = 60.0 / max(driven_rr, 0.1)
                    exp_fraction = max(0.1, 1.0 - insp_fraction)
                    capno_exp_duration = cycle_time * exp_fraction
        elif capno_is_spontaneous:
            capno_phase = self._phase_from_rr(resp_state.rr)

        capno_p_alv = resp_state.etco2 * self._airway_patency
        return self.capno.step(
            dt,
            capno_phase,
            capno_p_alv,
            is_spontaneous=capno_is_spontaneous,
            curare_cleft=capno_curare,
            exp_duration=capno_exp_duration,
            effort_scale=capno_context.effort_scale,
            airway_obstruction=self._capno_obstruction,
        )

    def _step_monitors(self: "SimulationEngine", dt: float, phase: str, hemo_state, resp_state, dist_vec):
        """Update Monitors and calculate smoothed state."""
        d_bis, d_map, _d_co, _d_svr, _d_sv, _d_hr = dist_vec
        state = self.state
        
        if self.patient.weight != self._remi_rate_weight:
            self._remi_rate_weight = self.patient.weight
            self._remi_rate_scale = 60.0 / self._remi_rate_weight
        remi_rate_ug_kg_min = self.remi_rate_ug_sec * self._remi_rate_scale

        mac_sevo = self.pk_sevo.state.mac
        bis_val = self.bis.step(dt, state.propofol_ce, state.remi_ce,
                                mac_sevo=mac_sevo,
                                remi_rate_ug_kg_min=remi_rate_ug_kg_min)
        capno_val = self._compute_capno_value(dt, phase, resp_state)
        
        # Neuromuscular PD uses step_recovery for effect-site & sugammadex binding.
        tof_val = self.tof_pd.step_recovery(dt, state.roc_cp, mac_sevo=mac_sevo)
        loc_val = self.loc_pd.compute_probability(state.propofol_ce, state.remi_ce)
        if getattr(self, "_tol_current", None) is not None:
            tol_val = self._tol_current
        else:
            tol_val = self.tol_pd.compute_probability(state.propofol_ce, state.remi_ce)

        rhythm = getattr(hemo_state, 'rhythm_type', None)
        state.ecg_voltage = self.ecg.step(dt, state_hr=hemo_state.hr, rhythm_type=rhythm)
        
        sao2 = resp_state.sao2        
        pleth, spo2_val = self.spo2_mon.step(dt, hr=hemo_state.hr, saturation=sao2)
        state.pleth_voltage = pleth
        
        self._update_nibp(dt, hemo_state)

        # Smoothing & final state updates.
        noise = self.rng.normal(0.0, self._monitor_noise_std)
        
        raw_map = hemo_state.map + d_map + noise[0]
        raw_hr = hemo_state.hr + noise[1]
        raw_bis = bis_val + d_bis + noise[2]
        
        alpha = 0.1
        self.smooth_map = (1 - alpha) * self.smooth_map + alpha * raw_map
        self.smooth_hr = (1 - alpha) * self.smooth_hr + alpha * raw_hr
        self.smooth_bis = (1 - alpha) * self.smooth_bis + alpha * raw_bis
        
        state.map = self.smooth_map
        state.hr = self.smooth_hr
        state.bis = clamp(self.smooth_bis, 0.0, 100.0)
        state.capno_co2 = capno_val
        state.etco2 = resp_state.etco2 if state.airway_mode != AirwayType.NONE else 0.0
        state.tof = tof_val
        state.loc = loc_val
        state.tol = tol_val
        state.spo2 = spo2_val
        
        monitor_vals = self._monitor_values
        monitor_vals["BIS"] = state.bis
        monitor_vals["MAP"] = state.map
        monitor_vals["HR"] = state.hr
        monitor_vals["EtCO2"] = state.etco2
        monitor_vals["SpO2"] = state.spo2
        state.alarms = self.alarms.update(monitor_vals)

    def _check_patient_viability(self: "SimulationEngine", dt: float):
        """
        Check if patient vitals are compatible with life.
        Only runs if enabled in config.
        """
        if not self.config.enable_death_detector:
            return
            
        if self.state.is_dead:
            return
            
        # Thresholds.
        MAP_CRITICAL_LOW = 20.0  # mmHg (Raised from 15 to ensure detection)
        HR_CRITICAL_LOW = 10.0   # bpm
        HR_CRITICAL_HIGH = 220.0 # bpm (matches HR clamp)
        
        if self.state.map < MAP_CRITICAL_LOW:
            self.time_hypotension += dt
        else:
            self.time_hypotension = max(0, self.time_hypotension - dt) # Decay accumulator
            
        if self.state.hr < HR_CRITICAL_LOW:
            self.time_brady += dt
        else:
            self.time_brady = max(0, self.time_brady - dt)
            
        if self.state.hr > HR_CRITICAL_HIGH:
            self.time_tachy += dt
        else:
             self.time_tachy = max(0, self.time_tachy - dt)
             
        if self.time_hypotension > self.DEATH_GRACE_PERIOD:
            self.state.is_dead = True
            self.state.death_reason = "Extreme Hypotension / Cardiac Arrest (MAP < 20 mmHg)"
            print(f"DEATH TRIGGERED: Hypotension ({self.state.map:.1f} mmHg)")
        elif self.time_brady > self.DEATH_GRACE_PERIOD:
            self.state.is_dead = True
            self.state.death_reason = "Asystole / Extreme Bradycardia (HR < 10 bpm)"
            print(f"DEATH TRIGGERED: Bradycardia ({self.state.hr:.1f} bpm)")
        elif self.time_tachy > self.DEATH_GRACE_PERIOD:
            self.state.is_dead = True
            self.state.death_reason = "Extreme Tachycardia / VFib (HR > 250 bpm)"
            print(f"DEATH TRIGGERED: Tachycardia ({self.state.hr:.1f} bpm)")
