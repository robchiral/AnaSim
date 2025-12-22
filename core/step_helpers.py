"""
Step Helper Methods Mixin for SimulationEngine.

This module contains the private _step_* methods that implement the core
simulation update logic. They are extracted here for maintainability while
preserving the SimulationEngine API.
"""

from typing import TYPE_CHECKING
import numpy as np

from .state import AirwayType
from core.constants import TEMP_METABOLIC_COEFFICIENT
from core.utils import pao2_to_sao2, clamp
from monitors.capno import Capnograph

if TYPE_CHECKING:
    from .engine import SimulationEngine


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
    
    def _step_temperature(self: "SimulationEngine", dt: float):
        """
        Update patient temperature based on metabolic heat production and heat loss.
        """
        # 1. Metabolic Heat Production (W)
        # Basal ~ 1.16 W/kg. Anesthetics reduce this by ~30% (Propofol/Volatiles).
        
        # Calculate depth factor (0 to 1) using MAC and Propofol concentration
        depth_index = self.state.mac + (self.state.propofol_ce / 4.0)
        depth_factor = min(1.0, depth_index)
        
        metabolic_reduction = 0.3 * depth_factor
        current_production = self.heat_production_basal * (1.0 - metabolic_reduction)
        
        # 2. Heat Loss (W)
        # Simplified: Linear transfer coefficient * (T_core - T_ambient)
        t_ambient = 20.0
        
        # Base conductance tuned for ~80-100W loss at steady state (37C vs 20C)
        base_conductance = 3.0 
        
        # Vasodilation (Anesthesia) increases heat loss via redistribution and surface perfusion
        # Increases conductance by up to 50%
        anest_conductance_boost = 0.5 * depth_factor
        total_conductance = base_conductance * (1.0 + anest_conductance_boost) * (self.surface_area / 1.9)
        
        heat_loss = total_conductance * (self.state.temp_c - t_ambient)
        
        # Redistribution phase (Phase 1): Fast drop in first hour
        # "Bolus Effect": Rapid deepening creates a surge of redistribution heat loss.
        # This is modeled proportional to the rate of depth increase.
        
        d_depth = (depth_index - self._last_depth_index) / dt
        self._last_depth_index = depth_index
        
        if d_depth > 0:
            # Rapid deepening -> rapid vasodilation -> heat mixing (core temp drop)
            k_redist = self.K_REDIST
            redistribution_flux = k_redist * d_depth
            heat_loss += redistribution_flux
        
        # 3. Active Warming (Bair Hugger)
        # Convection model: Gain = K * (Target - Temp)
        # Empirical: High setting (~43C) provides ~50-60W transfer
        warming_input = 0.0
        if self.state.bair_hugger_target > 0:
             k_bair = 7.0 
             dt_warming = max(0.0, self.state.bair_hugger_target - self.state.temp_c)
             warming_input = k_bair * dt_warming
        
        # 4. Net Flux (J/s)
        net_heat_flux = current_production + warming_input - heat_loss
        
        # 5. Temperature Change
        # dT = Q * dt / (mass * specific_heat)
        # mass * specific heat = Heat Capacity (J/C)
        heat_capacity = self.patient.weight * self.specific_heat
        
        d_temp = (net_heat_flux * dt) / heat_capacity
        
        self.state.temp_c += d_temp
        
        # Clamp to physiologic bounds
        self.state.temp_c = clamp(self.state.temp_c, 25.0, 42.0)

    def _step_disturbances(self: "SimulationEngine", dt: float) -> tuple:
        """Calculate disturbances and handle events."""
        # 0. Disturbances
        if not getattr(self, "disturbance_active", False) or not self.disturbances:
            dist_vec = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            t_rel = max(0.0, self.state.time - getattr(self, "disturbance_start_time", 0.0))
            dist_vec = self.disturbances.compute_dist(t_rel)
        if hasattr(dist_vec, "as_tuple"):
            dist_vec = dist_vec.as_tuple()
        d_bis, d_map, d_co, d_svr, d_sv, d_hr = dist_vec

        # Attenuate stimulus with anesthetic depth (deeper = less response).
        depth_index = self.state.mac + (self.state.propofol_ce / 4.0)
        depth_factor = min(1.0, depth_index)
        stim_gain = max(0.3, 1.0 - 0.6 * depth_factor)
        d_bis *= stim_gain
        d_svr *= stim_gain
        d_sv *= stim_gain
        d_hr *= stim_gain
        
        # Event Logic
        if self.active_hemorrhage:
            # Bleed rate (mL/s). Default 500 mL/min = 8.33 mL/s
            rate_sec = self.hemorrhage_rate_ml_min / 60.0
            self.hemo.add_volume(-rate_sec * dt)
        
        # Infusions (fluids / blood) with realistic rates
        if self.pending_infusions:
            finished = []
            for infusion in self.pending_infusions:
                rate_sec = infusion['rate'] / 60.0
                amount_this_step = min(infusion['remaining'], rate_sec * dt)
                infusion['remaining'] -= amount_this_step
                if self.hemo and amount_this_step > 0:
                    self.hemo.add_volume(amount_this_step, hematocrit=infusion['hematocrit'])
                if infusion['remaining'] <= 1e-3:
                    finished.append(infusion)
            for infusion in finished:
                self.pending_infusions.remove(infusion)
            
        # Anaphylaxis (gradual onset/offset for realistic hemodynamic changes)
        if self.active_anaphylaxis:
            # Ramp up severity
            self.anaphylaxis_severity = min(1.0, self.anaphylaxis_severity + self.anaphylaxis_onset_rate * dt)
        else:
            # Decay severity when not active (simulates treatment effect)
            self.anaphylaxis_severity = max(0.0, self.anaphylaxis_severity - self.anaphylaxis_decay_rate * dt)
        
        # Apply anaphylaxis effects based on current severity
        if self.anaphylaxis_severity > 0.01:
            # Vasodilation (SVR drop) - proportional to severity
            # Full severity: d_tpr = -10 (massive shock)
            d_svr -= 10.0 * self.anaphylaxis_severity
            
            # Bronchospasm - proportional to severity
            # Normal: 10 cmH2O/L/s, Severe: 30 cmH2O/L/s
            target_resistance = 10.0 + 20.0 * self.anaphylaxis_severity
            self.resp_mech.resistance = target_resistance
        else:
            if self.resp_mech:
                self.resp_mech.resistance = 10.0
        
        # Repack into tuple
        dist_vec = (d_bis, d_map, d_co, d_svr, d_sv, d_hr)
        return dist_vec

    def _step_tci(self: "SimulationEngine", dt: float):
        """Update TCI controllers."""
        tci_targets = (
            ("tci_prop", "propofol_rate_mg_sec"),
            ("tci_remi", "remi_rate_ug_sec"),
            ("tci_nore", "nore_rate_ug_sec"),
            ("tci_epi", "epi_rate_ug_sec"),
            ("tci_phenyl", "phenyl_rate_ug_sec"),
            ("tci_roc", "roc_rate_mg_sec"),
        )
        for tci_attr, rate_attr in tci_targets:
            controller = getattr(self, tci_attr, None)
            if controller:
                rate = controller.step(controller.target, sim_time=self.state.time)
                setattr(self, rate_attr, rate)

    def _step_machine(self: "SimulationEngine", dt: float) -> float:
        """
        Update Machine State (Circuit -> Composition).
        Returns: fi_sevo
        """
        # Use prior-step VA from state (includes spontaneous + mechanical breathing)
        # Note: self.state.va is set in _step_physiology with the combined ventilation
        connected = (self.state.airway_mode != AirwayType.NONE)
        total_va = self.state.va if (connected and self.state.va > 0) else 0.0
        
        # Determine Inspired Concentrations (Fi) from Circuit
        if not connected:
            fi_sevo = 0.0
            # Patient breathing room air when not connected to circuit
            self.state.fio2 = 0.21
        else:
            fi_vapor_circuit = self.circuit.composition.fi_agent 
            fi_sevo = fi_vapor_circuit if self.active_agent == "Sevoflurane" else 0.0
            # Update state FiO2 from circuit
            self.state.fio2 = self.circuit.composition.fio2

        # Update State Fi
        self.state.fi_sevo = fi_sevo * 100.0
        
        # Use prior-step alveolar partial pressure to compute uptake
        p_alv_prev = self.pk_sevo.state.p_alv
        uptake_sevo = (fi_sevo - p_alv_prev) * total_va

        # O2 uptake tied to metabolic rate and temperature
        uptake_o2 = 0.25
        if self.resp:
            metabolic_factor = TEMP_METABOLIC_COEFFICIENT ** (37.0 - self.state.temp_c)
            vco2_ml_min = self.resp.vco2 * metabolic_factor
            uptake_o2 = (vco2_ml_min / 1000.0) / max(self.resp.rq, 0.1)
        
        # Circuit Mixing Step
        self.circuit.step(dt, uptake_o2, uptake_sevo)
        
        return fi_sevo

    def _step_pk(self: "SimulationEngine", dt: float, fi_sevo: float, co_curr: float):
        """Update Pharmacokinetics."""
        self._update_pk_hemodynamics(co_curr)

        # Volatile PK
        self.pk_sevo.step(dt, fi_sevo, self.state.va, co_curr, temp_c=self.state.temp_c)
        self.state.et_sevo = self.pk_sevo.state.p_alv * 100.0
        self.state.mac = self.pk_sevo.state.mac
        
        # IV Drug PK Steps
        self.pk_prop.step(dt, self.propofol_rate_mg_sec)
        self.pk_remi.step(dt, self.remi_rate_ug_sec) 
        self.pk_nore.step(dt, self.nore_rate_ug_sec, propofol_conc_ug_ml=self.pk_prop.state.c1)
        self.pk_roc.step(dt, self.roc_rate_mg_sec)
        self.pk_epi.step(dt, self.epi_rate_ug_sec)
        self.pk_phenyl.step(dt, self.phenyl_rate_ug_sec)
        
        # Sync PK state to public state (single point of truth for PK sync)
        self._sync_pk_state()

    def _sync_pk_state(self: "SimulationEngine"):
        """Synchronize PK concentrations from subsystem states to public state.
        
        This is the SINGLE PLACE where PK subsystem → engine state copying occurs.
        Called at the end of _step_pk after all PK models have been updated.
        
        All vasopressors now use effect-site concentration (.ce) for hemodynamic
        effects, providing realistic 1-2 minute delays after bolus administration.
        """
        self.state.propofol_ce = self.pk_prop.state.ce
        self.state.propofol_cp = self.pk_prop.state.c1
        self.state.remi_ce = self.pk_remi.state.ce
        self.state.remi_cp = self.pk_remi.state.c1
        self.state.nore_ce = self.pk_nore.state.ce  # Effect-site (ke0 modeled)
        self.state.roc_ce = self.pk_roc.state.ce
        self.state.roc_cp = self.pk_roc.state.c1
        self.state.epi_ce = self.pk_epi.state.ce    # Effect-site (ke0 modeled)
        self.state.phenyl_ce = self.pk_phenyl.state.ce  # Effect-site (ke0 modeled)

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

    def _step_physiology(self: "SimulationEngine", dt: float, dist_vec: tuple):
        """Update Hemo and Respiration with enhanced ventilator dynamics."""
        d_bis, d_map, d_co, d_svr, d_sv, d_hr = dist_vec
        
        connected = self.state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
        vent_active = connected and self.vent.is_on
        bag_mask_active = self.bag_mask_active and connected and not vent_active

        # 1. Mechanical Lung Step (assisted ventilation only)
        if vent_active:
            mech_state = self.resp_mech.step(dt)
            total_peep_effect = self.resp_mech.get_total_peep()
        elif bag_mask_active:
            # Save current vent settings, apply bag-mask settings, then restore
            saved_settings = (
                self.resp_mech.set_rr,
                self.resp_mech.set_vt,
                self.resp_mech.set_peep,
                self.resp_mech.mode,
                self.resp_mech.set_p_insp,
                self.resp_mech.insp_time_fraction,
            )
            self.resp_mech.set_settings(self.bag_mask_rr, self.bag_mask_vt, 0.0, ie="1:2", mode="VCV")
            mech_state = self.resp_mech.step(dt)
            total_peep_effect = self.resp_mech.get_total_peep()
            (
                self.resp_mech.set_rr,
                self.resp_mech.set_vt,
                self.resp_mech.set_peep,
                self.resp_mech.mode,
                self.resp_mech.set_p_insp,
                self.resp_mech.insp_time_fraction,
            ) = saved_settings
        else:
            # No assisted ventilation: decay any residual volume without applying PEEP
            saved_settings = (
                self.resp_mech.set_rr,
                self.resp_mech.set_peep,
            )
            self.resp_mech.set_rr = 0.0
            self.resp_mech.set_peep = 0.0
            mech_state = self.resp_mech.step(dt)
            total_peep_effect = 0.0
            mech_state.paw_mean = 0.0
            mech_state.auto_peep = 0.0
            self.resp_mech.set_rr, self.resp_mech.set_peep = saved_settings
        
        # 2. Calculate intrathoracic pressure from mean Paw
        # Use the proper mean Paw from mechanics, with smoothing
        alpha_paw = 0.05  # Faster tracking of mean Paw changes
        if mech_state.paw_mean > 0:
            self.current_mean_paw = (1 - alpha_paw) * self.current_mean_paw + alpha_paw * mech_state.paw_mean
        else:
            self.current_mean_paw = (1 - alpha_paw) * self.current_mean_paw + alpha_paw * mech_state.paw
        
        # Intrathoracic pressure estimate including auto-PEEP effect
        # Higher mean Paw and auto-PEEP both increase Pit, reducing venous return
        pit_estimate = -2.0 + 0.4 * (self.current_mean_paw - 5.0) + 0.3 * mech_state.auto_peep

        # 3. Mech Vent MV calculation (mechanical ventilator)
        mech_rr = self.resp_mech.set_rr if vent_active else 0.0
        delivered_vt_l = 0.0
        if vent_active:
            if mech_state.delivered_vt > 0:
                delivered_vt_l = mech_state.delivered_vt / 1000.0
            else:
                delivered_vt_l = self.resp_mech.set_vt
        mech_vent_mv = mech_rr * delivered_vt_l if vent_active else 0.0
        
        # 3b. Bag-mask ventilation (separate from mechanical vent)
        # Bag-mask provides PPV when active and airway is connected
        bag_mask_mv = 0.0
        bag_mask_rr_for_resp = 0.0
        bag_mask_vt_for_resp = 0.0
        if bag_mask_active:
            bag_mask_mv = self.bag_mask_rr * self.bag_mask_vt
            bag_mask_rr_for_resp = self.bag_mask_rr
            bag_mask_vt_for_resp = self.bag_mask_vt
        
        # Total assisted MV = mechanical vent + bag-mask (mutually exclusive in practice)
        total_assisted_mv = mech_vent_mv + bag_mask_mv
        
        # Use ventilator RR/Vt or bag-mask RR/Vt for respiration model
        # Bag-mask takes precedence if active (typically used when vent is off)
        assisted_rr_for_resp = bag_mask_rr_for_resp if bag_mask_rr_for_resp > 0 else mech_rr
        assisted_vt_for_resp = bag_mask_vt_for_resp if bag_mask_vt_for_resp > 0 else delivered_vt_l

        # 4. Respiration step with PEEP for oxygenation
        resp_state = self.resp.step(
            dt, 
            ce_prop=self.state.propofol_ce, 
            ce_remi=self.state.remi_ce, 
            mech_vent_mv=total_assisted_mv,  # Use combined assisted MV
            fio2=self.state.fio2,
            ce_roc=self.state.roc_ce, 
            et_sevo=self.state.et_sevo,
            mac_sevo=self.state.mac,
            peep=total_peep_effect,  # Pass total PEEP for oxygenation
            mean_paw=self.current_mean_paw,
            temp_c=self.state.temp_c,
            mech_rr=assisted_rr_for_resp,
            mech_vt_l=assisted_vt_for_resp,
            hb_g_dl=self.hemo.hb_conc,
            oxygen_delivery_ratio=self._do2_ratio
        )
        
        # 5. Determine effective RR and phase
        spont_rr = resp_state.rr
        spont_vt_l = resp_state.vt / 1000.0
        
        # Calculate synchronized MV (prevent double counting)
        if vent_active or bag_mask_active:
             eff_rr = max(assisted_rr_for_resp, spont_rr)
             eff_vt = max(assisted_vt_for_resp, spont_vt_l)
             total_patient_mv = eff_rr * eff_vt
        else:
             total_patient_mv = spont_rr * spont_vt_l
        
        # Determine assisted RR (from vent or bag-mask)
        assisted_rr = 0.0
        if vent_active:
            assisted_rr = mech_rr
        elif bag_mask_active:
            assisted_rr = self.bag_mask_rr
        
        phase = mech_state.phase
        # Handle phase for bag-mask or spontaneous breathing
        def _phase_from_rr(rr: float) -> str:
            if rr <= 0:
                return "EXP"
            cycle_time = 60.0 / rr
            insp_time = cycle_time / 3.0
            t_cycle = self.state.time % cycle_time
            return "INSP" if t_cycle < insp_time else "EXP"

        assisted_active = vent_active or bag_mask_active
        if not assisted_active:
            phase = _phase_from_rr(spont_rr)
        elif bag_mask_active and not vent_active:
            phase = _phase_from_rr(self.bag_mask_rr)
        
        # Set displayed RR: matches clinical capnography/flow-based RR detection
        # When vent is on: patient gets at least vent breaths, so display max(vent, spont)
        # When vent is off: display spontaneous RR or bag-mask RR
        if vent_active:
            # Patient receives at least the vent rate; spontaneous efforts may add more
            self.state.rr = max(assisted_rr, spont_rr)
        elif bag_mask_active:
            self.state.rr = max(self.bag_mask_rr, spont_rr)
        else:
            self.state.rr = spont_rr

        # 6. Hemodynamics with enhanced Pit coupling
        mac_sevo = self.state.mac
        hemo_state = self.hemo.step(
            dt,
            self.state.propofol_ce,
            self.state.remi_ce,
            self.state.nore_ce,
            pit=pit_estimate,
            paco2=resp_state.p_alveolar_co2,
            pao2=resp_state.p_arterial_o2,
            dist_hr=d_hr,
            dist_sv=d_sv,
            dist_svr=d_svr,
            mac=self.state.mac,
            mac_sevo=mac_sevo,
            ce_epi=self.state.epi_ce,
            ce_phenyl=self.state.phenyl_ce,
            temp_c=self.state.temp_c,
        )
        
        # 7. Update Ventilator Monitors
        self.vent.step(dt, mech_state)
                                    
        # Update Hemo State for Monitor Input
        self.state.map = hemo_state.map # Raw
        self.state.hr = hemo_state.hr   # Raw
        self.state.sv = hemo_state.sv
        self.state.svr = hemo_state.svr 
        self.state.co = hemo_state.co
        
        # New hemodynamic fields
        self.state.sbp = getattr(hemo_state, 'sbp', self.state.map + 20)
        self.state.dbp = getattr(hemo_state, 'dbp', self.state.map - 20)
        self.state.blood_volume = getattr(self.hemo, 'blood_volume', 5000.0)
        self.state.hb_g_dl = getattr(self.hemo, 'hb_conc', self.state.hb_g_dl)
        if hasattr(self.hemo, 'get_hematocrit'):
            self.state.hct = self.hemo.get_hematocrit()

        # Save pit for state
        self.state.pit = pit_estimate
        
        # Save Resp/Mech state for Monitors
        self.state.paco2 = resp_state.p_alveolar_co2
        self.state.pao2 = resp_state.p_arterial_o2
        
        # Use delivered Vt from mechanics (important for PCV mode)
        if vent_active and mech_state.delivered_vt > 0:
            self.state.vt = mech_state.delivered_vt
        elif vent_active:
            self.state.vt = self.resp_mech.set_vt * 1000.0
        else:
            self.state.vt = resp_state.vt
            
        self.state.mv = total_patient_mv
        self.state.va = resp_state.va
        self.state.apnea = resp_state.apnea
        self.state.paw = mech_state.paw
        self.state.flow = mech_state.flow
        self.state.volume = mech_state.volume
        self._vent_active = vent_active
        pao2_est = max(0.0, resp_state.p_arterial_o2)
        sao2_est = pao2_to_sao2(pao2_est)
        co_for_do2 = max(0.1, self.state.co)
        self._do2_ratio = self.hemo.compute_do2_ratio(sao2_est / 100.0, pao2_est, co_for_do2)
        self.state.oxygen_delivery_ratio = self._do2_ratio
        
        return hemo_state, resp_state, phase

    def _step_monitors(self: "SimulationEngine", dt: float, phase: str, hemo_state, resp_state, dist_vec):
        """Update Monitors and calculate smoothed state."""
        d_bis, d_map, d_co, d_svr, d_sv, d_hr = dist_vec
        
        # Convert remi rate for BIS model
        remi_rate_ug_kg_min = self.remi_rate_ug_sec * 60.0 / self.patient.weight

        bis_val = self.bis.step(dt, self.state.propofol_ce, self.state.remi_ce,
                                mac_sevo=self.pk_sevo.state.mac,
                                remi_rate_ug_kg_min=remi_rate_ug_kg_min)
                                
        # Capnography context is computed in monitors/capno to avoid duplication
        bag_mask_active = self.bag_mask_active and self.state.airway_mode != AirwayType.NONE and not self._vent_active
        vent_rr = self.resp_mech.set_rr if self._vent_active else (self.bag_mask_rr if bag_mask_active else 0.0)
        insp_fraction = self.resp_mech.insp_time_fraction
        if bag_mask_active and not self._vent_active:
            insp_fraction = 1.0 / 3.0
        capno_context = Capnograph.build_context(
            resp_state,
            vent_rr=vent_rr,
            insp_fraction=insp_fraction,
            vent_active=(self._vent_active or bag_mask_active)
        )

        if self.state.airway_mode == AirwayType.NONE:
            capno_val = 0.0
        elif self.state.rr == 0:
            # Apneic - no gas flow, no waveform
            capno_val = 0.0
            self.capno.state.co2 = 0.0  # Reset capnograph
        else:
            capno_val = self.capno.step(dt, phase, resp_state.p_alveolar_co2, 
                                        is_spontaneous=capno_context.is_spontaneous,
                                        curare_cleft=capno_context.curare_active,
                                        exp_duration=capno_context.exp_duration,
                                        effort_scale=capno_context.effort_scale)
        
        # Other PD - Neuromuscular (uses step_recovery for proper effect-site & sugammadex binding)
        tof_val = self.tof_pd.step_recovery(dt, self.state.roc_cp, mac_sevo=self.pk_sevo.state.mac)
        loc_val = self.loc_pd.compute_probability(self.state.propofol_ce, self.state.remi_ce)
        tol_val = self.tol_pd.compute_probability(self.state.propofol_ce, self.state.remi_ce)

        # ECG / SpO2 / NIBP
        self.state.ecg_voltage = self.ecg.step(dt, state_hr=hemo_state.hr)
        
        pao2 = max(0.0, resp_state.p_arterial_o2)
        sao2 = pao2_to_sao2(pao2)
        pleth, spo2_val = self.spo2_mon.step(dt, hr=hemo_state.hr, saturation=sao2)
        self.state.pleth_voltage = pleth
        
        if self.state.time >= self._next_nibp_time and not self.nibp.is_cycling:
            self.nibp.trigger()
            self._next_nibp_time = self.state.time + self.nibp.interval

        prev_ts = self.state.nibp_timestamp
        cuff_p = self.nibp.step(dt, self.state.time, hemo_state.map, true_sys=getattr(hemo_state, "sbp", None))
        
        self.state.nibp_is_cycling = self.nibp.is_cycling
        self.state.nibp_cuff_pressure = cuff_p
        
        if self.nibp.latest_reading.timestamp > 0.0 and self.nibp.latest_reading.timestamp != prev_ts:
            self.state.nibp_sys = self.nibp.latest_reading.systolic
            self.state.nibp_dia = self.nibp.latest_reading.diastolic
            self.state.nibp_map = self.nibp.latest_reading.map
            self.state.nibp_timestamp = self.nibp.latest_reading.timestamp

        # Smoothing & Final State Updates
        # Apply noise - generate all values in single call for efficiency
        noise_std = np.array([0.5, 0.5, 0.2])  # MAP, HR, BIS standard deviations
        noise = self.rng.normal(0, 1, 3) * noise_std
        
        raw_map = hemo_state.map + d_map + noise[0]
        raw_hr = hemo_state.hr + noise[1]
        raw_bis = bis_val + d_bis + noise[2]
        
        alpha = 0.1
        self.smooth_map = (1 - alpha) * self.smooth_map + alpha * raw_map
        self.smooth_hr = (1 - alpha) * self.smooth_hr + alpha * raw_hr
        self.smooth_bis = (1 - alpha) * self.smooth_bis + alpha * raw_bis
        
        self.state.map = self.smooth_map
        self.state.hr = self.smooth_hr
        self.state.bis = clamp(self.smooth_bis, 0.0, 100.0)
        self.state.capno_co2 = capno_val
        self.state.etco2 = resp_state.etco2 if self.state.airway_mode != AirwayType.NONE else 0.0
        self.state.tof = tof_val
        self.state.loc = loc_val
        self.state.tol = tol_val
        self.state.spo2 = spo2_val
        
        # Alarms
        monitor_vals = {
            'BIS': self.state.bis,
            'MAP': self.state.map,
            'HR': self.state.hr,
            'EtCO2': self.state.etco2,
            'SpO2': self.state.spo2
        }
        self.state.alarms = self.alarms.update(monitor_vals)

    def _check_patient_viability(self: "SimulationEngine", dt: float):
        """
        Check if patient vitals are compatible with life.
        Only runs if enabled in config.
        """
        if not self.config.enable_death_detector:
            return
            
        if self.state.is_dead:
            return
            
        # Thresholds
        MAP_CRITICAL_LOW = 20.0  # mmHg (Raised from 15 to ensure detection)
        HR_CRITICAL_LOW = 10.0   # bpm
        HR_CRITICAL_HIGH = 220.0 # bpm (matches HR clamp)
        
        # 1. Extreme Hypotension (Cardiac Arrest / Shock)
        if self.state.map < MAP_CRITICAL_LOW:
            self.time_hypotension += dt
        else:
            self.time_hypotension = max(0, self.time_hypotension - dt) # Decay accumulator
            
        # 2. Extreme Bradycardia / Asystole
        if self.state.hr < HR_CRITICAL_LOW:
            self.time_brady += dt
        else:
            self.time_brady = max(0, self.time_brady - dt)
            
        # 3. Extreme Tachycardia (VFib equivalent or massive crisis)
        if self.state.hr > HR_CRITICAL_HIGH:
            self.time_tachy += dt
        else:
             self.time_tachy = max(0, self.time_tachy - dt)
             
        # Trigger Death
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
