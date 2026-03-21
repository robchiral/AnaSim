from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .state import AirwayType
from .utils import clamp

if TYPE_CHECKING:
    from .engine import SimulationEngine


@dataclass(slots=True)
class PhysiologyStepState:
    hemo_state: Any
    resp_state: Any
    phase: str
    pit_estimate: float
    rr_display: float
    vt_display_ml: float
    mv_display_l_min: float
    paw_display: float
    flow_display: float
    volume_display: float
    vent_active: bool


def sync_pk_state(engine: "SimulationEngine") -> None:
    """Synchronize PK concentrations from subsystem states to public state."""
    state = engine.state
    state.propofol_ce = engine.pk_prop.state.ce
    state.propofol_cp = engine.pk_prop.state.c1
    state.remi_ce = engine.pk_remi.state.ce
    state.remi_cp = engine.pk_remi.state.c1
    state.nore_ce = engine.pk_nore.state.ce
    state.roc_ce = engine.pk_roc.state.ce
    state.roc_cp = engine.pk_roc.state.c1
    state.epi_ce = engine.pk_epi.state.ce
    state.phenyl_ce = engine.pk_phenyl.state.ce
    if getattr(engine, "pk_vaso", None):
        state.vaso_ce = engine.pk_vaso.state.ce
    if getattr(engine, "pk_dobu", None):
        state.dobu_ce = engine.pk_dobu.state.ce
    if getattr(engine, "pk_mil", None):
        state.mil_ce = engine.pk_mil.state.ce


def sync_machine_state(engine: "SimulationEngine") -> None:
    """Copy volatile/circuit state into the public snapshot without advancing time."""
    state = engine.state
    connected = state.airway_mode != AirwayType.NONE
    composition = getattr(engine.circuit, "composition", None)
    if connected and composition is not None:
        fi_sevo = composition.fi_agent if getattr(engine, "_volatile_enabled", True) else 0.0
        fi_n2o = composition.fin2o
        state.fio2 = composition.fio2
    else:
        fi_sevo = 0.0
        fi_n2o = 0.0
        state.fio2 = 0.21

    state.fi_sevo = fi_sevo * 100.0
    state.fi_n2o = fi_n2o * 100.0
    state.et_sevo = engine.pk_sevo.state.p_alv * 100.0 if engine.pk_sevo else 0.0
    state.et_n2o = engine.pk_n2o.state.p_alv * 100.0 if getattr(engine, "pk_n2o", None) else 0.0
    state.mac_sevo = engine.pk_sevo.state.mac if engine.pk_sevo else 0.0
    state.mac_n2o = engine.pk_n2o.state.mac if getattr(engine, "pk_n2o", None) else 0.0
    state.mac = state.mac_sevo + state.mac_n2o


def project_hemodynamics(engine: "SimulationEngine", hemo_state: Any) -> None:
    """Copy hemodynamic model state into the public snapshot."""
    state = engine.state
    state.map = hemo_state.map
    state.hr = hemo_state.hr
    state.sv = hemo_state.sv
    state.svr = hemo_state.svr
    state.co = hemo_state.co
    state.sbp = getattr(hemo_state, "sbp", hemo_state.map + 20.0)
    state.dbp = getattr(hemo_state, "dbp", hemo_state.map - 20.0)
    state.blood_volume = getattr(engine.hemo, "blood_volume", state.blood_volume)
    state.hb_g_dl = getattr(engine.hemo, "hb_conc", state.hb_g_dl)
    state.hct = engine.hemo.get_hematocrit() if hasattr(engine.hemo, "get_hematocrit") else state.hct
    total_crystalloid = getattr(engine.hemo, "total_crystalloid_in_ml", 0.0)
    total_colloid = getattr(engine.hemo, "total_colloid_in_ml", 0.0)
    state.colloid_in_ml = total_colloid
    state.fluid_in_ml = total_crystalloid + total_colloid
    state.blood_in_ml = getattr(engine.hemo, "total_blood_in_ml", 0.0)
    state.urine_out_ml = getattr(engine.hemo, "total_urine_out_ml", 0.0)
    state.blood_out_ml = getattr(engine.hemo, "total_blood_out_ml", 0.0)
    state.net_fluid_ml = state.fluid_in_ml + state.blood_in_ml - state.urine_out_ml - state.blood_out_ml
    engine.smooth_map = hemo_state.map
    engine.smooth_hr = hemo_state.hr


def snapshot_respiratory_state(engine: "SimulationEngine", hemo_state: Any) -> Any:
    """Evaluate the respiratory model at the current subsystem state without advancing time."""
    if not engine.resp:
        return None

    state = engine.state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active

    if vent_active:
        mech_rr = engine.resp_mech.set_rr
        mech_vt_l = engine.resp_mech.set_vt
        total_assisted_mv = mech_rr * mech_vt_l
        peep = engine.resp_mech.get_total_peep()
        mean_paw = max(engine.current_mean_paw, peep)
    elif bag_mask_active:
        mech_rr = engine.bag_mask_rr
        mech_vt_l = engine.bag_mask_vt
        total_assisted_mv = mech_rr * mech_vt_l
        peep = 0.0
        mean_paw = 0.0
    else:
        mech_rr = 0.0
        mech_vt_l = 0.0
        total_assisted_mv = 0.0
        peep = 0.0
        mean_paw = 0.0

    if hemo_state and hemo_state.co > 0.0:
        cardiac_output = hemo_state.co
    else:
        cardiac_output = getattr(engine.hemo, "base_co_l_min", 5.0)

    return engine.resp.step(
        0.0,
        ce_prop=state.propofol_ce,
        ce_remi=state.remi_ce,
        mech_vent_mv=total_assisted_mv,
        fio2=state.fio2,
        ce_roc=state.roc_ce,
        et_sevo=state.et_sevo,
        mac_sevo=state.mac_sevo,
        peep=peep,
        mean_paw=mean_paw,
        temp_c=state.temp_c,
        mech_rr=mech_rr,
        mech_vt_l=mech_vt_l,
        airway_patency=engine._airway_patency,
        ventilation_efficiency=engine._ventilation_efficiency,
        vq_mismatch=engine._vq_mismatch,
        hb_g_dl=getattr(engine.hemo, "hb_conc", state.hb_g_dl),
        oxygen_delivery_ratio=getattr(engine, "_do2_ratio", 1.0),
        shiver_level=engine._shiver_level,
        cardiac_output=cardiac_output,
        metabolic_factor=max(0.5, getattr(engine, "_metabolic_factor", 1.0)),
    )


def project_respiration(engine: "SimulationEngine", resp_state: Any) -> None:
    """Copy respiratory model state into the public snapshot."""
    state = engine.state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active
    assisted_active = vent_active or bag_mask_active
    spontaneous_rr = resp_state.rr
    spontaneous_vt_l = resp_state.vt / 1000.0
    assisted_rr = 0.0
    assisted_vt_l = 0.0

    if vent_active:
        assisted_rr = engine.resp_mech.set_rr
        assisted_vt_l = engine.resp_mech.set_vt
        state.vt = max(0.0, assisted_vt_l) * 1000.0
        state.paw = max(0.0, engine.resp_mech.set_peep)
    elif bag_mask_active:
        assisted_rr = engine.bag_mask_rr
        assisted_vt_l = engine.bag_mask_vt
        state.vt = max(0.0, assisted_vt_l) * 1000.0
        state.paw = 0.0
    else:
        state.vt = resp_state.vt
        state.paw = 0.0

    if assisted_active:
        effective_rr = max(assisted_rr, spontaneous_rr)
        effective_vt_l = max(assisted_vt_l * engine._airway_patency, spontaneous_vt_l)
        state.rr = effective_rr
        state.mv = effective_rr * effective_vt_l
    else:
        state.rr = spontaneous_rr
        state.mv = spontaneous_rr * spontaneous_vt_l

    state.va = resp_state.va
    state.apnea = resp_state.apnea
    state.pa_co2 = resp_state.pa_co2
    state.alveolar_co2 = resp_state.p_alveolar_co2
    state.pao2 = resp_state.p_arterial_o2
    state.sao2 = resp_state.sao2
    state.spo2 = resp_state.sao2
    state.etco2 = resp_state.etco2 if connected else 0.0
    state.pit = getattr(engine.hemo, "pit_0", -2.0)
    state.flow = 0.0
    state.volume = 0.0


def project_runtime_physiology(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    state.pit = snapshot.pit_estimate
    state.pa_co2 = snapshot.resp_state.pa_co2
    state.alveolar_co2 = snapshot.resp_state.p_alveolar_co2
    state.pao2 = snapshot.resp_state.p_arterial_o2
    state.sao2 = snapshot.resp_state.sao2
    state.etco2 = snapshot.resp_state.etco2 if state.airway_mode != AirwayType.NONE else 0.0
    state.rr = snapshot.rr_display
    state.vt = snapshot.vt_display_ml
    state.mv = snapshot.mv_display_l_min
    state.va = snapshot.resp_state.va
    state.apnea = snapshot.resp_state.apnea
    state.paw = snapshot.paw_display
    state.flow = snapshot.flow_display
    state.volume = snapshot.volume_display
    engine._vent_active = snapshot.vent_active

    pao2_est = max(0.0, snapshot.resp_state.p_arterial_o2)
    sao2_est = state.sao2 if state.sao2 > 0.0 else 0.0
    co_for_do2 = max(0.1, state.co)
    engine._do2_ratio = engine.hemo.compute_do2_ratio(sao2_est / 100.0, pao2_est, co_for_do2)
    state.oxygen_delivery_ratio = engine._do2_ratio
    engine._sync_raw_vital_cache()


def sync_monitor_baselines(engine: "SimulationEngine") -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    state = engine.state
    state.bis = clamp(
        engine.bis.compute_bis(
            state.propofol_ce,
            state.remi_ce,
            u_volatile=state.mac_sevo,
        ),
        0.0,
        100.0,
    )
    if engine.bis:
        engine.bis.initialize(state.bis, dt=max(engine.config.dt, 1e-6))
    engine.smooth_bis = state.bis
    state.tof = engine.tof_pd.compute_tof_from_ce(
        state.roc_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    state.loc = engine.loc_pd.compute_probability(
        state.propofol_ce,
        state.remi_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    state.tol = engine.tol_pd.compute_probability(
        state.propofol_ce,
        state.remi_ce,
        mac=state.mac,
    )
    state.capno_co2 = 0.0
    state.ecg_voltage = 0.0
    state.pleth_voltage = 0.0
    engine._monitor_values["BIS"] = state.bis
    engine._monitor_values["MAP"] = state.map
    engine._monitor_values["HR"] = state.hr
    engine._monitor_values["EtCO2"] = state.etco2
    engine._monitor_values["SpO2"] = state.spo2


def sync_state_from_models(engine: "SimulationEngine") -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    state.blood_volume = getattr(engine.hemo, "blood_volume", state.blood_volume)
    state.temp_c = getattr(engine.patient, "baseline_temp", state.temp_c)
    sync_pk_state(engine)
    sync_machine_state(engine)
    hemo_state = engine.hemo.state if engine.hemo else None
    if hemo_state:
        project_hemodynamics(engine, hemo_state)
    resp_state = snapshot_respiratory_state(engine, hemo_state)
    if resp_state is not None:
        project_respiration(engine, resp_state)
    sync_monitor_baselines(engine)
    engine._sync_display_state_from_raw()
    engine._sync_raw_vital_cache()
