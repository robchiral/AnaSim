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


def set_state_float_fields(state, **values: float) -> None:
    """Assign numeric SimulationState fields as built-in floats."""
    for name, value in values.items():
        setattr(state, name, float(value))


def sync_pk_state(engine: "SimulationEngine") -> None:
    """Synchronize PK concentrations from subsystem states to public state."""
    state = engine.state
    set_state_float_fields(
        state,
        propofol_ce=engine.pk_prop.state.ce,
        propofol_cp=engine.pk_prop.state.c1,
        remi_ce=engine.pk_remi.state.ce,
        remi_cp=engine.pk_remi.state.c1,
        nore_ce=engine.pk_nore.state.ce,
        roc_ce=engine.pk_roc.state.ce,
        roc_cp=engine.pk_roc.state.c1,
        epi_ce=engine.pk_epi.state.ce,
        phenyl_ce=engine.pk_phenyl.state.ce,
        vaso_ce=engine.pk_vaso.state.ce,
        dobu_ce=engine.pk_dobu.state.ce,
        mil_ce=engine.pk_mil.state.ce,
    )


def sync_machine_state(engine: "SimulationEngine") -> None:
    """Copy volatile/circuit state into the public snapshot without advancing time."""
    state = engine.state
    connected = state.airway_mode != AirwayType.NONE
    composition = engine.circuit.composition
    if connected:
        fi_sevo = composition.fi_agent if engine._volatile_enabled else 0.0
        fi_n2o = composition.fin2o
        fio2 = composition.fio2
    else:
        fi_sevo = 0.0
        fi_n2o = 0.0
        fio2 = 0.21

    mac_sevo = engine.pk_sevo.state.mac
    mac_n2o = engine.pk_n2o.state.mac
    set_state_float_fields(
        state,
        fio2=fio2,
        fi_sevo=fi_sevo * 100.0,
        fi_n2o=fi_n2o * 100.0,
        et_sevo=engine.pk_sevo.state.p_alv * 100.0,
        et_n2o=engine.pk_n2o.state.p_alv * 100.0,
        mac_sevo=mac_sevo,
        mac_n2o=mac_n2o,
        mac=mac_sevo + mac_n2o,
    )


def project_hemodynamics(engine: "SimulationEngine", hemo_state: Any) -> None:
    """Copy hemodynamic model state into the public snapshot."""
    state = engine.state
    map_val = hemo_state.map
    hr_val = hemo_state.hr
    sv_val = hemo_state.sv
    svr_val = hemo_state.svr
    co_val = hemo_state.co
    sbp_val = hemo_state.sbp
    dbp_val = hemo_state.dbp
    hct_val = engine.hemo.get_hematocrit()
    total_crystalloid = engine.hemo.total_crystalloid_in_ml
    total_colloid = engine.hemo.total_colloid_in_ml
    blood_in_ml = engine.hemo.total_blood_in_ml
    urine_out_ml = engine.hemo.total_urine_out_ml
    blood_out_ml = engine.hemo.total_blood_out_ml
    fluid_in_ml = total_crystalloid + total_colloid
    set_state_float_fields(
        state,
        map=map_val,
        hr=hr_val,
        sv=sv_val,
        svr=svr_val,
        co=co_val,
        sbp=sbp_val,
        dbp=dbp_val,
        blood_volume=engine.hemo.blood_volume,
        hb_g_dl=engine.hemo.hb_conc,
        hct=hct_val,
        colloid_in_ml=total_colloid,
        fluid_in_ml=fluid_in_ml,
        blood_in_ml=blood_in_ml,
        urine_out_ml=urine_out_ml,
        blood_out_ml=blood_out_ml,
        net_fluid_ml=fluid_in_ml + blood_in_ml - urine_out_ml - blood_out_ml,
    )
    engine.smooth_map = float(map_val)
    engine.smooth_hr = float(hr_val)


def _project_respiratory_observables(
    engine: "SimulationEngine",
    resp_state: Any,
    *,
    rr_display: float,
    vt_display_ml: float,
    mv_display_l_min: float,
    paw_display: float,
    flow_display: float,
    volume_display: float,
    pit_display: float,
) -> None:
    """Copy respiratory fields shared by startup sync and runtime projection."""
    state = engine.state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    set_state_float_fields(
        state,
        rr=rr_display,
        vt=vt_display_ml,
        mv=mv_display_l_min,
        va=resp_state.va,
        pa_co2=resp_state.pa_co2,
        alveolar_co2=resp_state.p_alveolar_co2,
        pao2=resp_state.p_arterial_o2,
        sao2=resp_state.sao2,
        spo2=resp_state.sao2,
        etco2=resp_state.etco2 if connected else 0.0,
        pit=pit_display,
        paw=paw_display,
        flow=flow_display,
        volume=volume_display,
    )
    state.apnea = bool(resp_state.apnea)


def _current_respiratory_support(engine: "SimulationEngine") -> dict[str, Any]:
    """Return current assisted-ventilation settings used for state projection."""
    state = engine.state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active

    if vent_active:
        assisted_rr = engine.resp_mech.set_rr
        assisted_vt_l = engine.resp_mech.set_vt
        peep = engine.resp_mech.get_total_peep()
        mean_paw = max(engine.current_mean_paw, peep)
    elif bag_mask_active:
        assisted_rr = engine.bag_mask_rr
        assisted_vt_l = engine.bag_mask_vt
        peep = 0.0
        mean_paw = 0.0
    else:
        assisted_rr = 0.0
        assisted_vt_l = 0.0
        peep = 0.0
        mean_paw = 0.0

    return {
        "connected": connected,
        "vent_active": vent_active,
        "bag_mask_active": bag_mask_active,
        "assisted_active": vent_active or bag_mask_active,
        "assisted_rr": assisted_rr,
        "assisted_vt_l": assisted_vt_l,
        "total_assisted_mv": assisted_rr * assisted_vt_l,
        "peep": peep,
        "mean_paw": mean_paw,
    }


def snapshot_respiratory_state(engine: "SimulationEngine", hemo_state: Any) -> Any:
    """Evaluate the respiratory model at the current subsystem state without advancing time."""
    state = engine.state
    support = _current_respiratory_support(engine)

    if hemo_state.co > 0.0:
        cardiac_output = hemo_state.co
    else:
        cardiac_output = engine.hemo.base_co_l_min

    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=support["total_assisted_mv"],
        peep=support["peep"],
        mean_paw=support["mean_paw"],
        mech_rr=support["assisted_rr"],
        mech_vt_l=support["assisted_vt_l"],
        cardiac_output=cardiac_output,
        mac_sevo=state.mac_sevo,
    )
    return engine.resp.step(0.0, **kwargs)


def build_snapshot_from_models(engine: "SimulationEngine", hemo_state: Any, resp_state: Any) -> PhysiologyStepState:
    """Build a projection snapshot from current model state without advancing runtime."""
    support = _current_respiratory_support(engine)
    spontaneous_rr = resp_state.rr
    spontaneous_vt_l = resp_state.vt / 1000.0
    assisted_rr = support["assisted_rr"]
    assisted_vt_l = support["assisted_vt_l"]

    if support["vent_active"]:
        vt_display_ml = max(0.0, assisted_vt_l) * 1000.0
        paw_display = max(0.0, engine.resp_mech.set_peep)
    elif support["bag_mask_active"]:
        vt_display_ml = max(0.0, assisted_vt_l) * 1000.0
        paw_display = 0.0
    else:
        vt_display_ml = resp_state.vt
        paw_display = 0.0

    if support["assisted_active"]:
        effective_rr = max(assisted_rr, spontaneous_rr)
        effective_vt_l = max(assisted_vt_l * engine._airway_patency, spontaneous_vt_l)
        rr_display = effective_rr
        mv_display_l_min = effective_rr * effective_vt_l
    else:
        rr_display = spontaneous_rr
        mv_display_l_min = spontaneous_rr * spontaneous_vt_l

    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        phase=engine.resp_mech.state.phase,
        pit_estimate=engine.hemo.pit_0,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=mv_display_l_min,
        paw_display=paw_display,
        flow_display=0.0,
        volume_display=0.0,
        vent_active=support["vent_active"],
    )


def project_runtime_physiology(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    _project_respiratory_observables(
        engine,
        snapshot.resp_state,
        rr_display=snapshot.rr_display,
        vt_display_ml=snapshot.vt_display_ml,
        mv_display_l_min=snapshot.mv_display_l_min,
        paw_display=snapshot.paw_display,
        flow_display=snapshot.flow_display,
        volume_display=snapshot.volume_display,
        pit_display=snapshot.pit_estimate,
    )
    engine._vent_active = snapshot.vent_active

    pao2_est = max(0.0, snapshot.resp_state.p_arterial_o2)
    sao2_est = state.sao2 if state.sao2 > 0.0 else 0.0
    co_for_do2 = max(0.1, state.co)
    engine._do2_ratio = float(engine.hemo.compute_do2_ratio(sao2_est / 100.0, pao2_est, co_for_do2))
    set_state_float_fields(state, oxygen_delivery_ratio=engine._do2_ratio)
    engine._sync_raw_vital_cache()


def sync_monitor_baselines(engine: "SimulationEngine") -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    state = engine.state
    bis_val = clamp(
        engine.bis.compute_bis(
            state.propofol_ce,
            state.remi_ce,
            u_volatile=state.mac_sevo,
        ),
        0.0,
        100.0,
    )
    tof_val = engine.tof_pd.compute_tof_from_ce(
        state.roc_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    loc_val = engine.loc_pd.compute_probability(
        state.propofol_ce,
        state.remi_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    tol_val = engine.tol_pd.compute_probability(
        state.propofol_ce,
        state.remi_ce,
        mac=state.mac,
    )
    set_state_float_fields(
        state,
        bis=bis_val,
        tof=tof_val,
        loc=loc_val,
        tol=tol_val,
        capno_co2=0.0,
        ecg_voltage=0.0,
        pleth_voltage=0.0,
    )
    if engine.bis:
        engine.bis.initialize(state.bis, dt=max(engine.config.dt, 1e-6))
    engine.smooth_bis = state.bis
    engine._monitor_values["BIS"] = state.bis
    engine._monitor_values["MAP"] = state.map
    engine._monitor_values["HR"] = state.hr
    engine._monitor_values["EtCO2"] = state.etco2
    engine._monitor_values["SpO2"] = state.spo2


def sync_state_from_models(engine: "SimulationEngine") -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    set_state_float_fields(
        state,
        blood_volume=engine.hemo.blood_volume,
        temp_c=engine.patient.baseline_temp,
    )
    sync_pk_state(engine)
    sync_machine_state(engine)
    hemo_state = engine.hemo.state
    resp_state = snapshot_respiratory_state(engine, hemo_state)
    project_runtime_physiology(engine, build_snapshot_from_models(engine, hemo_state, resp_state))
    sync_monitor_baselines(engine)
    engine._sync_display_state_from_raw()
    engine._sync_raw_vital_cache()
