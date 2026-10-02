from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anasim.patient.pd import hypnotic_equivalent, opioid_equivalent

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
    paw_peak: float
    paw_plat: float
    paw_mean: float
    peep: float
    vent_active: bool


def measured_ventilation(
    engine: "SimulationEngine",
    assisted: bool,
    assisted_rr: float,
    assisted_vt_l: float,
    spontaneous_rr: float,
    spontaneous_vt_l: float,
) -> tuple[float, float, float]:
    """Return (RR, exhaled VT mL, MV L/min) as the ventilator measures them.

    Assisted VT is the last completed machine breath. Gas that upper-airway
    obstruction stops from reaching the lungs leaks at the mask, so exhaled VT
    falls. Bronchospasm lowers alveolar ventilation in the respiratory model,
    not the exhaled volume. MV is RR x VT.
    """
    if assisted:
        rr = max(assisted_rr, spontaneous_rr)
        vt_l = assisted_vt_l * engine._airway_patency
    else:
        rr, vt_l = spontaneous_rr, spontaneous_vt_l
    return rr, vt_l * 1000.0, rr * vt_l


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
        fentanyl_ce=engine.pk_fentanyl.state.ce,
        fentanyl_cp=engine.pk_fentanyl.state.c1,
        midazolam_ce=engine.pk_midazolam.state.ce,
        etomidate_ce=engine.pk_etomidate.state.ce,
        ketamine_ce=engine.pk_ketamine.state.ce,
        opioid_ce=opioid_equivalent(engine.pk_remi.state.ce, engine.pk_fentanyl.state.ce),
        opioid_cp=opioid_equivalent(engine.pk_remi.state.c1, engine.pk_fentanyl.state.c1),
        hypnotic_ce=hypnotic_equivalent(
            engine.pk_prop.state.ce,
            engine.pk_etomidate.state.ce,
            engine.pk_midazolam.state.ce,
            engine.midazolam_c50,
        ),
        nore_ce=engine.pk_nore.state.ce,
        roc_ce=engine.tof_pd.ce,
        roc_cp=engine.pk_roc.state.c1,
        epi_ce=engine.pk_epi.state.ce,
        phenyl_ce=engine.pk_phenyl.state.ce,
        vaso_ce=engine.pk_vaso.state.ce,
        dobu_ce=engine.pk_dobu.state.ce,
        mil_ce=engine.pk_mil.state.ce,
        esmolol_ce=engine.pk_esmolol.state.ce,
        labetalol_ce=engine.pk_labetalol.state.ce,
        glyco_ce=engine.pk_glyco.state.ce,
    )


def sync_inspired_gas(engine: "SimulationEngine") -> tuple[float, float]:
    """Project inspired gas at the airway and return (Fi sevo, Fi N2O) as fractions."""
    composition = engine.circuit.composition
    if engine.state.airway_mode == AirwayType.NONE:
        fio2, fi_sevo, fi_n2o = 0.21, 0.0, 0.0
    else:
        fio2 = composition.fio2
        fi_sevo = composition.fi_agent if engine._volatile_enabled else 0.0
        fi_n2o = composition.fin2o
    set_state_float_fields(engine.state, fio2=fio2, fi_sevo=fi_sevo * 100.0, fi_n2o=fi_n2o * 100.0)
    return fi_sevo, fi_n2o


def sync_inhaled_agents(engine: "SimulationEngine") -> None:
    """Project end-tidal and brain MAC values for sevoflurane and nitrous oxide."""
    sevo, n2o = engine.pk_sevo.state, engine.pk_n2o.state
    set_state_float_fields(
        engine.state,
        et_sevo=sevo.p_alv * 100.0,
        et_n2o=n2o.p_alv * 100.0,
        mac_sevo=sevo.mac,
        mac_n2o=n2o.mac,
        mac=sevo.mac + n2o.mac,
        et_mac=sevo.p_alv * 100.0 / engine.pk_sevo.mac_age + n2o.p_alv * 100.0 / engine.pk_n2o.mac_age,
    )


def project_hemodynamics(engine: "SimulationEngine", hemo_state: Any) -> None:
    """Copy hemodynamic model state into the public snapshot."""
    state = engine.state
    map_val = hemo_state.map
    hr_val = hemo_state.hr
    sv_val = hemo_state.sv
    svr_val = hemo_state.svr
    co_val = hemo_state.co
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


def _project_respiratory_observables(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Copy respiratory fields shared by startup sync and runtime projection."""
    state = engine.state
    resp_state = snapshot.resp_state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    set_state_float_fields(
        state,
        rr=snapshot.rr_display,
        vt=snapshot.vt_display_ml,
        mv=snapshot.mv_display_l_min,
        va=resp_state.va,
        pa_co2=resp_state.pa_co2,
        alveolar_co2=resp_state.p_alveolar_co2,
        pao2=resp_state.p_arterial_o2,
        sao2=resp_state.sao2,
        spo2=resp_state.sao2,
        etco2=resp_state.etco2 if connected else 0.0,
        et_o2=resp_state.eto2 if connected else 0.0,
        pit=snapshot.pit_estimate,
        paw=snapshot.paw_display,
        flow=snapshot.flow_display,
        volume=snapshot.volume_display,
        paw_peak=snapshot.paw_peak,
        paw_plat=snapshot.paw_plat,
        paw_mean=snapshot.paw_mean,
        peep=snapshot.peep,
    )
    state.apnea = bool(resp_state.apnea)


def _current_respiratory_support(engine: "SimulationEngine") -> dict[str, Any]:
    """Return current assisted-ventilation settings used for state projection."""
    state = engine.state
    connected = state.airway_mode in (AirwayType.ETT, AirwayType.MASK)
    vent_active = connected and engine.vent.is_on
    bag_mask_active = engine.bag_mask_active and connected and not vent_active

    if vent_active:
        assisted_rr = engine.resp_mech.state.rr
        assisted_vt_l = engine.resp_mech.state.delivered_vt / 1000.0
        peep = engine.resp_mech.get_total_peep()
        mean_paw = max(engine.current_mean_paw, peep)
    elif bag_mask_active:
        assisted_rr = engine.resp_mech.state.rr
        assisted_vt_l = engine.resp_mech.state.delivered_vt / 1000.0
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
    support = _current_respiratory_support(engine)

    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=support["total_assisted_mv"],
        peep=support["peep"],
        mean_paw=support["mean_paw"],
        mech_rr=support["assisted_rr"],
        mech_vt_l=support["assisted_vt_l"],
        cardiac_output=hemo_state.co,
    )
    return engine.resp.step(0.0, **kwargs)


def build_snapshot_from_models(engine: "SimulationEngine", hemo_state: Any, resp_state: Any) -> PhysiologyStepState:
    """Build a projection snapshot from current model state without advancing runtime."""
    support = _current_respiratory_support(engine)
    spontaneous_rr = resp_state.rr
    spontaneous_vt_l = resp_state.vt / 1000.0
    assisted_rr = support["assisted_rr"]
    assisted_vt_l = support["assisted_vt_l"]

    paw_display = max(0.0, engine.resp_mech.set_peep) if support["vent_active"] else 0.0
    rr_display, vt_display_ml, mv_display_l_min = measured_ventilation(
        engine, support["assisted_active"], assisted_rr, max(0.0, assisted_vt_l), spontaneous_rr, spontaneous_vt_l
    )

    mech = engine.resp_mech.state
    assisted = support["assisted_active"]
    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        phase=mech.phase,
        pit_estimate=engine.hemo.pit_0,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=mv_display_l_min,
        paw_display=paw_display,
        flow_display=0.0,
        volume_display=0.0,
        paw_peak=mech.paw_peak if assisted else 0.0,
        paw_plat=mech.paw_plat if assisted else 0.0,
        paw_mean=mech.paw_mean if assisted else 0.0,
        peep=engine.resp_mech.set_peep if support["vent_active"] else 0.0,
        vent_active=support["vent_active"],
    )


def project_runtime_physiology(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    _project_respiratory_observables(engine, snapshot)
    engine._vent_active = snapshot.vent_active
    set_state_float_fields(
        state,
        oxygen_delivery_ratio=engine.hemo.compute_do2_ratio(
            max(0.0, state.sao2) / 100.0, max(0.0, state.pao2), state.co
        ),
    )


def sync_monitor_baselines(engine: "SimulationEngine") -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    state = engine.state
    bis_val = clamp(engine.bis.compute_bis(state.hypnotic_ce, state.opioid_ce, state.mac_sevo), 0.0, 100.0)
    tof_val = engine.tof_pd.compute_tof_from_ce(
        state.roc_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    loc_val = engine.loc_pd.compute_probability(
        state.hypnotic_ce,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
    )
    tol_val = engine.tol_pd.compute_probability(
        state.hypnotic_ce, state.opioid_ce, mac=state.mac, ce_ketamine=state.ketamine_ce
    )
    engine._tol_current = tol_val
    cardiac_sample = engine.cardiac_cycle.seed(state.hr, engine.hemo.state.rhythm_type)
    arterial_sample = engine.arterial_waveform.step(
        cardiac_sample,
        state.map,
        state.sv,
    )
    art_reading = engine.art_line.seed(arterial_sample)
    engine.airway_sensor.seed(state.paw, state.flow)
    set_state_float_fields(
        state,
        bis=bis_val,
        tof=tof_val,
        loc=loc_val,
        tol=tol_val,
        capno_co2=0.0,
        ecg_voltage=0.0,
        pleth_voltage=0.0,
        sbp=arterial_sample.systolic,
        dbp=arterial_sample.diastolic,
        art_pressure=art_reading.pressure,
        art_sbp=art_reading.systolic,
        art_dbp=art_reading.diastolic,
        art_map=art_reading.mean,
        display_hr=cardiac_sample.display_hr,
        display_bis=bis_val,
        display_etco2=state.etco2,
        display_spo2=state.spo2,
    )
    engine.bis.initialize(state.bis)
    engine.smooth_bis = state.bis


def sync_state_from_models(engine: "SimulationEngine") -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    set_state_float_fields(
        state,
        blood_volume=engine.hemo.blood_volume,
        temp_c=engine.patient.baseline_temp,
    )
    sync_pk_state(engine)
    sync_inspired_gas(engine)
    sync_inhaled_agents(engine)
    hemo_state = engine.hemo.state
    resp_state = snapshot_respiratory_state(engine, hemo_state)
    project_runtime_physiology(engine, build_snapshot_from_models(engine, hemo_state, resp_state))
    sync_monitor_baselines(engine)
