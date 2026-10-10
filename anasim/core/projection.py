from __future__ import annotations

import math
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
    compliance_dyn: float


def mask_leak(engine: "SimulationEngine") -> float:
    """Share of each assisted breath lost at a facemask when the upper airway is obstructed.

    The gas cannot reach the lungs, so it leaks around the mask. A tracheal tube
    does not leak; obstruction raises its resistance instead.
    """
    assisted = engine.vent.is_on or engine.bag_mask_active
    return 1.0 - engine._airway_patency if assisted and engine.state.airway_mode == AirwayType.MASK else 0.0


def circuit_ventilation(engine: "SimulationEngine") -> tuple[float, float]:
    """Return (RR, mean VT L) of all recent measured breaths that reach the lungs."""
    spirometry = engine.vent.monitors
    rr = spirometry.rr_total
    vt_l = spirometry.mv_exp / rr * (1.0 - mask_leak(engine)) if rr > 0.0 else 0.0
    return rr, vt_l


def measured_ventilation(
    engine: "SimulationEngine",
    connected: bool,
    spontaneous_rr: float,
    spontaneous_vt_l: float,
) -> tuple[float, float, float]:
    """Return (RR, exhaled VT mL, MV L/min) as the workstation measures them.

    Assisted VT is the last completed breath and MV the exhaled volume of
    recent breaths, less any mask leak. Bronchospasm lowers alveolar
    ventilation in the respiratory model, not the exhaled volume.
    """
    if connected:
        spirometry = engine.vent.monitors
        kept = 1.0 - mask_leak(engine)
        # Before the first completed breath, the respiratory model supplies
        # the baseline rate; no tidal volume has yet been measured.
        rr = spirometry.rr_total if engine.vent.has_measured_breath else spontaneous_rr
        return rr, spirometry.tv_exp * kept, spirometry.mv_exp * kept
    # Impedance counts muscle effort even when obstruction prevents airflow.
    return engine.resp.state.effort_rr, spontaneous_vt_l * 1000.0, spontaneous_rr * spontaneous_vt_l


# Projections run every step, so they assign public fields directly, as built-in floats.
def sync_pk_state(engine: "SimulationEngine") -> None:
    """Synchronize PK concentrations from subsystem states to public state."""
    state = engine.state
    state.propofol_ce = float(engine.pk_prop.state.ce)
    state.propofol_cp = float(engine.pk_prop.state.c1)
    state.remi_ce = float(engine.pk_remi.state.ce)
    state.remi_cp = float(engine.pk_remi.state.c1)
    state.fentanyl_ce = float(engine.pk_fentanyl.state.ce)
    state.fentanyl_cp = float(engine.pk_fentanyl.state.c1)
    state.midazolam_ce = float(engine.pk_midazolam.state.ce)
    state.etomidate_ce = float(engine.pk_etomidate.state.ce)
    state.ketamine_ce = float(engine.pk_ketamine.state.ce)
    state.lidocaine_ce = float(engine.pk_lidocaine.state.ce)
    state.opioid_ce = float(opioid_equivalent(engine.pk_remi.state.ce, engine.pk_fentanyl.state.ce))
    state.opioid_cp = float(opioid_equivalent(engine.pk_remi.state.c1, engine.pk_fentanyl.state.c1))
    state.hypnotic_ce = float(hypnotic_equivalent(
        engine.pk_prop.state.ce,
        engine.pk_etomidate.state.ce,
        engine.pk_midazolam.state.ce,
        engine.midazolam_c50,
    ))
    state.hypnotic_response_ce = float(hypnotic_equivalent(
        engine.pk_prop.state.ce_response, engine.pk_etomidate.state.ce,
        engine.pk_midazolam.state.ce, engine.midazolam_c50,
    ))
    state.nore_ce = float(engine.pk_nore.state.ce)
    state.roc_ce = float(engine.tof_pd.ce)
    state.roc_cp = float(engine.pk_roc.state.c1)
    state.epi_ce = float(engine.pk_epi.state.ce)
    state.phenyl_ce = float(engine.pk_phenyl.state.ce)
    state.vaso_ce = float(engine.pk_vaso.state.ce)
    state.dobu_ce = float(engine.pk_dobu.state.ce)
    state.mil_ce = float(engine.pk_mil.state.ce)
    state.esmolol_ce = float(engine.pk_esmolol.state.ce)
    state.labetalol_ce = float(engine.pk_labetalol.state.ce)
    state.glyco_ce = float(engine.pk_glyco.state.ce)


def sync_inspired_gas(engine: "SimulationEngine") -> tuple[float, float]:
    """Project inspired gas at the airway and return (Fi sevo, Fi N2O) as fractions."""
    state = engine.state
    composition = engine.circuit.composition
    if state.airway_mode == AirwayType.NONE:
        fio2, fi_sevo, fi_n2o = 0.21, 0.0, 0.0
    else:
        fio2 = composition.fio2
        fi_sevo = composition.fi_agent
        fi_n2o = composition.fin2o
    state.fio2 = float(fio2)
    state.fi_sevo = float(fi_sevo * 100.0)
    state.fi_n2o = float(fi_n2o * 100.0)
    return fi_sevo, fi_n2o


def sync_inhaled_agents(engine: "SimulationEngine") -> None:
    """Project end-tidal and brain MAC values for sevoflurane and nitrous oxide."""
    state = engine.state
    sevo, n2o = engine.pk_sevo.state, engine.pk_n2o.state
    state.et_sevo = float(sevo.p_alv * 100.0)
    state.et_n2o = float(n2o.p_alv * 100.0)
    state.mac_sevo = float(sevo.mac)
    state.mac_n2o = float(n2o.mac)
    state.mac = float(sevo.mac + n2o.mac)
    state.et_mac = float(sevo.p_alv * 100.0 / engine.pk_sevo.mac_age + n2o.p_alv * 100.0 / engine.pk_n2o.mac_age)


def project_hemodynamics(engine: "SimulationEngine", hemo_state: Any) -> None:
    """Copy hemodynamic model state into the public snapshot."""
    state = engine.state
    total_colloid = engine.hemo.total_colloid_in_ml
    blood_in_ml = engine.hemo.total_blood_in_ml
    urine_out_ml = engine.hemo.total_urine_out_ml
    blood_out_ml = engine.hemo.total_blood_out_ml
    fluid_in_ml = engine.hemo.total_crystalloid_in_ml + total_colloid
    state.map = float(hemo_state.map)
    state.hr = float(hemo_state.hr)
    state.sv = float(hemo_state.sv)
    state.svr = float(hemo_state.svr)
    state.co = float(hemo_state.co)
    state.blood_volume = float(engine.hemo.blood_volume)
    state.hb_g_dl = float(engine.hemo.hb_conc)
    state.hct = float(engine.hemo.get_hematocrit())
    state.colloid_in_ml = float(total_colloid)
    state.fluid_in_ml = float(fluid_in_ml)
    state.blood_in_ml = float(blood_in_ml)
    state.urine_out_ml = float(urine_out_ml)
    state.blood_out_ml = float(blood_out_ml)
    state.net_fluid_ml = float(fluid_in_ml + blood_in_ml - urine_out_ml - blood_out_ml)
    state.lap = float(hemo_state.lap)
    state.lung_water = float(engine.hemo.lung_water_ml_kg)


def _project_respiratory_observables(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Copy respiratory fields shared by startup sync and runtime projection."""
    state = engine.state
    resp_state = snapshot.resp_state
    connected = state.airway_mode != AirwayType.NONE
    state.rr = float(snapshot.rr_display)
    state.vt = float(snapshot.vt_display_ml)
    state.mv = float(snapshot.mv_display_l_min)
    state.va = float(resp_state.va)
    state.pa_co2 = float(resp_state.pa_co2)
    state.alveolar_co2 = float(resp_state.p_alveolar_co2)
    state.pao2 = float(resp_state.p_arterial_o2)
    state.sao2 = float(resp_state.sao2)
    state.spo2 = float(resp_state.sao2)
    state.etco2 = float(resp_state.etco2 if connected else 0.0)
    state.et_o2 = float(resp_state.eto2 if connected else 0.0)
    state.pit = float(snapshot.pit_estimate)
    state.paw = float(snapshot.paw_display)
    state.flow = float(snapshot.flow_display)
    state.volume = float(snapshot.volume_display)
    state.paw_peak = float(snapshot.paw_peak)
    state.paw_plat = float(snapshot.paw_plat)
    state.paw_mean = float(snapshot.paw_mean)
    state.peep = float(snapshot.peep)
    state.compliance_dyn = float(snapshot.compliance_dyn)
    state.apnea = bool(resp_state.apnea)


def snapshot_respiratory_state(engine: "SimulationEngine", hemo_state: Any) -> Any:
    """Evaluate the respiratory model at the current subsystem state without advancing time."""
    connected = engine.state.airway_mode != AirwayType.NONE
    assisted_rr, assisted_vt_l = circuit_ventilation(engine) if connected else (0.0, 0.0)

    kwargs = engine.get_resp_step_kwargs(
        total_assisted_mv=assisted_rr * assisted_vt_l,
        mech_rr=assisted_rr,
        mech_vt_l=assisted_vt_l,
        cardiac_output=hemo_state.co,
    )
    return engine.resp.step(0.0, **kwargs)


def build_snapshot_from_models(
    engine: "SimulationEngine", hemo_state: Any, resp_state: Any, *, pit_estimate: float,
) -> PhysiologyStepState:
    """Build the respiratory projection shared by initialization and runtime."""
    connected = engine.state.airway_mode != AirwayType.NONE
    assisted = connected and (engine.vent.is_on or engine.bag_mask_active)
    rr_display, vt_display_ml, mv_display_l_min = measured_ventilation(
        engine, connected, resp_state.rr, resp_state.vt / 1000.0
    )
    vent = engine.vent
    spirometry = vent.monitors
    return PhysiologyStepState(
        hemo_state=hemo_state,
        resp_state=resp_state,
        pit_estimate=pit_estimate,
        rr_display=rr_display,
        vt_display_ml=vt_display_ml,
        mv_display_l_min=mv_display_l_min,
        paw_display=vent.paw if connected else 0.0,
        flow_display=vent.flow if connected else 0.0,
        volume_display=vent.volume if connected else 0.0,
        paw_peak=spirometry.paw_peak if connected else 0.0,
        paw_plat=spirometry.paw_plat if assisted else math.nan,
        paw_mean=spirometry.paw_mean if connected else 0.0,
        peep=spirometry.peep if assisted else 0.0,
        compliance_dyn=spirometry.compliance_dyn if assisted else math.nan,
    )


def project_runtime_physiology(engine: "SimulationEngine", snapshot: PhysiologyStepState) -> None:
    """Project a runtime physiology step back into the public SimulationState."""
    state = engine.state
    project_hemodynamics(engine, snapshot.hemo_state)
    _project_respiratory_observables(engine, snapshot)
    state.oxygen_delivery_ratio = float(engine.hemo.compute_do2_ratio(
        max(0.0, state.sao2) / 100.0, max(0.0, state.pao2), state.co
    ))


def sync_monitor_baselines(engine: "SimulationEngine") -> None:
    """Derive monitor baselines from the current physiologic snapshot."""
    state = engine.state
    bis_val = clamp(engine.bis.compute_bis(state.hypnotic_ce, state.mac_sevo), 0.0, 100.0)
    tof_val = engine.tof_pd.compute_tof_from_ce(
        state.roc_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
    )
    loc_val = engine.response.loss_of_response(
        state.hypnotic_response_ce,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
    )
    tol_val = engine.response.tolerance(
        state.hypnotic_response_ce,
        state.opioid_ce,
        mac_sevo=state.mac_sevo,
        mac_n2o=state.mac_n2o,
        ce_ketamine=state.ketamine_ce,
        ce_lidocaine=state.lidocaine_ce,
    )
    engine._tol_current = tol_val
    cardiac_sample = engine.cardiac_cycle.seed(state.hr, engine.hemo.state.rhythm_type)
    arterial_sample = engine.arterial_waveform.step(
        cardiac_sample,
        state.map,
        state.sv,
    )
    art_reading = engine.art_line.seed(arterial_sample)
    engine.airway_sensor.seed(state.paw, state.flow, state.volume)
    state.bis = float(bis_val)
    state.tof = float(tof_val)
    state.loc = float(loc_val)
    state.tol = float(tol_val)
    state.capno_co2 = state.ecg_voltage = state.pleth_voltage = 0.0
    state.sbp = float(arterial_sample.systolic)
    state.dbp = float(arterial_sample.diastolic)
    state.art_pressure = float(art_reading.pressure)
    state.art_sbp = float(art_reading.systolic)
    state.art_dbp = float(art_reading.diastolic)
    state.art_map = float(art_reading.mean)
    state.display_hr = float(cardiac_sample.display_hr)
    state.display_bis = float(bis_val)
    state.display_etco2 = float(state.etco2)
    state.display_spo2 = float(state.spo2)
    engine.bis.initialize(state.bis)
    engine.smooth_bis = state.bis


def sync_state_from_models(engine: "SimulationEngine") -> None:
    """Derive the public SimulationState from current subsystem state."""
    state = engine.state
    state.temp_c = float(engine.patient.baseline_temp)
    sync_pk_state(engine)
    sync_inspired_gas(engine)
    sync_inhaled_agents(engine)
    hemo_state = engine.hemo.state
    resp_state = snapshot_respiratory_state(engine, hemo_state)
    project_runtime_physiology(engine, build_snapshot_from_models(
        engine, hemo_state, resp_state, pit_estimate=engine.hemo.config.pit_0,
    ))
    sync_monitor_baselines(engine)
