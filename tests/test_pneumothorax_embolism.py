"""Quantitative physiology tests for tension pneumothorax, massive PE, and VAE."""

import pytest

from anasim.core.engine import SimulationEngine
from anasim.core.state import SimulationConfig
from anasim.patient.patient import Patient


def _advance(engine: SimulationEngine, seconds: float, dt: float = 0.5) -> None:
    engine.running = True
    for _ in range(round(seconds / dt)):
        engine.step(dt)


@pytest.mark.clinical
def test_tension_pneumothorax_ppv_vs_spontaneous_and_decompression_reverses_pea():
    """PPV drives rapid Ppeak+Pplat rise, hypoxemia, and obstructive PEA reversible by decompression
    (Nelson 2013; Roberts 2015), whereas spontaneous breathing and CPAP PEEP=0 progress slowly."""
    ppv = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    ppv.set_fgf(0.8, 1.2, 0.0)
    _advance(ppv, 30.0)
    ppeak0, pplat0, map0, pao2_0 = ppv.state.paw_peak, ppv.state.paw_plat, ppv.state.map, ppv.state.pao2
    ppv.start_pneumothorax(severity=1.25, initial_volume_l=0.60)

    # Early phase (~45 s): both Ppeak and Pplat climb together (narrow resistive gap) with shunt hypoxemia.
    _advance(ppv, 45.0)
    assert ppv.state.paw_peak > ppeak0 + 6.0
    assert ppv.state.paw_plat > pplat0 + 6.0
    assert (ppv.state.paw_peak - ppv.state.paw_plat) < 8.0
    assert ppv.state.pao2 < pao2_0 - 30.0
    assert ppv.state.map < map0 - 15.0

    # Sustained PPV progression drives obstructive PEA cardiac arrest (Nelson 2013).
    _advance(ppv, 140.0)
    assert ppv.state.pneumothorax_pressure >= 14.0
    assert ppv.state.co < 0.5
    assert ppv.state.cardiac_arrest

    # Immediate needle/chest decompression rapidly reverses PEA and restores perfusion (Nelson 2013).
    ppv.decompress_pneumothorax()
    _advance(ppv, 45.0)
    assert ppv.state.pneumothorax_pressure < 1.0
    assert not ppv.state.cardiac_arrest
    assert ppv.state.map >= 60.0 and ppv.state.co >= 3.0

    # Spontaneous breathing without PPV progresses much more slowly and avoids immediate arrest (Roberts 2015).
    spont = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=1))
    spont.start_pneumothorax(severity=1.25, initial_volume_l=0.60)
    _advance(spont, 185.0)
    assert not spont.state.cardiac_arrest
    assert spont.state.pneumothorax_pressure < 9.5
    assert spont.state.map >= 65.0

    # CPAP with PEEP=0 does not pump positive-pressure tension into an apneic pneumothorax.
    cpap = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    cpap.set_vent_settings(rr=0.0, vt=0.5, peep=0.0, ie="1:2", mode="CPAP")
    cpap.start_pneumothorax(severity=1.25, initial_volume_l=0.30)
    vol0 = cpap.pneumothorax.air_volume_l
    _advance(cpap, 30.0)
    assert cpap.pneumothorax.air_volume_l == pytest.approx(vol0, abs=0.02)


@pytest.mark.clinical
def test_pneumothorax_n2o_expansion_matches_eger_saidman_1965():
    """Closed intrapleural air under ~73% N2O doubles by ~10 min and approaches 1/(1-F_A_N2O) (Eger and Saidman 1965)."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1))
    engine.set_fgf(0.8, 0.0, 2.2)
    engine.set_vaporizer(1.2)
    engine.circuit.composition.fin2o = 0.73
    engine.circuit.composition.fio2 = 0.26
    engine.circuit.composition.fin2 = 0.01
    engine.pk_n2o.state.p_alv = 0.73
    engine.pk_n2o.state.p_vrg = 0.73
    engine.pneumothorax.set_closed_volume(0.30)

    _advance(engine, 600.0, dt=1.0)
    vol_10m = engine.pneumothorax.total_volume_l
    assert 0.50 <= vol_10m <= 0.72  # ~2x at 10 min (Eger and Saidman 1965)

    _advance(engine, 2100.0, dt=2.0)  # Total 45 min
    vol_45m = engine.pneumothorax.total_volume_l
    assert 0.85 <= vol_45m <= 1.15  # ~3x at 45 min

    # Turning N2O off shrinks the pleural cavity back toward baseline air volume.
    engine.set_fgf(8.0, 0.0, 0.0)
    _advance(engine, 1800.0, dt=2.0)
    assert engine.pneumothorax.total_volume_l < vol_45m - 0.35


@pytest.mark.clinical
def test_pneumothorax_pcv_tidal_volume_collapse():
    """Under PCV, rising pleural tension collapses delivered tidal volume (Nelson 2013)."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    engine.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="PCV", p_insp=14.0)
    _advance(engine, 30.0)
    vt0 = engine.state.vt
    engine.start_pneumothorax(severity=1.4, initial_volume_l=0.90)
    _advance(engine, 120.0)
    assert engine.state.vt < 0.35 * vt0


@pytest.mark.clinical
def test_massive_pe_deadspace_etco2_drop_pvr_heparin_and_thrombolysis():
    """Massive PE raises PVR, widens PaCO2-EtCO2 > 20 mmHg without bronchospastic capnogram slope
    (McIntyre and Sasahara 1971; Eriksson 1989), and responds to heparin and thrombolysis."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    engine.set_fgf(0.8, 1.2, 0.0)
    _advance(engine, 30.0)
    et0, map0, co0 = engine.state.etco2, engine.state.map, engine.state.co

    engine.start_pulmonary_embolism(0.62)
    _advance(engine, 90.0)
    gap = engine.state.pa_co2 - engine.state.etco2
    assert engine.state.etco2 < et0 - 12.0
    assert gap > 20.0  # Unconstrained parallel alveolar dead-space gradient (Eriksson 1989)
    assert engine.airway_status.capno_obstruction < 0.05  # Flat plateau, not bronchospasm
    assert engine.hemo.state.pvr > 3.5
    assert engine.state.co < 0.72 * co0
    assert engine.state.map < map0 - 18.0

    # Thrombolysis dissolves clot and restores pulmonary blood flow and EtCO2.
    engine.give_thrombolysis()
    _advance(engine, 240.0)
    assert engine.state.pe_obstruction < 0.25
    assert engine.state.etco2 > et0 - 8.0
    assert engine.state.map > map0 - 10.0

    # Therapeutic heparin drives slow endogenous lysis without reversing back toward baseline target.
    hep = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    ctrl = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    hep.start_pulmonary_embolism(0.55)
    ctrl.start_pulmonary_embolism(0.55)
    _advance(hep, 90.0)
    _advance(ctrl, 90.0)
    obs_peak = hep.state.pe_obstruction
    hep.give_drug_bolus("heparin", 7000.0)
    _advance(hep, 600.0, dt=1.0)
    _advance(ctrl, 600.0, dt=1.0)
    assert hep.state.pe_obstruction < obs_peak - 0.008
    assert hep.state.pe_obstruction < ctrl.state.pe_obstruction - 0.015

    # Early thrombolysis when obstruction is still below PE_RESIDUAL_AFTER_LYSIS never grows clot upward.
    early = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="tiva", rng_seed=1))
    early.start_pulmonary_embolism(0.55)
    early.embolism.pe_obstruction = 0.05
    early.give_thrombolysis()
    _advance(early, 30.0)
    assert early.state.pe_obstruction <= 0.055


@pytest.mark.clinical
def test_vae_n2o_expansion_o2_clearance_durant_and_cvc_aspiration():
    """VAE expands >2.5x under 74% N2O (Munson and Merrick 1966), clears faster on 100% O2,
    and responds to Durant positioning and CVC aspiration (Mirski 2007)."""
    o2_eng = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1))
    o2_eng.set_fgf(6.0, 0.0, 0.0)
    n2o_eng = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", maint_type="balanced", rng_seed=1))
    n2o_eng.set_fgf(0.8, 0.0, 2.2)
    n2o_eng.circuit.composition.fin2o = 0.74
    n2o_eng.circuit.composition.fio2 = 0.25
    n2o_eng.pk_n2o.state.p_alv = 0.74
    n2o_eng.pk_n2o.state.p_vrg = 0.74

    # Same 70 mL room-air bolus produces >2.5x effective volume and much worse shock/EtCO2 drop on 74% N2O.
    o2_eng.start_venous_air_embolism(rate_ml_min=0.0, initial_bolus_ml=70.0)
    o2_eng.stop_venous_air_embolism()
    n2o_eng.start_venous_air_embolism(rate_ml_min=0.0, initial_bolus_ml=70.0)
    n2o_eng.stop_venous_air_embolism()
    _advance(o2_eng, 45.0)
    _advance(n2o_eng, 45.0)
    assert n2o_eng.embolism.vae_n2o_factor > 2.5
    assert n2o_eng.state.etco2 < o2_eng.state.etco2 - 6.0
    assert n2o_eng.state.map < o2_eng.state.map - 10.0

    # Durant left-lateral position reduces effective RVOT obstruction; CVC aspiration removes ~45% air.
    obs_supine = n2o_eng.embolism.vae_obstruction
    n2o_eng.set_position("left_lateral")
    _advance(n2o_eng, 2.0)
    assert n2o_eng.embolism.vae_obstruction < obs_supine
    air_before = n2o_eng.state.vae_air_ml
    removed = n2o_eng.aspirate_vae_air()
    assert removed == pytest.approx(0.45 * air_before, rel=0.05)

    # 100% O2 washes out residual nitrogen markedly faster than room air (Mirski 2007).
    air_hi = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    air_lo = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", rng_seed=1))
    air_hi.set_fgf(8.0, 0.0, 0.0)
    air_lo.set_fgf(0.0, 8.0, 0.0)
    air_lo.circuit.composition.fio2 = 0.21
    air_lo.circuit.composition.fin2 = 0.79
    for eng in (air_hi, air_lo):
        eng.start_venous_air_embolism(rate_ml_min=0.0, initial_bolus_ml=80.0)
        eng.stop_venous_air_embolism()
        _advance(eng, 300.0)
    assert air_hi.state.vae_air_ml < 0.70 * air_lo.state.vae_air_ml


@pytest.mark.invariant
def test_pneumothorax_and_embolism_step_size_invariance():
    """Pneumothorax, PE, and VAE produce matching state at dt=0.1 s vs dt=0.5 s."""
    def run_pair(setup_fn):
        res = []
        for dt in (0.1, 0.5):
            e = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", dt=dt, rng_seed=1))
            setup_fn(e)
            _advance(e, 60.0, dt=dt)
            res.append((e.state.map, e.state.etco2, e.state.pneumothorax_pressure, e.state.pe_obstruction, e.state.vae_air_ml))
        return res

    for fn in (
        lambda e: e.start_pneumothorax(severity=0.9, initial_volume_l=0.4),
        lambda e: e.start_pulmonary_embolism(0.55),
        lambda e: e.start_venous_air_embolism(rate_ml_min=50.0, initial_bolus_ml=40.0),
    ):
        a, b = run_pair(fn)
        assert a[0] == pytest.approx(b[0], abs=2.0)
        assert a[1] == pytest.approx(b[1], abs=1.5)
        assert a[2] == pytest.approx(b[2], abs=0.5)
        assert a[3] == pytest.approx(b[3], abs=0.02)
        assert a[4] == pytest.approx(b[4], abs=1.5)
