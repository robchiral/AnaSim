"""Tests for difficult laryngoscopy (ASA 2022 / Mort 2004b / Joffe 2010) and esophageal intubation."""

import json

import pytest

from anasim.core.clinical_actions import check_airway_removal
from anasim.core.engine import SimulationEngine
from anasim.core.state import AirwayType, SimulationConfig
from anasim.patient.patient import Patient
from anasim.web import WebSession


def _advance(engine: SimulationEngine, seconds: float, dt: float = 0.5) -> None:
    engine.running = True
    for _ in range(round(seconds / dt)):
        engine.step(dt)


def _induce_paralyzed_mask(patient: Patient | None = None) -> SimulationEngine:
    engine = SimulationEngine(patient or Patient(), SimulationConfig(mode="awake", rng_seed=1))
    engine.set_airway_mode("Mask")
    engine.set_fgf(10.0, 0.0, 0.0)
    engine.set_apl_valve(20.0)
    _advance(engine, 30.0)
    engine.give_drug_bolus("fentanyl", 100.0)
    engine.give_drug_bolus("propofol", 200.0)
    engine.give_drug_bolus("roc", 50.0)
    engine.set_drug_rate("propofol", 110.0)
    engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.5, p_max=20.0)
    _advance(engine, 75.0)
    return engine


@pytest.mark.clinical
def test_anterior_difficult_laryngoscopy_usa_or_pathway_and_trauma_escalation():
    """Anterior difficult DL fails direct laryngoscopy, improves mask seal with OPA/2-hand (Joffe 2010),
    succeeds with VL+bougie, and escalates airway edema on repeated failed attempts (ASA 2022; calibrated from Mort 2004b)."""
    engine = _induce_paralyzed_mask()
    engine.set_mask_seal(0.65)
    engine.set_difficult_airway("anterior_dl")
    _advance(engine, 10.0)

    # 1. Direct laryngoscopy fails with Cormack-Lehane Grade III view.
    fb = engine.attempt_intubation("dl")
    assert fb is not None and fb.blocked
    assert engine.state.airway_mode == AirwayType.MASK
    assert engine.state.laryngoscopy_view == "Grade III"
    assert engine.state.laryngoscopy_attempts == 1

    # 2. OPA + two-handed jaw thrust relieve obstruction and improve delivered tidal volume (Joffe 2010).
    obs_before = engine.state.airway_obstruction
    vt_before = engine.state.vt
    assert engine.set_oral_airway(True) is None
    engine.set_two_hand_mask(True)
    engine.call_airway_help()
    _advance(engine, 15.0)
    assert engine.state.airway_obstruction < obs_before
    assert engine.state.vt > vt_before
    assert engine.state.airway_help_called

    # 3. Repeated futile DL attempts (>2) escalate airway trauma (calibrated from Mort 2004b) until plain VL fails,
    # while VL + bougie still succeeds.
    fb2 = engine.attempt_intubation("dl")
    fb3 = engine.attempt_intubation("dl")
    assert fb2 is not None and fb2.blocked
    assert fb3 is not None and fb3.blocked
    assert engine.airway_crisis.trauma_obstruction >= 0.35
    fb_vl = engine.attempt_intubation("vl")
    assert fb_vl is not None and fb_vl.blocked and fb_vl.code == "repeated_laryngoscopy"

    fb_bougie = engine.attempt_intubation("vl_bougie")
    assert fb_bougie is None or not fb_bougie.blocked
    assert engine.state.airway_mode == AirwayType.ETT
    assert not engine.state.esophageal_intubation
    engine.set_vent_power(True)
    _advance(engine, 30.0)
    assert engine.state.etco2 >= 25.0 and engine.state.sao2 >= 94.0


@pytest.mark.clinical
def test_sga_rescue_and_cico_efona_pathway():
    """2nd-gen SGA rescues anterior difficult airway; severe CICO fails intubation and SGA and is rescued by eFONA."""
    # Anterior difficult airway rescued by SGA.
    sga_eng = _induce_paralyzed_mask()
    sga_eng.set_difficult_airway("anterior_dl")
    fb_dl = sga_eng.attempt_intubation("dl")
    assert fb_dl is not None and fb_dl.blocked
    fb_sga = sga_eng.attempt_sga()
    assert fb_sga is None or not fb_sga.blocked
    assert sga_eng.state.airway_mode == AirwayType.SGA
    sga_eng.set_vent_power(True)
    _advance(sga_eng, 25.0)
    assert sga_eng.state.etco2 >= 25.0 and sga_eng.state.sao2 >= 95.0

    # Severe CICO: all supraglottic/intubation attempts fail, eFONA bypasses glottis with +3 cmH2O/(L/s) R.
    cico = _induce_paralyzed_mask()
    cico.set_difficult_airway("severe_cico")
    _advance(cico, 15.0)
    for fb_fail in (
        cico.attempt_intubation("dl"),
        cico.attempt_intubation("vl_bougie"),
        cico.attempt_intubation("fiberoptic"),
        cico.attempt_sga(),
    ):
        assert fb_fail is not None and fb_fail.blocked
    assert cico.state.airway_obstruction >= 0.70

    r_base = cico._base_airway_resistance
    fb_fona = cico.perform_fona()
    assert fb_fona is None
    assert cico.state.airway_mode == AirwayType.FONA
    cico.set_vent_power(True)
    _advance(cico, 30.0)
    assert cico.state.airway_obstruction == 0.0
    assert cico.resp_mech.resistance == pytest.approx(r_base + cico.airway_tuning.fona_resistance_add, abs=0.2)
    assert cico.state.etco2 >= 25.0 and cico.state.sao2 >= 94.0


@pytest.mark.clinical
def test_esophageal_intubation_decaying_co2_low_vte_desaturation_and_puma_removal():
    """Esophageal intubation shows decaying gastric CO2 (<7 breaths >7.5 mmHg; Linko 1983, Chrimes 2022),
    very low returned VTe (Sum-Ping 1989), true apneic desaturation, and penalty-free tube removal."""
    engine = _induce_paralyzed_mask()
    engine.set_bag_mask_ventilation(False)
    engine.arm_esophageal_intubation(True)
    # Blind DL enters esophagus; prior mask ventilation seeds ~18 mmHg gastric CO2.
    engine.attempt_intubation("dl")
    assert engine.state.airway_mode == AirwayType.ETT
    assert engine.state.esophageal_intubation is True
    assert engine.airway_crisis.esophageal_co2_mmhg == pytest.approx(18.0, abs=0.5)

    engine.set_vent_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    paco2_start = engine.state.pa_co2

    # First ~2 breaths show attenuated, per-breath decaying gastric CO2 (no stale mask EtCO2 hold)
    # and no spurious low-FiO2 alarm on 100% O2; by 65 s CO2 is flatline 0, VTe is ~70 mL, alveolar
    # ventilation is 0, and preoxygenated SpO2 stays >98% while PaCO2 rises (Linko 1983; Sum-Ping 1989; Chrimes 2022).
    _advance(engine, 6.0)
    et_b1 = engine.state.display_etco2
    assert 0.5 < et_b1 < 18.0
    _advance(engine, 5.0)
    assert engine.state.display_etco2 < et_b1
    assert 0.5 < engine.state.etco2 < 12.0
    assert "FiO2" not in engine.state.alarms
    assert engine.state.fio2 > 0.85
    pao2_peak = engine.state.pao2
    _advance(engine, 54.0)
    assert engine.state.etco2 == 0.0
    assert engine.state.display_etco2 < 1.0
    assert engine.state.sao2 > 98.0
    assert 30.0 <= engine.state.vt <= 120.0  # Sum-Ping 1989 low returned VTe
    assert engine.state.va == 0.0
    assert engine.state.pa_co2 > paco2_start + 3.0
    # Sustained unrecognized esophageal intubation steadily drains alveolar O2 and eventually desaturates.
    _advance(engine, 240.0)
    assert engine.state.pao2 < pao2_peak - 200.0
    assert engine.state.pa_co2 > paco2_start + 15.0

    # Direct ETT -> ETT re-intubation over an esophageal ETT clears esophageal placement for all techniques.
    for tech in ("dl", "vl", "vl_bougie", "fiberoptic"):
        direct = _induce_paralyzed_mask()
        direct.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
        direct.attempt_intubation(tech)
        assert direct.state.airway_mode == AirwayType.ETT
        assert not direct.state.esophageal_intubation

    # PUMA default tube removal incurs zero safety warning even during apnea + paralysis.
    assert check_airway_removal(engine) is None
    engine.set_airway_mode("Mask")
    assert not engine.state.esophageal_intubation
    assert engine._extubation_stimulus_until == 0.0
    assert engine.laryngospasm_severity < 1e-6

    # Re-intubation under VL places tube in trachea and restores sustained EtCO2 and oxygenation.
    engine.attempt_intubation("vl")
    assert engine.state.airway_mode == AirwayType.ETT and not engine.state.esophageal_intubation
    _advance(engine, 30.0)
    assert engine.state.display_etco2 >= 25.0
    assert engine.state.vt >= 420.0
    assert engine.state.sao2 >= 96.0


@pytest.mark.clinical
def test_esophageal_without_prior_mask_ventilation_has_zero_gastric_co2():
    """Without prior mask ventilation, esophageal placement produces 0 mmHg CO2 from breath 1 (Linko 1983)."""
    engine = SimulationEngine(Patient(), SimulationConfig(mode="awake", rng_seed=2))
    engine.give_drug_bolus("propofol", 200.0)
    engine.give_drug_bolus("roc", 50.0)
    _advance(engine, 60.0)
    engine.arm_esophageal_intubation(True)
    engine.attempt_intubation("dl")
    assert engine.state.esophageal_intubation
    assert engine.airway_crisis.esophageal_co2_mmhg == 0.0


def test_failed_laryngoscopy_independent_accepts_sugammadex_wakeup():
    """ASA 2022 independent pathway accepts high-dose sugammadex reversal + wake-up as a valid resolution."""
    session = WebSession({"scenario_id": "failed_laryngoscopy", "practice_mode": "independent"})
    session.engine.config.dt = 0.5
    session.command("run", '{"running": true}')
    session.command("speed", '{"value": 10}')
    session.command("airway", json.dumps({"mode": "ETT", "technique": "dl"}))
    session.command("call_airway_help", "{}")
    session.command("oral_airway", '{"placed": true}')
    session.command("two_hand_mask", '{"active": true}')
    session.command("drug_rate", '{"key": "propofol", "rate": 0}')
    session.command("sugammadex", '{"mg_per_kg": 16}')
    # Ventilate while propofol washes out, then allow CO2 to trigger spontaneous breathing.
    for _ in range(180):
        session.advance(0.2)
        if session.engine.state.tof >= 95.0 and session.engine.state.propofol_ce < 2.2:
            break
    session.command("bag_mask", '{"active": false}')
    for _ in range(150):
        session.advance(0.2)
        if not session.engine.resp.state.apnea and session.engine.state.tof >= 95.0:
            break
    debrief = json.loads(session.command("case_finish"))["debrief"]
    unmet = [o["id"] for o in debrief["outcomes"] if not o["achieved"]]
    assert unmet == []


def test_esophageal_volume_continuity_mask_adjunct_cleanup_mild_coma_block_and_direct_vl():
    """Esophageal transitions preserve lung volume without phantom +500 mL jumps, Mask -> ETT
    clears mask adjuncts, mild hypothermia does not bypass awake intubation block, and direct VL
    re-intubation satisfies esophageal Step 1."""
    from anasim.core.clinical_actions import check_intubation, check_sga_placement
    from anasim.scenarios.airway import _check_esoph_removed

    # 1. Esophageal entry/exit does not inject phantom lung volume into resp_mech.
    eng = SimulationEngine(Patient(), SimulationConfig(mode="steady_state", dt=0.5, rng_seed=1))
    eng.start()
    _advance(eng, 10.0)
    eng.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
    _advance(eng, 15.0)
    vol_before_exit = eng.resp_mech.volume
    eng.set_esophageal_intubation(False)
    assert abs(eng.resp_mech.volume - vol_before_exit) < 0.05

    # 2. Switching off Mask clears OPA and two-hand jaw thrust.
    masked = _induce_paralyzed_mask()
    masked.set_oral_airway(True)
    masked.set_two_hand_mask(True)
    masked.set_airway_mode("ETT")
    assert not masked.airway_crisis.oral_airway and not masked.state.oral_airway
    assert not masked.airway_crisis.two_hand_mask and not masked.state.two_hand_mask

    # 3. Mild hypothermia (33.5 °C) in an awake unparalyzed patient still blocks intubation and SGA.
    cold = SimulationEngine(Patient(), SimulationConfig(mode="awake", dt=0.5, rng_seed=1))
    cold.state.temp_c = 33.5
    fb_ett = check_intubation(cold)
    fb_sga = check_sga_placement(cold)
    assert fb_ett is not None and fb_ett.blocked and fb_ett.code == "awake_intubation"
    assert fb_sga is not None and fb_sga.blocked and fb_sga.code == "awake_sga"

    # 4. Direct VL re-intubation over an esophageal tube satisfies Step 1 without deadlock,
    # and eFONA from an esophageal tube does not trigger tracheal extubation stimulus.
    direct = _induce_paralyzed_mask()
    direct.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
    assert not _check_esoph_removed(direct)[0]
    direct.attempt_intubation("vl")
    assert _check_esoph_removed(direct)[0]

    fona = _induce_paralyzed_mask()
    fona.set_esophageal_intubation(True, gastric_co2_mmhg=18.0)
    fona._extubation_stimulus_until = -1.0
    fona.perform_fona()
    assert fona._extubation_stimulus_until < 0.0
