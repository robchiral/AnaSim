import pytest

from anasim.physiology.resp_mech import RespiratoryMechanics
from anasim.physiology.respiration import RespiratoryModel


def _run_mechanics(compliance, mode, seconds=12.0, dt=0.01, resistance=10.0, **settings):
    mech = RespiratoryMechanics(compliance=compliance, resistance=resistance)
    mech.set_settings(**{"rr": 12, "vt": 0.5, "peep": 5.0, "ie": "1:2", "mode": mode, **settings})
    for _ in range(round(seconds / dt)):
        mech.step(dt)
    return mech.state


def test_vcv_holds_volume_and_pcv_holds_pressure_when_compliance_falls():
    vcv = _run_mechanics(0.05, "VCV")
    vcv_stiff = _run_mechanics(0.025, "VCV")
    assert vcv.delivered_vt == pytest.approx(500.0, abs=50.0)
    assert vcv_stiff.paw_peak > vcv.paw_peak
    assert 5.0 < vcv.paw_mean < vcv.paw_peak

    pcv = _run_mechanics(0.05, "PCV", p_insp=15.0)
    pcv_stiff = _run_mechanics(0.025, "PCV", p_insp=15.0)
    assert pcv.paw_peak == pytest.approx(20.0, abs=2.0)
    assert 50.0 < pcv_stiff.delivered_vt < pcv.delivered_vt


def test_auto_peep_needs_a_short_expiratory_time():
    # RC = 0.75 s; RR 30 at 1:1 leaves 1 s to exhale.
    trapped = _run_mechanics(0.05, "VCV", seconds=10.0, resistance=15.0, rr=30, ie="1:1")
    normal = _run_mechanics(0.05, "VCV", seconds=10.0)
    assert trapped.auto_peep > 0.5
    assert normal.auto_peep < 1.0


def test_zero_rate_exhales_passively_to_peep():
    mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
    mech.set_settings(rr=0.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    mech.state.volume = 0.6

    result = mech.step(0.5)

    assert result.phase == "EXP"
    assert result.volume < 0.6
    assert result.flow <= 0.0
    assert result.paw == pytest.approx(5.0, abs=0.5)


def test_invalid_ie_ratio_is_rejected():
    mech = RespiratoryMechanics(compliance=0.05, resistance=10.0)
    with pytest.raises(ValueError, match="Invalid I:E ratio"):
        mech.set_settings(rr=12.0, vt=0.5, peep=5.0, ie="bad_format", mode="VCV")


def test_peep_improves_pao2(patient):
    pao2 = []
    for peep in (0.0, 10.0):
        resp = RespiratoryModel(patient)
        for _ in range(500):
            resp.step(0.01, ce_prop=0, ce_remi=0, mech_vent_mv=6.0, fio2=0.5, peep=peep)
        pao2.append(resp.state.p_arterial_o2)
    assert pao2[1] > pao2[0]


def test_bag_mask_ventilates_paralysis_only_through_an_airway(awake_engine, advance_time):
    engine = awake_engine
    engine.give_drug_bolus("Rocuronium", 0.8 * engine.patient.weight)
    engine.set_bag_mask_ventilation(True, rr=12.0, vt=0.6)
    advance_time(engine, 60.0, dt=0.25)
    assert engine.state.mv < 1.0
    apneic_pao2 = engine.state.pao2

    engine.set_airway_mode("Mask")
    advance_time(engine, 60.0, dt=0.25)
    assert engine.state.mv > 5.0
    assert engine.state.vt > 400.0
    assert engine.state.pao2 > apneic_pao2 + 20.0


def test_pressure_support_augments_spontaneous_breaths(awake_engine, advance_time):
    engine = awake_engine
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="CPAP", p_insp=0.0)
    advance_time(engine, 20.0, dt=0.1)
    vt_cpap = engine.state.vt
    assert vt_cpap > 200.0
    assert engine.resp_mech.state.paw_peak <= 10.0

    engine.set_vent_settings(rr=0.0, vt=0.0, peep=8.0, ie="1:2", mode="PSV", p_insp=10.0)
    advance_time(engine, 20.0, dt=0.1)
    assert engine.state.vt > vt_cpap + 50.0
