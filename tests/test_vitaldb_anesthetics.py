"""VitalDB cohort rules, preop-lab mapping, syringe resets, CV splits, and replay transitions."""

import numpy as np
import pytest

from anasim.monitors.capno import Capnograph
from validation.vitaldb import vitaldb_anesthetics


@pytest.fixture
def comparison():
    return vitaldb_anesthetics


@pytest.fixture
def case():
    return dict(age="40", weight="70", height="170", sex="M", opstart="600", opend="1500")


def test_patient_from_case_accepts_expanded_domain_and_wires_preop_labs(comparison, case):
    # Obesity and older adults within AnaSim's domain are accepted.
    obese = comparison.patient_for({**case, "weight": "110", "height": "190"})
    assert obese.weight == pytest.approx(110.0)
    obese2 = comparison.patient_for({**case, "weight": "100", "height": "170", "age": "82"})
    assert obese2.age == pytest.approx(82.0)

    # Valid preop Hb and creatinine map to baseline_hb and Cockcroft-Gault CrCl.
    with_labs = comparison.patient_for({**case, "preop_hb": "9.4", "preop_cr": "2.4"})
    assert with_labs.baseline_hb == pytest.approx(9.4)
    assert with_labs.creatinine_clearance_ml_min is not None
    assert with_labs.creatinine_clearance_ml_min < 50.0
    assert with_labs.renal_function < 0.5

    # Implausible creatinine outliers are ignored rather than crashing.
    from validation.vitaldb.vitaldb_data import has_plausible_creatinine

    assert has_plausible_creatinine({**case, "preop_cr": "2.4"})
    assert not has_plausible_creatinine({**case, "preop_cr": "0.05"})
    bad_cr = comparison.patient_for({**case, "preop_cr": "0.05"})
    assert bad_cr.creatinine_clearance_ml_min is None

    # Out-of-domain demographics or hemoglobin are rejected.
    with pytest.raises(comparison.Excluded, match="simulator patient domain"):
        comparison.patient_for({**case, "age": "90"})
    with pytest.raises(comparison.Excluded, match="preop hemoglobin"):
        comparison.patient_for({**case, "preop_hb": "4.5"})


def test_kfold_splits_preserve_subject_groups_and_stratification():
    from validation.vitaldb.vitaldb_data import kfold_splits

    items = [
        {"id": i, "subjectid": i // 2, "stratum": (i // 2) % 3}
        for i in range(60)
    ]
    splits = kfold_splits(
        items,
        n_folds=5,
        seed=42,
        group_key=lambda x: x["subjectid"],
        strata=lambda x: x["stratum"],
    )
    assert len(splits) == 5
    seen_val: list[int] = []
    for train_idx, val_idx in splits:
        train_subjs = {items[i]["subjectid"] for i in train_idx}
        val_subjs = {items[i]["subjectid"] for i in val_idx}
        assert train_subjs.isdisjoint(val_subjs)
        assert len(train_idx) + len(val_idx) == len(items)
        seen_val.extend(val_idx)
    assert sorted(seen_val) == list(range(len(items)))


def test_cv_hemodynamic_grid_selects_best_override_out_of_fold():
    from validation.vitaldb.vitaldb_replay import cv_hemodynamic_grid

    def make_rows(offset: float):
        rows = []
        for i in range(30):
            # 3 readings per patient with small patient-level jitter around offset.
            err = offset + ((i % 5) - 2) * 0.4
            rows.append({
                "caseid": i,
                "subjectid": i,
                "age": 35.0 + (i % 4) * 12.0,
                "treated": bool(i % 2),
                "hypertensive": bool(i % 3 == 0),
                "primary": {
                    "stopped": False,
                    "stats": [[3, 3.0 * err, 3.0 * (err**2 + 1.0)], [0, 0.0, 0.0]],
                },
            })
        return rows

    res = cv_hemodynamic_grid(
        [
            ({"scale": 0.8}, make_rows(-6.0)),
            ({"scale": 1.0}, make_rows(0.2)),
            ({"scale": 1.2}, make_rows(5.5)),
        ],
        n_folds=5,
        seed=7,
    )
    assert res["n_cases"] == 30
    assert res["best_full"]["override"] == {"scale": 1.0}
    assert all(fc["override"] == {"scale": 1.0} for fc in res["fold_choices"])
    assert abs(res["oof_mean"] - 0.2) < 0.5
    from validation.vitaldb.vitaldb_replay import cv_grid_lines, cv_stability_lines

    grid_md = "\n".join(cv_grid_lines(res))
    assert "Cross-validated parameter grid" in grid_md
    assert '"scale": 1.0' in grid_md
    assert cv_grid_lines(None) == []
    assert cv_stability_lines(None) == []


def test_ventilation_capnograph_helper_uses_pbw_and_avoids_class_mutation():
    from validation.vitaldb.vitaldb_ventilation import _make_capnograph

    orig_tau = Capnograph.ANALYZER_TAU
    orig_vol = Capnograph.PHASE_II_VOLUME
    capno = _make_capnograph(0.15, {"ANALYZER_TAU": 0.09, "PHASE_II_VOLUME": 0.20})
    assert capno.ANALYZER_TAU == pytest.approx(0.09)
    assert capno.PHASE_II_VOLUME == pytest.approx(0.20)
    assert capno._col_len == pytest.approx(0.15 - 0.5 * 0.20)
    assert Capnograph.ANALYZER_TAU == orig_tau
    assert Capnograph.PHASE_II_VOLUME == orig_vol


def test_sevo_and_tiva_expanded_eligibility():
    from validation.vitaldb import vitaldb_sevo, vitaldb_tiva

    assert "Primus/PPLAT_MBAR" not in vitaldb_sevo.REQUIRED
    base_case = {
        "weight": "115",
        "intraop_ppf": "180",
        "intraop_ftn": "100",
        "intraop_mdz": "0",
        "intraop_epi": "0",
        "intraop_rbc": "0",
        "intraop_ffp": "0",
        "intraop_ebl": "200",
        "asa": "3",
        "preop_htn": "1",
    }
    assert vitaldb_sevo.eligible(base_case, {"Primus/INSP_SEVO": "1"})
    tiva_case = {**base_case, "intraop_ppf": "0", "intraop_ftn": "0"}
    assert vitaldb_tiva.eligible(tiva_case, {"Orchestra/RFTN20_RATE": "1"})


def test_hemodynamic_replay_injects_inspired_sevoflurane():
    from validation.vitaldb.vitaldb_data import patient_from_case
    from validation.vitaldb.vitaldb_replay import run_hemodynamic_replay

    case = dict(
        age="45", weight="70", height="170", sex="M",
        anestart="0", aneend="3600", intraop_crystalloid="0", intraop_colloid="0",
    )
    patient = patient_from_case(case)
    grid = np.arange(0.0, 301.0, 1.0)
    propofol_mg_s = np.where(grid[:-1] < 20.0, 7.0, 0.0)
    remi_ug_s = np.zeros(len(grid) - 1)
    inspired = np.where(grid >= 90.0, 2.5, 0.0)
    out = run_hemodynamic_replay(
        case, patient, grid, onset=0.0, events=[],
        propofol_mg_s=propofol_mg_s, remi_ug_s=remi_ug_s, inspired_sevo=inspired,
    )
    assert out[-1, 2] > 1.2  # et_sevo (%) rises cleanly under injected FiSevo 2.5%


def tiva_recording(comparison):
    time = np.arange(2201, dtype=float)
    rate = np.where((time >= 60) & (time < 180), 500.0,
                    np.where((time >= 180) & (time < 1500), 12.0, 0.0))
    volume = np.r_[0.0, np.cumsum(rate[:-1] / 3600.0)]
    series = {name: (time, np.zeros(len(time))) for name in comparison.TIVA_TRACKS}
    series["Orchestra/PPF20_RATE"] = time, rate
    series["Orchestra/PPF20_VOL"] = time, volume
    series["Orchestra/RFTN20_RATE"] = time, np.full(len(time), 10.0)
    series["BIS/BIS"] = time, np.where(time < 60, 90.0, 40.0)
    series["BIS/SQI"] = time, np.where((time >= 800) & (time < 850), 0.0, 100.0)
    return series


def test_propofol_replay_preserves_delivered_dose_and_handles_syringe_reset(comparison, case, monkeypatch):
    series = tiva_recording(comparison)
    monkeypatch.setattr(comparison, "numeric_track", series.__getitem__)
    exact = comparison.evaluate_tiva(case, {name: name for name in series}, 1.0)
    assert exact["dose_mg"] == pytest.approx((500 * 120 + 12 * 1320) * 20 / 3600)
    assert exact["volume_disagreement_ml"] == pytest.approx(0.0, abs=1e-9)

    # Simulate a syringe change at t=900 s resetting the cumulative volume counter to 0.
    time, vol = series["Orchestra/PPF20_VOL"]
    reset_vol = np.where(time >= 900, vol - vol[900], vol)
    series["Orchestra/PPF20_VOL"] = time, reset_vol
    result = comparison.evaluate_tiva(case, {name: name for name in series}, 1.0)
    assert result["dose_mg"] == pytest.approx((500 * 120 + 12 * 1320) * 20 / 3600)
    assert result["volume_disagreement_ml"] == pytest.approx(12.0 / 3600.0, abs=1e-9)
    assert set(result["metrics"]) == {"all", "induction", "maintenance", "emergence"}
    time_out, _observed, predicted, valid, _onset = result["_trace"]
    assert not np.any(valid[(time_out >= 800) & (time_out < 850)])
    assert np.min(predicted) < 50
    assert predicted[-1] > predicted[np.searchsorted(time_out, 1500)]


def test_propofol_replay_rejects_unknown_history_and_dose_disagreement(comparison, case, monkeypatch):
    series = tiva_recording(comparison)
    tracks = {name: name for name in series}
    monkeypatch.setattr(comparison, "numeric_track", series.__getitem__)
    time, volume = series["Orchestra/PPF20_VOL"]
    series["Orchestra/PPF20_VOL"] = time, volume + 1.0
    with pytest.raises(comparison.Excluded, match="began before recording"):
        comparison.evaluate_tiva(case, tracks, 1.0)
    late = time >= 90
    series["Orchestra/PPF20_VOL"] = time[late], volume[late] - volume[late][0]
    with pytest.raises(comparison.Excluded, match="history starts too late"):
        comparison.evaluate_tiva(case, tracks, 1.0)
    series["Orchestra/PPF20_VOL"] = time, volume * 1.2
    with pytest.raises(comparison.Excluded, match="rate and volume disagree"):
        comparison.evaluate_tiva(case, tracks, 1.0)
    series["Orchestra/PPF20_VOL"] = time, volume
    rate_time, rate = series["Orchestra/PPF20_RATE"]
    keep = (rate_time < 800) | (rate_time > 820)
    series["Orchestra/PPF20_RATE"] = rate_time[keep], rate[keep]
    with pytest.raises(comparison.Excluded, match="incomplete input history"):
        comparison.evaluate_tiva(case, tracks, 1.0)


def test_sevo_replay_washin_washout_and_missing_ventilation(comparison, case, monkeypatch):
    time = np.arange(2401, dtype=float)
    inspired = np.where((time >= 60) & (time < 1800), 2.0, 0.0)
    series = {name: (time, np.zeros(len(time))) for name in comparison.SEVO_TRACKS}
    series.update({
        "Primus/INSP_SEVO": (time, inspired),
        "Primus/EXP_SEVO": (time, inspired * 0.8),
        "Primus/TV": (time, np.full(len(time), 500.0)),
        "Primus/RR_CO2": (time, np.full(len(time), 12.0)),
        "Primus/ETCO2": (time, np.full(len(time), 35.0)),
    })
    monkeypatch.setattr(comparison, "numeric_track", series.__getitem__)
    tracks = {name: name for name in series}
    result = comparison.evaluate_sevo(case, tracks, 1.0)
    assert set(result["metrics"]) == {"all", "washin", "maintenance", "washout"}
    assert set(result["cardiac_output_sensitivity"]) == {"3.0", "5.0", "7.0"}
    sample, _observed, predicted, _valid, _onset = result["_trace"]
    assert 0.0 < predicted[-1] < predicted[np.searchsorted(sample, 1800)] < 2.0
    keep = (time < 800) | (time > 840)
    series["Primus/TV"] = time[keep], np.full(keep.sum(), 500.0)
    with pytest.raises(comparison.Excluded, match="incomplete input history"):
        comparison.evaluate_sevo(case, tracks, 1.0)
