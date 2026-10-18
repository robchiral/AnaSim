"""Delivery units, missing-history rejection, and clinical replay transitions."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "validation" / "vitaldb"))
    return importlib.import_module("vitaldb_anesthetics")


@pytest.fixture
def case():
    return dict(age="40", weight="70", height="170", sex="M", opstart="600", opend="1500")


def test_evaluation_eligibility_remains_fixed_when_simulator_accepts_obesity(comparison, case):
    for changes in ({"weight": "110", "height": "190"}, {"weight": "100", "height": "170"}):
        with pytest.raises(comparison.Excluded, match="prespecified evaluation domain"):
            comparison.patient_for({**case, **changes})


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


def test_propofol_replay_preserves_delivered_dose_and_masks_bad_bis(comparison, case, monkeypatch):
    series = tiva_recording(comparison)
    monkeypatch.setattr(comparison, "numeric_track", series.__getitem__)
    result = comparison.evaluate_tiva(case, {name: name for name in series}, 1.0)
    assert result["dose_mg"] == pytest.approx((500 * 120 + 12 * 1320) * 20 / 3600)
    assert result["volume_disagreement_ml"] == pytest.approx(0.0, abs=1e-9)
    assert set(result["metrics"]) == {"all", "induction", "maintenance", "emergence"}
    time, _observed, predicted, valid, _onset = result["_trace"]
    assert not np.any(valid[(time >= 800) & (time < 850)])
    assert np.min(predicted) < 50
    assert predicted[-1] > predicted[np.searchsorted(time, 1500)]


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
    series.update({"Primus/INSP_SEVO": (time, inspired), "Primus/EXP_SEVO": (time, inspired * 0.8),
                   "Primus/TV": (time, np.full(len(time), 500.0)),
                   "Primus/RR_CO2": (time, np.full(len(time), 12.0)),
                   "Primus/ETCO2": (time, np.full(len(time), 35.0))})
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
