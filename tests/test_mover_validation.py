"""Unit tests for MOVER external validation parsers, medication normalization, and helpers."""

from __future__ import annotations

import math

import pytest

from validation.mover.mover_data import (
    classify_approach,
    classify_medication,
    demographics_in_domain,
    f_to_c,
    oz_to_kg,
    parse_bp_pair,
    parse_height_cm,
    patient_from_mover,
)


def test_mover_unit_and_bp_parsers():
    assert oz_to_kg(2564.39) == pytest.approx(72.70, abs=0.05)
    assert oz_to_kg(-10.0) is None
    assert oz_to_kg(None) is None

    assert parse_height_cm("5' 6") == pytest.approx(167.64, abs=0.05)
    assert parse_height_cm('5\' 10"') == pytest.approx(177.80, abs=0.05)
    assert parse_height_cm("5' .5") == pytest.approx(153.67, abs=0.05)
    assert parse_height_cm("6' .05") == pytest.approx(183.01, abs=0.05)
    assert parse_height_cm("170") == pytest.approx(170.0, abs=0.01)
    assert parse_height_cm("") is None

    assert f_to_c(98.6) == pytest.approx(37.0, abs=0.01)
    assert f_to_c(97.0) == pytest.approx(36.11, abs=0.02)
    assert f_to_c(36.5) == pytest.approx(36.5, abs=0.01)
    assert f_to_c(705.2) is None

    assert parse_bp_pair("112/64") == (112.0, 64.0)
    assert parse_bp_pair("60/62") is None
    assert parse_bp_pair("invalid") is None


def test_mover_medication_and_approach_classification():
    assert classify_medication("PROPOFOL 200 MG/20ML IV EMUL", "mg", "Given", 150.0) == (
        "propofol",
        "bolus",
        150.0,
        "mg",
    )
    assert classify_medication("PROPOFOL 1000 MG/100 ML INFUSION (UCI)", "mcg/kg/min", "Rate Change", 100.0) == (
        "propofol",
        "infusion",
        100.0,
        "mcg/kg/min",
    )
    assert classify_medication("REMIFENTANIL INFUSION IN 100ML (ANESTHESIA) (UCI)", "mcg/kg/min", "Stopped", 0.0) == (
        "remifentanil",
        "infusion",
        0.0,
        "mcg/kg/min",
    )
    assert classify_medication("EPINEPHRINE PF 1 MG/10ML IJ SOSY", "mg", "Given", 0.01) == (
        "epinephrine",
        "bolus",
        10.0,
        "mcg",
    )
    assert classify_medication("PHENYLEPHRINE DILUTION 100 MCG/ML IJ SOLN", "mcg", "Given", 100.0) == (
        "phenylephrine",
        "bolus",
        100.0,
        "mcg",
    )
    assert classify_medication("NEOSTIGMINE METHYLSULFATE 5 MG/5ML IV SOSY", "mg", "Given", 3.5) == (
        "neostigmine",
        "bolus",
        3.5,
        "mg",
    )
    assert classify_medication("GLYCOPYRROLATE 1 MG/5ML IV SOSY", "mg", "Given", 0.7) == (
        "glycopyrrolate",
        "bolus",
        0.7,
        "mg",
    )
    # Local anesthetic + epinephrine infiltration mixtures must be excluded.
    assert classify_medication("LIDOCAINE-EPINEPHRINE 1 %-1:100000 IJ SOLN", "mL", "Given", 10.0) is None
    assert classify_medication("PROPOFOL 200 MG/20ML IV EMUL", "mg", "Canceled Entry", 150.0) is None

    assert classify_approach("NEPHRECTOMY, PARTIAL, ROBOT-ASSISTED, LAPAROSCOPIC") == "Laparoscopic or robotic"
    assert classify_approach("GASTRECTOMY, SLEEVE, LAPAROSCOPIC") == "Laparoscopic or robotic"
    assert classify_approach("EGD (ESOPHAGOGASTRODUODENOSCOPY)") == "Endoscopic/other"
    assert classify_approach("ORIF, FRACTURE, RADIUS") == "Open"


def test_mover_patient_construction_and_domain():
    assert demographics_in_domain(54.0, 75.0, 170.0)
    assert not demographics_in_domain(15.0, 75.0, 170.0)
    assert not demographics_in_domain(54.0, 220.0, 170.0)

    p = patient_from_mover(
        {
            "age": 62.0,
            "weight_kg": 92.0,
            "height_cm": 175.0,
            "sex": "male",
            "dm": 1,
            "preop_hb": 12.8,
            "preop_cr": 1.1,
            "preop_glucose": 165.0,
        },
        baseline_map=96.0,
        baseline_hr=74.0,
    )
    assert p.age == 62.0
    assert p.weight == 92.0
    assert p.type2_diabetes is True
    assert p.baseline_hb == pytest.approx(12.8)
    assert p.baseline_glucose == pytest.approx(165.0)
    assert p.creatinine_clearance_ml_min is not None and math.isfinite(p.creatinine_clearance_ml_min)
