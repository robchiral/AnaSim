"""Shared MOVER (UCI EPIC) dataset loader, Parquet cache, parsers, and helpers."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from anasim.patient.domain import (
    AGE_RANGE_YEARS,
    BMI_RANGE_KG_M2,
    HEIGHT_RANGE_CM,
    HEMOGLOBIN_RANGE_G_DL,
    WEIGHT_RANGE_KG,
)
from anasim.patient.patient import Patient
from validation.vitaldb.vitaldb_data import cockcroft_gault_crcl

DEFAULT_MOVER_DIR = Path(
    os.environ.get("MOVER_DATA_DIR", "/Users/robertchen/Documents/GitHub/Outcomes/Raw/MOVER")
)
CACHE_DIR = Path(__file__).resolve().parent / ".mover_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)
SOURCE = (
    "Source: [Samad et al., JAMIA Open, 2023](https://pubmed.ncbi.nlm.nih.gov/37860605/) "
    "(MOVER EPIC dataset, UCI Health, 2017-2022, DOI 10.1093/jamiaopen/ooad084)."
)

OZ_TO_KG = 0.028349523125

_HEIGHT_FT_IN_RE = re.compile(r"^\s*(\d+)\s*'\s*(\d*(?:\.\d+)?)\s*\"?\s*$")
_BP_PAIR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_LAP_RE = re.compile(
    r"LAPAROSCOP|ROBOT(?:IC|\s*-?\s*ASSIST)|GASTRECTOMY.*SLEEVE|SLEEVE\s+GASTRECTOMY|"
    r"CHOLECYSTECTOMY.*LAP|COLECTOMY.*LAP|PROSTATECTOMY.*ROBOT|NEPHRECTOMY.*ROBOT",
    re.IGNORECASE,
)
_NON_PNEUMO_SCOPE_RE = re.compile(
    r"ARTHROSCOP|CYSTOSCOP|URETEROSCOP|HYSTEROSCOP|COLONOSCOP|BRONCHOSCOP|"
    r"ESOPHAGOGASTRO|EGD|LARYNGOSCOP|ENDOSCOPIC\s+SINUS",
    re.IGNORECASE,
)

# Vasoactive / autonomic drugs tracked for strict pressor-free filtering and bolus validation.
VASOACTIVE_DRUGS = frozenset(
    {
        "phenylephrine",
        "ephedrine",
        "norepinephrine",
        "epinephrine",
        "vasopressin",
        "esmolol",
        "labetalol",
        "atropine",
        "glycopyrrolate",
        "nitroglycerin",
        "nicardipine",
        "clevidipine",
        "hydralazine",
        "milrinone",
        "dobutamine",
        "dopamine",
        "dexmedetomidine",
    }
)


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required to query MOVER CSV/Parquet files (pip install duckdb)."
        ) from exc
    return duckdb


def _emr_csv(mover_dir: Path, filename: str) -> Path:
    """Locate an EMR CSV under `<mover_dir>/EMR/` or flat in `<mover_dir>/`."""
    for candidate in (mover_dir / "EMR" / filename, mover_dir / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"MOVER EMR file not found: {mover_dir / 'EMR' / filename}")


def _raw_flowsheet_files(mover_dir: Path) -> list[Path]:
    """Locate raw UCI MOVER `flowsheet_part*.csv` files if present."""
    for base in (mover_dir / "Flowsheets", mover_dir):
        found = sorted(p for p in base.glob("flowsheet_part*.csv") if p.is_file())
        if found:
            return found
    return []


def oz_to_kg(ounces: float | None) -> float | None:
    """Convert MOVER patient weight in ounces to kilograms."""
    if ounces is None or not math.isfinite(ounces) or ounces <= 0:
        return None
    kg = float(ounces) * OZ_TO_KG
    return kg if 20.0 <= kg <= 300.0 else None


def parse_height_cm(text: str | None) -> float | None:
    """Parse MOVER height strings such as `5' 6`, `5' .5`, or `5' 10\"` into centimeters."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = _HEIGHT_FT_IN_RE.match(s)
    if m:
        feet = int(m.group(1))
        inch_str = m.group(2)
        inches = float(inch_str) if inch_str else 0.0
        cm = (feet * 12.0 + inches) * 2.54
        return cm if 100.0 <= cm <= 230.0 else None
    try:
        val = float(s)
    except ValueError:
        return None
    if 100.0 <= val <= 230.0:
        return val
    if 40.0 <= val <= 90.0:
        return val * 2.54
    return None


def f_to_c(temp: float | None) -> float | None:
    """Convert MOVER temperature (°F when >= 50, °C when < 50) to plausible °C."""
    if temp is None or not math.isfinite(temp):
        return None
    c = (temp - 32.0) * 5.0 / 9.0 if temp >= 50.0 else float(temp)
    return c if 30.0 <= c <= 42.0 else None


def parse_bp_pair(text: str | None) -> tuple[float, float] | None:
    """Parse `'SBP/DBP'` string into `(sbp, dbp)` in mmHg."""
    if text is None:
        return None
    m = _BP_PAIR_RE.match(str(text))
    if not m:
        return None
    sbp, dbp = float(m.group(1)), float(m.group(2))
    if not (40.0 <= sbp <= 260.0 and 20.0 <= dbp <= 160.0 and sbp >= dbp + 5.0):
        return None
    return sbp, dbp


def classify_approach(proc_name: str | None) -> str:
    """Classify surgical procedure text into Laparoscopic/robotic, Open, or Endoscopic/other."""
    name = (proc_name or "").strip()
    if not name:
        return "Other"
    if _LAP_RE.search(name):
        return "Laparoscopic or robotic"
    if _NON_PNEUMO_SCOPE_RE.search(name):
        return "Endoscopic/other"
    return "Open"


def classify_medication(
    med_nm: str | None,
    dose_unit: str | None,
    mar_action: str | None,
    raw_value: float | None,
) -> tuple[str, str, float, str] | None:
    """Normalize an IV MOVER medication row into `(drug, kind, value, unit)`.

    `kind` is `'bolus'` or `'infusion'`.
    Canonical bolus units match AnaSim's `give_drug_bolus`:
      - propofol, midazolam, etomidate, ketamine, lidocaine, hydromorphone, morphine,
        rocuronium, succinylcholine, sugammadex, neostigmine, glycopyrrolate, atropine,
        ephedrine, esmolol, labetalol, hydralazine: `mg`
      - fentanyl, remifentanil, phenylephrine, norepinephrine, epinephrine: `mcg`
      - vasopressin: `units`
    Canonical infusion units:
      - propofol, remifentanil, esmolol, dexmedetomidine, dobutamine, dopamine, milrinone: `mcg/kg/min`
      - phenylephrine, norepinephrine, epinephrine, nitroglycerin, nicardipine: `mcg/min` (or `mcg/kg/min` preserved)
      - vasopressin: `units/min`
    """
    if not med_nm or not mar_action:
        return None
    name = med_nm.lower()
    action = mar_action.strip()
    unit = (dose_unit or "").strip().lower()

    is_bolus = action == "Given"
    is_inf = action in ("New Bag", "Rate Change", "Restarted", "Stopped")
    if not (is_bolus or is_inf):
        return None

    # Exclude local-anesthetic/epinephrine mixtures and topical/irrigant/regional solutions.
    if "lidocaine-epinephrine" in name or "bupivacaine" in name or "ropivacaine" in name:
        return None

    drug: str | None = None
    for key, token in (
        ("propofol", "propofol"),
        ("remifentanil", "remifentanil"),
        ("fentanyl", "fentanyl"),
        ("hydromorphone", "hydromorphone"),
        ("morphine", "morphine"),
        ("midazolam", "midazolam"),
        ("etomidate", "etomidate"),
        ("ketamine", "ketamine"),
        ("lidocaine", "lidocaine"),
        ("rocuronium", "rocuronium"),
        ("succinylcholine", "succinylcholine"),
        ("sugammadex", "sugammadex"),
        ("neostigmine", "neostigmine"),
        ("glycopyrrolate", "glycopyrrolate"),
        ("atropine", "atropine"),
        ("phenylephrine", "phenylephrine"),
        ("ephedrine", "ephedrine"),
        ("norepinephrine", "norepinephrine"),
        ("epinephrine", "epinephrine"),
        ("vasopressin", "vasopressin"),
        ("esmolol", "esmolol"),
        ("labetalol", "labetalol"),
        ("dexmedetomidine", "dexmedetomidine"),
        ("nitroglycerin", "nitroglycerin"),
        ("nicardipine", "nicardipine"),
        ("clevidipine", "clevidipine"),
        ("hydralazine", "hydralazine"),
        ("milrinone", "milrinone"),
        ("dobutamine", "dobutamine"),
        ("dopamine", "dopamine"),
    ):
        if token in name:
            drug = key
            break
    if drug is None:
        return None

    if action == "Stopped":
        return (drug, "infusion", 0.0, unit or "stopped")

    if raw_value is None or not math.isfinite(raw_value) or raw_value < 0:
        return None

    if is_inf or "/min" in unit or "/hr" in unit:
        if "/hr" in unit:
            if unit.startswith("mg/kg/hr"):
                return (drug, "infusion", raw_value * 1000.0 / 60.0, "mcg/kg/min")
            if unit.startswith("mcg/kg/hr"):
                return (drug, "infusion", raw_value / 60.0, "mcg/kg/min")
            if unit.startswith("mg/hr"):
                return (drug, "infusion", raw_value * 1000.0 / 60.0, "mcg/min")
            if unit.startswith("mcg/hr"):
                return (drug, "infusion", raw_value / 60.0, "mcg/min")
            if unit.startswith("unit"):
                return (drug, "infusion", raw_value / 60.0, "units/min")
        if unit in ("mcg/kg/min", "ug/kg/min"):
            return (drug, "infusion", raw_value, "mcg/kg/min")
        if unit in ("mcg/min", "ug/min"):
            return (drug, "infusion", raw_value, "mcg/min")
        if unit in ("units/min", "unit/min"):
            return (drug, "infusion", raw_value, "units/min")
        if unit == "mg/min":
            return (drug, "infusion", raw_value * 1000.0, "mcg/min")
        # Infusion rows without rate units (e.g. New Bag with NULL or mL) are skipped unless Stopped.
        return None

    # Bolus dose normalization
    if raw_value <= 0:
        return None
    mcg_drugs = {"fentanyl", "remifentanil", "phenylephrine", "norepinephrine", "epinephrine"}
    if drug == "vasopressin":
        if "unit" in unit:
            return (drug, "bolus", raw_value, "units")
        return None
    if drug in mcg_drugs:
        if unit in ("mcg", "ug"):
            return (drug, "bolus", raw_value, "mcg")
        if unit == "mg":
            return (drug, "bolus", raw_value * 1000.0, "mcg")
        return None
    # mg-based drugs
    if unit == "mg":
        return (drug, "bolus", raw_value, "mg")
    if unit in ("mcg", "ug"):
        return (drug, "bolus", raw_value / 1000.0, "mg")
    return None


def demographics_in_domain(age: float, weight: float, height: float) -> bool:
    """Check AnaSim adult patient domain bounds."""
    if not (AGE_RANGE_YEARS[0] <= age <= AGE_RANGE_YEARS[1]):
        return False
    if not (WEIGHT_RANGE_KG[0] <= weight <= WEIGHT_RANGE_KG[1]):
        return False
    if not (HEIGHT_RANGE_CM[0] <= height <= HEIGHT_RANGE_CM[1]):
        return False
    bmi = weight / (height / 100.0) ** 2
    return BMI_RANGE_KG_M2[0] <= bmi <= BMI_RANGE_KG_M2[1]


def patient_from_mover(
    row: dict[str, Any],
    *,
    baseline_map: float = 90.0,
    baseline_hr: float = 70.0,
    baseline_temp: float = 37.0,
    use_preop_labs: bool = True,
) -> Patient:
    """Construct an AnaSim Patient from a screened MOVER cohort row."""
    age = float(row["age"])
    weight = float(row["weight_kg"])
    height = float(row["height_cm"])
    sex = str(row["sex"]).strip().lower()
    kwargs: dict[str, Any] = {
        "age": age,
        "weight": weight,
        "height": height,
        "sex": sex,
        "baseline_map": float(baseline_map),
        "baseline_hr": float(baseline_hr),
        "baseline_temp": float(baseline_temp),
        "type2_diabetes": bool(row.get("dm")),
    }
    if use_preop_labs:
        hb = row.get("preop_hb")
        if hb is not None and math.isfinite(float(hb)):
            hb_f = float(hb)
            if HEMOGLOBIN_RANGE_G_DL[0] <= hb_f <= HEMOGLOBIN_RANGE_G_DL[1]:
                kwargs["baseline_hb"] = hb_f
        cr = row.get("preop_cr")
        if cr is not None and math.isfinite(float(cr)) and 0.2 <= float(cr) <= 15.0:
            kwargs["creatinine_clearance_ml_min"] = cockcroft_gault_crcl(
                age, weight, height, sex, float(cr)
            )
        glu = row.get("preop_glucose")
        if glu is not None and math.isfinite(float(glu)) and 50.0 <= float(glu) <= 450.0:
            kwargs["baseline_glucose"] = float(glu)
    return Patient(**kwargs)


def ensure_cache(
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    *,
    rebuild: bool = False,
) -> tuple[Path, Path]:
    """Build compact Parquet cache files (`cohort.parquet` and `meds.parquet`) if absent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = cache_dir / "cohort.parquet"
    meds_path = cache_dir / "meds.parquet"
    if not rebuild and cohort_path.exists() and meds_path.exists():
        return cohort_path, meds_path

    if not mover_dir.exists():
        raise FileNotFoundError(f"MOVER directory not found: {mover_dir}")

    duckdb = _duckdb()
    con = duckdb.connect()
    info_sql = str(_emr_csv(mover_dir, "patient_information.csv")).replace("'", "''")
    events_sql = str(_emr_csv(mover_dir, "patient_procedure events.csv")).replace("'", "''")
    history_sql = str(_emr_csv(mover_dir, "patient_history.csv")).replace("'", "''")
    labs_sql = str(_emr_csv(mover_dir, "patient_labs.csv")).replace("'", "''")
    meds_csv_sql = str(_emr_csv(mover_dir, "patient_medications.csv")).replace("'", "''")

    bp_subset = mover_dir / "Flowsheets" / "Subset" / "BP" / "intraop.csv"
    if bp_subset.exists():
        bp_source_sql = f"read_csv('{str(bp_subset).replace(chr(39), chr(39) * 2)}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false)"
    else:
        raw_parts = _raw_flowsheet_files(mover_dir)
        if not raw_parts:
            raise FileNotFoundError(
                f"Neither {bp_subset} nor raw flowsheet_part*.csv found under {mover_dir}. "
                "Run `python -m validation.mover.prepare_flowsheets --mover-dir <path>` to build Flowsheets/Subset/."
            )
        parts_sql = "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in raw_parts) + "]"
        bp_source_sql = f"read_csv({parts_sql}, header=true, all_varchar=true, ignore_errors=true, strict_mode=false, union_by_name=true)"

    # 1. Extract deduplicated, unique-MRN General Anesthesia adult-domain cohort with events,
    #    comorbidities, preop labs, and arterial-line presence.
    raw_rows = con.execute(
        f"""
        WITH info_dedup AS (
            SELECT *,
                   COALESCE(TRY_STRPTIME(AN_START_DATETIME, '%m/%d/%y %H:%M'),
                            TRY_CAST(AN_START_DATETIME AS TIMESTAMP)) AS t_an_start,
                   COALESCE(TRY_STRPTIME(AN_STOP_DATETIME, '%m/%d/%y %H:%M'),
                            TRY_CAST(AN_STOP_DATETIME AS TIMESTAMP)) AS t_an_stop,
                   COALESCE(TRY_STRPTIME(IN_OR_DTTM, '%m/%d/%y %H:%M'),
                            TRY_CAST(IN_OR_DTTM AS TIMESTAMP)) AS t_in_or,
                   COALESCE(TRY_STRPTIME(OUT_OR_DTTM, '%m/%d/%y %H:%M'),
                            TRY_CAST(OUT_OR_DTTM AS TIMESTAMP)) AS t_out_or,
                   ROW_NUMBER() OVER (PARTITION BY LOG_ID ORDER BY AN_START_DATETIME) AS log_rn
            FROM read_csv_auto('{info_sql}', ignore_errors=true)
            WHERE PRIMARY_ANES_TYPE_NM = 'General'
              AND ASA_RATING_C IN (1.0, 2.0, 3.0, 4.0)
              AND SEX IN ('Male', 'Female')
              AND HEIGHT IS NOT NULL AND WEIGHT IS NOT NULL
        ),
        ga_unique_mrn AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY MRN ORDER BY t_an_start NULLS LAST, LOG_ID) AS mrn_rn
            FROM info_dedup
            WHERE log_rn = 1 AND t_an_start IS NOT NULL AND t_an_stop IS NOT NULL
              AND t_an_stop > t_an_start
        ),
        base AS (
            SELECT LOG_ID, MRN,
                   TRY_CAST(BIRTH_DATE AS DOUBLE) AS age,
                   HEIGHT AS height_raw,
                   TRY_CAST(WEIGHT AS DOUBLE) AS weight_oz,
                   lower(SEX) AS sex,
                   CAST(ASA_RATING_C AS INT) AS asa,
                   PRIMARY_PROCEDURE_NM AS procedure_name,
                   t_an_start, t_an_stop, t_in_or, t_out_or
            FROM ga_unique_mrn
            WHERE mrn_rn = 1
              AND TRY_CAST(BIRTH_DATE AS DOUBLE) BETWEEN {AGE_RANGE_YEARS[0]} AND {AGE_RANGE_YEARS[1]}
        ),
        ev AS (
            SELECT LOG_ID,
                   MIN(CASE WHEN EVENT_DISPLAY_NAME = 'Induction' THEN ts END) AS t_induction,
                   MIN(CASE WHEN EVENT_DISPLAY_NAME = 'Intubation' THEN ts END) AS t_intubation,
                   MIN(CASE WHEN EVENT_DISPLAY_NAME = 'Anesthesia Ready' THEN ts END) AS t_ready,
                   MIN(CASE WHEN EVENT_DISPLAY_NAME = 'Emergence' THEN ts END) AS t_emergence,
                   MIN(CASE WHEN EVENT_DISPLAY_NAME = 'Extubation' THEN ts END) AS t_extubation,
                   MAX(CASE WHEN EVENT_DISPLAY_NAME IN ('CV Bypass Initiated', 'Aortic Clamp On') THEN 1 ELSE 0 END) AS has_cpb,
                   MAX(CASE WHEN EVENT_DISPLAY_NAME = 'One-Lung Ventilation' THEN 1 ELSE 0 END) AS has_olv,
                   MAX(CASE WHEN EVENT_DISPLAY_NAME LIKE 'LMA%' THEN 1 ELSE 0 END) AS has_lma
            FROM (
                SELECT LOG_ID, EVENT_DISPLAY_NAME,
                       COALESCE(TRY_STRPTIME(EVENT_TIME, '%m/%d/%y %H:%M'),
                                TRY_CAST(EVENT_TIME AS TIMESTAMP)) AS ts
                FROM read_csv_auto('{events_sql}', ignore_errors=true)
            )
            GROUP BY 1
        ),
        hist AS (
            SELECT mrn AS MRN,
                   MAX(CASE WHEN (diagnosis_code LIKE '401%' OR diagnosis_code LIKE '402%' OR lower(dx_name) LIKE '%hypertension%')
                            AND lower(dx_name) NOT LIKE '%pulmonary%' THEN 1 ELSE 0 END) AS htn,
                   MAX(CASE WHEN (diagnosis_code LIKE '250%' OR lower(dx_name) LIKE '%diabetes%')
                            AND lower(dx_name) NOT LIKE '%prediab%' AND lower(dx_name) NOT LIKE '%pre-diab%' THEN 1 ELSE 0 END) AS dm,
                   MAX(CASE WHEN diagnosis_code LIKE '491%' OR diagnosis_code LIKE '492%' OR diagnosis_code LIKE '496%'
                            OR lower(dx_name) LIKE '%copd%' OR lower(dx_name) LIKE '%chronic obstructive%' THEN 1 ELSE 0 END) AS copd,
                   MAX(CASE WHEN diagnosis_code LIKE '428%' OR lower(dx_name) LIKE '%heart failure%' THEN 1 ELSE 0 END) AS hf,
                   MAX(CASE WHEN diagnosis_code LIKE '571.2%' OR diagnosis_code LIKE '571.5%' OR lower(dx_name) LIKE '%cirrhosis%' THEN 1 ELSE 0 END) AS cirrhosis
            FROM read_csv_auto('{history_sql}', ignore_errors=true)
            GROUP BY 1
        ),
        aline AS (
            SELECT DISTINCT trim(LOG_ID) AS LOG_ID, 1 AS has_aline
            FROM {bp_source_sql}
            WHERE FLO_DISPLAY_NAME IN ('MAP-ART A-line', 'BP-ART A-line', 'Arterial Line MAP (ART)', 'Arterial Line BP (ART)')
        ),
        prelabs AS (
            SELECT l.LOG_ID,
                   ARG_MAX(CASE WHEN l."Lab Code" IN ('30350-3', '718-7') AND val BETWEEN 4 AND 22 THEN val END, ts) AS preop_hb,
                   ARG_MAX(CASE WHEN l."Lab Code" = '2160-0' AND val BETWEEN 0.2 AND 15 THEN val END, ts) AS preop_cr,
                   ARG_MAX(CASE WHEN l."Lab Code" IN ('2345-7', '2339-0') AND val BETWEEN 40 AND 600 THEN val END, ts) AS preop_glucose
            FROM (
                SELECT LOG_ID, "Lab Code",
                       TRY_CAST("Observation Value" AS DOUBLE) AS val,
                       TRY_CAST("Collection Datetime" AS TIMESTAMP) AS ts
                FROM read_csv('{labs_sql}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false)
                WHERE "Lab Code" IN ('30350-3', '718-7', '2160-0', '2345-7', '2339-0')
            ) l
            JOIN base b USING (LOG_ID)
            WHERE l.ts IS NOT NULL
              AND l.ts < b.t_an_start
              AND l.ts >= b.t_an_start - INTERVAL 30 DAY
            GROUP BY 1
        )
        SELECT b.LOG_ID, b.MRN, b.age, b.height_raw, b.weight_oz, b.sex, b.asa, b.procedure_name,
               b.t_an_start, b.t_an_stop, b.t_in_or, b.t_out_or,
               e.t_induction, e.t_intubation, e.t_ready, e.t_emergence, e.t_extubation,
               COALESCE(e.has_cpb, 0) AS has_cpb,
               COALESCE(e.has_olv, 0) AS has_olv,
               COALESCE(e.has_lma, 0) AS has_lma,
               COALESCE(h.htn, 0) AS htn,
               COALESCE(h.dm, 0) AS dm,
               COALESCE(h.copd, 0) AS copd,
               COALESCE(h.hf, 0) AS hf,
               COALESCE(h.cirrhosis, 0) AS cirrhosis,
               COALESCE(a.has_aline, 0) AS has_aline,
               pl.preop_hb, pl.preop_cr, pl.preop_glucose
        FROM base b
        LEFT JOIN ev e USING (LOG_ID)
        LEFT JOIN hist h USING (MRN)
        LEFT JOIN aline a USING (LOG_ID)
        LEFT JOIN prelabs pl USING (LOG_ID)
        """
    ).fetchall()

    # Screen strictly to AnaSim's adult patient domain in Python with exact height/weight parsing.
    screened: list[tuple[Any, ...]] = []
    for r in raw_rows:
        age = float(r[2])
        height_cm = parse_height_cm(r[3])
        weight_kg = oz_to_kg(r[4])
        if height_cm is None or weight_kg is None:
            continue
        if not demographics_in_domain(age, weight_kg, height_cm):
            continue
        bmi = weight_kg / (height_cm / 100.0) ** 2
        approach = classify_approach(r[7])
        screened.append(
            (
                r[0],
                r[1],
                age,
                round(weight_kg, 2),
                round(height_cm, 2),
                round(bmi, 2),
                r[5],
                int(r[6]),
                r[7] or "",
                approach,
                r[8],
                r[9],
                r[10],
                r[11],
                r[12],
                r[13],
                r[14],
                r[15],
                r[16],
                int(r[17]),
                int(r[18]),
                int(r[19]),
                int(r[20]),
                int(r[21]),
                int(r[22]),
                int(r[23]),
                int(r[24]),
                int(r[25]),
                r[26],
                r[27],
                r[28],
            )
        )

    con.execute(
        """
        CREATE TABLE cohort (
            log_id VARCHAR, mrn VARCHAR, age DOUBLE, weight_kg DOUBLE, height_cm DOUBLE,
            bmi DOUBLE, sex VARCHAR, asa INT, procedure_name VARCHAR, approach VARCHAR,
            t_an_start TIMESTAMP, t_an_stop TIMESTAMP, t_in_or TIMESTAMP, t_out_or TIMESTAMP,
            t_induction TIMESTAMP, t_intubation TIMESTAMP, t_ready TIMESTAMP,
            t_emergence TIMESTAMP, t_extubation TIMESTAMP,
            has_cpb INT, has_olv INT, has_lma INT,
            htn INT, dm INT, copd INT, hf INT, cirrhosis INT, has_aline INT,
            preop_hb DOUBLE, preop_cr DOUBLE, preop_glucose DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO cohort VALUES (" + ",".join(["?"] * 31) + ")",
        screened,
    )
    cohort_sql_path = str(cohort_path).replace("'", "''")
    con.execute(f"COPY cohort TO '{cohort_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # 2. Extract and normalize IV perioperative medications within [t_an_start - 90m, t_an_stop + 15m]
    med_raw = con.execute(
        f"""
        SELECT m.LOG_ID,
               TRY_CAST(m.MED_ACTION_TIME AS TIMESTAMP) AS ts,
               m.MEDICATION_NM,
               m.MAR_ACTION_NM,
               TRY_CAST(m.ADMIN_SIG AS DOUBLE) AS val,
               m.DOSE_UNIT_NM
        FROM read_csv('{meds_csv_sql}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false) m
        JOIN cohort c ON m.LOG_ID = c.log_id
        WHERE m.MED_ROUTE_NM = 'IntraVENOUS'
          AND m.MAR_ACTION_NM IN ('Given', 'New Bag', 'Rate Change', 'Restarted', 'Stopped')
          AND TRY_CAST(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN
              c.t_an_start - INTERVAL 90 MINUTE AND c.t_an_stop + INTERVAL 15 MINUTE
        """
    ).fetchall()

    seen: set[tuple[Any, ...]] = set()
    med_norm: list[tuple[Any, ...]] = []
    for log_id, ts, med_nm, action, val, unit in med_raw:
        if ts is None:
            continue
        parsed = classify_medication(med_nm, unit, action, val)
        if parsed is None:
            continue
        drug, kind, norm_val, norm_unit = parsed
        # Deduplicate duplicate MAR rows (e.g. identical IV SOSY + IJ SOLN entries at the same minute).
        dedup_key = (log_id, ts, drug, kind, round(norm_val, 4), norm_unit)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        med_norm.append((log_id, ts, drug, kind, float(norm_val), norm_unit, action))

    con.execute(
        """
        CREATE TABLE meds (
            log_id VARCHAR, ts TIMESTAMP, drug VARCHAR, kind VARCHAR,
            val DOUBLE, unit VARCHAR, action VARCHAR
        )
        """
    )
    con.executemany("INSERT INTO meds VALUES (?, ?, ?, ?, ?, ?, ?)", med_norm)
    meds_sql_path = str(meds_path).replace("'", "''")
    con.execute(
        f"COPY (SELECT * FROM meds ORDER BY log_id, ts, drug) TO '{meds_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return cohort_path, meds_path


def load_cohort(
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
    where: str = "1=1",
) -> list[dict[str, Any]]:
    """Query rows from `cohort.parquet` as a list of dicts."""
    cohort_path, _ = ensure_cache(mover_dir, cache_dir)
    con = _duckdb().connect()
    cur = con.execute(f"SELECT * FROM read_parquet('{cohort_path!s}') WHERE {where} ORDER BY log_id")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def load_meds_for_logs(
    log_ids: Sequence[str],
    mover_dir: Path = DEFAULT_MOVER_DIR,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """Load normalized perioperative IV medication events grouped by `log_id`."""
    if not log_ids:
        return {}
    _, meds_path = ensure_cache(mover_dir, cache_dir)
    con = _duckdb().connect()
    con.execute("CREATE TEMP TABLE ids (log_id VARCHAR)")
    con.executemany("INSERT INTO ids VALUES (?)", [(lid,) for lid in log_ids])
    cur = con.execute(
        f"""
        SELECT m.log_id, m.ts, m.drug, m.kind, m.val, m.unit, m.action
        FROM read_parquet('{meds_path!s}') m
        SEMI JOIN ids USING (log_id)
        ORDER BY m.log_id, m.ts, m.drug
        """
    )
    grouped: dict[str, list[dict[str, Any]]] = {lid: [] for lid in log_ids}
    for log_id, ts, drug, kind, val, unit, action in cur.fetchall():
        grouped[log_id].append(
            {
                "ts": ts,
                "drug": drug,
                "kind": kind,
                "val": float(val),
                "unit": unit,
                "action": action,
            }
        )
    return grouped


def load_flowsheet_subset(
    folder: str,
    log_ids: Sequence[str],
    display_names: Sequence[str] | None = None,
    *,
    mover_dir: Path = DEFAULT_MOVER_DIR,
    split: str = "intraop",
) -> dict[str, list[tuple[datetime, str, str]]]:
    """Read `(ts, display_name, meas_value)` rows from pre-subsetted or raw MOVER flowsheets.

    Prefers pre-subsetted `<mover_dir>/Flowsheets/Subset/<folder>/<split>.csv` (generated by
    `validation.mover.prepare_flowsheets`). When absent, falls back to scanning raw
    `flowsheet_part*.csv` files directly if present.
    """
    if not log_ids:
        return {}
    fn = mover_dir / "Flowsheets" / "Subset" / folder / f"{split}.csv"
    if fn.exists():
        fn_sql = str(fn).replace("'", "''")
        source_sql = f"read_csv('{fn_sql}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false)"
    else:
        raw_parts = _raw_flowsheet_files(mover_dir)
        if not raw_parts:
            raise FileNotFoundError(
                f"MOVER flowsheet subset not found: {fn}. "
                "Generate subsets from raw MOVER `flowsheet_part*.csv` with "
                "`python -m validation.mover.prepare_flowsheets --mover-dir <path>`."
            )
        parts_sql = "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in raw_parts) + "]"
        source_sql = (
            f"read_csv({parts_sql}, header=true, all_varchar=true, ignore_errors=true, "
            f"strict_mode=false, union_by_name=true)"
        )
    con = _duckdb().connect()
    con.execute("CREATE TEMP TABLE ids (LOG_ID VARCHAR)")
    con.executemany("INSERT INTO ids VALUES (?)", [(lid,) for lid in log_ids])
    name_filter = ""
    if display_names:
        quoted = ",".join("'" + n.replace("'", "''") + "'" for n in display_names)
        name_filter = f"AND f.FLO_DISPLAY_NAME IN ({quoted})"
    rows = con.execute(
        f"""
        SELECT trim(f.LOG_ID) AS LOG_ID, TRY_CAST(f.RECORDED_TIME AS TIMESTAMP) AS ts,
               f.FLO_DISPLAY_NAME, f.MEAS_VALUE
        FROM {source_sql} f
        SEMI JOIN ids ON trim(f.LOG_ID) = ids.LOG_ID
        WHERE f.MEAS_VALUE IS NOT NULL {name_filter}
        ORDER BY 1, ts
        """
    ).fetchall()
    out: dict[str, list[tuple[datetime, str, str]]] = {lid: [] for lid in log_ids}
    for lid, ts, name, val in rows:
        if ts is not None and lid in out:
            out[lid].append((ts, name, val))
    return out


def mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap 95% CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(arr.mean())
    if len(arr) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def median_iqr(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Median and interquartile range (25th, 75th percentiles)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan, math.nan)
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return (float(q50), float(q25), float(q75))


def fmt_mean_ci(triple: tuple[float, float, float], digits: int = 1) -> str:
    m, lo, hi = triple
    if not math.isfinite(m):
        return "-"
    return f"{m:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def fmt_med_iqr(triple: tuple[float, float, float], digits: int = 1, suffix: str = "") -> str:
    m, q25, q75 = triple
    if not math.isfinite(m):
        return "-"
    return f"{m:.{digits}f}{suffix} ({q25:.{digits}f} to {q75:.{digits}f})"


def write_results(
    stem: str,
    lines: Iterable[str],
    payload: dict[str, Any],
    *,
    output_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write Markdown report and JSON summary under `validation/mover/results/`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return md_path, json_path


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MOVER Parquet cache for AnaSim validation.")
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    c_path, m_path = ensure_cache(args.mover_dir, args.cache, rebuild=args.rebuild)
    con = _duckdb().connect()
    n_c = con.execute(f"SELECT COUNT(*) FROM read_parquet('{c_path}')").fetchone()[0]
    n_m = con.execute(f"SELECT COUNT(*) FROM read_parquet('{m_path}')").fetchone()[0]
    print(f"Cached {n_c} screened unique-MRN GA patients -> {c_path}")
    print(f"Cached {n_m} normalized IV medication events -> {m_path}")
