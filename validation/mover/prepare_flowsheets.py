"""Regenerate per-category MOVER flowsheet subsets from raw UCI EPIC `flowsheet_part*.csv` files.

Raw MOVER EPIC downloads provide patient measurements across 19 large `flowsheet_part*.csv`
files alongside `EMR/patient_information.csv`. This script builds `patient_timings.csv`
(`START_TIME = min(IN_OR_DTTM, AN_START_DATETIME)`, `END_TIME = max(OUT_OR_DTTM, AN_STOP_DATETIME)`)
and splits the required measurement categories into `Flowsheets/Subset/<Category>/intraop.csv`
(and optionally `preop.csv`, `postop.csv`, and `unknown.csv`) using DuckDB.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from validation.mover.mover_data import DEFAULT_MOVER_DIR, _duckdb

# Categories used by AnaSim's MOVER validation scripts plus standard perioperative groups.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "BP": (
        "BP",
        "MAP (mmHg)",
        "NIBP",
        "NIBP - MAP",
        "Arterial Line BP (ART)",
        "Arterial Line MAP (ART)",
        "BP-ART A-line",
        "MAP-ART A-line",
    ),
    "HR": ("Pulse", "Heart Rate"),
    "Gas": (
        "DES ET",
        "DES In",
        "ISO ET",
        "ISO In",
        "Nitrous Oxide",
        "SEV ET",
        "SEV In",
        "FIN20 (%)",
        "ETN2O",
    ),
    "O2": ("O2 Device", "O2 Flow Rate (L/min)", "FiO2 (%)"),
    "SpO2": ("SpO2",),
    "Temp": ("Temp",),
    "Blood": ("PRBC", "QBL"),
    "Volume": ("QBL", "Volume", "Additional Intake"),
    "Urine": ("Urine Output",),
    "Vent": ("Vent Mode",),
    "Airway": ("Airway Device",),
    "Body": ("Weight", "Height", "BMI kg/m2", "BMI (Calculated)"),
    "Rhythm": ("Cardiac Rhythm",),
    "HRSource": ("Heart Rate Source",),
    "RR": ("Resp",),
}

VALIDATION_CATEGORIES: tuple[str, ...] = ("BP", "HR", "Gas", "O2", "Temp", "Blood", "Volume")


def _find_patient_info(mover_dir: Path) -> Path:
    for candidate in (
        mover_dir / "EMR" / "patient_information.csv",
        mover_dir / "patient_information.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find patient_information.csv under {mover_dir} or {mover_dir / 'EMR'}"
    )


def _find_raw_flowsheets(mover_dir: Path, glob_pattern: str) -> list[Path]:
    for base in (mover_dir / "Flowsheets", mover_dir):
        matches = sorted(p for p in base.glob(glob_pattern) if p.is_file())
        if matches:
            return matches
    raise FileNotFoundError(
        f"No raw flowsheet files matching {glob_pattern!r} found under {mover_dir} or {mover_dir / 'Flowsheets'}"
    )


def generate_patient_timings(
    patient_info_csv: Path,
    output_csv: Path,
) -> int:
    """Build `patient_timings.csv` (`LOG_ID, MRN, START_TIME, END_TIME`) from `patient_information.csv`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    con = _duckdb().connect()
    in_sql = str(patient_info_csv).replace("'", "''")
    out_sql = str(output_csv).replace("'", "''")

    con.execute(
        f"""
        CREATE TEMP TABLE timings AS
        WITH parsed AS (
            SELECT trim(LOG_ID) AS LOG_ID,
                   trim(MRN) AS MRN,
                   COALESCE(TRY_STRPTIME(IN_OR_DTTM, '%m/%d/%y %H:%M'), TRY_CAST(IN_OR_DTTM AS TIMESTAMP)) AS t_in,
                   COALESCE(TRY_STRPTIME(OUT_OR_DTTM, '%m/%d/%y %H:%M'), TRY_CAST(OUT_OR_DTTM AS TIMESTAMP)) AS t_out,
                   COALESCE(TRY_STRPTIME(AN_START_DATETIME, '%m/%d/%y %H:%M'), TRY_CAST(AN_START_DATETIME AS TIMESTAMP)) AS t_as,
                   COALESCE(TRY_STRPTIME(AN_STOP_DATETIME, '%m/%d/%y %H:%M'), TRY_CAST(AN_STOP_DATETIME AS TIMESTAMP)) AS t_ae
            FROM read_csv('{in_sql}', header=true, all_varchar=true, ignore_errors=true, strict_mode=false)
        ),
        bounds AS (
            SELECT LOG_ID,
                   MRN,
                   LEAST(t_in, t_as) AS START_TIME,
                   GREATEST(t_out, t_ae) AS END_TIME
            FROM parsed
        )
        SELECT DISTINCT LOG_ID, MRN,
               strftime(START_TIME, '%Y-%m-%d %H:%M:%S') AS START_TIME,
               strftime(END_TIME, '%Y-%m-%d %H:%M:%S') AS END_TIME
        FROM bounds
        WHERE LOG_ID IS NOT NULL AND MRN IS NOT NULL
          AND START_TIME IS NOT NULL AND END_TIME IS NOT NULL
          AND END_TIME > START_TIME
        """
    )
    n_rows = int(con.execute("SELECT COUNT(*) FROM timings").fetchone()[0])
    con.execute(f"COPY timings TO '{out_sql}' (HEADER, DELIMITER ',')")
    return n_rows


def extract_flowsheet_subsets(
    mover_dir: Path,
    *,
    output_base: Path | None = None,
    categories: Sequence[str] = VALIDATION_CATEGORIES,
    glob_pattern: str = "flowsheet_part*.csv",
    all_splits: bool = False,
) -> dict[str, dict[str, int]]:
    """Extract category CSVs into `Flowsheets/Subset/<Category>/<split>.csv` from raw flowsheet parts."""
    for cat in categories:
        if cat not in CATEGORIES:
            raise ValueError(f"Unknown category {cat!r}; choose from {sorted(CATEGORIES)}")

    out_root = output_base or (mover_dir / "Flowsheets" / "Subset")
    out_root.mkdir(parents=True, exist_ok=True)
    patient_info = _find_patient_info(mover_dir)
    timings_csv = out_root / "patient_timings.csv"
    generate_patient_timings(patient_info, timings_csv)

    raw_files = _find_raw_flowsheets(mover_dir, glob_pattern)
    con = _duckdb().connect()
    timings_sql = str(timings_csv).replace("'", "''")
    con.execute(
        f"""
        CREATE TEMP TABLE timings AS
        SELECT LOG_ID, MRN,
               TRY_CAST(START_TIME AS TIMESTAMP) AS start_ts,
               TRY_CAST(END_TIME AS TIMESTAMP) AS end_ts
        FROM read_csv('{timings_sql}', header=true, all_varchar=true)
        """
    )

    # Read raw flowsheet parts once into a filtered temp table containing only requested display names.
    needed_names = sorted({nm for cat in categories for nm in CATEGORIES[cat]})
    names_sql = ",".join("'" + n.replace("'", "''") + "'" for n in needed_names)
    files_sql = "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in raw_files) + "]"

    con.execute(
        f"""
        CREATE TEMP TABLE filtered_flows AS
        SELECT trim(f.LOG_ID) AS LOG_ID,
               trim(f.MRN) AS MRN,
               f.FLO_NAME,
               f.FLO_DISPLAY_NAME,
               f.RECORD_TYPE,
               f.RECORDED_TIME,
               f.MEAS_VALUE,
               f.UNITS,
               CASE
                   WHEN t.start_ts IS NULL OR TRY_CAST(f.RECORDED_TIME AS TIMESTAMP) IS NULL THEN 'unknown'
                   WHEN TRY_CAST(f.RECORDED_TIME AS TIMESTAMP) < t.start_ts THEN 'preop'
                   WHEN TRY_CAST(f.RECORDED_TIME AS TIMESTAMP) > t.end_ts THEN 'postop'
                   ELSE 'intraop'
               END AS split_name
        FROM read_csv({files_sql}, header=true, all_varchar=true, ignore_errors=true, strict_mode=false, union_by_name=true) f
        LEFT JOIN timings t
          ON trim(f.LOG_ID) = t.LOG_ID AND trim(f.MRN) = t.MRN
        WHERE f.FLO_DISPLAY_NAME IN ({names_sql})
          AND f.MEAS_VALUE IS NOT NULL
          AND trim(f.MEAS_VALUE) <> ''
        """
    )

    splits = ("intraop", "preop", "postop", "unknown") if all_splits else ("intraop",)
    counts: dict[str, dict[str, int]] = {}
    for cat in categories:
        cat_dir = out_root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_names_sql = ",".join("'" + n.replace("'", "''") + "'" for n in CATEGORIES[cat])
        counts[cat] = {}
        for split in splits:
            out_csv = cat_dir / f"{split}.csv"
            out_sql = str(out_csv).replace("'", "''")
            query = (
                f"SELECT LOG_ID, MRN, FLO_NAME, FLO_DISPLAY_NAME, RECORD_TYPE, RECORDED_TIME, MEAS_VALUE, UNITS "
                f"FROM filtered_flows "
                f"WHERE FLO_DISPLAY_NAME IN ({cat_names_sql}) AND split_name = '{split}'"
            )
            n = int(con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0])
            con.execute(f"COPY ({query}) TO '{out_sql}' (HEADER, DELIMITER ',')")
            counts[cat][split] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Flowsheets/Subset category CSVs from raw MOVER flowsheet_part*.csv files."
    )
    parser.add_argument("--mover-dir", type=Path, default=DEFAULT_MOVER_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination subset directory (defaults to <mover-dir>/Flowsheets/Subset).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(VALIDATION_CATEGORIES),
        help=f"Categories to extract (default: {' '.join(VALIDATION_CATEGORIES)}; pass 'all' for all).",
    )
    parser.add_argument(
        "--glob",
        default="flowsheet_part*.csv",
        help="Filename glob for raw flowsheet parts inside Flowsheets/ or mover-dir.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Also write preop.csv, postop.csv, and unknown.csv (default writes intraop.csv only).",
    )
    args = parser.parse_args()
    cats = list(CATEGORIES) if args.categories == ["all"] else args.categories
    counts = extract_flowsheet_subsets(
        args.mover_dir,
        output_base=args.output_dir,
        categories=cats,
        glob_pattern=args.glob,
        all_splits=args.all_splits,
    )
    for cat, split_counts in counts.items():
        summary = ", ".join(f"{k}={v}" for k, v in split_counts.items())
        print(f"{cat}: {summary}")


if __name__ == "__main__":
    main()
