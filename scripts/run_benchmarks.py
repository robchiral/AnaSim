import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class BenchmarkResult:
    name: str
    seconds: float
    steps: int

    @property
    def us_per_step(self) -> float:
        if self.steps <= 0:
            return 0.0
        return (self.seconds / self.steps) * 1_000_000.0

    @property
    def steps_per_sec(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.steps / self.seconds


def _time_loop(fn, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(steps):
        fn()
    return perf_counter() - start


def _time_indexed(fn, steps: int, warmup: int) -> float:
    for i in range(warmup):
        fn(i)
    start = perf_counter()
    for i in range(steps):
        fn(i)
    return perf_counter() - start


def _print_profile(name: str, profiler, limit: int) -> None:
    import io
    import pstats

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("tottime").print_stats(limit)
    output = stream.getvalue().strip()
    if output:
        print(f"\n[Profile] {name}\n{output}\n")


def _resolve_profile_path(template: Optional[str], bench: str, multi: bool) -> Optional[str]:
    if not template:
        return None
    if "{bench}" in template:
        return template.format(bench=bench)
    if not multi:
        return template
    path = Path(template)
    if path.suffix:
        return str(path.with_name(f"{path.stem}.{bench}{path.suffix}"))
    return f"{template}.{bench}.prof"


# -----------------------------------------------------------------------------
# Hemodynamic Benchmarks
# -----------------------------------------------------------------------------


def bench_hemo_baseline(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel

    model = HemodynamicModel(Patient())
    dt = 1.0

    prop_vals = (0.0, 1.0, 2.0, 3.0)
    remi_vals = (0.0, 1.0, 2.0, 4.0)
    nore_vals = (0.0, 5.0, 10.0, 20.0)
    pit_vals = (-2.0, 0.0, 2.0, 4.0)
    paco2_vals = (35.0, 40.0, 45.0, 55.0)
    pao2_vals = (90.0, 95.0, 100.0, 75.0)
    mac_vals = (0.0, 0.5, 1.0, 1.5)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(
            dt,
            prop_vals[idx],
            remi_vals[idx],
            nore_vals[idx],
            pit_vals[idx],
            paco2_vals[idx],
            pao2_vals[idx],
            mac_sevo=mac_vals[idx],
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.baseline", elapsed, steps)


def bench_hemo_sepsis(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel

    model = HemodynamicModel(Patient())
    model.sepsis_severity = 0.7
    model.anaphylaxis_severity = 0.2
    dt = 1.0

    nore_vals = (10.0, 15.0, 20.0, 25.0)
    pit_vals = (-2.0, 0.0, 2.0, 4.0)
    paco2_vals = (40.0, 45.0, 50.0, 55.0)
    pao2_vals = (95.0, 90.0, 85.0, 80.0)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(
            dt,
            1.5,
            2.0,
            nore_vals[idx],
            pit_vals[idx],
            paco2_vals[idx],
            pao2_vals[idx],
            mac_sevo=0.5,
            temp_c=38.5,
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.sepsis", elapsed, steps)


def bench_hemo_hemorrhage(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel

    model = HemodynamicModel(Patient())
    model.add_volume(-1500.0)
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(dt, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.hemorrhage", elapsed, steps)


def bench_hemo_arrhythmia(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel
    from anasim.core.enums import RhythmType

    model = HemodynamicModel(Patient())
    model.rhythm_type = RhythmType.VTACH
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(dt, 0.5, 1.0, 0.0, -2.0, 40.0, 95.0)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.arrhythmia", elapsed, steps)


def bench_hemo_pressors(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel

    model = HemodynamicModel(Patient())
    dt = 1.0

    nore_vals = (5.0, 10.0, 15.0, 20.0)
    epi_vals = (0.0, 1.0, 2.0, 3.0)
    phenyl_vals = (0.0, 5.0, 10.0, 20.0)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(
            dt,
            1.0,
            1.0,
            nore_vals[idx],
            -2.0,
            40.0,
            95.0,
            ce_epi=epi_vals[idx],
            ce_phenyl=phenyl_vals[idx],
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.pressors", elapsed, steps)


def bench_hemo_hypothermia(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel

    model = HemodynamicModel(Patient())
    dt = 1.0

    temp_vals = (34.0, 35.0, 34.5, 35.5)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(dt, 0.5, 0.5, 0.0, -2.0, 40.0, 95.0, temp_c=temp_vals[idx])

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("hemo.hypothermia", elapsed, steps)


# -----------------------------------------------------------------------------
# Respiratory Benchmarks
# -----------------------------------------------------------------------------


def bench_resp_baseline(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.respiration import RespiratoryModel

    model = RespiratoryModel(Patient())
    dt = 1.0

    prop_vals = (0.0, 1.0, 2.0, 3.0)
    remi_vals = (0.0, 1.0, 2.0, 4.0)
    sevo_vals = (0.0, 0.5, 1.0, 1.5)
    fio2_vals = (0.21, 0.30, 0.50, 0.80)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(
            dt,
            prop_vals[idx],
            remi_vals[idx],
            fio2=fio2_vals[idx],
            mac_sevo=sevo_vals[idx],
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("resp.baseline", elapsed, steps)


def bench_resp_apnea(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.respiration import RespiratoryModel

    model = RespiratoryModel(Patient())
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(dt, 6.0, 6.0, ce_roc=2.0, fio2=0.3, mac_sevo=1.2)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("resp.apnea", elapsed, steps)


def bench_resp_hypercapnia(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.respiration import RespiratoryModel

    model = RespiratoryModel(Patient())
    model.state.p_alveolar_co2 = 60.0
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(dt, 0.5, 0.5, fio2=0.3, mac_sevo=0.2)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("resp.hypercapnia", elapsed, steps)


def bench_resp_obstruction(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.respiration import RespiratoryModel

    model = RespiratoryModel(Patient())
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(
            dt,
            0.5,
            0.5,
            airway_patency=0.4,
            ventilation_efficiency=0.6,
            vq_mismatch=0.5,
            fio2=0.4,
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("resp.obstruction", elapsed, steps)


def bench_resp_mech_vent(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.respiration import RespiratoryModel

    model = RespiratoryModel(Patient())
    dt = 1.0

    def run_step(_: int) -> None:
        model.step(
            dt,
            1.0,
            1.0,
            mech_rr=14.0,
            mech_vt_l=0.5,
            mech_vent_mv=7.0,
            fio2=0.5,
        )

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("resp.mech_vent", elapsed, steps)


# -----------------------------------------------------------------------------
# Respiratory Mechanics Benchmarks
# -----------------------------------------------------------------------------


def bench_mech_vcv(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.physiology.resp_mech import RespiratoryMechanics

    model = RespiratoryMechanics()
    model.set_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="VCV")
    dt = 0.05

    def run_step() -> None:
        model.step(dt)

    elapsed = _time_loop(run_step, steps, warmup)
    return BenchmarkResult("mech.vcv", elapsed, steps)


def bench_mech_pcv(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.physiology.resp_mech import RespiratoryMechanics

    model = RespiratoryMechanics()
    model.set_settings(rr=12.0, vt=0.5, peep=5.0, ie="1:2", mode="PCV", p_insp=15.0)
    dt = 0.05

    def run_step() -> None:
        model.step(dt)

    elapsed = _time_loop(run_step, steps, warmup)
    return BenchmarkResult("mech.pcv", elapsed, steps)


def bench_mech_autopeep(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.physiology.resp_mech import RespiratoryMechanics

    model = RespiratoryMechanics(compliance=0.04, resistance=15.0)
    model.set_settings(rr=30.0, vt=0.45, peep=8.0, ie="1:1", mode="VCV")
    dt = 0.05

    def run_step() -> None:
        model.step(dt)

    elapsed = _time_loop(run_step, steps, warmup)
    return BenchmarkResult("mech.autopeep", elapsed, steps)


# -----------------------------------------------------------------------------
# PK Benchmarks
# -----------------------------------------------------------------------------


def bench_pk_propofol(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.patient.pk_models import PropofolPKEleveld

    model = PropofolPKEleveld(Patient())
    dt = 1.0
    infusion_vals = (0.0, 0.5, 1.0, 2.0)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(dt, infusion_vals[idx])

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("pk.propofol_eleveld", elapsed, steps)


def bench_pk_norepi(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.patient.pk_models import NorepinephrinePK

    model = NorepinephrinePK(Patient(), model="Li")
    dt = 1.0
    infusion_vals = (0.0, 5.0, 10.0, 20.0)
    prop_vals = (0.0, 1.0, 2.0, 3.0)

    def run_step(i: int) -> None:
        idx = i & 3
        model.step(dt, infusion_vals[idx], propofol_conc_ug_ml=prop_vals[idx])

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("pk.norepi_li", elapsed, steps)


# -----------------------------------------------------------------------------
# Mixed Pipeline Benchmarks
# -----------------------------------------------------------------------------


def bench_mixed_baseline(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel
    from anasim.physiology.respiration import RespiratoryModel
    from anasim.physiology.resp_mech import RespiratoryMechanics
    from anasim.patient.pk_models import PropofolPKEleveld

    patient = Patient()
    hemo = HemodynamicModel(patient)
    resp = RespiratoryModel(patient)
    mech = RespiratoryMechanics()
    pk = PropofolPKEleveld(patient)

    dt = 1.0
    infusion_vals = (0.0, 0.5, 1.0, 2.0)
    remi_vals = (0.0, 1.0, 2.0, 4.0)

    def run_step(i: int) -> None:
        idx = i & 3
        pk_state = pk.step(dt, infusion_vals[idx])
        hemo_state = hemo.step(dt, pk_state.ce, remi_vals[idx], 0.0, -2.0, 40.0, 95.0)
        resp.step(dt, pk_state.ce, remi_vals[idx], mac_sevo=0.0, cardiac_output=hemo_state.co)
        mech.step(0.05)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("mixed.baseline", elapsed, steps)


def bench_mixed_sepsis(steps: int, warmup: int) -> BenchmarkResult:
    from anasim.patient.patient import Patient
    from anasim.physiology.hemodynamics import HemodynamicModel
    from anasim.physiology.respiration import RespiratoryModel
    from anasim.physiology.resp_mech import RespiratoryMechanics
    from anasim.patient.pk_models import PropofolPKEleveld

    patient = Patient()
    hemo = HemodynamicModel(patient)
    hemo.sepsis_severity = 0.7
    hemo.anaphylaxis_severity = 0.2
    resp = RespiratoryModel(patient)
    mech = RespiratoryMechanics()
    pk = PropofolPKEleveld(patient)

    dt = 1.0
    infusion_vals = (0.5, 1.0, 1.5, 2.0)
    remi_vals = (1.0, 2.0, 3.0, 4.0)
    nore_vals = (10.0, 15.0, 20.0, 25.0)

    def run_step(i: int) -> None:
        idx = i & 3
        pk_state = pk.step(dt, infusion_vals[idx])
        hemo_state = hemo.step(
            dt,
            pk_state.ce,
            remi_vals[idx],
            nore_vals[idx],
            -2.0,
            45.0,
            85.0,
            mac_sevo=0.5,
            temp_c=38.5,
        )
        resp.step(dt, pk_state.ce, remi_vals[idx], mac_sevo=0.5, cardiac_output=hemo_state.co)
        mech.step(0.05)

    elapsed = _time_indexed(run_step, steps, warmup)
    return BenchmarkResult("mixed.sepsis", elapsed, steps)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _format_results(results: list[BenchmarkResult]) -> str:
    name_width = max(len(r.name) for r in results)
    lines = []
    header = f"{'Benchmark':<{name_width}}  {'ms':>10}  {'us/step':>10}  {'steps/s':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        ms = result.seconds * 1_000.0
        lines.append(
            f"{result.name:<{name_width}}  {ms:>10.2f}  {result.us_per_step:>10.2f}  {result.steps_per_sec:>12.2f}"
        )
    return "\n".join(lines)


def _aggregate_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    buckets: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = buckets.setdefault(result.name, {"seconds": 0.0, "steps": 0.0, "runs": 0.0})
        bucket["seconds"] += result.seconds
        bucket["steps"] += result.steps
        bucket["runs"] += 1.0
    aggregated = []
    for name, data in buckets.items():
        runs = data["runs"] or 1.0
        aggregated.append(
            BenchmarkResult(
                name=name,
                seconds=data["seconds"] / runs,
                steps=int(data["steps"] / runs),
            )
        )
    return aggregated


def _format_slowest(results: list[BenchmarkResult], limit: int = 5) -> str:
    if not results:
        return ""
    ranked = sorted(results, key=lambda r: r.us_per_step, reverse=True)
    ranked = ranked[: min(limit, len(ranked))]
    lines = ["Slowest (avg us/step):"]
    for result in ranked:
        lines.append(f"  {result.name}: {result.us_per_step:.2f} us/step")
    return "\n".join(lines)


def _parse_bench_list(value: str, available: dict[str, callable]) -> list[str]:
    if value == "all":
        return list(available.keys())

    requested = [item.strip() for item in value.split(",") if item.strip()]
    available_keys = list(available.keys())
    selected: list[str] = []
    unknown: list[str] = []

    for name in requested:
        if name in available:
            selected.append(name)
            continue
        prefix = f"{name}."
        matches = [key for key in available_keys if key.startswith(prefix)]
        if matches:
            selected.extend(matches)
        else:
            unknown.append(name)

    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    return selected


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    benchmarks = {
        "hemo.baseline": bench_hemo_baseline,
        "hemo.sepsis": bench_hemo_sepsis,
        "hemo.hemorrhage": bench_hemo_hemorrhage,
        "hemo.arrhythmia": bench_hemo_arrhythmia,
        "hemo.pressors": bench_hemo_pressors,
        "hemo.hypothermia": bench_hemo_hypothermia,
        "resp.baseline": bench_resp_baseline,
        "resp.apnea": bench_resp_apnea,
        "resp.hypercapnia": bench_resp_hypercapnia,
        "resp.obstruction": bench_resp_obstruction,
        "resp.mech_vent": bench_resp_mech_vent,
        "mech.vcv": bench_mech_vcv,
        "mech.pcv": bench_mech_pcv,
        "mech.autopeep": bench_mech_autopeep,
        "pk.propofol_eleveld": bench_pk_propofol,
        "pk.norepi_li": bench_pk_norepi,
        "mixed.baseline": bench_mixed_baseline,
        "mixed.sepsis": bench_mixed_sepsis,
    }

    parser = argparse.ArgumentParser(description="AnaSim micro-benchmarks")
    parser.add_argument(
        "--bench",
        default="all",
        help=(
            "Comma-separated list of benchmarks, group prefix, or 'all'. "
            "Examples: hemo,resp.baseline,pk.propofol_eleveld"
        ),
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=30)
    parser.add_argument("--profile-out", default=None)
    args = parser.parse_args()

    try:
        selected = _parse_bench_list(args.bench, benchmarks)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    multi_profile = len(selected) > 1
    results = []

    for name in selected:
        bench = benchmarks[name]
        for _ in range(args.repeat):
            profiler = None
            if args.profile:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            result = bench(args.steps, args.warmup)
            results.append(result)
            if profiler is not None:
                profiler.disable()
                if args.profile_out:
                    profile_path = _resolve_profile_path(args.profile_out, name, multi_profile)
                    if profile_path:
                        profiler.dump_stats(profile_path)
                _print_profile(name, profiler, args.profile_limit)

    print(_format_results(results))

    aggregated = _aggregate_results(results)
    slowest = _format_slowest(aggregated)
    if slowest:
        print("\n" + slowest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
