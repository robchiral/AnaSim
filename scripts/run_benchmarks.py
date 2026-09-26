import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import mean
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


BENCHMARKS = {
    'hemo': ('baseline', 'sepsis', 'hemorrhage', 'arrhythmia', 'pressors', 'hypothermia'),
    'resp': ('baseline', 'apnea', 'hypercapnia', 'obstruction', 'mech_vent'),
    'mech': ('vcv', 'pcv', 'autopeep'),
    'pk': ('propofol_eleveld', 'norepi_li'),
    'mixed': ('baseline', 'sepsis'),
    'engine': ('awake', 'steady_state'),
}


def _cycle(step, **parameters):
    """Prebind four input sets; tuple values cycle, scalar values stay fixed."""
    calls = [
        partial(step, **{key: value[i] if isinstance(value, tuple) else value
                        for key, value in parameters.items()})
        for i in range(4)
    ]
    return lambda i: calls[i % 4]()


def build_step(name):
    """Build a fresh workload outside the timed loop."""
    from anasim.core.engine import SimulationEngine
    from anasim.core.enums import RhythmType
    from anasim.core.state import SimulationConfig
    from anasim.patient.patient import Patient
    from anasim.patient.pk_models import NorepinephrinePK, PropofolPKEleveld
    from anasim.physiology.hemodynamics import HemodynamicModel
    from anasim.physiology.resp_mech import RespiratoryMechanics
    from anasim.physiology.respiration import RespiratoryModel

    group, case = name.split('.')
    if case not in BENCHMARKS[group]:
        raise ValueError(f'Unknown benchmark: {name}')
    patient = Patient()

    if group == 'hemo':
        model = HemodynamicModel(patient)
        if case == 'sepsis':
            model.sepsis_severity, model.anaphylaxis_severity = 0.7, 0.2
        elif case == 'hemorrhage':
            model.add_volume(-1500.0)
        elif case == 'arrhythmia':
            model.rhythm_type = RhythmType.VTACH
        cases = {
            'baseline': dict(cp_prop=(0.0, 1.0, 2.0, 3.0), cp_remi=(0.0, 1.0, 2.0, 4.0),
                             ce_nore=(0.0, 5.0, 10.0, 20.0), pit=(-2.0, 0.0, 2.0, 4.0),
                             paco2=(35.0, 40.0, 45.0, 55.0), pao2=(90.0, 95.0, 100.0, 75.0),
                             mac_sevo=(0.0, 0.5, 1.0, 1.5)),
            'sepsis': dict(cp_prop=1.5, cp_remi=2.0, ce_nore=(10.0, 15.0, 20.0, 25.0),
                           pit=(-2.0, 0.0, 2.0, 4.0), paco2=(40.0, 45.0, 50.0, 55.0),
                           pao2=(95.0, 90.0, 85.0, 80.0), mac_sevo=0.5, temp_c=38.5),
            'hemorrhage': dict(cp_prop=0.0, cp_remi=0.0),
            'arrhythmia': dict(cp_prop=0.5, cp_remi=1.0),
            'pressors': dict(cp_prop=1.0, cp_remi=1.0, ce_nore=(5.0, 10.0, 15.0, 20.0),
                             ce_epi=(0.0, 1.0, 2.0, 3.0), ce_phenyl=(0.0, 5.0, 10.0, 20.0)),
            'hypothermia': dict(cp_prop=0.5, cp_remi=0.5, temp_c=(34.0, 35.0, 34.5, 35.5)),
        }
        inputs = dict(dt=1.0, ce_nore=0.0, pit=-2.0, paco2=40.0, pao2=95.0)
        return _cycle(model.step, **(inputs | cases[case]))

    if group == 'resp':
        model = RespiratoryModel(patient)
        if case == 'hypercapnia':
            model.state.p_alveolar_co2 = 60.0
        cases = {
            'baseline': dict(ce_prop=(0.0, 1.0, 2.0, 3.0), ce_remi=(0.0, 1.0, 2.0, 4.0),
                             mac_sevo=(0.0, 0.5, 1.0, 1.5), fio2=(0.21, 0.3, 0.5, 0.8)),
            'apnea': dict(ce_prop=6.0, ce_remi=6.0, ce_roc=2.0, fio2=0.3, mac_sevo=1.2),
            'hypercapnia': dict(ce_prop=0.5, ce_remi=0.5, fio2=0.3, mac_sevo=0.2),
            'obstruction': dict(ce_prop=0.5, ce_remi=0.5, airway_patency=0.4,
                                ventilation_efficiency=0.6, vq_mismatch=0.5, fio2=0.4),
            'mech_vent': dict(ce_prop=1.0, ce_remi=1.0, mech_rr=14.0, mech_vt_l=0.5,
                              mech_vent_mv=7.0, fio2=0.5),
        }
        return _cycle(model.step, dt=1.0, **cases[case])

    if group == 'mech':
        model = (RespiratoryMechanics(compliance=0.04, resistance=15.0)
                 if case == 'autopeep' else RespiratoryMechanics())
        settings = {
            'vcv': dict(rr=12.0, vt=0.5, peep=5.0, ie='1:2', mode='VCV'),
            'pcv': dict(rr=12.0, vt=0.5, peep=5.0, ie='1:2', mode='PCV', p_insp=15.0),
            'autopeep': dict(rr=30.0, vt=0.45, peep=8.0, ie='1:1', mode='VCV'),
        }
        model.set_settings(**settings[case])
        return lambda i: model.step(0.05)

    if group == 'pk':
        if case == 'propofol_eleveld':
            return _cycle(PropofolPKEleveld(patient).step, dt_sec=1.0,
                          input_rate_per_sec=(0.0, 0.5, 1.0, 2.0))
        return _cycle(NorepinephrinePK(patient, model='Li').step, dt_sec=1.0,
                      infusion_rate_ug_sec=(0.0, 5.0, 10.0, 20.0),
                      propofol_conc_ug_ml=(0.0, 1.0, 2.0, 3.0))

    if group == 'mixed':
        hemo = HemodynamicModel(patient)
        resp = RespiratoryModel(patient)
        mech = RespiratoryMechanics()
        pk = PropofolPKEleveld(patient)
        sepsis = case == 'sepsis'
        if sepsis:
            hemo.sepsis_severity, hemo.anaphylaxis_severity = 0.7, 0.2
        infusions = (0.5, 1.0, 1.5, 2.0) if sepsis else (0.0, 0.5, 1.0, 2.0)
        remi = (1.0, 2.0, 3.0, 4.0) if sepsis else (0.0, 1.0, 2.0, 4.0)
        nore = (10.0, 15.0, 20.0, 25.0) if sepsis else (0.0, 0.0, 0.0, 0.0)
        gas = dict(paco2=45.0, pao2=85.0, mac_sevo=0.5, temp_c=38.5) if sepsis else dict(paco2=40.0, pao2=95.0)

        def step(i):
            index = i % 4
            pk_state = pk.step(1.0, infusions[index])
            hemo_state = hemo.step(1.0, pk_state.c1, remi[index], nore[index], -2.0, **gas)
            resp.step(1.0, pk_state.ce, remi[index], mac_sevo=0.5 if sepsis else 0.0,
                      cardiac_output=hemo_state.co)
            mech.step(0.05)
        return step

    engine = SimulationEngine(patient, SimulationConfig(mode=case, rng_seed=123, dt=0.1))
    engine.start()
    return lambda i: engine.step(0.1)


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
    groups = defaultdict(list)
    for result in results:
        groups[result.name].append(result)
    return [BenchmarkResult(name, mean(r.seconds for r in runs), runs[0].steps)
            for name, runs in groups.items()]


def _format_slowest(results: list[BenchmarkResult], limit: int = 5) -> str:
    if not results:
        return ""
    ranked = sorted(results, key=lambda r: r.us_per_step, reverse=True)
    ranked = ranked[: min(limit, len(ranked))]
    lines = ["Slowest (avg us/step):"]
    for result in ranked:
        lines.append(f"  {result.name}: {result.us_per_step:.2f} us/step")
    return "\n".join(lines)


def _parse_bench_list(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return list(available)

    requested = [item.strip() for item in value.split(",") if item.strip()]
    available_keys = available
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
    benchmarks = [f"{group}.{case}" for group, cases in BENCHMARKS.items() for case in cases]

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
        for _ in range(args.repeat):
            profiler = None
            if args.profile:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            step = build_step(name)
            elapsed = _time_indexed(step, args.steps, args.warmup)
            result = BenchmarkResult(name, elapsed, args.steps)
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
