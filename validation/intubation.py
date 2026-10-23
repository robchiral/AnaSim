"""Induction and intubation replays. Run with python -m validation.intubation.

Billard 1994 supplies SBP and HR. Sawano 2013 adds MAP, HR, CI, and SVI in
normotensive patients. Sjøen 2023 and 2024 randomize propofol and remifentanil
induction doses with invasive pressure, and Poterman 2017 high-dose remifentanil
with or without atropine. All are compared with typical-patient predictions.
"""

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

from anasim.core.engine import SimulationEngine
from anasim.core.enums import RhythmType
from anasim.core.state import SimulationConfig
from anasim.monitors.arterial import ArterialWaveformRenderer
from anasim.monitors.cardiac_cycle import CardiacCycle
from anasim.patient.patient import Patient
from anasim.patient.pd.anesthesia import ClinicalResponseModel, opioid_equivalent
from anasim.patient.pk_models import FentanylPK, PropofolPKEleveld
from anasim.physiology.disturbances import Disturbances
from anasim.physiology.hemodynamics import HemodynamicModel


@dataclass(frozen=True)
class BillardCase:
    fentanyl: float  # mcg/kg
    propofol: float  # mg/kg
    age: float
    weight: float
    baseline_hr: float
    baseline_sbp: float
    pre_sbp: float
    post_sbp: float
    pre_hr: float
    post_hr: float


BILLARD_CASES = (
    BillardCase(0, 2.0, 44, 65, 86, 126, 110, 186, 76, 99),
    BillardCase(0, 2.5, 41, 71, 83, 132, 114, 170, 80, 99),
    BillardCase(0, 3.0, 46, 64, 82, 144, 114, 192, 82, 96),
    BillardCase(0, 3.5, 40, 66, 88, 140, 113, 164, 84, 96),
    BillardCase(2, 2.0, 50, 62, 84, 147, 87, 131, 66, 74),
    BillardCase(2, 2.5, 44, 68, 84, 148, 87, 153, 68, 82),
    BillardCase(2, 3.0, 48, 65, 85, 137, 77, 117, 70, 81),
    BillardCase(2, 3.5, 43, 60, 90, 141, 91, 141, 76, 91),
    BillardCase(4, 2.0, 46, 59, 78, 138, 89, 127, 67, 73),
    BillardCase(4, 2.5, 50, 62, 79, 147, 81, 106, 66, 70),
    BillardCase(4, 3.0, 46, 66, 79, 148, 90, 136, 65, 84),
    BillardCase(4, 3.5, 44, 62, 75, 138, 88, 124, 63, 73),
)


class Reading(NamedTuple):
    sbp: float
    hr: float
    map: float
    co: float
    svr: float  # MAP/CO, Wood units; excludes a CVP subtraction.


class IndexedReading(NamedTuple):
    map: float
    hr: float
    ci: float  # L/min/m2
    svi: float  # mL/m2


@dataclass(frozen=True)
class SawanoCase:
    fentanyl: float  # mcg/kg
    age: float
    height: float
    weight: float
    observed: tuple[IndexedReading, ...]
    map_sd: tuple[float, ...]
    hr_sd: tuple[float, ...]


# Sawano 2013 Tables 1a and 2a: awake, before intubation, then 1/2/3/4/5/10 min.
# The timing of "after induction" and "immediately after intubation" is not
# specified precisely, so neither column is assigned a point sampling time.
SAWANO_TIMES = (0.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 840.0)
SAWANO_CASES = (
    SawanoCase(0, 65, 161, 57, (
        IndexedReading(100, 72, 3.2, 44), IndexedReading(77, 64, 2.3, 36),
        IndexedReading(110, 80, 4.4, 52), IndexedReading(97, 75, 3.1, 41),
        IndexedReading(84, 72, 2.4, 34), IndexedReading(79, 70, 2.2, 32),
        IndexedReading(76, 69, 2.3, 34), IndexedReading(73, 68, 2.3, 35),
    ), (14, 14, 15, 10, 9, 9, 10, 9), (7, 11, 13, 10, 9, 9, 9, 9)),
    SawanoCase(2, 63, 157, 54, (
        IndexedReading(98, 75, 3.1, 42), IndexedReading(69, 67, 2.1, 32),
        IndexedReading(88, 78, 3.3, 42), IndexedReading(82, 76, 3.0, 39),
        IndexedReading(72, 72, 2.3, 32), IndexedReading(68, 71, 2.1, 29),
        IndexedReading(64, 68, 2.0, 28), IndexedReading(64, 65, 2.1, 32),
    ), (10, 8, 14, 15, 9, 7, 5, 4), (6, 12, 15, 13, 9, 8, 7, 8)),
    SawanoCase(4, 60, 164, 59, (
        IndexedReading(94, 75, 3.3, 45), IndexedReading(64, 71, 1.9, 28),
        IndexedReading(70, 74, 2.5, 33), IndexedReading(67, 73, 2.4, 32),
        IndexedReading(62, 71, 2.1, 30), IndexedReading(61, 71, 2.0, 29),
        IndexedReading(60, 70, 2.0, 29), IndexedReading(57, 68, 1.9, 29),
    ), (12, 10, 14, 13, 11, 11, 10, 8), (10, 9, 11, 9, 10, 9, 10, 8)),
)


@dataclass(frozen=True)
class SjoenArm:
    remifentanil_factor: float  # Syringe strength relative to the programmed 50 mcg/mL
    age: float
    weight: float
    height: float
    sap_change: float  # Mean relative change, baseline to minimum (%)
    hr_change: float
    propofol_factor: float = 1.0  # Syringe strength relative to the programmed 10 mg/mL
    propofol_target: float = 6.0  # Induction plasma targets, mcg/mL and ng/mL
    remifentanil_target: float = 4.0
    propofol_mg_kg: float = 1.8  # Reported total by 450 s


# Sjøen 2024 Tables 1-2: remifentanil doses giving peak effect-site concentrations of 2, 4, and 8 ng/mL.
SJOEN_ARMS = (
    SjoenArm(0.5, 36, 72, 169, -30.0, -20.0),
    SjoenArm(1.0, 36, 74, 169, -32.0, -21.0),
    SjoenArm(2.0, 35, 67, 166, -32.0, -21.0),
)

# Sjøen 2023 Tables 1-2: the same targets with 10 or 20 mg/mL propofol gave
# 1.4 or 2.7 mg/kg. The reported remifentanil totals (1.9-2.0 mcg/kg, peak
# effect-site concentration near 5 ng/mL) are 69% of Minto delivery to the
# stated 8 ng/mL target, so delivery is scaled to them.
SJOEN_2023_ARMS = (
    SjoenArm(0.69, 35, 74, 166, -31.0, -24.0, propofol_target=5.0, remifentanil_target=8.0, propofol_mg_kg=1.41),
    SjoenArm(0.69, 36, 69, 166, -36.0, -20.0, propofol_factor=2.0, propofol_target=5.0, remifentanil_target=8.0,
             propofol_mg_kg=2.74),
)



@dataclass(frozen=True)
class PotermanArm:
    age: float
    height: float
    weight: float
    map_t0: float
    hr_t0: float
    atropine_mg: float
    map_change: tuple[float, float, float]  # Median and IQR, T0 to T10 (mmHg)
    hr_change: tuple[float, float]  # Mean and SD (bpm)


# Poterman 2017 Tables 1-2: propofol Ce 2.5 mcg/mL and remifentanil Ce 8 ng/mL
# after 500 mL colloid, with methylatropine 0.5 mg or saline at induction.
POTERMAN_ARMS = {
    "saline": PotermanArm(50, 177, 81, 112, 78, 0.0, (-37.0, -41.0, -31.0), (-19.0, 11.0)),
    "atropine": PotermanArm(59, 175, 86, 107, 80, 0.5, (-24.0, -40.0, -21.0), (0.0, 13.0)),
}

class PlasmaTCI:
    """Three-compartment plasma-targeting pump; rates in model mass per second."""

    def __init__(self, v1, k10, k12, k13, k21, k31, max_rate, dt):
        n = 3
        system = np.zeros((n + 1, n + 1))
        system[:n, :n] = np.array([[-(k10 + k12 + k13), k21, k31], [k12, -k21, 0.0], [k13, 0.0, -k31]]) * dt / 60.0
        system[0, n] = dt / v1
        transition = expm(system)
        self.phi, self.gamma = transition[:n, :n], transition[:n, n]
        self.max_rate = max_rate
        self.x = np.zeros(n)

    def step(self, target: float) -> float:
        free = self.phi @ self.x
        rate = min(self.max_rate, max(0.0, (target - free[0]) / self.gamma[0]))
        self.x = free + self.gamma * rate
        return rate


def sjoen_delivery(arm: SjoenArm, switch_s: float, dt: float) -> np.ndarray:
    """Marsh propofol and Minto remifentanil plasma TCI (Sjøen 2023 and 2024 methods).

    Targets 1 mcg/mL and 1 ng/mL for 150 s, then the arm's induction targets;
    propofol falls to 2 when verbal contact is lost. Pumps run at up to
    1200 mL/h of the programmed 10 mg/mL propofol and 50 mcg/mL remifentanil.
    """
    lbm = 1.07 * arm.weight - 148.0 * (arm.weight / arm.height) ** 2
    v1 = 5.1 - 0.0201 * (arm.age - 40) + 0.072 * (lbm - 55)
    v2 = 9.82 - 0.0811 * (arm.age - 40) + 0.108 * (lbm - 55)
    cl1 = 2.6 - 0.0162 * (arm.age - 40) + 0.0191 * (lbm - 55)
    cl2 = 2.05 - 0.0301 * (arm.age - 40)
    cl3 = 0.076 - 0.00113 * (arm.age - 40)
    propofol = PlasmaTCI(0.228 * arm.weight, 0.119, 0.112, 0.0419, 0.055, 0.0033, 1200 / 3600 * 10.0, dt)
    remifentanil = PlasmaTCI(v1, cl1 / v1, cl2 / v1, cl3 / v1, cl2 / v2, cl3 / 5.42, 1200 / 3600 * 50.0, dt)
    rates = []
    for index in range(round(450.0 / dt)):
        time = index * dt
        prop_target = 1.0 if time < 150.0 else (arm.propofol_target if time < 150.0 + switch_s else 2.0)
        remi_target = 1.0 if time < 150.0 else arm.remifentanil_target
        rates.append((arm.propofol_factor * propofol.step(prop_target),
                      arm.remifentanil_factor * remifentanil.step(remi_target)))
    return np.array(rates)


def replay_sjoen(arm: SjoenArm, dt: float = 0.5) -> dict[str, float]:
    """Relative changes from the 55-5 s pre-bolus mean to the 0-300 s minimum (10-s means).

    The time of loss of verbal contact is unreported; the propofol target falls
    when the total reaches the reported dose by 450 s. Spontaneous breathing
    gives way to ventilation 60 s after the bolus. Lithotomy positioning and
    the laryngeal mask are not replayed.
    """
    switch_s = next(float(s) for s in range(5, 300)
                    if sjoen_delivery(arm, s, dt)[:, 0].sum() * dt / arm.weight >= arm.propofol_mg_kg)
    rates = sjoen_delivery(arm, switch_s, dt)
    engine = SimulationEngine(Patient(age=arm.age, weight=arm.weight, height=arm.height, sex="female",
                                      baseline_map=(130.0 + 2.0 * 75.0) / 3.0, baseline_hr=68.0),
                              SimulationConfig(mode="awake", rng_seed=4))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    samples = []
    for index, (propofol, remifentanil) in enumerate(rates):
        if index * dt == 210.0:
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(rr=12, vt=0.5, peep=0.0, ie="1:2", mode="VCV")
            engine.set_vent_power(True)
        engine.infusion_rates["propofol"], engine.infusion_rates["remi"] = propofol, remifentanil
        engine.step(dt)
        state = engine.state
        samples.append((state.sbp, state.hr, state.co, state.svr))
    trace = np.array(samples)
    window = round(10.0 / dt)
    smooth = np.array([trace[max(0, i - window + 1):i + 1].mean(axis=0) for i in range(len(trace))])
    time = (np.arange(len(trace)) + 1) * dt
    baseline = smooth[(time >= 95.0) & (time <= 145.0)].mean(axis=0)
    change = 100.0 * (smooth[(time > 150.0) & (time <= 450.0)].min(axis=0) / baseline - 1.0)
    return dict(sap=change[0], hr=change[1], co=change[2], svr=change[3],
                propofol_mg_kg=float(rates[:, 0].sum() * dt / arm.weight),
                remifentanil_mcg_kg=float(rates[:, 1].sum() * dt / arm.weight))


def replay_billard(case: BillardCase, dt: float = 0.2) -> tuple[Reading, Reading]:
    """Replay doses and timing through production PK, stimulus attenuation,
    circulation, and SBP, with blood gases fixed.

    Height was not reported; assume 165 cm and female sex for the mostly
    female cohorts. Initial MAP reproduces baseline SBP. Eleveld and Bae PK
    stand in for the study's predicted concentrations; fluid preload, manual
    ventilation, and vecuronium are omitted.
    """
    patient = Patient(age=case.age, weight=case.weight, height=165, sex="female",
                      baseline_hr=case.baseline_hr)
    hemo = HemodynamicModel(patient)
    renderer = ArterialWaveformRenderer(age=case.age)
    cycle = CardiacCycle(np.random.default_rng(0))
    initial = hemo.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
    initial_sbp = renderer.step(cycle.seed(initial.hr, RhythmType.SINUS), initial.map, initial.sv).systolic
    patient.baseline_map += case.baseline_sbp - initial_sbp
    hemo = HemodynamicModel(patient)
    initial = hemo.step(0.0, 0.0, 0.0, 0.0, -2.0, 40.0, 95.0)
    renderer = ArterialWaveformRenderer(age=case.age)
    cycle = CardiacCycle(np.random.default_rng(0))
    renderer.step(cycle.seed(initial.hr, RhythmType.SINUS), initial.map, initial.sv)
    propofol = PropofolPKEleveld(patient, concomitant_opioids=case.fentanyl > 0)
    fentanyl = FentanylPK(patient)
    response = ClinicalResponseModel(patient.age)
    fentanyl.state.c1 = case.fentanyl * case.weight / fentanyl.v1
    # Fentanyl, 5 min, propofol at 5 mg/s, then 4 min after its injection ends.
    end_injection = 300.0 + case.propofol * case.weight / 5.0
    intubation = end_injection + 240.0
    stimulus = Disturbances("stim_intubation_pulse")
    assert stimulus.spec is not None
    intensity = stimulus.spec.sympathetic_intensity
    before, after = [], []
    for index in range(round((intubation + 65.0) / dt)):
        time = index * dt
        injection = max(0.0, min(time + dt, end_injection) - max(time, 300.0))
        prop = propofol.step(dt, 5.0 * injection / dt)
        fent = fentanyl.step(dt, 0.0)
        effects = stimulus.compute_average(time - intubation, time + dt - intubation)
        pressure_fraction, hr_fraction = response.cardiovascular_response_fractions(
            prop.ce_response, 0.0, intensity=intensity, ce_fentanyl=fent.ce,
        )
        circulation = hemo.step(
            dt, prop.c1, opioid_equivalent(0.0, fent.ce), 0.0, -2.0, 40.0, 95.0,
            dist_hr=effects.hr * hr_fraction,
            dist_svr=effects.svr * pressure_fraction, dist_sv=effects.sv * pressure_fraction,
            remifentanil_cp=0.0,
        )
        pressure = renderer.step(
            cycle.step(dt, circulation.hr, circulation.rhythm_type), circulation.map, circulation.sv,
        )
        reading = (pressure.systolic, circulation.hr, circulation.map, circulation.co, circulation.svr)
        if intubation - 10.0 <= time < intubation:
            before.append(reading)
        if intubation + 55.0 <= time < intubation + 65.0:
            after.append(reading)
    return Reading(*np.mean(before, axis=0)), Reading(*np.mean(after, axis=0))



def replay_poterman(arm: PotermanArm, dt: float = 1.0) -> tuple[float, float]:
    """MAP and HR change from induction (T0) to 10 min in Poterman 2017.

    Eleveld effect-site TCI stands in for the study's Schnider and Minto
    targets, and atropine for methylatropine. Sex is unreported by arm; assume
    male. Intubation after cisatracurium is placed at 4 min; ventilation
    starts at 90 s.
    """
    engine = SimulationEngine(Patient(age=arm.age, height=arm.height, weight=arm.weight, sex="male",
                                      baseline_map=arm.map_t0, baseline_hr=arm.hr_t0),
                              SimulationConfig(mode="awake", tci_enabled=True, rng_seed=1))
    engine.start()
    engine.set_airway_mode("Mask")
    engine.set_fgf(6.0, 0.0)
    engine.give_albumin(500.0)
    for _ in range(round(600.0 / dt)):  # Preload completed before induction.
        engine.step(dt)
    map_t0, hr_t0 = engine.state.map, engine.state.hr
    if arm.atropine_mg:
        engine.give_drug_bolus("atropine", arm.atropine_mg)
    engine.enable_tci("propofol", 2.5)
    engine.enable_tci("remi", 8.0)
    for index in range(round(600.0 / dt)):
        time = index * dt
        if time == 90.0:
            engine.set_airway_mode("ETT")
            engine.set_vent_settings(rr=12, vt=0.008 * arm.weight, peep=5.0, ie="1:2", mode="VCV")
            engine.set_vent_power(True)
        if time == 240.0:
            engine.start_disturbance("stim_intubation_pulse")
        engine.step(dt)
    return engine.state.map - map_t0, engine.state.hr - hr_t0

def replay_sawano(case: SawanoCase, loading_dose_mg_kg: float = 0.0) -> tuple[IndexedReading, ...]:
    """Full-engine replay of the normotensive cohorts in Sawano 2013.

    Plasma-targeted Eleveld TCI stands in for the unspecified study TCI model.
    Sex and fluid rate were unreported; assume male and no fluid. Ventilation
    at 12/min, 500 mL, PEEP 0 runs from baseline in place of manual ventilation
    after induction.

    `loading_dose_mg_kg` adds an assumed initial bolus before the same plasma
    target, to test sensitivity to the unknown loading history.
    """
    patient = Patient(age=case.age, height=case.height, weight=case.weight,
                      sex="male", baseline_hr=case.observed[0].hr,
                      baseline_map=case.observed[0].map)
    engine = SimulationEngine(patient, SimulationConfig(
        mode="awake", tci_enabled=True, concomitant_opioids=case.fentanyl > 0,
        maintenance_fluid_ml_hr=0.0, rng_seed=4,
    ))
    engine.start()
    engine.set_airway_mode("ETT")
    engine.set_vent_settings(rr=12, vt=0.5, peep=0.0, ie="1:2", mode="VCV")
    engine.set_vent_power(True)
    engine.set_fgf(6.0, 0.0)
    if loading_dose_mg_kg:
        engine.give_drug_bolus("propofol", loading_dose_mg_kg * patient.weight)
    engine.enable_tci("propofol", 4.0, mode="plasma")

    def reading() -> IndexedReading:
        state = engine.state
        return IndexedReading(state.map, state.hr, state.co / patient.bsa,
                              1000.0 * state.co / state.hr / patient.bsa)

    readings = [reading()]
    rocuronium_given = False
    dt = 0.5
    for index in range(round(SAWANO_TIMES[-1] / dt)):
        time = index * dt
        if not rocuronium_given and engine.state.loc >= 0.5:
            engine.give_drug_bolus("roc", 0.9 * patient.weight)
            rocuronium_given = True
        if case.fentanyl and 120.0 <= time < 125.0:
            engine.give_drug_bolus("fentanyl", case.fentanyl * patient.weight * dt / 5.0)
        if time == 240.0:
            engine.start_disturbance("stim_intubation_pulse")
            engine.set_fgf(3.0, 3.0)
        engine.step(dt)
        if time + dt in SAWANO_TIMES[1:]:
            readings.append(reading())
    return tuple(readings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Challenge Sawano's unknown loading history with initial propofol boluses.")
    args = parser.parse_args()
    print("Billard 1994: typical-patient predictions / observed means (10 patients per cohort).")
    print("Fentanyl mcg/kg; propofol mg/kg; pressures mmHg; HR beats/min.")
    print("Billard did not report MAP, CO, or SVR; these predictions are not validation targets.")
    print(" F   P   SBP before     SBP after     SBP rise       HR rise     Model MAP rise")
    results = []
    for case in BILLARD_CASES:
        before, after = replay_billard(case)
        results.append((case, before, after))
        print(f"{case.fentanyl:2.0f} {case.propofol:3.1f}"
              f" {before.sbp:6.1f}/{case.pre_sbp:5.1f} {after.sbp:6.1f}/{case.post_sbp:5.1f}"
              f" {after.sbp - before.sbp:6.1f}/{case.post_sbp - case.pre_sbp:5.1f}"
              f" {after.hr - before.hr:6.1f}/{case.post_hr - case.pre_hr:5.1f}"
              f" {after.map - before.map:9.1f}", flush=True)
    print("\nMean signed errors: model minus observed.")
    print(" F   SBP before   SBP after   SBP rise   HR rise")
    for dose in (0, 2, 4):
        errors = [(before.sbp - case.pre_sbp, after.sbp - case.post_sbp,
                   (after.sbp - before.sbp) - (case.post_sbp - case.pre_sbp),
                   (after.hr - before.hr) - (case.post_hr - case.pre_hr))
                  for case, before, after in results if case.fentanyl == dose]
        print(f"{dose:2.0f}" + "".join(f"{error:12.1f}" for error in np.mean(errors, axis=0)))
    print("\nModel predictions only, before / after intubation.")
    print(" F   P       MAP mmHg          CO L/min         SVR Wood units")
    for case, before, after in results:
        print(f"{case.fentanyl:2.0f} {case.propofol:3.1f}"
              f" {before.map:8.1f}/{after.map:6.1f}"
              f" {before.co:10.2f}/{after.co:5.2f}"
              f" {before.svr:10.2f}/{after.svr:5.2f}")

    print("\nSjøen 2024: relative change from pre-bolus baseline to minimum, model / observed (%).")
    print("CO and SVR come from pulse contour in the study and are shown as model values only.")
    print(" Remi   SAP          HR           CO     SVR    Propofol mg/kg  Remi mcg/kg")
    for arm in SJOEN_ARMS:
        result = replay_sjoen(arm)
        print(f" x{arm.remifentanil_factor:<4} {result['sap']:5.0f}/{arm.sap_change:3.0f}  "
              f"{result['hr']:5.0f}/{arm.hr_change:3.0f}  {result['co']:6.0f} {result['svr']:6.0f}"
              f"  {result['propofol_mg_kg']:12.2f}  {result['remifentanil_mcg_kg']:10.2f}", flush=True)

    print("\nSjøen 2023: propofol dose arms, same measures; observed CO -35/-32% and SVR -20/-31%.")
    print(" Prop   SAP          HR           CO     SVR    Propofol mg/kg  Remi mcg/kg")
    for arm in SJOEN_2023_ARMS:
        result = replay_sjoen(arm)
        print(f" x{arm.propofol_factor:<4} {result['sap']:5.0f}/{arm.sap_change:3.0f}  "
              f"{result['hr']:5.0f}/{arm.hr_change:3.0f}  {result['co']:6.0f} {result['svr']:6.0f}"
              f"  {result['propofol_mg_kg']:12.2f}  {result['remifentanil_mcg_kg']:10.2f}", flush=True)

    print("\nPoterman 2017: change from induction to 10 min, model / observed (MAP median, HR mean).")
    for name, group in POTERMAN_ARMS.items():
        map_change, hr_change = replay_poterman(group)
        print(f" {name:8} MAP {map_change:5.0f}/{group.map_change[0]:3.0f} mmHg   "
              f"HR {hr_change:5.0f}/{group.hr_change[0]:3.0f} bpm", flush=True)

    print("\nSawano 2013: normotensive cohorts, typical-patient predictions / observed means.")
    print("Propofol plasma TCI 4 mcg/mL; study PK model and infusion histories unreported.")
    print("CI L/min/m2; SVI mL/m2. Point times exclude the undefined immediate peak.")
    print(" F  Time relative to intubation     MAP mmHg       HR bpm       CI            SVI")
    sawano_results = []
    for cohort in SAWANO_CASES:
        predicted = replay_sawano(cohort)
        sawano_results.append((cohort, predicted))
        for time, model, observed in zip(SAWANO_TIMES, predicted, cohort.observed, strict=True):
            if time == 0.0:
                label = "awake"
            elif time == 240.0:
                label = "before intubation"
            else:
                label = f"{(time - 240.0) / 60.0:g} min"
            print(f"{cohort.fentanyl:2.0f} {label:>28}"
                  f" {model.map:7.1f}/{observed.map:5.1f} {model.hr:6.1f}/{observed.hr:5.1f}"
                  f" {model.ci:6.2f}/{observed.ci:4.2f} {model.svi:7.1f}/{observed.svi:5.1f}",
                  flush=True)

    print("\nSawano late HR: 3 to 10 min after intubation, bpm.")
    print(" F    HR change model/observed    HR at 10 min model/observed    Error   Late HR within SD")
    for cohort, predicted in sawano_results:
        hr_change = predicted[-1].hr - predicted[4].hr
        observed_hr_change = cohort.observed[-1].hr - cohort.observed[4].hr
        hr_error = predicted[-1].hr - cohort.observed[-1].hr
        within_sd = all(abs(model.hr - observed.hr) <= sd for model, observed, sd in zip(
            predicted[4:], cohort.observed[4:], cohort.hr_sd[4:], strict=True,
        ))
        print(f"{cohort.fentanyl:2.0f} {hr_change:13.1f}/{observed_hr_change:5.1f}"
              f" {predicted[-1].hr:24.1f}/{cohort.observed[-1].hr:5.1f} {hr_error:10.1f}"
              f" {'PASS' if within_sd else 'FAIL':>19}")

    if args.sensitivity:
        print("\nSawano loading sensitivity: Eleveld TCI alone or an initial 1/2 mg/kg bolus.")
        print("All scenarios keep the 4 mcg/mL plasma target; the boluses are assumed histories.")
        print("Changes are from immediately before to 1 min after intubation.")
        print(" F       MAP change           HR change          CI change         SVI change")
        print("          model range / observed mean for each endpoint")
        for cohort, nominal in sawano_results:
            scenarios = [nominal, *(replay_sawano(cohort, dose) for dose in (1.0, 2.0))]
            changes = np.array([np.subtract(trace[2], trace[1]) for trace in scenarios])
            observed_change = np.subtract(cohort.observed[2], cohort.observed[1])
            lower, upper = changes.min(axis=0), changes.max(axis=0)
            print(f"{cohort.fentanyl:2.0f}" + "".join(
                f" {low:6.1f}..{high:5.1f}/{observed:5.1f}"
                for low, high, observed in zip(lower, upper, observed_change, strict=True)
            ), flush=True)


if __name__ == "__main__":
    main()
