from dataclasses import dataclass


@dataclass(frozen=True)
class HemodynamicConfig:
    """Centralized parameters for HemodynamicModel."""
    # Core model
    kout: float = 0.072

    # Baseline defaults
    baseline_hb: float = 13.5
    baseline_hct: float = 0.42
    base_hr: float = 56.0
    base_sv: float = 82.2
    base_tpr: float = 0.016
    bsa_fallback: float = 1.9
    ci_adult: float = 3.0
    ci_elderly: float = 2.5
    ci_elderly_age: float = 70.0

    # Arterial compliance
    arterial_compliance_ref: float = 1.5
    arterial_age_ref: float = 40.0
    arterial_age_slope: float = 0.008
    arterial_age_min: float = 0.5
    arterial_age_max: float = 1.2

    # Feedback
    fb: float = -0.66
    hr_sv_coupling: float = 0.312
    k_drift: float = 0.067

    # HR response smoothing
    tau_hr_fast: float = 5.0

    # Vasopressor SV factor (combined inotropy)
    vasopressor_sv_factor: float = 1.0

    # Propofol hemodynamic effects
    ec50_prop_tpr: float = 3.21
    emax_prop_tpr_clinical: float = -0.50
    emax_prop_tpr_literature: float = -0.78
    gamma_prop: float = 1.83
    ec50_prop_sv: float = 0.44
    emax_prop_sv_typ: float = -0.15
    age_emax_sv: float = 0.033

    # Remifentanil hemodynamic effects
    ec50_remi_tpr: float = 4.59
    emax_remi_tpr: float = -1.0
    gamma_remi_tpr: float = 1.0
    sl_remi_hr: float = 0.033
    sl_remi_sv: float = 0.058

    # Propofol-Remifentanil interaction
    int_tpr: float = 1.00
    int_hr: float = -0.12
    ec50_int_hr: float = 0.20
    int_sv: float = -0.21

    # Norepinephrine PD
    nore_c50: float = 7.04
    nore_gamma: float = 1.8
    nore_emax_map: float = 98.7
    nore_emax_hr: float = 10.0
    nore_emax_sv: float = 0.15

    # Volatile parameters
    ke0_sevo: float = 0.25
    sevo_emax_tpr: float = -0.45
    sevo_ec50_tpr: float = 1.0
    sevo_gamma_tpr: float = 1.5
    sevo_emax_sv: float = -0.15
    sevo_ec50_sv: float = 1.0
    vol_remi_shift_max: float = 0.6
    vol_remi_ec50: float = 2.0

    # Blood volume / preload tracking
    default_blood_volume: float = 5000.0
    unstressed_volume_fraction: float = 0.70
    venous_compliance: float = 100.0
    mcfp_floor: float = 1.0
    vol_clearance: float = 1.0
    crystalloid_retention_fraction: float = 0.30
    blood_retention_fraction: float = 1.0

    # Intrathoracic pressure coupling
    pit_0: float = -2.0
    alpha_peep: float = 0.04
    f_preload_pit: float = 1.0

    # Right heart / pulmonary circulation (lightweight coupling)
    # ESC/ERS RHC normal ranges: RAP 2–6 mmHg, PVR 0.3–2.0 WU.
    rap_baseline: float = 5.0
    # Baseline pulmonary vascular resistance (Wood units).
    # ESC/ERS normal: 0.3–2.0 WU.
    pvr_wood_baseline: float = 1.2
    # Pulmonary transit time (seconds). CMR median ~6.8 s in preserved EF cohort.
    pulmonary_transit_time_s: float = 6.8
    # Hypoxic pulmonary vasoconstriction (HPV) effect on PVR.
    pvr_o2_threshold: float = 60.0  # mmHg: below this, HPV begins to rise
    pvr_o2_floor: float = 30.0      # mmHg: severe hypoxia reference
    pvr_o2_max_factor: float = 2.0  # Max PVR multiplier at/under floor
    # PEEP effect on PVR (cmH2O above baseline).
    pvr_peep_ref: float = 5.0       # cmH2O baseline PEEP
    pvr_peep_slope: float = 0.03    # Fractional PVR increase per cmH2O above ref
    # Flow sensitivity to PVR (higher = more RV afterload sensitivity).
    pvr_flow_exponent: float = 0.7
    # Safety clamp
    pvr_max_factor: float = 3.0

    # Chemoreflex setpoints/gains
    paco2_set: float = 40.0
    pao2_set: float = 85.0
    g_hr_co2: float = 15.0
    g_hr_o2: float = 15.0
    k_tpr_co2: float = 0.3

    # Hemorrhage response defaults
    hemorrhage_hr_mult: float = 1.0
    hemorrhage_tpr_mult: float = 1.0

    # Sepsis / distributive shock
    # SIRS tachycardia threshold HR >90 bpm -> ~+20 bpm vs typical baseline 70 (Bone 1992).
    sepsis_hr_increase: float = 20.0          # bpm increase at full severity
    # Hyperdynamic septic shock often shows SVR <= 600 dyn·s/cm^5; targets 700-800 (Martin 1990).
    sepsis_svr_drop_wood: float = 6.0         # SVR drop (Wood units) at full severity
    # Severe vasoplegia can bring SVR toward ~400-500 dyn·s/cm^5 (~50-60% of normal) (Melo 1999).
    sepsis_tpr_floor: float = 0.55            # Min fraction of baseline TPR production
    # Pressor hyporesponsiveness: phenylephrine Emax ~39 vs 84 mmHg in controls (~50%) (Bellissant 2000).
    sepsis_pressor_resistance: float = 0.50   # Fractional blunting of pressor SVR effects
    # Capillary leak: TER of albumin ~6.7%/h in septic shock (Margarson 2002).
    sepsis_leak_fraction_per_hr: float = 0.067 # Fraction of blood volume lost per hour
    # Venous pooling: lower mean systemic pressure suggests ~20% stressed volume loss (Persichini 2012).
    sepsis_pooling_fraction: float = 0.06     # Fraction of baseline volume shifted to unstressed

    # Anaphylaxis (acute distributive shock)
    anaphylaxis_svr_drop_wood: float = 10.0   # Legacy heuristic (matches prior -10 Wood units)

    # Epinephrine PD
    epi_c50: float = 1.5
    epi_gamma: float = 2.5
    epi_emax_hr: float = 40.0
    epi_emax_sv: float = 0.25
    epi_emax_svr_low: float = -0.20
    epi_emax_svr_high: float = 0.40
    epi_threshold: float = 2.0

    # Phenylephrine PD
    phenyl_c50: float = 15.0
    phenyl_gamma: float = 1.5
    phenyl_emax_svr: float = 0.60

    # Vasopressin PD (units: mU/L)
    # Derived from clinical dosing (0.01–0.07 U/min) and DailyMed label PK.
    vaso_c50: float = 20.0
    vaso_gamma: float = 1.5
    vaso_emax_svr: float = 0.60
    vaso_emax_hr: float = -6.0

    # Dobutamine PD (units: ng/mL)
    # C50 anchored to plasma concentrations during DSE (≈27–403 ng/mL; Daly et al. 1997).
    dobu_c50: float = 80.0
    dobu_gamma: float = 1.3
    dobu_emax_hr: float = 15.0
    dobu_emax_sv: float = 0.35
    dobu_emax_svr: float = -0.25

    # Milrinone PD (units: ng/mL)
    # C50 anchored to therapeutic plasma range (~100–300 ng/mL; DailyMed label).
    mil_c50: float = 150.0
    mil_gamma: float = 1.2
    mil_emax_hr: float = 6.0
    mil_emax_sv: float = 0.45
    mil_emax_svr: float = -0.35

    # Hill cache tolerance
    cache_tolerance: float = 0.01
