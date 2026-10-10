from dataclasses import dataclass


@dataclass(frozen=True)
class HemodynamicConfig:
    """Parameters for HemodynamicModel. Sources are in docs/REFERENCES.md."""

    # Su et al. 2023 turnover model.
    kout: float = 0.072

    ci_adult: float = 3.0
    ci_elderly: float = 2.5
    ci_elderly_age: float = 70.0

    # Signed RMAP exponent. Su et al. report FB = 0.66 and apply -FB.
    fb: float = -0.66
    hr_sv_coupling: float = 0.312
    k_drift: float = 0.067
    tau_hr_fast: float = 5.0  # s

    # Fast baroreflex gains (bpm/mmHg) around a slowly resetting MAP set point.
    # Propofol depresses both limbs 65-73% (Sato 2005; Ebert 1994 found mainly
    # the tachycardic limb) and sevoflurane depresses cardiovagal gain (Umehara
    # 2006); one shared depression applies to both.
    baro_gain_brady: float = 0.8
    baro_gain_tachy: float = 0.5
    baro_anesthetic_depression: float = 0.7
    baro_max_hr_change: float = 35.0
    baro_reset_tau_s: float = 1800.0

    # Myocardial hypoxia: 83% of emergency-intubation arrests were associated
    # with SpO2 < 70% (Mort 2004). Depression grows from SaO2 70% to full at 30%.
    hypoxia_sao2_onset: float = 70.0
    hypoxia_sao2_full: float = 30.0
    hypoxia_tau_on_s: float = 45.0
    hypoxia_tau_off_s: float = 60.0
    hypoxia_hr_depression: float = 0.75
    hypoxia_sv_depression: float = 0.85

    # Propofol, remifentanil, and their interaction (Su et al. 2023).
    ec50_prop_tpr: float = 3.21
    emax_prop_tpr: float = -0.778
    gamma_prop: float = 1.83
    ec50_prop_sv: float = 0.44
    emax_prop_sv_typ: float = -0.15
    age_emax_sv: float = 0.033
    ec50_remi_tpr: float = 4.59
    emax_remi_tpr: float = -1.0
    gamma_remi_tpr: float = 1.0
    sl_remi_hr: float = 0.033
    sl_remi_sv: float = 0.058
    int_tpr: float = 1.00
    int_hr: float = -0.12
    ec50_int_hr: float = 0.20
    int_sv: float = -0.21

    # Integrated Li PK / Su circulation calibration: de Keijzer 2026 infusion
    # responses and Joachim 2024 bolus responses. These are simulation
    # coefficients, not either paper's standalone MAP-model parameters.
    nore_c50: float = 25.0
    nore_emax_svr: float = 1.75
    nore_emax_hr: float = 60.0
    nore_emax_sv: float = 0.7
    nore_anesthetic_alpha_gain: float = 0.6
    nore_anesthetic_sv_gain: float = 4.0

    # Sevoflurane, with remifentanil shifting the TPR EC50.
    ke0_sevo: float = 0.25
    sevo_emax_tpr: float = -0.45
    sevo_ec50_tpr: float = 1.0
    sevo_gamma_tpr: float = 1.5
    sevo_emax_sv: float = -0.15
    sevo_ec50_sv: float = 1.0
    vol_remi_shift_max: float = 0.6
    vol_remi_ec50: float = 2.0

    # Blood volume, preload, and fluid balance.
    unstressed_volume_fraction: float = 0.70
    venous_compliance: float = 100.0  # mL/mmHg
    mcfp_floor: float = 1.0
    # Reflex venoconstriction returns this share of a blood-volume deficit to
    # stressed volume, about 8 mL/kg at a 40% loss. Calibrated so slow losses
    # follow the ATLS hemorrhage classes.
    venous_recruitment_fraction: float = 0.3
    # Fixed urine output (mL/min); overrides uop_ml_kg_hr when set.
    vol_clearance: float | None = None
    # Urine output (mL/kg/hr), scaled by MAP and renal function.
    uop_ml_kg_hr: float = 0.5
    # Urine (mL/min) per unit fractional blood-volume expansion, awake (fit to
    # Reid 2003); anesthesia cuts it to 10-20% (Hahn 2010).
    volume_clearance_ml_min: float = 30.0
    anesthetic_clearance_reduction: float = 0.7
    crystalloid_retention_fraction: float = 0.30
    colloid_retention_fraction: float = 0.80
    blood_retention_fraction: float = 1.0
    # Teaching pressure-response curve, not an AKI or GFR threshold. Preserve
    # most basal output at MAP 60; taper to zero near circulatory collapse.
    renal_map_min: float = 20.0
    renal_map_norm: float = 65.0
    # Interstitial fluid returns only while blood volume is below its 30% share.
    third_space_refill_tau_hr: float = 6.0

    # LAP (mmHg) = A (exp(k x preload) - 1): 9 at baseline filling, 20 at double.
    lap_scale: float = 40.9
    lap_preload_gain: float = 0.2
    # Lung water filters above LAP 20 mmHg (mL/kg PBW/min per mmHg).
    lung_water_lap_threshold: float = 20.0
    lung_water_filtration: float = 0.02
    lung_water_clearance_tau_hr: float = 6.0

    # Intrathoracic pressure (mmHg) and preload.
    pit_0: float = -2.0
    alpha_peep: float = 0.04

    # Right heart and pulmonary circulation. Normal RAP 2-6 mmHg,
    # PVR 0.3-2.0 Wood units. Pulmonary transit median 6.8 s (Segeroth 2023).
    rap_baseline: float = 5.0
    pvr_wood_baseline: float = 1.2
    pulmonary_transit_time_s: float = 6.8
    # Hypoxic vasoconstriction raises PVR below PaO2 60 mmHg, up to 2x at 30.
    pvr_o2_threshold: float = 60.0
    pvr_o2_floor: float = 30.0
    pvr_o2_max_factor: float = 2.0
    # PVR rises 3% per cmH2O of PEEP above 5.
    pvr_peep_ref: float = 5.0
    pvr_peep_slope: float = 0.03
    pvr_flow_exponent: float = 0.7  # RV output sensitivity to PVR
    pvr_max_factor: float = 3.0

    # Chemoreflex set points (mmHg) and gains.
    paco2_set: float = 40.0
    pao2_set: float = 85.0
    g_hr_co2: float = 15.0
    g_hr_o2: float = 15.0
    k_tpr_co2: float = 0.3

    hemorrhage_hr_mult: float = 1.0
    hemorrhage_tpr_mult: float = 1.0

    # Sepsis at full severity.
    # SIRS HR > 90 bpm, about 20 above a typical baseline (Bone 1992).
    sepsis_hr_increase: float = 20.0  # bpm
    # Hyperdynamic septic shock SVR <= 600 dyn·s/cm^5 (Martin 1990).
    sepsis_svr_drop_wood: float = 6.0
    # Pressor response about halved (Bellissant 2000).
    sepsis_pressor_resistance: float = 0.50
    # Albumin transcapillary escape about 6.7%/h (Margarson 2002).
    sepsis_leak_fraction_per_hr: float = 0.067
    # Venous pooling lowers mean systemic pressure (Persichini 2012).
    sepsis_pooling_fraction: float = 0.06  # Fraction of baseline volume made unstressed

    anaphylaxis_svr_drop_wood: float = 10.0
    # Full severity shifts adrenergic agonist responses 3-fold to the right, so
    # 50-100 mcg epinephrine boluses (ANZAAG and ANZCA 2022) restore pressure
    # briefly without overshoot. Calibrated.
    anaphylaxis_agonist_shift: float = 2.0

    # Epinephrine: exogenous arterial concentrations in ng/mL. Responses and
    # delays are fitted to Freyschuss 1986 infusions and Takahashi 2002 boluses
    # (tests/test_epinephrine.py).
    epi_c50_hr: float = 3.2
    epi_gamma_hr: float = 1.5
    epi_emax_hr: float = 120.0
    epi_c50_sv: float = 2.0
    epi_emax_sv: float = 1.4
    epi_c50_beta2: float = 0.35
    epi_emax_svr_beta: float = -0.55
    epi_c50_alpha: float = 2.5
    epi_gamma_alpha: float = 2.0
    # Saturating alpha response preserves small-bolus pressure gains without
    # multiplying baseline resistance fourfold at larger doses. Calibrated.
    epi_emax_svr_alpha: float = 0.4
    epi_tau_hr_s: float = 18.0
    epi_tau_beta2_s: float = 20.0
    epi_tau_pressor_s: float = 55.0
    # Volatiles also blunt direct chronotropy; propofol does not.
    epi_volatile_hr_depression: float = 0.4

    # Phenylephrine fitted to 100-200 mcg boluses under propofol-remifentanil
    # (MAP +29.5 mmHg, HR -17 bpm; Meng 2011), with an SVR Emax near
    # norepinephrine's (Bellissant 2000).
    phenyl_c50: float = 4.0
    phenyl_gamma: float = 1.5
    phenyl_emax_svr: float = 1.2

    # Vasopressin (mU/L), from 0.01-0.07 U/min dosing and label PK.
    vaso_c50: float = 20.0
    vaso_gamma: float = 1.5
    vaso_emax_svr: float = 0.60
    vaso_emax_hr: float = -6.0

    # Dobutamine (ng/mL); stress-echo plasma levels 27-403 ng/mL (Daly 1997).
    dobu_c50: float = 80.0
    dobu_gamma: float = 1.3
    dobu_emax_hr: float = 15.0
    dobu_emax_sv: float = 0.35
    dobu_emax_svr: float = -0.25

    # Milrinone (ng/mL); therapeutic plasma range 100-300 ng/mL (label).
    mil_c50: float = 150.0
    mil_gamma: float = 1.2
    mil_emax_hr: float = 6.0
    mil_emax_sv: float = 0.45
    mil_emax_svr: float = -0.35

    # Adrenoceptor blockade. Competitive antagonists divide the agonist
    # concentration by the dose ratio 1 + sum(C / Kb), so higher agonist doses
    # overcome the block. Milrinone and vasopressin act on other receptors.
    # Occupancy (1 - 1 / dose ratio) reduces resting sympathetic tone and the
    # reflex, hemorrhage, and stimulation responses.
    # Esmolol (mcg/mL) blocks isoproterenol tachycardia about 50% at 0.3 and
    # 80% at 1 mcg/mL (Sum 1983).
    esmolol_kb: float = 0.3
    # Labetalol (ng/mL) is about 7 times more potent at beta than alpha
    # receptors after IV dosing (Hafsa 2022). 50 mg lowered HR for 3 h but
    # BP for under 30 min (Abernethy 1987).
    labetalol_kb_beta: float = 40.0
    labetalol_kb_alpha: float = 276.0
    # Fractions removed at full block, calibrated for AnaSim. Anesthesia lowers
    # resting sympathetic tone, so the resting fractions fall with depth.
    beta_block_hr: float = 0.30  # Resting HR
    beta_block_sv: float = 0.10
    beta_block_reflex: float = 0.8  # Reflex and stimulation tachycardia
    alpha_block_tpr: float = 0.30
    sympathetic_anesthetic_depression: float = 0.5
    # Slower AV conduction lowers the AF ventricular rate.
    beta_block_af_rate: float = 0.3

    # Glycopyrrolate (ng/mL): 11.8 ng/mL prevented the oculocardiac reflex in
    # 19 of 20 patients, and 3.1 ng/mL was insufficient (Ali-Melkkilä 1993).
    # Vagal block reduces resting vagal tone and baroreflex and opioid
    # bradycardia, and raises the sinus bradycardia rate.
    glyco_c50: float = 4.0
    glyco_gamma: float = 2.0
    glyco_emax_hr: float = 20.0
    glyco_brady_relief: float = 30.0  # bpm added to the 50 bpm sinus bradycardia rate

    # Ketamine (mcg/mL) raises sympathetic outflow: arterial pressure, HR, and
    # CO rose 15-30% during ketamine-nitrous oxide anesthesia at about 2.2
    # mcg/mL (Idvall 1979). Anesthetic depth and adrenoceptor block reduce it.
    ketamine_c50: float = 1.0
    ketamine_gamma: float = 1.5
    ketamine_emax_hr: float = 0.40  # Fraction of HR
    ketamine_emax_sv: float = 0.10
    ketamine_emax_tpr: float = 0.15
