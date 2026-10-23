from dataclasses import dataclass


@dataclass(frozen=True)
class HemodynamicConfig:
    """Parameters for HemodynamicModel. Sources are in docs/REFERENCES.md."""

    # Su et al. 2023 turnover rate, for the slow tone that MAP feedback,
    # chemoreflex, blood loss, and temperature adjust.
    kout: float = 0.072

    cardiac_index: float = 3.0  # L/min/m²

    # Signed RMAP exponent. Su et al. report FB = 0.66 and apply -FB.
    fb: float = -0.66
    hr_sv_coupling: float = 0.312
    k_drift: float = 0.067
    tau_hr_fast: float = 5.0  # s

    # Cardiovagal aging follows Kardos 2001. Reflex/beta HR scaling is calibrated
    # to Laitinen 2004, Bullington 1989, and Leenen 2005; vascular responses
    # are preserved (Monahan 2007).
    age_reference: float = 40.0
    age_sympathetic_hr: float = -0.0085  # Fraction per year
    age_vagal_baroreflex: float = -0.026  # ln gain per year

    # Fast baroreflex gains (bpm/mmHg) around a slowly resetting MAP set point.
    # Propofol depresses both limbs 65-73% (Sato 2005; Ebert 1994 found mainly
    # the tachycardic limb) and sevoflurane depresses cardiovagal gain (Umehara
    # 2006); one shared depression applies to both.
    baro_gain_brady: float = 0.8
    baro_gain_tachy: float = 0.5
    baro_anesthetic_depression: float = 0.7
    # Opioids also suppress reflex tachycardia, but not reflex slowing: HR fell
    # 20% as SAP fell 30% after propofol-remifentanil induction (Sjøen 2024).
    # Remifentanil-equivalent effect-site concentration (ng/mL) giving full
    # depth; calibrated.
    opioid_reflex_ce: float = 3.0
    baro_max_hr_change: float = 35.0
    baro_reset_tau_s: float = 1800.0

    # MAP-to-HR feedback is weak under propofol (Garraza-Obaldia 2024). Share of
    # the slow tachycardic feedback removed at full depth, calibrated to Sawano 2013.
    hr_feedback_anesthetic_depression: float = 1.0

    # Myocardial hypoxia: 83% of emergency-intubation arrests were associated
    # with SpO2 < 70% (Mort 2004). Depression grows from SaO2 70% to full at 30%.
    hypoxia_sao2_onset: float = 70.0
    hypoxia_sao2_full: float = 30.0
    hypoxia_tau_on_s: float = 45.0
    hypoxia_tau_off_s: float = 60.0
    hypoxia_hr_depression: float = 0.75
    hypoxia_sv_depression: float = 0.85

    # Propofol, remifentanil, and their interaction (Su et al. 2023), acting
    # from hemodynamic effect sites over the slow tone. Changes from Su's
    # volunteer estimates follow surgical patients. Propofol's resistance effect
    # is below Su's -0.778, as Fregolent 2025 found in surgical patients, so
    # induction without opioid lowers pressure less than with it (Billard 1994).
    ec50_prop_tpr: float = 3.21
    emax_prop_tpr: float = -0.5
    gamma_prop: float = 1.83
    ec50_prop_sv: float = 0.44
    emax_prop_sv_typ: float = -0.15
    # Kazama 1999 found similar propofol SBP sensitivity from 20 to 69 years
    # after equilibration; Su's 0.033 overstates the older SV effect.
    age_emax_sv: float = 0.015
    ec50_remi_tpr: float = 4.59
    # A quarter of Su's -1.0: Garraza-Obaldia 2024 found no direct
    # remifentanil effect on MAP.
    emax_remi_tpr: float = -0.25
    # Propofol pressure equilibration half-times (min) by age-group midpoint.
    # Young volunteers equilibrated over 10-13 min (Jeleazcov 2015), slower than
    # Kazama 1999's estimates below 60; these fit inductions without opioid
    # (Ebert 1992 and 1995; Billard 1994) and Kazama's older groups. Opioids
    # equilibrate at remifentanil's rate. Opioid effects level off above the
    # saturation: Sjøen 2024 found the same induction falls in SAP and HR with
    # remifentanil doses giving peak Ce of 2, 4, and 8 ng/mL.
    prop_cv_half_time_by_age: tuple[tuple[float, float], ...] = ((30.0, 11.0), (50.0, 10.5), (65.0, 9.5), (77.5, 10.22))
    opioid_cv_half_time: float = 1.2
    # Remifentanil's own vagal bradycardia, beyond the opioid terms below; it is
    # not seen with fentanyl (Sjøen 2024). Share of HR at saturation, which
    # Sjøen's equal falls at 2-8 ng/mL place below clinical doses. 4% matches
    # the HR fall after induction in VitalDB TIVA cases; randomized inductions
    # in younger patients fell further (Sjøen 2023; Poterman 2017).
    remi_vagal_hr: float = 0.04
    remi_vagal_c50: float = 0.5  # ng/mL
    opioid_cv_saturation: float = 2.5  # ng/mL remifentanil equivalent
    gamma_remi_tpr: float = 1.0
    # Half of Su's 0.033. Su's volunteers breathed spontaneously, so the positive
    # slope likely includes hypercapnia, which the chemoreflex models; Sawano
    # 2013 HR and glycopyrrolate responses bound the reduction.
    sl_remi_hr: float = 0.015
    sl_remi_sv: float = 0.058
    # Opioid interactions on resistance, HR, and stroke volume. Fitted so routine
    # propofol-remifentanil induction falls as VitalDB recordings do (median 31%
    # at the lowest reading, 27% before incision), with research inductions at
    # higher doses (Sjøen 2023 and 2024, Poterman 2017, Sawano 2013) within 2 SD
    # of their reported falls.
    int_tpr: float = 0.65
    int_hr: float = -0.09
    ec50_int_hr: float = 0.20
    int_sv: float = -0.2
    # Su shares the direct SV EC50 (0.44); a higher EC50 makes the stroke-volume
    # interaction grow with propofol dose, as the higher-dose trials fall further.
    ec50_int_sv: float = 4.0

    # Integrated Li PK / Su circulation calibration: de Keijzer 2026 infusion
    # responses and Joachim 2024 bolus responses. These are simulation
    # coefficients, not either paper's standalone MAP-model parameters.
    nore_c50: float = 25.0
    nore_emax_svr: float = 1.75
    nore_emax_hr: float = 60.0
    nore_emax_sv: float = 0.7
    nore_anesthetic_alpha_gain: float = 0.45
    nore_anesthetic_sv_gain: float = 3.0

    # Sevoflurane, fitted so routine bolus induction falls as VitalDB recordings
    # do: MAP a median 27% below the pre-induction reading before incision at
    # end-tidal 1.6% and 19% during surgery, regardless of opioid, with HR about
    # 10 bpm lower. Ebert 1995's rested volunteers (MAP -24, -29, and -40% at
    # end-tidal 1, 2, and 3%; HR unchanged) fall further and bound the model
    # within 2 SD. The 14-min equilibration half-time fits the gradual fall over
    # 10 minutes of mask sevoflurane after propofol (Ebert 1995, Fig. 1);
    # propofol's blood-pressure effect also equilibrates several times slower
    # than its BIS effect (Kazama 1999).
    ke0_sevo: float = 0.05
    sevo_emax_tpr: float = -0.6
    sevo_ec50_tpr: float = 0.55
    sevo_gamma_tpr: float = 1.5
    sevo_emax_sv: float = -0.15
    sevo_ec50_sv: float = 1.0
    sevo_emax_hr: float = -0.2
    sevo_ec50_hr: float = 0.5

    # Blood volume, preload, and fluid balance.
    unstressed_volume_fraction: float = 0.70
    venous_compliance: float = 100.0  # mL/mmHg
    mcfp_floor: float = 1.0
    # Reflex venoconstriction returns this share of a blood-volume deficit to
    # stressed volume, about 8 mL/kg at a 40% loss. Calibrated so slow losses
    # follow the ATLS hemorrhage classes.
    venous_recruitment_fraction: float = 0.3
    # de Wit 2016, Table 1: relative venous effects at Cp 3 to 6.5 mcg/mL.
    # Extrapolation below 3 is assumed; exposure is capped at 6.5.
    propofol_venous_pooling_slope: float = 0.0758
    propofol_venous_resistance_slope: float = 0.1007
    # Share of stressed volume pooled as propofol removes sympathetic venous
    # tone, without the fall in venous-return resistance, so induction lowers
    # output and filling pressure (mean systemic filling pressure -32%, Zucker
    # 2022; output-driven falls, Fairfield 1991 and Sjøen 2024). It follows
    # plasma depth with this half-time (min); epinephrine's alpha tone reverses it.
    propofol_sympathetic_pooling: float = 0.3
    propofol_sympathetic_half_time: float = 3.0
    # Propofol's resistance effect and venous pooling scale with pre-induction
    # MAP around the reference, since higher pressure carries more sympathetic
    # tone to withdraw (fractional gain per mmHg). The maximum MAP fall rose
    # 0.44 points per mmHg of baseline in Morley 2010 (n=130, reference = their
    # mean), about 0.33 after removing regression to the mean from a single
    # baseline reading; VitalDB TIVA cases show the same slope.
    propofol_baseline_sensitivity: float = 0.018
    propofol_baseline_map_ref: float = 97.0
    propofol_baseline_gain_range: tuple[float, float] = (0.5, 1.5)
    # Fixed urine output (mL/min); overrides uop_ml_kg_hr when set.
    vol_clearance: float | None = None
    # Urine output (mL/kg/hr), scaled by MAP and renal function.
    uop_ml_kg_hr: float = 0.5
    # Hahn 2017 kinetics with integrated urine normalization to Reid 2003.
    crystalloid_renal_scale: float = 0.20
    colloid_retention_fraction: float = 0.80
    blood_retention_fraction: float = 1.0
    donor_mchc_g_dl: float = 100.0 / 3.0  # Normal donor red cells; Hb = Hct / 0.03.
    # Teaching pressure-response curve, not an AKI or GFR threshold. Preserve
    # most basal output at MAP 60; taper to zero near circulatory collapse.
    renal_map_min: float = 20.0
    renal_map_norm: float = 65.0
    # Interstitial fluid returns only while blood volume is below its 30% share.
    third_space_refill_tau_hr: float = 6.0
    interstitial_refill_fraction: float = 0.30

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
    # About half of mean airway pressure reaches the pleura in normal lungs.
    # The share of inspiratory muscle pressure and its cap (cmH2O) are calibrated.
    paw_transmission: float = 0.54
    effort_transmission: float = 0.30
    effort_pressure_max: float = 20.0

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

    # Sepsis at full severity.
    # Inflammatory chronotropy follows severity directly, outside the slow
    # reflex feedback, so HR exceeds the SIRS 90 bpm threshold (Bone 1992)
    # under propofol-remifentanil. Calibrated.
    sepsis_hr_increase: float = 32.0  # bpm
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
    # delays are calibrated to Freyschuss 1986 infusions, Guinard 1995 awake
    # boluses, and Takahashi 2002 anesthetized boluses (tests/test_epinephrine.py).
    epi_c50_hr: float = 2.2
    epi_gamma_hr: float = 2.0
    epi_emax_hr: float = 97.0
    epi_c50_sv: float = 1.8
    epi_emax_sv: float = 1.35
    epi_c50_beta2: float = 0.35
    epi_emax_svr_beta: float = -0.55
    epi_c50_alpha: float = 2.5
    epi_gamma_alpha: float = 2.0
    # Saturating alpha response preserves small-bolus pressure gains without
    # multiplying baseline resistance fourfold at larger doses. Calibrated.
    # The same alpha occupancy restores propofol's pooled venous volume.
    epi_emax_svr_alpha: float = 0.3
    epi_tau_hr_s: float = 7.0
    epi_tau_beta2_s: float = 15.0
    epi_tau_pressor_s: float = 50.0
    # Volatiles also blunt direct chronotropy; propofol does not.
    epi_volatile_hr_depression: float = 0.4

    # Phenylephrine fitted to 100-200 mcg boluses under propofol-remifentanil
    # (MAP +29.5 mmHg, HR -17 bpm; Meng 2011), with an SVR Emax near
    # norepinephrine's (Bellissant 2000).
    phenyl_c50: float = 4.0
    phenyl_gamma: float = 1.5
    phenyl_emax_svr: float = 1.2
    # Recruitment and afterload calibrated to Rebet 2016 under propofol-
    # remifentanil: CI -19% at normal preload, -3% when preload-dependent.
    phenyl_venous_recruitment_fraction: float = 0.6  # Share of baseline stressed volume
    phenyl_afterload_gain: float = 0.5  # Acute SV loss per relative arterial resistance gain

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

    # Hinderling 1985 II tissue-linked response and Hill slope. Potency and
    # maximum fit awake patients (Horiguchi 2002) and Hinderling's infusions;
    # volatile and opioid anesthesia leave the response intact (Yamaguchi 1988).
    # Propofol (mcg/mL) limits the HR rise but not vagal block (Horiguchi 2002).
    atropine_c50: float = 2.4  # ng/mL
    atropine_gamma: float = 3.4
    atropine_emax_hr: float = 60.0
    atropine_propofol_cp50: float = 1.6

    # Meng 2011 IV peak calibration, checked against Sjøen 2026 induction.
    # Tolerance timing and Hill slope use Persky 2004 oral data. IV tolerance
    # potency and maximum are calibrated.
    # HR combines Persky's sustained slope (fraction of baseline per ng/mL)
    # with a peak response that shares the vascular tolerance.
    ephedrine_c50: float = 25.0
    ephedrine_gamma: float = 1.9
    ephedrine_emax_svr: float = 0.4
    ephedrine_emax_sv: float = 0.2
    ephedrine_hr_slope: float = 0.0014
    ephedrine_emax_hr: float = 0.4
    ephedrine_tolerance_half_min: float = 15.0
    ephedrine_tolerance_c50: float = 20.0
    ephedrine_tolerance_gamma: float = 3.7
    ephedrine_tolerance_max: float = 0.9

    # Ketamine (mcg/mL) raises sympathetic outflow. Without atropine, 2 mg/kg
    # raised HR 33%, cardiac index 29%, and aortic pressure 28%, with SV and SVR
    # unchanged (Tweed 1972). Calibrated through the Kamp PK. Anesthetic depth
    # and adrenoceptor block reduce it. HR rises by a fixed bpm: Bovill 1972
    # found similar 17-18 bpm rises with or without atropine despite different
    # pre-ketamine HR.
    ketamine_c50: float = 2.0
    ketamine_gamma: float = 3.0
    ketamine_emax_hr: float = 26.0  # bpm; calibrated
    ketamine_emax_sv: float = 0.11  # Offsets the rate-related fall in SV
