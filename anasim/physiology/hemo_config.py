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
    # Resting ventriculo-arterial coupling ratio Ea/Ees in normal left ventricles
    # (Asanoi 1989 Group A: 0.46 ± 0.17; Chen 1998: 0.96 ± 0.36; normal range 0.5-1.0 reviewed in Schwartzenberg 2012).
    normal_coupling_ratio: float = 0.70
    k_drift: float = 0.067
    tau_hr_fast: float = 5.0  # s

    # Cardiovagal aging follows Kardos 2001. Reflex/beta HR scaling is calibrated
    # to Laitinen 2004, Bullington 1989, and Leenen 2005; baroreflex vascular
    # control is preserved (Monahan 2007).
    age_reference: float = 40.0
    age_sympathetic_hr: float = -0.0085  # Fraction per year
    age_vagal_baroreflex: float = -0.026  # ln gain per year

    # Fast baroreflex gains around a slowly resetting MAP set point. Propofol
    # depresses both limbs 65-73% (Sato 2005; Ebert and Muzi 1994 found mainly the
    # tachycardic limb) and sevoflurane depresses cardiovagal gain (Umehara
    # 2006); one shared depression applies to both. When muscarinic block or a
    # fixed rhythm prevents sinus slowing, arterial sympathoinhibition lowers
    # endogenous resistance up to baro_vasc_max (Weissler 1957; Tweed 1972).
    baro_gain_brady: float = 0.8
    baro_gain_tachy: float = 0.5
    baro_gain_vasc: float = 0.05  # Fractional TPR withdrawal per mmHg
    baro_vasc_max: float = 0.38
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

    # Adequate coronary supply during CPR for this long restarts beats after
    # primary asystole while energy reserve remains, a teaching calibration.
    asystole_recovery_time_s: float = 30.0

    # Acute IV amiodarone effects, calibrated to Kosinski 1984 and Munoz 1988.
    # Vasodilation follows the rise in concentration during a rapid bolus and
    # fades within minutes. Contractility falls within minutes, which lowers
    # output more when EF is reduced. HR slows and its sympathetic reflexes
    # weaken over 15-30 minutes.
    amiodarone_c50: float = 4.0  # mcg/mL
    amiodarone_hr_reduction: float = 0.165
    amiodarone_sv_reduction: float = 0.055
    amiodarone_ees_reduction: float = 0.255
    amiodarone_beta_reflex_block: float = 0.65
    amiodarone_vasc_c50: float = 3.3  # mcg/mL rise above the adapted level
    amiodarone_vasc_gamma: float = 2.5
    amiodarone_vasc_adaptation_s: float = 90.0
    amiodarone_tpr_reduction: float = 0.24

    # Propofol, remifentanil, and their interaction (Su et al. 2023), acting
    # through hemodynamic effect sites on the slow tone. Departures from Su's
    # volunteer values follow surgical patients: propofol's resistance effect is
    # below Su's -0.778 (Fregolent 2025), so induction without opioid lowers
    # pressure less than with it (Billard 1994).
    ec50_prop_tpr: float = 3.21
    emax_prop_tpr: float = -0.5
    gamma_prop: float = 1.83
    ec50_prop_sv: float = 0.44
    emax_prop_sv_typ: float = -0.15
    # Kazama 1999 found similar propofol SBP sensitivity from 20 to 69 years
    # after equilibration; Su's 0.033 overstates the older SV effect.
    age_emax_sv: float = 0.015
    ec50_remi_tpr: float = 4.59
    # Garraza-Obaldia 2024 found minimal direct remifentanil effect on MAP;
    # modeled here as mild direct vasodilation.
    emax_remi_tpr: float = -0.12
    # Propofol pressure equilibration half-times (min) by age-group midpoint.
    # Young volunteers took 10-13 min (Jeleazcov 2015), longer than Kazama 1999's
    # estimates below 60; these values fit inductions without opioid (Ebert 1992
    # and 1995; Billard 1994) and Kazama's 60-69 group. At 70-85, Kazama's
    # patients equilibrated 1.8 times as slowly as at 20-39 and fell further (SBP
    # IC50 2.09 vs 3.96-4.61 mcg/mL); the potency multiplier is fitted to VitalDB
    # TIVA patients over 70 and Oh 2025. Opioids equilibrate at remifentanil's
    # rate and level off above the saturation concentration, since Sjøen 2024
    # found equal falls in SAP and HR at peak Ce of 2, 4, and 8 ng/mL.
    prop_cv_half_time_by_age: tuple[tuple[float, float], ...] = ((30.0, 11.0), (50.0, 10.5), (65.0, 9.5), (77.5, 19.8))
    prop_cv_potency_by_age: tuple[tuple[float, float], ...] = ((65.0, 1.0), (77.5, 1.5))
    opioid_cv_half_time: float = 1.2
    # Remifentanil's antimuscarinic-reversible vagal bradycardia (Poterman 2017),
    # not applied to fentanyl here. Share of HR at saturation, which Sjøen 2024's
    # equal falls at 2-8 ng/mL place below clinical doses. 4% matches VitalDB TIVA
    # inductions; younger trial patients fell further (Sjøen 2023, 2024; Poterman 2017).
    remi_vagal_hr: float = 0.04
    remi_vagal_c50: float = 0.5  # ng/mL
    opioid_cv_saturation: float = 2.5  # ng/mL remifentanil equivalent
    gamma_remi_tpr: float = 1.0
    # Direct opioid effects without propofol: no stroke-volume rise and
    # modest central vagal bradycardia across awake/balanced/etomidate states.
    sl_remi_hr: float = -0.020
    sl_remi_sv: float = 0.0
    # Opioid interactions on resistance, HR, and stroke volume. Fitted so routine
    # propofol-remifentanil induction falls as VitalDB recordings do (median 31%
    # at the lowest reading, 27% before incision), with research inductions at
    # higher doses (Sjøen 2023 and 2024, Poterman 2017, Sawano 2013) within 2 SD
    # of their reported falls.
    int_tpr: float = 0.65
    int_hr: float = -0.055
    ec50_int_hr: float = 0.20
    int_sv: float = -0.068
    # Su shares the direct SV EC50 (0.44); a higher EC50 makes the stroke-volume
    # interaction grow with propofol dose, as the higher-dose trials fall further.
    ec50_int_sv: float = 4.0

    # Calibrated with Li PK in the Su circulation to de Keijzer 2026 infusion and
    # Joachim 2024 bolus responses, with venous recruitment from Persichini 2012,
    # Hamzaoui 2010, and Carelli 2026; these differ from either paper's MAP model.
    nore_c50: float = 25.0
    nore_emax_svr: float = 1.75
    nore_emax_hr: float = 60.0
    nore_emax_sv: float = 0.7
    nore_anesthetic_alpha_gain: float = 0.45
    nore_anesthetic_sv_gain: float = 2.0
    nore_venous_recruitment_fraction: float = 0.65
    nore_venous_potency_ratio: float = 0.50

    # Sevoflurane, fitted to routine VitalDB bolus inductions: MAP a median 27%
    # below baseline before incision at end-tidal 1.6% and 19% during surgery,
    # regardless of opioid, with HR about 10 bpm lower. Ebert 1995's rested
    # volunteers fell further (MAP -24, -29, and -40% at 1, 2, and 3%; HR
    # unchanged) and bound the model within 2 SD; their gradual fall over 10
    # minutes of mask sevoflurane after propofol (Fig. 1) sets the 14-min
    # half-time. Most of the fall comes from stroke volume: in unstimulated
    # volunteers on controlled ventilation, cardiac index fell 29% at 1 and 1.5
    # MAC while SVR fell 3% and 15% (non-significant until 2 MAC; Malan 1995).
    ke0_sevo: float = 0.05
    sevo_emax_tpr: float = -0.4
    sevo_ec50_tpr: float = 1.05
    sevo_gamma_tpr: float = 2.5
    sevo_emax_sv: float = -0.5
    sevo_ec50_sv: float = 0.6
    sevo_emax_hr: float = -0.14
    sevo_ec50_hr: float = 0.5

    # Blood volume, preload, and fluid balance.
    unstressed_volume_fraction: float = 0.70
    # Non-recruitable anatomical unstressed vascular floor (heart chambers,
    # arteries, pulmonary bed, microcirculation) that neither endogenous
    # venoconstriction nor exogenous pressors can empty during hemorrhage.
    min_unstressed_volume_fraction: float = 0.48
    venous_compliance: float = 100.0  # mL/mmHg
    mcfp_floor: float = 1.0
    # Reflex venoconstriction returns this share of a blood-volume deficit to
    # stressed volume, about 8 mL/kg at a 40% loss. Calibrated so slow losses
    # follow the ATLS hemorrhage classes.
    venous_recruitment_fraction: float = 0.3
    # Extra recruitment for losses above 30%, calibrated to keep slow losses
    # near 45% compensated.
    severe_venous_recruitment_fraction: float = 0.6
    # de Wit 2016, Table 1: relative venous effects at Cp 3 to 6.5 mcg/mL.
    # Extrapolation below 3 is assumed; exposure is capped at 6.5.
    propofol_venous_pooling_slope: float = 0.0758
    propofol_venous_resistance_slope: float = 0.1007
    # Share of stressed volume pooled as propofol removes sympathetic venous
    # tone, without the fall in venous-return resistance, so induction lowers
    # output and filling pressure (mean systemic filling pressure -32%, Zucker
    # 2022; Sjøen 2024; delayed cardiac output decline in Fairfield 1991). It follows
    # plasma depth with this half-time (min); epinephrine's alpha tone reverses it.
    propofol_sympathetic_pooling: float = 0.3
    propofol_sympathetic_half_time: float = 5.0
    propofol_sympathetic_cp: float = 3.0  # Plasma mcg/mL for full depth
    # Propofol's resistance effect and venous pooling scale with pre-induction
    # MAP around the reference (fractional gain per mmHg), since higher pressure
    # reflects more sympathetic tone. The maximum MAP fall rose 0.44 points per
    # mmHg of baseline in Morley 2010 (n=130; the reference is their mean), about
    # 0.33 after correcting for regression to the mean; VitalDB TIVA cases show
    # the same slope.
    propofol_baseline_sensitivity: float = 0.018
    propofol_baseline_map_ref: float = 97.0
    propofol_baseline_gain_range: tuple[float, float] = (0.5, 1.5)
    # Fixed urine output (mL/min); overrides uop_ml_kg_hr when set.
    vol_clearance: float | None = None
    # Urine output (mL/kg/hr), scaled by MAP and renal function.
    uop_ml_kg_hr: float = 0.5
    # Hahn 2017 kinetics scaled to Reid 2003's 6 h urine after 2 L of Hartmann's;
    # gives ~8% 150-min intraoperative excretion (Ewaldsson and Hahn 2005: 7-11%).
    crystalloid_renal_scale: float = 0.28
    colloid_retention_fraction: float = 0.80
    blood_retention_fraction: float = 1.0
    donor_mchc_g_dl: float = 100.0 / 3.0  # Normal donor red cells; Hb = Hct / 0.03.
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
    # reflex feedback, so HR exceeds the SIRS 90 bpm threshold
    # under propofol-remifentanil. Calibrated.
    sepsis_hr_increase: float = 32.0  # bpm
    # Hyperdynamic septic shock SVR <= 600 dyn·s/cm^5 (Martin 1990).
    sepsis_svr_drop_wood: float = 6.0
    # Pressor response about halved (Bellissant and Annane 2000).
    sepsis_pressor_resistance: float = 0.50
    # Albumin transcapillary escape about 6.7%/h (Margarson and Soni 2002).
    sepsis_leak_fraction_per_hr: float = 0.067
    sepsis_pooling_fraction: float = 0.06  # Fraction of baseline volume made unstressed

    anaphylaxis_hr_increase: float = 24.0  # bpm; direct H2/endogenous sympathetic drive
    anaphylaxis_svr_drop_wood: float = 10.0
    # Fisher 1986 inferred plasma escape up to 35% of blood volume within 10 min.
    anaphylaxis_leak_fraction_per_min: float = 0.03
    # Full anaphylaxis blunts catecholamine vascular/cardiac efficacy while
    # right-shifting non-epinephrine agonists (1 + 1.7 = 2.7-fold) without
    # shifting epinephrine rightward into low-dose beta-2 dominance.
    anaphylaxis_pressor_resistance: float = 0.45
    anaphylaxis_agonist_shift: float = 1.7

    # Epinephrine: exogenous arterial concentrations in ng/mL. Responses and
    # delays are calibrated to Freyschuss 1986, Stratton 1985, Ensinger 1992a, 1992b,
    # Marker 1998, and Leenen 2005 infusions, Guinard 1995 awake boluses,
    # and Takahashi 2002 anesthetized boluses (tests/test_epinephrine.py).
    epi_pulse_gain: float = 0.6
    # Venous recruitment uses a separate calibrated potency ratio.
    epi_venous_potency_ratio: float = 0.34
    epi_age_sv: float = 1.3164436381410376
    epi_age_beta2: float = 0.6165682771729425
    epi_age_alpha: float = 0.45
    epi_c50_hr: float = 7.210482964309253
    epi_gamma_hr: float = 1.0413425804848178
    epi_emax_hr: float = 133.64470160862055
    epi_c50_sv: float = 1.3
    epi_emax_sv: float = 0.77
    epi_c50_beta2: float = 1.17
    epi_emax_svr_beta: float = -0.68
    epi_c50_alpha: float = 4.0
    epi_gamma_alpha: float = 4.4
    # Primary arterial alpha acts below the high-concentration component.
    epi_emax_svr_alpha: float = 0.46
    # A gradual alpha response at higher concentrations also supports CPR and
    # crosses over net SVR above baseline around 0.35-0.5 mcg/kg/min.
    epi_c50_alpha_high: float = 13.0
    epi_gamma_alpha_high: float = 3.2
    epi_tau_alpha_high_s: float = 180.0
    epi_emax_svr_alpha_high: float = 1.15
    epi_tau_hr_s: float = 15.515422502151596
    epi_tau_beta2_s: float = 10.000898034308412
    epi_tau_pressor_s: float = 35.01803824066735
    # Volatiles also blunt direct chronotropy; propofol does not.
    epi_volatile_hr_depression: float = 0.4

    # Phenylephrine fitted to 100-200 mcg boluses under propofol-remifentanil
    # (MAP +29.5 mmHg, HR -17 bpm; Meng 2011), with septic pressor resistance
    # from Bellissant and Annane 2000.
    phenyl_c50: float = 4.0
    phenyl_gamma: float = 1.5
    phenyl_emax_svr: float = 1.2
    # Recruitment and afterload calibrated to Rebet 2016 under propofol-
    # remifentanil: CI -19% at normal preload, -3% when preload-dependent.
    phenyl_venous_recruitment_fraction: float = 0.6  # Share of baseline stressed volume
    phenyl_afterload_gain: float = 0.5  # Acute SV loss per relative arterial resistance gain

    # Vasopressin (mU/L), from 0.01-0.07 U/min dosing and label PK.
    vaso_c50: float = 18.0
    vaso_gamma: float = 1.4
    vaso_emax_svr: float = 1.05
    vaso_emax_hr: float = -6.0

    # Dobutamine (ng/mL): Magnani 1977 (2.5-10 mcg/kg/min), Daly 1997 (5-30 mcg/kg/min,
    # 27-403 ng/mL, HR 76->120 bpm), Hernández 2013 (5 mcg/kg/min septic shock).
    dobu_c50: float = 80.0
    dobu_gamma: float = 1.3
    dobu_c50_hr: float = 210.0
    dobu_gamma_hr: float = 2.0
    dobu_emax_hr: float = 68.0
    dobu_emax_sv: float = 0.35
    dobu_emax_svr: float = -0.25

    # Milrinone (ng/mL): Baim 1983 (SVR -35%, CI +53%, MAP -6% in HFrEF); rapid
    # bolus causes vasodilatory hypotension in normal and HFpEF hearts.
    mil_c50: float = 150.0
    mil_gamma: float = 1.2
    mil_emax_hr: float = 10.0
    mil_emax_sv: float = 0.34
    mil_emax_svr: float = -0.52
    mil_venous_pooling_fraction: float = 0.34

    # Adrenoceptor blockade. Competitive antagonists divide the agonist
    # concentration by the dose ratio 1 + sum(C / Kb), so higher agonist doses
    # overcome the block. Milrinone and vasopressin act on other receptors.
    # Occupancy (1 - 1 / dose ratio) reduces resting sympathetic tone and the
    # reflex, hemorrhage, and stimulation responses.
    # Esmolol (mcg/mL) blocks isoproterenol tachycardia about 50% at 0.3 and
    # 80% at 1 mcg/mL (Sum 1983).
    esmolol_kb: float = 0.3
    # Labetalol (ng/mL) is about 7 times more potent at beta than alpha
    # receptors after IV dosing (Richards 1977). 50 mg lowered HR for 3 h but
    # BP for under 30 min (Abernethy 1987).
    labetalol_kb_beta: float = 40.0
    labetalol_kb_alpha: float = 276.0
    # Fractions removed at full block, calibrated for AnaSim. Anesthesia lowers
    # resting sympathetic tone, so the resting fractions fall with depth.
    beta_block_hr: float = 0.30  # Resting HR
    beta_block_sv: float = 0.10
    beta_block_reflex: float = 0.8  # Reflex and stimulation tachycardia
    alpha_block_tpr: float = 0.22
    sympathetic_anesthetic_depression: float = 0.5
    # Slower AV conduction lowers the AF ventricular rate.
    beta_block_af_rate: float = 0.3
    amiodarone_af_rate: float = 0.32

    # Glycopyrrolate (ng/mL): 11.8 ng/mL prevented the oculocardiac reflex in
    # 19 of 20 patients, and 3.1 ng/mL was insufficient (Ali-Melkkilä 1993).
    # Vagal block reduces resting vagal tone and baroreflex and opioid
    # bradycardia, and raises the sinus bradycardia rate.
    glyco_c50: float = 4.0
    glyco_gamma: float = 2.0
    glyco_emax_hr: float = 44.0
    glyco_brady_relief: float = 30.0  # bpm added to the 50 bpm sinus bradycardia rate

    # Hinderling 1985 II tissue-linked response and Hill slope. Potency and
    # maximum fit awake patients (Horiguchi and Nishikawa 2002) and Hinderling's
    # infusions; halothane, enflurane, and opioid anesthesia leave the response
    # largely intact (Yamaguchi 1988). Propofol (mcg/mL) limits the HR rise
    # (Horiguchi and Nishikawa 2002) without reducing modeled vagal block.
    # During unstimulated vagolytic tachycardia, central venous blood briefly
    # supports filling before unloading over several minutes, whereas preload
    # depletion removes that reserve (Weissler 1957; Berry 1959) and adrenergic
    # drive restores stroke volume at fixed rate (Ross 1965).
    atropine_c50: float = 1.80  # ng/mL
    atropine_gamma: float = 2.2
    atropine_emax_hr: float = 58.0
    atropine_propofol_cp50: float = 1.6
    unstim_sv_reservoir: float = 0.28
    unstim_sv_unload: float = 0.18
    unstim_sv_tau_s: float = 150.0

    # Meng 2011 IV peak calibration, checked against Sjøen 2026 induction.
    # Tolerance timing and Hill slope use Persky 2004 oral data. IV tolerance
    # potency and maximum are calibrated.
    # HR combines Persky's sustained slope (fraction of baseline per ng/mL)
    # with a peak response that shares the vascular tolerance.
    ephedrine_c50: float = 72.0
    ephedrine_gamma: float = 1.9
    ephedrine_emax_svr: float = 0.4
    ephedrine_emax_sv: float = 0.2
    ephedrine_hr_slope: float = 0.00050
    ephedrine_emax_hr: float = 0.4
    ephedrine_tolerance_half_min: float = 15.0
    ephedrine_tolerance_c50: float = 64.0
    ephedrine_tolerance_gamma: float = 3.7
    ephedrine_tolerance_max: float = 0.9

    # Ketamine (mcg/mL) raises central sympathetic outflow and depresses
    # baroreflex gain at anesthetic doses (rat data in Akine 2001). Without
    # atropine, 2 mg/kg raised HR 33%, cardiac index 29%, and aortic pressure
    # 28%, with SV and SVR near baseline; during atrial pacing, SV, SVR, and
    # pressure still rose (Tweed 1972). Calibrated through the Kamp PK,
    # including an upward fast-baroreflex shift fitted to the pressor rises
    # after atropine and during pacing (Bovill and Dundee 1972; Tweed 1972).
    # Adrenoceptor block reduces it (Bovill and Dundee 1972; Bålfors 1983), and
    # anesthetic depth attenuates it here. HR rises by a fixed bpm: Bovill and
    # Dundee 1972 found similar 17-18 bpm rises with or without atropine
    # despite different pre-ketamine HR.
    ketamine_c50: float = 2.0
    ketamine_gamma: float = 3.0
    ketamine_emax_hr: float = 24.0  # bpm; calibrated
    ketamine_emax_sv: float = 0.21
    ketamine_emax_svr: float = 0.085
    ketamine_baro_shift: float = 20.0  # mmHg; calibrated
