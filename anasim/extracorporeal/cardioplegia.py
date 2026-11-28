"""CPB cardioplegia protocols (Buckberg 4:1 blood and del Nido 1:4) and cannula pressure-flow.

Protocol compositions, dosing volumes, and redose windows follow Buckberg (1995),
Ikonomidis et al. (1994), Matte & del Nido (2012), and Ad et al. (2018).
Antegrade versus retrograde myocardial distribution and coronary sinus pressure
ceilings follow Partington et al. (1989) and Ikonomidis et al. (1994).
Cannula pressure-flow M-number scaling follows Sinard et al. (1991).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anasim.core.utils import clamp, clamp01
from anasim.patient.domain import bounded_number

CARDIOPLEGIA_PROTOCOLS = ("buckberg", "del_nido")
CARDIOPLEGIA_ROUTES = ("antegrade", "retrograde")
CARDIOPLEGIA_PHASES = ("induction", "maintenance", "hot_shot")

# Sinard 1991 Table 2 & 3 cannula M-numbers (DeltaP = 10^M / 7.161e6 * mu^0.25 * Q^1.75 * (rho/0.9982)^0.75).
ARTERIAL_CANNULA_M: dict[int, float] = {
    15: 3.30,
    17: 3.00,
    19: 2.80,
    20: 2.70,
    22: 2.50,
    24: 2.38,
}
VENOUS_CANNULA_M: dict[int, float] = {
    20: 3.60,
    24: 3.25,
    28: 2.85,
    32: 2.55,
    36: 2.45,
}
DEFAULT_ARTERIAL_CANNULA_FR = 22
DEFAULT_VENOUS_CANNULA_FR = 32
ARTERIAL_LINE_HIGH_MMHG = 300.0
CORONARY_SINUS_HIGH_MMHG = 40.0  # Ikonomidis 1994 safe ceiling <=40 mmHg (50 mmHg max in Partington 1989)
_BLOOD_DENSITY_G_ML = 1.055
_REFERENCE_VISCOSITY_POISE = 0.030  # Whole blood viscosity at Hct ~41%, 37 °C (Sinard 1991)


def cannula_m_number(french: int, table: dict[int, float]) -> float:
    """Interpolate Sinard 1991 M-number for an arterial or venous cannula French size."""
    sizes = sorted(table)
    fr = clamp(float(french), float(sizes[0]), float(sizes[-1]))
    for idx in range(len(sizes) - 1):
        lo, hi = sizes[idx], sizes[idx + 1]
        if lo <= fr <= hi:
            t = (fr - lo) / (hi - lo)
            return table[lo] + t * (table[hi] - table[lo])
    return table[sizes[-1]]


def sinard_pressure_drop_mmhg(
    flow_ml_min: float,
    m_number: float,
    *,
    viscosity_ratio: float = 1.0,
    density_g_ml: float = _BLOOD_DENSITY_G_ML,
) -> float:
    """Pressure drop (mmHg) across an extracorporeal cannula (Sinard 1991 Eq. 4)."""
    q = max(0.0, float(flow_ml_min))
    if q <= 1e-6:
        return 0.0
    mu = max(0.007, _REFERENCE_VISCOSITY_POISE * max(0.25, float(viscosity_ratio)))
    return (
        (10.0 ** float(m_number))
        / 7.161e6
        * (mu ** 0.25)
        * (q ** 1.75)
        * ((density_g_ml / 0.9982) ** 0.75)
    )


def venous_drainage_cannula_scale(venous_fr: int) -> float:
    """Relative CPB gravity venous drainage conductance normalized to a 32 Fr cannula."""
    m_ref = cannula_m_number(DEFAULT_VENOUS_CANNULA_FR, VENOUS_CANNULA_M)
    m_cur = cannula_m_number(venous_fr, VENOUS_CANNULA_M)
    # Conductance scales inversely with (10^M)^(1/1.75) at equal driving head.
    return clamp((10.0 ** ((m_ref - m_cur) / 1.75)), 0.35, 1.35)


@dataclass(frozen=True, slots=True)
class CardioplegiaRecipe:
    """Delivered mixture composition and protection parameters for one protocol + phase."""

    protocol: str
    phase: str
    crystalloid_fraction: float  # 0.20 for Buckberg 4:1 blood; 0.80 for del Nido 1:4
    delivered_k_meq_l: float
    delivered_mg_mmol_l: float
    delivered_na_meq_l: float
    delivered_cl_meq_l: float
    delivered_hco3_meq_l: float
    delivered_lidocaine_mg_l: float
    delivered_mannitol_g_l: float
    default_volume_ml: float
    default_flow_ml_min: float
    default_temp_c: float
    protection_window_s: float  # Cold redose window (s)


CARDIOPLEGIA_RECIPES: dict[tuple[str, str], CardioplegiaRecipe] = {
    # Buckberg 4:1 blood:crystalloid (Buckberg 1995; Ikonomidis 1994; Ad 2018)
    ("buckberg", "induction"): CardioplegiaRecipe(
        protocol="buckberg",
        phase="induction",
        crystalloid_fraction=0.20,
        delivered_k_meq_l=25.0,
        delivered_mg_mmol_l=2.0,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=122.0,
        delivered_hco3_meq_l=25.0,
        delivered_lidocaine_mg_l=0.0,
        delivered_mannitol_g_l=0.0,
        default_volume_ml=1000.0,
        default_flow_ml_min=250.0,
        default_temp_c=8.0,
        protection_window_s=1200.0,  # 20 min cold redose window
    ),
    ("buckberg", "maintenance"): CardioplegiaRecipe(
        protocol="buckberg",
        phase="maintenance",
        crystalloid_fraction=0.20,
        delivered_k_meq_l=10.0,
        delivered_mg_mmol_l=1.5,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=110.0,
        delivered_hco3_meq_l=24.0,
        delivered_lidocaine_mg_l=0.0,
        delivered_mannitol_g_l=0.0,
        default_volume_ml=500.0,
        default_flow_ml_min=250.0,
        default_temp_c=8.0,
        protection_window_s=1200.0,  # 20 min redose window
    ),
    ("buckberg", "hot_shot"): CardioplegiaRecipe(
        protocol="buckberg",
        phase="hot_shot",
        crystalloid_fraction=0.20,
        delivered_k_meq_l=10.0,
        delivered_mg_mmol_l=2.0,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=108.0,
        delivered_hco3_meq_l=26.0,
        delivered_lidocaine_mg_l=0.0,
        delivered_mannitol_g_l=0.0,
        default_volume_ml=500.0,
        default_flow_ml_min=200.0,
        default_temp_c=37.0,
        protection_window_s=600.0,
    ),
    # del Nido 1:4 blood:crystalloid (Matte & del Nido 2012; Ad 2018)
    ("del_nido", "induction"): CardioplegiaRecipe(
        protocol="del_nido",
        phase="induction",
        crystalloid_fraction=0.80,
        delivered_k_meq_l=24.0,
        delivered_mg_mmol_l=7.3,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=110.0,
        delivered_hco3_meq_l=18.0,
        delivered_lidocaine_mg_l=98.0,
        delivered_mannitol_g_l=2.46,
        default_volume_ml=1000.0,
        default_flow_ml_min=250.0,
        default_temp_c=10.0,
        protection_window_s=4500.0,  # 75 min (60-90 min single-dose protection)
    ),
    ("del_nido", "maintenance"): CardioplegiaRecipe(
        protocol="del_nido",
        phase="maintenance",
        crystalloid_fraction=0.80,
        delivered_k_meq_l=24.0,
        delivered_mg_mmol_l=7.3,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=110.0,
        delivered_hco3_meq_l=18.0,
        delivered_lidocaine_mg_l=98.0,
        delivered_mannitol_g_l=2.46,
        default_volume_ml=500.0,
        default_flow_ml_min=250.0,
        default_temp_c=10.0,
        protection_window_s=3600.0,  # 60 min redose window
    ),
    ("del_nido", "hot_shot"): CardioplegiaRecipe(
        protocol="del_nido",
        phase="hot_shot",
        crystalloid_fraction=0.80,
        delivered_k_meq_l=12.0,
        delivered_mg_mmol_l=4.0,
        delivered_na_meq_l=140.0,
        delivered_cl_meq_l=108.0,
        delivered_hco3_meq_l=22.0,
        delivered_lidocaine_mg_l=45.0,
        delivered_mannitol_g_l=1.23,
        default_volume_ml=400.0,
        default_flow_ml_min=200.0,
        default_temp_c=37.0,
        protection_window_s=600.0,
    ),
}


@dataclass(slots=True)
class CardioplegiaStepOutput:
    """Net biochemical, fluid, thermal, and protection increments from one cardioplegia step."""

    delivered_total_ml: float = 0.0
    net_crystalloid_ml: float = 0.0
    net_k_meq: float = 0.0
    net_mg_mmol: float = 0.0
    net_lidocaine_mg: float = 0.0
    infusate_temp_c: float = 10.0
    hot_shot_energy_boost: float = 0.0
    delivery_pressure_mmhg: float = 0.0
    coronary_sinus_pressure_mmhg: float = 0.0
    breakthrough_occurred: bool = False


class CardioplegiaController:
    """Manages Buckberg/del Nido infusion delivery, cannula pressures, and redose protection."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.protocol: str = "buckberg"
        self.route: str = "antegrade"
        self.phase: str = "induction"
        self.active_infusion: bool = False
        self.remaining_ml: float = 0.0
        self.flow_ml_min: float = 250.0
        self.temp_c: float = 8.0
        self.total_volume_ml: float = 0.0
        self.total_crystalloid_ml: float = 0.0
        self.dose_count: int = 0
        self.protection: float = 0.0  # 0-1 myocardial cardioplegic arrest & metabolic protection
        self.myocardial_temp_c: float = 37.0
        self.protection_window_s: float = 1200.0
        self.time_since_dose_s: float = 0.0
        self.manual_arrest_hold: bool = False  # Preserves legacy set_cpb_cardioplegia(True) behavior
        self.breakthrough: bool = False
        self.last_delivery_pressure_mmhg: float = 0.0
        self.last_cs_pressure_mmhg: float = 0.0
        self.last_protocol_used: str = ""

    @staticmethod
    def recipe(protocol: str, phase: str) -> CardioplegiaRecipe:
        p = str(protocol).strip().lower()
        if p not in CARDIOPLEGIA_PROTOCOLS:
            choices = ", ".join(CARDIOPLEGIA_PROTOCOLS)
            raise ValueError(f"Unsupported cardioplegia protocol {protocol!r}; choose one of: {choices}")
        ph = str(phase).strip().lower()
        if ph not in CARDIOPLEGIA_PHASES:
            choices = ", ".join(CARDIOPLEGIA_PHASES)
            raise ValueError(f"Unsupported cardioplegia phase {phase!r}; choose one of: {choices}")
        return CARDIOPLEGIA_RECIPES[(p, ph)]

    def start_dose(
        self,
        protocol: str = "buckberg",
        *,
        route: str = "antegrade",
        phase: str = "induction",
        volume_ml: float | None = None,
        flow_ml_min: float | None = None,
        temp_c: float | None = None,
    ) -> CardioplegiaRecipe:
        """Start an antegrade or retrograde Buckberg or del Nido cardioplegia dose."""
        rec = self.recipe(protocol, phase)
        norm_route = str(route).strip().lower()
        if norm_route not in CARDIOPLEGIA_ROUTES:
            choices = ", ".join(CARDIOPLEGIA_ROUTES)
            raise ValueError(f"Unsupported cardioplegia route {route!r}; choose one of: {choices}")
        vol = rec.default_volume_ml if volume_ml is None else bounded_number(
            "cardioplegia volume", volume_ml, 50.0, 2000.0, unit="mL",
        )
        flow = rec.default_flow_ml_min if flow_ml_min is None else bounded_number(
            "cardioplegia flow", flow_ml_min, 50.0, 500.0, unit="mL/min",
        )
        temp = rec.default_temp_c if temp_c is None else bounded_number(
            "cardioplegia temperature", temp_c, 4.0, 38.0, unit="°C",
        )
        self.protocol = rec.protocol
        self.last_protocol_used = rec.protocol
        self.route = norm_route
        self.phase = rec.phase
        self.remaining_ml = vol
        self.flow_ml_min = flow
        self.temp_c = temp
        self.active_infusion = True
        self.manual_arrest_hold = False
        self.breakthrough = False
        self.dose_count += 1
        self.protection_window_s = rec.protection_window_s
        return rec

    def stop_infusion(self) -> None:
        self.active_infusion = False
        self.remaining_ml = 0.0
        self.last_delivery_pressure_mmhg = 0.0
        self.last_cs_pressure_mmhg = 0.0

    def set_manual_hold(self, arrested: bool) -> None:
        """Legacy toggle support: holds arrest protection steady until unclamped or stopped."""
        if arrested:
            self.manual_arrest_hold = True
            self.protection = 1.0
            self.breakthrough = False
        else:
            self.manual_arrest_hold = False
            self.stop_infusion()
            self.protection = 0.0
            self.breakthrough = False

    def step(
        self,
        dt: float,
        *,
        cross_clamped: bool,
        core_temp_c: float,
        patient_k_meq_l: float,
        patient_mg_mmol_l: float,
        coronary_flow_reserve: float = 3.5,
        viscosity_ratio: float = 1.0,
    ) -> CardioplegiaStepOutput:
        """Advance infusion delivery, cannula/CS pressures, myocardial temperature, and protection."""
        out = CardioplegiaStepOutput(infusate_temp_c=self.temp_c)
        if dt <= 0.0:
            return out

        if not cross_clamped:
            self.stop_infusion()
            self.manual_arrest_hold = False
            self.breakthrough = False
            # Reperfusion rapidly washes out cardioplegic arrest and rewarms myocardium.
            self.protection *= math.exp(-dt / 15.0)
            if self.protection < 1e-3:
                self.protection = 0.0
            self.myocardial_temp_c += (core_temp_c - self.myocardial_temp_c) * -math.expm1(-dt / 30.0)
            return out

        cad_severity = clamp01((2.5 - coronary_flow_reserve) / 1.5)
        if self.active_infusion and self.remaining_ml > 1e-6:
            rec = self.recipe(self.protocol, self.phase)
            delivered_ml = min(self.remaining_ml, self.flow_ml_min * (dt / 60.0))
            self.remaining_ml = max(0.0, self.remaining_ml - delivered_ml)
            if self.remaining_ml <= 1e-6:
                self.active_infusion = False
                self.remaining_ml = 0.0

            delivered_l = delivered_ml / 1000.0
            blood_frac = 1.0 - rec.crystalloid_fraction
            out.delivered_total_ml = delivered_ml
            out.net_crystalloid_ml = delivered_ml * rec.crystalloid_fraction
            # Blood fraction comes from the CPB circuit; only excess over circuit concentration is net added.
            out.net_k_meq = max(0.0, (rec.delivered_k_meq_l - blood_frac * patient_k_meq_l) * delivered_l)
            out.net_mg_mmol = max(0.0, (rec.delivered_mg_mmol_l - blood_frac * patient_mg_mmol_l) * delivered_l)
            out.net_lidocaine_mg = rec.delivered_lidocaine_mg_l * delivered_l
            out.infusate_temp_c = self.temp_c

            self.total_volume_ml += delivered_ml
            self.total_crystalloid_ml += out.net_crystalloid_ml
            self.time_since_dose_s = 0.0

            # Delivery pressure mechanics (Partington 1989; Ikonomidis 1994).
            eff_visc = viscosity_ratio * (1.0 - 0.35 * rec.crystalloid_fraction)
            if self.route == "antegrade":
                # Aortic root pressure ~70-100 mmHg at 250 mL/min; rises modestly with CAD stenosis.
                root_r = (0.31 + 0.14 * cad_severity) * max(0.5, eff_visc)
                out.delivery_pressure_mmhg = clamp(self.flow_ml_min * root_r, 0.0, 250.0)
                out.coronary_sinus_pressure_mmhg = 0.0
                # Partington 1989: antegrade maldistribution behind severe CAD limits full arrest coverage.
                coverage = 1.0 - 0.30 * cad_severity
            else:
                # Retrograde CS catheter: target <=40 mmHg (200-250 mL/min -> ~25-35 mmHg; >350 mL/min >40).
                cs_p = self.flow_ml_min * 0.145 * (0.75 + 0.25 * max(0.4, eff_visc))
                out.delivery_pressure_mmhg = cs_p
                out.coronary_sinus_pressure_mmhg = cs_p
                # Retrograde bypasses epicardial coronary stenoses (Partington 1989).
                coverage = 0.94

            self.last_delivery_pressure_mmhg = out.delivery_pressure_mmhg
            self.last_cs_pressure_mmhg = out.coronary_sinus_pressure_mmhg

            # Cooling/warming myocardial temperature during cardioplegia perfusion (~300 g myocardium).
            self.myocardial_temp_c += (self.temp_c - self.myocardial_temp_c) * -math.expm1(-delivered_ml / 350.0)
            target_prot = coverage if rec.phase != "hot_shot" else max(self.protection, 0.70 * coverage)
            self.protection = max(self.protection, min(1.0, self.protection + (target_prot - self.protection) * -math.expm1(-delivered_ml / 220.0)))
            if rec.phase == "hot_shot":
                out.hot_shot_energy_boost = 0.18 * (delivered_ml / max(100.0, rec.default_volume_ml))
            self.breakthrough = False
        else:
            self.last_delivery_pressure_mmhg = 0.0
            self.last_cs_pressure_mmhg = 0.0
            if not self.manual_arrest_hold and self.protection > 0.0:
                self.time_since_dose_s += dt
                # Passive ambient/mediastinal rewarming of arrested myocardium toward core temperature.
                self.myocardial_temp_c += (core_temp_c - self.myocardial_temp_c) * -math.expm1(-dt / 900.0)
                # Warmer myocardium consumes arrest protection faster (Buckberg 1995).
                warm_rate = 1.0 + 1.2 * clamp01((self.myocardial_temp_c - 18.0) / 18.0)
                # Protection falls slowly during the safe window and drops sharply once overdue.
                window = max(300.0, self.protection_window_s / warm_rate)
                if self.time_since_dose_s <= window:
                    decay_tau = window * 1.8
                else:
                    decay_tau = 180.0
                self.protection = max(0.0, self.protection * math.exp(-dt / decay_tau))
                if self.protection < 0.25:
                    self.breakthrough = True
                    out.breakthrough_occurred = True
        return out

    @property
    def arrest_maintained(self) -> bool:
        """Whether cardioplegic arrest is currently suppressing electromechanical activity."""
        return self.manual_arrest_hold or self.protection >= 0.25

    @property
    def redose_due_s(self) -> float:
        """Estimated seconds remaining before cardioplegia redose is recommended (0 if due)."""
        if self.manual_arrest_hold or self.protection <= 0.0:
            return 0.0
        warm_rate = 1.0 + 1.2 * clamp01((self.myocardial_temp_c - 18.0) / 18.0)
        window = max(300.0, self.protection_window_s / warm_rate)
        return max(0.0, window - self.time_since_dose_s)
