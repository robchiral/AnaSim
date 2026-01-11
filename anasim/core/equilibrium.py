
import numpy as np
from scipy.optimize import minimize, root_scalar
from typing import Tuple, Optional
from anasim.patient.patient import Patient
from anasim.patient.pk_models import PropofolPKMarsh, RemifentanilPKMinto, NorepinephrinePK
from anasim.patient.pd_models import BISModel, TOLModel
from anasim.physiology.hemodynamics import HemodynamicModel

class EquilibriumSolver:
    """
    Solves for steady-state drug concentrations and infusion rates 
    to achieve target BIS, TOL, and MAP.
    """

    def __init__(self, patient: Patient):
        self.patient = patient

    def solve_bi_objective(
        self, 
        bis_model: BISModel,
        tol_model: TOLModel,
        bis_target: float, 
        tol_target: float,
        primary_hypnotic: str = 'propofol'
    ) -> Tuple[float, float, float]:
        """
        Find drug concentrations to minimize:
        J = (BIS - BIS_tgt)^2 / 100^2 + (TOL - TOL_tgt)^2
        
        Args:
            bis_model: BIS pharmacodynamic model
            tol_model: TOL pharmacodynamic model  
            bis_target: Target BIS value
            tol_target: Target TOL probability
            primary_hypnotic: 'propofol' or 'volatile'
            
        Returns:
            Tuple of (ce_prop, ce_remi, mac)
        """
        
        def objective(x):
            val_primary = x[0]
            ce_r = x[1]
            if val_primary < 0 or ce_r < 0:
                return 1e9
            
            ce_p = 0.0
            u_vol = 0.0
            
            if primary_hypnotic == 'propofol':
                ce_p = val_primary
            else:
                # Treat x[0] as normalized volatile load (MAC proxy)
                u_vol = val_primary
                
            bis = bis_model.compute_bis(ce_p, ce_r, u_volatile=u_vol)
            tol = tol_model.compute_probability(ce_p, ce_r, mac=u_vol)
             
            return ((bis - bis_target)**2) / 10000.0 + ((tol - tol_target)**2)

        # Initial guess and bounds
        if primary_hypnotic == 'propofol':
            x0 = [bis_model.params.c50p, bis_model.params.c50r / 2.0]
            bounds = ((0, 20), (0, 100))
        else:
            x0 = [0.8, bis_model.params.c50r / 2.0]
            bounds = ((0, 3.0), (0, 100))
        
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if primary_hypnotic == 'propofol':
            return res.x[0], res.x[1], 0.0
        else:
            return 0.0, res.x[1], res.x[0]

    def solve_map_target(
        self,
        hemo_model: HemodynamicModel,
        ce_prop: float,
        ce_remi: float,
        map_target: float,
        mac: float = 0.0
    ) -> float:
        """
        Find Norepinephrine Cp (ng/mL) to reach MAP target.
        
        Args:
            hemo_model: Hemodynamic model instance
            ce_prop: Propofol effect-site concentration (ug/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            map_target: Target MAP (mmHg)
            mac: MAC value for volatile anesthetics
            
        Returns:
            Required norepinephrine concentration (ng/mL)
        """
        def error_func(ce_nore):
            if ce_nore < 0:
                return -100
            state = hemo_model.calculate_steady_state(ce_prop, ce_remi, ce_nore, mac_sevo=mac)
            return state.map - map_target
            
        try:
            res = root_scalar(error_func, bracket=[0, 50], method='brentq')
            return res.root
        except ValueError:
            # Target MAP outside controllable range
            if error_func(0) > 0:
                return 0.0  # MAP already above target without norepi
            else:
                return 50.0  # Max out norepi

    def find_equilibrium(
        self,
        pk_prop: PropofolPKMarsh,
        pk_remi: RemifentanilPKMinto,
        pk_nore: NorepinephrinePK,
        bis_model: BISModel,
        tol_model: TOLModel,
        hemo_model: HemodynamicModel,
        bis_target: float = 40.0,
        tol_target: float = 0.9,
        map_target: float = 80.0,
        fixed_ratio: Optional[float] = None,
        primary_hypnotic: str = 'propofol'
    ) -> dict:
        """
        Compute full equilibrium solution for target BIS, TOL, and MAP.
        
        Args:
            pk_prop: Propofol PK model
            pk_remi: Remifentanil PK model
            pk_nore: Norepinephrine PK model
            bis_model: BIS PD model
            tol_model: TOL PD model
            hemo_model: Hemodynamic model
            bis_target: Target BIS value (default 40)
            tol_target: Target TOL probability (default 0.9)
            map_target: Target MAP in mmHg (default 80)
            fixed_ratio: If set, fixes Ce_remi = ratio * Ce_prop
            primary_hypnotic: 'propofol' or 'volatile'
            
        Returns:
            Dictionary with target Ce values and infusion rates
        """
        mac = 0.0
        
        if fixed_ratio:
            def bis_error(ce_p):
                ce_r = ce_p * fixed_ratio
                return bis_model.compute_bis(ce_p, ce_r) - bis_target
             
            try:
                res = root_scalar(bis_error, bracket=[0, 20], method='brentq')
                ce_prop = res.root
                ce_remi = ce_prop * fixed_ratio
            except ValueError:
                ce_prop = 4.0
                ce_remi = 4.0 * fixed_ratio
        else:
            ce_prop, ce_remi, mac = self.solve_bi_objective(
                bis_model, tol_model, bis_target, tol_target, primary_hypnotic
            )
            
        ce_nore = self.solve_map_target(hemo_model, ce_prop, ce_remi, map_target, mac)
        
        u_prop, u_remi, u_nore = self.calculate_infusion_rates(
            ce_prop, ce_remi, ce_nore, pk_prop, pk_remi, pk_nore
        )
        
        return {
            "ce_prop": ce_prop,
            "ce_remi": ce_remi,
            "ce_nore": ce_nore,
            "mac": mac,
            "rate_prop_mg_sec": u_prop,
            "rate_remi_ug_sec": u_remi,
            "rate_nore_ug_sec": u_nore
        }

    def calculate_infusion_rates(
        self,
        ce_prop: float,
        ce_remi: float,
        ce_nore: float,
        pk_prop,
        pk_remi,
        pk_nore
    ) -> Tuple[float, float, float]:
        """
        Convert steady-state Ce to infusion rate.
        
        At steady state: Rate = Clearance * Css = V1 * k10 * Ce
        
        Args:
            ce_prop: Propofol effect-site concentration (ug/mL)
            ce_remi: Remifentanil effect-site concentration (ng/mL)
            ce_nore: Norepinephrine concentration (ng/mL)
            pk_prop: Propofol PK model
            pk_remi: Remifentanil PK model
            pk_nore: Norepinephrine PK model
            
        Returns:
            Tuple of (rate_prop_mg_sec, rate_remi_ug_sec, rate_nore_ug_sec)
        """
        # Propofol: Ce (ug/mL=mg/L) * V1 (L) * k10 (1/min) = mg/min -> /60 = mg/sec
        u_prop_mg_sec = (ce_prop * pk_prop.v1 * pk_prop.k10) / 60.0
        
        # Remifentanil: Ce (ng/mL=ug/L) * V1 (L) * k10 (1/min) = ug/min -> /60 = ug/sec
        u_remi_ug_sec = (ce_remi * pk_remi.v1 * pk_remi.k10) / 60.0
        
        # Norepinephrine: Ce (ng/mL=ug/L) * V1 (L) * k10 (1/min) = ug/min -> /60 = ug/sec
        u_nore_ug_sec = (ce_nore * pk_nore.v1 * pk_nore.k10) / 60.0
        
        return u_prop_mg_sec, u_remi_ug_sec, u_nore_ug_sec
