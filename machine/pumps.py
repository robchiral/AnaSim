
from dataclasses import dataclass
from typing import Optional

@dataclass
class PumpStatus:
    is_infusing: bool = False
    rate: float = 0.0 # Unit depends on drug (ml/hr or mg/hr or ug/min)
    volume_infused: float = 0.0
    drug_name: str = ""
    concentration: float = 1.0 # mg/mL or ug/mL
    rate_unit: str = "ml/hr" # "ml/hr", "mg/hr", "ug/min", "ug/kg/min"

class InfusionPump:
    """
    Simulates a single channel infusion pump.
    """
    def __init__(self, drug_name: str, concentration: float, rate_unit: str = "ml/hr"):
        self.status = PumpStatus(
            drug_name=drug_name,
            concentration=concentration,
            rate_unit=rate_unit
        )
        self.target_rate = 0.0
        
    def set_rate(self, rate: float):
        """Set target rate."""
        self.target_rate = max(0.0, rate)
        self.status.rate = self.target_rate
        self.status.is_infusing = (self.target_rate > 0)
        
    def bolus(self, amount: float):
        """Standard pump usually doesn't do instant bolus, requires time.
        But for sim, we can track bolus volume."""
        self.status.volume_infused += amount
        # Helper to return amount for PK
        return amount

    def step(self, dt: float) -> float:
        """
        Advance pump.
        Returns amount of DRUG infused in this step (mg or ug).
        """
        if not self.status.is_infusing:
            return 0.0
            
        # volumetrics
        # Rate units handling
        # standardizing to "unit of drug amount" / sec
        
        amount = 0.0
        rate_sec = 0.0
        
        if self.status.rate_unit == "ml/hr":
            # amount (mg or ug) = rate (ml/hr) * conc (mg/ml) / 3600 * dt
            rate_ml_sec = self.status.rate / 3600.0
            vol = rate_ml_sec * dt
            amount = vol * self.status.concentration
            self.status.volume_infused += vol
        
        elif self.status.rate_unit == "mg/hr":
             # amount (mg) = rate (mg/hr) / 3600 * dt
             # vol = amount / conc
             rate_mg_sec = self.status.rate / 3600.0
             amount = rate_mg_sec * dt
             if self.status.concentration > 0:
                 self.status.volume_infused += amount / self.status.concentration
                 
        elif self.status.rate_unit == "ug/min":
             # amount (ug) = rate (ug/min) / 60 * dt
             rate_ug_sec = self.status.rate / 60.0
             amount = rate_ug_sec * dt
             if self.status.concentration > 0:
                 # vol (ml) = amount (ug) / (conc (ug/ml))
                 self.status.volume_infused += amount / self.status.concentration
                 
        elif self.status.rate_unit == "ug/kg/min":
             # Weight-based rate calculation is handled by the caller
             pass
             
        return amount
