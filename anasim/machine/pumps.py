from dataclasses import dataclass

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
        """Simulate bolus delivery.
        The simulation tracks bolus volume as an instantaneous addition."""
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
            
        # Volumetrics and rate unit handling
        unit = self.status.rate_unit
        rate = self.status.rate
        conc = self.status.concentration

        if unit == "ml/hr":
            # amount (mg or ug) = rate (ml/hr) * conc (mg/ml) / 3600 * dt
            vol = (rate / 3600.0) * dt
            amount = vol * conc
        elif unit == "mg/hr":
            # amount (mg) = rate (mg/hr) / 3600 * dt
            amount = (rate / 3600.0) * dt
            vol = amount / conc if conc > 0 else 0.0
        elif unit == "ug/min":
            # amount (ug) = rate (ug/min) / 60 * dt
            amount = (rate / 60.0) * dt
            vol = amount / conc if conc > 0 else 0.0
        elif unit == "ug/kg/min":
            # Weight-based rate calculation is handled by the caller
            return 0.0
        else:
            return 0.0

        self.status.volume_infused += vol
        return amount
