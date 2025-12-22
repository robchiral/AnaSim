from dataclasses import dataclass

@dataclass
class StandardPatient:
    age: int = 40
    height: int = 170
    weight: int = 70
    sex: int = 0  # 0: Male, 1: Female

    # Initial Physiological Parameters
    bis_initial: float = 98.0
    map_initial: float = 85.0
    hr_initial: float = 75.0
