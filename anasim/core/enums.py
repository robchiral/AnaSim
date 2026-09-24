from enum import Enum


class RhythmType(Enum):
    """Cardiac rhythms."""
    SINUS = "Sinus rhythm"
    SINUS_BRADY = "Sinus bradycardia"
    AFIB = "Atrial fibrillation"
    SVT = "Supraventricular tachycardia"
    VTACH = "Ventricular tachycardia"
    VFIB = "Ventricular fibrillation"
    ASYSTOLE = "Asystole"
