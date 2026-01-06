from enum import Enum, auto

class RhythmType(Enum):
    """Cardiac Rhythm Types"""
    SINUS = "Sinus rhythm"
    SINUS_BRADY = "Sinus bradycardia" 
    AFIB = "Atrial fibrillation"
    SVT = "Supraventricular tachycardia"
    VTACH = "Ventricular tachycardia"
    VFIB = "Ventricular fibrillation"
    ASYSTOLE = "Asystole"
