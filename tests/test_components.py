
import pytest
from anasim.monitors.alarms import AlarmSystem
from anasim.physiology.disturbances import Disturbances

# --- Alarm Tests ---

def test_alarm_bounds():
    alarms = AlarmSystem()
    
    # Normal state
    state = {'HR': 60, 'MAP': 80, 'BIS': 50}
    res = alarms.update(state)
    assert 'HR' not in res
    
    # Low HR
    state = {'HR': 30, 'MAP': 80, 'BIS': 50}
    res = alarms.update(state)
    # Delay is 0 by default for HR
    assert res['HR']['low'] == True
    
    # High MAP
    state = {'HR': 60, 'MAP': 150, 'BIS': 50}
    res = alarms.update(state)
    assert res['MAP']['high'] == True

def test_alarm_delay():
    # Delay 2 seconds, dt 1s
    alarms = AlarmSystem(delays={'HR': 2}, dt=1.0)
    
    # Step 1: Low
    res = alarms.update({'HR': 30})
    assert 'HR' not in res # Wait buffer fill
    
    # Buffer needs 2 samples (max(1, 2/1) = 2).
    # Step 2: Low
    res = alarms.update({'HR': 30})
    assert res['HR']['low'] == True # Full window low
    
    # If Intermittent
    alarms = AlarmSystem(delays={'HR': 3}, dt=1.0) # Size 3
    alarms.update({'HR': 30}) # [30]
    alarms.update({'HR': 100}) # [30, 100] -> Not all low
    res = alarms.update({'HR': 30}) # [30, 100, 30] -> Not all low
    assert 'HR' not in res
    
    alarms.update({'HR': 30}) # [100, 30, 30]
    alarms.update({'HR': 30}) # [30, 30, 30] -> All low
    res = alarms.update({'HR':30})
    assert res['HR']['low'] == True

# --- Disturbance Tests ---

def test_disturbance_none():
    dist = Disturbances(None)
    vals = dist.compute_dist(100.0)
    assert vals == [0.0]*6

def test_disturbance_step():
    dist = Disturbances("stim_intubation_pulse")
    
    # Before step
    vals0 = dist.compute_dist(0.0)
    assert vals0[0] == 0.0 # BIS
    
    # in step
    vals1 = dist.compute_dist(15.0) 
    assert vals1[0] > 0
    
def test_disturbance_vitaldb():
    dist = Disturbances("stim_sustained_surgery")
    vals = dist.compute_dist(100.0)
    assert len(vals) == 6
