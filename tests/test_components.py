
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
    assert res['HR']['low']
    
    # High MAP
    state = {'HR': 60, 'MAP': 150, 'BIS': 50}
    res = alarms.update(state)
    assert res['MAP']['high']

def test_alarm_delay():
    # Delay 2 seconds, dt 1s
    alarms = AlarmSystem(delays={'HR': 2}, dt=1.0)
    
    # Step 1: Low
    res = alarms.update({'HR': 30})
    assert 'HR' not in res # Wait buffer fill
    
    # Buffer needs 2 samples (max(1, 2/1) = 2).
    # Step 2: Low
    res = alarms.update({'HR': 30})
    assert res['HR']['low']  # Full window low
    
    # If Intermittent
    alarms = AlarmSystem(delays={'HR': 3}, dt=1.0) # Size 3
    alarms.update({'HR': 30}) # [30]
    alarms.update({'HR': 100}) # [30, 100] -> Not all low
    res = alarms.update({'HR': 30}) # [30, 100, 30] -> Not all low
    assert 'HR' not in res
    
    alarms.update({'HR': 30}) # [100, 30, 30]
    alarms.update({'HR': 30}) # [30, 30, 30] -> All low
    res = alarms.update({'HR':30})
    assert res['HR']['low']

# --- Disturbance Tests ---

def test_disturbance_none():
    dist = Disturbances(None)
    vals = dist.compute_dist(100.0)
    assert vals.bis == 0.0
    assert vals.svr == 0.0
    assert vals.sv == 0.0
    assert vals.hr == 0.0

def test_disturbance_profiles_interpolate_and_report_lifecycle():
    pulse = Disturbances("stim_intubation_pulse")
    assert pulse.compute_dist(0.0).bis == 0.0
    assert pulse.compute_dist(15.0).bis > 0.0
    assert pulse.compute_average(0.0, 60.0).bis > 0.0
    assert not pulse.is_complete(49.9)
    assert pulse.is_complete(50.0)

    sustained = Disturbances("stim_sustained_surgery")
    effects = sustained.compute_dist(100.0)
    averaged_effects = sustained.compute_average(90.0, 110.0)
    assert effects.bis > 0.0
    assert effects.svr > 0.0
    assert effects.hr > 0.0
    assert averaged_effects.bis > 0.0
    assert not sustained.is_complete(10_000.0)
