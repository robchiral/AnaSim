import pytest

from anasim.physiology.disturbances import Disturbances


@pytest.mark.parametrize("profile,start,end,expected", [
    ("stim_intubation_pulse", 299.0, 301.0, (3.0 / 480.0, 0.0, 0.115 / 480.0, 3.7 / 480.0)),
    ("stim_intubation_pulse", 301.0, 305.0, (0.0, 0.0, 0.0, 0.0)),
    ("stim_sustained_surgery", 59.0, 61.0, (7.98, 5.99, 1.596, 0.9975)),
    ("stim_sustained_surgery", 600.0, 3600.0, (8.0, 6.0, 1.6, 1.0)),
])
def test_average_retains_ramps_and_plateaus_across_step_boundaries(profile, start, end, expected):
    effects = Disturbances(profile).compute_average(start, end)
    assert (effects.bis, effects.hr, effects.svr, effects.sv) == pytest.approx(expected)
