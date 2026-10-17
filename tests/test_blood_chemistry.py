"""Rees 2010 reference chemistry; oxygen saturation is a supplied input."""

import math

import pytest

from anasim.physiology.blood_chemistry import BloodChemistry

MMHG_PER_KPA = 7.5006168270417


def test_rees_table_2_normal_oxygenated_blood():
    model = BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=9.3 / 21.0)
    state = model.equilibrate(ph=7.4, pco2_mmhg=5.33 * MMHG_PER_KPA, oxygen_saturation=1.0)
    # The printed parameters do not reproduce every rounded Table 2 value exactly.
    assert state.plasma_bicarbonate_mmol_l == pytest.approx(24.5, abs=0.05)
    assert state.rbc_bicarbonate_mmol_l == pytest.approx(12.8, abs=0.05)
    assert state.total_co2_mmol_l == pytest.approx(20.7, abs=0.05)
    assert state.plasma_buffer_base_meq_l == pytest.approx(41.7, abs=0.05)
    assert state.rbc_buffer_base_meq_l == pytest.approx(72.5, abs=0.35)
    assert state.buffer_base_meq_l == pytest.approx(55.3, abs=0.15)


@pytest.mark.parametrize(
    "pco2_kpa,saturation,buffer_base,expected_ph",
    [
        pytest.param(5.33, 0.982, 55.3, 7.400, id="normal"),
        pytest.param(6.0, 0.981, 55.3, 7.367, id="respiratory-acidosis"),
        pytest.param(6.0, 0.705, 55.3, 7.378, id="respiratory-acidosis-low-oxygen"),
        pytest.param(4.0, 0.985, 55.3, 7.480, id="respiratory-alkalosis"),
        pytest.param(4.0, 0.772, 55.3, 7.490, id="respiratory-alkalosis-low-oxygen"),
        pytest.param(5.33, 0.981, 52.8, 7.364, id="metabolic-acidosis"),
        pytest.param(5.33, 0.983, 57.8, 7.436, id="metabolic-alkalosis"),
        pytest.param(5.33, 0.743, 57.8, 7.444, id="metabolic-alkalosis-low-oxygen"),
    ],
)
def test_rees_table_6_acid_base_disturbances(pco2_kpa, saturation, buffer_base, expected_ph):
    """Test buffering independently of the paper's oxygen dissociation curve.

    The low-oxygen metabolic-acidosis row prints PCO2 6.0 kPa, inconsistent
    with the stated metabolic protocol and its displayed results; omit that row.
    """
    model = BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=9.3 / 21.0)
    state = model.solve_ph(
        pco2_mmhg=pco2_kpa * MMHG_PER_KPA, oxygen_saturation=saturation,
        buffer_base_meq_l=buffer_base,
    )
    assert state.ph == pytest.approx(expected_ph, abs=0.0025)


def test_hemoglobin_limits_the_ph_fall_during_acute_co2_retention():
    ph_falls = []
    for hb in (4.65, 9.3):
        model = BloodChemistry(hemoglobin_mmol_l=hb, hematocrit=hb / 21.0)
        baseline = model.equilibrate(ph=7.4, pco2_mmhg=40.0, oxygen_saturation=0.98)
        hypercapnic = model.solve_ph(
            pco2_mmhg=80.0, oxygen_saturation=0.98,
            buffer_base_meq_l=baseline.buffer_base_meq_l,
        )
        assert hypercapnic.plasma_bicarbonate_mmol_l > baseline.plasma_bicarbonate_mmol_l
        assert hypercapnic.total_co2_mmol_l > baseline.total_co2_mmol_l
        ph_falls.append(baseline.ph - hypercapnic.ph)
    assert 0.0 < ph_falls[1] < ph_falls[0]


def test_co2_capacity_matches_solved_buffering():
    model = BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=9.3 / 21.0)
    buffer_base = model.evaluate(7.4, 40.0, 0.98).buffer_base
    for pressure in (20.0, 40.0, 80.0):
        states = [model.solve(p, 0.98, buffer_base) for p in (pressure - 0.001, pressure, pressure + 0.001)]
        derivative = (states[2].total_co2 - states[0].total_co2) / 0.002
        assert states[1].co2_capacity == pytest.approx(derivative, rel=1e-6)


def test_solution_is_independent_of_initial_ph_including_search_limits():
    model = BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=9.3 / 21.0)
    for ph, pressure, saturation in ((6.0, 160.0, 0.1), (7.2, 80.0, 0.7), (7.6, 20.0, 0.98), (8.5, 5.0, 0.7)):
        target = model.evaluate(ph, pressure, saturation)
        for initial_ph in (6.0, 7.4, 8.5):
            state = model.solve(pressure, saturation, target.buffer_base, initial_ph)
            assert state.ph == pytest.approx(ph, abs=1e-10)
            assert state.total_co2 == pytest.approx(target.total_co2, abs=1e-9)
            assert state.co2_capacity == pytest.approx(target.co2_capacity, rel=1e-10)


def test_mixing_preserves_buffer_base_and_co2_for_different_cell_compositions():
    first = BloodChemistry(hemoglobin_mmol_l=7.0, hematocrit=0.45, plasma_buffer_mmol_l=18.0)
    second = BloodChemistry(hemoglobin_mmol_l=10.0, hematocrit=0.35, plasma_buffer_mmol_l=30.0)
    fraction = 0.4
    hct = fraction * first.hematocrit + (1.0 - fraction) * second.hematocrit
    mixed = BloodChemistry(
        hemoglobin_mmol_l=fraction * first.hemoglobin_mmol_l + (1.0 - fraction) * second.hemoglobin_mmol_l,
        hematocrit=hct,
        plasma_buffer_mmol_l=(
            fraction * (1.0 - first.hematocrit) * first.plasma_buffer_mmol_l
            + (1.0 - fraction) * (1.0 - second.hematocrit) * second.plasma_buffer_mmol_l
        ) / (1.0 - hct),
    )
    # Equal initial gases isolate component conservation from affinity changes.
    a, b, result = [model.equilibrate(ph=7.4, pco2_mmhg=40.0, oxygen_saturation=0.98)
                    for model in (first, second, mixed)]
    assert result.buffer_base_meq_l == pytest.approx(
        fraction * a.buffer_base_meq_l + (1.0 - fraction) * b.buffer_base_meq_l, abs=1e-12,
    )
    assert result.total_co2_mmol_l == pytest.approx(
        fraction * a.total_co2_mmol_l + (1.0 - fraction) * b.total_co2_mmol_l, abs=1e-12,
    )


def test_invalid_or_unsolvable_chemistry_is_rejected():
    with pytest.raises(ValueError, match="hematocrit"):
        BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=1.0)
    with pytest.raises(ValueError, match="hemoglobin_mmol_l"):
        BloodChemistry(hemoglobin_mmol_l=math.nan, hematocrit=0.45)
    model = BloodChemistry(hemoglobin_mmol_l=9.3, hematocrit=9.3 / 21.0)
    with pytest.raises(ValueError, match="initial_ph"):
        model.solve_ph(pco2_mmhg=40.0, oxygen_saturation=0.98,
                       buffer_base_meq_l=55.3, initial_ph=math.nan)
    with pytest.raises(ValueError, match="oxygen_saturation"):
        model.equilibrate(ph=7.4, pco2_mmhg=40.0, oxygen_saturation=98.0)
    with pytest.raises(ValueError, match="pco2_mmhg"):
        model.equilibrate(ph=7.4, pco2_mmhg=0.0, oxygen_saturation=0.98)
    with pytest.raises(ValueError, match="outside the solver range"):
        model.solve_ph(pco2_mmhg=40.0, oxygen_saturation=0.98, buffer_base_meq_l=0.0)
