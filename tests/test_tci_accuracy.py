import numpy as np
import pytest

from anasim.core.state import SimulationConfig
from anasim.core.tci import TCIController
from anasim.patient.pk_models import PropofolPKSchnider, RemifentanilPKMinto


@pytest.mark.parametrize(
    ("pk_cls", "compartment", "max_rate", "t90_limit_s"),
    [
        (PropofolPKSchnider, "effect_site", 50.0, 90),
        (PropofolPKSchnider, "plasma", 50.0, 30),
        (RemifentanilPKMinto, "effect_site", 500.0, 90),
    ],
)
def test_controller_reaches_target_without_overshoot_and_follows_a_decrease(
    patient_factory, pk_cls, compartment, max_rate, t90_limit_s
):
    """Shafer and Gregg 1992: the peak-constrained controller never overshoots."""
    pk = pk_cls(patient_factory(age=45))
    tci = TCIController(
        pk_model=pk, target_compartment=compartment, max_rate=max_rate, sampling_time=1.0
    )

    def concentration():
        return pk.state.ce if compartment == "effect_site" else pk.state.c1

    target = 4.0
    history = []
    for _ in range(300):
        pk.step(1.0, tci.step(target))
        history.append(concentration())
    history = np.asarray(history)

    reached = np.flatnonzero(history >= 0.9 * target)
    assert reached.size > 0
    assert reached[0] < t90_limit_s
    assert history.max() < 1.03 * target
    assert history[-60:].mean() == pytest.approx(target, rel=0.02)

    lower = 2.5
    for _ in range(300):
        pk.step(1.0, tci.step(lower))
    assert concentration() == pytest.approx(lower, rel=0.03)


class TestEngineTCI:
    """Pump-limited TCI inside the engine."""

    @pytest.mark.parametrize("dt", [0.1, 0.3, 1.0])
    def test_effect_site_induction_reaches_hypnosis_without_overshoot(self, engine_factory, dt):
        engine = engine_factory(config=SimulationConfig(mode="awake", tci_enabled=True, dt=dt, rng_seed=123), start=True)
        engine.set_airway_mode("Mask")
        engine.set_fgf(6.0, 0.0)
        engine.enable_tci("propofol", 4.0)
        peak = 0.0
        time_to_90 = None
        for _ in range(round(360.0 / dt)):
            engine.step(dt)
            peak = max(peak, engine.state.propofol_ce)
            if time_to_90 is None and engine.state.propofol_ce >= 3.6:
                time_to_90 = engine.state.time
        assert time_to_90 is not None and time_to_90 < 240.0
        assert peak < 4.0 * 1.03
        assert engine.state.propofol_ce == pytest.approx(4.0, rel=0.03)
        assert engine.state.bis < 60.0

    def test_plasma_controller_does_not_bolus_after_resync(self, engine_factory):
        """Resyncing to live PK state must not trigger max-rate boluses."""
        engine = engine_factory(config=SimulationConfig(mode="steady_state", tci_enabled=True), start=True)
        engine.set_drug_rate("nore", 5.0)
        for _ in range(300):
            engine.step(1.0)
        engine.enable_tci("nore", engine.pk_nore.state.c1, mode="plasma")
        target = engine.tci_nore.target
        peak = 0.0
        for _ in range(3000):
            engine.step(0.1)
            peak = max(peak, engine.pk_nore.state.c1)
        assert peak < target * 1.4

    @pytest.mark.parametrize("manual_rate", [0.0, 120.0])
    def test_manual_rate_remains_in_control_after_tci(self, engine_factory, manual_rate):
        engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
        engine.enable_tci("propofol", 4.0)
        engine.step(0.1)
        assert engine.propofol_rate_mg_sec > 0

        engine.set_drug_rate("propofol", manual_rate)
        for _ in range(120):
            engine.step(0.1)
            assert engine.propofol_rate_mg_sec == pytest.approx(manual_rate * engine.patient.weight / 60000)
        assert not engine.get_drug_state("propofol")["is_tci"]

    def test_target_mode_switch_uses_live_compartments(self, engine_factory):
        engine = engine_factory(config=SimulationConfig(tci_enabled=True), start=True)
        engine.give_drug_bolus("propofol", 100)
        engine.enable_tci("propofol", 4.0)
        engine.step(0.05)
        engine.enable_tci("propofol", 2.0, mode="plasma")

        assert engine.tci_prop.target_compartment == "plasma"
        for _ in range(20):
            engine.step(0.1)
            # Plasma already exceeds the new target after the manual bolus.
            assert engine.propofol_rate_mg_sec == 0.0
