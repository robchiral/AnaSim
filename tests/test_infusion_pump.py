from anasim.machine.pumps import InfusionPump


def test_pump_ml_per_hr_delivery_and_volume():
    pump = InfusionPump("TestDrug", concentration=10.0, rate_unit="ml/hr")
    pump.set_rate(60.0)  # 60 mL/hr = 1 mL/min

    amount = pump.step(60.0)  # 60 seconds -> 1 mL -> 10 mg
    assert amount == 10.0
    assert abs(pump.status.volume_infused - 1.0) < 1e-6


def test_pump_mg_per_hr_delivery_and_volume():
    pump = InfusionPump("TestDrug", concentration=10.0, rate_unit="mg/hr")
    pump.set_rate(60.0)  # 60 mg/hr = 1 mg/min

    amount = pump.step(60.0)  # 60 seconds -> 1 mg
    assert amount == 1.0
    assert abs(pump.status.volume_infused - 0.1) < 1e-6


def test_pump_ug_per_min_delivery_and_volume():
    pump = InfusionPump("TestDrug", concentration=100.0, rate_unit="ug/min")
    pump.set_rate(120.0)  # 120 ug/min = 2 ug/sec

    amount = pump.step(30.0)  # 30 seconds -> 60 ug
    assert amount == 60.0
    assert abs(pump.status.volume_infused - 0.6) < 1e-6


def test_pump_bolus_tracks_volume():
    pump = InfusionPump("TestDrug", concentration=5.0, rate_unit="ml/hr")
    pump.bolus(2.5)
    assert abs(pump.status.volume_infused - 2.5) < 1e-6
