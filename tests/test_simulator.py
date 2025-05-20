from simulator import SignalGenerator, DeviceModel
import numpy as np

def test_diode_generation():
    t, v = SignalGenerator.sweep(0, 1, 100)
    params = {'Is': 1e-14, 'N': 1.0, 'Rth': 50.0, 'Cth': 1.0, 'alphaT': 0.005, 'tauH': 0.1, 'beta_h': 0.1}
    model = DeviceModel(params, 'Диод')
    i = model.simulate(v)
    assert i.shape == v.shape
    assert np.all(np.isfinite(i)) 