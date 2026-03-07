from __future__ import annotations

import numpy as np

from pyisetcam.utils import blackbody, energy_to_quanta, param_format, quanta_to_energy


def test_param_format_string_and_key_value_list() -> None:
    assert param_format("Exposure Time") == "exposuretime"
    assert param_format(["Exposure Time", 1, "Some Flag", True]) == ["exposuretime", 1, "someflag", True]


def test_energy_quanta_round_trip() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array([0.2, 0.5, 0.8])
    quanta = energy_to_quanta(energy, wave)
    restored = quanta_to_energy(quanta, wave)
    assert np.allclose(restored, energy)


def test_blackbody_matlab_scaling() -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    spectra = blackbody(wave, np.array([3000.0, 5000.0]))
    assert spectra.shape == (wave.size, 2)
    assert np.all(spectra > 0.0)
    eq_index = int(np.argmin(np.abs(wave - 550.0)))
    assert np.isclose(spectra[eq_index, 0], spectra[eq_index, 1])
