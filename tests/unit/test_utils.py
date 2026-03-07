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


def test_blackbody_normalized() -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    spectrum = blackbody(wave, 3000.0)
    assert spectrum.shape == wave.shape
    assert np.isclose(np.max(spectrum), 1.0)

