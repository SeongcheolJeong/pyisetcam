from __future__ import annotations

import numpy as np

from pyisetcam import ieParameterOtype
from pyisetcam.utils import blackbody, energy_to_quanta, ie_parameter_otype, param_format, quanta_to_energy, unit_frequency_list


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


def test_unit_frequency_list_matches_matlab_even_and_odd() -> None:
    assert np.allclose(unit_frequency_list(4), np.array([-1.0, -0.5, 0.0, 0.5]))
    assert np.allclose(unit_frequency_list(5), np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_ie_parameter_otype_handles_direct_prefix_and_unique_params() -> None:
    assert ie_parameter_otype("scene") == ("scene", None)
    assert ie_parameter_otype("oi size") == ("oi", "size")
    assert ie_parameter_otype("pixel/size") == ("pixel", "size")
    assert ie_parameter_otype("wvf_zcoeffs") == ("wvf", "zcoeffs")
    assert ie_parameter_otype("display gamma") == ("display", "gamma")
    assert ie_parameter_otype("dsnu sigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("sensorspectralsr") == ("sensor", "sensorspectralsr")
    assert ie_parameter_otype("dynamic range") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("shot noise flag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("human cone densities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("human cone seed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("fnumber") == ("optics", "fnumber")
    assert ie_parameter_otype("asset light") == ("asset", "assetlight")
    assert ieParameterOtype("ip display") == ("ip", "display")


def test_ie_parameter_otype_returns_empty_type_for_ambiguous_or_unknown_params() -> None:
    assert ie_parameter_otype("size") == ("", "size")
    assert ie_parameter_otype("mystery parameter") == ("", "mysteryparameter")
