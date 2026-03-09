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
    assert ie_parameter_otype("pixel width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("pixel pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("size same fill factor") == ("pixel", "sizesamefillfactor")
    assert ie_parameter_otype("size constant fill factor") == ("pixel", "sizeconstantfillfactor")
    assert ie_parameter_otype("size keep fill factor") == ("pixel", "sizekeepfillfactor")
    assert ie_parameter_otype("dark voltage per pixel per sec") == ("pixel", "darkvoltageperpixelpersec")
    assert ie_parameter_otype("wvf_zcoeffs") == ("wvf", "zcoeffs")
    assert ie_parameter_otype("display gamma") == ("display", "gamma")
    assert ie_parameter_otype("dsnu sigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("sensorspectralsr") == ("sensor", "sensorspectralsr")
    assert ie_parameter_otype("dynamic range") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("shot noise flag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("black level") == ("sensor", "blacklevel")
    assert ie_parameter_otype("zero level") == ("sensor", "zerolevel")
    assert ie_parameter_otype("bits") == ("sensor", "bits")
    assert ie_parameter_otype("voltage") == ("sensor", "voltage")
    assert ie_parameter_otype("color") == ("sensor", "color")
    assert ie_parameter_otype("digital value") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("digital values") == ("sensor", "digitalvalues")
    assert ie_parameter_otype("electron") == ("sensor", "electron")
    assert ie_parameter_otype("electrons per area") == ("sensor", "electronsperarea")
    assert ie_parameter_otype("filter names cell array") == ("sensor", "filternamescellarray")
    assert ie_parameter_otype("filter color names cell array") == ("sensor", "filtercolornamescellarray")
    assert ie_parameter_otype("filter names cell") == ("sensor", "filternamescell")
    assert ie_parameter_otype("lut") == ("sensor", "lut")
    assert ie_parameter_otype("quantization lut") == ("sensor", "quantizationlut")
    assert ie_parameter_otype("qMethod") == ("sensor", "qmethod")
    assert ie_parameter_otype("quantization structure") == ("sensor", "quantizationstructure")
    assert ie_parameter_otype("dv or volts") == ("sensor", "dvorvolts")
    assert ie_parameter_otype("digital or volts") == ("sensor", "digitalorvolts")
    assert ie_parameter_otype("volt images") == ("sensor", "voltimages")
    assert ie_parameter_otype("fill factor") == ("pixel", "fillfactor")
    assert ie_parameter_otype("conversion gain") == ("pixel", "conversiongain")
    assert ie_parameter_otype("conversion gain v per electron") == ("pixel", "conversiongainvperelectron")
    assert ie_parameter_otype("well capacity") == ("pixel", "wellcapacity")
    assert ie_parameter_otype("volts per electron") == ("pixel", "voltsperelectron")
    assert ie_parameter_otype("saturation voltage") == ("pixel", "saturationvoltage")
    assert ie_parameter_otype("max voltage") == ("pixel", "maxvoltage")
    assert ie_parameter_otype("dark current density") == ("pixel", "darkcurrentdensity")
    assert ie_parameter_otype("dark current per pixel") == ("pixel", "darkcurrentperpixel")
    assert ie_parameter_otype("dark voltage per pixel") == ("pixel", "darkvoltageperpixel")
    assert ie_parameter_otype("volts per second") == ("pixel", "voltspersecond")
    assert ie_parameter_otype("read noise") == ("pixel", "readnoise")
    assert ie_parameter_otype("read noise volts") == ("pixel", "readnoisevolts")
    assert ie_parameter_otype("read standard deviation volts") == ("pixel", "readstandarddeviationvolts")
    assert ie_parameter_otype("read standard deviation electrons") == ("pixel", "readstandarddeviationelectrons")
    assert ie_parameter_otype("read noise std volts") == ("pixel", "readnoisestdvolts")
    assert ie_parameter_otype("read noise millivolts") == ("pixel", "readnoisemillivolts")
    assert ie_parameter_otype("width gap") == ("pixel", "widthgap")
    assert ie_parameter_otype("width between pixels") == ("pixel", "widthbetweenpixels")
    assert ie_parameter_otype("height between pixels") == ("pixel", "heightbetweenpixels")
    assert ie_parameter_otype("pixel width meters") == ("pixel", "widthmeters")
    assert ie_parameter_otype("pixel height meters") == ("pixel", "heightmeters")
    assert ie_parameter_otype("pd width") == ("pixel", "pdwidth")
    assert ie_parameter_otype("pd dimension") == ("pixel", "pddimension")
    assert ie_parameter_otype("layer thicknesses") == ("pixel", "layerthicknesses")
    assert ie_parameter_otype("refractive indices") == ("pixel", "refractiveindices")
    assert ie_parameter_otype("stack height") == ("pixel", "stackheight")
    assert ie_parameter_otype("pixel depth meters") == ("pixel", "depthmeters")
    assert ie_parameter_otype("pd xpos") == ("pixel", "pdxpos")
    assert ie_parameter_otype("photodetector x position") == ("pixel", "photodetectorxposition")
    assert ie_parameter_otype("pd position") == ("pixel", "pdposition")
    assert ie_parameter_otype("pixel quantum efficiency") == ("pixel", "quantumefficiency")
    assert ie_parameter_otype("photodetector quantum efficiency") == ("pixel", "photodetectorquantumefficiency")
    assert ie_parameter_otype("photodetector spectral quantum efficiency") == ("pixel", "photodetectorspectralquantumefficiency")
    assert ie_parameter_otype("diffusion MTF") == ("sensor", "diffusionmtf")
    assert ie_parameter_otype("ag") == ("sensor", "ag")
    assert ie_parameter_otype("ao") == ("sensor", "ao")
    assert ie_parameter_otype("human cone densities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("human cone seed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("mcc corner points") == ("sensor", "mcccornerpoints")
    assert ie_parameter_otype("mcc rect handles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("fnumber") == ("optics", "fnumber")
    assert ie_parameter_otype("asset light") == ("asset", "assetlight")
    assert ieParameterOtype("ip display") == ("ip", "display")


def test_ie_parameter_otype_returns_empty_type_for_ambiguous_or_unknown_params() -> None:
    assert ie_parameter_otype("size") == ("", "size")
    assert ie_parameter_otype("mystery parameter") == ("", "mysteryparameter")
