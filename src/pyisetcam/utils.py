"""Shared numerical utilities."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom

DEFAULT_WAVE = np.arange(400.0, 701.0, 10.0, dtype=float)

_PLANCK = 6.626176e-34
_LIGHT_SPEED = 299792458.0
_BOLTZMANN = 1.380662e-23

_PARAMETER_OTYPE_PREFIXES = {
    "scene": "scene",
    "oi": "oi",
    "optics": "optics",
    "wvf": "wvf",
    "sensor": "sensor",
    "pixel": "pixel",
    "vci": "ip",
    "ip": "ip",
    "display": "display",
    "l3": "l3",
}

_PARAMETER_OTYPE_UNIQUE = {
    "scene": {
        "objectdistance",
        "meanluminance",
        "luminance",
        "spatialsupportlinear",
        "spatialsupport",
        "radiancehline",
        "radiancevline",
        "hlineradiance",
        "vlineradiance",
        "luminancehline",
        "luminancevline",
        "hlineluminance",
        "vlineluminance",
        "illuminantcomment",
        "roiilluminantphotons",
        "roimeanilluminantphotons",
        "roiilluminantenergy",
        "roimeanilluminantenergy",
        "illuminanthlinephotons",
        "illuminantvlinephotons",
        "hlineilluminantphotons",
        "vlineilluminantphotons",
        "illuminanthlineenergy",
        "illuminantvlineenergy",
        "hlineilluminantenergy",
        "vlineilluminantenergy",
        "roimeanluminance",
        "chromaticity",
        "roichromaticitymean",
        "illuminant",
        "illuminantname",
        "illuminantenergy",
        "illuminantphotons",
        "illuminantxyz",
        "illuminantwave",
        "illuminantcomment",
        "illuminantformat",
    },
    "oi": {
        "optics",
        "opticsmodel",
        "irradiancehline",
        "irradiancevline",
        "irradianceenergyhline",
        "irradianceenergyvline",
        "hline",
        "vline",
        "hlineirradiance",
        "vlineirradiance",
        "hlineenergy",
        "vlineenergy",
        "hlineirradianceenergy",
        "vlineirradianceenergy",
        "illuminancehline",
        "illuminancevline",
        "hlineilluminance",
        "vlineilluminance",
        "chromaticity",
        "roichromaticitymean",
        "roiilluminance",
        "roimeanilluminance",
        "diffusermethod",
        "diffuserblur",
        "psfstruct",
        "sampledrtpsf",
        "psfsampleangles",
        "psfanglestep",
        "psfimageheights",
        "raytraceopticsname",
    },
    "optics": {
        "fnumber",
        "effectivefnumber",
        "focallength",
        "power",
        "imagedistance",
        "imageheight",
        "imagewidth",
        "numericalaperture",
        "aperturedameter",
        "apertureradius",
        "magnification",
        "pupilmagnification",
        "offaxismethod",
        "cos4thmethod",
        "cos4thdata",
        "otfdata",
        "otfsize",
        "otffx",
        "otffy",
        "otfsupport",
        "psfdata",
        "psfspacing",
        "psfsupport",
        "incoherentcutoffspatialfrequency",
        "maxincoherentcutoffspatialfrequency",
        "rtname",
        "raytrace",
        "rtopticsprogram",
        "rtlensfile",
        "rteffectivefnumber",
        "rtfnumber",
        "rtmagnification",
        "rtreferencewavelength",
        "rtobjectdistance",
        "rtfieldofview",
        "rteffectivefocallength",
        "rtpsf",
        "rtpsfdata",
        "rtpsfsize",
        "rtpsfwavelength",
        "rtpsffieldheight",
        "rtpsfsamplespacing",
        "rtpsfsupport",
        "rtpsfsupportrow",
        "rtpsfsupportcol",
        "rtotfdata",
        "rtrelillum",
        "rtrifunction",
        "rtriwavelength",
        "rtrifieldheight",
        "rtgeometry",
        "rtgeomfunction",
        "rtgeomwavelength",
        "rtgeomfieldheight",
        "rtgeommaxfieldheight",
    },
    "wvf": {
        "zcoeffs",
        "constantsampleintervaldomain",
        "refsizeoffieldmm",
    },
    "sensor": {
        "chiefrayangle",
        "chiefrayangledegrees",
        "sensoretendue",
        "microlens",
        "volts",
        "voltage",
        "voltimages",
        "digitalvalue",
        "digitalvalues",
        "electrons",
        "electron",
        "electronsperarea",
        "dvorvolts",
        "digitalorvolts",
        "dvorvoltsroielectrons",
        "roivoltsmean",
        "roielectronsmean",
        "chromaticity",
        "roichromaticitymean",
        "hlinevolts",
        "hlineelectrons",
        "hlinedv",
        "vlinevolts",
        "vlineelectrons",
        "vlinedv",
        "responseratio",
        "responsedr",
        "dr",
        "drdb20",
        "dynamicrange",
        "diffusionmtf",
        "analoggain",
        "ag",
        "analogoffset",
        "ao",
        "sensordynamicrange",
        "quantization",
        "nbits",
        "maxoutput",
        "quantizatonlut",
        "quantizationmethod",
        "filtertransmissivities",
        "infraredfilter",
        "cfaname",
        "filternames",
        "nfilters",
        "filtercolorletters",
        "filtercolorletterscell",
        "filterplotcolors",
        "spectralqe",
        "sensorspectralsr",
        "pattern",
        "dsnusigma",
        "prnusigma",
        "fpnparameters",
        "dsnuimage",
        "prnuimage",
        "columnfpn",
        "columndsnu",
        "columnprnu",
        "coloffsetfpnvector",
        "colgainfpnvector",
        "blacklevel",
        "zerolevel",
        "noiseflag",
        "shotnoiseflag",
        "reusenoise",
        "noiseseed",
        "pixel",
        "autoexpsoure",
        "exposuretime",
        "uniqueexptime",
        "exposureplane",
        "cds",
        "vignetting",
        "nsamplesperpixel",
        "sensormovement",
        "movementpositions",
        "framesperpositions",
        "sensorpositionsx",
        "sensorpositionsy",
        "human",
        "humanconetype",
        "humanconedensities",
        "humanconelocs",
        "humanrseed",
        "humanconeseed",
        "mccrecthandles",
        "mcccornerpoints",
    },
    "pixel": {
        "pdsize",
        "fillfactor",
        "pdarea",
        "pdspectralqe",
        "conversiongain",
        "voltageswing",
        "wellcapacity",
        "darkcurrentdensity",
        "darkcurrent",
        "darkvoltage",
        "darkelectrons",
        "readnoiseelectrons",
        "readnoisevolts",
        "readnoisemillivolts",
        "pdspectralsr",
        "pixeldr",
    },
    "ip": {
        "chromaticity",
        "roichromaticitymean",
        "render",
        "colorbalance",
        "colorbalancemethod",
        "demosaic",
        "demosaicmethod",
        "colorconversion",
        "colorconversionmethod",
        "internalcolorspace",
        "internalcolormatchingfunciton",
        "display",
        "displayxyz",
        "displayxy",
        "displaywhitepoint",
        "displaymaxluminance",
        "displayspd",
        "displaygamma",
        "displaymaxrgb",
        "displaydpi",
        "displayviewingdistance",
        "l3",
    },
    "l3": {
        "trainingilluminant",
        "clusters",
        "filters",
        "sensorpatches",
    },
    "asset": {
        "assetobject",
        "assetbranch",
        "assetlight",
    },
}


def param_format(value: Any) -> Any:
    """Mirror ISETCam's ieParamFormat behavior."""

    if isinstance(value, (int, float, complex, np.number, np.ndarray, bool)):
        return value
    if isinstance(value, str):
        return value.lower().replace(" ", "")
    if isinstance(value, list):
        formatted = list(value)
        for index in range(0, len(formatted), 2):
            formatted[index] = param_format(formatted[index])
        return formatted
    if isinstance(value, tuple):
        formatted = list(value)
        for index in range(0, len(formatted), 2):
            formatted[index] = param_format(formatted[index])
        return tuple(formatted)
    return value


def ie_parameter_otype(param: str) -> tuple[str, str | None]:
    """Infer the ISET object type associated with a parameter string."""

    if not param:
        raise ValueError("param is required")

    normalized = param_format(param)
    direct = _PARAMETER_OTYPE_PREFIXES.get(normalized)
    if direct is not None:
        return direct, None

    positions = [idx for idx in (param.find(" "), param.find("/"), param.find("_")) if idx >= 0]
    if positions:
        pos = min(positions)
        prefix = _PARAMETER_OTYPE_PREFIXES.get(param_format(param[:pos]))
        if prefix is not None:
            return prefix, param_format(param[(pos + 1) :])

    for object_type, unique_params in _PARAMETER_OTYPE_UNIQUE.items():
        if normalized in unique_params:
            return object_type, normalized
    return "", normalized


def split_prefixed_parameter(parameter: Any, prefixes: tuple[str, ...]) -> tuple[str | None, str]:
    """Split MATLAB-style object-prefixed parameter names.

    Examples:
        ``display spd`` -> (``display``, ``spd``)
        ``display/spd`` -> (``display``, ``spd``)
        ``displayspd`` -> (``display``, ``spd``)
    """

    if not isinstance(parameter, str):
        return None, ""

    normalized = param_format(parameter)
    ordered_prefixes = sorted((param_format(prefix) for prefix in prefixes), key=len, reverse=True)
    for prefix in ordered_prefixes:
        if normalized == prefix:
            return prefix, ""
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :]
            remainder = re.sub(r"^[/\\-]+", "", remainder)
            return prefix, remainder
    return None, normalized


def spectral_step(wave_nm: NDArray[np.float64]) -> float:
    wave_nm = np.asarray(wave_nm, dtype=float).reshape(-1)
    if wave_nm.size < 2:
        return 1.0
    return float(np.mean(np.diff(wave_nm)))


def unit_frequency_list(sample_count: int) -> NDArray[np.float64]:
    """Mirror ISETCam's unitFrequencyList() normalization and DC placement."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if count % 2:
        middle = (count + 1) // 2
    else:
        middle = (count // 2) + 1
    coordinates = np.arange(1, count + 1, dtype=float)
    coordinates = coordinates - coordinates[middle - 1]
    return coordinates / np.max(np.abs(coordinates))


def interp_spectra(
    source_wave_nm: NDArray[np.float64],
    values: NDArray[np.float64],
    target_wave_nm: NDArray[np.float64],
    *,
    left: float = 0.0,
    right: float = 0.0,
) -> NDArray[np.float64]:
    """Interpolate spectral columns onto a new wavelength basis."""

    source_wave_nm = np.asarray(source_wave_nm, dtype=float).reshape(-1)
    target_wave_nm = np.asarray(target_wave_nm, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    flattened = values.reshape(source_wave_nm.size, -1)
    interpolated = np.empty((target_wave_nm.size, flattened.shape[1]), dtype=float)
    for column in range(flattened.shape[1]):
        interpolated[:, column] = np.interp(
            target_wave_nm,
            source_wave_nm,
            flattened[:, column],
            left=left,
            right=right,
        )
    return interpolated.reshape((target_wave_nm.size, *values.shape[1:]))


def quanta_to_energy(quanta: NDArray[np.float64], wave_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert photons/quanta to energy."""

    wave_m = np.asarray(wave_nm, dtype=float) * 1e-9
    return np.asarray(quanta, dtype=float) * (_PLANCK * _LIGHT_SPEED / wave_m)


def energy_to_quanta(energy: NDArray[np.float64], wave_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert energy to photons/quanta."""

    wave_m = np.asarray(wave_nm, dtype=float) * 1e-9
    return np.asarray(energy, dtype=float) * wave_m / (_PLANCK * _LIGHT_SPEED)


def blackbody(
    wave_nm: NDArray[np.float64],
    temperature_k: float | NDArray[np.float64],
    *,
    kind: str = "energy",
    eq_wave_nm: float = 550.0,
) -> NDArray[np.float64]:
    """Approximate ISETCam's blackbody() scaling and unit conventions."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    temps = np.asarray(temperature_k, dtype=float).reshape(-1)
    wave_m = wave[:, None] * 1e-9

    # ISETCam uses pre-combined constants in Planck's law and reports
    # spectral radiance in W / (m^2 nm sr) before optional photon conversion.
    c1 = 3.741832e-16
    c2 = 1.438786e-2
    spec_emit = c1 / (wave_m**5 * (np.exp(c2 / (wave_m * temps[None, :])) - 1.0))
    spec_rad = spec_emit * 1e-9 / np.pi

    idx = int(np.argmin(np.abs(wave - float(eq_wave_nm))))

    from .assets import AssetStore

    store = AssetStore.default()
    _, xyz_energy = store.load_xyz(wave_nm=wave, energy=True)
    y_bar = np.asarray(xyz_energy, dtype=float)[:, 1]
    delta = spectral_step(wave)

    if param_format(kind) in {"watts", "energy"}:
        luminance = 683.0 * float(np.sum(spec_rad[:, 0] * y_bar * delta))
        spec_rad = spec_rad * (100.0 / max(luminance, 1e-12))
        scale = spec_rad[idx, 0] / np.maximum(spec_rad[idx, :], 1e-12)
        spec_rad = spec_rad * scale.reshape(1, -1)
        return spec_rad[:, 0] if temps.size == 1 else spec_rad

    if param_format(kind) in {"photons", "quanta"}:
        scale = spec_rad[idx, 0] / np.maximum(spec_rad[idx, :], 1e-12)
        spec_rad = spec_rad * scale.reshape(1, -1)
        luminance = 683.0 * float(np.sum(spec_rad[:, 0] * y_bar * delta))
        spec_rad = spec_rad * (100.0 / max(luminance, 1e-12))
        photons = energy_to_quanta(spec_rad, wave)
        return photons[:, 0] if temps.size == 1 else photons

    raise ValueError(f"Unsupported blackbody unit type: {kind}")


def resample_cube(cube: NDArray[np.float64], new_shape: tuple[int, int]) -> NDArray[np.float64]:
    """Resample a spectral image cube over rows and columns."""

    if cube.shape[:2] == new_shape:
        return np.asarray(cube, dtype=float).copy()
    factors = (
        new_shape[0] / cube.shape[0],
        new_shape[1] / cube.shape[1],
        1.0,
    )
    return zoom(np.asarray(cube, dtype=float), factors, order=1)


def gaussian_sigma_pixels(
    f_number: float,
    wavelength_nm: float,
    sample_spacing_m: float,
    *,
    extra_blur_pixels: float = 0.0,
) -> float:
    """Approximate diffraction blur as a Gaussian sigma in pixel units."""

    airy_radius_m = 1.22 * wavelength_nm * 1e-9 * f_number
    sigma_pixels = airy_radius_m / max(sample_spacing_m, 1e-12) / 2.355
    return float(max(0.0, sigma_pixels + extra_blur_pixels))


def apply_channelwise_gaussian(
    cube: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    *,
    mode: str = "constant",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Blur each wavelength slice with its own sigma."""

    output = np.empty_like(cube, dtype=float)
    for index, sigma in enumerate(np.asarray(sigmas, dtype=float).reshape(-1)):
        output[:, :, index] = gaussian_filter(cube[:, :, index], sigma=sigma, mode=mode, cval=cval)
    return output


def least_squares_matrix(
    source_sensitivities: NDArray[np.float64],
    target_sensitivities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the linear transform that maps one spectral basis to another."""

    matrix, _, _, _ = np.linalg.lstsq(
        np.asarray(source_sensitivities, dtype=float),
        np.asarray(target_sensitivities, dtype=float),
        rcond=None,
    )
    return matrix


def xyz_to_linear_srgb(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIE XYZ to linear sRGB."""

    transform = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=float,
    )
    reshaped = np.asarray(xyz, dtype=float).reshape(-1, 3)
    rgb = reshaped @ transform.T
    return rgb.reshape(np.asarray(xyz).shape)


def linear_to_srgb(linear_rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the standard sRGB transfer curve."""

    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, None)
    threshold = 0.0031308
    srgb = np.where(
        linear_rgb <= threshold,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def invert_gamma_table(linear_rgb: NDArray[np.float64], gamma_table: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a display gamma LUT."""

    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, 1.0)
    gamma_table = np.asarray(gamma_table, dtype=float)
    digital_axis = np.linspace(0.0, 1.0, gamma_table.shape[0])
    output = np.empty_like(linear_rgb, dtype=float)
    for channel in range(linear_rgb.shape[-1]):
        output[..., channel] = np.interp(
            linear_rgb[..., channel],
            gamma_table[:, channel],
            digital_axis,
            left=0.0,
            right=1.0,
        )
    return output


def tile_pattern(pattern: NDArray[np.int_], rows: int, cols: int) -> NDArray[np.int_]:
    """Tile a CFA pattern to image size."""

    pattern = np.asarray(pattern, dtype=int)
    tiled = np.tile(
        pattern,
        (
            int(np.ceil(rows / pattern.shape[0])),
            int(np.ceil(cols / pattern.shape[1])),
        ),
    )
    return tiled[:rows, :cols]


def ensure_multiple(value: int, multiple: int) -> int:
    """Round up to the next multiple."""

    if multiple <= 1:
        return int(value)
    return int(np.ceil(value / multiple) * multiple)


def array_percentile(value: NDArray[np.float64], percentile: float) -> float:
    return float(np.percentile(np.asarray(value, dtype=float), percentile))
