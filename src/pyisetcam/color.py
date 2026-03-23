"""Spectral and color helpers."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .utils import blackbody, energy_to_quanta, interp_spectra, param_format, quanta_to_energy, spectral_step, xyz_to_srgb


def xyz_color_matching(
    wave_nm: NDArray[np.float64],
    *,
    energy: bool = False,
    quanta: bool = False,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float)
    if energy and quanta:
        raise ValueError("xyz_color_matching cannot request both energy and quanta data.")
    if quanta:
        _, xyz = store.load_xyz_quanta(wave_nm=wave)
    else:
        _, xyz = store.load_xyz(wave_nm=wave, energy=energy)
    return np.asarray(xyz, dtype=float)


def luminance_from_photons(
    photons: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    energy = quanta_to_energy(np.asarray(photons, dtype=float), np.asarray(wave_nm, dtype=float))
    return luminance_from_energy(energy, wave_nm, asset_store=asset_store)


def luminance_from_energy(
    energy: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    xyz_energy = xyz_color_matching(wave_nm, energy=True, asset_store=asset_store)
    y_bar = xyz_energy[:, 1]
    return 683.0 * np.tensordot(
        np.asarray(energy, dtype=float),
        y_bar * spectral_step(np.asarray(wave_nm, dtype=float)),
        axes=([-1], [0]),
    )


def ie_xyz_from_photons(
    photons: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert photon spectra to XYZ using MATLAB ieXYZFromPhotons() semantics."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    energy = np.asarray(quanta_to_energy(np.asarray(photons, dtype=float), wave), dtype=float)
    xyz_energy = xyz_color_matching(wave, energy=True, asset_store=asset_store)
    return 683.0 * np.tensordot(energy, xyz_energy * spectral_step(wave), axes=([-1], [0]))


def ie_luminance_to_radiance(
    luminance: float,
    this_wave: float,
    *,
    sd: float = 10.0,
    wave: NDArray[np.float64] | None = None,
    asset_store: AssetStore | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Model monochromatic LED radiance from luminance using a Gaussian SPD."""

    center_wave = float(this_wave)
    if center_wave < 350.0 or center_wave > 720.0:
        raise ValueError("this_wave must be between 350 and 720 nm.")
    wave_array = np.asarray(np.arange(300.0, 771.0, 1.0, dtype=float) if wave is None else wave, dtype=float).reshape(-1)
    if wave_array.size == 0:
        raise ValueError("wave must not be empty.")
    sigma = float(sd)
    if sigma <= 0.0:
        raise ValueError("sd must be positive.")

    energy = np.exp(-0.5 * ((wave_array - center_wave) / sigma) ** 2)
    scale = float(luminance) / max(float(luminance_from_energy(energy, wave_array, asset_store=asset_store)), 1e-12)
    return np.asarray(energy * scale, dtype=float).reshape(-1), wave_array


def ie_scotopic_luminance_from_energy(
    energy: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Compute rod-weighted luminance from energy using MATLAB semantics."""

    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    _, rods = store.load_spectra("rods.mat", wave_nm=wave)
    v_prime = np.asarray(rods, dtype=float).reshape(-1)
    return 1745.0 * np.tensordot(
        np.asarray(energy, dtype=float),
        v_prime * spectral_step(wave),
        axes=([-1], [0]),
    )


def ie_responsivity_convert(
    responsivity: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    method: str = "e2q",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert responsivities between energy and quanta conventions."""

    response = np.asarray(responsivity, dtype=float)
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    if response.shape[0] != wave.size:
        raise ValueError("Responsivity rows must match the wavelength vector length.")

    peak = float(np.max(response)) if response.size else 0.0
    normalized_method = param_format(method)
    if normalized_method in {"e2q", "energy2quanta", "e2p", "energy2photons"}:
        scale = np.asarray(quanta_to_energy(np.ones((wave.size,), dtype=float), wave), dtype=float).reshape(-1)
        converted = scale[:, np.newaxis] * response
    elif normalized_method in {"q2e", "quanta2energy", "p2e", "photons2energy"}:
        scale = np.asarray(energy_to_quanta(np.ones((wave.size,), dtype=float), wave), dtype=float).reshape(-1)
        converted = scale[:, np.newaxis] * response
    else:
        raise UnsupportedOptionError("ieResponsivityConvert", method)

    if converted.size and peak > 0.0:
        converted_peak = float(np.max(converted))
        if converted_peak > 0.0:
            converted = converted * (peak / converted_peak)
    return np.asarray(converted, dtype=float), np.asarray(scale, dtype=float).reshape(-1)


def y_to_lstar(y_value: NDArray[np.float64] | float, white_y: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Convert luminance Y to CIELAB L* using MATLAB Y2Lstar() semantics."""

    ratio = np.asarray(y_value, dtype=float) / np.maximum(np.asarray(white_y, dtype=float), 1e-12)
    lstar = 116.0 * np.cbrt(ratio) - 16.0
    low = ratio < 0.008856
    if np.any(low):
        lstar = np.asarray(lstar, dtype=float)
        lstar[low] = 903.3 * ratio[low]
    return np.asarray(lstar, dtype=float)


def srgb_to_lrgb(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert nonlinear sRGB values to linear sRGB values."""

    values = np.asarray(rgb, dtype=float)
    if values.size and float(np.max(values)) > 1.0:
        warnings.warn("srgb appears to be outside the (0,1) range", RuntimeWarning, stacklevel=2)
    linear = values.copy()
    high = linear > 0.04045
    linear[~high] = linear[~high] / 12.92
    linear[high] = ((linear[high] + 0.055) / 1.055) ** 2.4
    return linear


def lrgb_to_srgb(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert linear sRGB values to nonlinear framebuffer sRGB values."""

    values = np.asarray(rgb, dtype=float)
    if values.size and (float(np.max(values)) > 1.0 or float(np.min(values)) < 0.0):
        raise ValueError("Linear rgb values must be between 0 and 1.")
    srgb = values.copy()
    high = srgb > 0.0031308
    srgb[~high] = srgb[~high] * 12.92
    srgb[high] = 1.055 * (srgb[high] ** (1.0 / 2.4)) - 0.055
    return srgb


def _xyy_to_xyz(xyy: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(xyy, dtype=float)
    reshaped = values.reshape(-1, 3)
    x = reshaped[:, 0]
    y = reshaped[:, 1]
    big_y = reshaped[:, 2]
    denominator = np.maximum(y, 1e-12)
    xyz = np.column_stack(
        [
            (x * big_y) / denominator,
            big_y,
            ((1.0 - x - y) * big_y) / denominator,
        ]
    )
    return xyz.reshape(values.shape)


def xyy_to_xyz(xyy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIE xyY values to CIE XYZ values."""

    values = np.asarray(xyy, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("xyy must have a trailing dimension of size 3.")
    return np.asarray(_xyy_to_xyz(values), dtype=float)


def ie_lab_to_xyz(lab: NDArray[np.float64], white_point: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIELAB values to XYZ using MATLAB ieLAB2XYZ() semantics."""

    lab_array = np.asarray(lab, dtype=float)
    white = np.asarray(white_point, dtype=float)
    if lab_array.shape[-1] != 3:
        raise ValueError("lab must have a trailing dimension of size 3.")
    if white.shape[-1] != 3:
        raise ValueError("white_point must have a trailing dimension of size 3.")

    delta = 6.0 / 29.0
    fy = (lab_array[..., 0] + 16.0) / 116.0
    fx = (lab_array[..., 1] / 500.0) + fy
    fz = fy - (lab_array[..., 2] / 200.0)

    def _inverse_f(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > delta, values**3, 3.0 * delta**2 * (values - (4.0 / 29.0)))

    xyz = np.empty_like(lab_array, dtype=float)
    xyz[..., 0] = _inverse_f(fx) * white[..., 0]
    xyz[..., 1] = _inverse_f(fy) * white[..., 1]
    xyz[..., 2] = _inverse_f(fz) * white[..., 2]
    return xyz


def _stockman_xyz_matrices(
    *,
    asset_store: AssetStore | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    store = asset_store or AssetStore.default()
    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    xyz = np.asarray(xyz_color_matching(wave, asset_store=store), dtype=float)
    _, lms = store.load_spectra("stockman.mat", wave_nm=wave)
    lms_array = np.asarray(lms, dtype=float)
    xyz_to_lms_matrix, _, _, _ = np.linalg.lstsq(xyz, lms_array, rcond=None)
    lms_to_xyz_matrix, _, _, _ = np.linalg.lstsq(lms_array, xyz, rcond=None)
    return np.asarray(xyz_to_lms_matrix, dtype=float), np.asarray(lms_to_xyz_matrix, dtype=float)


def _apply_tristimulus_transform(values: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    transform = np.asarray(matrix, dtype=float)
    if array.ndim >= 1 and array.shape[-1] == 3:
        reshaped = array.reshape(-1, 3)
        return np.asarray(reshaped @ transform, dtype=float).reshape(array.shape)
    if array.ndim == 2 and array.shape[0] == 3 and array.shape[1] != 3:
        return np.asarray(transform.T @ array, dtype=float)
    raise ValueError("Input must have a trailing dimension of size 3 or be a 3xN array.")


def xyz_to_lms(
    xyz: NDArray[np.float64],
    cb_type: int = 0,
    extrap_val: float = 0.0,
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert XYZ values to Stockman LMS using the direct MATLAB xyz2lms() path."""

    cb = int(cb_type)
    if cb > 0:
        raise UnsupportedOptionError("xyz2lms", f"cbType={cb_type}")
    xyz_to_lms_matrix, _ = _stockman_xyz_matrices(asset_store=asset_store)
    lms = np.asarray(_apply_tristimulus_transform(np.asarray(xyz, dtype=float), xyz_to_lms_matrix), dtype=float)
    if cb == 0:
        return lms
    channel_index = abs(cb) - 1
    if channel_index not in {0, 1, 2}:
        raise UnsupportedOptionError("xyz2lms", f"cbType={cb_type}")
    lms = np.asarray(lms, dtype=float).copy()
    if lms.ndim >= 1 and lms.shape[-1] == 3:
        lms[..., channel_index] = float(extrap_val)
        return lms
    lms[channel_index, :] = float(extrap_val)
    return lms


def lms_to_xyz(
    lms: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert Stockman LMS values to XYZ using the direct MATLAB lms2xyz() path."""

    _, lms_to_xyz_matrix = _stockman_xyz_matrices(asset_store=asset_store)
    return np.asarray(_apply_tristimulus_transform(np.asarray(lms, dtype=float), lms_to_xyz_matrix), dtype=float)


def lms_to_srgb(
    lms: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert LMS image data to sRGB for visualization."""

    from .utils import xyz_to_srgb

    xyz = np.asarray(lms_to_xyz(lms, asset_store=asset_store), dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("lms2srgb expects an RGB-format LMS image.")
    return np.asarray(xyz_to_srgb(xyz), dtype=float)


def srgb_parameters(value: str = "all") -> NDArray[np.float64]:
    """Return sRGB display parameters using MATLAB srgbParameters() semantics."""

    params = np.array(
        [
            [0.6400, 0.3000, 0.1500, 0.3127],
            [0.3300, 0.6000, 0.0600, 0.3290],
            [0.2126, 0.7152, 0.0722, 1.0000],
        ],
        dtype=float,
    )
    key = param_format(value)
    if key == "all":
        return params.copy()
    if key == "chromaticity":
        return params[:2, :3].copy()
    if key == "luminance":
        return params[2, :3].copy()
    if key == "xyywhite":
        return params[:, 3].copy()
    if key == "xyzwhite":
        return _xyy_to_xyz(params[:, 3]).reshape(3)
    raise UnsupportedOptionError("srgbParameters", value)


def adobergb_parameters(value: str = "all") -> NDArray[np.float64]:
    """Return Adobe RGB display parameters using MATLAB adobergbParameters() semantics."""

    params = np.array(
        [
            [0.64, 0.21, 0.15, 0.3127],
            [0.33, 0.71, 0.06, 0.3290],
            [47.5744, 100.3776, 12.0320, 160.0],
        ],
        dtype=float,
    )
    key = param_format(value)
    if key == "all":
        return params.copy()
    if key == "chromaticity":
        return params[:2, :3].copy()
    if key == "luminance":
        return params[2, :3].copy()
    if key == "xyywhite":
        return params[:, 3].copy()
    if key == "xyzwhite":
        return _xyy_to_xyz(params[:, 3]).reshape(3)
    if key == "xyzblack":
        return np.array([0.5282, 0.5557, 0.6052], dtype=float)
    raise UnsupportedOptionError("adobergbParameters", value)


def daylight(
    wave_nm: NDArray[np.float64],
    cct_k: float | NDArray[np.float64] = 6500.0,
    units: str = "energy",
    *,
    return_xyz: bool = False,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate CIE daylight spectra from correlated color temperature."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    cct = np.asarray(cct_k, dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("wave_nm must not be empty.")
    if cct.size == 0:
        raise ValueError("cct_k must not be empty.")
    if np.any((cct < 4000.0) | (cct >= 30000.0)):
        raise ValueError("daylight supports 4000 K <= cct_k < 30000 K.")

    lower_mask = (cct >= 4000.0) & (cct < 7000.0)
    upper_mask = cct >= 7000.0
    xdt = np.empty((2, cct.size), dtype=float)
    xdt[0, :] = (-4.6070e9 / cct**3) + (2.9678e6 / cct**2) + (0.09911e3 / cct) + 0.244063
    xdt[1, :] = (-2.0064e9 / cct**3) + (1.9018e6 / cct**2) + (0.24748e3 / cct) + 0.237040
    xd = lower_mask.astype(float) * xdt[0, :] + upper_mask.astype(float) * xdt[1, :]
    yd = (-3.0 * xd**2) + (2.87 * xd) - 0.275

    denominator = 0.0241 + (0.2562 * xd) - (0.7341 * yd)
    weights = np.empty((2, cct.size), dtype=float)
    weights[0, :] = (-1.3515 - (1.7703 * xd) + (5.9114 * yd)) / denominator
    weights[1, :] = (0.03 - (31.4424 * xd) + (30.0717 * yd)) / denominator

    store = asset_store or AssetStore.default()
    _, day_basis = store.load_spectra("cieDaylightBasis.mat", wave_nm=wave)
    basis = np.asarray(day_basis, dtype=float)
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    energy = basis[:, [0]] + basis[:, 1:3] @ weights

    normalized_units = param_format(units)
    if normalized_units in {"photons", "quanta"}:
        spectra = np.asarray(energy_to_quanta(energy, wave), dtype=float)
        first_luminance = float(luminance_from_photons(spectra[:, 0], wave, asset_store=store))
    elif normalized_units in {"energy", "watts"}:
        spectra = energy
        first_luminance = float(luminance_from_energy(spectra[:, 0], wave, asset_store=store))
    else:
        raise UnsupportedOptionError("daylight", units)
    spectra = (spectra / max(first_luminance, 1e-12)) * 100.0

    if not return_xyz:
        return spectra[:, 0] if cct.size == 1 else spectra

    xyz_energy = xyz_color_matching(wave, energy=True, asset_store=store)
    xyz = 683.0 * np.tensordot(
        np.asarray(spectra, dtype=float).T,
        xyz_energy * spectral_step(wave),
        axes=([-1], [0]),
    )
    if cct.size == 1:
        return spectra[:, 0], np.asarray(xyz, dtype=float).reshape(3)
    return spectra, np.asarray(xyz, dtype=float)


def cct_to_sun(
    wave_nm: NDArray[np.float64] | None,
    cct_k: float | NDArray[np.float64],
    units: str = "energy",
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Legacy MATLAB ``cct2sun`` compatibility wrapper."""

    wave = np.arange(400.0, 701.0, 1.0, dtype=float) if wave_nm is None else np.asarray(wave_nm, dtype=float).reshape(-1)
    return np.asarray(daylight(wave, cct_k, units, asset_store=asset_store), dtype=float)


def ie_ctemp_to_srgb(
    c_temp: float,
    *,
    wave: NDArray[np.float64] | None = None,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert a blackbody color temperature into a headless sRGB triplet."""

    wave_array = np.asarray(np.arange(400.0, 701.0, 10.0, dtype=float) if wave is None else wave, dtype=float).reshape(-1)
    energy = np.asarray(blackbody(wave_array, float(c_temp), kind="energy"), dtype=float).reshape(1, -1)
    from .metrics import xyz_from_energy

    xyz = np.asarray(xyz_from_energy(energy, wave_array, asset_store=asset_store), dtype=float).reshape(1, 1, 3)
    return np.asarray(xyz_to_srgb(xyz), dtype=float).reshape(3)


def ie_circle_points(rad_spacing: float = 2.0 * np.pi / 60.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return samples on the unit circle using the MATLAB ``ieCirclePoints`` contract."""

    spacing = float(rad_spacing)
    if spacing <= 0.0:
        raise ValueError("rad_spacing must be positive.")
    theta = np.arange(0.0, (2.0 * np.pi) + (spacing * 0.5), spacing, dtype=float)
    return np.cos(theta), np.sin(theta)


def mk_inv_gamma_table(g_table: NDArray[np.float64], num_entries: int | None = None) -> NDArray[np.float64]:
    """Compute a MATLAB-style inverse gamma lookup table."""

    gamma_table = np.asarray(g_table, dtype=float)
    if gamma_table.ndim == 1:
        gamma_table = gamma_table.reshape(-1, 1)
    if gamma_table.ndim != 2 or gamma_table.shape[0] == 0:
        raise ValueError("g_table must be a non-empty 1D or 2D gamma table.")

    entry_count = int(4 * gamma_table.shape[0] if num_entries is None else num_entries)
    if entry_count <= 0:
        raise ValueError("num_entries must be positive.")

    result = np.zeros((entry_count, gamma_table.shape[1]), dtype=float)
    target_axis = np.arange(entry_count, dtype=float) / max(entry_count - 1, 1)

    for column in range(gamma_table.shape[1]):
        this_table = np.asarray(gamma_table[:, column], dtype=float).reshape(-1)
        if np.any(np.diff(this_table) <= 0.0):
            this_table = np.sort(this_table)
            positive_locs = np.where(np.diff(this_table) > 0.0)[0] + 1
            pos_locs = np.concatenate(([0], positive_locs)).astype(float)
            monotone_table = this_table[pos_locs.astype(int)]
        else:
            monotone_table = this_table
            pos_locs = np.arange(this_table.size, dtype=float)
        result[:, column] = np.interp(target_axis, monotone_table, pos_locs)

    return result


def _surface_reflectances(
    surfaces: str,
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore,
) -> NDArray[np.float64]:
    normalized = param_format(surfaces)
    if normalized in {"mcc", "mccoptimized"}:
        _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=np.asarray(wave_nm, dtype=float))
        return np.asarray(reflectances, dtype=float)
    if normalized in {"esser", "esseroptimized"}:
        data = asset_store.load_mat("data/surfaces/charts/esser/reflectance/esserChart.mat")
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        reflectances = np.asarray(data["data"], dtype=float)
        return np.asarray(
            interp_spectra(wavelengths, reflectances, np.asarray(wave_nm, dtype=float)),
            dtype=float,
        )
    raise UnsupportedOptionError("ieColorTransform", surfaces)


def _target_qe(
    target_space: str,
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore,
) -> NDArray[np.float64]:
    normalized = param_format(target_space)
    if normalized == "xyz":
        return xyz_color_matching(wave_nm, quanta=True, asset_store=asset_store)
    raise UnsupportedOptionError("ieColorTransform", target_space)


def sensor_to_target_matrix(
    wave_nm: NDArray[np.float64],
    filter_spectra: NDArray[np.float64],
    *,
    target_space: str = "xyz",
    illuminant: str = "D65",
    surfaces: str = "mcc",
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float)
    sensor_qe = np.asarray(filter_spectra, dtype=float)
    target_qe = _target_qe(target_space, wave, asset_store=store)
    _, illuminant_energy = store.load_illuminant(illuminant, wave_nm=wave)
    illuminant_quanta = energy_to_quanta(np.asarray(illuminant_energy, dtype=float), wave)
    reflectances = _surface_reflectances(surfaces, wave, asset_store=store)
    weighted_surfaces = reflectances * illuminant_quanta.reshape(-1, 1)
    sensor_response = weighted_surfaces.T @ sensor_qe
    target_response = weighted_surfaces.T @ target_qe
    matrix, _, _, _ = np.linalg.lstsq(sensor_response, target_response, rcond=None)
    return matrix


def sensor_to_xyz_matrix(
    wave_nm: NDArray[np.float64],
    filter_spectra: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    return sensor_to_target_matrix(
        wave_nm,
        filter_spectra,
        target_space="xyz",
        illuminant="D65",
        surfaces="mcc",
        asset_store=asset_store,
    )


def internal_to_display_matrix(
    wave_nm: NDArray[np.float64],
    display_spd: NDArray[np.float64],
    *,
    internal_cs: str = "xyz",
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    normalized = param_format(internal_cs)
    if normalized != "xyz":
        raise UnsupportedOptionError("displayRender", internal_cs)
    internal_cmf = xyz_color_matching(np.asarray(wave_nm, dtype=float), energy=False, asset_store=asset_store)
    return np.linalg.inv(np.asarray(display_spd, dtype=float).T @ internal_cmf)


cct2sun = cct_to_sun
ieCTemp2SRGB = ie_ctemp_to_srgb
ieCirclePoints = ie_circle_points
mkInvGammaTable = mk_inv_gamma_table
