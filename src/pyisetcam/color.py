"""Spectral and color helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .utils import energy_to_quanta, param_format, quanta_to_energy, spectral_step


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
        _, reflectances = asset_store.load_reflectances("esserChart.mat", wave_nm=np.asarray(wave_nm, dtype=float))
        return np.asarray(reflectances, dtype=float)
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
