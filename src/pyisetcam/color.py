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
    xyz_energy = xyz_color_matching(wave_nm, energy=True, asset_store=asset_store)
    energy = quanta_to_energy(np.asarray(photons, dtype=float), np.asarray(wave_nm, dtype=float))
    y_bar = xyz_energy[:, 1]
    return 683.0 * np.tensordot(energy, y_bar * spectral_step(np.asarray(wave_nm, dtype=float)), axes=([2], [0]))


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
