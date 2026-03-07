"""Spectral and color helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore
from .utils import least_squares_matrix, quanta_to_energy, spectral_step


def xyz_color_matching(
    wave_nm: NDArray[np.float64],
    *,
    energy: bool = False,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    store = asset_store or AssetStore.default()
    _, xyz = store.load_xyz(wave_nm=np.asarray(wave_nm, dtype=float), energy=energy)
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


def sensor_to_xyz_matrix(
    wave_nm: NDArray[np.float64],
    filter_spectra: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    xyz = xyz_color_matching(wave_nm, energy=False, asset_store=asset_store)
    return least_squares_matrix(np.asarray(filter_spectra, dtype=float), xyz)

