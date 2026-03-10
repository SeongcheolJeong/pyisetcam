"""Metric and validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore
from .color import xyz_color_matching
from .exceptions import UnsupportedOptionError
from .utils import DEFAULT_WAVE, blackbody, param_format, spectral_step


def _vector(value: Any, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _paired_arrays(reference: Any, actual: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    if reference_array.shape != actual_array.shape:
        raise ValueError("reference and actual must have the same shape.")
    return reference_array, actual_array


def _default_wave(length: int) -> NDArray[np.float64]:
    if int(length) != int(DEFAULT_WAVE.size):
        raise ValueError("wave must be provided when SPD length does not match DEFAULT_WAVE.")
    return DEFAULT_WAVE.copy()


def xyz_from_energy(
    energy: Any,
    wave_nm: Any,
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert energy spectra to XYZ tristimulus values."""

    wave = _vector(wave_nm, name="wave_nm")
    energy_array = np.asarray(energy, dtype=float)
    if energy_array.shape[-1] != wave.size:
        raise ValueError("The last energy dimension must match the wavelength vector.")
    xyz_energy = xyz_color_matching(wave, energy=True, asset_store=asset_store)
    return 683.0 * np.tensordot(energy_array, xyz_energy * spectral_step(wave), axes=([-1], [0]))


def xyz_to_lab(xyz: Any, white_point: Any) -> NDArray[np.float64]:
    """Convert XYZ tristimulus values to CIELAB."""

    xyz_array = np.asarray(xyz, dtype=float)
    white = np.asarray(white_point, dtype=float)
    if xyz_array.shape[-1] != 3:
        raise ValueError("xyz must have a trailing dimension of size 3.")
    if white.shape[-1] != 3:
        raise ValueError("white_point must have a trailing dimension of size 3.")
    ratio = xyz_array / np.maximum(white, 1e-12)
    delta = 6.0 / 29.0
    threshold = delta**3
    f_ratio = np.where(ratio > threshold, np.cbrt(ratio), (ratio / (3.0 * delta**2)) + (4.0 / 29.0))
    lab = np.empty_like(f_ratio, dtype=float)
    lab[..., 0] = (116.0 * f_ratio[..., 1]) - 16.0
    lab[..., 1] = 500.0 * (f_ratio[..., 0] - f_ratio[..., 1])
    lab[..., 2] = 200.0 * (f_ratio[..., 1] - f_ratio[..., 2])
    return lab


def xyz_to_luv(xyz: Any, white_point: Any) -> NDArray[np.float64]:
    """Convert XYZ tristimulus values to CIELUV."""

    xyz_array = np.asarray(xyz, dtype=float)
    white = np.asarray(white_point, dtype=float)
    if xyz_array.shape[-1] != 3:
        raise ValueError("xyz must have a trailing dimension of size 3.")
    if white.shape[-1] != 3:
        raise ValueError("white_point must have a trailing dimension of size 3.")
    y_ratio = xyz_array[..., 1] / np.maximum(white[..., 1], 1e-12)
    delta = 6.0 / 29.0
    threshold = delta**3
    lstar = np.where(y_ratio > threshold, (116.0 * np.cbrt(y_ratio)) - 16.0, (903.3 * y_ratio))

    def _uv_prime(values: NDArray[np.float64]) -> NDArray[np.float64]:
        denominator = np.maximum(values[..., 0] + (15.0 * values[..., 1]) + (3.0 * values[..., 2]), 1e-12)
        uv = np.empty(values.shape[:-1] + (2,), dtype=float)
        uv[..., 0] = (4.0 * values[..., 0]) / denominator
        uv[..., 1] = (9.0 * values[..., 1]) / denominator
        return uv

    uv = _uv_prime(xyz_array)
    uv_white = _uv_prime(white)
    luv = np.empty_like(xyz_array, dtype=float)
    luv[..., 0] = lstar
    luv[..., 1] = 13.0 * lstar * (uv[..., 0] - uv_white[..., 0])
    luv[..., 2] = 13.0 * lstar * (uv[..., 1] - uv_white[..., 1])
    return luv


def chromaticity_xy(xyz: Any) -> NDArray[np.float64]:
    xyz_array = np.asarray(xyz, dtype=float)
    denominator = np.maximum(np.sum(xyz_array, axis=-1, keepdims=True), 1e-12)
    return xyz_array[..., :2] / denominator


def xyz_to_uv(xyz: Any) -> NDArray[np.float64]:
    xyz_array = np.asarray(xyz, dtype=float)
    denominator = np.maximum(xyz_array[..., 0] + (15.0 * xyz_array[..., 1]) + (3.0 * xyz_array[..., 2]), 1e-12)
    uv = np.empty(xyz_array.shape[:-1] + (2,), dtype=float)
    uv[..., 0] = (4.0 * xyz_array[..., 0]) / denominator
    uv[..., 1] = (6.0 * xyz_array[..., 1]) / denominator
    return uv


def _cct_from_uv(uv: Any, *, asset_store: AssetStore | None = None) -> NDArray[np.float64]:
    uv_array = np.asarray(uv, dtype=float)
    if uv_array.shape[-1] != 2:
        raise ValueError("uv must have a trailing dimension of size 2.")
    points = uv_array.reshape(-1, 2)
    table = np.asarray((asset_store or AssetStore.default()).load_mat("color/cct.mat")["table"], dtype=float)
    temperatures = table[:, 0]
    u_table = table[:, 1][:, None]
    v_table = table[:, 2][:, None]
    slopes = table[:, 3][:, None]
    us = points[:, 0][None, :]
    vs = points[:, 1][None, :]
    distance = ((us - u_table) - (slopes * (vs - v_table))) / np.sqrt(1.0 + np.square(slopes))
    signs = np.sign(distance)
    signs = np.where(signs == 0.0, 1.0, signs)
    signs = np.vstack([signs, np.zeros((1, points.shape[0]), dtype=float)])
    transitions = np.abs(np.diff(signs, axis=0)) == 2.0
    if np.any(np.sum(transitions, axis=0) != 1):
        raise ValueError("uv coordinates are outside the supported cct.mat lookup range.")
    row_index = np.argmax(transitions, axis=0)
    column_index = np.arange(points.shape[0])
    d0 = distance[row_index, column_index]
    d1 = distance[row_index + 1, column_index]
    t0 = temperatures[row_index]
    t1 = temperatures[row_index + 1]
    cct = 1.0 / ((1.0 / t0) + (d0 / (d0 - d1)) * ((1.0 / t1) - (1.0 / t0)))
    return cct.reshape(uv_array.shape[:-1])


def cct_from_uv(
    uv: Any,
    *,
    asset_store: AssetStore | None = None,
) -> float | NDArray[np.float64]:
    """Estimate correlated color temperature from CIE 1960 uv coordinates using the upstream cct.mat table."""

    cct = _cct_from_uv(uv, asset_store=asset_store)
    return float(cct) if np.ndim(cct) == 0 else cct


def delta_e_ab(xyz1: Any, xyz2: Any, white_point: Any) -> NDArray[np.float64]:
    """Compute the CIELAB 1976 Delta E between XYZ values."""

    xyz1_array, xyz2_array = _paired_arrays(xyz1, xyz2)
    lab1 = xyz_to_lab(xyz1_array, white_point)
    lab2 = xyz_to_lab(xyz2_array, white_point)
    return np.linalg.norm(lab1 - lab2, axis=-1)


def spectral_angle(spd1: Any, spd2: Any, *, degrees: bool = True) -> float:
    """Compute the angle between two spectral vectors."""

    first = _vector(spd1, name="spd1")
    second = _vector(spd2, name="spd2")
    if first.size != second.size:
        raise ValueError("spd1 and spd2 must have the same length.")
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    return float(np.rad2deg(angle)) if degrees else angle


def correlated_color_temperature(
    xyz: Any,
    *,
    asset_store: AssetStore | None = None,
) -> float | NDArray[np.float64]:
    """Estimate correlated color temperature from XYZ using the upstream cct.mat lookup table."""

    cct = cct_from_uv(xyz_to_uv(np.asarray(xyz, dtype=float)), asset_store=asset_store)
    return float(cct) if np.ndim(cct) == 0 else cct


def mired_difference(cct1_k: float, cct2_k: float) -> float:
    """Compute the absolute mired difference between two color temperatures."""

    return float(abs((1.0 / max(float(cct2_k), 1e-12)) - (1.0 / max(float(cct1_k), 1e-12))) * 1e6)


def mean_absolute_error(reference: Any, actual: Any) -> float:
    reference_array, actual_array = _paired_arrays(reference, actual)
    return float(np.mean(np.abs(actual_array - reference_array)))


def root_mean_squared_error(reference: Any, actual: Any) -> float:
    reference_array, actual_array = _paired_arrays(reference, actual)
    return float(np.sqrt(np.mean(np.square(actual_array - reference_array))))


def mean_relative_error(reference: Any, actual: Any, *, epsilon: float = 1e-12) -> float:
    reference_array, actual_array = _paired_arrays(reference, actual)
    denominator = np.maximum(np.abs(reference_array), float(epsilon))
    return float(np.mean(np.abs(actual_array - reference_array) / denominator))


def peak_signal_to_noise_ratio(reference: Any, actual: Any, *, data_range: float | None = None) -> float:
    reference_array, actual_array = _paired_arrays(reference, actual)
    rmse = root_mean_squared_error(reference_array, actual_array)
    if np.isclose(rmse, 0.0):
        return float(np.inf)
    if data_range is None:
        lower = min(float(np.min(reference_array)), float(np.min(actual_array)))
        upper = max(float(np.max(reference_array)), float(np.max(actual_array)))
        data_range = upper - lower
    if not np.isfinite(data_range) or float(data_range) <= 0.0:
        data_range = 1.0
    return float(20.0 * np.log10(float(data_range) / rmse))


def comparison_metrics(reference: Any, actual: Any, *, data_range: float | None = None) -> dict[str, float]:
    reference_array, actual_array = _paired_arrays(reference, actual)
    return {
        "mae": mean_absolute_error(reference_array, actual_array),
        "rmse": root_mean_squared_error(reference_array, actual_array),
        "mean_rel": mean_relative_error(reference_array, actual_array),
        "max_abs": float(np.max(np.abs(actual_array - reference_array))),
        "psnr": peak_signal_to_noise_ratio(reference_array, actual_array, data_range=data_range),
    }


def metrics_spd(
    spd1: Any,
    spd2: Any,
    *,
    metric: str = "angle",
    luminance: float = 100.0,
    white_point: Any | None = None,
    wave: Any | None = None,
    asset_store: AssetStore | None = None,
    return_params: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Compare two spectral power distributions using MATLAB-style metrics."""

    first = _vector(spd1, name="spd1")
    second = _vector(spd2, name="spd2")
    if first.size != second.size:
        raise ValueError("spd1 and spd2 must have the same length.")
    wave_array = _default_wave(first.size) if wave is None else _vector(wave, name="wave")
    if wave_array.size != first.size:
        raise ValueError("wave must match the SPD length.")
    normalized_metric = param_format(metric)
    params: dict[str, Any] = {}

    if normalized_metric == "angle":
        value = spectral_angle(first, second, degrees=True)
        return (value, params) if return_params else value

    if normalized_metric == "cielab":
        xyz1 = xyz_from_energy(first, wave_array, asset_store=asset_store)
        xyz2 = xyz_from_energy(second, wave_array, asset_store=asset_store)
        spd1_scaled = first * (float(luminance) / max(float(xyz1[1]), 1e-12))
        spd2_scaled = second * (float(luminance) / max(float(xyz2[1]), 1e-12))
        xyz1 = xyz_from_energy(spd1_scaled, wave_array, asset_store=asset_store)
        xyz2 = xyz_from_energy(spd2_scaled, wave_array, asset_store=asset_store)
        white_xyz = xyz_from_energy(spd1_scaled, wave_array, asset_store=asset_store) if white_point is None else np.asarray(white_point, dtype=float)
        if white_point is None:
            white_xyz = white_xyz * (float(luminance) / max(float(white_xyz[1]), 1e-12))
        lab1 = xyz_to_lab(xyz1, white_xyz)
        lab2 = xyz_to_lab(xyz2, white_xyz)
        value = float(np.linalg.norm(lab1 - lab2))
        params = {
            "xyz1": xyz1,
            "xyz2": xyz2,
            "lab1": lab1,
            "lab2": lab2,
            "white_point": np.asarray(white_xyz, dtype=float),
        }
        return (value, params) if return_params else value

    if normalized_metric in {"mired", "cct"}:
        xyz1 = xyz_from_energy(first, wave_array, asset_store=asset_store)
        xyz2 = xyz_from_energy(second, wave_array, asset_store=asset_store)
        cct_values = np.array(
            [
                correlated_color_temperature(xyz1, asset_store=asset_store),
                correlated_color_temperature(xyz2, asset_store=asset_store),
            ],
            dtype=float,
        )
        value = mired_difference(float(cct_values[0]), float(cct_values[1]))
        params = {
            "xyz": np.vstack([xyz1, xyz2]),
            "uv": np.column_stack([xyz_to_uv(xyz1), xyz_to_uv(xyz2)]),
            "cct_k": cct_values,
        }
        return (value, params) if return_params else value

    raise UnsupportedOptionError("metricsSPD", metric)


def example_spd_pair(*, wave: Any | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a deterministic SPD pair for documentation/examples."""

    wave_array = DEFAULT_WAVE.copy() if wave is None else _vector(wave, name="wave")
    return (
        wave_array,
        np.asarray(blackbody(wave_array, 6500.0, kind="energy"), dtype=float),
        np.asarray(blackbody(wave_array, 5000.0, kind="energy"), dtype=float),
    )


# MATLAB-style aliases.
ieXYZFromEnergy = xyz_from_energy
ieXYZ2LAB = xyz_to_lab
xyz2luv = xyz_to_luv
deltaEab = delta_e_ab
iePSNR = peak_signal_to_noise_ratio
metricsSPD = metrics_spd
