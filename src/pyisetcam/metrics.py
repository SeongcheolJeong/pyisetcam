"""Metric and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat
from skimage.color import deltaE_ciede2000, deltaE_ciede94

from .assets import AssetStore
from .color import xyz_color_matching
from .exceptions import UnsupportedOptionError
from .utils import DEFAULT_WAVE, blackbody, param_format, rgb_to_xw_format, spectral_step, srgb_to_xyz


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


def _paired_color_vectors(
    reference: Any,
    actual: Any,
    *,
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
    reference_array, actual_array = _paired_arrays(reference, actual)
    if reference_array.ndim < 2 or reference_array.shape[-1] != 3:
        raise ValueError(f"{name} inputs must have a trailing dimension of size 3.")
    return (
        reference_array.reshape(-1, 3),
        actual_array.reshape(-1, 3),
        reference_array.shape[:-1],
    )


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


def delta_e_ab(
    xyz1: Any,
    xyz2: Any,
    white_point: Any,
    delta_e_version: str = "1976",
) -> NDArray[np.float64]:
    """Compute CIELAB Delta E between XYZ values."""

    xyz1_array, xyz2_array = _paired_arrays(xyz1, xyz2)
    lab1 = xyz_to_lab(xyz1_array, white_point)
    lab2 = xyz_to_lab(xyz2_array, white_point)
    normalized_version = param_format(delta_e_version)
    if normalized_version in {"1976", "76", "cie1976"}:
        return np.linalg.norm(lab1 - lab2, axis=-1)
    if normalized_version in {"1994", "94", "cie1994"}:
        return np.asarray(deltaE_ciede94(lab1, lab2), dtype=float)
    if normalized_version in {"2000", "00", "cie2000", "ciede2000"}:
        return np.asarray(deltaE_ciede2000(lab1, lab2), dtype=float)
    raise UnsupportedOptionError("deltaEab", delta_e_version)


def delta_e_2000(
    lab_std: Any,
    lab_sample: Any,
    klch: Any | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Legacy MATLAB deltaE2000() compatibility wrapper."""

    lab_std_flat, lab_sample_flat, out_shape = _paired_color_vectors(lab_std, lab_sample, name="deltaE2000")
    if klch is None:
        kl, kc, kh = 1.0, 1.0, 1.0
    else:
        weights = np.asarray(klch, dtype=float).reshape(-1)
        if weights.size != 3:
            raise ValueError("deltaE2000 requires KLCH to be a three-element vector.")
        kl, kc, kh = (float(weights[0]), float(weights[1]), float(weights[2]))

    l_std = lab_std_flat[:, 0]
    a_std = lab_std_flat[:, 1]
    b_std = lab_std_flat[:, 2]
    c_std = np.sqrt(a_std**2 + b_std**2)

    l_sample = lab_sample_flat[:, 0]
    a_sample = lab_sample_flat[:, 1]
    b_sample = lab_sample_flat[:, 2]
    c_sample = np.sqrt(a_sample**2 + b_sample**2)

    c_mean = (c_std + c_sample) / 2.0
    g = 0.5 * (1.0 - np.sqrt((c_mean**7) / np.maximum((c_mean**7) + (25.0**7), 1.0e-12)))

    a_std_prime = (1.0 + g) * a_std
    a_sample_prime = (1.0 + g) * a_sample
    c_std_prime = np.sqrt(a_std_prime**2 + b_std**2)
    c_sample_prime = np.sqrt(a_sample_prime**2 + b_sample**2)
    c_prod = c_std_prime * c_sample_prime
    zero_chroma = c_prod == 0.0

    h_std_prime = np.arctan2(b_std, a_std_prime)
    h_std_prime = h_std_prime + (2.0 * np.pi * (h_std_prime < 0.0))
    h_std_prime[(np.abs(a_std_prime) + np.abs(b_std)) == 0.0] = 0.0

    h_sample_prime = np.arctan2(b_sample, a_sample_prime)
    h_sample_prime = h_sample_prime + (2.0 * np.pi * (h_sample_prime < 0.0))
    h_sample_prime[(np.abs(a_sample_prime) + np.abs(b_sample)) == 0.0] = 0.0

    delta_l = l_sample - l_std
    delta_c = c_sample_prime - c_std_prime

    delta_h_prime = h_sample_prime - h_std_prime
    delta_h_prime = delta_h_prime - (2.0 * np.pi * (delta_h_prime > np.pi))
    delta_h_prime = delta_h_prime + (2.0 * np.pi * (delta_h_prime < -np.pi))
    delta_h_prime[zero_chroma] = 0.0
    delta_h = 2.0 * np.sqrt(c_prod) * np.sin(delta_h_prime / 2.0)

    l_prime = (l_sample + l_std) / 2.0
    c_prime = (c_std_prime + c_sample_prime) / 2.0
    h_prime = (h_std_prime + h_sample_prime) / 2.0
    h_prime = h_prime - ((np.abs(h_std_prime - h_sample_prime) > np.pi) * np.pi)
    h_prime = h_prime + ((h_prime < 0.0) * 2.0 * np.pi)
    h_prime[zero_chroma] = h_sample_prime[zero_chroma] + h_std_prime[zero_chroma]

    lp_minus_50_sq = (l_prime - 50.0) ** 2
    s_l = 1.0 + (0.015 * lp_minus_50_sq / np.sqrt(20.0 + lp_minus_50_sq))
    s_c = 1.0 + (0.045 * c_prime)
    t = (
        1.0
        - 0.17 * np.cos(h_prime - (np.pi / 6.0))
        + 0.24 * np.cos(2.0 * h_prime)
        + 0.32 * np.cos((3.0 * h_prime) + (np.pi / 30.0))
        - 0.20 * np.cos((4.0 * h_prime) - (63.0 * np.pi / 180.0))
    )
    s_h = 1.0 + (0.015 * c_prime * t)
    delta_theta = (30.0 * np.pi / 180.0) * np.exp(-(((180.0 / np.pi * h_prime) - 275.0) / 25.0) ** 2)
    r_c = 2.0 * np.sqrt((c_prime**7) / np.maximum((c_prime**7) + (25.0**7), 1.0e-12))
    r_t = -np.sin(2.0 * delta_theta) * r_c

    d_l = delta_l / (kl * s_l)
    d_c = delta_c / (kc * s_c)
    d_h = delta_h / (kh * s_h)
    interaction = r_t * d_c * d_h
    delta_e = np.sqrt(np.maximum((d_l**2) + (d_c**2) + (d_h**2) + interaction, 0.0))

    reshaped_delta_e = np.asarray(delta_e, dtype=float).reshape(out_shape)
    components = {
        "dL": np.asarray(d_l, dtype=float).reshape(out_shape),
        "dC": np.asarray(d_c, dtype=float).reshape(out_shape),
        "dH": np.asarray(d_h, dtype=float).reshape(out_shape),
        "RT": np.asarray(interaction, dtype=float).reshape(out_shape),
    }
    return reshaped_delta_e, components


def delta_e_94(
    lab1: Any,
    lab2: Any,
    k: Any | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Legacy MATLAB deltaE94() compatibility wrapper."""

    lab1_flat, lab2_flat, out_shape = _paired_color_vectors(lab1, lab2, name="deltaE94")
    if k is None:
        k_l, k_c, k_h = 1.0, 1.0, 1.0
    else:
        weights = np.asarray(k, dtype=float).reshape(-1)
        if weights.size != 3:
            raise ValueError("deltaE94 requires k to be a three-element vector.")
        k_l, k_c, k_h = (float(weights[0]), float(weights[1]), float(weights[2]))

    c_ab1 = np.sqrt(lab1_flat[:, 1] ** 2 + lab1_flat[:, 2] ** 2)
    c_ab2 = np.sqrt(lab2_flat[:, 1] ** 2 + lab2_flat[:, 2] ** 2)
    delta_c = c_ab1 - c_ab2
    delta_l = lab1_flat[:, 0] - lab2_flat[:, 0]
    delta_e_76 = np.sqrt(np.sum((lab1_flat - lab2_flat) ** 2, axis=1))
    delta_h_sq = np.maximum((delta_e_76**2) - (delta_l**2) - (delta_c**2), 0.0)
    delta_h = np.sqrt(delta_h_sq)

    s_l = np.ones_like(c_ab1, dtype=float)
    s_c = 1.0 + (0.045 * c_ab1)
    s_h = 1.0 + (0.015 * c_ab1)

    d_l = delta_l / (s_l * k_l)
    d_c = delta_c / (s_c * k_c)
    d_h = delta_h / (s_h * k_h)
    delta_e = np.sqrt(np.maximum((d_l**2) + (d_c**2) + (d_h**2), 0.0))

    reshaped_delta_e = np.asarray(delta_e, dtype=float).reshape(out_shape)
    components = {
        "dL": np.asarray(delta_l / s_l, dtype=float).reshape(out_shape),
        "dC": np.asarray(delta_c / s_c, dtype=float).reshape(out_shape),
        "dH": np.asarray(delta_h / s_h, dtype=float).reshape(out_shape),
    }
    return reshaped_delta_e, components


def delta_e_uv(
    xyz1: Any,
    xyz2: Any,
    white_point: Any,
) -> NDArray[np.float64]:
    """Legacy MATLAB deltaEuv() compatibility wrapper."""

    xyz1_array, xyz2_array = _paired_arrays(xyz1, xyz2)
    if xyz1_array.ndim < 2 or xyz1_array.shape[-1] != 3:
        raise ValueError("deltaEuv expects XYZ inputs with a trailing dimension of size 3.")

    if isinstance(white_point, (list, tuple)) and len(white_point) == 2:
        luv1 = xyz_to_luv(xyz1_array, white_point[0])
        luv2 = xyz_to_luv(xyz2_array, white_point[1])
    else:
        luv1 = xyz_to_luv(xyz1_array, white_point)
        luv2 = xyz_to_luv(xyz2_array, white_point)
    return np.sqrt(np.sum((luv1 - luv2) ** 2, axis=-1))


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


def spd_to_cct(
    wave_nm: Any,
    spd: Any,
    *,
    asset_store: AssetStore | None = None,
) -> float | NDArray[np.float64]:
    """Estimate correlated color temperature from spectral power distributions."""

    wave = _vector(wave_nm, name="wave_nm")
    spd_array = np.asarray(spd, dtype=float)
    if spd_array.ndim == 0:
        raise ValueError("spd must not be scalar.")
    if spd_array.ndim == 1:
        if spd_array.size != wave.size:
            raise ValueError("spd must match wave_nm length.")
        xyz = xyz_from_energy(spd_array, wave, asset_store=asset_store)
        return correlated_color_temperature(xyz, asset_store=asset_store)
    if spd_array.shape[0] == wave.size:
        spectra = np.moveaxis(spd_array, 0, -1)
    elif spd_array.shape[-1] == wave.size:
        spectra = spd_array
    else:
        raise ValueError("spd must have a spectral dimension matching wave_nm.")
    xyz = xyz_from_energy(spectra, wave, asset_store=asset_store)
    cct = correlated_color_temperature(xyz, asset_store=asset_store)
    return float(cct) if np.ndim(cct) == 0 else np.asarray(cct, dtype=float)


def srgb_to_color_temp(
    rgb: Any,
    method: str = "bright",
    *,
    return_table: bool = False,
    asset_store: AssetStore | None = None,
) -> float | tuple[float, NDArray[np.float64]]:
    """Estimate color temperature from an sRGB image using the upstream bright-pixel chromaticity heuristic."""

    rgb_input = np.asarray(rgb)
    if rgb_input.ndim < 3 or rgb_input.shape[-1] != 3:
        raise ValueError("rgb must have a trailing dimension of size 3.")

    if rgb_input.dtype.kind in {"u", "i"}:
        info = np.iinfo(rgb_input.dtype)
        rgb_float = np.asarray(rgb_input, dtype=float) / max(float(info.max), 1.0)
    else:
        rgb_float = np.asarray(rgb_input, dtype=float)

    img_xyz = np.asarray(srgb_to_xyz(rgb_float), dtype=float)
    img_xyz_xw, _, _, _ = rgb_to_xw_format(img_xyz)
    method_key = param_format(method)

    if method_key in {"bright"}:
        y_channel = img_xyz_xw[:, 1]
        top_y = float(np.percentile(y_channel, 98.0))
        mask = y_channel > top_y
        if not np.any(mask):
            mask = y_channel >= top_y
        top_xy = np.mean(chromaticity_xy(img_xyz_xw[mask, :]), axis=0, dtype=float)
    elif method_key in {"gray"}:
        top_xy = np.mean(chromaticity_xy(img_xyz_xw), axis=0, dtype=float)
    else:
        raise UnsupportedOptionError("srgb2colortemp", method)

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    c_temps = np.arange(2500.0, 10501.0, 500.0, dtype=float)
    xy = np.vstack(
        [
            np.asarray(chromaticity_xy(xyz_from_energy(blackbody(wave, c_temp, kind="energy"), wave, asset_store=asset_store)), dtype=float)
            for c_temp in c_temps
        ]
    )
    c_table = np.column_stack([c_temps, xy])
    index = int(np.argmin(np.linalg.norm(xy - top_xy.reshape(1, 2), axis=1)))
    c_temp = float(c_temps[index])
    if return_table:
        return c_temp, c_table
    return c_temp


def cpiq_csf(frequency_cpd: Any) -> NDArray[np.float64]:
    """Return the normalized CPIQ contrast-sensitivity weighting."""

    frequency = np.asarray(frequency_cpd, dtype=float)
    if np.any(frequency < 0.0):
        raise ValueError("frequency_cpd must be nonnegative.")

    csf = 75.0 * np.power(frequency, 0.8) * np.exp(-0.2 * frequency) / 34.05
    scale = float(np.max(csf)) if csf.size else 0.0
    if scale <= 1.0e-12:
        return np.zeros_like(frequency, dtype=float)
    return np.asarray(csf / scale, dtype=float)


def iso_acutance(cpd: Any, luminance_mtf: Any) -> float:
    """Compute ISO acutance from luminance MTF samples in cycles per degree."""

    frequency = _vector(cpd, name="cpd")
    mtf = _vector(luminance_mtf, name="luminance_mtf")
    if frequency.size != mtf.size:
        raise ValueError("cpd and luminance_mtf must have the same length.")
    if frequency.size < 2:
        raise ValueError("cpd must contain at least two samples.")

    csf = cpiq_csf(frequency)
    delta_v = float(frequency[1] - frequency[0])
    weighted = float(np.sum(mtf * csf, dtype=float) * delta_v)
    reference = float(np.sum(csf, dtype=float) * delta_v)
    return weighted / max(reference, 1.0e-12)


srgb2colortemp = srgb_to_color_temp


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


def _metrics_handle_copy(handles: Any) -> dict[str, Any]:
    if handles is None:
        raise ValueError("metrics handles are required.")
    if isinstance(handles, dict):
        return dict(handles)
    raise TypeError("metrics handles must be a dict in the headless Python API.")


def _metrics_names(handles: dict[str, Any], key: str, default_prefix: str) -> list[str]:
    names = handles.get(key)
    if names is None:
        return [str(handles.get(f"{default_prefix}1_name", "image1")), str(handles.get(f"{default_prefix}2_name", "image2"))]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return [str(names)]


def _metrics_metric_key(value: Any) -> str:
    return str(param_format(value)).replace("(", "").replace(")", "").replace("_", "").replace("-", "")


def _metrics_rect(handles: dict[str, Any], which_image: str, rect: Any | None = None) -> np.ndarray:
    if rect is not None:
        return np.asarray(rect, dtype=float).reshape(-1)

    normalized = _metrics_metric_key(which_image)
    if normalized in {"img1", "image1", "upperleftimage"}:
        keys = (
            "img1_rect",
            "image1_rect",
            "upperleftimage_rect",
            "rect_img1",
            "rect_image1",
        )
    elif normalized in {"img2", "image2", "upperrightimage"}:
        keys = (
            "img2_rect",
            "image2_rect",
            "upperrightimage_rect",
            "rect_img2",
            "rect_image2",
        )
    elif normalized in {"metricimage", "lowerimage", "metricimg"}:
        keys = (
            "metric_rect",
            "metricimage_rect",
            "lowerimage_rect",
            "rect_metric",
            "rect_metricimage",
        )
    else:
        raise UnsupportedOptionError("metricsROI", which_image)

    for key in keys:
        if key in handles and handles[key] is not None:
            return np.asarray(handles[key], dtype=float).reshape(-1)

    roi_rects = handles.get("roi_rects")
    if isinstance(roi_rects, dict):
        for key in keys:
            if key in roi_rects and roi_rects[key] is not None:
                return np.asarray(roi_rects[key], dtype=float).reshape(-1)

    raise ValueError(f"metricsROI requires an explicit stored rect for {which_image!r} in headless mode.")


def metrics_roi(handles: Any, which_image: str, rect: Any | None = None) -> NDArray[np.int_]:
    """Return ROI locations for a metrics image using a stored headless rect."""

    from .roi import ie_rect2_locs

    current = _metrics_handle_copy(handles)
    return np.asarray(ie_rect2_locs(_metrics_rect(current, which_image, rect)), dtype=int)


def metrics_compare_roi(
    handles: Any,
    roi_locs: Any | None = None,
    *,
    rect: Any | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Compare Delta Eab across a metrics ROI using the two selected IP objects."""

    from .ip import ip_get

    current = _metrics_handle_copy(handles)
    vci1, vci2 = metrics_get_vci_pair(current)
    locs = metrics_roi(current, "img1", rect) if roi_locs is None else np.asarray(roi_locs, dtype=int)
    xyz1 = np.asarray(ip_get(vci1, "roixyz", locs), dtype=float)
    xyz2 = np.asarray(ip_get(vci2, "roixyz", locs), dtype=float)
    delta = np.asarray(delta_e_ab(xyz1, xyz2, _metrics_white_point(vci1, vci2)), dtype=float)
    return delta, np.asarray(locs, dtype=int)


def metrics_get_vci_pair(handles: Any) -> tuple[Any, Any]:
    """Return the selected pair of IP objects from a headless metrics handle dict."""

    current = _metrics_handle_copy(handles)
    if "vci1" in current and "vci2" in current:
        return current["vci1"], current["vci2"]
    pair = current.get("vcipair") or current.get("vcimagepair")
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]

    images = current.get("images")
    if isinstance(images, dict):
        image1_name = str(current.get("image1_name", current.get("image1name", "")))
        image2_name = str(current.get("image2_name", current.get("image2name", "")))
        if image1_name in images and image2_name in images:
            return images[image1_name], images[image2_name]

    raise ValueError("metrics handles do not contain a resolvable IP pair.")


def metrics_get(handles: Any, param: str, *args: Any) -> Any:
    """Headless MATLAB-style metricsGet() wrapper."""

    current = _metrics_handle_copy(handles)
    key = param_format(param)
    if key == "image1name":
        return str(current.get("image1_name", current.get("image1name", "image1")))
    if key == "image2name":
        return str(current.get("image2_name", current.get("image2name", "image2")))
    if key in {"metricaxes", "metricsaxes"}:
        return current.get("imgMetric")
    if key in {"metricdata", "metricuserdata"}:
        return current.get("metric_image", current.get("metricImage"))
    if key in {"currentmetric", "curmetric"}:
        return str(current.get("current_metric", current.get("currentMetric", "CIELAB (dE)")))
    if key in {"listofmetricnames", "metricnames"}:
        return list(current.get("metric_names", current.get("metricNames", ["CIELAB (dE)", "CIELUV (dE)", "MSE", "RMSE", "PSNR"])))
    if key in {"metricimagedata", "metricimage"}:
        image = current.get("metric_image", current.get("metricImage"))
        if image is None:
            return None
        metric_name = str(metrics_get(current, "currentmetric"))
        image_array = np.asarray(image, dtype=float)
        if _metrics_metric_key(metric_name) == "cielabde":
            return image_array / 30.0
        image_max = float(np.max(image_array)) if image_array.size else 0.0
        return image_array if image_max <= 0.0 else image_array / image_max
    if key in {"vcipair", "vcimagepair"}:
        vci1, vci2 = metrics_get_vci_pair(current)
        return {"vci1": vci1, "vci2": vci2}
    raise KeyError(f"Unknown metricsGet parameter: {param}")


def metrics_set(handles: Any, param: str, val: Any, *args: Any) -> dict[str, Any]:
    """Headless MATLAB-style metricsSet() wrapper."""

    current = _metrics_handle_copy(handles)
    key = param_format(param)
    if key in {"metricdata", "metricuserdata"}:
        current["metric_image"] = np.asarray(val, dtype=float)
        return current
    raise KeyError(f"Unknown metricsSet parameter: {param}")


def _metrics_white_point(vci1: Any, vci2: Any) -> NDArray[np.float64]:
    from .ip import ip_get

    wp1 = ip_get(vci1, "datawhitepoint")
    wp2 = ip_get(vci2, "datawhitepoint")
    if wp1 is None and wp2 is None:
        raise ValueError("Metrics calculations require at least one IP white point.")
    if wp1 is None:
        return np.asarray(wp2, dtype=float)
    if wp2 is None:
        return np.asarray(wp1, dtype=float)
    wp1_array = np.asarray(wp1, dtype=float)
    wp2_array = np.asarray(wp2, dtype=float)
    return np.asarray((wp1_array + wp2_array) / 2.0, dtype=float)


def metrics_description(handles: Any) -> str:
    """Return the headless text summary that metricsDescription() writes into the GUI."""

    from .ip import ip_get

    current = _metrics_handle_copy(handles)
    vci1, vci2 = metrics_get_vci_pair(current)
    names = [str(metrics_get(current, "image1name")), str(metrics_get(current, "image2name"))]
    chunks: list[str] = []
    for index, (name, vci) in enumerate(((names[0], vci1), (names[1], vci2)), start=1):
        size = np.asarray(ip_get(vci, "size"), dtype=float).reshape(-1)
        wp = ip_get(vci, "datawhitepoint")
        chunks.append(f"Image {index} ({name}):")
        if size.size >= 2:
            chunks.append(f"size: ({size[0]:.0f}, {size[1]:.0f})")
        if wp is None:
            chunks.append("No image white point")
        else:
            wp_array = np.asarray(wp, dtype=float).reshape(-1)
            chunks.append(f"White (X,Y,Z): ({wp_array[0]:.1f}, {wp_array[1]:.1f}, {wp_array[2]:.1f})")
        if index == 1:
            chunks.append("")
            chunks.append("-----------------")
            chunks.append("")
    return "\n".join(chunks)


def metrics_masked_error(img_v: Any, error: Any, method: str = "ls") -> float:
    """Estimate the correlated error term using the MATLAB metricsMaskedError() contract."""

    normalized_method = param_format(method)
    if normalized_method != "ls":
        raise UnsupportedOptionError("metricsMaskedError", method)
    img_array = np.asarray(img_v, dtype=float).reshape(-1)
    error_array = np.asarray(error, dtype=float).reshape(-1)
    if img_array.size != error_array.size:
        raise ValueError("img_v and error must have the same number of elements.")
    return float(np.linalg.lstsq(img_array.reshape(-1, 1), error_array, rcond=None)[0][0])


def metrics_compute(vc1: Any, vc2: Any, metric_name: str = "difference") -> tuple[NDArray[np.float64] | None, float | None]:
    """Compute the headless metrics image/value for a pair of IP objects."""

    from .ip import ip_get

    normalized_metric = _metrics_metric_key(metric_name)
    if normalized_metric in {"cielab", "cielabde"}:
        xyz1 = np.asarray(ip_get(vc1, "dataxyz"), dtype=float)
        xyz2 = np.asarray(ip_get(vc2, "dataxyz"), dtype=float)
        delta = np.asarray(delta_e_ab(xyz1, xyz2, _metrics_white_point(vc1, vc2)), dtype=float)
        return delta, None
    if normalized_metric in {"cieluv", "cieluvde"}:
        xyz1 = np.asarray(ip_get(vc1, "dataxyz"), dtype=float)
        xyz2 = np.asarray(ip_get(vc2, "dataxyz"), dtype=float)
        white = _metrics_white_point(vc1, vc2)
        luv1 = np.asarray(xyz_to_luv(xyz1, white), dtype=float)
        luv2 = np.asarray(xyz_to_luv(xyz2, white), dtype=float)
        delta = np.linalg.norm(luv1 - luv2, axis=-1)
        return np.asarray(delta, dtype=float), None
    result1 = np.asarray(ip_get(vc1, "result"), dtype=float)
    result2 = np.asarray(ip_get(vc2, "result"), dtype=float)
    if normalized_metric in {"mse", "meansquarederror"}:
        diff = result1 - result2
        img = np.sum(np.square(diff), axis=-1) if diff.ndim == 3 else np.square(diff)
        return np.asarray(img, dtype=float), float(np.mean(img))
    if normalized_metric in {"rmse", "rootmeansquarederror"}:
        diff = result1 - result2
        img = np.sqrt(np.sum(np.square(diff), axis=-1)) if diff.ndim == 3 else np.abs(diff)
        return np.asarray(img, dtype=float), float(np.mean(img))
    if normalized_metric in {"psnr", "peaksnr"}:
        return None, peak_signal_to_noise_ratio(result1, result2)
    raise UnsupportedOptionError("metricsCompute", metric_name)


def metrics_show_image(handles: Any) -> dict[str, Any]:
    """Return the rendered-image payload that metricsShowImage() would display."""

    from .ip import ip_get

    current = _metrics_handle_copy(handles)
    vci1, vci2 = metrics_get_vci_pair(current)
    gamma = float(current.get("gamma", 1.0))
    return {
        "image1": None if ip_get(vci1, "result") is None else np.asarray(ip_get(vci1, "result"), dtype=float),
        "image2": None if ip_get(vci2, "result") is None else np.asarray(ip_get(vci2, "result"), dtype=float),
        "gamma": gamma,
    }


def metrics_show_metric(handles: Any) -> dict[str, Any]:
    """Return the metric-image payload that metricsShowMetric() would display."""

    current = _metrics_handle_copy(handles)
    image = metrics_get(current, "metricimagedata")
    return {
        "metric": str(metrics_get(current, "currentmetric")),
        "image": None if image is None else np.asarray(image, dtype=float),
    }


def metrics_save_image(handles: Any, path: str | Path) -> tuple[str, str]:
    """Save the headless metric image to a TIFF file."""

    current = _metrics_handle_copy(handles)
    image = metrics_get(current, "metricimagedata")
    if image is None:
        raise ValueError("No metric image data present.")
    full_path = Path(path).with_suffix(".tiff")
    iio.imwrite(full_path, np.asarray(image, dtype=np.float32))
    return str(full_path), str(metrics_get(current, "currentmetric"))


def metrics_save_data(handles: Any, path: str | Path) -> tuple[str, str]:
    """Save the current headless metrics payload to a MAT file."""

    current = _metrics_handle_copy(handles)
    image = metrics_get(current, "metricdata")
    if image is None:
        raise ValueError("No metric data present.")
    full_path = Path(path).with_suffix(".mat")
    savemat(
        full_path,
        {
            "data": np.asarray(image, dtype=float),
            "metricName": str(metrics_get(current, "currentmetric")),
            "image1": str(metrics_get(current, "image1name")),
            "image2": str(metrics_get(current, "image2name")),
        },
    )
    return str(full_path), str(metrics_get(current, "currentmetric"))


def metrics_close(handles: Any | None = None) -> dict[str, Any]:
    """Close the headless metrics window state and drop GUI-only handles."""

    current: dict[str, Any] = {} if handles is None else _metrics_handle_copy(handles)
    current.pop("metricsWindow", None)
    current.pop("metrics_window", None)
    current.pop("figure1", None)
    current["closed"] = True
    return current


def metrics_refresh(handles: Any) -> dict[str, Any]:
    """Refresh the headless metrics-window payloads."""

    current = _metrics_handle_copy(handles)
    if "images" in current and isinstance(current["images"], dict):
        names = [str(name) for name in current["images"].keys()]
    else:
        names = _metrics_names(current, "image_names", "image")
    current["popImageList1"] = list(names)
    current["popImageList2"] = list(names)
    current["image_names"] = list(names)
    current["shown_images"] = metrics_show_image(current)
    current["shown_metric"] = metrics_show_metric(current)
    current["description_text"] = metrics_description(current)
    return current


def metrics_key_press(handles: Any, key: Any) -> dict[str, Any]:
    """Dispatch the small upstream metrics keyboard-control surface headlessly."""

    current = _metrics_handle_copy(handles)
    if isinstance(key, str):
        if len(key) != 1:
            raise ValueError("metricsKeyPress string keys must be single characters.")
        key_code = ord(key)
    else:
        key_code = int(key)

    if key_code == 8:
        current["last_action"] = "help"
        return current
    if key_code == 16:
        vci1, vci2 = metrics_get_vci_pair(current)
        metric_name = str(metrics_get(current, "currentmetric"))
        metric_image, metric_value = metrics_compute(vci1, vci2, metric_name)
        if metric_image is not None:
            current["metric_image"] = np.asarray(metric_image, dtype=float)
        current["metric_value"] = None if metric_value is None else float(metric_value)
        current["last_action"] = "compute"
        return current
    if key_code == 18:
        refreshed = metrics_refresh(current)
        refreshed["last_action"] = "refresh"
        return refreshed
    return current


def metrics_camera(
    camera: Any,
    metric_name: str,
    *,
    asset_store: AssetStore | None = None,
    session: Any | None = None,
) -> Any:
    """Compute a predefined camera metric using the MATLAB metricsCamera() gateway shape."""

    from .camera import camera_acutance, camera_color_accuracy, camera_mtf, camera_vsnr

    normalized_metric = param_format(metric_name)
    if normalized_metric == "mcccolor":
        metric, _ = camera_color_accuracy(camera, asset_store=asset_store, session=session)
        if isinstance(metric, dict) and "vci" in metric:
            metric = dict(metric)
            metric.pop("vci", None)
        return metric
    if normalized_metric == "slantededge":
        return camera_mtf(camera, asset_store=asset_store, session=session)
    if normalized_metric == "vsnr":
        return camera_vsnr(camera, asset_store=asset_store, session=session)
    if normalized_metric == "acutance":
        return float(camera_acutance(camera, asset_store=asset_store, session=session))
    raise UnsupportedOptionError("metricsCamera", metric_name)


def ie_sqri(sf: Any, d_mtf: Any, luminance: Any, *args: Any) -> tuple[float, NDArray[np.float64]]:
    """Return the Barten SQRI value and corresponding human CSF."""

    width = 40.0
    if len(args) % 2 != 0:
        raise ValueError("ieSQRI optional arguments must be key/value pairs.")
    for index in range(0, len(args), 2):
        key = param_format(args[index])
        value = args[index + 1]
        if key == "width":
            width = float(np.asarray(value, dtype=float).reshape(-1)[0])
        else:
            raise UnsupportedOptionError("ieSQRI", str(args[index]))

    sf_array = np.asarray(sf, dtype=float).reshape(-1)
    d_mtf_array = np.asarray(d_mtf, dtype=float).reshape(-1)
    luminance_value = float(np.asarray(luminance, dtype=float).reshape(-1)[0])
    if sf_array.size < 2:
        raise ValueError("ieSQRI requires at least two spatial-frequency samples.")
    if sf_array.size != d_mtf_array.size:
        raise ValueError("sf and dMTF must have the same length.")
    if np.any(sf_array < 0.0):
        raise ValueError("sf must be nonnegative.")
    if np.any(d_mtf_array < 0.0) or np.any(d_mtf_array > 1.0 + (100.0 * np.finfo(float).eps)):
        raise ValueError("dMTF must lie within [0, 1].")
    if luminance_value < 0.0:
        raise ValueError("L must be nonnegative.")

    a = 540.0 * (1.0 + (0.7 / max(luminance_value, 1e-12))) ** (-0.2)
    a /= 1.0 + (12.0 / (float(width) * (1.0 + (sf_array / 3.0)) ** 2))
    b = 0.3 * (1.0 + (100.0 / max(luminance_value, 1e-12))) ** 0.15
    c = 0.06
    h_csf = (a * sf_array) * np.exp(-b * sf_array) * np.sqrt(1.0 + c * np.exp(b * sf_array))
    h_csf = np.asarray(h_csf, dtype=float).reshape(-1)

    du = np.diff(sf_array)
    u = sf_array[1:]
    dm = 0.5 * (d_mtf_array[:-1] + d_mtf_array[1:])
    dh = 0.5 * (h_csf[:-1] + h_csf[1:])
    sqri = (1.0 / np.log(2.0)) * np.sum(np.sqrt(dm * dh) * (du / np.maximum(u, 1e-12)))
    return float(sqri), h_csf


def exposure_value(oi: Any, sensor: Any) -> float:
    """Compute MATLAB-style exposure value from OI f-number and sensor exposure time."""

    from .optics import oi_get
    from .sensor import sensor_get

    f_number = float(oi_get(oi, "fnumber"))
    exposure_time = float(sensor_get(sensor, "exposuretime"))
    return float(np.log2((f_number**2) / max(exposure_time, 1e-30)))


def photometric_exposure(oi: Any, sensor: Any) -> float:
    """Compute MATLAB-style photometric exposure in lux-seconds."""

    from .optics import oi_get
    from .sensor import sensor_get

    return float(oi_get(oi, "meanilluminance")) * float(sensor_get(sensor, "exposuretime"))


def _chart_crop(image: NDArray[np.float64], rect: Any) -> NDArray[np.float64]:
    rectangle = np.asarray(rect, dtype=float).reshape(-1)
    if rectangle.size < 4:
        raise ValueError("rectangles must contain [x, y, width, height].")
    x, y, width, height = rectangle[:4]
    left = int(np.floor(x))
    top = int(np.floor(y))
    right = left + max(int(np.round(width)), 0) + 1
    bottom = top + max(int(np.round(height)), 0) + 1
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image.shape[1])
    bottom = min(bottom, image.shape[0])
    if right <= left or bottom <= top:
        raise ValueError("chart patch rectangle must intersect the image bounds.")
    return np.asarray(image[top:bottom, left:right, :], dtype=float)


def chart_patch_compare(
    img_l: Any,
    img_s: Any,
    rect_l: Any,
    rect_s: Any,
    *,
    patch_size: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compare two 24-patch sRGB charts using the legacy MATLAB template contract."""

    image_l = np.asarray(img_l, dtype=float)
    image_s = np.asarray(img_s, dtype=float)
    if image_l.ndim != 3 or image_l.shape[-1] != 3:
        raise ValueError("img_l must be an RGB image.")
    if image_s.ndim != 3 or image_s.shape[-1] != 3:
        raise ValueError("img_s must be an RGB image.")

    rects_l = np.asarray(rect_l, dtype=float)
    rects_s = np.asarray(rect_s, dtype=float)
    if rects_l.shape != rects_s.shape or rects_l.ndim != 2 or rects_l.shape[1] != 4:
        raise ValueError("rect_l and rect_s must be matching Nx4 rectangle arrays.")
    if rects_l.shape[0] != 24:
        raise ValueError("chartPatchCompare expects 24 chart rectangles in MATLAB patch order.")
    if int(patch_size) <= 0 or int(patch_size) % 2 != 0:
        raise ValueError("patch_size must be a positive even integer.")

    patch_size = int(patch_size)
    chart_template = np.zeros((patch_size * 4, patch_size * 6, 3), dtype=float)
    delta_e_map = np.zeros((patch_size * 4, patch_size * 6), dtype=float)
    white_point = np.asarray(srgb_to_xyz(np.ones((1, 1, 3), dtype=float)), dtype=float)[0, 0, :]

    count = 0
    for column in range(6):
        c_start = column * patch_size
        for row in range(4):
            r_start = row * patch_size
            mean_patch_l = np.mean(_chart_crop(image_l, rects_l[count]), axis=(0, 1), dtype=float)
            mean_patch_s = np.mean(_chart_crop(image_s, rects_s[count]), axis=(0, 1), dtype=float)

            chart_template[r_start : r_start + patch_size, c_start : c_start + patch_size, :] = mean_patch_l.reshape(1, 1, 3)
            half = patch_size // 2
            chart_template[r_start + half : r_start + patch_size, c_start + half : c_start + patch_size, :] = mean_patch_s.reshape(1, 1, 3)

            xyz_l = np.asarray(srgb_to_xyz(mean_patch_l.reshape(1, 1, 3)), dtype=float)[0, 0, :]
            xyz_s = np.asarray(srgb_to_xyz(mean_patch_s.reshape(1, 1, 3)), dtype=float)[0, 0, :]
            lab_l = xyz_to_lab(xyz_l.reshape(1, 1, 3), white_point.reshape(1, 1, 3))
            lab_s = xyz_to_lab(xyz_s.reshape(1, 1, 3), white_point.reshape(1, 1, 3))
            delta_e = float(deltaE_ciede2000(lab_l, lab_s)[0, 0])
            delta_e_map[r_start : r_start + patch_size, c_start : c_start + patch_size] = delta_e
            count += 1

    return chart_template, delta_e_map


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
deltaE2000 = delta_e_2000
deltaE94 = delta_e_94
deltaEuv = delta_e_uv
iePSNR = peak_signal_to_noise_ratio
ieSQRI = ie_sqri
metricsCamera = metrics_camera
metricsCompareROI = metrics_compare_roi
metricsCompute = metrics_compute
metricsDescription = metrics_description
metricsClose = metrics_close
metricsGet = metrics_get
metricsGetVciPair = metrics_get_vci_pair
metricsKeyPress = metrics_key_press
metricsMaskedError = metrics_masked_error
metricsROI = metrics_roi
metricsRefresh = metrics_refresh
metricsSaveData = metrics_save_data
metricsSaveImage = metrics_save_image
metricsSet = metrics_set
metricsShowImage = metrics_show_image
metricsShowMetric = metrics_show_metric
exposureValue = exposure_value
photometricExposure = photometric_exposure
chartPatchCompare = chart_patch_compare
metricsSPD = metrics_spd
spd2cct = spd_to_cct
cpiqCSF = cpiq_csf
ISOAcutance = iso_acutance
