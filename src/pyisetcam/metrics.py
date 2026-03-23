"""Metric and validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
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
iePSNR = peak_signal_to_noise_ratio
exposureValue = exposure_value
photometricExposure = photometric_exposure
chartPatchCompare = chart_patch_compare
metricsSPD = metrics_spd
spd2cct = spd_to_cct
cpiqCSF = cpiq_csf
ISOAcutance = iso_acutance
