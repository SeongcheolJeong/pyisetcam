"""ISO 12233-oriented helpers used by metrics tutorials."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

from .ip import ip_get
from .roi import vc_get_roi_data
from .types import ImageProcessor


def ie_cxcorr(a: Any, b: Any) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Circular cross correlation using MATLAB ieCXcorr() semantics."""

    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    if first.size != second.size:
        raise ValueError("Vector length mismatch.")

    norm_first = max(float(np.linalg.norm(first)), 1.0e-12)
    norm_second = max(float(np.linalg.norm(second)), 1.0e-12)
    first = first / norm_first
    second = second / norm_second

    xc = np.empty(first.size, dtype=float)
    shifted = second.copy()
    for index in range(first.size):
        xc[index] = float(np.dot(first, shifted))
        shifted = np.roll(shifted, 1)

    return xc, np.arange(first.size, dtype=int)


def _matlab_round_scalar(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    radius = (int(size) - 1) / 2.0
    sample = np.arange(-radius, radius + 1.0, dtype=float)
    xx, yy = np.meshgrid(sample, sample, indexing="xy")
    kernel = np.exp(-((xx**2 + yy**2) / max(2.0 * float(sigma) ** 2, 1.0e-12)))
    return kernel / max(float(np.sum(kernel, dtype=float)), 1.0e-12)


def iso_find_slanted_bar(ip: ImageProcessor, blur_flag: bool = False) -> NDArray[np.int_]:
    """Return a good slanted-bar ROI rectangle following MATLAB ISOFindSlantedBar()."""

    image = np.asarray(ip_get(ip, "data display"), dtype=float)
    if image.ndim != 3:
        raise ValueError("ISOFindSlantedBar expects an RGB-format image-processor result.")

    monochrome = np.sum(image, axis=2, dtype=float)
    monochrome = monochrome / max(float(np.max(monochrome)), 1.0e-12)

    black_border = True
    if blur_flag:
        size = monochrome.shape
        kernel_size = max(_matlab_round_scalar(min(size) / 10.0), 1)
        kernel = _gaussian_kernel(kernel_size, max(float(np.rint(kernel_size * 0.7)), 1.0))
        monochrome = convolve2d(monochrome, kernel, mode="same")
        black_border = False

    y_sums = np.sum(monochrome, axis=1, dtype=float)
    dy_sums = np.diff(y_sums)
    below_average = y_sums < (float(np.mean(y_sums)) / 5.0)
    remove = below_average[:-1] & below_average[1:]
    dy_sums = dy_sums.copy()
    dy_sums[remove] = 0.0

    if float(np.max(np.abs(dy_sums))) > 5.0 and black_border:
        row_max = int(np.argmin(dy_sums)) + 1
        row_min = int(np.argmax(dy_sums[: max(row_max - 4, 1)])) + 1
    else:
        rows = int(ip_get(ip, "row"))
        skip = _matlab_round_scalar(rows * 0.05)
        row_min = max(skip, 1)
        row_max = max(rows - skip, row_min + 1)

    dx_row_min = np.diff(np.sum(monochrome[row_min + 1 : row_min + 5, :], axis=0, dtype=float))
    dx_row_max = np.diff(np.sum(monochrome[row_max - 5 : row_max - 1, :], axis=0, dtype=float))

    upper_left_x = int(np.argmin(dx_row_min)) + 1
    lower_right_x = int(np.argmin(dx_row_max)) + 1
    upper_left = np.array([upper_left_x, row_min], dtype=float)
    lower_right = np.array([lower_right_x, row_max], dtype=float)
    midpoint = (lower_right + upper_left) / 2.0

    width = int(np.rint(abs(lower_right[0] - upper_left[0])))
    height = _matlab_round_scalar(1.5 * float(width))
    col_min = _matlab_round_scalar(midpoint[0] - (float(width) / 2.0))
    row_min_rect = _matlab_round_scalar(midpoint[1] - (float(height) / 2.0))

    if width < 5 or height < 5:
        return np.array([], dtype=int)
    return np.array([col_min, row_min_rect, width, height], dtype=int)


def edge_to_mtf(
    bar_image: Any,
    *,
    channel: int = 2,
    fixed_row: int = 20,
) -> dict[str, NDArray[np.float64] | NDArray[np.int_]]:
    """Estimate an edge-derived MTF using the workflow from s_metricsEdge2MTF.m."""

    image = np.asarray(bar_image, dtype=float)
    if image.ndim != 3:
        raise ValueError("edge_to_mtf expects an RGB-format bar image.")
    channel_index = int(channel) - 1
    if channel_index < 0 or channel_index >= image.shape[2]:
        raise ValueError("Requested channel is out of range.")

    mono = image[:, :, channel_index]
    dimg = np.abs(np.diff(mono, axis=1))
    row_count, col_count = dimg.shape
    aligned = np.zeros_like(dimg, dtype=float)
    lags = np.zeros(row_count, dtype=int)
    fixed = dimg[min(max(int(fixed_row) - 1, 0), row_count - 1), :]

    for row_index in range(row_count):
        corr, row_lags = ie_cxcorr(fixed, dimg[row_index, :])
        best_index = int(np.argmax(corr))
        shift = int(row_lags[best_index])
        aligned[row_index, :] = np.roll(dimg[row_index, :], shift)
        lags[row_index] = shift

    lsf = np.mean(aligned, axis=0, dtype=float)
    lsf = lsf / max(float(np.sum(lsf, dtype=float)), 1.0e-12)
    mtf = np.abs(np.fft.fft(lsf))
    freq = np.arange(_matlab_round_scalar(col_count / 2.0), dtype=float)

    return {
        "dimg": np.asarray(dimg, dtype=float),
        "aligned": np.asarray(aligned, dtype=float),
        "lsf": np.asarray(lsf, dtype=float),
        "mtf": np.asarray(mtf[: freq.size], dtype=float),
        "freq": np.asarray(freq, dtype=float),
        "lags": np.asarray(lags, dtype=int),
    }


ISOFindSlantedBar = iso_find_slanted_bar
ieCXcorr = ie_cxcorr
