"""Headless plot-data wrappers for selected MATLAB plotting APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .color import xyz_color_matching
from .exceptions import UnsupportedOptionError
from .ip import ip_get
from .metrics import xyz_to_lab, xyz_to_luv
from .optics import oi_get
from .scene import scene_get
from .sensor import pixel_snr, sensor_get, sensor_snr
from .types import ImageProcessor, OpticalImage, Scene, Sensor
from .utils import linear_to_srgb, param_format, tile_pattern, xyz_to_linear_srgb


def _roi_required(function_name: str, plot_type: str, roi_locs: Any | None) -> Any:
    if roi_locs is None:
        raise ValueError(f"ROI required for {function_name}(..., '{plot_type}').")
    return roi_locs


def _roi_payload(roi_locs: Any) -> dict[str, Any]:
    from .roi import ie_locs2_rect

    roi = np.asarray(roi_locs, dtype=int)
    payload: dict[str, Any] = {"roiLocs": roi.copy()}
    if roi.ndim == 1 and roi.size == 4:
        payload["rect"] = roi.copy()
    elif roi.ndim == 2 and roi.shape[1] == 2:
        payload["rect"] = ie_locs2_rect(roi)
    return payload


def _line_index(function_name: str, plot_type: str, xy: Any | None, orientation: str) -> tuple[int, np.ndarray]:
    if xy is None:
        raise ValueError(f"Line selector required for {function_name}(..., '{plot_type}').")
    xy_array = np.rint(np.asarray(xy, dtype=float)).astype(int).reshape(-1)
    if xy_array.size == 1:
        return int(xy_array[0]), xy_array.copy()
    if xy_array.size != 2:
        raise ValueError("Line selector must be a scalar index or [col, row].")
    index = int(xy_array[1] if orientation == "h" else xy_array[0])
    return index, xy_array.copy()


def _plot_option(args: tuple[Any, ...], key: str, default: Any = None) -> Any:
    if len(args) % 2 != 0:
        raise ValueError("Optional plotting arguments must be key/value pairs.")
    normalized_key = param_format(key)
    for index in range(0, len(args), 2):
        if param_format(args[index]) == normalized_key:
            return args[index + 1]
    return default


def _sensor_plot_line_data(sensor: Sensor, line_key: str, xy: Any) -> dict[str, Any]:
    key = param_format(line_key)
    orientation = "h" if "hline" in key else "v"
    data_type = "electrons" if "electrons" in key else "dv" if "dv" in key else "volts"
    line_index, xy_array = _line_index("plotSensor", line_key, xy, orientation)
    profile = sensor_get(sensor, f"{orientation}line {data_type}", line_index)
    if profile is None:
        raise ValueError(f"Sensor has no {data_type} data for {line_key}.")
    pix_color = np.array(
        [
            color_index
            for color_index, values in enumerate(profile["data"], start=1)
            if np.asarray(values, dtype=float).size > 0
        ],
        dtype=int,
    )
    return {
        "xy": xy_array,
        "ori": orientation,
        "dataType": data_type,
        "data": [np.asarray(values, dtype=float).copy() for values in profile["data"]],
        "pos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pos"]],
        "pixPos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pixPos"]],
        "pixColor": pix_color,
        "filterPlotColors": sensor_get(sensor, "filter plot colors"),
        "xLabel": "Position (um)",
        "yLabel": "digital value" if data_type == "dv" else data_type,
        "titleString": f"{'Horizontal' if orientation == 'h' else 'Vertical'} line {line_index}",
    }


def _sensor_plot_two_lines(sensor: Sensor, line_key: str, xy: Any) -> dict[str, Any]:
    key = param_format(line_key)
    orientation = "h" if "hline" in key else "v"
    data_type = "electrons" if "electrons" in key else "dv" if "dv" in key else "volts"
    line_index, xy_array = _line_index("plotSensor", line_key, xy, orientation)
    second_xy = xy_array.copy()
    if second_xy.size == 1:
        second_xy[0] = line_index + 1
    elif orientation == "h":
        second_xy[1] = line_index + 1
    else:
        second_xy[0] = line_index + 1

    max_index = int(sensor_get(sensor, "rows" if orientation == "h" else "cols"))
    if line_index >= max_index:
        raise IndexError("Two-line sensor plot requires an adjacent line within the sensor bounds.")

    first_line = _sensor_plot_line_data(sensor, line_key, xy_array)
    second_line = _sensor_plot_line_data(sensor, line_key, second_xy)

    pix_pos: list[np.ndarray] = []
    pix_data: list[np.ndarray] = []
    pix_color: list[int] = []
    for line in (first_line, second_line):
        for color_index, (positions, values) in enumerate(zip(line["pos"], line["data"]), start=1):
            if np.asarray(values).size == 0:
                continue
            pix_pos.append(np.asarray(positions, dtype=float).copy())
            pix_data.append(np.asarray(values, dtype=float).copy())
            pix_color.append(color_index)

    return {
        "xy": xy_array.copy(),
        "xy2": second_xy.copy(),
        "ori": orientation,
        "dataType": data_type,
        "pixPos": pix_pos,
        "pixData": pix_data,
        "pixColor": np.asarray(pix_color, dtype=int),
        "filterPlotColors": sensor_get(sensor, "filter plot colors"),
        "xLabel": "Position (um)",
        "yLabel": "digital value" if data_type == "dv" else data_type,
        "titleString": f"{'Horizontal' if orientation == 'h' else 'Vertical'} line {line_index}",
    }


def _sensor_plot_histogram(sensor: Sensor, data_type: str, roi_locs: Any) -> dict[str, Any]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    data = np.asarray(vc_get_roi_data(sensor, roi, data_type), dtype=float)
    payload = _roi_payload(roi)
    payload["data"] = data
    payload["unitType"] = data_type
    payload["filterPlotColors"] = sensor_get(sensor, "filter plot colors")
    payload["xLabel"] = {
        "volts": "Volts",
        "electrons": "Electrons",
        "dv": "Digital value",
    }.get(param_format(data_type), str(data_type))
    payload["yLabel"] = "Count"
    return payload


def _sensor_plot_chromaticity(sensor: Sensor, roi_locs: Any | None) -> dict[str, Any]:
    roi = roi_locs if roi_locs is not None else sensor_get(sensor, "roi")
    roi = _roi_required("plotSensor", "chromaticity", roi)
    if int(sensor_get(sensor, "nfilters")) < 2:
        raise UnsupportedOptionError("plotSensor", "chromaticity")
    rg = np.asarray(sensor_get(sensor, "chromaticity", roi), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    sums = np.sum(spectral_qe, axis=1, keepdims=True)
    spectrum_locus = np.divide(
        spectral_qe[:, :2],
        sums,
        out=np.full((spectral_qe.shape[0], 2), np.nan, dtype=float),
        where=sums > 0.0,
    )
    payload = _roi_payload(roi)
    payload["rg"] = rg.copy()
    payload["spectrumlocus"] = spectrum_locus.copy()
    payload["xLabel"] = "r-chromaticity"
    payload["yLabel"] = "g-chromaticity"
    payload["titleString"] = "rg sensor chromaticity"
    return payload


def _sensor_plot_spectra(sensor: Sensor, data_type: str) -> dict[str, Any]:
    key = param_format(data_type)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    if key == "colorfilters":
        data = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Transmittance"
    elif key == "irfilter":
        data = np.asarray(sensor_get(sensor, "ir filter"), dtype=float).reshape(-1, 1)
        names = ["o"]
        y_label = "Transmittance"
    elif key in {"pdspectralqe", "pixelspectralqe"}:
        data = np.asarray(sensor_get(sensor, "pixel spectral qe"), dtype=float).reshape(-1, 1)
        names = ["k"]
        y_label = "QE"
    elif key in {"spectralsr", "sr", "pdspectralsr", "pixelspectralsr"}:
        data = np.asarray(sensor_get(sensor, "pixel spectral sr"), dtype=float).reshape(-1, 1)
        names = ["k"]
        y_label = "Responsivity:  Volts/Watt"
    elif key in {"spectralqe", "sensorspectralqe"}:
        data = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Quantum efficiency"
    elif key in {"sensorspectralsr"}:
        data = np.asarray(sensor_get(sensor, "sensor spectral sr"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Responsivity:  Volts/Watt"
    else:
        raise UnsupportedOptionError("plotSensor", data_type)
    return {
        "x": wave.copy(),
        "y": data.copy(),
        "filterNames": names,
        "dataType": key,
        "xLabel": "Wavelength (nm)",
        "yLabel": y_label,
        "nameString": f"ISET: {key}",
    }


def _cfa_scale_factor(rows: int) -> int:
    if rows < 2:
        return 32
    if rows < 8:
        return 8
    return 1


def _sensor_pattern_letters(sensor: Sensor, pattern: np.ndarray) -> np.ndarray:
    letters = np.array(list(sensor_get(sensor, "filter color letters")), dtype="<U16")
    if letters.size == 0:
        return np.empty(np.asarray(pattern, dtype=int).shape, dtype="<U16")
    pattern_array = np.asarray(pattern, dtype=int)
    return letters[np.clip(pattern_array - 1, 0, letters.size - 1)]


def _sensor_filter_display_colors(sensor: Sensor) -> np.ndarray:
    letters = list(sensor_get(sensor, "filter color letters"))
    filter_spectra = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    colors = np.zeros((filter_spectra.shape[1], 3), dtype=float)
    standard = {
        "r": np.array([1.0, 0.0, 0.0], dtype=float),
        "g": np.array([0.0, 1.0, 0.0], dtype=float),
        "b": np.array([0.0, 0.0, 1.0], dtype=float),
        "c": np.array([0.0, 1.0, 1.0], dtype=float),
        "m": np.array([1.0, 0.0, 1.0], dtype=float),
        "y": np.array([1.0, 1.0, 0.0], dtype=float),
        "w": np.array([1.0, 1.0, 1.0], dtype=float),
        "k": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    fallback_indices: list[int] = []
    for index in range(colors.shape[0]):
        letter = letters[index].lower() if index < len(letters) and letters[index] else ""
        mapped = standard.get(letter)
        if mapped is not None:
            colors[index] = mapped
        else:
            fallback_indices.append(index)

    if fallback_indices:
        wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
        xyz = np.asarray(
            filter_spectra[:, fallback_indices].T @ xyz_color_matching(wave, quanta=True),
            dtype=float,
        )
        linear_rgb = np.clip(xyz_to_linear_srgb(xyz), 0.0, None)
        row_max = np.max(linear_rgb, axis=1, keepdims=True)
        normalized = np.divide(
            linear_rgb,
            np.maximum(row_max, 1e-12),
            out=np.zeros_like(linear_rgb),
            where=row_max > 1e-12,
        )
        low_energy = np.ravel(row_max <= 1e-12)
        if np.any(low_energy):
            normalized[low_energy] = 1.0
        colors[fallback_indices] = linear_to_srgb(normalized)

    return np.clip(colors, 0.0, 1.0)


def _sensor_plot_cfa(sensor: Sensor, *, full_array: bool) -> dict[str, Any]:
    unit_pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    if full_array:
        rows, cols = sensor_get(sensor, "size")
        pattern = tile_pattern(unit_pattern, int(rows), int(cols))
        mode = "full"
    else:
        pattern = unit_pattern.copy()
        mode = "block"
    filter_colors = _sensor_filter_display_colors(sensor)
    small_img = filter_colors[np.clip(pattern - 1, 0, filter_colors.shape[0] - 1)]
    scale = _cfa_scale_factor(int(small_img.shape[0]))
    if scale > 1:
        img = np.repeat(np.repeat(small_img, scale, axis=0), scale, axis=1)
    else:
        img = small_img.copy()
    return {
        "img": img.copy(),
        "imgSmall": small_img.copy(),
        "pattern": pattern.copy(),
        "unitPattern": unit_pattern.copy(),
        "patternColors": _sensor_pattern_letters(sensor, pattern),
        "unitPatternColors": _sensor_pattern_letters(sensor, unit_pattern),
        "filterNames": list(sensor_get(sensor, "filter color letters cell")),
        "filterColors": filter_colors.copy(),
        "scale": scale,
        "mode": mode,
    }


def _sensor_plot_data(sensor: Sensor, *, cfa_constant: bool = False) -> tuple[np.ndarray, str]:
    data_type = "dv" if sensor_get(sensor, "dv") is not None else "volts"
    data = sensor_get(sensor, data_type)
    if data is None:
        rows, cols = sensor_get(sensor, "size")
        fill = 1.0 if cfa_constant else 0.0
        return np.full((int(rows), int(cols)), fill, dtype=float), data_type
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    if cfa_constant:
        array = np.ones(array.shape[:2], dtype=float)
    return np.asarray(array, dtype=float), data_type


def _sensor_plot_scale(sensor: Sensor, data_type: str) -> float:
    if param_format(data_type) == "dv":
        return float(max((2 ** int(sensor_get(sensor, "nbits"))) - 1, 1))
    return float(max(sensor.fields["pixel"]["voltage_swing"], 1e-12))


def _sensor_render_image(sensor: Sensor, data: np.ndarray, data_type: str) -> np.ndarray:
    normalized = np.clip(np.asarray(data, dtype=float) / _sensor_plot_scale(sensor, data_type), 0.0, 1.0)
    if int(sensor_get(sensor, "nfilters")) <= 1:
        return normalized.copy()
    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), normalized.shape[0], normalized.shape[1])
    filter_colors = _sensor_filter_display_colors(sensor)
    linear_rgb = np.zeros(normalized.shape + (3,), dtype=float)
    for index, color in enumerate(filter_colors, start=1):
        mask = pattern == index
        if np.any(mask):
            linear_rgb[mask] = normalized[mask, None] * color.reshape(1, 3)
    return linear_to_srgb(linear_rgb)


def _sensor_plot_true_size(sensor: Sensor) -> dict[str, Any]:
    data, data_type = _sensor_plot_data(sensor)
    return {
        "img": _sensor_render_image(sensor, data, data_type),
        "dataType": data_type,
    }


def _sensor_plot_cfa_image(sensor: Sensor) -> dict[str, Any]:
    data, data_type = _sensor_plot_data(sensor, cfa_constant=True)
    return {
        "img": _sensor_render_image(sensor, data, data_type),
        "dataType": data_type,
    }


def _sensor_plot_channels(sensor: Sensor) -> dict[str, Any]:
    data, data_type = _sensor_plot_data(sensor)
    rows, cols = data.shape[:2]
    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), rows, cols)
    filter_colors = _sensor_filter_display_colors(sensor)
    normalized = np.clip(np.asarray(data, dtype=float) / _sensor_plot_scale(sensor, data_type), 0.0, 1.0)
    channel_data: list[np.ndarray] = []
    channel_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, color in enumerate(filter_colors, start=1):
        mask = pattern == index
        plane = np.full((rows, cols), np.nan, dtype=float)
        plane[mask] = data[mask]
        channel_data.append(plane)
        masks.append(mask.copy())
        tinted = np.zeros((rows, cols, 3), dtype=float)
        if np.any(mask):
            tinted[mask] = normalized[mask, None] * color.reshape(1, 3)
        channel_images.append(linear_to_srgb(tinted))
    return {
        "channelData": channel_data,
        "channelImages": channel_images,
        "masks": masks,
        "pattern": pattern.copy(),
        "filterNames": list(sensor_get(sensor, "filter color letters cell")),
        "dataType": data_type,
    }


def _sensor_plot_etendue(sensor: Sensor) -> dict[str, Any]:
    return {
        "support": sensor_get(sensor, "spatial support", "um"),
        "sensorEtendue": np.asarray(sensor_get(sensor, "etendue"), dtype=float),
        "xLabel": "Position (um)",
        "yLabel": "Position (um)",
        "zLabel": "Relative illumination",
        "nameString": f"ISET: Etendue ({sensor_get(sensor, 'vignetting name')})",
    }


def _sensor_fft_payload(sensor: Sensor, orientation: str, data_type: str, xy: Any) -> dict[str, Any]:
    if int(sensor_get(sensor, "nfilters")) > 1:
        raise UnsupportedOptionError("plotSensorFFT", "color sensors")
    line_index, xy_array = _line_index("plotSensorFFT", orientation, xy, orientation)
    data = sensor_get(sensor, data_type)
    if data is None:
        raise ValueError(f"Sensor has no {data_type} data for plotSensorFFT.")
    array = np.asarray(data, dtype=float)
    if array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal sensor FFT line index is out of range.")
        line = np.asarray(array[line_index - 1, :], dtype=float)
        title_string = f"ISET:  Horizontal fft {line_index}"
        x_label = "Cycles/deg (col)"
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical sensor FFT line index is out of range.")
        line = np.asarray(array[:, line_index - 1], dtype=float)
        title_string = f"ISET:  Vertical fft {line_index}"
        x_label = "Cycles/deg (row)"
    fov = float(sensor_get(sensor, "fov"))
    cpd = np.arange(0, round((line.size - 1) / 2) + 1, dtype=float) / max(fov, 1e-12)
    n_freq = int(cpd.size)
    mean_value = float(np.mean(line))
    amp = np.abs(np.fft.fft(line - mean_value)) / max(float(n_freq), 1.0)
    peak_contrast = float(np.max(amp) / mean_value) if not np.isclose(mean_value, 0.0) else float(np.inf)
    return {
        "xy": xy_array.copy(),
        "ori": orientation,
        "dataType": data_type,
        "cpd": cpd.copy(),
        "amp": np.asarray(amp, dtype=float).copy(),
        "ampPlot": np.asarray(amp[:n_freq], dtype=float).copy(),
        "mean": mean_value,
        "peakContrast": peak_contrast,
        "titleString": title_string,
        "xLabel": x_label,
        "yLabel": "Abs(fft(data))",
    }


def _sensor_noise_title(the_noise: np.ndarray, voltage_swing: float) -> str:
    noise = np.asarray(the_noise, dtype=float)
    return f"Max/min: [{float(np.max(noise)):.2E},{float(np.min(noise)):.2E}] on voltage swing {float(voltage_swing):.2f}"


def _sensor_plot_shot_noise(sensor: Sensor) -> dict[str, Any]:
    electrons = sensor_get(sensor, "electrons")
    if electrons is None:
        electrons = np.zeros(sensor_get(sensor, "size"), dtype=float)
    electron_image = np.clip(np.asarray(electrons, dtype=float), 0.0, None)
    rng = np.random.default_rng(0)
    electron_noise = np.sqrt(electron_image) * rng.standard_normal(electron_image.shape)
    low_count = electron_image < 25.0
    if np.any(low_count):
        poisson_counts = rng.poisson(electron_image[low_count])
        electron_noise[low_count] = poisson_counts - electron_image[low_count]
    conversion_gain = float(sensor.fields["pixel"]["conversion_gain_v_per_electron"])
    the_noise = conversion_gain * electron_noise
    noisy_image = conversion_gain * np.rint(electron_image + electron_noise)
    return {
        "noiseType": "shotnoise",
        "nameString": "ISET:  Shot noise",
        "titleString": _sensor_noise_title(the_noise, float(sensor_get(sensor, "voltage swing"))),
        "signal": electron_image.copy(),
        "noisyImage": np.asarray(noisy_image, dtype=float).copy(),
        "theNoise": np.asarray(the_noise, dtype=float).copy(),
    }


def _sensor_plot_fixed_pattern_noise(sensor: Sensor, noise_type: str) -> dict[str, Any]:
    rows, cols = sensor_get(sensor, "size")
    rng = np.random.default_rng(0)
    normalized = param_format(noise_type)
    voltage_swing = float(sensor_get(sensor, "voltage swing"))
    if normalized == "dsnu":
        the_noise = sensor_get(sensor, "dsnu image")
        if the_noise is None:
            sigma = float(sensor_get(sensor, "dsnu sigma"))
            the_noise = rng.normal(0.0, sigma, size=(int(rows), int(cols)))
        the_noise = np.asarray(the_noise, dtype=float)
        noisy_image = the_noise.copy()
        name_string = "ISET:  DSNU"
        title_string = _sensor_noise_title(the_noise, voltage_swing)
    elif normalized == "prnu":
        the_noise = sensor_get(sensor, "prnu image")
        if the_noise is None:
            sigma = float(sensor.fields["pixel"]["prnu_sigma"])
            the_noise = 1.0 + rng.normal(0.0, sigma, size=(int(rows), int(cols)))
        the_noise = np.asarray(the_noise, dtype=float)
        noisy_image = the_noise.copy()
        name_string = "ISET:  PRNU"
        title_string = f"Max/min: [{float(np.max(the_noise)):.2E},{float(np.min(the_noise)):.2E}] slope"
    else:
        raise UnsupportedOptionError("plotSensor", noise_type)
    return {
        "noiseType": normalized,
        "nameString": name_string,
        "titleString": title_string,
        "noisyImage": noisy_image.copy(),
        "theNoise": the_noise.copy(),
    }


def _ip_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line", xy, orientation)
    data = ip_get(ip, "result")
    if data is None:
        raise ValueError("IP has no result data for line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError("IP result data must be an RGB image for line plotting.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        values = np.asarray(array[line_index - 1, :, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        values = np.asarray(array[:, line_index - 1, :], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, values.shape[0] + 1, dtype=float),
        "values": values.copy(),
    }


def _ip_luminance_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line luminance", xy, orientation)
    data = ip_get(ip, "data luminance")
    if data is None:
        raise ValueError("IP has no XYZ data for luminance line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("IP luminance data must be a 2D image.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        line = np.asarray(array[line_index - 1, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        line = np.asarray(array[:, line_index - 1], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, line.shape[0] + 1, dtype=float),
        "data": line.copy(),
    }


def _ip_plot_color_data(ip: ImageProcessor, roi_locs: Any) -> tuple[np.ndarray, np.ndarray]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    rgb = np.asarray(vc_get_roi_data(ip, roi, "result"), dtype=float)
    xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    return rgb, xyz


def _ip_plot_white_point(ip: ImageProcessor, roi_xyz: np.ndarray) -> np.ndarray:
    white_point = ip_get(ip, "data white point")
    if white_point is not None:
        return np.asarray(white_point, dtype=float).reshape(3)
    return np.mean(np.asarray(roi_xyz, dtype=float), axis=0).reshape(3)


def scene_plot(
    scene: Scene,
    p_type: str = "hlineluminance",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotScene` user-data without opening a figure."""

    key = param_format(p_type)
    if key == "radianceenergyroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        energy = np.mean(np.asarray(scene_get(scene, "roi energy", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "energy": energy}, None
    if key == "radiancephotonsroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        photons = np.mean(np.asarray(scene_get(scene, "roi photons", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "photons": photons}, None
    if key in {"radiancehline", "hlineradiance", "radiancevline", "vlineradiance", "luminancehline", "hlineluminance", "luminancevline", "vlineluminance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        udata = dict(scene_get(scene, p_type, roi, *(args[:1] if args else ()), asset_store=asset_store))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key in {"reflectanceroi", "reflectance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        reflectance = np.mean(np.asarray(scene_get(scene, "roi reflectance", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "reflectance": reflectance}, None
    if key == "luminanceroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        return {"lum": np.asarray(scene_get(scene, "roi luminance", roi, asset_store=asset_store), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"chromaticityroi", "chromaticity"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        data = np.asarray(scene_get(scene, "chromaticity", roi, asset_store=asset_store), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"illuminantenergyroi", "illuminantenergy"}:
        if "roi" in key:
            roi = _roi_required("scenePlot", p_type, roi_locs)
            energy = np.asarray(scene_get(scene, "roi mean illuminant energy", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            energy = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float).reshape(-1)
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "energy": energy,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    if key in {"illuminantphotonsroi", "illuminantphotons"}:
        if "roi" in key:
            roi = _roi_required("scenePlot", p_type, roi_locs)
            photons = np.asarray(scene_get(scene, "roi mean illuminant photons", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            photons = np.asarray(scene_get(scene, "illuminant photons", asset_store=asset_store), dtype=float).reshape(-1)
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "photons": photons,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    raise UnsupportedOptionError("scenePlot", p_type)


def oi_plot(
    oi: OpticalImage,
    p_type: str = "illuminance hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `oiPlot` user-data without opening a figure."""

    key = param_format(p_type)
    if key == "irradiancephotonsroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi photons", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "irradianceenergyroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi energy", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {
        "irradiancehline",
        "hline",
        "hlineirradiance",
        "irradiancevline",
        "vline",
        "vlineirradiance",
        "irradianceenergyhline",
        "hlineenergy",
        "hlineirradianceenergy",
        "irradianceenergyvline",
        "vlineenergy",
        "vlineirradianceenergy",
        "illuminancehline",
        "horizontallineilluminance",
        "hlineilluminance",
        "illuminancevline",
        "vlineilluminance",
    }:
        roi = _roi_required("oiPlot", p_type, roi_locs)
        udata = dict(oi_get(oi, p_type, roi, *(args[:1] if args else ())))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key == "illuminanceroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        return {"illum": np.asarray(oi_get(oi, "roi illuminance", roi), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "chromaticityroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        data = np.asarray(oi_get(oi, "chromaticity", roi), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    raise UnsupportedOptionError("oiPlot", p_type)


def _sensor_plot_select_capture(sensor: Sensor, capture: Any) -> Sensor:
    capture_index = int(np.rint(float(capture)))
    if capture_index < 1:
        raise IndexError("Sensor capture index must be positive and 1-based.")

    n_captures = int(sensor_get(sensor, "n captures"))
    if capture_index > n_captures:
        raise IndexError(f"Requested sensor capture {capture_index} exceeds available captures ({n_captures}).")
    if n_captures <= 1:
        return sensor

    selected = sensor.clone()
    capture_zero_based = capture_index - 1
    for key in ("volts", "dv"):
        data = selected.data.get(key)
        if data is None:
            continue
        array = np.asarray(data, dtype=float)
        if array.ndim >= 3:
            selected.data[key] = np.asarray(array[:, :, capture_zero_based], dtype=float).copy()
    integration_time = np.asarray(selected.fields.get("integration_time"))
    if integration_time.ndim > 0 and integration_time.size > 1:
        selected.fields["integration_time"] = float(integration_time.reshape(-1)[capture_zero_based])
    return selected


def sensor_plot(
    sensor: Sensor,
    p_type: str = "volts hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensor` user-data without opening a figure."""

    two_lines = bool(_plot_option(args, "twolines", False))
    sensor = _sensor_plot_select_capture(sensor, _plot_option(args, "capture", 1))
    key = param_format(p_type)
    if key in {"electronshline", "hlineelectrons", "electronsvline", "vlineelectrons", "voltshline", "hlinevolts", "voltsvline", "vlinevolts", "dvhline", "hlinedv", "dvvline", "vlinedv"}:
        xy = _roi_required("plotSensor", p_type, roi_locs)
        if two_lines:
            return _sensor_plot_two_lines(sensor, key, xy), None
        return _sensor_plot_line_data(sensor, key, xy), None
    if key == "chromaticity":
        return _sensor_plot_chromaticity(sensor, roi_locs), None
    if key == "shotnoise":
        return _sensor_plot_shot_noise(sensor), None
    if key in {"dsnu", "prnu"}:
        return _sensor_plot_fixed_pattern_noise(sensor, key), None
    if key == "etendue":
        return _sensor_plot_etendue(sensor), None
    if key == "channels":
        return _sensor_plot_channels(sensor), None
    if key == "truesize":
        return _sensor_plot_true_size(sensor), None
    if key in {"cfa", "cfablock"}:
        return _sensor_plot_cfa(sensor, full_array=False), None
    if key == "cfaimage":
        return _sensor_plot_cfa_image(sensor), None
    if key == "cfafull":
        return _sensor_plot_cfa(sensor, full_array=True), None
    if key == "pixelsnr":
        snr, volts, snr_shot, snr_read = pixel_snr(sensor)
        return {"volts": volts, "snr": snr, "snrShot": snr_shot, "snrRead": snr_read}, None
    if key in {"sensorsnr", "snr"}:
        snr, volts, snr_shot, snr_read, snr_dsnu, snr_prnu = sensor_snr(sensor)
        return {
            "volts": volts,
            "snr": snr,
            "snrShot": snr_shot,
            "snrRead": snr_read,
            "snrDSNU": snr_dsnu,
            "snrPRNU": snr_prnu,
        }, None
    if key == "colorfilters":
        return _sensor_plot_spectra(sensor, key), None
    if key == "irfilter":
        return _sensor_plot_spectra(sensor, key), None
    if key in {"pdspectralqe", "pixelspectralqe"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"spectralsr", "sr", "pdspectralsr", "pixelspectralsr"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"spectralqe", "sensorspectralqe"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"sensorspectralsr"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"voltshistogram", "voltshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "volts", roi), None
    if key in {"electronshistogram", "electronshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "electrons", roi), None
    if key in {"dvhistogram", "dvhist", "digitalcountshistogram", "digitalcountshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "dv", roi), None
    raise UnsupportedOptionError("plotSensor", p_type)


def sensor_plot_fft(
    sensor: Sensor,
    ori: str = "h",
    data_type: str = "volts",
    xy: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensorFFT` user-data without opening a figure."""

    sensor = _sensor_plot_select_capture(sensor, _plot_option(args, "capture", 1))
    orientation_key = param_format(ori)
    if orientation_key in {"h", "horizontal"}:
        orientation = "h"
    elif orientation_key in {"v", "vertical"}:
        orientation = "v"
    else:
        raise UnsupportedOptionError("plotSensorFFT", ori)
    selector = _roi_required("plotSensorFFT", f"{ori} {data_type}", xy)
    return _sensor_fft_payload(sensor, orientation, param_format(data_type), selector), None


def ip_plot(
    ip: ImageProcessor,
    p_type: str = "horizontal line",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `ipPlot` user-data without opening a figure."""

    del args, asset_store
    key = param_format(p_type)
    if key in {"horizontalline", "hline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "h", xy), None
    if key in {"verticalline", "vline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "v", xy), None
    if key in {"horizontallineluminance", "hlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "h", xy), None
    if key in {"verticallineluminance", "vlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "v", xy), None
    if key == "chromaticity":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        data = np.asarray(ip_get(ip, "chromaticity", roi), dtype=float)
        xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
        payload = _roi_payload(roi)
        payload["x"] = data[:, 0].copy()
        payload["y"] = data[:, 1].copy()
        payload["XYZ"] = xyz.copy()
        return payload, None
    if key in {"rgbhistogram", "rgb"}:
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        payload["meanRGB"] = np.mean(rgb, axis=0).reshape(-1)
        return payload, None
    if key == "rgb3d":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        return payload, None
    if key == "luminance":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        luminance = np.asarray(xyz[:, 1], dtype=float).reshape(-1)
        payload = _roi_payload(roi)
        payload["luminance"] = luminance.copy()
        payload["meanL"] = float(np.mean(luminance))
        payload["stdLum"] = float(np.std(luminance))
        return payload, None
    if key == "cielab":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        lab = np.asarray(xyz_to_lab(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LAB"] = lab.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLAB"] = np.mean(lab, axis=0).reshape(-1)
        return payload, None
    if key == "cieluv":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        luv = np.asarray(xyz_to_luv(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LUV"] = luv.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLUV"] = np.mean(luv, axis=0).reshape(-1)
        return payload, None
    raise UnsupportedOptionError("ipPlot", p_type)


plotScene = scene_plot
oiPlot = oi_plot
plotSensor = sensor_plot
plotSensorFFT = sensor_plot_fft
ipPlot = ip_plot
