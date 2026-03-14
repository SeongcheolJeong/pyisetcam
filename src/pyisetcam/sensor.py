"""Sensor creation and computation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_color_filter
from .color import luminance_from_photons, xyz_color_matching
from .exceptions import UnsupportedOptionError
from .optics import DEFAULT_FOCAL_LENGTH_M
from .optics import oi_get
from .session import track_session_object
from .types import OpticalImage, Scene, Sensor, SessionContext
from .utils import DEFAULT_WAVE, blackbody, ensure_multiple, ie_parameter_otype, linear_to_srgb, param_format, tile_pattern, xyz_to_linear_srgb

_DEFAULT_PIXEL = {
    "name": "aps",
    "type": "pixel",
    "size_m": np.array([2.8e-6, 2.8e-6], dtype=float),
    "fill_factor": 0.75,
    "layer_thickness_m": np.array([], dtype=float),
    "refractive_indices": np.array([], dtype=float),
    "spectrum": {},
    "conversion_gain_v_per_electron": 1.0e-4,
    "voltage_swing": 1.0,
    "dark_voltage_v_per_sec": 1.0e-3,
    "read_noise_v": 1.0e-3,
    "dsnu_sigma_v": 0.0,
    "prnu_sigma": 0.0,
}
_ELEMENTARY_CHARGE_C = 1.602177e-19
_PLANCK_CONSTANT_J_S = 6.62607015e-34
_LIGHT_SPEED_M_S = 2.99792458e8
_SPATIAL_UNIT_SCALE = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "millimeters": 1e3,
    "millimeter": 1e3,
    "mm": 1e3,
    "microns": 1e6,
    "micron": 1e6,
    "um": 1e6,
}
_TIME_UNIT_SCALE = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e3,
    "msec": 1e3,
    "msecs": 1e3,
    "millisecond": 1e3,
    "milliseconds": 1e3,
    "us": 1e6,
    "usec": 1e6,
    "usecs": 1e6,
    "microsecond": 1e6,
    "microseconds": 1e6,
    "ns": 1e9,
    "nsec": 1e9,
    "nanosecond": 1e9,
    "nanoseconds": 1e9,
}


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def sensor_color_filter(
    cf_type: str = "gaussian",
    wave: np.ndarray | list[float] | tuple[float, ...] | None = None,
    *args: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Create MATLAB-style sensor color filter curves."""

    normalized_type = param_format(cf_type or "gaussian")
    wave_array = (
        np.arange(400.0, 701.0, 1.0, dtype=float)
        if wave is None
        else np.asarray(wave, dtype=float).reshape(-1)
    )
    smooth = -1.0

    if normalized_type == "gaussian":
        c_pos = np.asarray(args[0], dtype=float).reshape(-1) if len(args) >= 1 else np.array([450.0, 550.0, 650.0], dtype=float)
        widths = np.asarray(args[1], dtype=float).reshape(-1) if len(args) >= 2 else np.full(c_pos.shape, 40.0, dtype=float)
        if widths.size == 1 and c_pos.size > 1:
            widths = np.full(c_pos.shape, float(widths[0]), dtype=float)
        if widths.size != c_pos.size:
            raise ValueError("Gaussian filter widths must match the number of center positions.")
        filters = np.zeros((wave_array.size, c_pos.size), dtype=float)
        for idx, (center, width) in enumerate(zip(c_pos, widths, strict=False)):
            filters[:, idx] = np.exp(-0.5 * ((wave_array - float(center)) / float(width)) ** 2)
    elif normalized_type == "block":
        c_pos = np.asarray(args[0], dtype=float).reshape(-1) if len(args) >= 1 else np.array([450.0, 550.0, 650.0], dtype=float)
        widths = np.asarray(args[1], dtype=float).reshape(-1) if len(args) >= 2 else np.full(c_pos.shape, 40.0, dtype=float)
        if widths.size == 1 and c_pos.size > 1:
            widths = np.full(c_pos.shape, float(widths[0]), dtype=float)
        if widths.size != c_pos.size:
            raise ValueError("Block filter widths must match the number of center positions.")
        filters = np.zeros((wave_array.size, c_pos.size), dtype=float)
        for idx, (center, width) in enumerate(zip(c_pos, widths, strict=False)):
            filters[:, idx] = np.abs(wave_array - float(center)) <= (float(width) / 2.0)
    elif normalized_type == "irfilter":
        cut = float(args[0]) if len(args) >= 1 else 700.0
        if len(args) >= 2:
            smooth = float(args[1])
        filters = np.ones((wave_array.size, 1), dtype=float)
        filters[wave_array > cut, 0] = 0.0
    elif normalized_type == "uvfilter":
        cut = float(args[0]) if len(args) >= 1 else 400.0
        if len(args) >= 2:
            smooth = float(args[1])
        filters = np.ones((wave_array.size, 1), dtype=float)
        filters[wave_array < cut, 0] = 0.0
    else:
        raise UnsupportedOptionError("sensorColorFilter", cf_type)

    if smooth > 0:
        filters = gaussian_filter(filters, sigma=float(smooth))

    return np.asarray(filters, dtype=float), wave_array


sensorColorFilter = sensor_color_filter


def _default_pixel(pixel: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_PIXEL)
    if pixel:
        merged.update(pixel)
    merged["size_m"] = np.asarray(merged["size_m"], dtype=float)
    merged["layer_thickness_m"] = np.asarray(merged.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).reshape(-1)
    merged["refractive_indices"] = np.asarray(merged.get("refractive_indices", np.array([], dtype=float)), dtype=float).reshape(-1)
    stored_spectrum = merged.get("spectrum", {})
    if isinstance(stored_spectrum, dict):
        merged["spectrum"] = copy.deepcopy(stored_spectrum)
    elif stored_spectrum is None:
        merged["spectrum"] = {}
    else:
        merged["spectrum"] = copy.deepcopy(dict(vars(stored_spectrum)))
    return merged


def _sensor_base(
    name: str,
    wave: np.ndarray,
    size: tuple[int, int],
    pixel: dict[str, Any] | None,
) -> Sensor:
    sensor = Sensor(name=name)
    sensor.fields.update(
        {
            "wave": np.asarray(wave, dtype=float),
            "spectrum": {"wave": np.asarray(wave, dtype=float).copy()},
            "size": (int(size[0]), int(size[1])),
            "pixel": _default_pixel(pixel),
            "render": {"gamma": 1.0, "scale": False},
            "zero_level": 0.0,
            "analog_gain": 1.0,
            "analog_offset": 0.0,
            "nbits": 10,
            "noise_flag": 2,
            "reuse_noise": False,
            "noise_seed": 0,
            "response_type": "linear",
            "auto_exposure": True,
            "integration_time": 0.0,
            "quantization": "analog",
            "quantization_lut": None,
            "mosaic": True,
            "n_samples_per_pixel": 1,
            "vignetting": 0,
            "etendue": None,
            "pixel_qe": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
            "ir_filter": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
        }
    )
    return sensor


def _spatial_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _time_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _TIME_UNIT_SCALE.get(param_format(unit), 1.0)


def _pixel_size_m(pixel: dict[str, Any]) -> np.ndarray:
    size = np.asarray(pixel.get("size_m", np.array([0.0, 0.0], dtype=float)), dtype=float).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 2)
    if size.size < 2:
        raise ValueError("pixel size must have at least two elements.")
    return size[:2]


def _pixel_gaps_m(pixel: dict[str, Any]) -> np.ndarray:
    row_gap = pixel.get("height_gap_m", pixel.get("heightGapM", pixel.get("heightgap", pixel.get("height_gap", pixel.get("heightGap", 0.0)))))
    col_gap = pixel.get("width_gap_m", pixel.get("widthGapM", pixel.get("widthgap", pixel.get("width_gap", pixel.get("widthGap", 0.0)))))
    return np.array([float(row_gap), float(col_gap)], dtype=float)


def _sensor_spatial_resolution_m(sensor: Sensor) -> np.ndarray:
    pixel = sensor.fields["pixel"]
    return _pixel_size_m(pixel) + _pixel_gaps_m(pixel)


def _sensor_rows_cols(sensor: Sensor) -> tuple[int, int]:
    volts = sensor.data.get("volts")
    if volts is not None:
        shape = np.asarray(volts).shape
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    dv = sensor.data.get("dv")
    if dv is not None:
        shape = np.asarray(dv).shape
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return int(sensor.fields["size"][0]), int(sensor.fields["size"][1])


def _sensor_unit_block(sensor: Sensor) -> tuple[int, int]:
    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    return int(pattern.shape[0]), int(pattern.shape[1])


def _sensor_filter_color_letters(sensor: Sensor) -> str:
    names = sensor.fields.get("filter_names", [])
    return "".join(str(name)[0].lower() if str(name) else "k" for name in names)


def _sensor_cfa_pattern(sensor: Sensor) -> np.ndarray:
    return np.asarray(sensor.fields["pattern"], dtype=int)


def _sensor_spectrum_struct(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("spectrum")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["spectrum"] = stored
    spectrum = dict(stored)
    spectrum["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
    return spectrum


def _sensor_color_struct(sensor: Sensor) -> dict[str, Any]:
    return {
        "filterSpectra": np.asarray(sensor.fields["filter_spectra"], dtype=float).copy(),
        "filterNames": list(sensor.fields["filter_names"]),
        "irFilter": _sensor_ir_filter(sensor),
    }


def _sensor_chart_parameters(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("chartP")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["chartP"] = stored
    chart = dict(stored)
    for key in ("cornerPoints", "rects", "currentRect"):
        if key in chart and chart[key] is not None:
            chart[key] = np.asarray(chart[key]).copy()
    return chart


def _sensor_integration_time_value(sensor: Sensor, unit: Any = None) -> Any:
    integration_time = sensor.fields.get("integration_time", 0.0)
    scale = _time_unit_scale(unit)
    if np.isscalar(integration_time):
        return float(integration_time) * scale
    return np.asarray(integration_time, dtype=float).copy() * scale


def _sensor_microlens(sensor: Sensor) -> dict[str, Any] | None:
    stored = sensor.fields.get("ml")
    if stored is None:
        return None
    return copy.deepcopy(stored)


def _sensor_movement(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("movement")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["movement"] = stored
    movement = dict(stored)
    for key, entry in movement.items():
        if isinstance(entry, np.ndarray):
            movement[key] = entry.copy()
    return movement


def _sensor_human(sensor: Sensor) -> dict[str, Any] | None:
    stored = sensor.fields.get("human")
    if stored is None:
        return None
    return copy.deepcopy(stored)


def _sensor_column_fpn(sensor: Sensor) -> np.ndarray:
    stored = sensor.fields.get("column_fpn")
    if stored is None:
        return np.array([0.0, 0.0], dtype=float)
    return np.asarray(stored, dtype=float).reshape(-1).copy()


def _sensor_dynamic_range(sensor: Sensor, integration_time: Any = None) -> Any:
    if integration_time is None:
        integration_time = sensor_get(sensor, "integration time")
    integration_time_array = np.asarray(integration_time, dtype=float).reshape(-1)
    if integration_time_array.size == 0:
        return None
    if integration_time_array.size == 1 and np.isclose(integration_time_array[0], 0.0):
        return None
    integration_time_array = np.sort(integration_time_array)
    if integration_time_array.size > 1:
        integration_time_array = integration_time_array[[0, -1]]

    pixel = sensor.fields["pixel"]
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    dsnu_sigma = float(pixel["dsnu_sigma_v"])
    voltage_swing = float(pixel["voltage_swing"])

    dark_variance = (dark_voltage * integration_time_array / conversion_gain) * (conversion_gain**2)
    read_variance = read_noise**2
    offset_variance = dsnu_sigma**2
    noise_sd = np.sqrt(dark_variance + read_variance + offset_variance)
    max_voltage = voltage_swing - (dark_voltage * integration_time_array)

    dr = np.full(noise_sd.shape, np.inf, dtype=float)
    positive = noise_sd > 0.0
    dr[positive] = 10.0 * np.log10(np.maximum(max_voltage[positive], 0.0) / noise_sd[positive])
    if dr.size == 1:
        return float(dr[0])
    return dr


def _pixel_dynamic_range(sensor: Sensor, integration_time: Any = None) -> Any:
    if integration_time is None:
        integration_time = sensor_get(sensor, "integration time")
    integration_time_array = np.asarray(integration_time, dtype=float).reshape(-1)
    if integration_time_array.size == 0:
        return None
    if integration_time_array.size > 1:
        return None
    integration_time_value = float(integration_time_array[0])
    if np.isclose(integration_time_value, 0.0):
        return None

    pixel = sensor.fields["pixel"]
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    noise_sd = np.sqrt((dark_voltage * integration_time_value) + (read_noise**2))
    max_voltage = float(pixel["voltage_swing"]) - (dark_voltage * integration_time_value)
    if np.isclose(noise_sd, 0.0):
        return float(np.inf)
    return float(20.0 * np.log10(max(max_voltage, 0.0) / noise_sd))


def _sensor_plane_images(sensor: Sensor, data: np.ndarray | None, *, empty_value: float = np.nan) -> np.ndarray | None:
    if data is None:
        return None
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    rows, cols = array.shape[:2]
    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), rows, cols)
    n_planes = int(sensor_get(sensor, "nfilters"))
    plane_stack = np.full((rows, cols, n_planes), float(empty_value), dtype=float)
    for index in range(1, n_planes + 1):
        mask = pattern == index
        plane_stack[:, :, index - 1][mask] = array[mask]
    return plane_stack


def _sensor_color_data(sensor: Sensor, data: np.ndarray | None, which_sensor: Any) -> np.ndarray | None:
    plane_stack = _sensor_plane_images(sensor, data)
    if plane_stack is None:
        return None
    sensor_index = int(np.rint(float(np.asarray(which_sensor).reshape(-1)[0])))
    if sensor_index < 1 or sensor_index > plane_stack.shape[2]:
        raise ValueError("Requested sensor channel is out of range.")
    channel = np.asarray(plane_stack[:, :, sensor_index - 1], dtype=float)
    return channel[~np.isnan(channel)]


def _pixel_pd_area_m2(sensor: Sensor) -> float:
    return float(np.prod(_pixel_pd_size_from_pixel(sensor.fields["pixel"])))


def _sensor_pd_size_m(sensor: Sensor) -> np.ndarray:
    return _pixel_pd_size_from_pixel(sensor.fields["pixel"])


def _pixel_spectrum_struct(sensor: Sensor) -> dict[str, Any]:
    pixel = sensor.fields["pixel"]
    stored = pixel.get("spectrum", {})
    if isinstance(stored, dict):
        spectrum = copy.deepcopy(stored)
    elif stored is None:
        spectrum = {}
    else:
        spectrum = copy.deepcopy(dict(vars(stored)))
    spectrum["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
    return spectrum


def _pixel_pd_position_from_pixel(pixel: dict[str, Any]) -> np.ndarray:
    stored = pixel.get("pd_position_m")
    if stored is not None:
        position = np.asarray(stored, dtype=float).reshape(-1)
        if position.size == 1:
            position = np.repeat(position, 2)
        return position[:2].copy()
    pixel_size = _pixel_size_m(pixel)
    pd_size = _pixel_pd_size_from_pixel(pixel)
    return ((pixel_size - pd_size) / 2.0).astype(float, copy=False)


def _pixel_pd_size_from_pixel(pixel: dict[str, Any]) -> np.ndarray:
    stored = pixel.get("pd_size_m")
    if stored is not None:
        pd_size = np.asarray(stored, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        return pd_size[:2].copy()
    pixel_size = _pixel_size_m(pixel)
    return np.sqrt(float(pixel.get("fill_factor", 1.0))) * pixel_size[:2]


def _sync_pixel_pd_state(pixel: dict[str, Any]) -> None:
    pixel_size = _pixel_size_m(pixel)
    pd_size = _pixel_pd_size_from_pixel(pixel)
    pd_position = _pixel_pd_position_from_pixel(pixel)
    if np.any(pd_size < 0.0):
        raise ValueError("photodetector size must be nonnegative.")
    if np.any(pd_size - pixel_size > 1e-18):
        raise ValueError("photodetector size must not exceed the pixel size.")
    if np.any(pd_position < 0.0):
        raise ValueError("photodetector position must be nonnegative.")
    if np.any(pd_position + pd_size - pixel_size > 1e-18):
        raise ValueError("photodetector position must keep the photodetector inside the pixel.")
    pixel["pd_size_m"] = pd_size
    pixel["pd_position_m"] = pd_position
    pixel["fill_factor"] = float(np.prod(pd_size) / max(np.prod(pixel_size), 1e-30))


def _sensor_pixel_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    pixel = sensor.fields["pixel"]
    spatial_scale = _spatial_unit_scale(args[0] if args else None)
    pixel_size = _pixel_size_m(pixel)
    pixel_gaps = _pixel_gaps_m(pixel)
    pixel_spacing = pixel_size + pixel_gaps
    pd_size = _sensor_pd_size_m(sensor)
    if key == "name":
        return str(pixel.get("name", ""))
    if key == "type":
        return str(pixel.get("type", "pixel"))
    if key in {"width", "pixelwidth", "pixelwidthmeters", "widthmeters"}:
        return float(pixel_size[1]) * spatial_scale
    if key in {"height", "pixelheight", "pixelheightmeters", "heightmeters"}:
        return float(pixel_size[0]) * spatial_scale
    if key in {"pixelwidthgap", "widthgap", "widthbetweenpixels"}:
        return float(pixel_gaps[1]) * spatial_scale
    if key in {"pixelheightgap", "heightgap", "heightbetweenpixels"}:
        return float(pixel_gaps[0]) * spatial_scale
    if key in {"wspatialresolution", "deltax"}:
        return float(pixel_spacing[1]) * spatial_scale
    if key in {"hspatialresolution", "deltay"}:
        return float(pixel_spacing[0]) * spatial_scale
    if key in {"xyspacing", "dimension"}:
        return np.array([pixel_spacing[1], pixel_spacing[0]], dtype=float) * spatial_scale
    if key in {"size", "pixelsize"}:
        return np.array([pixel_spacing[0], pixel_spacing[1]], dtype=float) * spatial_scale
    if key in {"pixelarea", "area"}:
        return float(np.prod(pixel_spacing) * (spatial_scale**2))
    if key == "fillfactor":
        return float(pixel["fill_factor"])
    if key in {"photodetectorwidth", "pdwidth"}:
        return float(pd_size[1]) * spatial_scale
    if key in {"photodetectorheight", "pdheight"}:
        return float(pd_size[0]) * spatial_scale
    if key in {"pdsize", "photodetectorsize"}:
        return pd_size * spatial_scale
    pd_position = _pixel_pd_position_from_pixel(pixel)
    if key in {"pdxpos", "photodetectorxposition"}:
        return float(pd_position[1]) * spatial_scale
    if key in {"pdypos", "photodetectoryposition"}:
        return float(pd_position[0]) * spatial_scale
    if key == "pdposition":
        return np.array([pd_position[1], pd_position[0]], dtype=float) * spatial_scale
    if key == "pddimension":
        return np.array([pd_size[1], pd_size[0]], dtype=float) * spatial_scale
    if key == "pdarea":
        return float(_pixel_pd_area_m2(sensor) * (spatial_scale**2))
    if key in {"layerthickness", "layerthicknesses"}:
        return np.asarray(pixel.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).copy() * spatial_scale
    if key in {"pixeldepth", "depth", "pixeldepthmeters", "depthmeters", "stackheight"}:
        layer_thickness = np.asarray(pixel.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).reshape(-1)
        return float(np.sum(layer_thickness)) * spatial_scale
    if key in {"refractiveindex", "refractiveindices", "n"}:
        return np.asarray(pixel.get("refractive_indices", np.array([], dtype=float)), dtype=float).copy()
    if key in {"spectrum", "pixelspectrum"}:
        return _pixel_spectrum_struct(sensor)
    if key in {"wave", "wavelength", "wavelengthsamples", "pixelwavelength", "pixelwavelengthsamples"}:
        return np.asarray(sensor.fields["wave"], dtype=float).copy()
    if key in {"binwidth", "wavelengthresolution", "pixelbinwidth"}:
        wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
        return float(wave[1] - wave[0]) if wave.size > 1 else 1.0
    if key in {"nwave", "nwaves", "numberofwavelengthsamples", "pixelnwave"}:
        return int(np.asarray(sensor.fields["wave"], dtype=float).size)
    if key in {
        "pdspectralqe",
        "spectralqe",
        "qe",
        "pixelspectralqe",
        "pixelqe",
        "quantumefficiency",
        "pixelquantumefficiency",
        "photodetectorquantumefficiency",
        "photodetectorspectralquantumefficiency",
    }:
        return _sensor_pixel_qe(sensor)
    if key in {"pdspectralsr", "spectralsr", "sr"}:
        return _pixel_spectral_sr(sensor)
    if key in {"pixeldr", "pixeldynamicrange", "dr", "dynamicrange"}:
        return _pixel_dynamic_range(sensor, args[0] if args else None)
    if key in {"conversiongain", "conversiongainvpelectron", "conversiongainvperelectron", "voltsperelectron"}:
        return float(pixel["conversion_gain_v_per_electron"])
    if key in {"voltageswing", "vswing", "saturationvoltage", "maxvoltage"}:
        return float(pixel["voltage_swing"])
    if key == "wellcapacity":
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["voltage_swing"]) / conversion_gain
    if key in {"darkvolt", "darkvoltage", "darkvolts", "darkvoltageperpixelpersec", "darkvoltageperpixel", "voltspersecond"}:
        return float(pixel["dark_voltage_v_per_sec"])
    if key == "darkelectrons":
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["dark_voltage_v_per_sec"]) / conversion_gain
    if key in {"darkcurrent", "darkcurrentperpixel"}:
        return float(_sensor_pixel_get(sensor, "darkelectrons")) * _ELEMENTARY_CHARGE_C
    if key == "darkcurrentdensity":
        return float(_sensor_pixel_get(sensor, "darkcurrent")) / max(_pixel_pd_area_m2(sensor), 1e-30)
    if key in {"readnoise", "readnoisevolts", "readstandarddeviationvolts", "readnoisestdvolts"}:
        return float(pixel["read_noise_v"])
    if key == "readnoisemillivolts":
        return float(pixel["read_noise_v"]) * 1e3
    if key in {"readnoiseelectrons", "readstandarddeviationelectrons"}:
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["read_noise_v"]) / conversion_gain
    raise KeyError(f"Unsupported sensor pixel parameter: {parameter}")


def _sensor_pixel_set(sensor: Sensor, parameter: str, value: Any) -> Sensor:
    key = param_format(parameter)
    if key in {"pixel", "pixelfields"}:
        sensor.fields["pixel"] = _default_pixel(dict(value))
        sensor.fields["etendue"] = None
        return sensor
    pixel = sensor.fields["pixel"]
    pixel_size = _pixel_size_m(pixel)
    pixel_gaps = _pixel_gaps_m(pixel)
    pd_size = _sensor_pd_size_m(sensor)
    if key == "name":
        pixel["name"] = str(value)
        return sensor
    if key == "type":
        pixel["type"] = str(value)
        return sensor
    if key in {"width", "pixelwidth", "pixelwidthmeters"}:
        pixel["size_m"] = np.array([pixel_size[0], float(value)], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"height", "pixelheight", "pixelheightmeters"}:
        pixel["size_m"] = np.array([float(value), pixel_size[1]], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"pixelwidthgap", "widthgap", "widthbetweenpixels"}:
        pixel["width_gap_m"] = float(value)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"pixelheightgap", "heightgap", "heightbetweenpixels"}:
        pixel["height_gap_m"] = float(value)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"wspatialresolution", "deltax"}:
        width = float(value) - float(pixel_gaps[1])
        if width <= 0.0:
            raise ValueError("pixel width must stay positive after subtracting the width gap.")
        pixel["size_m"] = np.array([pixel_size[0], width], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"hspatialresolution", "deltay"}:
        height = float(value) - float(pixel_gaps[0])
        if height <= 0.0:
            raise ValueError("pixel height must stay positive after subtracting the height gap.")
        pixel["size_m"] = np.array([height, pixel_size[1]], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {
        "sizeconstantfillfactor",
        "sizekeepfillfactor",
        "sizesamefillfactor",
        "pixelsizeconstantfillfactor",
        "pixelsizekeepfillfactor",
        "pixelsizesamefillfactor",
    }:
        size = np.asarray(value, dtype=float).reshape(-1)
        if size.size == 1:
            size = np.repeat(size, 2)
        current_size = np.array([pixel_size[0], pixel_size[1]], dtype=float)
        current_pd_size = _pixel_pd_size_from_pixel(sensor.fields["pixel"])
        stored_pd_position = sensor.fields["pixel"].get("pd_position_m")
        current_pd_position = (
            _pixel_pd_position_from_pixel(sensor.fields["pixel"])
            if stored_pd_position is not None
            else None
        )
        scale_factor = size[:2] / np.maximum(current_size, 1e-30)
        sensor.fields["pixel"]["size_m"] = size[:2].copy()
        sensor.fields["pixel"]["pd_size_m"] = current_pd_size * scale_factor
        if current_pd_position is not None:
            sensor.fields["pixel"]["pd_position_m"] = current_pd_position * scale_factor
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"widthheight", "widthandheight"}:
        size = np.asarray(value, dtype=float).reshape(-1)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = np.array([size[1], size[0]], dtype=float)
        if sensor.fields["pixel"].get("pd_size_m") is not None:
            _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"size", "pixelsize"}:
        size = np.asarray(value, dtype=float)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = size
        if sensor.fields["pixel"].get("pd_size_m") is not None:
            _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"photodetectorwidth", "pdwidth"}:
        sensor.fields["pixel"]["pd_size_m"] = np.array([pd_size[0], float(value)], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"photodetectorheight", "pdheight"}:
        sensor.fields["pixel"]["pd_size_m"] = np.array([float(value), pd_size[1]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "fillfactor":
        sensor.fields["pixel"]["fill_factor"] = float(value)
        sensor.fields["pixel"].pop("pd_size_m", None)
        sensor.fields["pixel"].pop("pd_position_m", None)
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdsize", "photodetectorsize"}:
        pd_size = np.asarray(value, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        sensor.fields["pixel"]["pd_size_m"] = pd_size[:2].copy()
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pdwidthandheight":
        pd_size = np.asarray(value, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        sensor.fields["pixel"]["pd_size_m"] = np.array([pd_size[1], pd_size[0]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdxpos", "photodetectorxposition"}:
        pd_position = _pixel_pd_position_from_pixel(sensor.fields["pixel"])
        sensor.fields["pixel"]["pd_position_m"] = np.array([pd_position[0], float(value)], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdypos", "photodetectoryposition"}:
        pd_position = _pixel_pd_position_from_pixel(sensor.fields["pixel"])
        sensor.fields["pixel"]["pd_position_m"] = np.array([float(value), pd_position[1]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pdposition":
        pd_position = np.asarray(value, dtype=float).reshape(-1)
        if pd_position.size == 1:
            pd_position = np.repeat(pd_position, 2)
        sensor.fields["pixel"]["pd_position_m"] = np.array([pd_position[1], pd_position[0]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pddimension":
        pd_dimension = np.asarray(value, dtype=float).reshape(-1)
        if pd_dimension.size == 1:
            pd_dimension = np.repeat(pd_dimension, 2)
        return _sensor_pixel_set(sensor, "pd size", np.array([pd_dimension[1], pd_dimension[0]], dtype=float))
    if key in {"layerthickness", "layerthicknesses"}:
        sensor.fields["pixel"]["layer_thickness_m"] = np.asarray(value, dtype=float).reshape(-1).copy()
        return sensor
    if key in {"refractiveindex", "refractiveindices", "n"}:
        sensor.fields["pixel"]["refractive_indices"] = np.asarray(value, dtype=float).reshape(-1).copy()
        return sensor
    if key in {"spectrum", "pixelspectrum"}:
        payload = dict(value) if isinstance(value, dict) else dict(vars(value))
        if "wave" in payload:
            sensor = _sensor_update_wave(sensor, np.asarray(payload["wave"], dtype=float).reshape(-1))
        payload["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
        sensor.fields["pixel"]["spectrum"] = copy.deepcopy(payload)
        return sensor
    if key in {"wave", "wavelength", "wavelengthsamples", "pixelwavelength", "pixelwavelengthsamples"}:
        sensor = _sensor_update_wave(sensor, np.asarray(value, dtype=float).reshape(-1))
        spectrum = _pixel_spectrum_struct(sensor)
        sensor.fields["pixel"]["spectrum"] = copy.deepcopy(spectrum)
        return sensor
    if key in {
        "pdspectralqe",
        "spectralqe",
        "qe",
        "pixelspectralqe",
        "pixelqe",
        "quantumefficiency",
        "pixelquantumefficiency",
        "photodetectorquantumefficiency",
        "photodetectorspectralquantumefficiency",
    }:
        qe = np.asarray(value, dtype=float).reshape(-1)
        if qe.size == 1:
            sensor.fields["pixel_qe"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(qe[0]), dtype=float)
        elif qe.size == np.asarray(sensor.fields["wave"], dtype=float).size:
            sensor.fields["pixel_qe"] = qe
        else:
            raise ValueError("pixel spectral QE must match the sensor wavelength sampling.")
        return sensor
    if key in {"conversiongain", "conversiongainvpelectron", "conversiongainvperelectron", "voltsperelectron"}:
        sensor.fields["pixel"]["conversion_gain_v_per_electron"] = float(value)
        return sensor
    if key in {"voltageswing", "vswing", "saturationvoltage", "maxvoltage"}:
        sensor.fields["pixel"]["voltage_swing"] = float(value)
        return sensor
    if key in {"darkvolt", "darkvoltage", "darkvolts", "darkvoltageperpixelpersec", "darkvoltageperpixel", "voltspersecond"}:
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = float(value)
        return sensor
    if key in {"readnoise", "readnoiseelectrons", "readstandarddeviationelectrons"}:
        conversion_gain = max(float(sensor.fields["pixel"]["conversion_gain_v_per_electron"]), 1e-30)
        sensor.fields["pixel"]["read_noise_v"] = float(value) * conversion_gain
        return sensor
    if key in {"readnoisevolts", "readstandarddeviationvolts", "readnoisestdvolts"}:
        sensor.fields["pixel"]["read_noise_v"] = float(value)
        return sensor
    if key == "readnoisemillivolts":
        sensor.fields["pixel"]["read_noise_v"] = float(value) * 1e-3
        return sensor
    raise KeyError(f"Unsupported sensor pixel parameter: {parameter}")


def _copy_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _spectrum_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        spectrum = dict(value)
    else:
        spectrum = {}
        if hasattr(value, "__dict__"):
            spectrum.update(vars(value))
    if "wave" not in spectrum:
        wave = getattr(value, "wave", None)
        if wave is None:
            raise ValueError("Spectrum value must include a wave field.")
        spectrum["wave"] = wave
    spectrum["wave"] = np.asarray(spectrum["wave"], dtype=float).reshape(-1)
    return spectrum


def _sensor_color_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        color = dict(value)
    else:
        color = {}
        if hasattr(value, "__dict__"):
            color.update(vars(value))
    return color


def _microlens_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    microlens = {}
    if hasattr(value, "__dict__"):
        microlens.update(vars(value))
    return copy.deepcopy(microlens)


def _human_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    human = {}
    if hasattr(value, "__dict__"):
        human.update(vars(value))
    return copy.deepcopy(human)


def _movement_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        movement = dict(value)
    else:
        movement = {}
        if hasattr(value, "__dict__"):
            movement.update(vars(value))
    for key, entry in list(movement.items()):
        if isinstance(entry, np.ndarray):
            movement[key] = entry.copy()
    return movement


def _movement_positions_from_value(value: Any) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.size == 0:
        return positions.reshape(0, 2)
    if positions.ndim == 1:
        if positions.size != 2:
            raise ValueError("sensor movement positions must be Nx2.")
        return positions.reshape(1, 2)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("sensor movement positions must be Nx2.")
    return positions.copy()


def _sensor_unit_block_config(sensor: Sensor) -> np.ndarray:
    pixel = sensor.fields["pixel"]
    pixel_size = _pixel_size_m(pixel)
    pattern = _sensor_cfa_pattern(sensor)
    x, y = np.meshgrid(
        np.arange(pattern.shape[1], dtype=float) * pixel_size[1],
        np.arange(pattern.shape[0], dtype=float) * pixel_size[0],
    )
    return np.column_stack([x.ravel(), y.ravel()])


def _sensor_cfa_struct(sensor: Sensor) -> dict[str, Any]:
    pattern = _sensor_cfa_pattern(sensor)
    return {
        "pattern": pattern.copy(),
        "unitBlock": {
            "rows": int(pattern.shape[0]),
            "cols": int(pattern.shape[1]),
            "config": _sensor_unit_block_config(sensor),
        },
    }


def _sensor_cfa_name(sensor: Sensor) -> str:
    pattern = _sensor_cfa_pattern(sensor)
    filter_colors = "".join(sorted(_sensor_filter_color_letters(sensor)))
    if pattern.size == 1:
        return "Monochrome"
    if pattern.shape != (2, 2):
        return "Other"
    if filter_colors == "bgr":
        return "Bayer RGB"
    if filter_colors == "cmy":
        return "Bayer CMY"
    if filter_colors == "bgrw":
        return "RGBW"
    return "Other"


def _cfa_pattern_from_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "pattern" in value:
            return np.asarray(value["pattern"], dtype=int)
        if "cfapattern" in value:
            return np.asarray(value["cfapattern"], dtype=int)
    pattern = getattr(value, "pattern", None)
    if pattern is not None:
        return np.asarray(pattern, dtype=int)
    raise ValueError("CFA value must include a pattern.")


def _sensor_filter_display_colors(sensor: Sensor) -> np.ndarray:
    letters = list(_sensor_filter_color_letters(sensor))
    filter_spectra = _sensor_combined_qe(sensor, dtype=float)
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
        wave = np.asarray(sensor.fields["wave"], dtype=float)
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


def _color_block_matrix(wave_nm: np.ndarray, extrap_val: float = 0.0) -> np.ndarray:
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    default_wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    blue_count = 10
    green_count = 8
    red_count = default_wave.size - blue_count - green_count
    default_matrix = np.column_stack(
        (
            np.concatenate((np.zeros(blue_count + green_count), np.ones(red_count))),
            np.concatenate((np.zeros(blue_count), np.ones(green_count), np.zeros(red_count))),
            np.concatenate((np.ones(blue_count), np.zeros(green_count + red_count))),
        )
    )
    if np.array_equal(wave, default_wave):
        matrix = default_matrix.copy()
    else:
        matrix = np.empty((wave.size, 3), dtype=float)
        for index in range(3):
            matrix[:, index] = np.interp(
                wave,
                default_wave,
                default_matrix[:, index],
                left=float(extrap_val),
                right=float(extrap_val),
            )
    column_sums = np.sum(matrix, axis=0, keepdims=True)
    matrix = np.divide(
        matrix,
        np.maximum(column_sums, 1e-12),
        out=np.zeros_like(matrix),
        where=column_sums > 0.0,
    )
    white_spd = blackbody(wave, 6500.0, kind="quanta").reshape(-1)
    white_spd = white_spd / max(float(np.max(white_spd)), 1e-12)
    rgb = white_spd @ matrix
    matrix = matrix @ np.diag(1.0 / np.maximum(rgb, 1e-12))
    return matrix


def _sensor_display_transform(sensor: Sensor) -> np.ndarray:
    block_matrix = _color_block_matrix(np.asarray(sensor.fields["wave"], dtype=float), extrap_val=0.2)
    filter_spectra = np.asarray(sensor_get(sensor, "filterspectra"), dtype=float)
    filter_rgb = block_matrix.T @ filter_spectra
    transform = filter_rgb.T
    scale = max(float(np.max(transform)), 1e-12)
    return transform / scale


def _image_linear_transform(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    rows, cols, channels = array.shape
    xw = array.reshape(rows * cols, channels)
    transformed = xw @ np.asarray(transform, dtype=float)
    return transformed.reshape(rows, cols, -1)


def _sensor_render_state(sensor: Sensor) -> dict[str, Any]:
    render = sensor.fields.get("render")
    if not isinstance(render, dict):
        render = {}
        sensor.fields["render"] = render
    render.setdefault("gamma", 1.0)
    render.setdefault("scale", False)
    return render


def _sensor_pixel_qe(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    stored = sensor.fields.get("pixel_qe")
    if stored is None:
        return np.ones(np.asarray(sensor.fields["wave"], dtype=float).size, dtype=dtype)
    qe = np.asarray(stored, dtype=dtype).reshape(-1)
    if qe.size == 1:
        return np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(qe[0]), dtype=dtype)
    return qe


def _sensor_ir_filter(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    stored = sensor.fields.get("ir_filter")
    if stored is None:
        return np.ones(np.asarray(sensor.fields["wave"], dtype=float).size, dtype=dtype)
    ir_filter = np.asarray(stored, dtype=dtype).reshape(-1)
    if ir_filter.size == 1:
        return np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=dtype)
    return ir_filter


def _sensor_combined_qe(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    filter_spectra = np.asarray(sensor.fields["filter_spectra"], dtype=dtype)
    if filter_spectra.ndim == 1:
        filter_spectra = filter_spectra.reshape(-1, 1)
    pixel_qe = _sensor_pixel_qe(sensor, dtype=dtype).reshape(-1, 1)
    ir_filter = _sensor_ir_filter(sensor, dtype=dtype).reshape(-1, 1)
    return filter_spectra * pixel_qe * ir_filter


def _sensor_combined_sr(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    wave_m = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1, 1) * 1e-9
    conversion = (wave_m * _ELEMENTARY_CHARGE_C) / (_PLANCK_CONSTANT_J_S * _LIGHT_SPEED_M_S)
    return (conversion * _sensor_combined_qe(sensor, dtype=dtype)).astype(dtype, copy=False)


def _pixel_spectral_sr(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    wave_m = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1) * 1e-9
    pixel_qe = _sensor_pixel_qe(sensor, dtype=dtype).reshape(-1)
    return ((wave_m * _ELEMENTARY_CHARGE_C) / (_PLANCK_CONSTANT_J_S * _LIGHT_SPEED_M_S) * pixel_qe).astype(dtype, copy=False)


def _sensor_rgb_source(sensor: Sensor, data_type: str) -> tuple[np.ndarray | None, str]:
    key = param_format(data_type)
    resolved = key
    if key in {"dvorvolts", "digitalorvolts"}:
        if sensor.data.get("dv") is not None:
            resolved = "dv"
            source = sensor.data.get("dv")
        else:
            resolved = "volts"
            source = sensor.data.get("volts")
    elif key in {"dv", "digitalvalue", "digitalvalues"}:
        source = sensor.data.get("dv")
    elif key == "electrons":
        source = _sensor_electrons(sensor)
    else:
        resolved = "volts"
        source = sensor.data.get("volts")

    if source is None:
        return None, resolved

    array = np.asarray(source, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    return np.asarray(array, dtype=float), resolved


def _sensor_display_scale(sensor: Sensor, data: np.ndarray, data_type: str, *, scale_max: bool) -> float:
    if scale_max:
        return float(max(np.max(np.asarray(data, dtype=float)), 1e-12))
    key = param_format(data_type)
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        return float(max(sensor_get(sensor, "max digital value"), 1.0))
    return float(max(sensor_get(sensor, "max output"), 1e-12))


def _sensor_rgb_image(
    sensor: Sensor,
    data_type: str = "volts",
    gamma: float | None = None,
    scale_max: bool | None = None,
) -> np.ndarray | None:
    data, resolved_type = _sensor_rgb_source(sensor, data_type)
    if data is None:
        return None

    gamma_value = float(sensor_get(sensor, "gamma") if gamma is None else gamma)
    scale_max_value = bool(sensor_get(sensor, "scale max") if scale_max is None else scale_max)
    normalized = np.clip(
        np.asarray(data, dtype=float) / _sensor_display_scale(sensor, data, resolved_type, scale_max=scale_max_value),
        0.0,
        1.0,
    )

    if int(sensor_get(sensor, "nfilters")) <= 1:
        return np.power(normalized, gamma_value)

    linear = _sensor_plane_images(sensor, np.asarray(data, dtype=float), empty_value=0.0)
    if linear is None:
        return None
    filter_letters = str(sensor_get(sensor, "filtercolorletters")).lower()
    if filter_letters == "rgb":
        transformed = linear
    elif filter_letters == "wrgb":
        transformed = _image_linear_transform(
            linear,
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
        )
    elif filter_letters == "rgbw":
        transformed = _image_linear_transform(
            linear,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=float,
            ),
        )
    else:
        transformed = _image_linear_transform(linear, _sensor_display_transform(sensor))
    transformed = np.clip(transformed / _sensor_display_scale(sensor, data, resolved_type, scale_max=scale_max_value), 0.0, 1.0)
    transformed = np.power(transformed, gamma_value)
    if transformed.shape[2] == 3:
        return linear_to_srgb(transformed)
    return transformed


def _interp_spectral_array(old_wave: np.ndarray, new_wave: np.ndarray, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(new_wave.shape, float(array), dtype=float)
    if array.ndim == 1:
        if array.size == 1:
            return np.full(new_wave.shape, float(array[0]), dtype=float)
        return np.interp(new_wave, old_wave, array, left=0.0, right=0.0)
    if array.shape[0] == 1:
        return np.repeat(array.astype(float), new_wave.size, axis=0)
    interpolated = np.empty((new_wave.size, array.shape[1]), dtype=float)
    for index in range(array.shape[1]):
        interpolated[:, index] = np.interp(new_wave, old_wave, array[:, index], left=0.0, right=0.0)
    return interpolated


def _sensor_update_wave(sensor: Sensor, new_wave: np.ndarray) -> Sensor:
    old_wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
    new_wave = np.asarray(new_wave, dtype=float).reshape(-1)

    if old_wave.size > 0 and not np.array_equal(old_wave, new_wave):
        filter_spectra = np.asarray(sensor.fields.get("filter_spectra"), dtype=float)
        if filter_spectra.ndim >= 1 and filter_spectra.shape[0] == old_wave.size:
            sensor.fields["filter_spectra"] = _interp_spectral_array(old_wave, new_wave, filter_spectra)

        pixel_qe = np.asarray(sensor.fields.get("pixel_qe"), dtype=float).reshape(-1)
        if pixel_qe.size == old_wave.size:
            sensor.fields["pixel_qe"] = _interp_spectral_array(old_wave, new_wave, pixel_qe)

        ir_filter = np.asarray(sensor.fields.get("ir_filter"), dtype=float).reshape(-1)
        if ir_filter.size == old_wave.size:
            sensor.fields["ir_filter"] = _interp_spectral_array(old_wave, new_wave, ir_filter)

    sensor.fields["wave"] = new_wave
    spectrum = _sensor_spectrum_struct(sensor)
    spectrum["wave"] = new_wave.copy()
    sensor.fields["spectrum"] = spectrum
    return sensor


def _sensor_aligned_dimension(value: Any, block_size: int) -> int:
    dimension = int(np.floor(float(value)))
    if block_size <= 1:
        return dimension
    return int(np.floor(dimension / block_size) * block_size)


def _sensor_clear_data(sensor: Sensor) -> None:
    sensor.data.clear()


def _sensor_electrons(sensor: Sensor) -> np.ndarray | None:
    volts = sensor.data.get("volts")
    if volts is None:
        return None
    analog_gain = float(sensor.fields["analog_gain"])
    analog_offset = float(sensor.fields["analog_offset"])
    conversion_gain = float(sensor.fields["pixel"]["conversion_gain_v_per_electron"])
    return np.clip((np.asarray(volts, dtype=float) * analog_gain) - analog_offset, 0.0, None) / max(
        conversion_gain, 1e-12
    )


def _filter_bundle(
    filter_name: str | list[str] | tuple[str, ...],
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(filter_name, (list, tuple)):
        spectra_parts: list[np.ndarray] = []
        names: list[str] = []
        for current_filter in filter_name:
            _, current_spectra, current_names = asset_store.load_color_filters(current_filter, wave_nm=wave)
            current_spectra = np.asarray(current_spectra, dtype=float)
            if current_spectra.ndim == 1:
                current_spectra = current_spectra.reshape(-1, 1)
            spectra_parts.append(current_spectra)
            names.extend(current_names)
        return np.concatenate(spectra_parts, axis=1), names
    _, spectra, names = asset_store.load_color_filters(filter_name, wave_nm=wave)
    return np.asarray(spectra, dtype=float), names


def _matlab_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in np.atleast_1d(value)]


def _sensor_from_upstream_model(
    relative_path: str,
    *,
    asset_store: AssetStore,
) -> Sensor:
    data = asset_store.load_mat(relative_path)
    model = data["sensor"]
    wave = np.asarray(model.spectrum.wave, dtype=float).reshape(-1)
    size = (int(model.rows), int(model.cols))
    pixel_width = float(model.pixel.width)
    pixel_height = float(model.pixel.height)
    pd_width = float(getattr(model.pixel, "pdWidth", pixel_width))
    pd_height = float(getattr(model.pixel, "pdHeight", pixel_height))
    fill_factor = float((pd_width * pd_height) / max(pixel_width * pixel_height, 1e-12))
    pixel = {
        "size_m": np.array([pixel_height, pixel_width], dtype=float),
        "height_gap_m": float(getattr(model.pixel, "heightGap", 0.0)),
        "width_gap_m": float(getattr(model.pixel, "widthGap", 0.0)),
        "fill_factor": fill_factor,
        "conversion_gain_v_per_electron": float(model.pixel.conversionGain),
        "voltage_swing": float(model.pixel.voltageSwing),
        "dark_voltage_v_per_sec": float(model.pixel.darkVoltage),
        "read_noise_v": float(model.pixel.readNoise),
    }
    sensor = _sensor_base(str(model.name), wave, size, pixel)
    pattern = np.asarray(model.cfa.pattern, dtype=int)
    if pattern.ndim == 0:
        pattern = pattern.reshape(1, 1)
    filter_spectra = np.asarray(model.color.filterSpectra, dtype=float)
    if filter_spectra.ndim == 1:
        filter_spectra = filter_spectra.reshape(-1, 1)
    sensor.fields["pattern"] = pattern
    sensor.fields["filter_spectra"] = filter_spectra
    sensor.fields["filter_names"] = _matlab_string_list(model.color.filterNames)
    sensor.fields["pixel_qe"] = np.asarray(getattr(model.pixel, "spectralQE", np.ones(wave.size)), dtype=float).reshape(-1)
    sensor.fields["ir_filter"] = np.asarray(getattr(model.color, "irFilter", np.ones(wave.size)), dtype=float).reshape(-1)
    sensor.fields["analog_gain"] = float(model.analogGain)
    sensor.fields["analog_offset"] = float(model.analogOffset)
    quantization = getattr(model, "quantization", "analog")
    if hasattr(quantization, "bits") and hasattr(quantization, "method"):
        bits_value = np.asarray(quantization.bits, dtype=float).reshape(-1)
        bits = int(np.rint(float(bits_value[0]))) if bits_value.size else 0
        method = str(quantization.method)
        if param_format(method) == "linear" and bits > 0:
            sensor.fields["quantization"] = f"{bits} bit"
        else:
            sensor.fields["quantization"] = method
        sensor.fields["nbits"] = bits
    else:
        sensor.fields["quantization"] = str(quantization)
    sensor.fields["integration_time"] = float(model.integrationTime)
    sensor.fields["auto_exposure"] = bool(model.AE)
    sensor.fields["noise_flag"] = int(model.noiseFlag)
    sensor.fields["cds"] = bool(getattr(model, "CDS", False))
    if hasattr(model, "sigmaOffsetFPN"):
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(model.sigmaOffsetFPN)
    if hasattr(model, "sigmaGainFPN"):
        sensor.fields["pixel"]["prnu_sigma"] = float(model.sigmaGainFPN)
    if hasattr(model, "offsetFPNimage") and np.size(model.offsetFPNimage):
        sensor.fields["offset_fpn_image"] = np.asarray(model.offsetFPNimage, dtype=float).copy()
    if hasattr(model, "gainFPNimage") and np.size(model.gainFPNimage):
        sensor.fields["gain_fpn_image"] = np.asarray(model.gainFPNimage, dtype=float).copy()
    return sensor


def _sensor_variant_name(args: tuple[Any, ...], default: str) -> str:
    if not args:
        return default
    return str(args[0])


def _sensor_vendor_mt9v024(variant: str, *, asset_store: AssetStore) -> Sensor:
    normalized = param_format(variant)
    mapping = {
        "mono": "data/sensor/auto/MT9V024SensorMono.mat",
        "monochrome": "data/sensor/auto/MT9V024SensorMono.mat",
        "rgb": "data/sensor/auto/MT9V024SensorRGB.mat",
        "rccc": "data/sensor/auto/MT9V024SensorRCCC.mat",
        "rgbw": "data/sensor/auto/MT9V024SensorRGBW.mat",
    }
    if normalized not in mapping:
        raise UnsupportedOptionError("sensorCreate", f"MT9V024/{variant}")
    return _sensor_from_upstream_model(mapping[normalized], asset_store=asset_store)


def _sensor_vendor_ar0132at(variant: str, *, asset_store: AssetStore) -> Sensor:
    normalized = param_format(variant)
    mapping = {
        "rgb": "data/sensor/auto/ar0132atSensorRGB.mat",
        "rccc": "data/sensor/auto/ar0132atSensorRCCC.mat",
        "rgbw": "data/sensor/auto/ar0132atSensorRGBW.mat",
    }
    if normalized not in mapping:
        raise UnsupportedOptionError("sensorCreate", f"ar0132at/{variant}")
    return _sensor_from_upstream_model(mapping[normalized], asset_store=asset_store)


def _sensor_vendor_imx363(*, asset_store: AssetStore) -> Sensor:
    return _sensor_from_upstream_model("data/sensor/sony/imx363.mat", asset_store=asset_store)


def _matlab_kv_pairs(args: tuple[Any, ...], *, function_name: str) -> list[tuple[str, Any]]:
    if len(args) % 2 != 0:
        raise ValueError(f"{function_name} expects MATLAB-style key/value pairs.")
    return [(param_format(str(args[index])), args[index + 1]) for index in range(0, len(args), 2)]


def _apply_sensor_settings(sensor: Sensor, settings: list[tuple[str, Any]]) -> Sensor:
    updated = sensor
    for parameter, value in settings:
        updated = sensor_set(updated, parameter, value)
    return updated


def _sensor_create_ovt_large_pair(*, asset_store: AssetStore) -> tuple[Sensor, Sensor]:
    common_settings = [
        ("size", np.array([968, 1288], dtype=int)),
        ("pixel size same fill factor", 2.8e-6),
        ("pixel voltage swing", 22000.0 * 49e-6),
        ("pixel conversion gain", 49e-6),
        ("pixel fill factor", 1.0),
        ("pixel read noise electrons", 3.05),
        ("pixel dark voltage", 25.6 * 49e-6),
        ("analog gain", 1.0),
        ("quantization", "12 bit"),
        ("bits", 12),
        ("name", "ovt-LPDLCG"),
    ]
    filter_spectra, filter_names, _ = ie_read_color_filter(
        DEFAULT_WAVE,
        "data/sensor/colorfilters/OVT/ovt-large.mat",
        asset_store=asset_store,
    )
    common_settings.extend(
        [
            ("filter spectra", filter_spectra),
            ("filter names", filter_names),
        ]
    )
    lpd_lcg = _apply_sensor_settings(sensor_create(asset_store=asset_store), common_settings)
    lpd_hcg = _apply_sensor_settings(
        lpd_lcg.clone(),
        [
            ("pixel read noise electrons", 0.83),
            ("analog gain", 49.0 / 200.0),
            ("name", "ovt-LPDHCG"),
        ],
    )
    return lpd_lcg, lpd_hcg


def _sensor_create_ovt_small(*, asset_store: AssetStore) -> Sensor:
    filter_spectra, filter_names, _ = ie_read_color_filter(
        DEFAULT_WAVE,
        "data/sensor/colorfilters/OVT/ovt-large.mat",
        asset_store=asset_store,
    )
    settings = [
        ("size", np.array([968, 1288], dtype=int)),
        ("pixel size same fill factor", 2.8e-6),
        ("pixel voltage swing", 7900.0 * 49e-6),
        ("pixel conversion gain", 49e-6),
        ("pixel fill factor", 1e-2),
        ("pixel read noise electrons", 0.83),
        ("pixel dark voltage", 4.2 * 49e-6),
        ("quantization", "12 bit"),
        ("bits", 12),
        ("name", "ovt-SPDLCG"),
        ("filter spectra", filter_spectra),
        ("filter names", filter_names),
    ]
    return _apply_sensor_settings(sensor_create(asset_store=asset_store), settings)


def sensor_create(
    sensor_type: str = "default",
    pixel: dict[str, Any] | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Create a supported sensor."""

    store = _store(asset_store)
    normalized = param_format(sensor_type)
    if normalized in {"mt9v024", "ar0132at"} and isinstance(pixel, str):
        args = (pixel, *args)
        pixel = None
    pixel_dict = _default_pixel(pixel)
    wave = np.asarray(pixel_dict.get("wave", DEFAULT_WAVE), dtype=float)
    size = tuple(pixel_dict.get("size", (72, 88)))

    if normalized in {"default", "color", "bayer", "rgb", "bayergrbg", "bayer-grbg"}:
        sensor = _sensor_base("bayer-grbg", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 1], [3, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayerrggb", "bayer-rggb"}:
        sensor = _sensor_base("bayer-rggb", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "monochrome":
        sensor = _sensor_base("monochrome", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("monochrome", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"rgbw", "interleaved"}:
        sensor = _sensor_base("rgbw", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [3, 4]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("interleavedrgbw", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "rccc":
        sensor = _sensor_base("rccc", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 2], [2, 1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle(["r", "w"], wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayer-ycmy", "ycmy", "bayer(ycmy)"}:
        sensor = _sensor_base("bayer-ycmy", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 1], [3, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("cym", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayer-cyym", "cyym", "bayer(cyym)"}:
        sensor = _sensor_base("bayer-cyym", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("cym", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "mt9v024":
        return track_session_object(session, _sensor_vendor_mt9v024(_sensor_variant_name(args, "rgb"), asset_store=store))

    if normalized == "ar0132at":
        return track_session_object(session, _sensor_vendor_ar0132at(_sensor_variant_name(args, "rgb"), asset_store=store))

    if normalized in {"imx363", "googlepixel4a"}:
        sensor = _sensor_vendor_imx363(asset_store=store)
        if args:
            if len(args) % 2 != 0:
                raise ValueError("sensorCreate('IMX363', ...) expects key/value pairs.")
            current_gain = float(sensor.fields["analog_gain"])
            current_offset = float(sensor.fields["analog_offset"])
            black_level_v = current_offset / max(current_gain, 1e-12)
            for index in range(0, len(args), 2):
                parameter = param_format(args[index])
                value = args[index + 1]
                if parameter in {"rowcol", "rowcolsize"}:
                    rows, cols = np.rint(np.asarray(value, dtype=float).reshape(-1)[:2]).astype(int)
                    sensor = sensor_set(sensor, "rows", int(rows))
                    sensor = sensor_set(sensor, "cols", int(cols))
                    continue
                if parameter in {"isospeed", "iso"}:
                    iso_speed = float(value)
                    analog_gain = 55.0 / max(iso_speed, 1e-12)
                    sensor = sensor_set(sensor, "analog gain", analog_gain)
                    sensor = sensor_set(sensor, "analog offset", black_level_v * analog_gain)
                    continue
                if parameter in {"exposuretime", "exptime", "integrationtime"}:
                    sensor = sensor_set(sensor, "integration time", value)
                    continue
                if parameter == "wave":
                    sensor = sensor_set(sensor, "wave", value)
                    continue
                sensor = sensor_set(sensor, parameter, value)
        return track_session_object(session, sensor)

    if normalized == "ideal":
        return sensor_create_ideal("xyz", None, asset_store=store, session=session)

    raise UnsupportedOptionError("sensorCreate", sensor_type)


def sensor_create_ideal(
    ideal_type: str = "xyz",
    sensor_example: Sensor | None = None,
    pixel_size_m: float | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Create an ideal milestone-one sensor."""

    store = _store(asset_store)
    normalized = param_format(ideal_type)
    pixel = _default_pixel(sensor_example.fields["pixel"] if sensor_example is not None else None)
    if pixel_size_m is not None:
        pixel["size_m"] = np.array([pixel_size_m, pixel_size_m], dtype=float)
    pixel["fill_factor"] = 1.0
    size = sensor_example.fields["size"] if sensor_example is not None else (72, 88)
    wave = sensor_example.fields["wave"] if sensor_example is not None else DEFAULT_WAVE.copy()

    if normalized in {"monochrome"}:
        sensor = _sensor_base("ideal-monochrome", np.asarray(wave, dtype=float), size, pixel)
        sensor.fields["pattern"] = np.array([[1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("monochrome", np.asarray(wave, dtype=float), asset_store=store)
        sensor.fields["noise_flag"] = 0
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
        sensor.fields["pixel"]["read_noise_v"] = 0.0
        sensor.fields["pixel"]["voltage_swing"] = 1e6
        return track_session_object(session, sensor)

    if normalized in {"xyz", "matchxyz"}:
        sensor = _sensor_base("ideal-xyz", np.asarray(wave, dtype=float), size, pixel)
        sensor.fields["pattern"] = np.array([[1, 2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("xyz", np.asarray(wave, dtype=float), asset_store=store)
        sensor.fields["noise_flag"] = 0
        sensor.fields["mosaic"] = False
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
        sensor.fields["pixel"]["read_noise_v"] = 0.0
        sensor.fields["pixel"]["voltage_swing"] = 1e6
        return track_session_object(session, sensor)

    if normalized == "match" and sensor_example is not None:
        sensor = sensor_example.clone()
        sensor.name = f"ideal-{sensor_example.name}"
        sensor.fields["pixel"] = pixel
        sensor.fields["noise_flag"] = 0
        return track_session_object(session, sensor)

    raise UnsupportedOptionError("sensorCreateIdeal", ideal_type)


def sensor_create_split_pixel(
    *args: Any,
    asset_store: AssetStore | None = None,
) -> list[Sensor]:
    """Create a MATLAB-style split-pixel sensor array."""

    store = _store(asset_store)
    settings = _matlab_kv_pairs(args, function_name="sensorCreateSplitPixel")
    array_type = "ovt"
    shared_settings: list[tuple[str, Any]] = []
    for parameter, value in settings:
        if parameter == "arraytype":
            array_type = str(value)
            continue
        shared_settings.append((parameter, value))

    normalized_array_type = param_format(array_type)
    if normalized_array_type != "ovt":
        raise UnsupportedOptionError("sensorCreateSplitPixel", array_type)

    large_lcg, large_hcg = _sensor_create_ovt_large_pair(asset_store=store)
    small_lcg = _sensor_create_ovt_small(asset_store=store)
    return [_apply_sensor_settings(sensor.clone(), shared_settings) for sensor in (large_lcg, large_hcg, small_lcg)]


def sensor_create_array(
    *args: Any,
    asset_store: AssetStore | None = None,
) -> list[Sensor]:
    """Create a MATLAB-style coordinated sensor array."""

    settings = _matlab_kv_pairs(args, function_name="sensorCreateArray")
    array_type = "ovt"
    forward_args: list[Any] = []
    for parameter, value in settings:
        if parameter == "arraytype":
            array_type = str(value)
        forward_args.extend((parameter, value))

    normalized_array_type = param_format(array_type)
    if normalized_array_type == "ovt":
        return sensor_create_split_pixel(*forward_args, asset_store=asset_store)
    raise UnsupportedOptionError("sensorCreateArray", array_type)


def sensor_compute_array(
    sensor_array: list[Sensor] | tuple[Sensor, ...],
    oi: OpticalImage,
    *args: Any,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> tuple[Sensor, list[Sensor]]:
    """Compute a split-pixel sensor array and its combined response."""

    settings = _matlab_kv_pairs(args, function_name="sensorComputeArray")
    method = "saturated"
    saturated_fraction = 0.95
    for parameter, value in settings:
        if parameter == "method":
            method = str(value)
        elif parameter == "saturated":
            saturated_fraction = float(value)
        else:
            raise UnsupportedOptionError("sensorComputeArray", parameter)

    computed_sensors = [
        sensor_compute(sensor, oi, seed=None if seed is None else int(seed) + index)
        for index, sensor in enumerate(sensor_array)
    ]
    if not computed_sensors:
        raise ValueError("sensorComputeArray requires at least one sensor.")

    design_name = str(sensor_get(computed_sensors[0], "name"))
    if not param_format(design_name).startswith("ovt"):
        raise UnsupportedOptionError("sensorComputeArray", design_name)

    reference_shape = tuple(int(value) for value in sensor_get(computed_sensors[0], "size"))
    input_referred = np.zeros(reference_shape + (len(computed_sensors),), dtype=float)
    saturated_mask = np.zeros(reference_shape + (len(computed_sensors),), dtype=bool)
    voltage_swings = np.zeros(len(computed_sensors), dtype=float)

    for index, sensor in enumerate(computed_sensors):
        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        voltage_swings[index] = float(sensor_get(sensor, "pixel voltage swing"))
        saturated_mask[:, :, index] = volts >= (saturated_fraction * voltage_swings[index])
        if index == 0:
            input_referred[:, :, index] = volts
        elif index == 1:
            input_referred[:, :, index] = volts * float(sensor_get(sensor, "analog gain"))
        elif index == 2:
            input_referred[:, :, index] = volts / max(float(sensor.fields["pixel"]["fill_factor"]), 1e-12)
        else:
            raise UnsupportedOptionError("sensorComputeArray", "OVT arrays beyond 3 captures")

    combined = computed_sensors[0].clone()
    normalized_method = param_format(method)
    if normalized_method == "saturated":
        combined_volts = np.zeros(reference_shape, dtype=float)
        good_first = ~saturated_mask[:, :, 0]
        good_second = ~saturated_mask[:, :, 1]
        both_good = good_first & good_second
        combined_volts[both_good] = 0.5 * (input_referred[:, :, 0][both_good] + input_referred[:, :, 1][both_good])
        first_only = good_first & ~good_second
        combined_volts[first_only] = input_referred[:, :, 0][first_only]
        neither = ~good_first & ~good_second
        combined_volts[neither] = input_referred[:, :, 2][neither]

        target_swing = float(sensor_get(combined, "pixel voltage swing"))
        max_value = float(np.max(combined_volts))
        if max_value > 0.0:
            combined_volts = combined_volts * (target_swing / max_value)
        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["saturated"] = saturated_mask
    elif normalized_method == "bestsnr":
        electrons = []
        for sensor in computed_sensors:
            signal = np.asarray(sensor_get(sensor, "electrons"), dtype=float)
            well_capacity = float(sensor_get(sensor, "pixel well capacity"))
            electrons.append(np.where(signal < well_capacity, signal, 0.0))
        electron_stack = np.stack(electrons, axis=2)
        best_pixel = np.argmax(electron_stack, axis=2) + 1
        best_signal = np.max(electron_stack, axis=2)
        combined_volts = best_signal * float(sensor_get(combined, "pixel conversion gain"))
        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["bestPixel"] = best_pixel.astype(int)
    else:
        raise UnsupportedOptionError("sensorComputeArray", method)

    design_root = design_name.split("-", 1)[0]
    combined = sensor_set(combined, "name", f"{design_root}-{normalized_method}")
    return track_session_object(session, combined), computed_sensors


sensorCreateSplitPixel = sensor_create_split_pixel
sensorCreateArray = sensor_create_array
sensorComputeArray = sensor_compute_array


def _snr_voltage_levels(pixel: dict[str, Any], volts: Any | None) -> np.ndarray:
    if volts is None:
        return np.logspace(-4.0, 0.0, 50, dtype=float) * max(float(pixel["voltage_swing"]), 1e-12)
    return np.asarray(volts, dtype=float)


def _snr_db(signal_power: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
    safe_noise = np.maximum(np.asarray(noise_power, dtype=float), 1e-30)
    return 10.0 * np.log10(np.asarray(signal_power, dtype=float) / safe_noise)


def pixel_snr(sensor: Sensor, volts: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray]:
    """Compute MATLAB-style pixel SNR curves in dB over voltage levels."""

    pixel = sensor.fields["pixel"]
    voltage_levels = _snr_voltage_levels(pixel, volts)
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    read_sd = float(pixel["read_noise_v"]) / conversion_gain

    electron_mean = np.maximum(np.asarray(voltage_levels, dtype=float) / conversion_gain, 0.0)
    shot_sd = np.sqrt(electron_mean)
    signal_power = np.square(np.asarray(voltage_levels, dtype=float) / conversion_gain)
    snr = _snr_db(signal_power, np.square(read_sd) + np.square(shot_sd))
    snr_shot = _snr_db(signal_power, np.square(shot_sd))
    snr_read: float | np.ndarray
    if np.isclose(read_sd, 0.0):
        snr_read = float(np.inf)
    else:
        snr_read = _snr_db(signal_power, np.square(read_sd))
    return snr, np.asarray(voltage_levels, dtype=float), snr_shot, snr_read


def sensor_snr(
    sensor: Sensor,
    volts: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Compute MATLAB-style sensor SNR curves in dB over voltage levels."""

    pixel = sensor.fields["pixel"]
    voltage_levels = _snr_voltage_levels(pixel, volts)
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    read_sd = float(pixel["read_noise_v"]) / conversion_gain
    dsnu_sd = float(pixel["dsnu_sigma_v"]) / conversion_gain
    prnu_gain_sd = float(pixel["prnu_sigma"])

    electron_mean = np.maximum(np.asarray(voltage_levels, dtype=float) / conversion_gain, 0.0)
    shot_sd = np.sqrt(electron_mean)
    prnu_sd = prnu_gain_sd * (np.asarray(voltage_levels, dtype=float) / conversion_gain)
    signal_power = np.square(np.asarray(voltage_levels, dtype=float) / conversion_gain)
    noise_power = np.square(shot_sd) + np.square(read_sd) + np.square(dsnu_sd) + np.square(prnu_sd)

    snr = _snr_db(signal_power, noise_power)
    snr_shot = _snr_db(signal_power, np.square(shot_sd))
    if np.isclose(read_sd, 0.0):
        snr_read: float | np.ndarray = float(np.inf)
    else:
        snr_read = _snr_db(signal_power, np.square(read_sd))
    if np.isclose(dsnu_sd, 0.0):
        snr_dsnu: float | np.ndarray = float(np.inf)
    else:
        snr_dsnu = _snr_db(signal_power, np.square(dsnu_sd))
    if np.isclose(prnu_gain_sd, 0.0):
        snr_prnu: float | np.ndarray = float(np.inf)
    else:
        snr_prnu = _snr_db(signal_power, np.square(prnu_sd))

    return snr, np.asarray(voltage_levels, dtype=float), snr_shot, snr_read, snr_dsnu, snr_prnu


def _sensor_line_profile(sensor: Sensor, line_key: str, rc: Any) -> dict[str, list[np.ndarray]] | None:
    from .roi import _sensor_signal_cube

    normalized = param_format(line_key)
    if normalized not in {"hlinevolts", "hlineelectrons", "hlinedv", "vlinevolts", "vlineelectrons", "vlinedv"}:
        raise KeyError(f"Unsupported sensor line profile parameter: {line_key}")
    data_type = "electrons" if "electrons" in normalized else "dv" if "dv" in normalized else "volts"
    cube = _sensor_signal_cube(sensor, data_type)
    support = sensor_get(sensor, "spatial support")
    nfilters = int(sensor_get(sensor, "nfilters"))
    line_index = int(np.rint(float(rc)))
    if normalized.startswith("h"):
        if line_index < 1 or line_index > cube.shape[0]:
            raise IndexError("Horizontal sensor line index is out of range.")
        position = np.asarray(support["x"], dtype=float)
        oriented = cube[line_index - 1, :, :]
    else:
        if line_index < 1 or line_index > cube.shape[1]:
            raise IndexError("Vertical sensor line index is out of range.")
        position = np.asarray(support["y"], dtype=float)
        oriented = cube[:, line_index - 1, :]

    data: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for filter_index in range(nfilters):
        channel = np.asarray(oriented[:, filter_index], dtype=float)
        valid = ~np.isnan(channel)
        data.append(channel[valid].copy())
        positions.append(position[valid].copy())
    return {"data": data, "pos": positions, "pixPos": [values.copy() for values in positions]}


def _sensor_rect_for_chromaticity(rect_or_locs: Any) -> np.ndarray | None:
    if rect_or_locs is None:
        return None
    rect_array = np.asarray(rect_or_locs)
    if rect_array.size == 0:
        return None
    if rect_array.ndim == 1 and rect_array.size == 4:
        rect = np.rint(rect_array.astype(float)).astype(int)
    else:
        from .roi import ie_locs2_rect

        rect = ie_locs2_rect(rect_array)
    rect = rect.reshape(-1).astype(int)
    if rect.size != 4:
        raise ValueError("Chromaticity ROI must be [col, row, width, height] or Nx2 locations.")
    even = (rect % 2) == 0
    rect[even] -= 1
    rect[:2] = np.maximum(rect[:2], 1)
    rect[2:] = np.maximum(rect[2:], 0)
    return rect


def _sensor_crop_rect(data: np.ndarray | None, rect: np.ndarray | None) -> np.ndarray | None:
    if data is None or rect is None:
        return None if data is None else np.asarray(data).copy()
    array = np.asarray(data)
    row_start = max(int(rect[1]) - 1, 0)
    col_start = max(int(rect[0]) - 1, 0)
    row_end = min(int(rect[1] + rect[3]), array.shape[0])
    col_end = min(int(rect[0] + rect[2]), array.shape[1])
    return np.asarray(array[row_start:row_end, col_start:col_end, ...], dtype=float).copy()


def _sensor_cfa_crop_rect(sensor: Sensor, rect: Any) -> np.ndarray:
    rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)[:4]).astype(int)
    if rect_array.size != 4:
        raise ValueError("Crop rect must be [x, y, width, height].")

    cfa_rows, cfa_cols = _sensor_unit_block(sensor)
    cfa_xy = np.array([max(int(cfa_cols), 1), max(int(cfa_rows), 1)], dtype=int)
    adjusted = rect_array.copy()
    for axis in range(2):
        start_index = axis
        size_index = axis + 2
        remainder = adjusted[start_index] % cfa_xy[axis]
        if remainder != 1:
            if remainder == 0:
                adjusted[start_index] += 1
            else:
                adjusted[start_index] += cfa_xy[axis] - remainder + 1
        count = adjusted[size_index] + 1
        remainder = count % cfa_xy[axis]
        if remainder != 0:
            adjusted[size_index] += cfa_xy[axis] - remainder

    adjusted[:2] = np.maximum(adjusted[:2], 1)
    adjusted[2:] = np.maximum(adjusted[2:], 0)
    return adjusted


def sensor_crop(sensor: Sensor, rect: Any) -> Sensor:
    """Crop a sensor while preserving CFA alignment."""

    adjusted_rect = _sensor_cfa_crop_rect(sensor, rect)
    cropped = sensor.clone()

    volts = sensor.data.get("volts")
    dv = sensor.data.get("dv")
    new_size: tuple[int, int] | None = None
    if volts is not None:
        new_volts = _sensor_crop_rect(volts, adjusted_rect)
        if new_volts is not None:
            cropped.data["volts"] = new_volts
            new_size = tuple(int(value) for value in new_volts.shape[:2])
    if dv is not None:
        new_dv = _sensor_crop_rect(dv, adjusted_rect)
        if new_dv is not None:
            cropped.data["dv"] = np.rint(new_dv).astype(np.asarray(dv).dtype, copy=False)
            if new_size is None:
                new_size = tuple(int(value) for value in new_dv.shape[:2])
    if new_size is not None:
        cropped.fields["size"] = new_size
    cropped.metadata["crop"] = adjusted_rect.copy()
    return cropped


def _sensor_chromaticity(sensor: Sensor, rect_or_locs: Any = None, mode: str = "vec") -> np.ndarray | None:
    from .ip import _sensor_space

    volts = sensor.data.get("volts")
    if volts is None:
        return None

    rect = _sensor_rect_for_chromaticity(rect_or_locs)
    cropped = sensor.clone()
    cropped_volts = _sensor_crop_rect(volts, rect)
    if cropped_volts is None:
        return None
    cropped.data["volts"] = cropped_volts
    cropped_dv = _sensor_crop_rect(sensor.data.get("dv"), rect)
    if cropped_dv is not None:
        cropped.data["dv"] = cropped_dv
    elif "dv" in cropped.data:
        cropped.data.pop("dv", None)
    cropped.fields["size"] = tuple(int(value) for value in cropped_volts.shape[:2])

    sensor_space = np.asarray(_sensor_space(cropped), dtype=float)
    if sensor_space.ndim == 2:
        sensor_space = sensor_space[:, :, np.newaxis]
    nchannels = int(sensor_space.shape[2])
    output_shape = sensor_space.shape[:2] + (max(nchannels - 1, 0),)
    denominator = np.sum(sensor_space, axis=2, keepdims=True)
    chromaticity = np.divide(
        sensor_space[:, :, : max(nchannels - 1, 0)],
        denominator,
        out=np.full(output_shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )

    normalized_mode = param_format(mode)
    if normalized_mode == "matrix":
        return chromaticity
    return chromaticity.reshape(-1, output_shape[2])


def sensor_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
    object_type, object_param = ie_parameter_otype(parameter)
    if object_type == "pixel":
        try:
            if object_param is None:
                return sensor.fields["pixel"]
            return _sensor_pixel_get(sensor, object_param, *args)
        except KeyError:
            pass
    key = param_format(parameter)
    if key == "type":
        return sensor.type
    if key == "name":
        return sensor.name
    if key == "human":
        return _sensor_human(sensor)
    if key in {"humanconetype", "conetype"}:
        human = sensor.fields.get("human", {})
        value = human.get("coneType")
        return None if value is None else np.asarray(value).copy()
    if key in {"humanconedensities", "densities"}:
        human = sensor.fields.get("human", {})
        value = human.get("densities")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"humanconelocs", "conexy", "conelocs", "xy"}:
        human = sensor.fields.get("human", {})
        value = human.get("xy")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"humanrseed", "humanconeseed", "rseed"}:
        human = sensor.fields.get("human", {})
        return _copy_metadata_value(human.get("rSeed"))
    if key in {"sensormovement", "eyemovement"}:
        return _sensor_movement(sensor)
    if key in {"movementpositions", "sensorpositions"}:
        movement = sensor.fields.get("movement", {})
        value = movement.get("pos")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key == "sensorpositionsx":
        positions = sensor_get(sensor, "sensor positions")
        return None if positions is None else np.asarray(positions, dtype=float)[:, 0].copy()
    if key == "sensorpositionsy":
        positions = sensor_get(sensor, "sensor positions")
        return None if positions is None else np.asarray(positions, dtype=float)[:, 1].copy()
    if key in {"framesperposition", "framesperpositions", "exposuretimesperposition", "etimeperpos"}:
        movement = sensor.fields.get("movement", {})
        return _copy_metadata_value(movement.get("framesPerPosition"))
    if key in {"wave", "wavelength", "wavelengthsamples"}:
        return np.asarray(sensor.fields["wave"], dtype=float)
    if key in {"chartparameters"}:
        return _sensor_chart_parameters(sensor)
    if key in {"cornerpoints", "chartcornerpoints", "chartcorners"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("cornerPoints")
        return None if value is None else np.asarray(value).copy()
    if key == "mcccornerpoints":
        return sensor_get(sensor, "chart corner points")
    if key in {"chartrects", "chartrectangles"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("rects")
        return None if value is None else np.asarray(value).copy()
    if key in {"currentrect", "chartcurrentrect"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("currentRect")
        return None if value is None else np.asarray(value).copy()
    if key == "mccrecthandles":
        return _copy_metadata_value(sensor.fields.get("mccRectHandles"))
    if key in {"spectrum", "sensorspectrum"}:
        return _sensor_spectrum_struct(sensor)
    if key == "color":
        return _sensor_color_struct(sensor)
    if key in {"binwidth", "waveresolution", "wavelengthresolution"}:
        wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
        return float(wave[1] - wave[0]) if wave.size > 1 else 1.0
    if key in {"nwave", "nwaves", "numberofwavelengthsamples"}:
        return int(np.asarray(sensor.fields["wave"]).size)
    if key == "pattern":
        return np.asarray(sensor.fields["pattern"], dtype=int)
    if key in {"filterspectra", "colorfilters", "filtertransmissivities"}:
        return np.asarray(sensor.fields["filter_spectra"], dtype=float)
    if key in {"spectralqe", "sensorqe", "sensorspectralqe", "qe"}:
        return _sensor_combined_qe(sensor)
    if key in {"sensorspectralsr"}:
        return _sensor_combined_sr(sensor)
    if key in {"pixelspectralqe", "pdspectralqe", "pixelqe"}:
        return _sensor_pixel_qe(sensor)
    if key in {"spectralsr", "pdspectralsr", "pixelspectralsr", "sr"}:
        return _pixel_spectral_sr(sensor)
    if key in {"ir", "infraredfilter", "irfilter", "otherfilter"}:
        return _sensor_ir_filter(sensor)
    if key in {"consistency", "sensorconsistency"}:
        return bool(sensor.fields.get("consistency", False))
    if key in {"sensorcompute", "sensorcomputemethod"}:
        return _copy_metadata_value(sensor.fields.get("sensor_compute_method"))
    if key in {"filternames", "filtername"}:
        return list(sensor.fields["filter_names"])
    if key in {"nfilters", "nfilter", "ncolors", "ncolor", "nsensors", "nsensor"}:
        return int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
    if key == "size":
        return _sensor_rows_cols(sensor)
    if key in {"rows", "row"}:
        return _sensor_rows_cols(sensor)[0]
    if key in {"cols", "col"}:
        return _sensor_rows_cols(sensor)[1]
    if key in {"height", "arrayheight"}:
        height = float(sensor_get(sensor, "rows")) * float(sensor_get(sensor, "deltay"))
        return height * _spatial_unit_scale(args[0] if args else None)
    if key in {"width", "arraywidth"}:
        width = float(sensor_get(sensor, "cols")) * float(sensor_get(sensor, "deltax"))
        return width * _spatial_unit_scale(args[0] if args else None)
    if key == "dimension":
        dimension = np.array([sensor_get(sensor, "height"), sensor_get(sensor, "width")], dtype=float)
        return dimension * _spatial_unit_scale(args[0] if args else None)
    if key in {"pixelfields", "pixel"}:
        return sensor.fields["pixel"]
    if key in {"pixelsize", "pixelsizesamefillfactor"}:
        return np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    if key in {"wspatialresolution", "wres", "deltax", "widthspatialresolution"}:
        return float(_sensor_spatial_resolution_m(sensor)[1]) * _spatial_unit_scale(args[0] if args else None)
    if key in {"hspatialresolution", "hres", "deltay", "heightspatialresolultion"}:
        return float(_sensor_spatial_resolution_m(sensor)[0]) * _spatial_unit_scale(args[0] if args else None)
    if key in {"spatialsupport", "xyvaluesinmeters"}:
        rows, cols = _sensor_rows_cols(sensor)
        deltay = float(sensor_get(sensor, "deltay"))
        deltax = float(sensor_get(sensor, "deltax"))
        support = {
            "y": _sample_centers(rows, deltay),
            "x": _sample_centers(cols, deltax),
        }
        scale = _spatial_unit_scale(args[0] if args else None)
        if not np.isclose(scale, 1.0):
            support = {axis: values * scale for axis, values in support.items()}
        return support
    if key == "filtercolorletters":
        return _sensor_filter_color_letters(sensor)
    if key == "filtercolorletterscell":
        return list(_sensor_filter_color_letters(sensor))
    if key in {"filternamescellarray", "filtercolornamescellarray", "filternamescell"}:
        return [str(letter) for letter in _sensor_filter_color_letters(sensor)]
    if key in {"cfa", "colorfilterarray"}:
        return _sensor_cfa_struct(sensor)
    if key in {"cfapattern", "pattern"}:
        return _sensor_cfa_pattern(sensor).copy()
    if key == "cfaname":
        return _sensor_cfa_name(sensor)
    if key == "diffusionmtf":
        value = sensor.fields.get("diffusion_mtf")
        return None if value is None else copy.deepcopy(value)
    if key in {"filterplotcolor", "filterplotcolors"}:
        colors = "".join(letter if letter in "rgbcmyk" else "k" for letter in _sensor_filter_color_letters(sensor))
        if args:
            index = int(np.rint(float(args[0]))) - 1
            return colors[index]
        return colors
    if key == "unitblockrows":
        return _sensor_unit_block(sensor)[0]
    if key == "unitblockcols":
        return _sensor_unit_block(sensor)[1]
    if key in {"cfasize", "unitblocksize"}:
        return _sensor_unit_block(sensor)
    if key == "unitblockconfig":
        return _sensor_unit_block_config(sensor)
    if key in {"patterncolors", "pcolors", "blockcolors"}:
        letters = np.array(list(_sensor_filter_color_letters(sensor)), dtype="<U1")
        if letters.size == 0:
            return np.empty(np.asarray(sensor.fields["pattern"], dtype=int).shape, dtype="<U1")
        known = np.array(list("rgbcmykw"), dtype="<U1")
        unknown = ~np.isin(letters, known)
        letters[unknown] = "k"
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        return letters[np.clip(pattern - 1, 0, letters.size - 1)]
    if key in {
        "integrationtime",
        "integrationtimes",
        "exptime",
        "exptimes",
        "exposuretimes",
        "exposuretime",
        "expduration",
        "exposureduration",
        "exposuredurations",
    }:
        return _sensor_integration_time_value(sensor, args[0] if args else None)
    if key in {"uniqueintegrationtimes", "uniqueexptime", "uniqueexptimes"}:
        return np.unique(np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float).reshape(-1))
    if key in {"centralexposure", "geometricmeanexposuretime"}:
        exposure_times = np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float).reshape(-1)
        return float(np.prod(exposure_times) ** (1.0 / max(len(exposure_times), 1)))
    if key in {"exposuremethod", "expmethod"}:
        stored = sensor.fields.get("exposure_method")
        if stored is not None:
            return _copy_metadata_value(stored)
        exposure_times = np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float)
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        if exposure_times.size <= 1:
            return "singleExposure"
        if exposure_times.ndim == 1:
            return "bracketedExposure"
        if exposure_times.shape == pattern.shape:
            return "cfaExposure"
        return None
    if key == "nexposures":
        return int(np.asarray(sensor.fields.get("integration_time")).size)
    if key == "exposureplane":
        if "exposure_plane" in sensor.fields:
            return int(sensor.fields["exposure_plane"])
        return int(np.floor(sensor_get(sensor, "n exposures") / 2.0) + 1)
    if key in {"cds", "correlateddoublesampling"}:
        return bool(sensor.fields.get("cds", False))
    if key == "gamma":
        return float(_sensor_render_state(sensor)["gamma"])
    if key in {"maxbright", "scalemax", "scaleintensity"}:
        return bool(_sensor_render_state(sensor)["scale"])
    if key in {"autoexp", "autoexposure", "automaticexposure"}:
        return bool(sensor.fields["auto_exposure"])
    if key in {"analoggain", "ag"}:
        return float(sensor.fields["analog_gain"])
    if key in {"analogoffset", "ao"}:
        return float(sensor.fields["analog_offset"])
    if key in {"pixelpdarea", "pdarea"}:
        area = _pixel_pd_area_m2(sensor)
        if args:
            area *= _spatial_unit_scale(args[0]) ** 2
        return float(area)
    if key in {"noiseflag", "shotnoiseflag"}:
        return int(sensor.fields["noise_flag"])
    if key in {"dr", "drdb20", "dynamicrange", "sensordynamicrange"}:
        integration_time = args[0] if args else None
        return _sensor_dynamic_range(sensor, integration_time)
    if key in {"fpnparameters", "fpn", "fpnoffsetgain", "fpnoffsetandgain"}:
        return np.array([sensor_get(sensor, "dsnu sigma"), sensor_get(sensor, "prnu sigma")], dtype=float)
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "offsetnoisevalue", "dsnusigma", "sigmadsnu"}:
        return float(sensor.fields["pixel"]["dsnu_sigma_v"])
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "gainnoisevalue", "prnusigma", "sigmaprnu", "prnulevel"}:
        return float(sensor.fields["pixel"]["prnu_sigma"]) * 100.0
    if key in {"dsnuimage", "offsetfpnimage"}:
        stored = sensor.fields.get("offset_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"prnuimage", "gainfpnimage"}:
        stored = sensor.fields.get("gain_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"columnfpn", "columnfixedpatternnoise", "colfpn"}:
        return _sensor_column_fpn(sensor)
    if key in {"columndsnu", "columnfpnoffset", "colfpnoffset", "coldsnu"}:
        column_fpn = _sensor_column_fpn(sensor)
        return float(column_fpn[0]) if column_fpn.size >= 1 else 0.0
    if key in {"columnprnu", "columnfpngain", "colfpngain", "colprnu"}:
        column_fpn = _sensor_column_fpn(sensor)
        return float(column_fpn[1]) if column_fpn.size >= 2 else 0.0
    if key in {"coloffsetfpnvector", "coloffsetfpn", "coloffset"}:
        stored = sensor.fields.get("column_offset_fpn")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"colgainfpnvector", "colgainfpn", "colgain"}:
        stored = sensor.fields.get("column_gain_fpn")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key == "reusenoise":
        return bool(sensor.fields.get("reuse_noise", False))
    if key == "noiseseed":
        return _copy_metadata_value(sensor.fields.get("noise_seed", 0))
    if key == "responsedr":
        volts = sensor.data.get("volts")
        if volts is None:
            return None
        volts_array = np.asarray(volts, dtype=float)
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        v_max = float(np.max(volts_array))
        v_min = max(float(np.min(volts_array)), voltage_swing / float(2**12))
        return v_max / v_min
    if key in {"nbits", "bits"}:
        return int(sensor.fields["nbits"])
    if key in {"lut", "quantizationlut", "quantizatonlut"}:
        stored = sensor.fields.get("quantization_lut")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key == "quantizationstructure":
        return {
            "bits": int(sensor.fields["nbits"]),
            "method": _copy_metadata_value(sensor.fields["quantization"]),
            "lut": sensor_get(sensor, "lut"),
        }
    if key in {
        "vignetting",
        "vignettingflag",
        "pixelvignetting",
        "sensorvignetting",
        "bareetendue",
        "sensorbareetendue",
        "nomicrolensetendue",
    }:
        return sensor.fields.get("vignetting", 0)
    if key == "vignettingname":
        vignetting = sensor.fields.get("vignetting", 0)
        if isinstance(vignetting, str):
            normalized = param_format(vignetting)
            if normalized in {"", "skip"}:
                return "skip"
            if normalized in {"bare"}:
                return "bare"
            if normalized in {"centered"}:
                return "centered"
            if normalized in {"optimal"}:
                return "optimal"
            return str(vignetting)
        if vignetting in {0, None}:
            return "skip"
        if int(vignetting) == 1:
            return "bare"
        if int(vignetting) == 2:
            return "centered"
        if int(vignetting) == 3:
            return "optimal"
        return str(vignetting)
    if key in {"microlens", "ulens", "mlens", "ml"}:
        return _sensor_microlens(sensor)
    if key in {"microlensoffset", "mloffset", "microlensoffsetmicrons"}:
        return _copy_metadata_value(sensor.fields.get("mlOffset"))
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        stored = sensor.fields.get("etendue")
        if stored is None:
            return np.ones(sensor.fields["size"], dtype=float)
        return np.asarray(stored, dtype=float).copy()
    if key in {"ngridsamples", "pixelsamples", "nsamplesperpixel", "npixelsamplesforcomputing", "spatialsamplesperpixel"}:
        return int(sensor.fields.get("n_samples_per_pixel", 1))
    if key == "quantization":
        return sensor.fields["quantization"]
    if key in {"qmethod", "quantizationmethod"}:
        return sensor.fields["quantization"]
    if key == "responsetype":
        return str(sensor.fields.get("response_type", "linear"))
    if key in {"pixelvoltageswing", "voltageswing"}:
        return float(sensor.fields["pixel"]["voltage_swing"])
    if key in {"maxvoltage", "max", "maxoutput"}:
        return float(sensor_get(sensor, "pixel voltage swing"))
    if key in {"metadatascenename", "scenename", "scene_name"}:
        value = sensor.metadata.get("scenename", sensor.metadata.get("scene_name"))
        return _copy_metadata_value(value)
    if key in {"metadataopticsname", "metadatalensname", "metadatalens", "lens"}:
        value = sensor.metadata.get("opticsname", sensor.metadata.get("lens"))
        return _copy_metadata_value(value)
    if key == "metadatasensorname":
        return _copy_metadata_value(sensor.metadata.get("sensorname"))
    if key == "metadatacrop":
        return _copy_metadata_value(sensor.metadata.get("crop"))
    if key in {"blacklevel", "zerolevel", "zero"}:
        return float(sensor.fields.get("zero_level", 0.0))
    if key in {"maxdigital", "maxdigitalvalue"}:
        nbits = int(sensor_get(sensor, "nbits"))
        zero_level = float(sensor_get(sensor, "zero level"))
        return float((2**nbits) - zero_level)
    if key in {"roi", "roilocs"}:
        roi = sensor.fields.get("roi")
        if roi is None:
            return None
        roi_array = np.asarray(roi, dtype=int)
        if key == "roilocs" and roi_array.ndim == 1 and roi_array.size == 4:
            from .roi import ie_rect2_locs

            return ie_rect2_locs(roi_array)
        return roi_array.copy()
    if key == "roirect":
        roi = sensor.fields.get("roi")
        if roi is None:
            return None
        roi_array = np.asarray(roi, dtype=int)
        if roi_array.ndim == 1 and roi_array.size == 4:
            return roi_array.copy()
        from .roi import ie_locs2_rect

        return ie_locs2_rect(roi_array)
    if key in {"volts", "voltage"}:
        volts = sensor.data.get("volts")
        if args and volts is not None:
            return _sensor_color_data(sensor, volts, args[0])
        return volts
    if key == "voltimages":
        return _sensor_plane_images(sensor, sensor.data.get("volts"))
    if key in {"electrons", "electron"}:
        electrons = _sensor_electrons(sensor)
        if args and electrons is not None:
            return _sensor_color_data(sensor, electrons, args[0])
        return electrons
    if key == "electronsperarea":
        units = args[0] if args else "m"
        channel = args[1] if len(args) > 1 else None
        electrons = sensor_get(sensor, "electrons")
        if electrons is None:
            return None
        values = np.asarray(electrons, dtype=float) / max(_pixel_pd_area_m2(sensor), 1e-30)
        values = values / (_spatial_unit_scale(units) ** 2)
        if channel is not None:
            return _sensor_color_data(sensor, values, channel)
        return values
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        dv = sensor.data.get("dv")
        if args and dv is not None:
            return _sensor_color_data(sensor, dv, args[0])
        return dv
    if key in {"ncaptures", "ncapture"}:
        volts = sensor.data.get("volts")
        if volts is not None and np.asarray(volts).ndim >= 3:
            return int(np.asarray(volts).shape[2])
        dv = sensor.data.get("dv")
        if dv is not None and np.asarray(dv).ndim >= 3:
            return int(np.asarray(dv).shape[2])
        integration_time = np.asarray(sensor.fields.get("integration_time"))
        if integration_time.ndim == 1 and integration_time.size > 1:
            return int(integration_time.size)
        return 1
    if key in {"dvorvolts", "digitalorvolts"}:
        return sensor.data.get("dv", sensor.data.get("volts"))
    if key in {"hlinevolts", "hlineelectrons", "hlinedv", "vlinevolts", "vlineelectrons", "vlinedv"}:
        if not args:
            raise ValueError("Specify row or col.")
        return _sensor_line_profile(sensor, key, args[0])
    if key in {"roivolts", "roidata", "roidatav", "roidatavolts"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "volts")
    if key in {"roielectrons", "roidatae", "roidataelectrons"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "electrons")
    if key == "chromaticity":
        rect = args[0] if args else None
        mode = args[1] if len(args) >= 2 else "vec"
        return _sensor_chromaticity(sensor, rect, str(mode))
    if key == "roichromaticitymean":
        if not args:
            return None
        chromaticity = sensor_get(sensor, "chromaticity", args[0], "vec")
        if chromaticity is None:
            return None
        return np.nanmean(np.asarray(chromaticity, dtype=float), axis=0)
    if key in {"roidv", "roidigitalcount"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "dv")
    if key == "roivoltsmean":
        roi_data = sensor_get(sensor, "roi volts")
        if roi_data is None:
            return None
        return np.nanmean(np.asarray(roi_data, dtype=float), axis=0)
    if key == "roielectronsmean":
        roi_data = sensor_get(sensor, "roi electrons")
        if roi_data is None:
            return None
        return np.nanmean(np.asarray(roi_data, dtype=float), axis=0)
    if key in {"responseratio", "volts2maxratio"}:
        volts = sensor.data.get("volts")
        if volts is not None:
            voltage_swing = float(sensor.fields["pixel"]["voltage_swing"])
            return float(np.max(np.asarray(volts, dtype=float)) / max(voltage_swing, 1e-12))
        dv = sensor.data.get("dv")
        if dv is not None:
            nbits = int(sensor.fields["nbits"])
            max_digital = float(2**nbits)
            return float(np.max(np.asarray(dv, dtype=float)) / max(max_digital, 1e-12))
        return 0.0
    if key == "rgb":
        data_type = str(args[0]) if args else "volts"
        gamma = float(args[1]) if len(args) > 1 else None
        scale_max = bool(args[2]) if len(args) > 2 else None
        return _sensor_rgb_image(sensor, data_type, gamma, scale_max)
    if key in {"fovhorizontal", "fov"}:
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
        width = float(sensor_get(sensor, "width"))
        return float(np.rad2deg(2.0 * np.arctan2(width / 2.0, focal_length)))
    if key in {"fovvertical", "vfov"}:
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
        height = float(sensor_get(sensor, "height"))
        return float(np.rad2deg(2.0 * np.arctan2(height / 2.0, focal_length)))
    raise KeyError(f"Unsupported sensorGet parameter: {parameter}")


def sensor_set(sensor: Sensor, parameter: str, value: Any) -> Sensor:
    object_type, object_param = ie_parameter_otype(parameter)
    if object_type == "pixel":
        try:
            return _sensor_pixel_set(sensor, object_param or "pixel", value)
        except KeyError:
            pass
    key = param_format(parameter)
    if key == "name":
        sensor.name = str(value)
        return sensor
    if key == "human":
        sensor.fields["human"] = None if value is None else _human_struct_from_value(value)
        return sensor
    if key in {"humanconetype", "conetype"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["coneType"] = np.asarray(value).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanconedensities", "densities"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["densities"] = np.asarray(value, dtype=float).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanconelocs", "conexy", "conelocs", "xy"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["xy"] = np.asarray(value, dtype=float).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanrseed", "humanconeseed", "rseed"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["rSeed"] = _copy_metadata_value(value)
        sensor.fields["human"] = human
        return sensor
    if key in {"sensormovement", "eyemovement"}:
        sensor.fields["movement"] = _movement_struct_from_value(value)
        return sensor
    if key in {"movementpositions", "sensorpositions"}:
        movement = _sensor_movement(sensor)
        movement["pos"] = _movement_positions_from_value(value)
        sensor.fields["movement"] = movement
        return sensor
    if key in {"framesperposition", "framesperpositions", "exposuretimesperposition", "etimeperpos"}:
        movement = _sensor_movement(sensor)
        movement["framesPerPosition"] = _copy_metadata_value(value)
        sensor.fields["movement"] = movement
        return sensor
    if key in {"spectrum", "sensorspectrum"}:
        spectrum = _spectrum_struct_from_value(value)
        sensor = _sensor_update_wave(sensor, np.asarray(spectrum["wave"], dtype=float).reshape(-1))
        sensor.fields["spectrum"] = spectrum
        sensor.fields["spectrum"]["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
        return sensor
    if key == "color":
        color = _sensor_color_struct_from_value(value)
        if "filterSpectra" in color:
            sensor.fields["filter_spectra"] = np.asarray(color["filterSpectra"], dtype=float)
        if "filterNames" in color:
            sensor.fields["filter_names"] = list(color["filterNames"])
        if "irFilter" in color:
            ir_filter = np.asarray(color["irFilter"], dtype=float).reshape(-1)
            if ir_filter.size == 1:
                sensor.fields["ir_filter"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=float)
            elif ir_filter.size == np.asarray(sensor.fields["wave"], dtype=float).size:
                sensor.fields["ir_filter"] = ir_filter
            else:
                raise ValueError("IR filter must match the sensor wavelength sampling.")
        return sensor
    if key in {"chartparameters"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"].update(dict(value))
        return sensor
    if key in {"chartcornerpoints", "cornerpoints", "chartcorners"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["cornerPoints"] = np.asarray(value).copy()
        return sensor
    if key == "mcccornerpoints":
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["cornerPoints"] = np.asarray(value).copy()
        return sensor
    if key in {"chartrects", "chartrectangles"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["rects"] = np.asarray(value).copy()
        return sensor
    if key in {"chartcurrentrect", "currentrect"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["currentRect"] = np.asarray(value).copy()
        return sensor
    if key == "mccrecthandles":
        sensor.fields["mccRectHandles"] = _copy_metadata_value(value)
        return sensor
    if key in {"wave", "wavelength", "wavelengthsamples"}:
        sensor = _sensor_update_wave(sensor, np.asarray(value, dtype=float).reshape(-1))
        return sensor
    if key == "size":
        sensor = sensor_set(sensor, "rows", value[0])
        sensor = sensor_set(sensor, "cols", value[1])
        return sensor
    if key == "matchoi":
        if not isinstance(value, OpticalImage):
            raise ValueError("match oi requires an optical image.")
        pixel_size = float(oi_get(value, "width spatial resolution", "m"))
        sensor = sensor_set(sensor, "pixel size same fill factor", pixel_size)
        oi_size = np.asarray(oi_get(value, "size"), dtype=int).reshape(-1)
        if oi_size.size != 2:
            raise ValueError("match oi requires an optical image with a 2D size.")
        return sensor_set(sensor, "size", oi_size)
    if key in {"rows", "row"}:
        block_rows, _ = _sensor_unit_block(sensor)
        rows = _sensor_aligned_dimension(value, block_rows)
        sensor.fields["size"] = (rows, int(sensor.fields["size"][1]))
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"cols", "col"}:
        _, block_cols = _sensor_unit_block(sensor)
        cols = _sensor_aligned_dimension(value, block_cols)
        sensor.fields["size"] = (int(sensor.fields["size"][0]), cols)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key == "pattern":
        sensor.fields["pattern"] = np.asarray(value, dtype=int)
        return sensor
    if key in {"patternandsize", "patternsize", "cfapatternandsize"}:
        pattern = np.asarray(value, dtype=int)
        sensor.fields["pattern"] = pattern
        rows, cols = _sensor_rows_cols(sensor)
        block_rows, block_cols = pattern.shape
        new_rows = ensure_multiple(rows, block_rows)
        new_cols = ensure_multiple(cols, block_cols)
        if new_rows != rows or new_cols != cols:
            sensor.fields["size"] = (new_rows, new_cols)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"colorfilterarray", "cfa"}:
        sensor.fields["pattern"] = _cfa_pattern_from_value(value)
        return sensor
    if key in {"filterspectra", "colorfilters", "filtertransmissivities"}:
        sensor.fields["filter_spectra"] = np.asarray(value, dtype=float)
        return sensor
    if key in {"pixelspectralqe", "pdspectralqe", "pixelqe"}:
        current = _sensor_pixel_qe(sensor)
        qe = np.asarray(value, dtype=float).reshape(-1)
        if qe.size == 1:
            sensor.fields["pixel_qe"] = current * float(qe[0])
            return sensor
        if qe.size != np.asarray(sensor.fields["wave"], dtype=float).size:
            raise ValueError("pixel spectral QE must match the sensor wavelength sampling.")
        sensor.fields["pixel_qe"] = qe
        return sensor
    if key in {"ir", "infraredfilter", "irfilter", "otherfilter"}:
        ir_filter = np.asarray(value, dtype=float).reshape(-1)
        if ir_filter.size == 1:
            sensor.fields["ir_filter"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=float)
            return sensor
        if ir_filter.size != np.asarray(sensor.fields["wave"], dtype=float).size:
            raise ValueError("IR filter must match the sensor wavelength sampling.")
        sensor.fields["ir_filter"] = ir_filter
        return sensor
    if key in {"filternames", "filtername"}:
        sensor.fields["filter_names"] = list(value)
        return sensor
    if key in {"consistency", "sensorconsistency"}:
        sensor.fields["consistency"] = bool(value)
        return sensor
    if key in {"sensorcompute", "sensorcomputemethod"}:
        sensor.fields["sensor_compute_method"] = _copy_metadata_value(value)
        return sensor
    if key in {"roi", "roilocs"}:
        roi = np.asarray(value, dtype=float)
        if roi.ndim == 1 and roi.size == 4:
            sensor.fields["roi"] = np.rint(roi).astype(int)
            return sensor
        if roi.ndim == 2 and roi.shape[1] == 2:
            sensor.fields["roi"] = np.rint(roi).astype(int)
            return sensor
        raise ValueError("sensor ROI must be a rect or an Nx2 location array.")
    if key == "roirect":
        rect = np.rint(np.asarray(value, dtype=float).reshape(-1)).astype(int)
        if rect.size != 4:
            raise ValueError("sensor ROI rect must contain [col, row, width, height].")
        sensor.fields["roi"] = rect
        return sensor
    if key in {"pixelsizesamefillfactor", "pixelsize"}:
        size_value = np.asarray(value, dtype=float)
        if size_value.size == 1:
            size_value = np.repeat(size_value, 2)
        sensor.fields["pixel"]["size_m"] = size_value
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {
        "integrationtime",
        "integrationtimes",
        "exptime",
        "exptimes",
        "exposuretimes",
        "exposuretime",
        "expduration",
        "exposureduration",
        "exposuredurations",
    }:
        integration_time = np.asarray(value, dtype=float)
        if integration_time.ndim == 0:
            sensor.fields["integration_time"] = float(integration_time)
        else:
            sensor.fields["integration_time"] = integration_time.copy()
        sensor.fields["auto_exposure"] = False
        return sensor
    if key == "exposureplane":
        sensor.fields["exposure_plane"] = int(np.rint(float(value)))
        return sensor
    if key in {"exposuremethod", "expmethod"}:
        sensor.fields["exposure_method"] = _copy_metadata_value(value)
        return sensor
    if key in {"cds", "correlateddoublesampling"}:
        sensor.fields["cds"] = bool(value)
        return sensor
    if key == "diffusionmtf":
        sensor.fields["diffusion_mtf"] = None if value is None else copy.deepcopy(value)
        return sensor
    if key in {"blacklevel", "zerolevel", "zero"}:
        sensor.fields["zero_level"] = float(value)
        return sensor
    if key in {"metadatascenename", "scenename", "scene_name"}:
        copied = _copy_metadata_value(value)
        sensor.metadata["scenename"] = copied
        sensor.metadata["scene_name"] = _copy_metadata_value(copied)
        return sensor
    if key in {"metadataopticsname", "metadatalensname", "metadatalens", "lens"}:
        copied = _copy_metadata_value(value)
        sensor.metadata["opticsname"] = copied
        sensor.metadata["lens"] = _copy_metadata_value(copied)
        return sensor
    if key == "metadatasensorname":
        sensor.metadata["sensorname"] = _copy_metadata_value(value)
        return sensor
    if key == "metadatacrop":
        sensor.metadata["crop"] = _copy_metadata_value(value)
        return sensor
    if key == "gamma":
        _sensor_render_state(sensor)["gamma"] = float(value)
        return sensor
    if key in {"maxbright", "scalemax", "scaleintensity"}:
        _sensor_render_state(sensor)["scale"] = bool(value)
        return sensor
    if key in {"autoexp", "autoexposure", "automaticexposure"}:
        if isinstance(value, str):
            enabled = param_format(value) == "on"
        else:
            enabled = bool(value)
        sensor.fields["auto_exposure"] = enabled
        if enabled:
            integration_time = np.asarray(sensor.fields.get("integration_time", 0.0))
            pattern = np.asarray(sensor.fields["pattern"], dtype=int)
            if integration_time.ndim > 0 and integration_time.shape == pattern.shape:
                sensor.fields["integration_time"] = np.zeros(pattern.shape, dtype=float)
            else:
                sensor.fields["integration_time"] = 0.0
        return sensor
    if key in {"analoggain", "ag"}:
        sensor.fields["analog_gain"] = float(value)
        return sensor
    if key in {"analogoffset", "ao"}:
        sensor.fields["analog_offset"] = float(value)
        return sensor
    if key == "noiseflag":
        sensor.fields["noise_flag"] = int(value)
        return sensor
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "offsetnoisevalue", "dsnusigma", "sigmadsnu"}:
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(value)
        sensor.fields.pop("offset_fpn_image", None)
        return sensor
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "gainnoisevalue", "prnusigma", "sigmaprnu", "prnulevel"}:
        sensor.fields["pixel"]["prnu_sigma"] = float(value) / 100.0
        sensor.fields.pop("gain_fpn_image", None)
        return sensor
    if key in {"dsnuimage", "offsetfpnimage"}:
        if value is None:
            sensor.fields.pop("offset_fpn_image", None)
            return sensor
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("DSNU image must match the sensor size.")
        sensor.fields["offset_fpn_image"] = image
        return sensor
    if key in {"prnuimage", "gainfpnimage"}:
        if value is None:
            sensor.fields.pop("gain_fpn_image", None)
            return sensor
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("PRNU image must match the sensor size.")
        sensor.fields["gain_fpn_image"] = image
        return sensor
    if key in {"columnfpnparameters", "columnfpn", "columnfixedpatternnoise", "colfpn"}:
        column_fpn = np.asarray(value, dtype=float).reshape(-1)
        if column_fpn.size == 0:
            sensor.fields.pop("column_fpn", None)
            return sensor
        if column_fpn.size != 2:
            raise ValueError("Column FPN must be in [offset, gain] format.")
        sensor.fields["column_fpn"] = column_fpn.copy()
        return sensor
    if key in {"colgainfpnvector", "columnprnu"}:
        column_gain = np.asarray(value, dtype=float).reshape(-1)
        if column_gain.size == 0:
            sensor.fields.pop("column_gain_fpn", None)
            return sensor
        if column_gain.size != int(sensor_get(sensor, "cols")):
            raise ValueError("Bad column gain data.")
        sensor.fields["column_gain_fpn"] = column_gain.copy()
        return sensor
    if key in {"coloffsetfpnvector", "columndsnu"}:
        column_offset = np.asarray(value, dtype=float).reshape(-1)
        if column_offset.size == 0:
            sensor.fields.pop("column_offset_fpn", None)
            return sensor
        if column_offset.size != int(sensor_get(sensor, "cols")):
            raise ValueError("Bad column offset data.")
        sensor.fields["column_offset_fpn"] = column_offset.copy()
        return sensor
    if key == "reusenoise":
        sensor.fields["reuse_noise"] = bool(value)
        return sensor
    if key == "noiseseed":
        sensor.fields["noise_seed"] = _copy_metadata_value(value)
        return sensor
    if key in {
        "vignetting",
        "vignettingflag",
        "pixelvignetting",
        "sensorvignetting",
        "bareetendue",
        "sensorbareetendue",
        "nomicrolensetendue",
    }:
        sensor.fields["vignetting"] = value
        sensor.fields["etendue"] = None
        return sensor
    if key in {"microlens", "ulens", "mlens", "ml"}:
        sensor.fields["ml"] = None if value is None else _microlens_struct_from_value(value)
        return sensor
    if key in {"microlensoffset", "mloffset", "microlensoffsetmicrons"}:
        sensor.fields["mlOffset"] = _copy_metadata_value(value)
        return sensor
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        etendue = np.asarray(value, dtype=float)
        if etendue.shape != tuple(sensor.fields["size"]):
            raise ValueError("sensor etendue must match the sensor size.")
        sensor.fields["etendue"] = etendue
        return sensor
    if key in {"ngridsamples", "pixelsamples", "nsamplesperpixel", "npixelsamplesforcomputing", "spatialsamplesperpixel"}:
        sensor.fields["n_samples_per_pixel"] = int(value)
        return sensor
    if key in {"quantization", "qmethod", "quantizationmethod"}:
        sensor.fields["quantization"] = str(value)
        return sensor
    if key in {"nbits", "bits"}:
        sensor.fields["nbits"] = int(value)
        return sensor
    if key in {"lut", "quantizationlut", "quantizatonlut"}:
        sensor.fields["quantization_lut"] = None if value is None else np.asarray(value, dtype=float).copy()
        return sensor
    if key == "quantizationstructure":
        payload = dict(value) if isinstance(value, dict) else dict(vars(value))
        if "bits" in payload:
            sensor.fields["nbits"] = int(payload["bits"])
        if "method" in payload:
            sensor.fields["quantization"] = str(payload["method"])
        if "lut" in payload:
            sensor.fields["quantization_lut"] = None if payload["lut"] is None else np.asarray(payload["lut"], dtype=float).copy()
        return sensor
    if key == "responsetype":
        normalized = str(param_format(value))
        if normalized not in {"linear", "log"}:
            raise ValueError("response type must be 'linear' or 'log'.")
        sensor.fields["response_type"] = normalized
        return sensor
    if key in {"volts", "voltage"}:
        volts = np.asarray(value, dtype=float)
        sensor.data["volts"] = volts
        if param_format(sensor.fields.get("quantization", "analog")) == "analog":
            sensor.data.pop("dv", None)
        if volts.ndim >= 2:
            sensor.fields["size"] = (int(volts.shape[0]), int(volts.shape[1]))
        return sensor
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        dv = np.asarray(value, dtype=float)
        sensor.data["dv"] = dv
        if dv.ndim >= 2:
            sensor.fields["size"] = (int(dv.shape[0]), int(dv.shape[1]))
        return sensor
    raise KeyError(f"Unsupported sensorSet parameter: {parameter}")


def sensor_set_size_to_fov(sensor: Sensor, fov: float | tuple[float, float], oi: OpticalImage) -> Sensor:
    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    pattern_rows, pattern_cols = pattern.shape
    focal_length = float(oi_get(oi, "focal length"))
    if isinstance(fov, (tuple, list, np.ndarray)):
        hfov = float(fov[0])
        vfov = float(fov[1] if len(fov) > 1 else fov[0])
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        height = 2.0 * focal_length * np.tan(np.deg2rad(vfov) / 2.0)
        cols = max(pattern_cols, int(round(width / float(sensor_get(sensor, "deltax")))))
        rows = max(pattern_rows, int(round(height / float(sensor_get(sensor, "deltay")))))
    else:
        hfov = float(fov)
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        current_width = float(sensor_get(sensor, "width"))
        scale = width / max(current_width, 1e-12)
        rows = max(pattern_rows, int(round(sensor_get(sensor, "rows") * scale)))
        cols = max(pattern_cols, int(round(sensor_get(sensor, "cols") * scale)))
    cols = ensure_multiple(cols, pattern_cols)
    rows = ensure_multiple(rows, pattern_rows)
    sensor = sensor_set(sensor, "rows", rows)
    sensor = sensor_set(sensor, "cols", cols)
    return sensor


def _scene_distance_m(scene_or_distance: Scene | OpticalImage | float | None) -> float:
    if scene_or_distance is None:
        return np.inf
    if isinstance(scene_or_distance, OpticalImage):
        return np.inf
    if isinstance(scene_or_distance, Scene):
        return float(scene_or_distance.fields.get("distance_m", np.inf))
    return float(scene_or_distance)


def _sensor_image_distance_m(
    scene_or_distance: Scene | OpticalImage | float | None,
    oi: OpticalImage | None,
) -> float:
    if oi is None:
        return DEFAULT_FOCAL_LENGTH_M

    optics = oi.fields.get("optics", {})
    focal_length = float(optics.get("focal_length_m", oi_get(oi, "focal length")))
    if param_format(optics.get("model", "")) == "skip":
        return focal_length

    scene_distance = _scene_distance_m(scene_or_distance)
    if not np.isfinite(scene_distance) or scene_distance <= focal_length:
        return focal_length
    return 1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12)


def _shot_noise_electrons(rng: np.random.Generator, electrons: np.ndarray) -> np.ndarray:
    electrons = np.asarray(electrons, dtype=float)
    clipped = np.clip(electrons, 0.0, None)
    noisy = clipped + (np.sqrt(clipped) * rng.standard_normal(clipped.shape))
    low_count = clipped < 25.0
    if np.any(low_count):
        noisy[low_count] = rng.poisson(clipped[low_count])
    return np.rint(noisy)


def _pixel_plane(volume: np.ndarray, values: np.ndarray) -> np.ndarray:
    plane = np.asarray(values, dtype=float)
    if np.asarray(volume).ndim == 2:
        return plane
    return plane[:, :, np.newaxis]


def _apply_read_noise(rng: np.random.Generator, volts: np.ndarray, sigma_v: float) -> np.ndarray:
    if sigma_v <= 0.0:
        return volts
    return volts + _pixel_plane(volts, rng.normal(0.0, sigma_v, size=volts.shape[:2]))


def _apply_fixed_pattern_noise(
    rng: np.random.Generator,
    volts: np.ndarray,
    *,
    dsnu_sigma_v: float,
    prnu_sigma: float,
    integration_time: float | np.ndarray,
    auto_exposure: bool,
) -> np.ndarray:
    dsnu = _pixel_plane(volts, rng.normal(0.0, dsnu_sigma_v, size=volts.shape[:2]))
    integration_time_array = np.asarray(integration_time, dtype=float)
    if np.all(np.isclose(integration_time_array, 0.0)) and not auto_exposure:
        return dsnu
    prnu = _pixel_plane(volts, 1.0 + rng.normal(0.0, prnu_sigma, size=volts.shape[:2]))
    return (volts * prnu) + dsnu


def _sample_centers(count: int, spacing_m: float) -> np.ndarray:
    return ((np.arange(count, dtype=float) + 0.5) - (count / 2.0)) * float(spacing_m)


def _sample2space(samples: np.ndarray, spacing_m: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    return (samples - np.mean(samples)) * float(spacing_m)


def _gaussian_kernel(shape: tuple[int, int], sigma: float) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 1 and cols <= 1:
        return np.ones((1, 1), dtype=float)
    y = np.arange(rows, dtype=float) - ((rows - 1.0) / 2.0)
    x = np.arange(cols, dtype=float) - ((cols - 1.0) / 2.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / max(2.0 * sigma * sigma, 1e-12))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.ones((1, 1), dtype=float)
    return kernel / kernel_sum


def _sensor_etendue(sensor: Sensor) -> np.ndarray:
    stored = sensor.fields.get("etendue")
    if stored is not None:
        etendue = np.asarray(stored, dtype=float)
        if etendue.shape == tuple(sensor.fields["size"]):
            return etendue

    vignetting = sensor.fields.get("vignetting", 0)
    normalized = param_format(vignetting) if isinstance(vignetting, str) else vignetting
    if normalized in {0, "skip", "", None}:
        etendue = np.ones(sensor.fields["size"], dtype=float)
        sensor.fields["etendue"] = etendue
        return etendue

    raise UnsupportedOptionError("sensorCompute", f"vignetting {vignetting}")


def _pixel_pd_size_m(pixel: dict[str, Any]) -> np.ndarray:
    return _pixel_pd_size_from_pixel(pixel)


def _sensor_pd_array(sensor: Sensor, spacing: float) -> np.ndarray:
    if spacing <= 0.0 or spacing > 1.0:
        raise ValueError("spacing must be within (0, 1].")
    pixel = sensor.fields["pixel"]
    pixel_size = np.asarray(pixel["size_m"], dtype=float)
    pd_size = _pixel_pd_size_m(pixel)
    pd_position = _pixel_pd_position_from_pixel(pixel)

    normalized_pd_min = pd_position / (spacing * pixel_size)
    normalized_pd_max = (pd_size + pd_position) / (spacing * pixel_size)
    grid_positions = np.arange(0.0, 1.0 + spacing, spacing) / spacing
    n_squares = max(len(grid_positions) - 1, 1)
    in_pd_rows = np.zeros(n_squares, dtype=float)
    in_pd_cols = np.zeros(n_squares, dtype=float)
    for index in range(n_squares):
        lower = max(grid_positions[index], normalized_pd_min[0])
        upper = min(grid_positions[index + 1], normalized_pd_max[0])
        in_pd_rows[index] = max(0.0, upper - lower)

        lower = max(grid_positions[index], normalized_pd_min[1])
        upper = min(grid_positions[index + 1], normalized_pd_max[1])
        in_pd_cols[index] = max(0.0, upper - lower)
    return np.outer(in_pd_rows, in_pd_cols)


def _interpolated_cfa(sensor: Sensor, spacing: float, row_count: int, col_count: int) -> np.ndarray:
    pattern = tile_pattern(np.asarray(sensor.fields["pattern"], dtype=int), sensor.fields["size"][0], sensor.fields["size"][1])
    if np.isclose(spacing, 1.0):
        return pattern
    row_coords = np.floor(spacing * np.arange(row_count, dtype=float)).astype(int)
    col_coords = np.floor(spacing * np.arange(col_count, dtype=float)).astype(int)
    row_coords = np.clip(row_coords, 0, pattern.shape[0] - 1)
    col_coords = np.clip(col_coords, 0, pattern.shape[1] - 1)
    return pattern[row_coords[:, None], col_coords[None, :]]


def _interp2_linear_constant_zero(
    plane: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
) -> np.ndarray:
    if source_cols.size <= 1:
        col_coords = np.zeros_like(target_cols, dtype=float)
    else:
        col_coords = (np.asarray(target_cols, dtype=float) - float(source_cols[0])) / float(source_cols[1] - source_cols[0])
    if source_rows.size <= 1:
        row_coords = np.zeros_like(target_rows, dtype=float)
    else:
        row_coords = (np.asarray(target_rows, dtype=float) - float(source_rows[0])) / float(source_rows[1] - source_rows[0])
    # Match MATLAB's endpoint-inclusive behavior for coordinates that only miss
    # the valid range by floating-point roundoff.
    eps = 1e-9
    max_row = float(np.asarray(plane).shape[0] - 1)
    max_col = float(np.asarray(plane).shape[1] - 1)
    row_coords = np.where(np.abs(row_coords) <= eps, 0.0, row_coords)
    col_coords = np.where(np.abs(col_coords) <= eps, 0.0, col_coords)
    row_coords = np.where(np.abs(row_coords - max_row) <= eps, max_row, row_coords)
    col_coords = np.where(np.abs(col_coords - max_col) <= eps, max_col, col_coords)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    return map_coordinates(
        np.asarray(plane, dtype=float),
        [row_grid, col_grid],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _sensor_qe_on_wave(sensor: Sensor, target_wave: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    target_wave = np.asarray(target_wave, dtype=float).reshape(-1)
    sensor_wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
    spectral_qe = _sensor_combined_qe(sensor, dtype=dtype)
    if spectral_qe.ndim == 1:
        spectral_qe = spectral_qe.reshape(-1, 1)
    if np.array_equal(target_wave, sensor_wave):
        return spectral_qe
    if sensor_wave.size <= 1:
        raise ValueError("Sensor and optical image wavelength samplings do not match.")
    interpolated = np.empty((target_wave.size, spectral_qe.shape[1]), dtype=spectral_qe.dtype)
    for index in range(spectral_qe.shape[1]):
        interpolated[:, index] = np.interp(target_wave, sensor_wave, spectral_qe[:, index], left=0.0, right=0.0)
    return interpolated


def _signal_current_density(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    irradiance = np.asarray(oi.data["photons"], dtype=np.float32)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    spectral_qe = _sensor_qe_on_wave(sensor, wave, dtype=np.float32)
    bin_width = np.float32(np.mean(np.diff(wave)) if wave.size > 1 else 1.0)
    weighted_qe = spectral_qe * bin_width
    return np.tensordot(irradiance, weighted_qe, axes=([2], [0])).astype(np.float32) * np.float32(_ELEMENTARY_CHARGE_C)


def _spatial_integrate_current_density(scdi: np.ndarray, oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    n_samples_per_pixel = int(sensor.fields.get("n_samples_per_pixel", 1))
    if n_samples_per_pixel <= 0:
        raise ValueError("n_samples_per_pixel must be positive.")
    spacing = 1.0 / float(n_samples_per_pixel)
    if n_samples_per_pixel % 2 == 0:
        raise NotImplementedError("sensorCompute only supports odd nSamplesPerPixel values.")

    oi_rows, oi_cols = scdi.shape[:2]
    sensor_rows, sensor_cols = sensor.fields["size"]
    oi_height_spacing = float(oi_get(oi, "hspatialresolution"))
    oi_width_spacing = float(oi_get(oi, "wspatialresolution"))
    sensor_height_spacing = float(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)[0])
    sensor_width_spacing = float(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)[1])

    source_rows = _sample2space(np.arange(oi_rows, dtype=float), oi_height_spacing)
    source_cols = _sample2space(np.arange(oi_cols, dtype=float), oi_width_spacing)
    target_row_samples = np.arange(0.0, sensor_rows, spacing, dtype=float) + (spacing / 2.0)
    target_col_samples = np.arange(0.0, sensor_cols, spacing, dtype=float) + (spacing / 2.0)
    target_rows = _sample2space(target_row_samples, sensor_height_spacing)
    target_cols = _sample2space(target_col_samples, sensor_width_spacing)

    interpolated_cfa = _interpolated_cfa(sensor, spacing, target_rows.size, target_cols.size)

    height_samples_per_pixel = max(1, int(np.ceil(sensor_height_spacing / max(oi_height_spacing, 1e-12))))
    width_samples_per_pixel = max(1, int(np.ceil(sensor_width_spacing / max(oi_width_spacing, 1e-12))))
    kernel = _gaussian_kernel((height_samples_per_pixel, width_samples_per_pixel), height_samples_per_pixel / 4.0)

    flat_scdi = np.zeros((target_rows.size, target_cols.size), dtype=float)
    for channel_index in range(scdi.shape[2]):
        plane = convolve2d(np.asarray(scdi[:, :, channel_index], dtype=float), kernel, mode="same")
        sampled = _interp2_linear_constant_zero(plane, source_rows, source_cols, target_rows, target_cols)
        mask = interpolated_cfa == (channel_index + 1)
        flat_scdi = flat_scdi + (mask * sampled)

    pixel_area = float(np.prod(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)))
    if n_samples_per_pixel == 1:
        return flat_scdi * float(sensor.fields["pixel"]["fill_factor"]) * pixel_area

    pd_array = _sensor_pd_array(sensor, spacing)
    photo_detector_array = np.tile(pd_array, sensor.fields["size"])
    signal_current_large = flat_scdi * photo_detector_array
    filt = pixel_area * (np.ones((n_samples_per_pixel, n_samples_per_pixel), dtype=float) / float(n_samples_per_pixel**2))
    blurred = convolve2d(signal_current_large, filt, mode="same")
    start = n_samples_per_pixel // 2
    return blurred[start::n_samples_per_pixel, start::n_samples_per_pixel]


def signal_current(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    """Compute signal current in Amps/pixel using the MATLAB signalCurrent path."""

    working = sensor.clone()
    if working.fields["mosaic"]:
        current_density = _signal_current_density(oi, working)
        return np.asarray(_spatial_integrate_current_density(current_density, oi, working), dtype=float)

    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel = working.fields["pixel"]
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * float(pixel["fill_factor"]))
    filter_spectra = _sensor_qe_on_wave(working, wave)
    electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
    electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, working) * pixel_area
    return np.asarray(electron_rate * _ELEMENTARY_CHARGE_C, dtype=float)


def _regrid_electron_rate_density(
    density_cube: np.ndarray,
    oi: OpticalImage,
    sensor: Sensor,
) -> np.ndarray:
    oi_rows, oi_cols = density_cube.shape[:2]
    sensor_rows, sensor_cols = sensor.fields["size"]
    if oi_rows == 1 and oi_cols == 1:
        return np.broadcast_to(
            np.asarray(density_cube[0, 0, :], dtype=float),
            (sensor_rows, sensor_cols, density_cube.shape[2]),
        ).copy()
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    oi_spacing = float(oi.fields.get("sample_spacing_m") or (oi.fields["width_m"] / max(oi_cols, 1)))

    oi_y = _sample_centers(oi_rows, oi_spacing)
    oi_x = _sample_centers(oi_cols, oi_spacing)
    sensor_y = _sample_centers(sensor_rows, pixel_size[0])
    sensor_x = _sample_centers(sensor_cols, pixel_size[1])
    sensor_yy, sensor_xx = np.meshgrid(sensor_y, sensor_x, indexing="ij")

    row_samples_per_pixel = max(1.0, float(np.ceil(pixel_size[0] / max(oi_spacing, 1e-12))))
    col_samples_per_pixel = max(1.0, float(np.ceil(pixel_size[1] / max(oi_spacing, 1e-12))))
    kernel_shape = (int(row_samples_per_pixel), int(col_samples_per_pixel))
    kernel_sigma = row_samples_per_pixel / 4.0

    regridded = np.empty((sensor_rows, sensor_cols, density_cube.shape[2]), dtype=float)
    for channel_index in range(density_cube.shape[2]):
        plane = density_cube[:, :, channel_index]
        if row_samples_per_pixel > 1.0 or col_samples_per_pixel > 1.0:
            plane = convolve2d(plane, _gaussian_kernel(kernel_shape, kernel_sigma), mode="same")
        interpolator = RegularGridInterpolator((oi_y, oi_x), plane, bounds_error=False, fill_value=0.0)
        regridded[:, :, channel_index] = interpolator(np.stack([sensor_yy, sensor_xx], axis=-1))
    return regridded


def _auto_exposure_default(sensor: Sensor, oi: OpticalImage) -> float:
    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    voltage_swing = float(sensor.fields["pixel"]["voltage_swing"])

    illuminance = luminance_from_photons(cube, wave, asset_store=AssetStore.default())
    bright_row, bright_col = np.unravel_index(int(np.argmax(illuminance)), illuminance.shape)
    # MATLAB oiExtractBright/oiCrop uses a [x y width-1 height-1] rect
    # convention, so the "1x1" bright patch in auto exposure is a 2x2 crop.
    row_start = min(bright_row, max(cube.shape[0] - 2, 0))
    col_start = min(bright_col, max(cube.shape[1] - 2, 0))
    row_stop = min(row_start + 2, cube.shape[0])
    col_stop = min(col_start + 2, cube.shape[1])

    small_oi = oi.clone()
    small_oi.data["photons"] = cube[row_start:row_stop, col_start:col_stop, :].copy()
    small_oi.fields["optics"] = dict(small_oi.fields["optics"])
    small_oi.fields["optics"]["model"] = "skip"
    small_oi.fields["optics"]["compute_method"] = "skip"
    small_oi.fields["optics"]["offaxis_method"] = "skip"

    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    small_sensor = sensor.clone()
    small_sensor.fields["size"] = (8 * pattern.shape[0], 8 * pattern.shape[1])
    small_sensor.fields["integration_time"] = 1.0
    small_sensor.fields["auto_exposure"] = False
    small_sensor.fields["pixel"] = dict(small_sensor.fields["pixel"])
    small_sensor.fields["pixel"]["voltage_swing"] = 1e6
    small_sensor.data.clear()

    sensor_hfov = float(sensor_get(small_sensor, "fov", None, oi))
    sensor_vfov = float(sensor_get(small_sensor, "vfov", None, oi))
    image_distance = float(oi.fields.get("image_distance_m", oi_get(oi, "focal length")))
    target_hfov = 2.0 * sensor_hfov
    target_vfov = 2.0 * sensor_vfov
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(target_hfov) / 2.0)
    height_m = 2.0 * image_distance * np.tan(np.deg2rad(target_vfov) / 2.0)
    small_oi.fields["width_m"] = width_m
    small_oi.fields["height_m"] = height_m
    small_oi.fields["fov_deg"] = target_hfov
    small_oi.fields["vfov_deg"] = target_vfov
    small_oi.fields["rows"] = int(small_oi.data["photons"].shape[0])
    small_oi.fields["cols"] = int(small_oi.data["photons"].shape[1])
    small_oi.fields["sample_spacing_m"] = width_m / max(int(small_oi.data["photons"].shape[1]), 1)

    signal_sensor = sensor_compute(small_sensor, small_oi, seed=0)
    signal_voltage = np.asarray(signal_sensor.data["volts"], dtype=float)
    max_signal_voltage = float(np.max(signal_voltage))
    return (0.95 * voltage_swing) / max(max_signal_voltage, 1e-12)


def sensor_compute(
    sensor: Sensor,
    oi: OpticalImage,
    show_bar: bool | None = None,
    *,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Compute sensor response from an optical image."""

    del show_bar
    computed = sensor.clone()
    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    pixel = computed.fields["pixel"]
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * pixel["fill_factor"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])

    integration_time_value = np.asarray(computed.fields["integration_time"], dtype=float)
    integration_time_matrix: np.ndarray | None = None
    if integration_time_value.ndim > 1:
        pattern = np.asarray(computed.fields["pattern"], dtype=int)
        if not computed.fields["mosaic"]:
            raise UnsupportedOptionError("sensorCompute", "matrix integration times on nonmosaic sensors")
        if integration_time_value.shape != pattern.shape:
            raise UnsupportedOptionError("sensorCompute", "matrix integration times with non-pattern shape")
        if computed.fields["auto_exposure"]:
            raise UnsupportedOptionError("sensorCompute", "auto exposure with matrix integration times")
        if np.any(integration_time_value <= 0.0):
            raise UnsupportedOptionError("sensorCompute", "nonpositive integration times")
        integration_time_matrix = np.asarray(integration_time_value, dtype=float).copy()
        computed.fields["integration_time"] = integration_time_matrix.copy()
        integration_time_array = np.empty(0, dtype=float)
    else:
        integration_time_array = integration_time_value.reshape(-1)

    if integration_time_matrix is None and integration_time_array.size == 0:
        integration_time_array = np.array([0.0], dtype=float)

    if integration_time_matrix is None and integration_time_array.size == 1 and (
        computed.fields["auto_exposure"] or float(integration_time_array[0]) <= 0.0
    ):
        computed.fields["integration_time"] = _auto_exposure_default(computed, oi)
        integration_time_array = np.array([float(computed.fields["integration_time"])], dtype=float)
    elif integration_time_matrix is None and integration_time_array.size > 1:
        if computed.fields["auto_exposure"]:
            raise UnsupportedOptionError("sensorCompute", "auto exposure with multiple integration times")
        if np.any(integration_time_array <= 0.0):
            raise UnsupportedOptionError("sensorCompute", "nonpositive integration times")
        computed.fields["integration_time"] = integration_time_array.copy()
    elif integration_time_matrix is None:
        computed.fields["integration_time"] = float(integration_time_array[0])

    seed_value = sensor.fields.get("noise_seed", 0) if seed is None else seed
    rng = np.random.default_rng(seed_value)
    noise_flag = int(computed.fields["noise_flag"])

    if computed.fields["mosaic"]:
        current_density = _signal_current_density(oi, computed)
        signal_current = _spatial_integrate_current_density(current_density, oi, computed)
        exposure_map_m: np.ndarray | None = None
        if integration_time_matrix is not None:
            pattern_rows, pattern_cols = integration_time_matrix.shape
            row_tiles = int(np.ceil(signal_current.shape[0] / pattern_rows))
            col_tiles = int(np.ceil(signal_current.shape[1] / pattern_cols))
            exposure_map_m = np.tile(integration_time_matrix, (row_tiles, col_tiles))[
                : signal_current.shape[0], : signal_current.shape[1]
            ]

        def _base_volts(current_integration_time: float) -> tuple[np.ndarray, None]:
            volts = signal_current * (current_integration_time * conversion_gain / _ELEMENTARY_CHARGE_C)
            return np.asarray(volts, dtype=float).copy(), None

        def _base_volts_matrix() -> tuple[np.ndarray, None]:
            assert exposure_map_m is not None
            volts = signal_current * (exposure_map_m * conversion_gain / _ELEMENTARY_CHARGE_C)
            return np.asarray(volts, dtype=float).copy(), None

    else:
        filter_spectra = _sensor_qe_on_wave(computed, wave)
        electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
        electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, computed) * pixel_area

        def _base_volts(current_integration_time: float) -> tuple[np.ndarray, np.ndarray]:
            electrons = electron_rate * current_integration_time
            volts_full = electrons * conversion_gain
            volts_full = np.asarray(volts_full, dtype=float)
            return volts_full.copy(), volts_full.copy()

    etendue = _sensor_etendue(computed)
    analog_gain = float(computed.fields["analog_gain"])
    analog_offset = float(computed.fields["analog_offset"])
    voltage_swing = float(pixel["voltage_swing"])
    volts_captures: list[np.ndarray] = []
    channel_volts_captures: list[np.ndarray] = []
    dv_captures: list[np.ndarray] = []

    if integration_time_matrix is not None:
        volts, channel_volts = _base_volts_matrix()
        if volts.ndim == 2:
            volts = volts * etendue
        else:
            volts = volts * etendue[:, :, None]

        if noise_flag in {1, 2, -2}:
            if noise_flag == 2:
                volts = volts + (float(pixel["dark_voltage_v_per_sec"]) * exposure_map_m)
            volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain
            if noise_flag == 2:
                volts = _apply_read_noise(rng, volts, float(pixel["read_noise_v"]))
            if noise_flag in {1, 2}:
                volts = _apply_fixed_pattern_noise(
                    rng,
                    volts,
                    dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                    prnu_sigma=float(pixel["prnu_sigma"]),
                    integration_time=exposure_map_m,
                    auto_exposure=bool(computed.fields["auto_exposure"]),
                )
        elif noise_flag not in {0, -1}:
            raise UnsupportedOptionError("sensorCompute", f"noise flag {noise_flag}")

        volts = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)
        computed.data["volts"] = np.asarray(volts, dtype=float).copy()
        computed.data["channel_volts"] = None if channel_volts is None else np.asarray(channel_volts, dtype=float).copy()

        if param_format(computed.fields["quantization"]) != "analog":
            nbits = int(computed.fields["nbits"])
            max_digital = (2**nbits) - 1
            computed.data["dv"] = np.round(volts / voltage_swing * max_digital).astype(np.int32)
        return track_session_object(session, computed)

    for integration_time in integration_time_array:
        volts, channel_volts = _base_volts(float(integration_time))
        if volts.ndim == 2:
            volts = volts * etendue
        else:
            volts = volts * etendue[:, :, None]

        if noise_flag in {1, 2, -2}:
            if noise_flag == 2:
                volts = volts + (float(pixel["dark_voltage_v_per_sec"]) * float(integration_time))
            volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain
            if noise_flag == 2:
                volts = _apply_read_noise(rng, volts, float(pixel["read_noise_v"]))
            if noise_flag in {1, 2}:
                volts = _apply_fixed_pattern_noise(
                    rng,
                    volts,
                    dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                    prnu_sigma=float(pixel["prnu_sigma"]),
                    integration_time=float(integration_time),
                    auto_exposure=bool(computed.fields["auto_exposure"]),
                )
        elif noise_flag not in {0, -1}:
            raise UnsupportedOptionError("sensorCompute", f"noise flag {noise_flag}")

        volts = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)
        volts_captures.append(np.asarray(volts, dtype=float).copy())
        if channel_volts is not None:
            channel_volts_captures.append(np.asarray(channel_volts, dtype=float).copy())

        if param_format(computed.fields["quantization"]) != "analog":
            nbits = int(computed.fields["nbits"])
            max_digital = (2**nbits) - 1
            dv_captures.append(np.round(volts / voltage_swing * max_digital).astype(np.int32))

    if len(volts_captures) == 1:
        computed.data["volts"] = volts_captures[0]
        computed.data["channel_volts"] = channel_volts_captures[0] if channel_volts_captures else None
        if dv_captures:
            computed.data["dv"] = dv_captures[0]
    else:
        computed.data["volts"] = np.stack(volts_captures, axis=2)
        computed.data["channel_volts"] = (
            np.stack(channel_volts_captures, axis=3) if channel_volts_captures else None
        )
        if dv_captures:
            computed.data["dv"] = np.stack(dv_captures, axis=2)
        exposure_plane = int(np.rint(float(computed.fields.get("exposure_plane", np.floor(len(volts_captures) / 2.0) + 1))))
        computed.fields["exposure_plane"] = int(np.clip(exposure_plane, 1, len(volts_captures)))

    return track_session_object(session, computed)
