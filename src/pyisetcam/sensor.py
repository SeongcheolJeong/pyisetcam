"""Sensor creation and computation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d

from .assets import AssetStore
from .color import luminance_from_photons
from .exceptions import UnsupportedOptionError
from .optics import DEFAULT_FOCAL_LENGTH_M
from .optics import oi_get
from .session import track_session_object
from .types import OpticalImage, Scene, Sensor, SessionContext
from .utils import DEFAULT_WAVE, ensure_multiple, param_format, tile_pattern

_DEFAULT_PIXEL = {
    "size_m": np.array([2.8e-6, 2.8e-6], dtype=float),
    "fill_factor": 0.75,
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


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _default_pixel(pixel: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_PIXEL)
    if pixel:
        merged.update(pixel)
    merged["size_m"] = np.asarray(merged["size_m"], dtype=float)
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
            "size": (int(size[0]), int(size[1])),
            "pixel": _default_pixel(pixel),
            "analog_gain": 1.0,
            "analog_offset": 0.0,
            "nbits": 10,
            "noise_flag": 2,
            "auto_exposure": True,
            "integration_time": 0.0,
            "quantization": "analog",
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
    sensor.fields["quantization"] = str(model.quantization)
    sensor.fields["integration_time"] = float(model.integrationTime)
    sensor.fields["auto_exposure"] = bool(model.AE)
    sensor.fields["noise_flag"] = int(model.noiseFlag)
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

    if normalized == "mt9v024":
        return track_session_object(session, _sensor_vendor_mt9v024(_sensor_variant_name(args, "rgb"), asset_store=store))

    if normalized == "ar0132at":
        return track_session_object(session, _sensor_vendor_ar0132at(_sensor_variant_name(args, "rgb"), asset_store=store))

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
    key = param_format(parameter)
    if key == "type":
        return sensor.type
    if key == "name":
        return sensor.name
    if key == "wave":
        return np.asarray(sensor.fields["wave"], dtype=float)
    if key == "nwave":
        return int(np.asarray(sensor.fields["wave"]).size)
    if key == "pattern":
        return np.asarray(sensor.fields["pattern"], dtype=int)
    if key in {"filterspectra", "colorfilters"}:
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
    if key in {"patterncolors", "pcolors", "blockcolors"}:
        letters = np.array(list(_sensor_filter_color_letters(sensor)), dtype="<U1")
        if letters.size == 0:
            return np.empty(np.asarray(sensor.fields["pattern"], dtype=int).shape, dtype="<U1")
        known = np.array(list("rgbcmykw"), dtype="<U1")
        unknown = ~np.isin(letters, known)
        letters[unknown] = "k"
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        return letters[np.clip(pattern - 1, 0, letters.size - 1)]
    if key in {"integrationtime", "exptime"}:
        return float(sensor.fields["integration_time"])
    if key == "autoexposure":
        return bool(sensor.fields["auto_exposure"])
    if key == "analoggain":
        return float(sensor.fields["analog_gain"])
    if key == "analogoffset":
        return float(sensor.fields["analog_offset"])
    if key == "noiseflag":
        return int(sensor.fields["noise_flag"])
    if key in {"fpnparameters", "fpn", "fpnoffsetgain", "fpnoffsetandgain"}:
        return np.array([sensor_get(sensor, "dsnu sigma"), sensor_get(sensor, "prnu sigma")], dtype=float)
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "dsnusigma", "sigmadsnu"}:
        return float(sensor.fields["pixel"]["dsnu_sigma_v"])
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "prnusigma", "sigmaprnu", "prnulevel"}:
        return float(sensor.fields["pixel"]["prnu_sigma"]) * 100.0
    if key in {"dsnuimage", "offsetfpnimage"}:
        stored = sensor.fields.get("offset_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"prnuimage", "gainfpnimage"}:
        stored = sensor.fields.get("gain_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key == "nbits":
        return int(sensor.fields["nbits"])
    if key in {"vignetting", "vignettingflag", "pixelvignetting"}:
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
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        stored = sensor.fields.get("etendue")
        if stored is None:
            return np.ones(sensor.fields["size"], dtype=float)
        return np.asarray(stored, dtype=float).copy()
    if key in {"nsamplesperpixel", "spatialsamplesperpixel"}:
        return int(sensor.fields.get("n_samples_per_pixel", 1))
    if key in {"quantization", "quantizationmethod"}:
        return sensor.fields["quantization"]
    if key in {"pixelvoltageswing", "voltageswing"}:
        return float(sensor.fields["pixel"]["voltage_swing"])
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
    if key == "volts":
        return sensor.data.get("volts")
    if key == "electrons":
        return _sensor_electrons(sensor)
    if key == "dv":
        return sensor.data.get("dv")
    if key in {"ncaptures", "ncapture"}:
        volts = sensor.data.get("volts")
        if volts is not None and np.asarray(volts).ndim >= 3:
            return int(np.asarray(volts).shape[2])
        dv = sensor.data.get("dv")
        if dv is not None and np.asarray(dv).ndim >= 3:
            return int(np.asarray(dv).shape[2])
        integration_time = np.asarray(sensor.fields.get("integration_time"))
        if integration_time.ndim > 0 and integration_time.size > 1:
            return int(integration_time.size)
        return 1
    if key == "dvorvolts":
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
    key = param_format(parameter)
    if key == "name":
        sensor.name = str(value)
        return sensor
    if key == "wave":
        sensor.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return sensor
    if key == "size":
        sensor = sensor_set(sensor, "rows", value[0])
        sensor = sensor_set(sensor, "cols", value[1])
        return sensor
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
    if key in {"filterspectra", "colorfilters"}:
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
    if key in {"integrationtime", "exptime"}:
        sensor.fields["integration_time"] = float(value)
        sensor.fields["auto_exposure"] = False
        return sensor
    if key == "autoexposure":
        enabled = bool(value)
        sensor.fields["auto_exposure"] = enabled
        if enabled:
            sensor.fields["integration_time"] = 0.0
        return sensor
    if key == "analoggain":
        sensor.fields["analog_gain"] = float(value)
        return sensor
    if key == "analogoffset":
        sensor.fields["analog_offset"] = float(value)
        return sensor
    if key == "noiseflag":
        sensor.fields["noise_flag"] = int(value)
        return sensor
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "dsnusigma", "sigmadsnu"}:
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(value)
        sensor.fields.pop("offset_fpn_image", None)
        return sensor
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "prnusigma", "sigmaprnu", "prnulevel"}:
        sensor.fields["pixel"]["prnu_sigma"] = float(value) / 100.0
        sensor.fields.pop("gain_fpn_image", None)
        return sensor
    if key in {"dsnuimage", "offsetfpnimage"}:
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("DSNU image must match the sensor size.")
        sensor.fields["offset_fpn_image"] = image
        return sensor
    if key in {"prnuimage", "gainfpnimage"}:
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("PRNU image must match the sensor size.")
        sensor.fields["gain_fpn_image"] = image
        return sensor
    if key in {"vignetting", "vignettingflag", "pixelvignetting"}:
        sensor.fields["vignetting"] = value
        sensor.fields["etendue"] = None
        return sensor
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        etendue = np.asarray(value, dtype=float)
        if etendue.shape != tuple(sensor.fields["size"]):
            raise ValueError("sensor etendue must match the sensor size.")
        sensor.fields["etendue"] = etendue
        return sensor
    if key in {"nsamplesperpixel", "spatialsamplesperpixel"}:
        sensor.fields["n_samples_per_pixel"] = int(value)
        return sensor
    if key in {"quantization", "quantizationmethod"}:
        sensor.fields["quantization"] = str(value)
        return sensor
    if key == "volts":
        volts = np.asarray(value, dtype=float)
        sensor.data["volts"] = volts
        if param_format(sensor.fields.get("quantization", "analog")) == "analog":
            sensor.data.pop("dv", None)
        if volts.ndim >= 2:
            sensor.fields["size"] = (int(volts.shape[0]), int(volts.shape[1]))
        return sensor
    if key in {"dv", "digitalvalues"}:
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
    integration_time: float,
    auto_exposure: bool,
) -> np.ndarray:
    dsnu = _pixel_plane(volts, rng.normal(0.0, dsnu_sigma_v, size=volts.shape[:2]))
    if np.isclose(integration_time, 0.0) and not auto_exposure:
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
    pixel_size = np.asarray(pixel["size_m"], dtype=float)
    fill_factor = float(pixel["fill_factor"])
    if fill_factor <= 0.0:
        return np.zeros(2, dtype=float)
    return np.sqrt(fill_factor) * pixel_size


def _sensor_pd_array(sensor: Sensor, spacing: float) -> np.ndarray:
    if spacing <= 0.0 or spacing > 1.0:
        raise ValueError("spacing must be within (0, 1].")
    pixel = sensor.fields["pixel"]
    pixel_size = np.asarray(pixel["size_m"], dtype=float)
    pd_size = _pixel_pd_size_m(pixel)
    pd_position = (pixel_size - pd_size) / 2.0

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
    seed: int = 0,
    session: SessionContext | None = None,
) -> Sensor:
    """Compute sensor response from an optical image."""

    del show_bar
    computed = sensor.clone()
    cube = np.asarray(oi.data["photons"], dtype=float)
    rows, cols = computed.fields["size"]
    wave = np.asarray(oi.fields["wave"], dtype=float)
    pattern = np.asarray(computed.fields["pattern"], dtype=int)
    pixel = computed.fields["pixel"]
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * pixel["fill_factor"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])

    if computed.fields["auto_exposure"] or computed.fields["integration_time"] <= 0.0:
        computed.fields["integration_time"] = _auto_exposure_default(computed, oi)

    integration_time = float(computed.fields["integration_time"])
    rng = np.random.default_rng(seed)
    noise_flag = int(computed.fields["noise_flag"])

    if computed.fields["mosaic"]:
        current_density = _signal_current_density(oi, computed)
        signal_current = _spatial_integrate_current_density(current_density, oi, computed)
        volts = signal_current * (integration_time * conversion_gain / _ELEMENTARY_CHARGE_C)
        computed.data["channel_volts"] = None
    else:
        filter_spectra = _sensor_qe_on_wave(computed, wave)
        electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
        electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, computed) * pixel_area
        electrons = electron_rate * integration_time
        volts_full = electrons * conversion_gain
        computed.data["channel_volts"] = volts_full.copy()
        volts = volts_full.copy()

    etendue = _sensor_etendue(computed)
    if volts.ndim == 2:
        volts = volts * etendue
    else:
        volts = volts * etendue[:, :, None]

    if noise_flag in {1, 2, -2}:
        if noise_flag == 2:
            volts = volts + (float(pixel["dark_voltage_v_per_sec"]) * integration_time)
        volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain
        if noise_flag == 2:
            volts = _apply_read_noise(rng, volts, float(pixel["read_noise_v"]))
        if noise_flag in {1, 2}:
            volts = _apply_fixed_pattern_noise(
                rng,
                volts,
                dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                prnu_sigma=float(pixel["prnu_sigma"]),
                integration_time=integration_time,
                auto_exposure=bool(computed.fields["auto_exposure"]),
            )
    elif noise_flag not in {0, -1}:
        raise UnsupportedOptionError("sensorCompute", f"noise flag {noise_flag}")

    analog_gain = float(computed.fields["analog_gain"])
    analog_offset = float(computed.fields["analog_offset"])
    computed.data["volts"] = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, float(pixel["voltage_swing"]))

    if param_format(computed.fields["quantization"]) != "analog":
        nbits = int(computed.fields["nbits"])
        max_digital = (2**nbits) - 1
        computed.data["dv"] = np.round(
            computed.data["volts"] / float(pixel["voltage_swing"]) * max_digital
        ).astype(np.int32)

    return track_session_object(session, computed)
