"""Sensor creation and computation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from .assets import AssetStore
from .color import luminance_from_photons
from .exceptions import UnsupportedOptionError
from .optics import DEFAULT_FOCAL_LENGTH_M
from .optics import oi_get
from .types import OpticalImage, Sensor
from .utils import DEFAULT_WAVE, ensure_multiple, param_format, tile_pattern

_DEFAULT_PIXEL = {
    "size_m": np.array([2.8e-6, 2.8e-6], dtype=float),
    "fill_factor": 0.75,
    "conversion_gain_v_per_electron": 1.0e-4,
    "voltage_swing": 1.0,
    "read_noise_v": 1.0e-3,
    "dsnu_sigma_v": 0.0,
    "prnu_sigma": 0.0,
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
        }
    )
    return sensor


def _filter_bundle(
    filter_name: str,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, list[str]]:
    _, spectra, names = asset_store.load_color_filters(filter_name, wave_nm=wave)
    return np.asarray(spectra, dtype=float), names


def sensor_create(
    sensor_type: str = "default",
    pixel: dict[str, Any] | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> Sensor:
    """Create a supported sensor."""

    store = _store(asset_store)
    normalized = param_format(sensor_type)
    pixel_dict = _default_pixel(pixel)
    wave = np.asarray(pixel_dict.get("wave", DEFAULT_WAVE), dtype=float)
    size = tuple(pixel_dict.get("size", (72, 88)))

    if normalized in {"default", "color", "bayer", "rgb", "bayergrbg", "bayer-grbg"}:
        sensor = _sensor_base("bayer-grbg", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 1], [3, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return sensor

    if normalized in {"bayerrggb", "bayer-rggb"}:
        sensor = _sensor_base("bayer-rggb", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return sensor

    if normalized == "monochrome":
        sensor = _sensor_base("monochrome", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("monochrome", wave, asset_store=store)
        return sensor

    if normalized == "ideal":
        return sensor_create_ideal("xyz", None, asset_store=store)

    raise UnsupportedOptionError("sensorCreate", sensor_type)


def sensor_create_ideal(
    ideal_type: str = "xyz",
    sensor_example: Sensor | None = None,
    pixel_size_m: float | None = None,
    *,
    asset_store: AssetStore | None = None,
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
        return sensor

    if normalized in {"xyz", "matchxyz"}:
        sensor = _sensor_base("ideal-xyz", np.asarray(wave, dtype=float), size, pixel)
        sensor.fields["pattern"] = np.array([[1, 2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("xyz", np.asarray(wave, dtype=float), asset_store=store)
        sensor.fields["noise_flag"] = 0
        sensor.fields["mosaic"] = False
        return sensor

    if normalized == "match" and sensor_example is not None:
        sensor = sensor_example.clone()
        sensor.name = f"ideal-{sensor_example.name}"
        sensor.fields["pixel"] = pixel
        sensor.fields["noise_flag"] = 0
        return sensor

    raise UnsupportedOptionError("sensorCreateIdeal", ideal_type)


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
    if key in {"spectralqe", "qe"}:
        return np.asarray(sensor.fields["filter_spectra"], dtype=float)
    if key in {"filternames", "filtername"}:
        return list(sensor.fields["filter_names"])
    if key == "nfilters":
        return int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
    if key == "size":
        return tuple(sensor.fields["size"])
    if key == "rows":
        return int(sensor.fields["size"][0])
    if key == "cols":
        return int(sensor.fields["size"][1])
    if key in {"pixelfields", "pixel"}:
        return sensor.fields["pixel"]
    if key in {"pixelsize", "pixelsizesamefillfactor"}:
        return np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
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
    if key == "nbits":
        return int(sensor.fields["nbits"])
    if key in {"quantization", "quantizationmethod"}:
        return sensor.fields["quantization"]
    if key in {"pixelvoltageswing", "voltageswing"}:
        return float(sensor.fields["pixel"]["voltage_swing"])
    if key == "volts":
        return sensor.data.get("volts")
    if key == "dv":
        return sensor.data.get("dv")
    if key == "dvorvolts":
        return sensor.data.get("dv", sensor.data.get("volts"))
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
        oi = args[1] if len(args) >= 2 else args[0] if args else None
        focal_length = DEFAULT_FOCAL_LENGTH_M if oi is None else float(oi_get(oi, "focal length"))
        pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
        width = sensor.fields["size"][1] * pixel_size[1]
        return float(np.rad2deg(2.0 * np.arctan2(width / 2.0, focal_length)))
    if key in {"fovvertical", "vfov"}:
        oi = args[1] if len(args) >= 2 else args[0] if args else None
        focal_length = DEFAULT_FOCAL_LENGTH_M if oi is None else float(oi_get(oi, "focal length"))
        pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
        height = sensor.fields["size"][0] * pixel_size[0]
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
        sensor.fields["size"] = (int(value[0]), int(value[1]))
        return sensor
    if key == "pattern":
        sensor.fields["pattern"] = np.asarray(value, dtype=int)
        return sensor
    if key in {"filterspectra", "colorfilters"}:
        sensor.fields["filter_spectra"] = np.asarray(value, dtype=float)
        return sensor
    if key in {"filternames", "filtername"}:
        sensor.fields["filter_names"] = list(value)
        return sensor
    if key in {"pixelsizesamefillfactor", "pixelsize"}:
        size_value = np.asarray(value, dtype=float)
        if size_value.size == 1:
            size_value = np.repeat(size_value, 2)
        sensor.fields["pixel"]["size_m"] = size_value
        return sensor
    if key in {"integrationtime", "exptime"}:
        sensor.fields["integration_time"] = float(value)
        return sensor
    if key == "autoexposure":
        sensor.fields["auto_exposure"] = bool(value)
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
    if key in {"quantization", "quantizationmethod"}:
        sensor.fields["quantization"] = str(value)
        return sensor
    if key == "volts":
        sensor.data["volts"] = np.asarray(value, dtype=float)
        return sensor
    raise KeyError(f"Unsupported sensorSet parameter: {parameter}")


def sensor_set_size_to_fov(sensor: Sensor, fov: float | tuple[float, float], oi: OpticalImage) -> Sensor:
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    pattern_rows, pattern_cols = pattern.shape
    focal_length = float(oi_get(oi, "focal length"))
    if isinstance(fov, (tuple, list, np.ndarray)):
        hfov = float(fov[0])
        vfov = float(fov[1] if len(fov) > 1 else fov[0])
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        height = 2.0 * focal_length * np.tan(np.deg2rad(vfov) / 2.0)
        cols = max(2, int(round(width / pixel_size[1])))
        rows = max(2, int(round(height / pixel_size[0])))
    else:
        hfov = float(fov)
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        current_width = sensor.fields["size"][1] * pixel_size[1]
        scale = width / max(current_width, 1e-12)
        rows = max(2, int(round(sensor.fields["size"][0] * scale)))
        cols = max(2, int(round(sensor.fields["size"][1] * scale)))
    cols = ensure_multiple(cols, pattern_cols)
    rows = ensure_multiple(rows, pattern_rows)
    sensor.fields["size"] = (rows, cols)
    return sensor


def _poisson_like(rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    lam = np.clip(np.asarray(lam, dtype=float), 0.0, None)
    out = np.empty_like(lam, dtype=float)
    low = lam < 1e6
    out[low] = rng.poisson(lam[low])
    out[~low] = rng.normal(lam[~low], np.sqrt(lam[~low]))
    return np.clip(out, 0.0, None)


def _sample_centers(count: int, spacing_m: float) -> np.ndarray:
    return ((np.arange(count, dtype=float) + 0.5) - (count / 2.0)) * float(spacing_m)


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
    sigma = (row_samples_per_pixel / 4.0, col_samples_per_pixel / 4.0)

    regridded = np.empty((sensor_rows, sensor_cols, density_cube.shape[2]), dtype=float)
    for channel_index in range(density_cube.shape[2]):
        plane = density_cube[:, :, channel_index]
        if row_samples_per_pixel > 1.0 or col_samples_per_pixel > 1.0:
            plane = gaussian_filter(plane, sigma=sigma, mode="nearest")
        interpolator = RegularGridInterpolator((oi_y, oi_x), plane, bounds_error=False, fill_value=0.0)
        regridded[:, :, channel_index] = interpolator(np.stack([sensor_yy, sensor_xx], axis=-1))
    return regridded


def _auto_exposure_default(sensor: Sensor, oi: OpticalImage) -> float:
    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(sensor.fields["wave"], dtype=float)
    voltage_swing = float(sensor.fields["pixel"]["voltage_swing"])

    illuminance = luminance_from_photons(cube, wave, asset_store=AssetStore.default())
    bright_row, bright_col = np.unravel_index(int(np.argmax(illuminance)), illuminance.shape)

    small_oi = oi.clone()
    small_oi.data["photons"] = cube[bright_row : bright_row + 1, bright_col : bright_col + 1, :].copy()
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

    sensor_fov = float(sensor_get(small_sensor, "fov", oi))
    image_distance = float(oi.fields.get("image_distance_m", oi_get(oi, "focal length")))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(2.0 * sensor_fov) / 2.0)
    small_oi.fields["width_m"] = width_m
    small_oi.fields["height_m"] = width_m
    small_oi.fields["sample_spacing_m"] = width_m

    signal_sensor = sensor_compute(small_sensor, small_oi, seed=0)
    signal_voltage = np.asarray(signal_sensor.data["volts"], dtype=float)
    max_signal_voltage = float(np.max(signal_voltage))
    return (0.95 * voltage_swing) / max(max_signal_voltage, 1e-12)


def sensor_compute(sensor: Sensor, oi: OpticalImage, show_bar: bool | None = None, *, seed: int = 0) -> Sensor:
    """Compute sensor response from an optical image."""

    del show_bar
    computed = sensor.clone()
    cube = np.asarray(oi.data["photons"], dtype=float)
    rows, cols = computed.fields["size"]
    wave = np.asarray(computed.fields["wave"], dtype=float)
    filter_spectra = np.asarray(computed.fields["filter_spectra"], dtype=float)
    pattern = np.asarray(computed.fields["pattern"], dtype=int)
    pixel = computed.fields["pixel"]
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * pixel["fill_factor"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])
    electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
    electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, computed) * pixel_area

    if computed.fields["auto_exposure"] or computed.fields["integration_time"] <= 0.0:
        computed.fields["integration_time"] = _auto_exposure_default(computed, oi)

    integration_time = float(computed.fields["integration_time"])
    electrons = electron_rate * integration_time
    rng = np.random.default_rng(seed)
    noise_flag = int(computed.fields["noise_flag"])

    if noise_flag in {1, 2}:
        electrons = _poisson_like(rng, electrons)

    volts_full = electrons * conversion_gain
    if noise_flag == 2:
        volts_full *= 1.0 + rng.normal(0.0, float(pixel["prnu_sigma"]), size=volts_full.shape)
        volts_full += rng.normal(0.0, float(pixel["dsnu_sigma_v"]), size=volts_full.shape)
        volts_full += rng.normal(0.0, float(pixel["read_noise_v"]), size=volts_full.shape)

    analog_gain = float(computed.fields["analog_gain"])
    analog_offset = float(computed.fields["analog_offset"])
    volts_full = np.clip((volts_full + analog_offset) / max(analog_gain, 1e-12), 0.0, float(pixel["voltage_swing"]))

    computed.data["channel_volts"] = volts_full

    if computed.fields["mosaic"]:
        tiled_pattern = tile_pattern(pattern, rows, cols)
        mosaic = np.zeros((rows, cols), dtype=float)
        for channel_index in range(volts_full.shape[2]):
            mask = tiled_pattern == (channel_index + 1)
            mosaic[mask] = volts_full[:, :, channel_index][mask]
        computed.data["volts"] = mosaic
    else:
        computed.data["volts"] = volts_full

    if param_format(computed.fields["quantization"]) != "analog":
        nbits = int(computed.fields["nbits"])
        max_digital = (2**nbits) - 1
        computed.data["dv"] = np.round(
            computed.data["volts"] / float(pixel["voltage_swing"]) * max_digital
        ).astype(np.int32)

    return computed
