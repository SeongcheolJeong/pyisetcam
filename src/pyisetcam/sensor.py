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
from .types import OpticalImage, Scene, Sensor
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
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
        sensor.fields["pixel"]["read_noise_v"] = 0.0
        sensor.fields["pixel"]["voltage_swing"] = 1e6
        return sensor

    if normalized in {"xyz", "matchxyz"}:
        sensor = _sensor_base("ideal-xyz", np.asarray(wave, dtype=float), size, pixel)
        sensor.fields["pattern"] = np.array([[1, 2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("xyz", np.asarray(wave, dtype=float), asset_store=store)
        sensor.fields["noise_flag"] = 0
        sensor.fields["mosaic"] = False
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
        sensor.fields["pixel"]["read_noise_v"] = 0.0
        sensor.fields["pixel"]["voltage_swing"] = 1e6
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
    if key in {"nsamplesperpixel", "spatialsamplesperpixel"}:
        return int(sensor.fields.get("n_samples_per_pixel", 1))
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
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
        pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
        width = sensor.fields["size"][1] * pixel_size[1]
        return float(np.rad2deg(2.0 * np.arctan2(width / 2.0, focal_length)))
    if key in {"fovvertical", "vfov"}:
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
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
    if key in {"nsamplesperpixel", "spatialsamplesperpixel"}:
        sensor.fields["n_samples_per_pixel"] = int(value)
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


def _signal_current_density(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    irradiance = np.asarray(oi.data["photons"], dtype=np.float32)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    sensor_wave = np.asarray(sensor.fields["wave"], dtype=float)
    spectral_qe = np.asarray(sensor.fields["filter_spectra"], dtype=np.float32)
    if not np.array_equal(wave, sensor_wave):
        if sensor_wave.size > 1:
            interpolated = np.empty((wave.size, spectral_qe.shape[1]), dtype=np.float32)
            for index in range(spectral_qe.shape[1]):
                interpolated[:, index] = np.interp(wave, sensor_wave, spectral_qe[:, index], left=0.0, right=0.0)
            spectral_qe = interpolated
        else:
            raise ValueError("Sensor and optical image wavelength samplings do not match.")
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
    wave = np.asarray(sensor.fields["wave"], dtype=float)
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

    if computed.fields["mosaic"]:
        current_density = _signal_current_density(oi, computed)
        signal_current = _spatial_integrate_current_density(current_density, oi, computed)
        volts = signal_current * (integration_time * conversion_gain / _ELEMENTARY_CHARGE_C)
        computed.data["channel_volts"] = None
    else:
        volts_full = electrons * conversion_gain
        computed.data["channel_volts"] = volts_full.copy()
        volts = volts_full.copy()

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

    return computed
