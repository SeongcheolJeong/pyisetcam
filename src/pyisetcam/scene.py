"""Scene creation and manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.signal import convolve2d

from .assets import AssetStore
from .color import luminance_from_photons
from .exceptions import UnsupportedOptionError
from .types import Scene
from .utils import DEFAULT_WAVE, blackbody, energy_to_quanta, param_format, quanta_to_energy

DEFAULT_DISTANCE_M = 1.2
DEFAULT_FOV_DEG = 10.0
_MACBETH_GRID = (4, 6)
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


def _wave_or_default(wave: Any | None) -> np.ndarray:
    if wave is None:
        return DEFAULT_WAVE.copy()
    return np.asarray(wave, dtype=float).reshape(-1)


def _scene_image_input(input_data: Any) -> tuple[np.ndarray, str, str]:
    if isinstance(input_data, (str, Path)):
        path = Path(input_data).expanduser()
        image = np.asarray(iio.imread(path), dtype=float)
        return image, str(path), path.stem
    image = np.asarray(input_data, dtype=float)
    return image, "numerical", "numerical"


def _scene_display(display: Any, wave: np.ndarray | None, *, asset_store: AssetStore) -> Any:
    from .display import display_create, display_set

    if display is None:
        return display_create("default", asset_store=asset_store, wave=wave)
    if isinstance(display, str):
        return display_create(display, asset_store=asset_store, wave=wave)

    current = display
    if wave is None:
        return current
    display_wave = np.asarray(current.fields.get("wave", wave), dtype=float).reshape(-1)
    if not np.array_equal(display_wave, wave):
        current = display_set(current.clone(), "wave", wave)
    return current


def _prepare_display_image(image: np.ndarray, im_type: str, n_primaries: int) -> np.ndarray:
    current = np.asarray(image, dtype=float)
    if current.ndim == 2:
        current = current[:, :, None]
    if current.ndim != 3:
        raise ValueError("scene_from_file expects a 2D or 3D image array.")
    if current.shape[2] == 4:
        current = current[:, :, :3]

    normalized_type = param_format(im_type)
    if normalized_type in {"monochrome", "unispectral"}:
        if current.shape[2] > 1:
            current = np.mean(current[:, :, : min(3, current.shape[2])], axis=2, keepdims=True)
        current = np.repeat(current, n_primaries, axis=2)
    elif normalized_type == "rgb":
        if current.shape[2] == 1:
            current = np.repeat(current, n_primaries, axis=2)
        elif current.shape[2] < n_primaries:
            current = np.pad(current, ((0, 0), (0, 0), (0, n_primaries - current.shape[2])), mode="constant")
        elif current.shape[2] > n_primaries:
            current = current[:, :, :n_primaries]
    else:
        raise UnsupportedOptionError("sceneFromFile", im_type)
    return current


def _display_linear_rgb(image: np.ndarray, gamma_table: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma_table, dtype=float)
    if gamma.ndim != 2:
        raise ValueError("Display gamma table must be a 2D matrix.")

    current = np.asarray(image, dtype=float)
    max_value = float(np.max(current)) if current.size else 0.0
    if max_value <= 1.0:
        digital = np.rint(current * (gamma.shape[0] - 1))
    elif max_value <= 255.0:
        digital = np.rint(current / 255.0 * (gamma.shape[0] - 1))
    else:
        digital = np.rint(current / max(max_value, 1e-12) * (gamma.shape[0] - 1))

    digital = np.clip(digital.astype(int), 0, gamma.shape[0] - 1)
    linear = np.empty_like(current, dtype=float)
    for channel in range(current.shape[2]):
        gamma_channel = min(channel, gamma.shape[1] - 1)
        linear[:, :, channel] = gamma[digital[:, :, channel], gamma_channel]
    return linear


def _invalidate_scene_caches(scene: Scene) -> None:
    scene.fields.pop("luminance", None)
    scene.fields.pop("mean_luminance", None)


def _spatial_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _update_scene_geometry(scene: Scene) -> Scene:
    photons = np.asarray(scene.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.size == 0:
        return scene
    rows, cols = photons.shape[:2]
    distance_m = float(scene.fields.get("distance_m", DEFAULT_DISTANCE_M))
    fov_deg = float(scene.fields.get("fov_deg", DEFAULT_FOV_DEG))
    width_m = 2.0 * distance_m * np.tan(np.deg2rad(fov_deg) / 2.0)
    height_m = width_m * rows / max(cols, 1)
    vfov_deg = np.rad2deg(2.0 * np.arctan2(height_m / 2.0, distance_m))
    scene.fields.update(
        {
            "wave": np.asarray(scene.fields.get("wave", DEFAULT_WAVE), dtype=float),
            "distance_m": distance_m,
            "fov_deg": fov_deg,
            "vfov_deg": float(vfov_deg),
            "width_m": float(width_m),
            "height_m": float(height_m),
            "rows": int(rows),
            "cols": int(cols),
        }
    )
    return scene


def _macbeth_cube(
    reflectances: np.ndarray,
    illuminant_photons: np.ndarray,
    patch_size: int,
    *,
    black_border: bool,
) -> np.ndarray:
    rows, cols = _MACBETH_GRID
    wave_count = reflectances.shape[0]

    # MATLAB uses reshape(transpose(macbethChart), 4, 6, nWaves), which keeps
    # the patch numbering column-major: 1:4 in the first column, 5:8 in the second.
    chart = np.reshape(reflectances.T, (rows, cols, wave_count), order="F")
    chart = chart * illuminant_photons.reshape(1, 1, -1)
    cube = np.repeat(np.repeat(chart, patch_size, axis=0), patch_size, axis=1)

    if not black_border:
        return cube

    data = cube.copy()
    border_px = int(np.floor(0.2 * patch_size))
    if border_px <= 0:
        return data
    for col in range(1, cols + 1):
        start = int(np.floor(col * patch_size - border_px))
        stop = min(col * patch_size, data.shape[1] - 1)
        data[:, start : stop + 1, :] = 0.0
    for row in range(1, rows + 1):
        start = int(np.floor(row * patch_size - border_px))
        stop = min(row * patch_size, data.shape[0] - 1)
        data[start : stop + 1, :, :] = 0.0

    pad = ((border_px, 0), (border_px, 0), (0, 0))
    return np.pad(data, pad, mode="constant")


def _load_d65(wave: np.ndarray, asset_store: AssetStore) -> tuple[np.ndarray, np.ndarray]:
    wave_nm, energy = asset_store.load_illuminant("D65.mat", wave_nm=wave)
    return wave_nm, energy


def _scale_energy_to_luminance(
    energy: np.ndarray,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
    luminance_cd_m2: float = 100.0,
) -> np.ndarray:
    _, xyz_energy = asset_store.load_xyz(wave_nm=wave, energy=True)
    y_bar = np.asarray(xyz_energy, dtype=float)[:, 1]
    delta = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    current = 683.0 * float(np.sum(np.asarray(energy, dtype=float) * y_bar * delta))
    if current <= 0.0:
        return np.asarray(energy, dtype=float)
    return np.asarray(energy, dtype=float) * (float(luminance_cd_m2) / current)


def _scene_size_2d(size: Any, *, default: int) -> tuple[int, int]:
    if size is None:
        return (int(default), int(default))
    if np.isscalar(size):
        side = int(size)
        return (side, side)
    values = np.asarray(size, dtype=int).reshape(-1)
    if values.size == 1:
        side = int(values[0])
        return (side, side)
    return (int(values[0]), int(values[1]))


def _matlab_round_scalar(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _spectral_illuminant(
    spectral_type: str,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
    blackbody_temperature_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spectral_key = param_format(spectral_type)
    if spectral_key == "d65":
        _, illuminant_energy = _load_d65(wave, asset_store)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    if spectral_key in {"ee", "equalenergy"}:
        illuminant_energy = np.ones(wave.size, dtype=float)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    if spectral_key in {"ep", "equalphoton", "equalphotons"}:
        illuminant_photons = np.ones(wave.size, dtype=float)
        return quanta_to_energy(illuminant_photons, wave), illuminant_photons
    if spectral_key in {"bb", "blackbody"}:
        temperature_k = 5000.0 if blackbody_temperature_k is None else float(blackbody_temperature_k)
        illuminant_energy = np.asarray(blackbody(wave, temperature_k, kind="energy"), dtype=float).reshape(-1)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    raise UnsupportedOptionError("sceneCreate", f"spectralType={spectral_type}")


def _create_macbeth_scene(
    patch_size: int,
    wave: np.ndarray,
    surface_file: str,
    black_border: bool,
    *,
    asset_store: AssetStore,
) -> Scene:
    _, reflectances = asset_store.load_reflectances(surface_file, wave_nm=wave)
    _, illuminant_energy = _load_d65(wave, asset_store)
    illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    scene = Scene(name="Macbeth D65")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = "D65.mat"
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = _macbeth_cube(
        reflectances,
        illuminant_photons,
        patch_size,
        black_border=black_border,
    )
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _uniform_scene(
    name: str,
    size: int,
    wave: np.ndarray,
    illuminant_energy: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    scene = Scene(name=name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, wave)
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    photons = np.broadcast_to(scene.fields["illuminant_photons"], (size, size, wave.size)).copy()
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _checkerboard_scene(
    pixels_per_check: int,
    number_of_checks: int,
    wave: np.ndarray,
    spectral_type: str,
    *,
    asset_store: AssetStore,
) -> Scene:
    size = 2 * pixels_per_check * number_of_checks
    yy, xx = np.indices((size, size))
    tiles = 1 - ((yy // pixels_per_check + xx // pixels_per_check) % 2)
    pattern = np.clip(tiles.astype(float), 1e-6, 1.0)

    spectral_key = param_format(spectral_type)
    if spectral_key == "d65":
        _, illuminant_energy = _load_d65(wave, asset_store)
        illuminant_energy = _scale_energy_to_luminance(illuminant_energy, wave, asset_store=asset_store)
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    elif spectral_key in {"ee", "equalenergy"}:
        illuminant_energy = _scale_energy_to_luminance(
            np.ones(wave.size, dtype=float),
            wave,
            asset_store=asset_store,
        )
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    elif spectral_key in {"ep", "equalphoton", "equalphotons"}:
        illuminant_photons = np.ones(wave.size, dtype=float)
        illuminant_energy = _scale_energy_to_luminance(
            quanta_to_energy(illuminant_photons, wave),
            wave,
            asset_store=asset_store,
        )
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    else:
        raise UnsupportedOptionError("sceneCreate", f"checkerboard spectralType={spectral_type}")

    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="Checkerboard")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _slanted_bar_scene(
    image_size: int,
    edge_slope: float,
    wave: np.ndarray,
    fov_deg: float,
    dark_level: float,
    *,
    asset_store: AssetStore,
) -> Scene:
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    center = image_size / 2.0
    threshold = center + edge_slope * (yy - center)
    bar = np.where(xx >= threshold, 1.0, dark_level)
    illuminant_energy = _scale_energy_to_luminance(
        np.ones(wave.size, dtype=float),
        wave,
        asset_store=asset_store,
    )
    photons = bar[:, :, None] * energy_to_quanta(illuminant_energy, wave).reshape(1, 1, -1)
    scene = Scene(name="Slanted Bar")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, wave)
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = float(fov_deg)
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _uniform_blackbody_scene(
    size: Any,
    temperature_k: float,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    illuminant_energy = np.asarray(blackbody(wave, float(temperature_k), kind="energy"), dtype=float).reshape(-1)
    return _uniform_scene(f"Uniform BB {int(round(float(temperature_k)))}K", _scene_size_2d(size, default=32)[0], wave, illuminant_energy, asset_store=asset_store)


def _uniform_monochromatic_scene(
    size: Any,
    wavelength: Any,
    *,
    asset_store: AssetStore,
) -> Scene:
    wave = _wave_or_default(wavelength)
    scene = _uniform_scene("Narrow Band", _scene_size_2d(size, default=128)[0], wave, np.ones(wave.size, dtype=float), asset_store=asset_store)
    scene.name = "Narrow Band"
    return scene


def _line_scene(
    spectral_type: str,
    size: Any,
    offset: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=64)
    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    line_pos = _matlab_round_scalar(cols / 2.0) - 1 + int(offset)
    line_pos = min(max(line_pos, 0), cols - 1)
    photons = np.full((rows, cols, wave.size), 1e-4, dtype=float)
    photons[:, line_pos, :] = 1.0
    photons = photons * illuminant_photons.reshape(1, 1, -1)

    scene = Scene(name=f"line-{param_format(spectral_type)}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def _bar_scene(size: Any, width: int, wave: np.ndarray, *, asset_store: AssetStore) -> Scene:
    rows, cols = _scene_size_2d(size, default=64)
    bar_width = max(int(width), 1)
    start = _matlab_round_scalar((cols - bar_width) / 2.0)
    stop = min(start + bar_width, cols)
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    photons = np.full((rows, cols, wave.size), 1e-8, dtype=float)
    photons[:, start:stop, :] = 1.0
    photons = photons * illuminant_photons.reshape(1, 1, -1)

    scene = Scene(name=f"bar-{bar_width}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def _point_array_scene(
    size: Any,
    spacing: int,
    spectral_type: str,
    point_size: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    spacing_px = max(int(spacing), 1)
    pattern = np.zeros((rows, cols), dtype=float)
    x_positions = np.arange(max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0), cols, spacing_px, dtype=int)
    y_positions = np.arange(max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0), rows, spacing_px, dtype=int)
    pattern[np.ix_(y_positions, x_positions)] = 1.0
    if int(point_size) > 1:
        kernel = np.ones((int(point_size), int(point_size)), dtype=float)
        pattern = convolve2d(pattern, kernel, mode="same", boundary="fill")

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="pointarray")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def _grid_lines_scene(
    size: Any,
    spacing: int,
    spectral_type: str,
    thickness: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    spacing_px = max(int(spacing), 1)
    line_thickness = max(int(thickness), 1)
    pattern = np.zeros((rows, cols), dtype=float)
    row_start = max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0)
    col_start = max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0)
    for delta in range(line_thickness):
        row_positions = np.arange(row_start + delta, rows, spacing_px, dtype=int)
        col_positions = np.arange(col_start + delta, cols, spacing_px, dtype=int)
        pattern[row_positions, :] = 1.0
        pattern[:, col_positions] = 1.0
    pattern[pattern == 0.0] = 1e-5

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="gridlines")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def _white_noise_scene(
    size: Any,
    contrast: float,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    sigma = float(contrast) / 100.0 if float(contrast) > 1.0 else float(contrast)
    rng = np.random.default_rng(0)
    pattern = np.maximum(0.0, rng.normal(loc=1.0, scale=sigma, size=(rows, cols)))
    illuminant_energy = _scale_energy_to_luminance(_load_d65(wave, asset_store)[1], wave, asset_store=asset_store)
    illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="white noise")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 1.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def scene_create(scene_name: str = "default", *args: Any, asset_store: AssetStore | None = None) -> Scene:
    """Create a supported milestone-one scene."""

    store = _store(asset_store)
    name = param_format(scene_name)

    if name in {"default", "macbeth", "macbethd65"}:
        patch_size = int(args[0]) if len(args) > 0 else 16
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = str(args[2]) if len(args) > 2 else "macbethChart.mat"
        black_border = bool(args[3]) if len(args) > 3 else False
        return _create_macbeth_scene(patch_size, wave, surface_file, black_border, asset_store=store)

    if name == "empty":
        scene = _create_macbeth_scene(16, _wave_or_default(None), "macbethChart.mat", False, asset_store=store)
        return scene_clear_data(scene)

    if name in {"uniformd65", "uniform d65".replace(" ", "")}:
        size = int(args[0]) if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        _, illuminant_energy = _load_d65(wave, store)
        return _uniform_scene("Uniform D65", size, wave, illuminant_energy, asset_store=store)

    if name in {"uniformee", "uniformequalenergy"}:
        size = int(args[0]) if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return _uniform_scene("Uniform EE", size, wave, np.ones(wave.size, dtype=float), asset_store=store)

    if name in {"uniformbb", "uniformblackbody"}:
        size = args[0] if len(args) > 0 else 32
        temperature_k = float(args[1]) if len(args) > 1 else 5000.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return _uniform_blackbody_scene(size, temperature_k, wave, asset_store=store)

    if name in {"uniformmonochromatic", "narrowband"}:
        wavelength = args[0] if len(args) > 0 else 500.0
        size = args[1] if len(args) > 1 else 128
        return _uniform_monochromatic_scene(size, wavelength, asset_store=store)

    if name in {"line", "lined65", "impulse1dd65"}:
        size = args[0] if len(args) > 0 else 64
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return _line_scene("d65", size, 0, wave, asset_store=store)

    if name in {"lineee", "impulse1dee"}:
        size = args[0] if len(args) > 0 else 64
        offset = int(args[1]) if len(args) > 1 else 0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return _line_scene("ee", size, offset, wave, asset_store=store)

    if name in {"lineequalphoton", "lineep"}:
        size = args[0] if len(args) > 0 else 64
        offset = int(args[1]) if len(args) > 1 else 0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return _line_scene("ep", size, offset, wave, asset_store=store)

    if name == "bar":
        size = args[0] if len(args) > 0 else 64
        width = int(args[1]) if len(args) > 1 else 3
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return _bar_scene(size, width, wave, asset_store=store)

    if name in {"whitenoise", "noise"}:
        size = args[0] if len(args) > 0 else 128
        contrast = float(args[1]) if len(args) > 1 else 20.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return _white_noise_scene(size, contrast, wave, asset_store=store)

    if name in {"pointarray", "manypoints", "point array".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = int(args[1]) if len(args) > 1 else 16
        spectral_type = str(args[2]) if len(args) > 2 else "ep"
        point_size = int(args[3]) if len(args) > 3 else 1
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return _point_array_scene(size, spacing, spectral_type, point_size, wave, asset_store=store)

    if name in {"gridlines", "distortiongrid", "grid lines".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = int(args[1]) if len(args) > 1 else 16
        spectral_type = str(args[2]) if len(args) > 2 else "ep"
        thickness = int(args[3]) if len(args) > 3 else 1
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return _grid_lines_scene(size, spacing, spectral_type, thickness, wave, asset_store=store)

    if name == "checkerboard":
        pixels_per_check = int(args[0]) if len(args) > 0 else 16
        number_of_checks = int(args[1]) if len(args) > 1 else 8
        spectral_type = str(args[2]) if len(args) > 2 and isinstance(args[2], str) else "ep"
        wave_arg = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not isinstance(args[2], str) else None)
        wave = _wave_or_default(wave_arg)
        return _checkerboard_scene(
            pixels_per_check,
            number_of_checks,
            wave,
            spectral_type,
            asset_store=store,
        )

    if name in {"slantedbar", "slanted bar".replace(" ", "")}:
        image_size = int(args[0]) if len(args) > 0 else 256
        edge_slope = float(args[1]) if len(args) > 1 else 0.5
        fov_deg = float(args[2]) if len(args) > 2 else 2.0
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        dark_level = float(args[4]) if len(args) > 4 else 0.0
        return _slanted_bar_scene(image_size, edge_slope, wave, fov_deg, dark_level, asset_store=store)

    raise UnsupportedOptionError("sceneCreate", scene_name)


def scene_from_file(
    input_data: Any,
    im_type: str,
    mean_luminance: float | None = None,
    display: Any = None,
    wave: np.ndarray | None = None,
    illuminant_energy: Any | None = None,
    scale_reflectance: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Create a scene from RGB or monochrome image data and a display model."""

    del scale_reflectance
    store = _store(asset_store)
    normalized_type = param_format(im_type)
    if normalized_type not in {"rgb", "monochrome", "unispectral"}:
        raise UnsupportedOptionError("sceneFromFile", im_type)

    requested_wave = None if wave is None else _wave_or_default(wave)
    current_display = _scene_display(display, requested_wave, asset_store=store)

    from .display import display_get

    if not bool(display_get(current_display, "is emissive")):
        raise UnsupportedOptionError("sceneFromFile", "reflective display")

    wave_nm = np.asarray(display_get(current_display, "wave"), dtype=float).reshape(-1)
    image, filename, source_name = _scene_image_input(input_data)
    spd = np.asarray(display_get(current_display, "spd"), dtype=float)
    n_primaries = spd.shape[1]
    prepared = _prepare_display_image(image, normalized_type, n_primaries)
    linear_rgb = _display_linear_rgb(prepared, np.asarray(display_get(current_display, "gamma table"), dtype=float))
    energy = linear_rgb.reshape(-1, n_primaries) @ spd.T
    photons = energy_to_quanta(energy, wave_nm).reshape(prepared.shape[0], prepared.shape[1], wave_nm.size)

    scene = Scene(name=f"{source_name} - {current_display.name}")
    scene.fields["wave"] = wave_nm
    scene.fields["illuminant_format"] = "spectral"
    if illuminant_energy is None:
        source_illuminant_energy = np.sum(spd, axis=1)
        illuminant_comment = current_display.name
    else:
        source_illuminant_energy, illuminant_comment = _resolve_illuminant_input(
            illuminant_energy,
            wave_nm,
            asset_store=store,
        )
    scene.fields["illuminant_energy"] = np.asarray(source_illuminant_energy, dtype=float).reshape(-1)
    scene.fields["illuminant_photons"] = energy_to_quanta(scene.fields["illuminant_energy"], wave_nm)
    scene.fields["illuminant_comment"] = str(illuminant_comment)
    scene.fields["distance_m"] = float(display_get(current_display, "viewing distance"))
    scene.fields["fov_deg"] = float(prepared.shape[1]) * float(display_get(current_display, "deg per dot"))
    scene.fields["filename"] = filename
    scene.fields["source_type"] = normalized_type
    scene.fields["display_name"] = current_display.name
    scene.data["photons"] = photons
    _update_scene_geometry(scene)

    if mean_luminance is not None:
        scene = scene_adjust_luminance(scene, float(mean_luminance), asset_store=store)
    return scene


def scene_calculate_luminance(scene: Scene, *, asset_store: AssetStore | None = None) -> np.ndarray:
    store = _store(asset_store)
    luminance = luminance_from_photons(scene.data["photons"], np.asarray(scene.fields["wave"], dtype=float), asset_store=store)
    scene.fields["luminance"] = luminance
    scene.fields["mean_luminance"] = float(np.mean(luminance))
    return luminance


def scene_adjust_luminance(
    scene: Scene,
    target_luminance: float,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    current_luminance = scene_get(scene, "mean luminance", asset_store=asset_store)
    if current_luminance <= 0.0:
        return scene
    scale = float(target_luminance) / current_luminance
    scene.data["photons"] = np.asarray(scene.data["photons"], dtype=float) * scale
    if "illuminant_photons" in scene.fields:
        scene.fields["illuminant_photons"] = np.asarray(scene.fields["illuminant_photons"], dtype=float) * scale
    if "illuminant_energy" in scene.fields:
        scene.fields["illuminant_energy"] = np.asarray(scene.fields["illuminant_energy"], dtype=float) * scale
    _invalidate_scene_caches(scene)
    scene_calculate_luminance(scene, asset_store=asset_store)
    return scene


def _resolve_illuminant_input(
    ill_energy: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, str]:
    if isinstance(ill_energy, str):
        path = Path(ill_energy)
        if path.exists():
            data = asset_store.load_mat(path)
            name = path.name
        elif ill_energy.upper() == "D65":
            _, data_values = asset_store.load_illuminant("D65.mat", wave_nm=wave)
            return data_values, "D65.mat"
        else:
            data = asset_store.load_mat(ill_energy)
            name = ill_energy
        energy = np.asarray(data["data"], dtype=float).reshape(-1)
        source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        return np.interp(wave, source_wave, energy, left=0.0, right=0.0), name
    if isinstance(ill_energy, dict) and "energy" in ill_energy:
        return np.asarray(ill_energy["energy"], dtype=float).reshape(-1), str(ill_energy.get("name", "custom"))
    if ill_energy is None:
        return blackbody(wave, 6500.0, kind="energy"), "blackbody-6500K"
    return np.asarray(ill_energy, dtype=float).reshape(-1), "custom"


def scene_adjust_illuminant(
    scene: Scene,
    ill_energy: Any,
    preserve_mean: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    store = _store(asset_store)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    original_mean = float(scene_get(scene, "mean luminance", asset_store=store))
    new_energy, comment = _resolve_illuminant_input(ill_energy, wave, asset_store=store)
    new_photons = energy_to_quanta(new_energy, wave)
    current_illuminant = np.asarray(
        scene.fields.get("illuminant_photons", np.ones_like(new_photons)),
        dtype=float,
    )
    factor = new_photons / np.maximum(current_illuminant, 1e-12)
    scene.data["photons"] = np.asarray(scene.data["photons"], dtype=float) * factor.reshape(1, 1, -1)
    scene.fields["illuminant_energy"] = new_energy
    scene.fields["illuminant_photons"] = new_photons
    scene.fields["illuminant_comment"] = comment
    _invalidate_scene_caches(scene)
    if preserve_mean:
        scene_adjust_luminance(scene, original_mean, asset_store=store)
    return scene


def scene_clear_data(scene: Scene) -> Scene:
    cleared = scene.clone()
    if "photons" in cleared.data:
        cleared.data["photons"] = np.zeros_like(cleared.data["photons"], dtype=float)
    _invalidate_scene_caches(cleared)
    return cleared


def scene_get(scene: Scene, parameter: str, *args: Any, asset_store: AssetStore | None = None) -> Any:
    key = param_format(parameter)
    if key == "type":
        return scene.type
    if key == "name":
        return scene.name
    if key == "filename":
        return scene.fields.get("filename")
    if key == "sourcetype":
        return scene.fields.get("source_type")
    if key == "metadata":
        return scene.metadata
    if key == "wave":
        return np.asarray(scene.fields["wave"], dtype=float)
    if key == "photons":
        return np.asarray(scene.data["photons"], dtype=float)
    if key == "data":
        return scene.data
    if key == "rows":
        return int(scene.fields["rows"])
    if key == "cols":
        return int(scene.fields["cols"])
    if key == "size":
        return (int(scene.fields["rows"]), int(scene.fields["cols"]))
    if key in {"distance", "distancem"}:
        return float(scene.fields["distance_m"])
    if key == "depthmap":
        depth_map = scene.fields.get("depth_map_m")
        if depth_map is None:
            depth_map = np.full(scene_get(scene, "size"), float(scene.fields["distance_m"]), dtype=float)
        scale = _spatial_unit_scale(args[0] if args else None)
        return np.asarray(depth_map, dtype=float) * scale
    if key == "depthrange":
        depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
        positive = depth_map[depth_map > 0.0]
        if positive.size == 0:
            return np.empty(0, dtype=float)
        scale = _spatial_unit_scale(args[0] if args else None)
        return np.array([positive.min(), positive.max()], dtype=float) * scale
    if key in {"fov", "hfov", "fovhorizontal", "wangular"}:
        return float(scene.fields["fov_deg"])
    if key in {"vfov", "fovvertical"}:
        return float(scene.fields["vfov_deg"])
    if key == "width":
        return float(scene.fields["width_m"])
    if key == "height":
        return float(scene.fields["height_m"])
    if key == "illuminantformat":
        return scene.fields.get("illuminant_format", "spectral")
    if key == "illuminantphotons":
        return np.asarray(scene.fields["illuminant_photons"], dtype=float)
    if key == "illuminantenergy":
        return np.asarray(scene.fields["illuminant_energy"], dtype=float)
    if key == "luminance":
        if "luminance" not in scene.fields:
            scene_calculate_luminance(scene, asset_store=asset_store)
        return np.asarray(scene.fields["luminance"], dtype=float)
    if key in {"meanluminance", "luminancemean"}:
        if "mean_luminance" not in scene.fields:
            scene_calculate_luminance(scene, asset_store=asset_store)
        return float(scene.fields["mean_luminance"])
    raise KeyError(f"Unsupported sceneGet parameter: {parameter}")


def scene_set(scene: Scene, parameter: str, value: Any) -> Scene:
    key = param_format(parameter)
    if key == "name":
        scene.name = str(value)
        return scene
    if key == "filename":
        scene.fields["filename"] = str(value)
        return scene
    if key == "metadata":
        scene.metadata = dict(value)
        return scene
    if key == "wave":
        scene.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        _invalidate_scene_caches(scene)
        return scene
    if key == "photons":
        scene.data["photons"] = np.asarray(value, dtype=float)
        _invalidate_scene_caches(scene)
        return _update_scene_geometry(scene)
    if key == "depthmap":
        depth_map = np.asarray(value, dtype=float)
        if depth_map.shape != scene_get(scene, "size"):
            raise ValueError("Scene depth map must match the scene size.")
        scene.fields["depth_map_m"] = depth_map
        return scene
    if key in {"distance", "distancem"}:
        scene.fields["distance_m"] = float(value)
        return _update_scene_geometry(scene)
    if key in {"fov", "hfov", "wangular"}:
        scene.fields["fov_deg"] = float(value)
        return _update_scene_geometry(scene)
    if key == "illuminantenergy":
        wave = np.asarray(scene.fields["wave"], dtype=float)
        energy = np.asarray(value, dtype=float).reshape(-1)
        scene.fields["illuminant_energy"] = energy
        scene.fields["illuminant_photons"] = energy_to_quanta(energy, wave)
        _invalidate_scene_caches(scene)
        return scene
    if key == "illuminantphotons":
        scene.fields["illuminant_photons"] = np.asarray(value, dtype=float).reshape(-1)
        _invalidate_scene_caches(scene)
        return scene
    raise KeyError(f"Unsupported sceneSet parameter: {parameter}")
