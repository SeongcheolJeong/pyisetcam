"""Scene creation and manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import rotate, zoom
from scipy.signal import convolve2d

from .assets import AssetStore
from .color import luminance_from_photons
from .exceptions import UnsupportedOptionError
from .metrics import chromaticity_xy, xyz_from_energy
from .session import track_session_object
from .types import Scene, SessionContext
from .utils import DEFAULT_WAVE, blackbody, energy_to_quanta, interp_spectra, param_format, quanta_to_energy

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


def _mat_struct_field(structure: Any, field: str, default: Any = None) -> Any:
    if structure is None:
        return default
    return getattr(structure, field, default)


def _resample_wave_last(values: np.ndarray, source_wave_nm: np.ndarray, target_wave_nm: np.ndarray) -> np.ndarray:
    wave_first = np.moveaxis(np.asarray(values, dtype=float), -1, 0)
    resampled = interp_spectra(np.asarray(source_wave_nm, dtype=float), wave_first, np.asarray(target_wave_nm, dtype=float))
    return np.moveaxis(np.asarray(resampled, dtype=float), 0, -1)


def _multispectral_scene_input(
    input_data: Any,
    wave: np.ndarray | None,
    *,
    asset_store: AssetStore,
) -> dict[str, Any]:
    if not isinstance(input_data, (str, Path)):
        raise ValueError("scene_from_file(..., 'multispectral', ...) requires a MAT file path.")

    path = asset_store.resolve(Path(input_data).expanduser())
    data = asset_store.load_mat(path)
    target_wave = None if wave is None else np.asarray(wave, dtype=float).reshape(-1)

    if "mcCOEF" in data:
        basis_struct = data["basis"]
        source_wave = np.asarray(_mat_struct_field(basis_struct, "wave"), dtype=float).reshape(-1)
        basis_matrix = np.asarray(_mat_struct_field(basis_struct, "basis"), dtype=float)
        if basis_matrix.shape[0] != source_wave.size and basis_matrix.shape[1] == source_wave.size:
            basis_matrix = basis_matrix.T
        wave_nm = source_wave if target_wave is None else target_wave
        if target_wave is not None and not np.array_equal(source_wave, target_wave):
            basis_matrix = interp_spectra(source_wave, basis_matrix, target_wave)
        photons = np.tensordot(np.asarray(data["mcCOEF"], dtype=float), np.asarray(basis_matrix, dtype=float).T, axes=([2], [0]))
        if "imgMean" in data:
            image_mean = np.asarray(data["imgMean"], dtype=float).reshape(-1)
            if image_mean.size != source_wave.size:
                raise ValueError("imgMean wavelength length does not match basis wavelength samples.")
            if target_wave is not None and not np.array_equal(source_wave, target_wave):
                image_mean = interp_spectra(source_wave, image_mean, target_wave).reshape(-1)
            photons = photons + image_mean.reshape(1, 1, -1)
    else:
        if "photons" in data:
            photons = np.asarray(data["photons"], dtype=float)
        elif "data" in data:
            photons = np.asarray(data["data"], dtype=float)
        else:
            raise ValueError("Multispectral MAT file must contain either 'mcCOEF', 'photons', or 'data'.")
        if "wave" in data:
            source_wave = np.asarray(data["wave"], dtype=float).reshape(-1)
        elif "wavelength" in data:
            source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        else:
            raise ValueError("Multispectral MAT file is missing wavelength samples.")
        if photons.ndim == 3 and photons.shape[-1] != source_wave.size and photons.shape[0] == source_wave.size:
            photons = np.moveaxis(photons, 0, -1)
        wave_nm = source_wave if target_wave is None else target_wave
        if target_wave is not None and not np.array_equal(source_wave, target_wave):
            photons = _resample_wave_last(photons, source_wave, target_wave)

    illuminant = data.get("illuminant")
    illuminant_photons: np.ndarray | None = None
    illuminant_energy: np.ndarray | None = None
    illuminant_format = "spectral"
    if illuminant is not None:
        spectrum_struct = _mat_struct_field(illuminant, "spectrum")
        illuminant_wave = np.asarray(_mat_struct_field(spectrum_struct, "wave", wave_nm), dtype=float).reshape(-1)
        illuminant_data = _mat_struct_field(illuminant, "data")
        stored_photons = _mat_struct_field(illuminant_data, "photons")
        if stored_photons is not None:
            illuminant_photons = np.asarray(stored_photons, dtype=float)
            if illuminant_photons.ndim == 3 and illuminant_photons.shape[-1] != illuminant_wave.size and illuminant_photons.shape[0] == illuminant_wave.size:
                illuminant_photons = np.moveaxis(illuminant_photons, 0, -1)
            if not np.array_equal(illuminant_wave, wave_nm):
                if illuminant_photons.ndim == 1:
                    illuminant_photons = interp_spectra(illuminant_wave, illuminant_photons, wave_nm).reshape(-1)
                else:
                    illuminant_photons = _resample_wave_last(illuminant_photons, illuminant_wave, wave_nm)
            illuminant_format = "spatial spectral" if illuminant_photons.ndim == 3 else "spectral"
            illuminant_energy = quanta_to_energy(illuminant_photons, wave_nm)
            if np.asarray(illuminant_energy).ndim == 3:
                illuminant_energy = np.mean(np.asarray(illuminant_energy, dtype=float), axis=(0, 1))

    if illuminant_photons is None:
        illuminant_photons = np.maximum(np.mean(np.asarray(photons, dtype=float), axis=(0, 1)), 1e-12)
        illuminant_energy = quanta_to_energy(illuminant_photons, wave_nm)
        illuminant_format = "spectral"

    comment = data.get("comment")
    if comment is None:
        illuminant_comment = path.name
    elif hasattr(comment, "_fieldnames"):
        illuminant_comment = path.name
    else:
        illuminant_comment = str(comment)

    return {
        "photons": np.maximum(np.asarray(photons, dtype=float), 0.0),
        "wave": np.asarray(wave_nm, dtype=float).reshape(-1),
        "illuminant_photons": np.asarray(illuminant_photons, dtype=float),
        "illuminant_energy": np.asarray(illuminant_energy, dtype=float),
        "illuminant_format": illuminant_format,
        "illuminant_comment": illuminant_comment,
        "filename": str(path),
        "source_name": path.stem,
        "distance_m": float(np.asarray(data.get("dist", DEFAULT_DISTANCE_M), dtype=float).reshape(-1)[0]),
        "fov_deg": float(np.asarray(data.get("fov", DEFAULT_FOV_DEG), dtype=float).reshape(-1)[0]),
    }


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


def _scene_resize_array(data: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    current = np.asarray(data, dtype=float)
    if current.ndim == 2:
        factors = (new_rows / max(current.shape[0], 1), new_cols / max(current.shape[1], 1))
    elif current.ndim == 3:
        factors = (
            new_rows / max(current.shape[0], 1),
            new_cols / max(current.shape[1], 1),
            1.0,
        )
    else:
        raise ValueError("Scene resize only supports 2D or 3D arrays.")
    return zoom(current, factors, order=1)


def _scene_resize(scene: Scene, new_size: Any) -> Scene:
    values = np.asarray(new_size, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scene resize requires a target size.")
    if values.size == 1:
        values = np.repeat(values, 2)
    new_rows = max(int(np.rint(values[0])), 1)
    new_cols = max(int(np.rint(values[1])), 1)

    resized = scene.clone()
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    resized = scene_set(resized, "photons", _scene_resize_array(photons, new_rows, new_cols))

    depth_map = scene.fields.get("depth_map_m")
    if depth_map is not None:
        resized.fields["depth_map_m"] = _scene_resize_array(np.asarray(depth_map, dtype=float), new_rows, new_cols)

    illuminant_photons = scene.fields.get("illuminant_photons")
    if illuminant_photons is not None:
        illuminant_array = np.asarray(illuminant_photons, dtype=float)
        if illuminant_array.ndim >= 2:
            resized.fields["illuminant_photons"] = _scene_resize_array(illuminant_array, new_rows, new_cols)

    return resized


def scene_combine(scene1: Scene, scene2: Scene, *args: Any) -> Scene:
    direction = "horizontal"
    if args:
        if len(args) != 2 or param_format(args[0]) != "direction":
            raise ValueError("scene_combine expects MATLAB-style 'direction', value arguments.")
        direction = str(args[1]).lower()
    normalized = param_format(direction)

    wave1 = np.asarray(scene_get(scene1, "wave"), dtype=float)
    wave2 = np.asarray(scene_get(scene2, "wave"), dtype=float)
    if not np.array_equal(wave1, wave2):
        raise ValueError("scene_combine requires matching wavelength samples.")

    if normalized == "horizontal":
        if int(scene_get(scene1, "rows")) != int(scene_get(scene2, "rows")):
            raise ValueError("Horizontal scene_combine requires matching row counts.")
        photons = np.concatenate(
            (
                np.asarray(scene_get(scene1, "photons"), dtype=float),
                np.asarray(scene_get(scene2, "photons"), dtype=float),
            ),
            axis=1,
        )
        combined = scene_set(scene1.clone(), "photons", photons)
        return scene_set(
            combined,
            "hfov",
            float(scene_get(scene1, "fov")) + float(scene_get(scene2, "fov")),
        )

    if normalized == "vertical":
        if int(scene_get(scene1, "cols")) != int(scene_get(scene2, "cols")):
            raise ValueError("Vertical scene_combine requires matching column counts.")
        photons = np.concatenate(
            (
                np.asarray(scene_get(scene1, "photons"), dtype=float),
                np.asarray(scene_get(scene2, "photons"), dtype=float),
            ),
            axis=0,
        )
        return scene_set(scene1.clone(), "photons", photons)

    if normalized == "both":
        return scene_combine(scene_combine(scene1, scene2, "direction", "horizontal"), scene_combine(scene1, scene2, "direction", "horizontal"), "direction", "vertical")

    if normalized == "centered":
        scene_mid = scene_combine(scene2, scene_combine(scene1, scene2, "direction", "horizontal"), "direction", "horizontal")
        scene_edge = scene_combine(scene_combine(scene2, scene2, "direction", "horizontal"), scene2, "direction", "horizontal")
        return scene_combine(scene_combine(scene_edge, scene_mid, "direction", "vertical"), scene_edge, "direction", "vertical")

    raise ValueError(f"Unsupported scene_combine direction: {direction}")


def _scene_rotation_degrees(value: Any) -> float:
    if isinstance(value, str):
        normalized = param_format(value)
        if normalized in {"cw", "clockwise"}:
            return -90.0
        if normalized in {"ccw", "counterclockwise"}:
            return 90.0
        raise ValueError(f"Unsupported scene rotation parameter: {value}")
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def scene_rotate(scene: Scene, degrees: Any) -> Scene:
    angle_deg = _scene_rotation_degrees(degrees)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    if photons.ndim != 3:
        raise ValueError("scene_rotate requires a scene photons cube.")

    rotated = scene.clone()
    rotated_photons = rotate(
        photons,
        angle_deg,
        axes=(1, 0),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated = scene_set(rotated, "photons", rotated_photons)

    if param_format(scene_get(scene, "illuminant format")) == "spatialspectral":
        illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float)
        if illuminant.ndim < 2:
            raise ValueError("Spatial-spectral illuminant photons must be at least 2-D.")
        rotated.fields["illuminant_photons"] = rotate(
            illuminant,
            angle_deg,
            axes=(1, 0),
            reshape=True,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        _invalidate_scene_caches(rotated)

    return rotated


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


def _parse_image_size(size: Any) -> tuple[int, int]:
    return _scene_size_2d(size, default=32)


def _matlab_round_scalar(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _matlab_round(values: Any) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    return np.sign(numeric) * np.floor(np.abs(numeric) + 0.5)


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
    size: Any,
    wave: np.ndarray,
    illuminant_energy: np.ndarray,
    illuminant_comment: str | None = None,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _parse_image_size(size)
    scene = Scene(name=name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, wave)
    if illuminant_comment is not None:
        scene.fields["illuminant_comment"] = str(illuminant_comment)
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    photons = np.broadcast_to(scene.fields["illuminant_photons"], (rows, cols, wave.size)).copy()
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
    return _uniform_scene(
        f"Uniform BB {int(round(float(temperature_k)))}K",
        _scene_size_2d(size, default=32)[0],
        wave,
        illuminant_energy,
        illuminant_comment=f"blackbody-{int(round(float(temperature_k)))}K",
        asset_store=asset_store,
    )


def _uniform_monochromatic_scene(
    size: Any,
    wavelength: Any,
    *,
    asset_store: AssetStore,
) -> Scene:
    wave = _wave_or_default(wavelength)
    scene = _uniform_scene(
        "Narrow Band",
        _scene_size_2d(size, default=128)[0],
        wave,
        np.ones(wave.size, dtype=float),
        illuminant_comment="equal-energy",
        asset_store=asset_store,
    )
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


def _star_pattern_scene(
    image_size: int,
    spectral_type: str,
    n_lines: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    im_size = max(int(image_size), 1)
    n_rays = max(int(n_lines), 1)
    radians = np.pi * np.arange(n_rays, dtype=float) / float(n_rays)
    end_points = np.column_stack((np.cos(radians), np.sin(radians))) * (im_size / 2.0)
    end_points = _matlab_round(end_points).astype(int)

    image = np.zeros((im_size, im_size), dtype=float)
    center = im_size / 2.0
    for x, y in end_points:
        u = -int(x)
        v = -int(y)
        if x > 0:
            x, y, u, v = u, v, int(x), int(y)
        if u != x:
            slope = (float(y) - float(v)) / (float(u) - float(x))
            jj_values = np.arange(float(x), float(u) + 0.2001, 0.2, dtype=float)
            kk_values = _matlab_round(jj_values * slope).astype(int)
            row_idx = _matlab_round(kk_values + center).astype(int)
            col_idx = _matlab_round(jj_values + center).astype(int)
            valid = (
                (row_idx >= 0)
                & (row_idx < im_size)
                & (col_idx >= 0)
                & (col_idx < im_size)
            )
            image[row_idx[valid], col_idx[valid]] = 1.0
        else:
            image[:, int(im_size / 2)] = 1.0

    image[image == 0.0] = 1e-4
    image = image / max(float(np.max(image)), 1e-12)

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = image[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name=f"radialLine-{param_format(spectral_type)}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


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


def _normalized_parameter_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        raw = value
    elif hasattr(value, "items"):
        raw = dict(value.items())
    elif hasattr(value, "__dict__"):
        raw = vars(value)
    else:
        return {"imagesize": value}
    return {param_format(str(key)): item for key, item in raw.items()}


def _broadcast_parameter_vector(values: np.ndarray, count: int, name: str) -> np.ndarray:
    if values.size == count:
        return values.astype(float, copy=False)
    if values.size == 1:
        return np.repeat(values.astype(float, copy=False), count)
    raise ValueError(f"Harmonic parameter '{name}' must be scalar or length {count}.")


def _scale_range(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    current = np.asarray(values, dtype=float)
    current_min = float(np.min(current))
    current_max = float(np.max(current))
    if current_max <= current_min:
        return np.full_like(current, float(out_min), dtype=float)
    scaled = (current - current_min) / (current_max - current_min)
    return scaled * (float(out_max) - float(out_min)) + float(out_min)


def _harmonic_parameters(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    default = {
        "name": "harmonicP",
        "ang": np.array([0.0], dtype=float),
        "contrast": np.array([1.0], dtype=float),
        "freq": np.array([1.0], dtype=float),
        "ph": np.array([np.pi / 2.0], dtype=float),
        "row": 65,
        "col": 65,
        "center": np.array([0.0, 0.0], dtype=float),
        "gaborflag": 0.0,
    }

    imagesize = normalized.get("imagesize")
    row = int(np.rint(normalized.get("row", default["row"])))
    col = int(np.rint(normalized.get("col", default["col"])))
    if imagesize is not None:
        image_size = np.asarray(imagesize, dtype=float).reshape(-1)
        if image_size.size == 1:
            row = int(np.rint(image_size[0]))
            col = int(np.rint(image_size[0]))
        elif image_size.size >= 2:
            row = int(np.rint(image_size[0]))
            col = int(np.rint(image_size[1]))

    angle = np.asarray(normalized.get("orientation", normalized.get("ang", default["ang"])), dtype=float).reshape(-1)
    contrast = np.asarray(normalized.get("contrast", default["contrast"]), dtype=float).reshape(-1)
    frequency = np.asarray(normalized.get("frequency", normalized.get("freq", default["freq"])), dtype=float).reshape(-1)
    phase = np.asarray(normalized.get("phase", normalized.get("ph", default["ph"])), dtype=float).reshape(-1)
    count = max(angle.size, contrast.size, frequency.size, phase.size)

    center = np.asarray(normalized.get("center", default["center"]), dtype=float).reshape(-1)
    if center.size == 1:
        center = np.repeat(center, 2)
    if center.size != 2:
        raise ValueError("Harmonic 'center' must contain one or two values.")

    return {
        "name": str(normalized.get("name", default["name"])),
        "ang": _broadcast_parameter_vector(angle, count, "ang"),
        "contrast": _broadcast_parameter_vector(contrast, count, "contrast"),
        "freq": _broadcast_parameter_vector(frequency, count, "freq"),
        "ph": _broadcast_parameter_vector(phase, count, "ph"),
        "row": max(row, 1),
        "col": max(col, 1),
        "center": center.astype(float, copy=False),
        "gaborflag": float(normalized.get("gaborflag", default["gaborflag"])),
    }


def _image_harmonic(params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    row = int(params["row"])
    col = int(params["col"])
    center = np.asarray(params["center"], dtype=float)

    x = np.arange(col, dtype=float) / max(col, 1)
    y = np.arange(row, dtype=float) / max(row, 1)
    x = x - (x[-1] / 2.0)
    y = y - (y[-1] / 2.0)
    x = x - center[0] / max(col, 1)
    y = y - center[1] / max(row, 1)
    xx, yy = np.meshgrid(x, y)

    x_pixels = np.arange(col, dtype=float) - ((col - 1) / 2.0) - center[0]
    y_pixels = np.arange(row, dtype=float) - ((row - 1) / 2.0) - center[1]
    xx_pixels, yy_pixels = np.meshgrid(x_pixels, y_pixels)

    gabor_flag = float(params["gaborflag"])
    if gabor_flag > 0.0:
        sigma = gabor_flag * min(row, col)
        window = np.exp(-((xx_pixels**2 + yy_pixels**2) / (2.0 * sigma**2)))
        window /= max(float(np.max(window)), 1e-12)
    elif gabor_flag < 0.0:
        radius = -gabor_flag * min(row, col)
        x_arg = np.pi * xx_pixels / (2.0 * radius)
        y_arg = np.pi * yy_pixels / (2.0 * radius)
        window = np.cos(x_arg) * np.cos(y_arg)
        mask = (np.abs(x_arg) <= (np.pi / 2.0)) & (np.abs(y_arg) <= (np.pi / 2.0))
        window = np.where(mask, window, 0.0)
        window /= max(float(np.max(window)), 1e-12)
    else:
        window = np.ones((row, col), dtype=float)

    image = np.zeros((row, col), dtype=float)
    count = len(params["freq"])
    for ii in range(count):
        image += (
            params["contrast"][ii]
            * window
            * np.cos(
                2.0
                * np.pi
                * params["freq"][ii]
                * (np.cos(params["ang"][ii]) * xx + np.sin(params["ang"][ii]) * yy)
                + params["ph"][ii]
            )
            + 1.0
        )
    image /= max(count, 1)
    return image, params


def _frequency_orientation_parameters(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    angles = np.asarray(normalized.get("angles", np.linspace(0.0, np.pi / 2.0, 8)), dtype=float).reshape(-1)
    freqs = np.asarray(normalized.get("freqs", np.arange(1.0, 9.0, dtype=float)), dtype=float).reshape(-1)
    contrast = float(normalized.get("contrast", 1.0))
    block_size = normalized.get("blocksize")
    if block_size is None:
        requested_size = normalized.get("imagesize")
        if requested_size is None:
            block_size = 32
        else:
            requested = np.asarray(requested_size, dtype=float).reshape(-1)
            if requested.size == 1:
                block_size = max(int(np.rint(requested[0] / max(len(angles), len(freqs)))), 1)
            else:
                block_size = max(int(np.rint(min(requested[0] / max(len(freqs), 1), requested[1] / max(len(angles), 1)))), 1)
    return {
        "angles": angles,
        "freqs": freqs,
        "contrast": contrast,
        "blocksize": max(int(np.rint(block_size)), 1),
    }


def _frequency_orientation_image(params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    block_size = int(params["blocksize"])
    x = np.arange(block_size, dtype=float) / block_size
    xx, yy = np.meshgrid(x, x)
    rows: list[np.ndarray] = []
    for frequency in params["freqs"]:
        blocks: list[np.ndarray] = []
        for angle in params["angles"]:
            block = 0.5 * (
                1.0
                + params["contrast"]
                * np.sin(2.0 * np.pi * frequency * (np.cos(angle) * xx + np.sin(angle) * yy))
            )
            blocks.append(block)
        rows.append(np.concatenate(blocks, axis=1))
    image = np.concatenate(rows, axis=0).T
    return image, params


def _sweep_frequency_image(
    size: Any,
    max_frequency: float,
    y_contrast: Any | None = None,
) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    x = np.arange(1.0, cols + 1.0, dtype=float) / cols
    frequency = (x**2) * float(max_frequency)
    x_image = np.sin(2.0 * np.pi * (frequency * x))
    if y_contrast is None:
        contrast = np.linspace(1.0, 0.0, rows, dtype=float)
    else:
        contrast = np.asarray(y_contrast, dtype=float).reshape(-1)
        if contrast.size == 1:
            contrast = np.repeat(contrast, rows)
        elif contrast.size != rows:
            raise ValueError("Sweep-frequency yContrast must be scalar or length rows.")
    image = contrast[:, None] * x_image[None, :] + 0.5
    image = _scale_range(image, 1.0, 256.0)
    return image / max(float(np.max(image)), 1e-12)


def _linear_intensity_ramp_image(size: Any, dynamic_range: float) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    max_column = _matlab_round_scalar(cols / 2.0)
    min_column = -(max_column - 1)
    x_image = np.arange(min_column, min_column + cols, dtype=float)
    y_contrast = np.arange(rows, 0, -1, dtype=float) / max(rows, 1)
    image = (y_contrast[:, None] * x_image[None, :]) + 0.5
    image = _scale_range(image, 1.0, max(float(dynamic_range), 1.0))
    return image / max(float(np.max(image)), 1e-12)


def _exponential_intensity_ramp_image(size: Any, dynamic_range: float) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    if float(dynamic_range) <= 1.0:
        ramp = np.ones(cols, dtype=float)
    else:
        ramp = np.logspace(0.0, np.log10(float(dynamic_range)), cols, dtype=float)
    image = np.tile(ramp.reshape(1, -1), (rows, 1))
    image = _scale_range(image, 1.0, max(float(dynamic_range), 1.0))
    return image / max(float(np.max(image)), 1e-12)


def _equal_photon_pattern_scene(
    name: str,
    image: np.ndarray,
    wave: np.ndarray,
    *,
    fov_deg: float,
    asset_store: AssetStore,
    illuminant_comment: str = "equal photons",
) -> Scene:
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    scene = Scene(name=name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = illuminant_comment
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = float(fov_deg)
    scene.data["photons"] = np.asarray(image, dtype=float)[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _default_reflectance_chart_files() -> list[str]:
    return [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "skin/HyspexSkinReflectance.mat",
    ]


def _reflectance_chart_sources(value: Any | None) -> list[Any]:
    if value is None:
        return _default_reflectance_chart_files()
    if isinstance(value, (str, Path, np.ndarray)):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value.copy()
    return [value]


def _load_reflectance_source(
    source: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists():
            data = asset_store.load_mat(source_path)
        else:
            data = asset_store.load_mat(Path("data/surfaces/reflectances") / source_path)
        reflectance = np.asarray(data["data"], dtype=float)
        source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        if reflectance.ndim == 1:
            reflectance = reflectance.reshape(-1, 1)
        if not np.array_equal(source_wave, wave):
            from .utils import interp_spectra

            reflectance = interp_spectra(source_wave, reflectance, wave)
        return np.asarray(reflectance, dtype=float)
    reflectance = np.asarray(source, dtype=float)
    if reflectance.ndim == 1:
        reflectance = reflectance.reshape(-1, 1)
    if reflectance.shape[0] != wave.size:
        raise ValueError("Reflectance matrices must be provided on the target wavelength grid.")
    return reflectance


def _reflectance_sample_lists(
    reflectance_sets: list[np.ndarray],
    sample_spec: Any,
    sampling: str,
) -> list[np.ndarray]:
    normalized_sampling = param_format(sampling)
    with_replacement = normalized_sampling.startswith("r")
    use_all = normalized_sampling == "all"

    if isinstance(sample_spec, (list, tuple)) and len(sample_spec) == len(reflectance_sets):
        explicit = []
        is_explicit = True
        for item in sample_spec:
            if np.isscalar(item):
                is_explicit = False
                break
            values = np.asarray(item, dtype=int).reshape(-1)
            if values.size == 0:
                explicit.append(values)
                continue
            explicit.append(values)
        if is_explicit:
            return explicit

    if sample_spec is None:
        counts = [24] * len(reflectance_sets)
    else:
        counts_array = np.asarray(sample_spec, dtype=int).reshape(-1)
        if counts_array.size != len(reflectance_sets):
            raise ValueError("Reflectance chart sample counts must match the number of reflectance sources.")
        counts = counts_array.tolist()

    rng = np.random.default_rng(0)
    sample_lists: list[np.ndarray] = []
    for count, reflectance in zip(counts, reflectance_sets, strict=True):
        n_reflectances = reflectance.shape[1]
        if use_all:
            sample_lists.append(np.arange(1, n_reflectances + 1, dtype=int))
            continue
        count = int(count)
        if with_replacement:
            sample_lists.append(rng.integers(1, n_reflectances + 1, size=count, endpoint=False, dtype=int))
        else:
            if count > n_reflectances:
                raise ValueError("Requested more reflectance samples than available without replacement.")
            perm = rng.permutation(n_reflectances)[:count]
            sample_lists.append(np.asarray(perm + 1, dtype=int))
    return sample_lists


def _reflectance_chart_parameters(
    value: Any | None,
    *,
    wave: np.ndarray | None = None,
) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    files = _reflectance_chart_sources(normalized.get("sfiles"))
    samples = normalized.get("ssamples", np.array([50, 40, 10], dtype=int))
    patch_size = int(np.rint(normalized.get("psize", 24)))
    chart_wave = _wave_or_default(normalized.get("wave", wave))
    gray_flag = bool(normalized.get("grayflag", 1))
    sampling = str(normalized.get("sampling", "r"))
    return {
        "sfiles": files,
        "ssamples": samples,
        "psize": max(patch_size, 1),
        "wave": chart_wave,
        "grayflag": gray_flag,
        "sampling": sampling,
    }


def _reflectance_chart_scene(
    source_files: list[Any],
    sample_spec: Any,
    patch_size: int,
    wave: np.ndarray,
    gray_flag: bool,
    sampling: str,
    *,
    asset_store: AssetStore,
) -> Scene:
    reflectance_sets = [_load_reflectance_source(source, wave, asset_store=asset_store) for source in source_files]
    sample_lists = _reflectance_sample_lists(reflectance_sets, sample_spec, sampling)
    sampled_blocks = []
    for reflectance, sample_list in zip(reflectance_sets, sample_lists, strict=True):
        if sample_list.size == 0:
            continue
        sampled_blocks.append(reflectance[:, sample_list.astype(int) - 1])
    if sampled_blocks:
        reflectances = np.concatenate(sampled_blocks, axis=1)
    else:
        reflectances = np.zeros((wave.size, 0), dtype=float)

    n_samples = reflectances.shape[1]
    rows = int(np.ceil(np.sqrt(n_samples))) if n_samples > 0 else 1
    cols = int(np.ceil(n_samples / max(rows, 1))) if n_samples > 0 else 1
    if gray_flag:
        gray_strip = np.ones((wave.size, rows), dtype=float) * np.logspace(0.0, np.log10(0.05), rows, dtype=float)
        reflectances = np.concatenate((reflectances, gray_strip), axis=1)
        cols += 1

    illuminant_energy, illuminant_photons = _spectral_illuminant("ee", wave, asset_store=asset_store)
    radiance = reflectances * illuminant_photons.reshape(-1, 1)
    patch_cube = np.zeros((rows, cols, wave.size), dtype=float)
    index_map = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            idx = row + col * rows
            if idx < radiance.shape[1]:
                patch_cube[row, col, :] = radiance[:, idx]
                index_map[row, col] = idx + 1
            else:
                patch_cube[row, col, :] = 0.2 * illuminant_photons

    xyz = xyz_from_energy(
        quanta_to_energy(patch_cube.reshape(-1, wave.size), wave),
        wave,
        asset_store=asset_store,
    ).reshape(rows, cols, 3)

    photons = np.repeat(np.repeat(patch_cube, patch_size, axis=0), patch_size, axis=1)
    scene = Scene(name="Reflectance Chart (EE)")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = "Equal energy"
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.fields["chart_parameters"] = {
        "sFiles": [str(item) for item in source_files],
        "sSamples": [np.asarray(item, dtype=int).copy() for item in sample_lists],
        "grayFlag": bool(gray_flag),
        "sampling": str(sampling),
        "pSize": int(patch_size),
        "wave": np.asarray(wave, dtype=float).copy(),
        "XYZ": xyz.copy(),
        "rowcol": np.array([rows, cols], dtype=int),
        "rIdxMap": np.repeat(np.repeat(index_map, patch_size, axis=0), patch_size, axis=1),
    }
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def scene_create(
    scene_name: str = "default",
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene:
    """Create a supported milestone-one scene."""

    store = _store(asset_store)
    name = param_format(scene_name)

    if name in {"default", "macbeth", "macbethd65"}:
        patch_size = int(args[0]) if len(args) > 0 else 16
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = str(args[2]) if len(args) > 2 else "macbethChart.mat"
        black_border = bool(args[3]) if len(args) > 3 else False
        return track_session_object(
            session,
            _create_macbeth_scene(patch_size, wave, surface_file, black_border, asset_store=store),
        )

    if name == "empty":
        scene = _create_macbeth_scene(16, _wave_or_default(None), "macbethChart.mat", False, asset_store=store)
        return track_session_object(session, scene_clear_data(scene))

    if name in {"uniformd65", "uniform d65".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        _, illuminant_energy = _load_d65(wave, store)
        return track_session_object(
            session,
            _uniform_scene("Uniform D65", size, wave, illuminant_energy, illuminant_comment="D65.mat", asset_store=store),
        )

    if name in {"uniform", "uniformee", "uniformequalenergy"}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return track_session_object(
            session,
            _uniform_scene(
                "Uniform EE",
                size,
                wave,
                np.ones(wave.size, dtype=float),
                illuminant_comment="equal-energy",
                asset_store=store,
            ),
        )

    if name in {"uniformep", "uniformequalphoton", "uniformequalphotons"}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        illuminant_photons = np.ones(wave.size, dtype=float)
        illuminant_energy = quanta_to_energy(illuminant_photons, wave)
        return track_session_object(
            session,
            _uniform_scene(
                "Uniform EP",
                size,
                wave,
                illuminant_energy,
                illuminant_comment="equal-photons",
                asset_store=store,
            ),
        )

    if name in {"uniformbb", "uniformblackbody"}:
        size = args[0] if len(args) > 0 else 32
        temperature_k = float(args[1]) if len(args) > 1 else 5000.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _uniform_blackbody_scene(size, temperature_k, wave, asset_store=store))

    if name in {"uniformmonochromatic", "narrowband"}:
        wavelength = args[0] if len(args) > 0 else 500.0
        size = args[1] if len(args) > 1 else 128
        return track_session_object(session, _uniform_monochromatic_scene(size, wavelength, asset_store=store))

    if name in {"reflectancechart", "reflectance"}:
        if args and isinstance(args[0], dict):
            params = _reflectance_chart_parameters(args[0])
        elif len(args) == 1 and hasattr(args[0], "items"):
            params = _reflectance_chart_parameters(args[0])
        else:
            params = _reflectance_chart_parameters(None)
            if len(args) > 0:
                params["psize"] = max(int(np.rint(args[0])), 1)
            if len(args) > 1:
                params["ssamples"] = args[1]
            if len(args) > 2:
                params["sfiles"] = _reflectance_chart_sources(args[2])
            if len(args) > 3:
                params["wave"] = _wave_or_default(args[3])
            if len(args) > 4:
                params["grayflag"] = bool(args[4])
            if len(args) > 5:
                params["sampling"] = str(args[5])
        return track_session_object(
            session,
            _reflectance_chart_scene(
                params["sfiles"],
                params["ssamples"],
                params["psize"],
                params["wave"],
                params["grayflag"],
                params["sampling"],
                asset_store=store,
            ),
        )

    if name in {"line", "lined65", "impulse1dd65"}:
        size = args[0] if len(args) > 0 else 64
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return track_session_object(session, _line_scene("d65", size, 0, wave, asset_store=store))

    if name in {"lineee", "impulse1dee"}:
        size = args[0] if len(args) > 0 else 64
        offset = int(args[1]) if len(args) > 1 else 0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _line_scene("ee", size, offset, wave, asset_store=store))

    if name in {"lineequalphoton", "lineep"}:
        size = args[0] if len(args) > 0 else 64
        offset = int(args[1]) if len(args) > 1 else 0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _line_scene("ep", size, offset, wave, asset_store=store))

    if name in {"frequencyorientation", "demosaictarget", "freqorientpattern", "freqorient"}:
        params = _frequency_orientation_parameters(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        image, params = _frequency_orientation_image(params)
        image = np.clip(image, 1e-4, 1.0)
        image = image / max(float(np.max(image)), 1e-12)
        scene = _equal_photon_pattern_scene("FOTarget", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store)
        scene.fields["frequency_orientation_params"] = params
        return track_session_object(session, scene)

    if name in {"harmonic", "sinusoid"}:
        params = _harmonic_parameters(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        image, params = _image_harmonic(params)
        image = np.where(image == 0.0, 1e-4, image)
        image = image / (2.0 * max(float(np.max(image)), 1e-12))
        scene = _equal_photon_pattern_scene("harmonic", image, wave, fov_deg=1.0, asset_store=store)
        scene.fields["harmonic_params"] = params
        return track_session_object(session, scene)

    if name in {"sweep", "sweepfrequency"}:
        size = args[0] if len(args) > 0 else 128
        if np.isscalar(size):
            default_max_frequency = float(size) / 16.0
        else:
            size_vec = np.asarray(size, dtype=float).reshape(-1)
            default_max_frequency = float(size_vec[1] if size_vec.size > 1 else size_vec[0]) / 16.0
        max_frequency = float(args[1]) if len(args) > 1 else default_max_frequency
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        y_contrast = args[3] if len(args) > 3 else None
        image = _sweep_frequency_image(size, max_frequency, y_contrast)
        scene = _equal_photon_pattern_scene("sweep", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store)
        scene.fields["sweep_params"] = {
            "size": _scene_size_2d(size, default=128),
            "max_frequency": max_frequency,
            "wave": wave.copy(),
        }
        return track_session_object(session, scene)

    if name in {"linearintensityramp", "linearramp"}:
        size = args[0] if len(args) > 0 else 128
        dynamic_range = float(args[1]) if len(args) > 1 else 256.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        image = _linear_intensity_ramp_image(size, dynamic_range)
        return track_session_object(
            session,
            _equal_photon_pattern_scene("linearRamp", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store),
        )

    if name in {"exponentialintensityramp", "expintensityramp", "expramp"}:
        size = args[0] if len(args) > 0 else 128
        dynamic_range = float(args[1]) if len(args) > 1 else 256.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        image = _exponential_intensity_ramp_image(size, dynamic_range)
        return track_session_object(
            session,
            _equal_photon_pattern_scene("expRamp", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store),
        )

    if name in {"starpattern", "radiallines"}:
        image_size = int(args[0]) if len(args) > 0 else 256
        spectral_type = str(args[1]) if len(args) > 1 else "ep"
        n_lines = int(args[2]) if len(args) > 2 else 8
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        return track_session_object(
            session,
            _star_pattern_scene(image_size, spectral_type, n_lines, wave, asset_store=store),
        )

    if name == "bar":
        size = args[0] if len(args) > 0 else 64
        width = int(args[1]) if len(args) > 1 else 3
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _bar_scene(size, width, wave, asset_store=store))

    if name in {"whitenoise", "noise"}:
        size = args[0] if len(args) > 0 else 128
        contrast = float(args[1]) if len(args) > 1 else 20.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _white_noise_scene(size, contrast, wave, asset_store=store))

    if name in {"pointarray", "manypoints", "point array".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = int(args[1]) if len(args) > 1 else 16
        spectral_type = str(args[2]) if len(args) > 2 else "ep"
        point_size = int(args[3]) if len(args) > 3 else 1
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return track_session_object(
            session,
            _point_array_scene(size, spacing, spectral_type, point_size, wave, asset_store=store),
        )

    if name in {"gridlines", "distortiongrid", "grid lines".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = int(args[1]) if len(args) > 1 else 16
        spectral_type = str(args[2]) if len(args) > 2 else "ep"
        thickness = int(args[3]) if len(args) > 3 else 1
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return track_session_object(
            session,
            _grid_lines_scene(size, spacing, spectral_type, thickness, wave, asset_store=store),
        )

    if name == "checkerboard":
        pixels_per_check = int(args[0]) if len(args) > 0 else 16
        number_of_checks = int(args[1]) if len(args) > 1 else 8
        spectral_type = str(args[2]) if len(args) > 2 and isinstance(args[2], str) else "ep"
        wave_arg = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not isinstance(args[2], str) else None)
        wave = _wave_or_default(wave_arg)
        return track_session_object(
            session,
            _checkerboard_scene(
                pixels_per_check,
                number_of_checks,
                wave,
                spectral_type,
                asset_store=store,
            ),
        )

    if name in {"slantedbar", "slanted bar".replace(" ", "")}:
        image_size = int(args[0]) if len(args) > 0 else 256
        edge_slope = float(args[1]) if len(args) > 1 else 0.5
        fov_deg = float(args[2]) if len(args) > 2 else 2.0
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        dark_level = float(args[4]) if len(args) > 4 else 0.0
        return track_session_object(
            session,
            _slanted_bar_scene(image_size, edge_slope, wave, fov_deg, dark_level, asset_store=store),
        )

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
    session: SessionContext | None = None,
) -> Scene:
    """Create a scene from RGB, monochrome, or multispectral image data."""

    store = _store(asset_store)
    normalized_type = param_format(im_type)
    if normalized_type not in {"rgb", "monochrome", "unispectral", "spectral", "multispectral", "hyperspectral"}:
        raise UnsupportedOptionError("sceneFromFile", im_type)

    requested_wave = None if wave is None else _wave_or_default(wave)

    if normalized_type in {"spectral", "multispectral", "hyperspectral"}:
        multispectral = _multispectral_scene_input(input_data, requested_wave, asset_store=store)
        scene = Scene(name=str(multispectral["source_name"]))
        scene.fields["wave"] = np.asarray(multispectral["wave"], dtype=float)
        scene.fields["illuminant_format"] = str(multispectral["illuminant_format"])
        scene.fields["illuminant_energy"] = np.asarray(multispectral["illuminant_energy"], dtype=float)
        scene.fields["illuminant_photons"] = np.asarray(multispectral["illuminant_photons"], dtype=float)
        scene.fields["illuminant_comment"] = str(multispectral["illuminant_comment"])
        scene.fields["distance_m"] = float(multispectral["distance_m"])
        scene.fields["fov_deg"] = float(multispectral["fov_deg"])
        scene.fields["filename"] = str(multispectral["filename"])
        scene.fields["source_type"] = "multispectral"
        scene.data["photons"] = np.asarray(multispectral["photons"], dtype=float)
        _update_scene_geometry(scene)
        if mean_luminance is not None:
            scene = scene_adjust_luminance(scene, float(mean_luminance), asset_store=store)
        return track_session_object(session, scene)

    del scale_reflectance
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
    return track_session_object(session, scene)


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


def scene_interpolate_w(
    scene: Scene,
    wave: Any,
    preserve_luminance: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    source_wave = np.asarray(scene.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    target_wave = np.asarray(wave, dtype=float).reshape(-1)

    if target_wave.size == 0:
        raise ValueError("Target wavelength samples must not be empty.")
    if source_wave.size == 0 or np.array_equal(source_wave, target_wave):
        scene.fields["wave"] = target_wave
        _invalidate_scene_caches(scene)
        return scene
    if float(np.min(target_wave)) < float(np.min(source_wave)) or float(np.max(target_wave)) > float(np.max(source_wave)):
        raise ValueError("sceneInterpolateW does not support extrapolation outside the current wavelength support.")

    store = _store(asset_store)
    scene_photons = np.asarray(scene.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    original_mean: float | None = None
    if scene_photons.size > 0:
        if preserve_luminance:
            original_mean = float(scene_get(scene, "mean luminance", asset_store=store))
        scene.data["photons"] = _resample_wave_last(scene_photons, source_wave, target_wave)

    illuminant_photons = scene.fields.get("illuminant_photons")
    illuminant_energy = scene.fields.get("illuminant_energy")
    if illuminant_photons is not None:
        illuminant_array = np.asarray(illuminant_photons, dtype=float)
        if illuminant_array.ndim == 1:
            resampled_illuminant = np.asarray(interp_spectra(source_wave, illuminant_array, target_wave), dtype=float).reshape(-1)
        else:
            resampled_illuminant = _resample_wave_last(illuminant_array, source_wave, target_wave)
        scene.fields["illuminant_photons"] = resampled_illuminant
        energy = np.asarray(quanta_to_energy(resampled_illuminant, target_wave), dtype=float)
        if energy.ndim == 3:
            energy = np.mean(energy, axis=(0, 1))
        scene.fields["illuminant_energy"] = energy
    elif illuminant_energy is not None:
        illuminant_array = np.asarray(illuminant_energy, dtype=float)
        if illuminant_array.ndim == 1:
            resampled_energy = np.asarray(interp_spectra(source_wave, illuminant_array, target_wave), dtype=float).reshape(-1)
        else:
            resampled_energy = _resample_wave_last(illuminant_array, source_wave, target_wave)
            if resampled_energy.ndim == 3:
                resampled_energy = np.mean(resampled_energy, axis=(0, 1))
        scene.fields["illuminant_energy"] = np.asarray(resampled_energy, dtype=float)
        scene.fields["illuminant_photons"] = np.asarray(energy_to_quanta(resampled_energy, target_wave), dtype=float)

    scene.fields["wave"] = target_wave
    _invalidate_scene_caches(scene)
    if original_mean is not None:
        scene_adjust_luminance(scene, original_mean, asset_store=store)
    return scene


def _scene_roi_xyz(scene: Scene, roi_locs: Any | None = None, *, asset_store: AssetStore | None = None) -> np.ndarray:
    store = _store(asset_store)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    if roi_locs is None:
        photons = np.asarray(scene.data["photons"], dtype=float).reshape(-1, wave.size)
    else:
        photons = np.asarray(scene_get(scene, "roi photons", roi_locs, asset_store=store), dtype=float)
    energy = quanta_to_energy(photons, wave)
    return xyz_from_energy(energy, wave, asset_store=store)


def _scene_spatial_support_linear(scene: Scene, unit: Any | None = None) -> dict[str, np.ndarray]:
    rows, cols = scene_get(scene, "size")
    if rows <= 0 or cols <= 0:
        return {"x": np.empty(0, dtype=float), "y": np.empty(0, dtype=float)}
    width_m = float(scene.fields["width_m"])
    height_m = float(scene.fields["height_m"])
    dx = width_m / cols
    dy = height_m / rows
    x = ((np.arange(cols, dtype=float) + 0.5) - (cols / 2.0)) * dx
    y = ((np.arange(rows, dtype=float) + 0.5) - (rows / 2.0)) * dy
    scale = _spatial_unit_scale(unit)
    return {"x": x * scale, "y": y * scale}


def _line_index(line_arg: Any, orientation: str) -> int:
    values = np.asarray(line_arg, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Line location must be a scalar or [col, row] locator.")
    if values.size == 1:
        return int(np.rint(values[0]))
    return int(np.rint(values[1 if orientation == "h" else 0]))


def _scene_line_profile(
    scene: Scene,
    data_type: str,
    orientation: str,
    line_arg: Any,
    *,
    unit: Any | None = "mm",
    asset_store: AssetStore | None = None,
) -> dict[str, np.ndarray]:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    support = _scene_spatial_support_linear(scene, unit)
    line_index = _line_index(line_arg, orientation)
    if data_type == "photons":
        data = np.asarray(scene.data["photons"], dtype=float)
    elif data_type == "illuminant_photons":
        illuminant = np.asarray(scene_get(scene, "illuminant photons", asset_store=asset_store), dtype=float).reshape(1, 1, -1)
        rows, cols = scene_get(scene, "size")
        data = np.broadcast_to(illuminant, (rows, cols, illuminant.shape[2])).copy()
    elif data_type == "illuminant_energy":
        illuminant = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float).reshape(1, 1, -1)
        rows, cols = scene_get(scene, "size")
        data = np.broadcast_to(illuminant, (rows, cols, illuminant.shape[2])).copy()
    elif data_type == "luminance":
        data = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    else:
        raise KeyError(f"Unsupported scene line profile data type: {data_type}")

    if orientation == "h":
        if line_index < 1 or line_index > data.shape[0]:
            raise IndexError("Horizontal scene line index is out of range.")
        pos = np.asarray(support["x"], dtype=float)
        line = np.asarray(data[line_index - 1, ...], dtype=float)
    else:
        if line_index < 1 or line_index > data.shape[1]:
            raise IndexError("Vertical scene line index is out of range.")
        pos = np.asarray(support["y"], dtype=float)
        line = np.asarray(data[:, line_index - 1, ...], dtype=float)

    if line.ndim == 1:
        return {"pos": pos.copy(), "data": line.copy(), "unit": str(unit or "m")}
    return {"pos": pos.copy(), "wave": wave.copy(), "data": line.T.copy(), "unit": str(unit or "m")}


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
    if key == "spatialsupportlinear":
        return _scene_spatial_support_linear(scene, args[0] if args else None)
    if key == "spatialsupport":
        support = _scene_spatial_support_linear(scene, args[0] if args else None)
        xx, yy = np.meshgrid(support["x"], support["y"])
        return np.stack((xx, yy), axis=2)
    if key == "illuminantformat":
        return scene.fields.get("illuminant_format", "spectral")
    if key == "illuminantcomment":
        return scene.fields.get("illuminant_comment")
    if key == "chartparameters":
        return scene.fields.get("chart_parameters")
    if key == "illuminantphotons":
        return np.asarray(scene.fields["illuminant_photons"], dtype=float)
    if key == "illuminantenergy":
        return np.asarray(scene.fields["illuminant_energy"], dtype=float)
    if key in {"roiilluminantphotons", "illuminantphotonsroi"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi illuminant photons').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "illuminant photons")
    if key in {"roimeanilluminantphotons", "illuminantphotonsroimean"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean illuminant photons').")
        return np.mean(np.asarray(scene_get(scene, "roi illuminant photons", args[0], asset_store=asset_store), dtype=float), axis=0).reshape(-1)
    if key in {"roiilluminantenergy", "illuminantenergyroi"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi illuminant energy').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "illuminant energy")
    if key in {"roimeanilluminantenergy", "illuminantenergyroimean"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean illuminant energy').")
        return np.mean(np.asarray(scene_get(scene, "roi illuminant energy", args[0], asset_store=asset_store), dtype=float), axis=0).reshape(-1)
    if key in {"illuminanthlinephotons", "hlineilluminantphotons"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant hline photons').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_photons", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminantvlinephotons", "vlineilluminantphotons"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant vline photons').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_photons", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminanthlineenergy", "hlineilluminantenergy"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant hline energy').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_energy", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminantvlineenergy", "vlineilluminantenergy"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant vline energy').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_energy", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"radiancehline", "hlineradiance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'radiance hline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "photons", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"radiancevline", "vlineradiance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'radiance vline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "photons", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"luminancehline", "hlineluminance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'luminance hline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "luminance", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"luminancevline", "vlineluminance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'luminance vline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "luminance", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"roiphotons", "roiphotonsspd"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi photons').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "photons")
    if key == "roienergy":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi energy').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "energy")
    if key == "roimeanenergy":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean energy').")
        return np.mean(np.asarray(scene_get(scene, "roi energy", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roimeanphotons":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean photons').")
        return np.mean(np.asarray(scene_get(scene, "roi photons", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roireflectance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi reflectance').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "reflectance")
    if key == "roimeanreflectance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean reflectance').")
        return np.mean(np.asarray(scene_get(scene, "roi reflectance", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roiluminance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi luminance').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "luminance")
    if key == "roimeanluminance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean luminance').")
        return float(np.mean(np.asarray(scene_get(scene, "roi luminance", args[0], asset_store=asset_store), dtype=float)))
    if key in {"chromaticity", "roichromaticity"}:
        roi_locs = args[0] if args else None
        return chromaticity_xy(_scene_roi_xyz(scene, roi_locs, asset_store=asset_store))
    if key in {"roichromaticitymean", "roimeanchromaticity"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi chromaticity mean').")
        return np.mean(
            np.asarray(scene_get(scene, "chromaticity", args[0], asset_store=asset_store), dtype=float),
            axis=0,
        ).reshape(-1)
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
        return scene_interpolate_w(scene, value)
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
    if key == "resize":
        return _scene_resize(scene, value)
    if key in {"meanluminance", "luminancemean"}:
        return scene_adjust_luminance(scene, float(value))
    if key == "illuminantenergy":
        wave = np.asarray(scene.fields["wave"], dtype=float)
        energy = np.asarray(value, dtype=float).reshape(-1)
        scene.fields["illuminant_energy"] = energy
        scene.fields["illuminant_photons"] = energy_to_quanta(energy, wave)
        _invalidate_scene_caches(scene)
        return scene
    if key == "illuminantphotons":
        scene.fields["illuminant_photons"] = np.asarray(value, dtype=float)
        _invalidate_scene_caches(scene)
        return scene
    if key == "chartparameters":
        scene.fields["chart_parameters"] = dict(value)
        return scene
    raise KeyError(f"Unsupported sceneSet parameter: {parameter}")
