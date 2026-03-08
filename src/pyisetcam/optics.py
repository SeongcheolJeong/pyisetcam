"""Optical image creation and computation."""

from __future__ import annotations

from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates, rotate, uniform_filter
from scipy.signal import fftconvolve

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .scene import scene_get
from .types import OpticalImage, Scene
from .utils import (
    DEFAULT_WAVE,
    apply_channelwise_gaussian,
    gaussian_sigma_pixels,
    param_format,
    unit_frequency_list,
)

DEFAULT_FOCAL_LENGTH_M = 0.003862755099228
DEFAULT_WVF_FOCAL_LENGTH_M = 0.0171883
DEFAULT_WVF_MEASURED_PUPIL_MM = 8.0
DEFAULT_WVF_MEASURED_WAVELENGTH_NM = 550.0
DEFAULT_WVF_SPATIAL_SAMPLES = 201
DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM = 16.212
DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM = 3.0
DEFAULT_CAMERA_WVF_CALC_PUPIL_DIAMETER_MM = 9.6569e-01
DEFAULT_RAYTRACE_ANGLE_STEP_DEG = 10.0
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


def _mat_to_native(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "_fieldnames"):
        return {str(name): _mat_to_native(getattr(value, name)) for name in getattr(value, "_fieldnames", [])}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return _mat_to_native(value.item())
            flattened = [_mat_to_native(item) for item in value.reshape(-1)]
            return np.asarray(flattened, dtype=object).reshape(value.shape)
        return np.asarray(value)
    return value


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _normalize_raytrace_table(raw: dict[str, Any] | None) -> dict[str, Any]:
    current = {} if raw is None else dict(raw)
    return {
        "field_height_mm": np.asarray(
            current.get("field_height_mm", current.get("fieldHeight", np.empty(0, dtype=float))),
            dtype=float,
        ).reshape(-1),
        "wavelength_nm": np.asarray(
            current.get("wavelength_nm", current.get("wavelength", np.empty(0, dtype=float))),
            dtype=float,
        ).reshape(-1),
        "function": np.asarray(current.get("function", np.empty(0, dtype=float)), dtype=float),
    }


def _normalize_raytrace_psf(raw: dict[str, Any] | None) -> dict[str, Any]:
    current = _normalize_raytrace_table(raw)
    source = {} if raw is None else dict(raw)
    current["sample_spacing_mm"] = np.asarray(
        source.get("sample_spacing_mm", source.get("sampleSpacing", np.array([0.0, 0.0], dtype=float))),
        dtype=float,
    ).reshape(-1)
    return current


def _normalize_raytrace_optics(raw: dict[str, Any]) -> dict[str, Any]:
    if "raytrace" in raw and "geometry" in raw["raytrace"]:
        normalized = dict(raw)
        transmittance = dict(normalized.get("transmittance", {}))
        wave = np.asarray(transmittance.get("wave", DEFAULT_WAVE.copy()), dtype=float).reshape(-1)
        scale = np.asarray(transmittance.get("scale", np.ones(wave.size, dtype=float)), dtype=float).reshape(-1)
        if scale.size == 1 and wave.size > 1:
            scale = np.full(wave.size, float(scale[0]), dtype=float)
        if scale.size != wave.size:
            scale = np.ones(wave.size, dtype=float)
        normalized["transmittance"] = {
            "wave": wave.copy(),
            "scale": scale.copy(),
        }
        normalized["raytrace"] = dict(normalized["raytrace"])
        normalized["raytrace"]["geometry"] = _normalize_raytrace_table(normalized["raytrace"].get("geometry"))
        normalized["raytrace"]["relative_illumination"] = _normalize_raytrace_table(
            normalized["raytrace"].get("relative_illumination")
        )
        normalized["raytrace"]["psf"] = _normalize_raytrace_psf(normalized["raytrace"].get("psf"))
        return normalized

    raytrace = dict(raw.get("rayTrace", raw.get("raytrace", {})))
    transmittance = dict(raw.get("transmittance", {}))
    transmittance_wave = np.asarray(transmittance.get("wave", DEFAULT_WAVE.copy()), dtype=float).reshape(-1)
    transmittance_scale = np.asarray(
        transmittance.get("scale", np.ones(transmittance_wave.size, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if transmittance_scale.size == 1 and transmittance_wave.size > 1:
        transmittance_scale = np.full(transmittance_wave.size, float(transmittance_scale[0]), dtype=float)
    if transmittance_scale.size != transmittance_wave.size:
        transmittance_scale = np.ones(transmittance_wave.size, dtype=float)

    effective_focal_length_m = _scalar(raytrace.get("effectiveFocalLength"), DEFAULT_FOCAL_LENGTH_M * 1e3) / 1e3
    nominal_f_number = _scalar(raytrace.get("fNumber", raw.get("fNumber")), 4.0)
    return {
        "model": "raytrace",
        "name": str(raw.get("name", raytrace.get("name", "raytrace"))),
        "f_number": nominal_f_number,
        "focal_length_m": effective_focal_length_m,
        "nominal_focal_length_m": _scalar(raw.get("focalLength"), DEFAULT_FOCAL_LENGTH_M),
        "compute_method": "",
        "aberration_scale": 0.0,
        "offaxis_method": "skip",
        "transmittance": {
            "wave": transmittance_wave.copy(),
            "scale": transmittance_scale.copy(),
        },
        "raytrace": {
            "program": str(raytrace.get("program", "")),
            "lens_file": str(raytrace.get("lensFile", "")),
            "reference_wavelength_nm": _scalar(
                raytrace.get("referenceWavelength"),
                DEFAULT_WVF_MEASURED_WAVELENGTH_NM,
            ),
            "object_distance_m": _scalar(raytrace.get("objectDistance"), np.inf) / 1e3,
            "magnification": _scalar(raytrace.get("mag"), 0.0),
            "f_number": nominal_f_number,
            "effective_focal_length_m": effective_focal_length_m,
            "effective_f_number": _scalar(raytrace.get("effectiveFNumber"), nominal_f_number),
            "max_fov_deg": _scalar(raytrace.get("maxfov", raytrace.get("fov")), np.inf),
            "geometry": _normalize_raytrace_table(raytrace.get("geometry")),
            "relative_illumination": _normalize_raytrace_table(raytrace.get("relIllum", raytrace.get("relative_illumination"))),
            "psf": _normalize_raytrace_psf(raytrace.get("psf")),
            "name": str(raytrace.get("name", raw.get("name", "raytrace"))),
        },
    }


def _load_raytrace_optics(source: Any, *, asset_store: AssetStore) -> dict[str, Any]:
    if source is None:
        raw = asset_store.load_mat("data/optics/rtZemaxExample.mat")["optics"]
        return _normalize_raytrace_optics(_mat_to_native(raw))
    if isinstance(source, dict):
        return _normalize_raytrace_optics(source)

    path = Path(source)
    if path.is_absolute() or path.exists():
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)["optics"]
        return _normalize_raytrace_optics(_mat_to_native(raw))

    candidates = [Path(str(source))]
    if not str(source).lower().endswith(".mat"):
        candidates.append(Path("data/optics") / f"{source}.mat")
    candidates.append(Path("data/optics") / str(source))
    for candidate in candidates:
        try:
            raw = asset_store.load_mat(candidate)["optics"]
            return _normalize_raytrace_optics(_mat_to_native(raw))
        except Exception:
            continue
    raise UnsupportedOptionError("oiCreate", f"ray trace optics {source}")


def wvf_create(
    *,
    wave: np.ndarray | None = None,
    focal_length_m: float = DEFAULT_WVF_FOCAL_LENGTH_M,
    f_number: float | None = None,
    aberration_scale: float = 0.0,
    measured_pupil_diameter_mm: float = DEFAULT_WVF_MEASURED_PUPIL_MM,
    measured_wavelength_nm: float = DEFAULT_WVF_MEASURED_WAVELENGTH_NM,
    calc_pupil_diameter_mm: float = DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM,
    zcoeffs: np.ndarray | None = None,
    lca_method: str = "none",
    flip_psf_upside_down: bool = False,
    rotate_psf_90_degs: bool = False,
    compute_sce: bool = False,
    sce_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calc_pupil = float(calc_pupil_diameter_mm)
    focal_length = float(focal_length_m)
    wave_values = np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float)
    if f_number is None:
        f_number = (focal_length * 1e3) / max(calc_pupil, 1e-12)
    return {
        "type": "wvf",
        "wave": wave_values,
        "focal_length_m": focal_length,
        "f_number": float(f_number),
        "aberration_scale": float(aberration_scale),
        "measured_pupil_diameter_mm": float(measured_pupil_diameter_mm),
        "measured_wavelength_nm": float(measured_wavelength_nm),
        "sample_interval_domain": "psf",
        "spatial_samples": DEFAULT_WVF_SPATIAL_SAMPLES,
        "ref_pupil_plane_size_mm": DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM,
        "calc_pupil_diameter_mm": calc_pupil,
        "zcoeffs": np.asarray([0.0] if zcoeffs is None else zcoeffs, dtype=float).reshape(-1),
        "lca_method": str(lca_method),
        "flip_psf_upside_down": bool(flip_psf_upside_down),
        "rotate_psf_90_degs": bool(rotate_psf_90_degs),
        "compute_sce": bool(compute_sce),
        "sce_params": _normalize_sce_params(wave_values, sce_params),
    }


def oi_create(
    oi_type: str = "diffraction limited",
    *args: Any,
    asset_store: AssetStore | None = None,
) -> OpticalImage:
    """Create a supported optical image object."""

    normalized = param_format(oi_type)
    oi = OpticalImage(name="opticalimage")
    optics: dict[str, Any]

    if normalized in {"default", "diffractionlimited", "diffraction"}:
        optics = {
            "model": "diffractionlimited",
            "f_number": 4.0,
            "focal_length_m": DEFAULT_FOCAL_LENGTH_M,
            "compute_method": "opticsotf",
            "aberration_scale": 0.0,
            "offaxis_method": "cos4th",
        }
    elif normalized in {"wvf", "shiftinvariant"}:
        wavefront = (
            args[0]
            if args and isinstance(args[0], dict)
            else wvf_create(
                wave=DEFAULT_WAVE.copy(),
                focal_length_m=DEFAULT_FOCAL_LENGTH_M,
                f_number=4.0,
                calc_pupil_diameter_mm=DEFAULT_CAMERA_WVF_CALC_PUPIL_DIAMETER_MM,
            )
        )
        optics = {
            "model": "shiftinvariant",
            "f_number": float(wavefront.get("f_number", 4.0)),
            "focal_length_m": float(wavefront.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M)),
            "compute_method": "opticspsf",
            "aberration_scale": float(wavefront.get("aberration_scale", 0.0)),
            "wavefront": wavefront,
            "offaxis_method": "cos4th",
        }
    elif normalized == "raytrace":
        source = args[0] if args else None
        optics = _load_raytrace_optics(source, asset_store=_store(asset_store))
    elif normalized == "pinhole":
        optics = {
            "model": "skip",
            "f_number": 1e-3,
            "focal_length_m": 1e-2,
            "compute_method": "skip",
            "aberration_scale": 0.0,
            "offaxis_method": "skip",
        }
    elif normalized == "empty":
        optics = {
            "model": "diffractionlimited",
            "f_number": 4.0,
            "focal_length_m": DEFAULT_FOCAL_LENGTH_M,
            "compute_method": "opticsotf",
            "aberration_scale": 0.0,
            "offaxis_method": "cos4th",
        }
    else:
        raise UnsupportedOptionError("oiCreate", oi_type)

    wave_index = 1 if normalized in {"wvf", "shiftinvariant", "raytrace"} else 0
    wave = np.asarray(args[wave_index], dtype=float) if len(args) > wave_index else DEFAULT_WAVE.copy()
    optics.setdefault(
        "transmittance",
        {
            "wave": np.asarray(wave, dtype=float).reshape(-1).copy(),
            "scale": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
        },
    )
    oi.fields["optics"] = optics
    oi.fields["wave"] = wave
    oi.fields["compute_method"] = optics["compute_method"]
    oi.fields["diffuser_method"] = "skip"
    oi.fields["diffuser_blur_m"] = 2e-6
    oi.fields["psf_angle_step_deg"] = DEFAULT_RAYTRACE_ANGLE_STEP_DEG
    oi.fields["psf_sample_angles_deg"] = None
    oi.fields["psf_image_heights_m"] = None
    oi.fields["psf_wavelength_nm"] = None
    oi.fields["psf_optics_name"] = None
    oi.fields["psf_struct"] = None
    oi.fields["sample_spacing_m"] = None
    oi.data["photons"] = np.empty((0, 0, 0), dtype=float)
    return oi


def _scene_sample_spacing(scene: Scene) -> float:
    width = float(scene.fields["width_m"])
    cols = int(scene.fields["cols"])
    return width / max(cols, 1)


def _image_distance_m(optics: dict[str, Any], scene: Scene) -> float:
    focal_length = float(optics["focal_length_m"])
    model = param_format(optics.get("model", ""))
    if model == "skip":
        return focal_length
    if model == "raytrace":
        raytrace = optics.get("raytrace", {})
        return float(raytrace.get("effective_focal_length_m", focal_length))
    scene_distance = float(scene.fields.get("distance_m", np.inf))
    if not np.isfinite(scene_distance) or scene_distance <= focal_length:
        return focal_length
    return 1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12)


def _magnification(optics: dict[str, Any], scene: Scene) -> float:
    model = param_format(optics.get("model", ""))
    if model == "skip":
        return -1.0
    if model == "raytrace":
        return float(optics.get("raytrace", {}).get("magnification", 0.0))
    scene_distance = float(scene.fields.get("distance_m", np.inf))
    if not np.isfinite(scene_distance) or scene_distance <= 0.0:
        return 0.0
    return -_image_distance_m(optics, scene) / scene_distance


def _oi_geometry(optics: dict[str, Any], scene: Scene) -> tuple[float, float, float]:
    image_distance = _image_distance_m(optics, scene)
    hfov_deg = float(scene.fields.get("fov_deg", 10.0))
    vfov_deg = float(scene.fields.get("vfov_deg", hfov_deg))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(hfov_deg) / 2.0)
    height_m = 2.0 * image_distance * np.tan(np.deg2rad(vfov_deg) / 2.0)
    return image_distance, width_m, height_m


def _radiance_to_irradiance(scene_cube: np.ndarray, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    transmittance = _optics_transmittance_scale(optics, wave)
    raytrace = optics.get("raytrace", {})
    if param_format(optics.get("model", "")) == "raytrace":
        f_number = float(raytrace.get("effective_f_number", optics["f_number"]))
        magnification = float(raytrace.get("magnification", 0.0))
    else:
        f_number = float(optics["f_number"])
        magnification = _magnification(optics, scene)
    scale = np.pi / (1.0 + 4.0 * (f_number**2) * ((1.0 + abs(magnification)) ** 2))
    irradiance = np.asarray(scene_cube, dtype=float)
    if np.any(transmittance != 1.0):
        irradiance = irradiance * transmittance.reshape(1, 1, -1)
    return irradiance * scale


def _cos4th_factor(rows: int, cols: int, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    _, width_m, height_m = _oi_geometry(optics, scene)
    # ISETCam's cos4th.m uses the OI spatial support that reflects the
    # scene-focused image size, but then calls opticsGet('imageDistance')
    # without a scene-distance argument. That getter falls back to the focal
    # length, so we mirror that behavior here.
    image_distance = float(optics["focal_length_m"])
    x = np.linspace(-width_m / 2.0 + width_m / (2.0 * cols), width_m / 2.0 - width_m / (2.0 * cols), cols)
    y = np.linspace(-height_m / 2.0 + height_m / (2.0 * rows), height_m / 2.0 - height_m / (2.0 * rows), rows)
    xx, yy = np.meshgrid(x, y)
    s_factor = np.sqrt(image_distance**2 + xx**2 + yy**2)
    image_diagonal = np.sqrt(width_m**2 + height_m**2)
    if image_distance > 10.0 * image_diagonal:
        return (image_distance / np.maximum(s_factor, 1e-12)) ** 4

    magnification = _magnification(optics, scene)
    cos_phi = image_distance / np.maximum(s_factor, 1e-12)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, None))
    tan_phi = sin_phi / np.maximum(cos_phi, 1e-12)

    f_number = float(optics["f_number"])
    sin_theta = 1.0 / (1.0 + 4.0 * (f_number * (1.0 - magnification)) ** 2)
    cos_theta = np.sqrt(max(1e-12, 1.0 - sin_theta**2))
    tan_theta = sin_theta / cos_theta
    numerator = 1.0 - tan_theta**2 + tan_phi**2
    denominator = np.sqrt(tan_phi**4 + 2.0 * tan_phi**2 * (1.0 - tan_theta**2) + 1.0 / (cos_theta**4))
    spatial_fall = (np.pi / 2.0) * (1.0 - numerator / np.maximum(denominator, 1e-12))
    return spatial_fall / max(np.pi * (sin_theta**2), 1e-12)


def _pad_scene(
    scene_cube: np.ndarray,
    pad_pixels: tuple[int, int],
    pad_value: str,
) -> tuple[np.ndarray, str, float]:
    pad_rows, pad_cols = int(pad_pixels[0]), int(pad_pixels[1])
    if pad_rows <= 0 and pad_cols <= 0:
        return scene_cube, "nearest", 0.0
    mode = param_format(pad_value)
    if mode == "zero":
        return np.pad(scene_cube, ((pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0))), "constant", 0.0
    if mode == "border":
        padded = np.empty(
            (scene_cube.shape[0] + 2 * pad_rows, scene_cube.shape[1] + 2 * pad_cols, scene_cube.shape[2]),
            dtype=float,
        )
        band_corner = scene_cube[0, 0, :]
        row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
        col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
        for band_index, corner_value in enumerate(band_corner):
            padded[:, :, band_index] = corner_value
            padded[row_slice, col_slice, band_index] = scene_cube[:, :, band_index]
        return padded, "constant", float(np.mean(band_corner))
    if mode == "mean":
        padded = np.empty(
            (scene_cube.shape[0] + 2 * pad_rows, scene_cube.shape[1] + 2 * pad_cols, scene_cube.shape[2]),
            dtype=float,
        )
        band_means = scene_cube.mean(axis=(0, 1))
        row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
        col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
        for band_index, mean_value in enumerate(band_means):
            padded[:, :, band_index] = mean_value
            padded[row_slice, col_slice, band_index] = scene_cube[:, :, band_index]
        return padded, "constant", float(np.mean(band_means))
    raise UnsupportedOptionError("oiCompute", pad_value)


def _pad_depth_map(scene: Scene, pad_pixels: tuple[int, int]) -> np.ndarray:
    depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
    pad_rows, pad_cols = int(pad_pixels[0]), int(pad_pixels[1])
    if pad_rows <= 0 and pad_cols <= 0:
        return depth_map.copy()
    return np.pad(depth_map, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="constant", constant_values=0.0)


def _diffraction_otf(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    scene: Scene,
) -> np.ndarray:
    rows, cols = shape
    image_distance = _image_distance_m(optics, scene)
    width_m = float(cols) * float(sample_spacing_m)
    height_m = float(rows) * float(sample_spacing_m)
    fov_width = float(np.rad2deg(2.0 * np.arctan2(width_m / 2.0, image_distance)))
    fov_height = float(np.rad2deg(2.0 * np.arctan2(height_m / 2.0, image_distance)))
    distance_per_degree = width_m / max(fov_width, 1e-12)
    deg_per_dist = 1.0 / max(distance_per_degree, 1e-12)
    fx = unit_frequency_list(cols) * ((cols / 2.0) / max(fov_width, 1e-12) * deg_per_dist)
    fy = unit_frequency_list(rows) * ((rows / 2.0) / max(fov_height, 1e-12) * deg_per_dist)
    rho = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    aperture_diameter = float(optics["focal_length_m"]) / max(float(optics["f_number"]), 1e-12)
    # ISETCam's dlMTF() uses opticsGet(optics, 'focalPlaneDistance') without
    # passing the scene distance, which resolves to the focal length rather
    # than the thin-lens image distance for finite scene depth.
    focal_plane_distance = float(optics["focal_length_m"])
    wavelengths_m = np.asarray(wave, dtype=float) * 1e-9
    cutoff = (aperture_diameter / max(focal_plane_distance, 1e-12)) / np.maximum(wavelengths_m, 1e-12)

    otf = np.zeros((rows, cols, wavelengths_m.size), dtype=float)
    for index, cutoff_frequency in enumerate(cutoff):
        normalized = rho / max(float(cutoff_frequency), 1e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        current = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        current[normalized >= 1.0] = 0.0
        otf[:, :, index] = np.fft.ifftshift(current)
    return otf


def _apply_otf(cube: np.ndarray, otf: np.ndarray) -> np.ndarray:
    result = np.empty_like(cube, dtype=float)
    for band_index in range(cube.shape[2]):
        filtered = np.fft.ifft2(np.fft.fft2(cube[:, :, band_index]) * otf[:, :, band_index])
        result[:, :, band_index] = np.abs(filtered)
    return result


def _resize_image(image: np.ndarray, output_shape: tuple[int, int], *, method: str) -> np.ndarray:
    out_rows = max(int(round(output_shape[0])), 1)
    out_cols = max(int(round(output_shape[1])), 1)
    image = np.asarray(image, dtype=float)
    in_rows, in_cols = image.shape
    if (in_rows, in_cols) == (out_rows, out_cols):
        return image.copy()

    row_positions = np.linspace(0.0, max(in_rows - 1, 0), out_rows)
    col_positions = np.linspace(0.0, max(in_cols - 1, 0), out_cols)
    row_grid, col_grid = np.meshgrid(row_positions, col_positions, indexing="ij")
    if method == "nearest":
        row_index = np.clip(np.rint(row_grid).astype(int), 0, max(in_rows - 1, 0))
        col_index = np.clip(np.rint(col_grid).astype(int), 0, max(in_cols - 1, 0))
        return image[row_index, col_index]
    if method == "linear":
        return map_coordinates(image, [row_grid, col_grid], order=1, mode="nearest", prefilter=False)
    raise ValueError(f"Unsupported resize method: {method}")


def _resize_nearest(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    return _resize_image(image, output_shape, method="nearest")


def _support_resample_positions(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Resample step must be positive.")
    if stop <= start:
        return np.array([float(start)], dtype=float)
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return float(start) + float(step) * np.arange(max(count, 1), dtype=float)


def _resample_plane_on_support(
    plane: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    if x_support.size <= 1:
        col_coords = np.zeros_like(x_query, dtype=float)
    else:
        col_coords = (np.asarray(x_query, dtype=float) - float(x_support[0])) / float(x_support[1] - x_support[0])
    if y_support.size <= 1:
        row_coords = np.zeros_like(y_query, dtype=float)
    else:
        row_coords = (np.asarray(y_query, dtype=float) - float(y_support[0])) / float(y_support[1] - y_support[0])
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    if method == "linear":
        order = 1
    elif method == "nearest":
        order = 0
    else:
        raise ValueError(f"Unsupported resample method: {method}")
    return map_coordinates(plane, [row_grid, col_grid], order=order, mode="nearest", prefilter=False)


def _oi_spatial_resample(oi: OpticalImage, sample_spacing_m: float, *, method: str = "linear") -> OpticalImage:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.size == 0:
        return oi.clone()

    support = oi_get(oi, "spatial support linear", "m")
    x_support = np.asarray(support["x"], dtype=float)
    y_support = np.asarray(support["y"], dtype=float)
    x_query = _support_resample_positions(float(x_support[0]), float(x_support[-1]), float(sample_spacing_m))
    y_query = _support_resample_positions(float(y_support[0]), float(y_support[-1]), float(sample_spacing_m))

    resampled_cube = np.empty((y_query.size, x_query.size, photons.shape[2]), dtype=float)
    for band_index in range(photons.shape[2]):
        resampled_cube[:, :, band_index] = _resample_plane_on_support(
            photons[:, :, band_index],
            x_support,
            y_support,
            x_query,
            y_query,
            method=method,
        )

    resampled = oi.clone()
    original_fov = float(oi_get(oi, "fov"))
    resampled.data["photons"] = resampled_cube
    depth_map = oi.fields.get("depth_map_m")
    if depth_map is not None:
        resampled.fields["depth_map_m"] = _resample_plane_on_support(
            np.asarray(depth_map, dtype=float),
            x_support,
            y_support,
            x_query,
            y_query,
            method="nearest",
        )
    _sync_oi_geometry_fields(resampled)
    current_spacing = float(oi_get(resampled, "wspatialresolution"))
    if current_spacing > 0.0:
        resampled = oi_set(resampled, "fov", original_fov * float(sample_spacing_m) / current_spacing)
    return resampled


def _image_bounding_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or cols.size == 0:
        return (0.0, 0.0, 1.0, 1.0)

    min_row = float(rows.min() + 1)
    max_row = float(rows.max() + 1)
    min_col = float(cols.min() + 1)
    max_col = float(cols.max() + 1)
    center_x = (min_col + max_col) / 2.0
    center_y = (min_row + max_row) / 2.0
    max_diff = max(abs(max_row - center_y), abs(max_col - center_x))
    return (center_x - max_diff, center_y - max_diff, 2.0 * max_diff, 2.0 * max_diff)


def _ensure_optics_transmittance(optics: dict[str, Any], wave: np.ndarray | None = None) -> dict[str, np.ndarray]:
    current = optics.get("transmittance")
    if isinstance(current, dict) and "wave" in current and "scale" in current:
        wave_values = np.asarray(current["wave"], dtype=float).reshape(-1)
        scale_values = np.asarray(current["scale"], dtype=float).reshape(-1)
        if wave_values.size == scale_values.size and wave_values.size > 0:
            current["wave"] = wave_values
            current["scale"] = scale_values
            return current

    base_wave = np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float).reshape(-1)
    optics["transmittance"] = {
        "wave": base_wave.copy(),
        "scale": np.ones(base_wave.size, dtype=float),
    }
    return optics["transmittance"]


def _optics_transmittance_scale(optics: dict[str, Any], wave: np.ndarray) -> np.ndarray:
    transmittance = _ensure_optics_transmittance(optics, wave=np.asarray(wave, dtype=float).reshape(-1))
    source_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
    source_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    if source_wave.size == 0 or source_scale.size == 0:
        return np.ones(target_wave.size, dtype=float)
    if source_wave.size == target_wave.size and np.array_equal(source_wave, target_wave):
        return source_scale.copy()
    return np.interp(target_wave, source_wave, source_scale, left=1.0, right=1.0)


def _normalize_sce_params(wave: np.ndarray, sce_params: dict[str, Any] | None = None) -> dict[str, Any]:
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    current = {} if sce_params is None else dict(sce_params)
    wave_nm = np.asarray(current.get("wave", current.get("wavelengths", target_wave)), dtype=float).reshape(-1)
    rho = np.asarray(current.get("rho", np.zeros(wave_nm.size, dtype=float)), dtype=float).reshape(-1)
    if rho.size == 1 and wave_nm.size > 1:
        rho = np.full(wave_nm.size, float(rho[0]), dtype=float)
    if rho.size != wave_nm.size:
        raise ValueError("SCE rho must match the SCE wavelength sampling.")
    return {
        "wave": wave_nm.copy(),
        "rho": rho.copy(),
        "xo_mm": float(current.get("xo_mm", current.get("xo", 0.0))),
        "yo_mm": float(current.get("yo_mm", current.get("yo", 0.0))),
    }


def _sce_rho_for_wave(sce_params: dict[str, Any], wavelength_nm: float) -> float:
    wave_nm = np.asarray(sce_params.get("wave", np.array([], dtype=float)), dtype=float).reshape(-1)
    rho = np.asarray(sce_params.get("rho", np.array([], dtype=float)), dtype=float).reshape(-1)
    if wave_nm.size == 0 or rho.size == 0:
        return 0.0
    if wave_nm.size == 1:
        return float(rho[0])
    return float(np.interp(float(wavelength_nm), wave_nm, rho, left=float(rho[0]), right=float(rho[-1])))


def _osa_index_to_nm(index: int) -> tuple[int, int]:
    normalized_index = int(index)
    if normalized_index < 0:
        raise ValueError("OSA index must be non-negative.")
    order = 0
    offset = 0
    while True:
        ms = list(range(-order, order + 1, 2))
        next_offset = offset + len(ms)
        if normalized_index < next_offset:
            return order, ms[normalized_index - offset]
        offset = next_offset
        order += 1


def _zernike_radial(n: int, m_abs: int, radius: np.ndarray) -> np.ndarray:
    radial = np.zeros_like(radius, dtype=float)
    half_sum = (n + m_abs) // 2
    half_diff = (n - m_abs) // 2
    for s in range(half_diff + 1):
        coefficient = (
            ((-1) ** s)
            * factorial(n - s)
            / (
                factorial(s)
                * factorial(half_sum - s)
                * factorial(half_diff - s)
            )
        )
        radial = radial + (coefficient * np.power(radius, n - (2 * s)))
    return radial


def _zernike_surface_osa(zcoeffs: np.ndarray, norm_radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(zcoeffs, dtype=float).reshape(-1)
    if coefficients.size == 0 or np.allclose(coefficients, 0.0):
        return np.zeros_like(norm_radius, dtype=float)

    valid = norm_radius <= 1.0
    surface = np.zeros_like(norm_radius, dtype=float)
    radius_valid = norm_radius[valid]
    theta_valid = theta[valid]

    for osa_index, coefficient in enumerate(coefficients):
        if np.isclose(coefficient, 0.0):
            continue
        n, m = _osa_index_to_nm(osa_index)
        radial = _zernike_radial(n, abs(m), radius_valid)
        if m == 0:
            angular = np.ones_like(theta_valid, dtype=float)
            normalization = np.sqrt(n + 1.0)
        elif m > 0:
            angular = np.cos(m * theta_valid)
            normalization = np.sqrt(2.0 * (n + 1.0))
        else:
            angular = np.sin(abs(m) * theta_valid)
            normalization = np.sqrt(2.0 * (n + 1.0))
        term = normalization * radial * angular
        surface[valid] = surface[valid] + (float(coefficient) * term)

    return surface


def _wvf_aperture_mask(
    n_pixels: int,
    calc_radius_index: np.ndarray,
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    if aperture is None:
        return calc_radius_index.astype(float)

    current = np.asarray(aperture, dtype=float)
    if current.shape != (n_pixels, n_pixels):
        current = _resize_image(current, (n_pixels, n_pixels), method="linear")

    bounding_box = _image_bounding_box(calc_radius_index)
    target_shape = (
        max(int(round(bounding_box[3])), 1),
        max(int(round(bounding_box[2])), 1),
    )
    current = _resize_nearest(current, target_shape)

    pad = int(round((n_pixels - bounding_box[2]) / 2.0) - 2.0)
    if pad > 0:
        current = np.pad(current, ((pad, pad), (pad, pad)), mode="constant")

    current = _resize_nearest(current, (n_pixels, n_pixels))
    current = np.clip(current, 0.0, 1.0)
    current[~calc_radius_index] = 0.0
    return current


def _wvf_psf_stack(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = shape
    n_pixels = int(max(rows, cols))
    wavefront = dict(optics.get("wavefront", {}))
    measured_pupil_mm = float(wavefront.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM))
    measured_wavelength_nm = float(wavefront.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    lca_method = param_format(wavefront.get("lca_method", "none"))
    if lca_method not in {"none", ""}:
        raise UnsupportedOptionError("oiCompute", f"wvf lca method {wavefront.get('lca_method')}")
    compute_sce = bool(wavefront.get("compute_sce", False))
    sce_params = _normalize_sce_params(np.asarray(wave, dtype=float), wavefront.get("sce_params"))
    zcoeffs = np.asarray(wavefront.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1)
    focal_length_mm = float(optics["focal_length_m"]) * 1e3
    calc_pupil_mm = float(
        wavefront.get(
            "calc_pupil_diameter_mm",
            focal_length_mm / max(float(optics["f_number"]), 1e-12),
        )
    )
    psf_spacing_mm = float(sample_spacing_m) * 1e3
    ref_field_size_mm = measured_wavelength_nm * 1e-6 * focal_length_mm / max(psf_spacing_mm, 1e-12)

    middle_row = np.floor(n_pixels / 2.0) + 1.0
    sample_positions = (np.arange(n_pixels, dtype=float) + 1.0) - middle_row
    calc_radius = calc_pupil_mm / max(measured_pupil_mm, 1e-12)

    psf_stack = np.empty((n_pixels, n_pixels, len(wave)), dtype=float)
    for band_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        pupil_plane_size_mm = ref_field_size_mm * (float(wavelength_nm) / max(measured_wavelength_nm, 1e-12))
        pupil_sample_spacing_mm = pupil_plane_size_mm / max(n_pixels, 1)
        pupil_pos = sample_positions * pupil_sample_spacing_mm
        xpos, ypos = np.meshgrid(pupil_pos, -pupil_pos)
        norm_radius = np.sqrt(xpos**2 + ypos**2) / max(measured_pupil_mm / 2.0, 1e-12)
        theta = np.arctan2(ypos, xpos)
        calc_radius_index = norm_radius <= calc_radius
        aperture_mask = _wvf_aperture_mask(n_pixels, calc_radius_index, aperture=aperture)
        if compute_sce:
            rho = _sce_rho_for_wave(sce_params, float(wavelength_nm))
            xo_mm = float(sce_params.get("xo_mm", 0.0))
            yo_mm = float(sce_params.get("yo_mm", 0.0))
            aperture_mask = aperture_mask * np.power(10.0, -rho * ((xpos - xo_mm) ** 2 + (ypos - yo_mm) ** 2))
        wavefront_aberrations_um = _zernike_surface_osa(zcoeffs, norm_radius, theta)
        pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront_aberrations_um / max(float(wavelength_nm) * 1e-3, 1e-12))
        pupil_phase[norm_radius > calc_radius] = 0.0
        pupil_function = aperture_mask * pupil_phase

        amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))
        intensity = np.real(amp * np.conj(amp))
        psf = intensity / max(float(np.sum(intensity)), 1e-12)
        if bool(wavefront.get("flip_psf_upside_down", False)):
            psf = np.flipud(psf)
        if bool(wavefront.get("rotate_psf_90_degs", False)):
            psf = np.rot90(psf)
        psf_stack[:, :, band_index] = psf

    return psf_stack


def _apply_psf(cube: np.ndarray, psf_stack: np.ndarray) -> np.ndarray:
    rows, cols = cube.shape[:2]
    result = np.empty_like(cube, dtype=float)

    for band_index in range(cube.shape[2]):
        plane = cube[:, :, band_index]
        psf = psf_stack[:, :, band_index]

        if rows != cols:
            delta = abs(cols - rows)
            pre = delta // 2
            post = delta - pre
            if cols < rows:
                padded_plane = np.pad(plane, ((0, 0), (pre, post)))
                filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(padded_plane) * np.fft.fft2(psf)))
                plane_result = np.real(filtered[:, pre : pre + cols])
            else:
                padded_plane = np.pad(plane, ((pre, post), (0, 0)))
                filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(padded_plane) * np.fft.fft2(psf)))
                plane_result = np.real(filtered[pre : pre + rows, :])
        else:
            filtered = np.fft.ifft2(np.fft.fft2(plane) * np.fft.fft2(np.fft.ifftshift(psf)))
            plane_result = np.real(filtered)

        result[:, :, band_index] = plane_result

    return result


def _scene_diagonal_fov_deg(scene: Scene) -> float:
    hfov = float(scene.fields.get("fov_deg", 10.0))
    vfov = float(scene.fields.get("vfov_deg", hfov))
    tangent = np.sqrt(np.tan(np.deg2rad(hfov) / 2.0) ** 2 + np.tan(np.deg2rad(vfov) / 2.0) ** 2)
    return float(np.rad2deg(2.0 * np.arctan(tangent)))


def _raytrace_curve(table: dict[str, Any], wavelength_nm: float) -> np.ndarray:
    values = np.asarray(table.get("function", np.empty(0, dtype=float)), dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    wavelengths = np.asarray(table.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if values.ndim == 1 or wavelengths.size <= 1:
        return np.asarray(values, dtype=float).reshape(-1)
    wave_index = int(np.argmin(np.abs(wavelengths - float(wavelength_nm))))
    return np.asarray(values[:, wave_index], dtype=float).reshape(-1)


def _raytrace_geometry(cube: np.ndarray, wave: np.ndarray, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    raytrace = dict(optics.get("raytrace", {}))
    max_fov = float(raytrace.get("max_fov_deg", np.inf))
    if np.isfinite(max_fov) and (_scene_diagonal_fov_deg(scene) > max_fov + 1e-9):
        raise ValueError("Scene field of view exceeds the loaded ray-trace analysis.")

    rows, cols = cube.shape[:2]
    width_mm = _oi_geometry(optics, scene)[1] * 1e3
    dx_mm = width_mm / max(cols, 1)
    zeropad = 16
    padded_rows = rows + 2 * zeropad
    padded_cols = cols + 2 * zeropad
    row_center = (padded_rows + 1.0) / 2.0
    col_center = (padded_cols + 1.0) / 2.0

    r = (np.arange(padded_rows, dtype=float) + 1.0) - row_center
    c = (np.arange(padded_cols, dtype=float) + 1.0) - col_center
    cc, rr = np.meshgrid(c, r)
    pixdist_units = np.sqrt(cc**2 + rr**2) * dx_mm
    pixang = np.arctan2(cc, rr)

    field_heights = np.asarray(raytrace.get("geometry", {}).get("field_height_mm", np.empty(0, dtype=float)), dtype=float)
    if field_heights.size < 2:
        return np.asarray(cube, dtype=float).copy()

    degree = min(8, max(field_heights.size - 2, 0))
    result = np.empty_like(cube, dtype=float)
    for band_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        distorted_height = _raytrace_curve(raytrace["geometry"], float(wavelength_nm))
        relative_illumination = _raytrace_curve(raytrace["relative_illumination"], float(wavelength_nm))
        if distorted_height.size != field_heights.size or relative_illumination.size != field_heights.size:
            result[:, :, band_index] = cube[:, :, band_index]
            continue

        distortion_poly = np.polyfit(distorted_height, field_heights, degree)
        ri_degree = min(2, max(distorted_height.size - 1, 0))
        illumination_poly = np.polyfit(distorted_height, relative_illumination, ri_degree)

        ideal_height = np.polyval(distortion_poly, pixdist_units)
        rel_illum = np.clip(np.polyval(illumination_poly, pixdist_units), 0.0, None)
        pad_r = np.clip(row_center + ideal_height * np.cos(pixang) / max(dx_mm, 1e-12), 1.0, float(padded_rows)) - 1.0
        pad_c = np.clip(col_center + ideal_height * np.sin(pixang) / max(dx_mm, 1e-12), 1.0, float(padded_cols)) - 1.0

        padded = np.pad(np.asarray(cube[:, :, band_index], dtype=float), ((zeropad, zeropad), (zeropad, zeropad)))
        distorted = map_coordinates(padded, [pad_r, pad_c], order=1, mode="nearest", prefilter=False)
        result[:, :, band_index] = distorted[zeropad:-zeropad, zeropad:-zeropad] * rel_illum[zeropad:-zeropad, zeropad:-zeropad]

    return result


def _raytrace_psf_support_axes(psf_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    psf_function = np.asarray(psf_data.get("function", np.empty(0, dtype=float)), dtype=float)
    if psf_function.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    sample_spacing = np.asarray(psf_data.get("sample_spacing_mm", np.array([0.0, 0.0], dtype=float)), dtype=float).reshape(-1)
    if sample_spacing.size == 1:
        sample_spacing = np.repeat(sample_spacing, 2)
    rows, cols = psf_function.shape[:2]
    y_support = np.arange((-rows / 2.0) + 1.0, (rows / 2.0) + 1.0, dtype=float) * float(sample_spacing[0])
    x_support = np.arange((-cols / 2.0) + 1.0, (cols / 2.0) + 1.0, dtype=float) * float(sample_spacing[1])
    return x_support, y_support


def _raytrace_sampling_axes(rows: int, cols: int, sample_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(
        -(cols * sample_spacing_mm) / 2.0 + sample_spacing_mm / 2.0,
        (cols * sample_spacing_mm) / 2.0 - sample_spacing_mm / 2.0,
        cols,
    )
    y = np.linspace(
        -(rows * sample_spacing_mm) / 2.0 + sample_spacing_mm / 2.0,
        (rows * sample_spacing_mm) / 2.0 - sample_spacing_mm / 2.0,
        rows,
    )
    return x, y


def _raytrace_data_angle_height(rows: int, cols: int, sample_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    x_support, y_support = _raytrace_sampling_axes(rows, cols, sample_spacing_mm)
    xx, yy = np.meshgrid(x_support, y_support[::-1])
    data_angle = np.rint(np.rad2deg(np.arctan2(yy, xx) + np.pi)).astype(int)
    data_angle[data_angle == 0] = 1
    data_angle[data_angle > 360] = 360
    data_height = np.sqrt(xx**2 + yy**2)
    return data_angle, data_height


def _raytrace_sample_heights(field_heights_mm: np.ndarray, data_height_mm: np.ndarray) -> np.ndarray:
    heights = np.asarray(field_heights_mm, dtype=float).reshape(-1)
    if heights.size == 0:
        return heights
    max_height = float(np.max(np.asarray(data_height_mm, dtype=float)))
    keep = np.zeros(heights.size, dtype=bool)
    for index, value in enumerate(heights):
        keep[index] = True
        if float(value) > max_height:
            break
    return heights[keep]


def _raytrace_psf_grid(sample_spacing_mm: float, psf_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source_x, source_y = _raytrace_psf_support_axes(psf_data)
    if source_x.size == 0 or source_y.size == 0:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
    x_positive = np.arange(0.0, float(source_x[-1]) + sample_spacing_mm * 0.5, sample_spacing_mm)
    x_negative = -np.flip(np.arange(0.0, abs(float(source_x[0])) + sample_spacing_mm * 0.5, sample_spacing_mm))
    y_positive = np.arange(0.0, float(source_y[-1]) + sample_spacing_mm * 0.5, sample_spacing_mm)
    y_negative = -np.flip(np.arange(0.0, abs(float(source_y[0])) + sample_spacing_mm * 0.5, sample_spacing_mm))
    x_grid = np.concatenate((x_negative[:-1], x_positive))
    y_grid = np.concatenate((y_negative[:-1], y_positive))
    return x_grid, y_grid


def _raytrace_resample_psf(
    psf: np.ndarray,
    source_x_mm: np.ndarray,
    source_y_mm: np.ndarray,
    target_x_mm: np.ndarray,
    target_y_mm: np.ndarray,
) -> np.ndarray:
    kernel = _resample_plane_on_support(
        np.asarray(psf, dtype=float),
        np.asarray(source_x_mm, dtype=float),
        np.asarray(source_y_mm, dtype=float),
        np.asarray(target_x_mm, dtype=float),
        np.asarray(target_y_mm, dtype=float),
        method="linear",
    )
    kernel = np.clip(np.asarray(kernel, dtype=float), 0.0, None)
    total = float(np.sum(kernel))
    if total <= 0.0:
        kernel = np.zeros_like(kernel, dtype=float)
        kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
        return kernel
    return kernel / total


def _raytrace_angle_lut(sample_angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_angles = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        return np.zeros(360, dtype=int), np.ones(360, dtype=float)
    angle_step = float(sample_angles[1] - sample_angles[0])
    lower_index = np.zeros(360, dtype=int)
    weight = np.zeros(360, dtype=float)
    for degree in range(1, 361):
        nearest = int(np.argmin(np.abs(float(degree) - sample_angles)))
        value = float(abs(float(degree) - sample_angles[nearest]))
        if float(degree) <= sample_angles[nearest]:
            nearest = max(nearest - 1, 0)
            value = angle_step - value
        lower_index[degree - 1] = nearest
        weight[degree - 1] = value / max(angle_step, 1e-12)
    return lower_index, weight


def _raytrace_psf_struct_matches(
    psf_struct: dict[str, Any] | None,
    optics: dict[str, Any],
    wave: np.ndarray,
    rows: int,
    cols: int,
    sample_spacing_m: float,
    sample_angles_deg: np.ndarray,
) -> bool:
    if not isinstance(psf_struct, dict):
        return False
    if tuple(psf_struct.get("cube_shape", ())) != (rows, cols):
        return False
    if not np.isclose(float(psf_struct.get("sample_spacing_m", -1.0)), float(sample_spacing_m)):
        return False
    if not np.array_equal(np.asarray(psf_struct.get("wavelength_nm", np.empty(0))), np.asarray(wave, dtype=float)):
        return False
    current_angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0)), dtype=float).reshape(-1)
    if not np.array_equal(current_angles, np.asarray(sample_angles_deg, dtype=float).reshape(-1)):
        return False
    raytrace = optics.get("raytrace", {})
    return str(psf_struct.get("optics_name", "")) == str(raytrace.get("name", optics.get("name", "")))


def _raytrace_default_sample_angles(angle_step_deg: float) -> np.ndarray:
    step = float(angle_step_deg)
    if step <= 0.0:
        raise ValueError("PSF angle step must be positive.")
    return np.arange(0.0, 360.0 + step, step, dtype=float)


def _raytrace_requested_sample_angles(
    angle_step_deg: float,
    sample_angles_deg: Any | None,
) -> np.ndarray:
    if sample_angles_deg is None:
        return _raytrace_default_sample_angles(angle_step_deg)
    sample_angles = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        raise ValueError("Ray-trace PSF sample angles must contain at least two entries.")
    if not np.isclose(sample_angles[0], 0.0):
        raise ValueError("Ray-trace PSF sample angles must start at 0 degrees.")
    if sample_angles[-1] < 360.0 and not np.isclose(sample_angles[-1], 360.0):
        raise ValueError("Ray-trace PSF sample angles must include 360 degrees.")
    diffs = np.diff(sample_angles)
    if np.any(diffs <= 0.0):
        raise ValueError("Ray-trace PSF sample angles must be strictly increasing.")
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("Ray-trace PSF sample angles must be uniformly spaced.")
    return sample_angles


def _coerce_psf_stack(value: Any) -> np.ndarray:
    psf = np.asarray(value)
    if psf.dtype != object:
        numeric = np.asarray(value, dtype=float)
        if numeric.ndim != 5:
            raise ValueError("PSF stack must be a 5D array or a 3D object array of 2D kernels.")
        return numeric

    if psf.ndim != 3:
        raise ValueError("MATLAB-style PSF cell arrays must be a 3D object array of 2D kernels.")

    first_kernel: np.ndarray | None = None
    for index in np.ndindex(psf.shape):
        candidate = np.asarray(psf[index], dtype=float)
        if candidate.size == 0:
            continue
        if candidate.ndim != 2:
            raise ValueError("Each PSF kernel must be 2D.")
        first_kernel = candidate
        break
    if first_kernel is None:
        return np.empty(psf.shape + (0, 0), dtype=float)

    stack = np.empty(psf.shape + first_kernel.shape, dtype=float)
    for index in np.ndindex(psf.shape):
        kernel = np.asarray(psf[index], dtype=float)
        if kernel.shape != first_kernel.shape:
            raise ValueError("All PSF kernels must share the same shape.")
        stack[index] = kernel
    return stack


def _coerce_psf_sample_angles(value: Any) -> np.ndarray:
    sample_angles = np.asarray(value, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        raise ValueError("PSF sample angles must contain at least two entries.")
    diffs = np.diff(sample_angles)
    if np.any(diffs <= 0.0):
        raise ValueError("PSF sample angles must be strictly increasing.")
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("PSF sample angles must be uniformly spaced.")
    return sample_angles


def _normalize_psf_struct(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    current = dict(value)
    normalized = dict(current)
    if "psf" in current:
        normalized["psf"] = _coerce_psf_stack(current["psf"])
    sample_angles = current.get("sample_angles_deg", current.get("sampleAngles", current.get("sampAngles")))
    if sample_angles is not None:
        normalized["sample_angles_deg"] = _coerce_psf_sample_angles(sample_angles)
        if normalized["sample_angles_deg"].size >= 2:
            normalized["angle_step_deg"] = float(normalized["sample_angles_deg"][1] - normalized["sample_angles_deg"][0])
    img_height = current.get("img_height_mm", current.get("imgHeight"))
    if img_height is not None:
        normalized["img_height_mm"] = np.asarray(img_height, dtype=float).reshape(-1)
    wavelength = current.get("wavelength_nm", current.get("wavelength"))
    if wavelength is not None:
        normalized["wavelength_nm"] = np.asarray(wavelength, dtype=float).reshape(-1)
    optics_name = current.get("optics_name", current.get("opticsName"))
    if optics_name is not None:
        normalized["optics_name"] = str(optics_name)
    return normalized


def _sync_psf_metadata_fields(oi: OpticalImage) -> None:
    psf_struct = oi.fields.get("psf_struct")
    if not isinstance(psf_struct, dict):
        return
    sample_angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if sample_angles.size > 0:
        oi.fields["psf_sample_angles_deg"] = sample_angles.copy()
        if sample_angles.size >= 2:
            oi.fields["psf_angle_step_deg"] = float(sample_angles[1] - sample_angles[0])
    img_height_mm = np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if img_height_mm.size > 0:
        oi.fields["psf_image_heights_m"] = img_height_mm / 1e3
    wavelength_nm = np.asarray(psf_struct.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if wavelength_nm.size > 0:
        oi.fields["psf_wavelength_nm"] = wavelength_nm.copy()
    optics_name = psf_struct.get("optics_name")
    if optics_name is not None:
        oi.fields["psf_optics_name"] = str(optics_name)


def _spatial_unit_scale(unit: Any | None) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _raw_raytrace_psf_data(oi: OpticalImage) -> dict[str, Any]:
    return dict(oi.fields["optics"].get("raytrace", {}).get("psf", {}))


def _raw_raytrace_psf_size(oi: OpticalImage) -> tuple[int, int]:
    psf_function = np.asarray(_raw_raytrace_psf_data(oi).get("function", np.empty(0, dtype=float)), dtype=float)
    if psf_function.ndim < 2:
        return (0, 0)
    return int(psf_function.shape[0]), int(psf_function.shape[1])


def _raw_raytrace_psf_spacing(oi: OpticalImage, unit: Any | None = None) -> np.ndarray:
    spacing_mm = np.asarray(_raw_raytrace_psf_data(oi).get("sample_spacing_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if spacing_mm.size == 0:
        return np.empty(0, dtype=float)
    return (spacing_mm / 1e3) * _spatial_unit_scale(unit)


def _raw_raytrace_support_axis(length: int, spacing: float) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=float)
    return np.arange((-length / 2.0) + 1.0, (length / 2.0) + 1.0, dtype=float) * float(spacing)


def _raw_raytrace_psf_support_axes(oi: OpticalImage, unit: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _raw_raytrace_psf_size(oi)
    spacing = _raw_raytrace_psf_spacing(oi, unit)
    if spacing.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    y_support = _raw_raytrace_support_axis(rows, float(spacing[0]))
    x_support = _raw_raytrace_support_axis(cols, float(spacing[1]))
    return x_support, y_support


def _raw_raytrace_frequency_axes(oi: OpticalImage, unit: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _raw_raytrace_psf_size(oi)
    spacing = _raw_raytrace_psf_spacing(oi, unit)
    if rows <= 0 or cols <= 0 or spacing.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    fy = _raw_raytrace_support_axis(rows, 1.0 / max(rows * float(spacing[0]), 1e-12))
    fx = _raw_raytrace_support_axis(cols, 1.0 / max(cols * float(spacing[1]), 1e-12))
    return fx, fy


def _raytrace_padding_pixels(psf_struct: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(psf_struct, dict):
        return (0, 0)
    psf = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)))
    if psf.ndim != 5 or psf.shape[3] <= 0 or psf.shape[4] <= 0:
        return (0, 0)
    rows = int(psf.shape[3])
    cols = int(psf.shape[4])
    row_extent = int(np.floor(rows / 2.0))
    col_extent = int(np.floor(cols / 2.0))
    pad_rows = int(2 * np.ceil((np.ceil(row_extent) + 2.0) / 2.0))
    pad_cols = int(2 * np.ceil((np.ceil(col_extent) + 2.0) / 2.0))
    return pad_rows, pad_cols


def _raytrace_precompute_psf(
    optics: dict[str, Any],
    wave: np.ndarray,
    rows: int,
    cols: int,
    sample_spacing_m: float,
    *,
    sample_angles_deg: np.ndarray,
) -> dict[str, Any]:
    raytrace = dict(optics.get("raytrace", {}))
    psf_data = dict(raytrace.get("psf", {}))
    psf_function = np.asarray(psf_data.get("function", np.empty(0, dtype=float)), dtype=float)
    field_heights_all = np.asarray(psf_data.get("field_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    angle_step_deg = float(sample_angles_deg[1] - sample_angles_deg[0]) if sample_angles_deg.size >= 2 else 0.0
    if psf_function.size == 0 or field_heights_all.size == 0:
        return {
            "optics_name": str(raytrace.get("name", optics.get("name", "raytrace"))),
            "psf": np.empty((0, 0, 0, 0, 0), dtype=np.float32),
            "sample_angles_deg": np.empty(0, dtype=float),
            "img_height_mm": np.empty(0, dtype=float),
            "wavelength_nm": np.asarray(wave, dtype=float).copy(),
            "angle_step_deg": float(angle_step_deg),
            "cube_shape": (rows, cols),
            "sample_spacing_m": float(sample_spacing_m),
        }

    sample_spacing_mm = float(sample_spacing_m) * 1e3
    _, data_height_mm = _raytrace_data_angle_height(rows, cols, sample_spacing_mm)
    img_height_mm = _raytrace_sample_heights(field_heights_all, data_height_mm)
    source_x_mm, source_y_mm = _raytrace_psf_support_axes(psf_data)
    target_x_mm, target_y_mm = _raytrace_psf_grid(sample_spacing_mm, psf_data)
    n_angles = int(sample_angles_deg.size)
    n_heights = int(img_height_mm.size)
    n_wave = int(np.asarray(wave, dtype=float).size)
    psf_stack = np.empty((n_angles, n_heights, n_wave, target_y_mm.size, target_x_mm.size), dtype=np.float32)
    psf_wavelengths = np.asarray(psf_data.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)

    for wave_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        source_wave_index = int(np.argmin(np.abs(psf_wavelengths - float(wavelength_nm)))) if psf_wavelengths.size > 0 else 0
        for height_index, height_mm in enumerate(img_height_mm):
            source_height_index = int(np.argmin(np.abs(field_heights_all - float(height_mm))))
            current = np.asarray(psf_function[:, :, source_height_index, source_wave_index], dtype=float)
            current = np.rot90(current, 2)
            for angle_index, sample_angle in enumerate(sample_angles_deg):
                rotation_deg = 0.0 if height_index < 2 else float(sample_angle)
                rotated = rotate(current, rotation_deg, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
                rotated = uniform_filter(rotated, size=3, mode="nearest")
                kernel = _raytrace_resample_psf(rotated, source_x_mm, source_y_mm, target_x_mm, target_y_mm)
                psf_stack[angle_index, height_index, wave_index, :, :] = kernel.astype(np.float32, copy=False)

    lower_index, angle_weight = _raytrace_angle_lut(sample_angles_deg)
    return {
        "optics_name": str(raytrace.get("name", optics.get("name", "raytrace"))),
        "psf": psf_stack,
        "sample_angles_deg": sample_angles_deg,
        "img_height_mm": img_height_mm,
        "wavelength_nm": np.asarray(wave, dtype=float).copy(),
        "angle_step_deg": float(angle_step_deg),
        "cube_shape": (rows, cols),
        "sample_spacing_m": float(sample_spacing_m),
        "target_x_mm": target_x_mm,
        "target_y_mm": target_y_mm,
        "angle_lut_index": lower_index,
        "angle_lut_weight": angle_weight,
    }


def _raytrace_apply_psf(
    cube: np.ndarray,
    psf_struct: dict[str, Any],
    sample_spacing_m: float,
    *,
    pad_pixels: tuple[int, int] = (0, 0),
) -> np.ndarray:
    psf_stack = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)), dtype=float)
    if psf_stack.size == 0:
        return np.asarray(cube, dtype=float).copy()

    rows, cols = cube.shape[:2]
    sample_spacing_mm = float(sample_spacing_m) * 1e3
    data_angle, data_height = _raytrace_data_angle_height(rows, cols, sample_spacing_mm)
    img_height_mm = np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    angle_lut_index = np.asarray(psf_struct.get("angle_lut_index", np.zeros(360, dtype=int)), dtype=int).reshape(-1)
    angle_lut_weight = np.asarray(psf_struct.get("angle_lut_weight", np.ones(360, dtype=float)), dtype=float).reshape(-1)
    if img_height_mm.size < 2:
        return np.asarray(cube, dtype=float).copy()

    pad_rows = int(pad_pixels[0])
    pad_cols = int(pad_pixels[1])
    lower_angle = angle_lut_index[np.clip(data_angle - 1, 0, 359)]
    lower_weight = angle_lut_weight[np.clip(data_angle - 1, 0, 359)]
    unique_angles = np.unique(lower_angle)
    if pad_rows > 0 or pad_cols > 0:
        padded_shape = (rows + 2 * pad_rows, cols + 2 * pad_cols)
    else:
        padded_shape = (rows, cols)
    result = np.zeros((padded_shape[0], padded_shape[1], cube.shape[2]), dtype=float)

    for wave_index in range(cube.shape[2]):
        plane = np.asarray(cube[:, :, wave_index], dtype=float)
        plane_result = np.zeros(padded_shape, dtype=float)
        for height_index in range(1, img_height_mm.size):
            band_mask = (data_height >= img_height_mm[height_index - 1]) & (data_height < img_height_mm[height_index])
            if not np.any(band_mask):
                continue
            inner_weight = 1.0 - (
                np.abs(data_height - img_height_mm[height_index - 1])
                / max(img_height_mm[height_index] - img_height_mm[height_index - 1], 1e-12)
            )
            for angle_index in unique_angles:
                mask = band_mask & (lower_angle == angle_index)
                if not np.any(mask):
                    continue
                masked_plane = plane * mask
                radial_inner = inner_weight * mask
                angular_lower = lower_weight * mask
                if pad_rows > 0 or pad_cols > 0:
                    masked_plane = np.pad(masked_plane, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                    radial_inner = np.pad(radial_inner, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                    angular_lower = np.pad(angular_lower, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                plane_result = plane_result + fftconvolve(
                    masked_plane * radial_inner * angular_lower,
                    psf_stack[angle_index, height_index - 1, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * (1.0 - radial_inner) * angular_lower,
                    psf_stack[angle_index, height_index, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * radial_inner * (1.0 - angular_lower),
                    psf_stack[angle_index + 1, height_index - 1, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * (1.0 - radial_inner) * (1.0 - angular_lower),
                    psf_stack[angle_index + 1, height_index, wave_index, :, :],
                    mode="same",
                )
        result[:, :, wave_index] = plane_result

    return result


def oi_compute(
    oi: OpticalImage,
    scene: Scene,
    *args: Any,
    pad_value: str = "zero",
    crop: bool = False,
    pixel_size: float | None = None,
    aperture: np.ndarray | None = None,
) -> OpticalImage:
    """Compute a supported optical image from a scene."""

    del args
    optics = dict(oi.fields["optics"])
    compute_scene = scene
    if pixel_size is not None:
        from .scene import scene_set

        compute_scene = scene.clone()
        scene_cols = int(compute_scene.fields["cols"])
        focal_length_m = float(optics["focal_length_m"])
        w_angular = float(np.rad2deg(2.0 * np.arctan((float(pixel_size) * scene_cols / 2.0) / focal_length_m)))
        compute_scene = scene_set(compute_scene, "wAngular", w_angular)
    if param_format(optics.get("model", "")) == "raytrace":
        object_distance = float(optics.get("raytrace", {}).get("object_distance_m", compute_scene.fields.get("distance_m", np.inf)))
        if not np.isclose(float(compute_scene.fields.get("distance_m", np.inf)), object_distance):
            compute_scene = compute_scene.clone()
            compute_scene.fields["distance_m"] = object_distance
    scene_photons = np.asarray(compute_scene.data["photons"], dtype=float)
    wave = np.asarray(compute_scene.fields["wave"], dtype=float)
    image_distance_m, width_m, height_m = _oi_geometry(optics, compute_scene)
    sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
    photons = _radiance_to_irradiance(scene_photons, optics, compute_scene)
    if param_format(optics.get("offaxis_method", "cos4th")) == "cos4th":
        photons = photons * _cos4th_factor(photons.shape[0], photons.shape[1], optics, compute_scene)[:, :, None]
    extra_blur = float(optics.get("aberration_scale", 0.0))

    pad_pixels = (
        int(np.round(scene_photons.shape[0] / 8.0)),
        int(np.round(scene_photons.shape[1] / 8.0)),
    )
    model = param_format(optics.get("model", ""))
    psf_struct: dict[str, Any] | None = None
    if model == "raytrace":
        angle_step_deg = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
        sample_angles_deg = _raytrace_requested_sample_angles(
            angle_step_deg,
            oi.fields.get("psf_sample_angles_deg"),
        )
        current_psf_struct = oi.fields.get("psf_struct")
        if _raytrace_psf_struct_matches(
            current_psf_struct,
            optics,
            wave,
            scene_photons.shape[0],
            scene_photons.shape[1],
            sample_spacing_m,
            sample_angles_deg,
        ):
            psf_struct = dict(current_psf_struct)
        else:
            psf_struct = _raytrace_precompute_psf(
                optics,
                wave,
                scene_photons.shape[0],
                scene_photons.shape[1],
                sample_spacing_m,
                sample_angles_deg=sample_angles_deg,
            )
        pad_pixels = _raytrace_padding_pixels(psf_struct)
        depth_map = _pad_depth_map(compute_scene, pad_pixels)
        result = _raytrace_apply_psf(
            _raytrace_geometry(photons, wave, optics, compute_scene),
            psf_struct,
            sample_spacing_m,
            pad_pixels=pad_pixels,
        )
        if crop and (pad_pixels[0] > 0 or pad_pixels[1] > 0):
            row_slice = slice(pad_pixels[0], None if pad_pixels[0] == 0 else -pad_pixels[0])
            col_slice = slice(pad_pixels[1], None if pad_pixels[1] == 0 else -pad_pixels[1])
            result = result[row_slice, col_slice, :]
            depth_map = depth_map[row_slice, col_slice]
    else:
        padded, blur_mode, blur_cval = _pad_scene(photons, pad_pixels, pad_value)
        if model == "diffractionlimited":
            otf = _diffraction_otf(padded.shape[:2], sample_spacing_m, wave, optics, compute_scene)
            blurred = _apply_otf(padded, otf)
        elif model == "skip":
            blurred = padded
        elif model == "shiftinvariant":
            psf_stack = _wvf_psf_stack(padded.shape[:2], sample_spacing_m, wave, optics, aperture=aperture)
            blurred = _apply_psf(padded, psf_stack)
            if extra_blur > 0.0:
                blurred = apply_channelwise_gaussian(
                    blurred,
                    np.full(wave.shape, extra_blur, dtype=float),
                    mode=blur_mode,
                    cval=blur_cval,
                )
        else:
            sigmas = np.array(
                [
                    gaussian_sigma_pixels(
                        float(optics["f_number"]),
                        float(wavelength),
                        sample_spacing_m,
                        extra_blur_pixels=extra_blur,
                    )
                    for wavelength in wave
                ],
                dtype=float,
            )
            blurred = apply_channelwise_gaussian(padded, sigmas, mode=blur_mode, cval=blur_cval)

        pad_rows, pad_cols = pad_pixels
        depth_map = _pad_depth_map(compute_scene, pad_pixels)
        if crop and (pad_rows > 0 or pad_cols > 0):
            row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
            col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
            result = blurred[row_slice, col_slice, :]
            depth_map = depth_map[row_slice, col_slice]
        else:
            result = blurred

    output_sample_spacing_m = float(sample_spacing_m)
    output_width_m = float(result.shape[1] * output_sample_spacing_m)
    output_height_m = float(result.shape[0] * output_sample_spacing_m)
    output_fov_deg = float(np.rad2deg(2.0 * np.arctan2(output_width_m / 2.0, image_distance_m)))
    output_vfov_deg = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))

    if crop:
        focal_length_m = float(optics["focal_length_m"])
        output_fov_deg = float(
            np.rad2deg(2.0 * np.arctan2((result.shape[1] * sample_spacing_m) / 2.0, focal_length_m))
        )
        output_width_m = float(2.0 * image_distance_m * np.tan(np.deg2rad(output_fov_deg) / 2.0))
        output_sample_spacing_m = output_width_m / max(result.shape[1], 1)
        output_height_m = float(result.shape[0] * output_sample_spacing_m)
        output_vfov_deg = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))

    computed = oi.clone()
    computed.name = compute_scene.name
    computed.fields["wave"] = wave
    computed.fields["compute_method"] = optics.get("compute_method", computed.fields.get("compute_method", "opticsotf"))
    computed.fields["pad_value"] = pad_value
    computed.fields["crop"] = bool(crop)
    computed.fields["padding_pixels"] = pad_pixels
    computed.fields["sample_spacing_m"] = output_sample_spacing_m
    computed.fields["image_distance_m"] = image_distance_m
    computed.fields["depth_map_m"] = depth_map
    computed.fields["width_m"] = output_width_m
    computed.fields["height_m"] = output_height_m
    computed.fields["fov_deg"] = output_fov_deg
    computed.fields["vfov_deg"] = output_vfov_deg
    if model == "raytrace":
        computed.fields["psf_angle_step_deg"] = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
        computed.fields["psf_struct"] = psf_struct
        _sync_psf_metadata_fields(computed)
    computed.data["photons"] = result
    if pixel_size is not None:
        computed = _oi_spatial_resample(computed, float(pixel_size), method="linear")
        computed.fields["sample_spacing_m"] = float(pixel_size)
        computed.fields["requested_pixel_size_m"] = float(pixel_size)
    return computed


def _oi_shape(oi: OpticalImage) -> tuple[int, int]:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.ndim >= 2:
        return int(photons.shape[0]), int(photons.shape[1])
    return int(oi.fields.get("rows", 0)), int(oi.fields.get("cols", 0))


def _oi_depth_distance_m(oi: OpticalImage) -> float | None:
    depth_map = oi.fields.get("depth_map_m")
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=float)
    positive = depth[depth > 0.0]
    if positive.size == 0:
        return None
    if depth.shape[0] < 20 or depth.shape[1] < 20:
        return float(np.mean(positive))
    center_half = 10
    row_center = depth.shape[0] // 2
    col_center = depth.shape[1] // 2
    center = depth[
        max(row_center - center_half, 0) : min(row_center + center_half, depth.shape[0]),
        max(col_center - center_half, 0) : min(col_center + center_half, depth.shape[1]),
    ]
    center_positive = center[center > 0.0]
    if center_positive.size > 0:
        return float(np.mean(center_positive))
    return float(np.mean(positive))


def _oi_image_distance_m(oi: OpticalImage) -> float:
    if oi.fields.get("image_distance_m") is not None:
        return float(oi.fields["image_distance_m"])
    scene_distance = _oi_depth_distance_m(oi)
    focal_length = float(oi.fields["optics"]["focal_length_m"])
    if scene_distance is not None:
        if param_format(oi.fields["optics"].get("model", "")) == "skip":
            return focal_length
        if scene_distance <= focal_length:
            return focal_length
        return float(1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12))
    return float(oi.fields["optics"]["focal_length_m"])


def _oi_width_m(oi: OpticalImage) -> float:
    rows, cols = _oi_shape(oi)
    if oi.fields.get("width_m") is not None:
        return float(oi.fields["width_m"])
    image_distance = _oi_image_distance_m(oi)
    fov_deg = float(oi.fields.get("fov_deg", 10.0))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(fov_deg) / 2.0)
    if cols > 0 and rows > 0:
        return float(width_m)
    return float(width_m)


def _oi_height_m(oi: OpticalImage) -> float:
    rows, cols = _oi_shape(oi)
    if oi.fields.get("height_m") is not None:
        return float(oi.fields["height_m"])
    width_m = _oi_width_m(oi)
    if rows > 0 and cols > 0:
        return float(width_m * rows / max(cols, 1))
    image_distance = _oi_image_distance_m(oi)
    vfov_deg = float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    return float(2.0 * image_distance * np.tan(np.deg2rad(vfov_deg) / 2.0))


def _oi_sample_size_m(oi: OpticalImage) -> float | None:
    sample_spacing = oi.fields.get("sample_spacing_m")
    if sample_spacing is not None:
        return float(sample_spacing)
    _, cols = _oi_shape(oi)
    if cols <= 0:
        return None
    return float(_oi_width_m(oi) / cols)


def _oi_distance_per_degree_m(oi: OpticalImage) -> float:
    return _oi_width_m(oi) / max(float(oi.fields.get("fov_deg", 10.0)), 1e-12)


def _oi_spatial_support_linear(oi: OpticalImage) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    spatial = np.asarray(oi_get(oi, "distancepersample"), dtype=float)
    if rows <= 0 or cols <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    y = np.linspace(-(rows * spatial[0]) / 2.0 + spatial[0] / 2.0, (rows * spatial[0]) / 2.0 - spatial[0] / 2.0, rows)
    x = np.linspace(-(cols * spatial[1]) / 2.0 + spatial[1] / 2.0, (cols * spatial[1]) / 2.0 - spatial[1] / 2.0, cols)
    return x, y


def _oi_angular_support_linear_deg(oi: OpticalImage) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    angular = np.asarray(oi_get(oi, "angularresolution"), dtype=float)
    if rows <= 0 or cols <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    x = angular[1] * (np.arange(cols, dtype=float) + 1.0)
    y = angular[0] * (np.arange(rows, dtype=float) + 1.0)
    return x - np.mean(x), y - np.mean(y)


def _oi_frequency_support_1d(oi: OpticalImage, unit: str = "cyclesperdegree") -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    fov_width = float(oi.fields.get("fov_deg", 10.0))
    fov_height = float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    max_frequency_cpd = np.array(
        [
            (cols / 2.0) / max(fov_width, 1e-12),
            (rows / 2.0) / max(fov_height, 1e-12),
        ],
        dtype=float,
    )
    normalized_unit = param_format(unit)
    if normalized_unit in {"cyclesperdegree", "cycperdeg"}:
        max_frequency = max_frequency_cpd
    elif normalized_unit in {"meters", "m", "millimeters", "mm", "microns", "um"}:
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        deg_per_dist = 1.0 / max(_oi_distance_per_degree_m(oi) * unit_scale[normalized_unit], 1e-12)
        max_frequency = max_frequency_cpd * deg_per_dist
    else:
        raise KeyError(f"Unsupported oiGet frequency unit: {unit}")
    fx = unit_frequency_list(cols) * max_frequency[0]
    fy = unit_frequency_list(rows) * max_frequency[1]
    return fx, fy


def _sync_oi_geometry_fields(oi: OpticalImage) -> None:
    rows, cols = _oi_shape(oi)
    oi.fields["rows"] = rows
    oi.fields["cols"] = cols
    if rows <= 0 or cols <= 0:
        return

    image_distance = _oi_image_distance_m(oi)
    fov_deg = float(oi.fields.get("fov_deg", 10.0))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(fov_deg) / 2.0)
    sample_spacing_m = width_m / cols
    height_m = sample_spacing_m * rows
    vfov_deg = float(np.rad2deg(2.0 * np.arctan2(height_m / 2.0, image_distance)))

    oi.fields["image_distance_m"] = image_distance
    oi.fields["width_m"] = float(width_m)
    oi.fields["height_m"] = float(height_m)
    oi.fields["sample_spacing_m"] = float(sample_spacing_m)
    oi.fields["vfov_deg"] = vfov_deg


def oi_get(oi: OpticalImage, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "type":
        return oi.type
    if key == "name":
        return oi.name
    if key == "metadata":
        return oi.metadata
    if key == "data":
        return oi.data
    if key == "wave":
        return np.asarray(oi.fields["wave"], dtype=float)
    if key == "photons":
        return np.asarray(oi.data["photons"], dtype=float)
    if key == "depthmap":
        depth_map = oi.fields.get("depth_map_m")
        if depth_map is None:
            rows, cols = _oi_shape(oi)
            if rows <= 0 or cols <= 0:
                depth_map = np.empty((0, 0), dtype=float)
            else:
                scene_distance = _oi_depth_distance_m(oi)
                if scene_distance is None:
                    scene_distance = 1.2
                depth_map = np.full((rows, cols), float(scene_distance), dtype=float)
        scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]), 1.0) if args else 1.0
        return np.asarray(depth_map, dtype=float) * scale
    if key == "depthrange":
        depth_map = np.asarray(oi_get(oi, "depth map"), dtype=float)
        positive = depth_map[depth_map > 0.0]
        if positive.size == 0:
            return np.empty(0, dtype=float)
        scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]), 1.0) if args else 1.0
        return np.array([positive.min(), positive.max()], dtype=float) * scale
    if key == "rows":
        return _oi_shape(oi)[0]
    if key == "cols":
        return _oi_shape(oi)[1]
    if key == "size":
        return _oi_shape(oi)
    if key in {"imagedistance", "focalplanedistance", "distance"}:
        return _oi_image_distance_m(oi)
    if key in {"wangular", "widthangular", "hfov", "horizontalfieldofview", "fov"}:
        return float(oi.fields.get("fov_deg", 10.0))
    if key in {"hangular", "heightangular", "vfov", "verticalfieldofview"}:
        return float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    if key == "width":
        return _oi_width_m(oi)
    if key == "height":
        return _oi_height_m(oi)
    if key in {"diagonal", "diagonalsize"}:
        return float(np.hypot(_oi_height_m(oi), _oi_width_m(oi)))
    if key in {"heightwidth", "heightandwidth"}:
        return np.array([_oi_height_m(oi), _oi_width_m(oi)], dtype=float)
    if key in {"samplespacing", "sample spacing"}:
        rows, cols = _oi_shape(oi)
        width_spacing = 0.0 if cols <= 0 else _oi_width_m(oi) / cols
        height_spacing = 0.0 if rows <= 0 else _oi_height_m(oi) / rows
        return np.array([width_spacing, height_spacing], dtype=float)
    if key in {"samplesize"}:
        return _oi_sample_size_m(oi)
    if key in {"hspatialresolution", "heightspatialresolution", "hres"}:
        rows, _ = _oi_shape(oi)
        return 0.0 if rows <= 0 else float(_oi_height_m(oi) / rows)
    if key in {"wspatialresolution", "widthspatialresolution", "wres"}:
        _, cols = _oi_shape(oi)
        return 0.0 if cols <= 0 else float(_oi_width_m(oi) / cols)
    if key in {"spatialresolution", "distancepersample", "distpersamp"}:
        return np.array(
            [
                oi_get(oi, "hspatialresolution"),
                oi_get(oi, "wspatialresolution"),
            ],
            dtype=float,
        )
    if key in {"distanceperdegree", "distperdeg"}:
        unit = param_format(args[0]) if args else "m"
        val = _oi_distance_per_degree_m(oi)
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        return val * unit_scale.get(unit, 1.0)
    if key in {"degreesperdistance", "degperdist"}:
        unit = args[0] if args else "m"
        return 1.0 / max(float(oi_get(oi, "distanceperdegree", unit)), 1e-12)
    if key in {"hangularresolution", "heightangularresolution"}:
        spatial = float(oi_get(oi, "hspatialresolution"))
        image_distance = _oi_image_distance_m(oi)
        return float(np.rad2deg(2.0 * np.arctan((spatial / image_distance) / 2.0)))
    if key in {"wangularresolution", "widthangularresolution"}:
        spatial = float(oi_get(oi, "wspatialresolution"))
        image_distance = _oi_image_distance_m(oi)
        return float(np.rad2deg(2.0 * np.arctan((spatial / image_distance) / 2.0)))
    if key in {"angularresolution", "degperpixel", "degpersample", "degreepersample", "degreeperpixel"}:
        return np.array(
            [
                oi_get(oi, "hangularresolution"),
                oi_get(oi, "wangularresolution"),
            ],
            dtype=float,
        )
    if key in {"spatialsupportlinear"}:
        x, y = _oi_spatial_support_linear(oi)
        unit = param_format(args[0]) if args else "m"
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        scale = unit_scale.get(unit, 1.0)
        return {"x": x * scale, "y": y * scale}
    if key in {"spatialsupportmesh", "spatialsupport"}:
        support = oi_get(oi, "spatialsupportlinear", *(args[:1] if args else ()))
        xx, yy = np.meshgrid(support["x"], support["y"])
        return np.stack((xx, yy), axis=2)
    if key in {"angularsupport", "angularsamplingpositions"}:
        x, y = _oi_angular_support_linear_deg(oi)
        unit = param_format(args[0]) if args else "deg"
        if unit == "deg":
            scale = 1.0
        elif unit == "min":
            scale = 60.0
        elif unit == "sec":
            scale = 3600.0
        elif unit == "radians":
            scale = np.pi / 180.0
        else:
            raise KeyError(f"Unsupported oiGet angular support unit: {args[0]}")
        xx, yy = np.meshgrid(x * scale, y * scale)
        return np.stack((xx, yy), axis=2)
    if key in {"frequencyresolution", "freqres"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, fy = _oi_frequency_support_1d(oi, str(unit))
        return {"fx": fx, "fy": fy}
    if key in {"maxfrequencyresolution", "maxfreqres"}:
        unit = args[0] if args else "cyclesperdegree"
        support = oi_get(oi, "frequencyresolution", unit)
        return float(max(np.max(support["fx"]), np.max(support["fy"])))
    if key in {"frequencysupport", "fsupportxy", "fsupport2d", "fsupport"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, fy = _oi_frequency_support_1d(oi, str(unit))
        xx, yy = np.meshgrid(fx, fy)
        return np.stack((xx, yy), axis=2)
    if key in {"frequencysupportcol", "fsupportx"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, _ = _oi_frequency_support_1d(oi, str(unit))
        return fx
    if key in {"frequencysupportrow", "fsupporty"}:
        unit = args[0] if args else "cyclesperdegree"
        _, fy = _oi_frequency_support_1d(oi, str(unit))
        return fy
    if key == "optics":
        return oi.fields["optics"]
    if key in {"focallength", "opticsfocallength"}:
        return float(oi.fields["optics"]["focal_length_m"])
    if key in {"fnumber", "opticsfnumber"}:
        return float(oi.fields["optics"]["f_number"])
    if key in {"opticsmodel", "model"}:
        return oi.fields["optics"]["model"]
    if key in {"computemethod"}:
        return oi.fields["optics"].get("compute_method", oi.fields.get("compute_method"))
    if key in {"diffusermethod"}:
        return oi.fields.get("diffuser_method", "skip")
    if key in {"diffuserblur"}:
        return float(oi.fields.get("diffuser_blur_m", 0.0))
    if key in {"offaxismethod", "opticsoffaxismethod"}:
        return oi.fields["optics"].get("offaxis_method", "cos4th")
    if key in {"opticswvf"}:
        return oi.fields["optics"].get("wavefront")
    if key in {"opticsraytrace", "raytrace", "rt"}:
        return oi.fields["optics"].get("raytrace")
    if key in {"raytraceopticsname", "psfopticsname"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict) and "optics_name" in psf_struct:
            return psf_struct["optics_name"]
        stored = oi.fields.get("psf_optics_name")
        if stored is not None:
            return str(stored)
        raytrace = oi.fields["optics"].get("raytrace", {})
        return raytrace.get("name", oi.fields["optics"].get("name"))
    if key in {"psfstruct", "shiftvariantstructure"}:
        return oi.fields.get("psf_struct")
    if key in {"svpsf", "sampledrtpsf", "shiftvariantpsf"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return psf_struct.get("psf")
        return None
    if key in {"psfsampleangles"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float)
            if angles.size > 0:
                return angles
        stored = oi.fields.get("psf_sample_angles_deg")
        if stored is not None:
            return np.asarray(stored, dtype=float).reshape(-1)
        return _raytrace_default_sample_angles(float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG)))
    if key in {"psfanglestep"}:
        sample_angles = np.asarray(oi_get(oi, "psf sample angles"), dtype=float).reshape(-1)
        if sample_angles.size >= 2:
            return float(sample_angles[1] - sample_angles[0])
        return float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
    if key in {"psfimageheights"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]) if args else "mm", 1.0)
            return np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float) / 1e3 * scale
        stored = oi.fields.get("psf_image_heights_m")
        if stored is not None:
            scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]) if args else "mm", 1.0)
            return np.asarray(stored, dtype=float).reshape(-1) * scale
        return np.empty(0, dtype=float)
    if key in {"psfwavelength"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
        stored = oi.fields.get("psf_wavelength_nm")
        if stored is not None:
            return np.asarray(stored, dtype=float).reshape(-1)
        return np.empty(0, dtype=float)
    if key in {"psfimageheightsn", "psfimageheightcount"}:
        return int(np.asarray(oi_get(oi, "psf image heights"), dtype=float).size)
    if key in {"psfwavelengthn", "psfwn"}:
        return int(np.asarray(oi_get(oi, "psf wavelength"), dtype=float).size)
    if key in {"psfspatialsupportx"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("target_x_mm", np.empty(0, dtype=float)), dtype=float)
        return np.empty(0, dtype=float)
    if key in {"psfspatialsupporty"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("target_y_mm", np.empty(0, dtype=float)), dtype=float)
        return np.empty(0, dtype=float)
    if key in {"rtpsfsize"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            psf = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)))
            if psf.ndim == 5 and psf.shape[3] > 0 and psf.shape[4] > 0:
                return (int(psf.shape[3]), int(psf.shape[4]))
        return (0, 0)
    if key in {"rtobjectdistance", "rtobjdist", "rtreferenceobjectdistance"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("object_distance_m", np.inf))
    if key in {"rtfov"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("max_fov_deg", np.inf))
    if key in {"rtpsffieldheight"}:
        field_height_mm = np.asarray(_raw_raytrace_psf_data(oi).get("field_height_mm", np.empty(0, dtype=float)), dtype=float)
        return (field_height_mm / 1e3) * _spatial_unit_scale(args[0] if args else None)
    if key in {"rtpsfwavelength"}:
        return np.asarray(_raw_raytrace_psf_data(oi).get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
    if key in {"rtpsfsamplespacing", "rtpsfspacing"}:
        return _raw_raytrace_psf_spacing(oi, args[0] if args else None)
    if key in {"rtsupport", "rtpsfsupport"}:
        x_support, y_support = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        if x_support.size == 0 or y_support.size == 0:
            return np.empty((0, 0, 2), dtype=float)
        xx, yy = np.meshgrid(x_support, y_support)
        return np.stack((xx, yy), axis=2)
    if key in {"rtpsfsupportrow", "rtpsfsupporty"}:
        _, y_support = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        return y_support.reshape(-1, 1)
    if key in {"rtpsfsupportcol", "rtpsfsupportx"}:
        x_support, _ = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        return x_support.reshape(-1)
    if key in {"rtfreqsupportcol", "rtfreqsupportx"}:
        fx, _ = _raw_raytrace_frequency_axes(oi, args[0] if args else None)
        return fx.reshape(-1)
    if key in {"rtfreqsupportrow", "rtfreqsupporty"}:
        _, fy = _raw_raytrace_frequency_axes(oi, args[0] if args else None)
        return fy.reshape(-1, 1)
    if key in {"rtfreqsupport"}:
        fx, fy = _raw_raytrace_frequency_axes(oi, args[0] if args else "mm")
        return {"fx": fx, "fy": fy}
    if key in {"transmittance", "transmittancescale", "lenstransmittance", "opticstransmittance", "opticstransmittancescale"}:
        target_wave = np.asarray(args[0], dtype=float).reshape(-1) if args else np.asarray(oi.fields["wave"], dtype=float)
        return _optics_transmittance_scale(oi.fields["optics"], target_wave)
    if key in {"transmittancewave", "opticstransmittancewave"}:
        transmittance = _ensure_optics_transmittance(oi.fields["optics"], wave=np.asarray(oi.fields["wave"], dtype=float))
        return np.asarray(transmittance["wave"], dtype=float).copy()
    if key in {"transmittancenwave", "opticstransmittancenwave"}:
        transmittance = _ensure_optics_transmittance(oi.fields["optics"], wave=np.asarray(oi.fields["wave"], dtype=float))
        return int(np.asarray(transmittance["wave"], dtype=float).size)
    if key in {"crop"}:
        return bool(oi.fields.get("crop", False))
    if key in {"padvalue"}:
        return oi.fields.get("pad_value", "zero")
    if key in {"paddingpixels"}:
        return tuple(oi.fields.get("padding_pixels", (0, 0)))
    raise KeyError(f"Unsupported oiGet parameter: {parameter}")


def oi_set(oi: OpticalImage, parameter: str, value: Any) -> OpticalImage:
    key = param_format(parameter)
    if key == "name":
        oi.name = str(value)
        return oi
    if key == "wave":
        oi.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key == "photons":
        oi.data["photons"] = np.asarray(value, dtype=float)
        _sync_oi_geometry_fields(oi)
        return oi
    if key == "depthmap":
        depth_map = np.asarray(value, dtype=float)
        rows, cols = _oi_shape(oi)
        if rows > 0 and cols > 0 and depth_map.shape != (rows, cols):
            raise ValueError("Depth map must match the optical image size.")
        oi.fields["depth_map_m"] = depth_map
        return oi
    if key in {"wangular", "widthangular", "hfov", "horizontalfieldofview", "fov"}:
        oi.fields["fov_deg"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"hangular", "heightangular", "vfov", "verticalfieldofview"}:
        oi.fields["vfov_deg"] = float(value)
        return oi
    if key in {"imagedistance", "focalplanedistance", "distance"}:
        oi.fields["image_distance_m"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key == "optics":
        oi.fields["optics"] = dict(value)
        oi.fields["compute_method"] = oi.fields["optics"].get("compute_method", oi.fields.get("compute_method"))
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"focallength", "opticsfocallength"}:
        oi.fields["optics"]["focal_length_m"] = float(value)
        if oi.fields.get("image_distance_m") is None:
            oi.fields["image_distance_m"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"fnumber", "opticsfnumber"}:
        oi.fields["optics"]["f_number"] = float(value)
        return oi
    if key in {"opticsmodel", "model"}:
        oi.fields["optics"]["model"] = str(value)
        return oi
    if key in {"transmittance", "transmittancescale", "opticstransmittance", "opticstransmittancescale"}:
        transmittance = _ensure_optics_transmittance(oi.fields["optics"], wave=np.asarray(oi.fields["wave"], dtype=float))
        scale = np.asarray(value, dtype=float).reshape(-1)
        if scale.size != np.asarray(transmittance["wave"], dtype=float).size:
            raise ValueError("Transmittance must match wave dimension.")
        if np.any((scale < 0.0) | (scale > 1.0)):
            raise ValueError("Transmittance should be in [0, 1].")
        transmittance["scale"] = scale
        return oi
    if key in {"transmittancewave", "opticstransmittancewave"}:
        transmittance = _ensure_optics_transmittance(oi.fields["optics"], wave=np.asarray(oi.fields["wave"], dtype=float))
        old_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
        old_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
        new_wave = np.asarray(value, dtype=float).reshape(-1)
        transmittance["wave"] = new_wave
        transmittance["scale"] = np.interp(new_wave, old_wave, old_scale)
        return oi
    if key in {"computemethod"}:
        oi.fields["compute_method"] = str(value)
        oi.fields["optics"]["compute_method"] = str(value)
        return oi
    if key in {"diffusermethod"}:
        oi.fields["diffuser_method"] = str(value)
        return oi
    if key in {"diffuserblur"}:
        oi.fields["diffuser_blur_m"] = float(value)
        return oi
    if key in {"offaxismethod", "opticsoffaxismethod"}:
        oi.fields["optics"]["offaxis_method"] = str(value)
        return oi
    if key == "opticswvf":
        oi.fields["optics"]["wavefront"] = dict(value)
        return oi
    if key in {"psfstruct", "shiftvariantstructure"}:
        oi.fields["psf_struct"] = _normalize_psf_struct(value)
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"svpsf", "sampledrtpsf", "shiftvariantpsf"}:
        current = dict(oi.fields.get("psf_struct") or {})
        current["psf"] = _coerce_psf_stack(value)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfanglestep"}:
        oi.fields["psf_angle_step_deg"] = float(value)
        oi.fields["psf_sample_angles_deg"] = None
        oi.fields["psf_struct"] = None
        return oi
    if key in {"psfsampleangles"}:
        sample_angles = _raytrace_requested_sample_angles(DEFAULT_RAYTRACE_ANGLE_STEP_DEG, value)
        oi.fields["psf_angle_step_deg"] = float(sample_angles[1] - sample_angles[0])
        oi.fields["psf_sample_angles_deg"] = sample_angles
        oi.fields["psf_struct"] = None
        return oi
    if key in {"psfopticsname", "raytraceopticsname"}:
        oi.fields["psf_optics_name"] = str(value)
        current = dict(oi.fields.get("psf_struct") or {})
        current["optics_name"] = str(value)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfimageheights"}:
        oi.fields["psf_image_heights_m"] = np.asarray(value, dtype=float).reshape(-1) / 1e3
        current = dict(oi.fields.get("psf_struct") or {})
        current["img_height_mm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfwavelength"}:
        oi.fields["psf_wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        current = dict(oi.fields.get("psf_struct") or {})
        current["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"rtpsfwavelength"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = None
        return oi
    if key in {"rtpsffieldheight"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["field_height_mm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = None
        return oi
    if key in {"rtpsfsamplespacing", "rtpsfspacing"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["sample_spacing_mm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = None
        return oi
    if key in {"opticsraytrace", "raytrace", "rt"}:
        oi.fields["optics"]["raytrace"] = dict(value)
        oi.fields["optics"]["model"] = "raytrace"
        return oi
    raise KeyError(f"Unsupported oiSet parameter: {parameter}")
