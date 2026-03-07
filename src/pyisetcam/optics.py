"""Optical image creation and computation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom

from .exceptions import UnsupportedOptionError
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
DIFFRACTION_CUTOFF_GRID_SCALE = 1.001


def wvf_create(
    *,
    wave: np.ndarray | None = None,
    focal_length_m: float = DEFAULT_WVF_FOCAL_LENGTH_M,
    f_number: float | None = None,
    aberration_scale: float = 0.0,
    measured_pupil_diameter_mm: float = DEFAULT_WVF_MEASURED_PUPIL_MM,
    measured_wavelength_nm: float = DEFAULT_WVF_MEASURED_WAVELENGTH_NM,
    calc_pupil_diameter_mm: float = DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM,
) -> dict[str, Any]:
    calc_pupil = float(calc_pupil_diameter_mm)
    focal_length = float(focal_length_m)
    if f_number is None:
        f_number = (focal_length * 1e3) / max(calc_pupil, 1e-12)
    return {
        "type": "wvf",
        "wave": np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float),
        "focal_length_m": focal_length,
        "f_number": float(f_number),
        "aberration_scale": float(aberration_scale),
        "measured_pupil_diameter_mm": float(measured_pupil_diameter_mm),
        "measured_wavelength_nm": float(measured_wavelength_nm),
        "sample_interval_domain": "psf",
        "spatial_samples": DEFAULT_WVF_SPATIAL_SAMPLES,
        "ref_pupil_plane_size_mm": DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM,
        "calc_pupil_diameter_mm": calc_pupil,
    }


def oi_create(oi_type: str = "diffraction limited", *args: Any) -> OpticalImage:
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

    oi.fields["optics"] = optics
    oi.fields["wave"] = np.asarray(args[1], dtype=float) if len(args) > 1 else DEFAULT_WAVE.copy()
    oi.fields["sample_spacing_m"] = None
    oi.data["photons"] = np.empty((0, 0, 0), dtype=float)
    return oi


def _scene_sample_spacing(scene: Scene) -> float:
    width = float(scene.fields["width_m"])
    cols = int(scene.fields["cols"])
    return width / max(cols, 1)


def _image_distance_m(optics: dict[str, Any], scene: Scene) -> float:
    focal_length = float(optics["focal_length_m"])
    if param_format(optics.get("model", "")) == "skip":
        return focal_length
    scene_distance = float(scene.fields.get("distance_m", np.inf))
    if not np.isfinite(scene_distance) or scene_distance <= focal_length:
        return focal_length
    return 1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12)


def _magnification(optics: dict[str, Any], scene: Scene) -> float:
    if param_format(optics.get("model", "")) == "skip":
        return -1.0
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
    f_number = float(optics["f_number"])
    magnification = _magnification(optics, scene)
    scale = np.pi / (1.0 + 4.0 * (f_number**2) * ((1.0 + abs(magnification)) ** 2))
    return np.asarray(scene_cube, dtype=float) * scale


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
        return np.pad(scene_cube, ((pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0)), mode="edge"), "nearest", 0.0
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


def _diffraction_otf(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    scene: Scene,
) -> np.ndarray:
    rows, cols = shape
    nyquist = 1.0 / max(2.0 * sample_spacing_m, 1e-12)
    fx = np.fft.ifftshift(unit_frequency_list(cols) * nyquist)
    fy = np.fft.ifftshift(unit_frequency_list(rows) * nyquist)
    rho = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    aperture_diameter = float(optics["focal_length_m"]) / max(float(optics["f_number"]), 1e-12)
    # ISETCam's dlMTF() uses opticsGet(optics, 'focalPlaneDistance') without
    # passing the scene distance, which resolves to the focal length rather
    # than the thin-lens image distance for finite scene depth.
    focal_plane_distance = float(optics["focal_length_m"])
    wavelengths_m = np.asarray(wave, dtype=float) * 1e-9
    cutoff = (
        (aperture_diameter / max(focal_plane_distance, 1e-12))
        / np.maximum(wavelengths_m, 1e-12)
        * DIFFRACTION_CUTOFF_GRID_SCALE
    )

    otf = np.zeros((rows, cols, wavelengths_m.size), dtype=float)
    for index, cutoff_frequency in enumerate(cutoff):
        normalized = rho / max(float(cutoff_frequency), 1e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        current = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        current[normalized >= 1.0] = 0.0
        otf[:, :, index] = current
    return otf


def _apply_otf(cube: np.ndarray, otf: np.ndarray) -> np.ndarray:
    result = np.empty_like(cube, dtype=float)
    for band_index in range(cube.shape[2]):
        filtered = np.fft.ifft2(np.fft.fft2(cube[:, :, band_index]) * otf[:, :, band_index])
        result[:, :, band_index] = np.abs(filtered)
    return result


def _wvf_psf_stack(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
) -> np.ndarray:
    rows, cols = shape
    n_pixels = int(max(rows, cols))
    wavefront = dict(optics.get("wavefront", {}))
    measured_pupil_mm = float(wavefront.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM))
    measured_wavelength_nm = float(wavefront.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
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
        pupil_function = (norm_radius < calc_radius).astype(float)

        amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))
        intensity = np.real(amp * np.conj(amp))
        psf_stack[:, :, band_index] = intensity / max(float(np.sum(intensity)), 1e-12)

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

    del args, aperture
    optics = dict(oi.fields["optics"])
    scene_photons = np.asarray(scene.data["photons"], dtype=float)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    image_distance_m, width_m, height_m = _oi_geometry(optics, scene)
    sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
    photons = _radiance_to_irradiance(scene_photons, optics, scene)
    if param_format(optics.get("offaxis_method", "cos4th")) == "cos4th":
        photons = photons * _cos4th_factor(photons.shape[0], photons.shape[1], optics, scene)[:, :, None]
    extra_blur = float(optics.get("aberration_scale", 0.0))

    pad_pixels = (
        int(np.round(scene_photons.shape[0] / 8.0)),
        int(np.round(scene_photons.shape[1] / 8.0)),
    )
    padded, blur_mode, blur_cval = _pad_scene(photons, pad_pixels, pad_value)
    model = param_format(optics.get("model", ""))
    if model == "diffractionlimited":
        otf = _diffraction_otf(padded.shape[:2], sample_spacing_m, wave, optics, scene)
        blurred = _apply_otf(padded, otf)
    elif model == "skip":
        blurred = padded
    elif model == "shiftinvariant":
        psf_stack = _wvf_psf_stack(padded.shape[:2], sample_spacing_m, wave, optics)
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
    if crop and (pad_rows > 0 or pad_cols > 0):
        row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
        col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
        result = blurred[row_slice, col_slice, :]
    else:
        result = blurred

    output_width_m = float(result.shape[1] * sample_spacing_m)
    output_height_m = float(result.shape[0] * sample_spacing_m)

    if pixel_size is not None:
        current_spacing = sample_spacing_m
        factor = current_spacing / float(pixel_size)
        result = zoom(result, (factor, factor, 1.0), order=1)
        sample_spacing_m = float(pixel_size)
        output_width_m = float(result.shape[1] * sample_spacing_m)
        output_height_m = float(result.shape[0] * sample_spacing_m)

    computed = oi.clone()
    computed.name = scene.name
    computed.fields["wave"] = wave
    computed.fields["pad_value"] = pad_value
    computed.fields["crop"] = bool(crop)
    computed.fields["padding_pixels"] = pad_pixels
    computed.fields["sample_spacing_m"] = sample_spacing_m
    computed.fields["image_distance_m"] = image_distance_m
    computed.fields["width_m"] = output_width_m
    computed.fields["height_m"] = output_height_m
    computed.fields["fov_deg"] = float(np.rad2deg(2.0 * np.arctan2(output_width_m / 2.0, image_distance_m)))
    computed.fields["vfov_deg"] = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))
    computed.data["photons"] = result
    return computed


def oi_get(oi: OpticalImage, parameter: str) -> Any:
    key = param_format(parameter)
    if key == "type":
        return oi.type
    if key == "name":
        return oi.name
    if key == "wave":
        return np.asarray(oi.fields["wave"], dtype=float)
    if key == "photons":
        return np.asarray(oi.data["photons"], dtype=float)
    if key in {"focallength", "opticsfocallength"}:
        return float(oi.fields["optics"]["focal_length_m"])
    if key in {"fnumber", "opticsfnumber"}:
        return float(oi.fields["optics"]["f_number"])
    if key in {"opticsmodel", "model"}:
        return oi.fields["optics"]["model"]
    if key == "samplespacing":
        return oi.fields["sample_spacing_m"]
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
        return oi
    if key in {"focallength", "opticsfocallength"}:
        oi.fields["optics"]["focal_length_m"] = float(value)
        return oi
    if key in {"fnumber", "opticsfnumber"}:
        oi.fields["optics"]["f_number"] = float(value)
        return oi
    raise KeyError(f"Unsupported oiSet parameter: {parameter}")
