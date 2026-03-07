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
    spectral_step,
)

DEFAULT_FOCAL_LENGTH_M = 0.003862755099228


def wvf_create(
    *,
    wave: np.ndarray | None = None,
    focal_length_m: float = DEFAULT_FOCAL_LENGTH_M,
    f_number: float = 4.0,
    aberration_scale: float = 0.35,
) -> dict[str, Any]:
    return {
        "type": "wvf",
        "wave": np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float),
        "focal_length_m": float(focal_length_m),
        "f_number": float(f_number),
        "aberration_scale": float(aberration_scale),
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
        }
    elif normalized in {"wvf", "shiftinvariant"}:
        wavefront = args[0] if args and isinstance(args[0], dict) else wvf_create()
        optics = {
            "model": "shiftinvariant",
            "f_number": float(wavefront.get("f_number", 4.0)),
            "focal_length_m": float(wavefront.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M)),
            "compute_method": "opticspsf",
            "aberration_scale": float(wavefront.get("aberration_scale", 0.35)),
            "wavefront": wavefront,
        }
    elif normalized == "pinhole":
        optics = {
            "model": "skip",
            "f_number": 1e-3,
            "focal_length_m": 1e-2,
            "compute_method": "skip",
            "aberration_scale": 0.0,
        }
    elif normalized == "empty":
        optics = {
            "model": "diffractionlimited",
            "f_number": 4.0,
            "focal_length_m": DEFAULT_FOCAL_LENGTH_M,
            "compute_method": "opticsotf",
            "aberration_scale": 0.0,
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


def _pad_scene(scene_cube: np.ndarray, pad_pixels: int, pad_value: str) -> tuple[np.ndarray, str, float]:
    if pad_pixels <= 0:
        return scene_cube, "nearest", 0.0
    mode = param_format(pad_value)
    if mode == "zero":
        return np.pad(scene_cube, ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0))), "constant", 0.0
    if mode == "border":
        return np.pad(scene_cube, ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)), mode="edge"), "nearest", 0.0
    if mode == "mean":
        padded = np.empty(
            (scene_cube.shape[0] + 2 * pad_pixels, scene_cube.shape[1] + 2 * pad_pixels, scene_cube.shape[2]),
            dtype=float,
        )
        band_means = scene_cube.mean(axis=(0, 1))
        for band_index, mean_value in enumerate(band_means):
            padded[:, :, band_index] = mean_value
            padded[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, band_index] = scene_cube[:, :, band_index]
        return padded, "constant", float(np.mean(band_means))
    raise UnsupportedOptionError("oiCompute", pad_value)


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
    photons = np.asarray(scene.data["photons"], dtype=float)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    sample_spacing_m = _scene_sample_spacing(scene)
    extra_blur = float(optics.get("aberration_scale", 0.0))
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

    pad_pixels = int(np.ceil(max(float(sigmas.max()) * 3.0, 0.0))) if sigmas.size else 0
    padded, blur_mode, blur_cval = _pad_scene(photons, pad_pixels, pad_value)
    blurred = apply_channelwise_gaussian(padded, sigmas, mode=blur_mode, cval=blur_cval)

    if crop and pad_pixels > 0:
        result = blurred[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    else:
        result = blurred

    if pixel_size is not None:
        current_spacing = sample_spacing_m
        factor = current_spacing / float(pixel_size)
        result = zoom(result, (factor, factor, 1.0), order=1)
        sample_spacing_m = float(pixel_size)

    computed = oi.clone()
    computed.name = scene.name
    computed.fields["wave"] = wave
    computed.fields["pad_value"] = pad_value
    computed.fields["crop"] = bool(crop)
    computed.fields["padding_pixels"] = pad_pixels
    computed.fields["sample_spacing_m"] = sample_spacing_m
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

