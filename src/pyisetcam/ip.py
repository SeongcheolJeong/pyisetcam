"""Image processing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import convolve2d

from .assets import AssetStore
from .color import sensor_to_xyz_matrix
from .display import Display, display_create
from .exceptions import UnsupportedOptionError
from .types import ImageProcessor, Sensor
from .utils import invert_gamma_table, linear_to_srgb, param_format, tile_pattern, xyz_to_linear_srgb


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def ip_create(
    ip_name: str = "default",
    sensor: Sensor | None = None,
    display: Display | str | None = None,
    l3: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> ImageProcessor:
    """Create an image processor."""

    del l3
    store = _store(asset_store)
    ip = ImageProcessor(name=str(ip_name))
    if sensor is not None:
        ip.fields["wave"] = np.asarray(sensor.fields["wave"], dtype=float)
    else:
        ip.fields["wave"] = np.arange(400.0, 701.0, 10.0, dtype=float)
    if display is None:
        ip.fields["display"] = display_create("lcdExample.mat", wave=ip.fields["wave"], asset_store=store)
    elif isinstance(display, str):
        ip.fields["display"] = display_create(display, wave=ip.fields["wave"], asset_store=store)
    else:
        ip.fields["display"] = display
    ip.fields.update(
        {
            "transform_method": "adaptive",
            "demosaic_method": "bilinear",
            "illuminant_correction_method": "none",
            "internal_cs": "xyz",
            "conversion_method_sensor": "mcc optimized",
            "render": {"renderflag": "rgb", "scale": True},
        }
    )
    ip.data["input"] = None if sensor is None else sensor.data.get("dv", sensor.data.get("volts"))
    ip.fields["datamax"] = None if sensor is None else float(sensor.fields["pixel"]["voltage_swing"])
    return ip


def _ie_bilinear(planes: np.ndarray, cfa_pattern: np.ndarray) -> np.ndarray:
    rows, cols, nplanes = planes.shape
    extended = np.pad(planes, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    rgb = np.zeros((rows, cols, nplanes), dtype=float)
    for channel_index in range(nplanes):
        plane = extended[:, :, channel_index]
        mask = cfa_pattern == (channel_index + 1)
        if (mask[0, 0] and mask[-1, -1]) or (mask[0, -1] and mask[-1, 0]):
            kernel = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], dtype=float)
            rgb[:, :, channel_index] = convolve2d(plane, kernel, mode="valid")
        else:
            horizontal = convolve2d(plane, np.array([[0.5, 1.0, 0.5]], dtype=float), mode="valid")
            rgb[:, :, channel_index] = convolve2d(horizontal, np.array([[0.5], [1.0], [0.5]], dtype=float), mode="valid")
    return rgb


def _sensor_space(sensor: Sensor) -> np.ndarray:
    volts = sensor.data.get("volts")
    if volts is None:
        raise ValueError("Sensor has no computed volts.")
    if np.asarray(volts).ndim == 3 and not sensor.fields["mosaic"]:
        return np.asarray(volts, dtype=float)
    if np.asarray(volts).ndim == 3:
        return np.asarray(volts, dtype=float)
    if sensor.fields["mosaic"]:
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        rows, cols = np.asarray(volts).shape
        nfilters = int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
        tiled = tile_pattern(pattern, rows, cols)
        planes = np.zeros((rows, cols, nfilters), dtype=float)
        for channel_index in range(nfilters):
            mask = tiled == (channel_index + 1)
            planes[:, :, channel_index][mask] = np.asarray(volts, dtype=float)[mask]
        if nfilters == 1:
            return np.repeat(planes, 3, axis=2)
        return _ie_bilinear(planes, pattern)
    return np.repeat(np.asarray(volts, dtype=float)[..., None], 3, axis=2)


def ip_compute(
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    hdr_white: bool = False,
    hdr_level: float = 0.95,
    wgt_blur: float = 2.0,
    network_demosaic: str | None = None,
    asset_store: AssetStore | None = None,
) -> ImageProcessor:
    """Compute the default image processing pipeline."""

    del wgt_blur, network_demosaic
    store = _store(asset_store)
    computed = ip.clone()
    computed.data["input"] = sensor.data.get("dv", sensor.data.get("volts"))
    sensor_space = _sensor_space(sensor)
    if sensor_space.shape[2] == 1:
        sensor_space = np.repeat(sensor_space, 3, axis=2)
    if sensor_space.shape[2] > 3:
        raise UnsupportedOptionError("ipCompute", f"{sensor_space.shape[2]}-channel sensor space")

    filter_spectra = np.asarray(sensor.fields["filter_spectra"], dtype=float)
    if filter_spectra.shape[1] == 1:
        xyz = np.repeat(sensor_space, 3, axis=2)
    else:
        transform = sensor_to_xyz_matrix(np.asarray(sensor.fields["wave"], dtype=float), filter_spectra, asset_store=store)
        xyz = sensor_space @ transform

    linear_rgb = xyz_to_linear_srgb(xyz)
    linear_rgb = np.clip(linear_rgb, 0.0, None)

    if hdr_white:
        max_channel = np.max(linear_rgb, axis=2, keepdims=True)
        blend = np.clip((max_channel - hdr_level) / max(1e-6, 1.0 - hdr_level), 0.0, 1.0)
        linear_rgb = linear_rgb * (1.0 - blend) + blend

    if computed.fields["render"]["scale"]:
        scale = np.percentile(linear_rgb, 99.5)
        if scale > 0.0:
            linear_rgb = linear_rgb / scale

    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
    display = computed.fields["display"]
    display_rgb = invert_gamma_table(linear_rgb, np.asarray(display.fields["gamma"], dtype=float))
    srgb = linear_to_srgb(linear_rgb)

    computed.data["sensorspace"] = sensor_space
    computed.data["xyz"] = xyz
    computed.data["display_rgb"] = display_rgb
    computed.data["srgb"] = srgb
    computed.data["result"] = linear_rgb
    return computed


def ip_get(ip: ImageProcessor, parameter: str) -> Any:
    key = param_format(parameter)
    if key == "type":
        return ip.type
    if key == "name":
        return ip.name
    if key == "wave":
        return np.asarray(ip.fields["wave"], dtype=float)
    if key == "display":
        return ip.fields["display"]
    if key == "input":
        return ip.data.get("input")
    if key == "sensorspace":
        return ip.data.get("sensorspace")
    if key == "result":
        return ip.data.get("result")
    if key == "srgb":
        return ip.data.get("srgb")
    if key == "datamax":
        return ip.fields.get("datamax")
    raise KeyError(f"Unsupported ipGet parameter: {parameter}")


def ip_set(ip: ImageProcessor, parameter: str, value: Any) -> ImageProcessor:
    key = param_format(parameter)
    if key == "name":
        ip.name = str(value)
        return ip
    if key == "display":
        ip.fields["display"] = value
        return ip
    if key == "input":
        ip.data["input"] = np.asarray(value, dtype=float)
        return ip
    if key == "result":
        ip.data["result"] = np.asarray(value, dtype=float)
        return ip
    raise KeyError(f"Unsupported ipSet parameter: {parameter}")

