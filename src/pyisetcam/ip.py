"""Image processing pipeline."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_spectra
from .color import internal_to_display_matrix, sensor_to_target_matrix, xyz_color_matching
from .display import Display, display_create, display_get, display_set
from .exceptions import UnsupportedOptionError
from .metrics import chromaticity_xy, xyz_from_energy
from .session import track_ip_session_state, track_session_object
from .sensor import sensor_get
from .types import ImageProcessor, Sensor, SessionContext
from .utils import (
    image_linear_transform,
    invert_gamma_table,
    linear_to_srgb,
    param_format,
    split_prefixed_parameter,
    tile_pattern,
)


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _copy_metadata_value(value: Any) -> Any:
    return copy.deepcopy(value)


def _ip_chart_parameters(ip: ImageProcessor) -> dict[str, Any]:
    chart = ip.fields.get("chartP")
    if not isinstance(chart, dict):
        chart = {}
        ip.fields["chartP"] = chart
    return chart


def _identity_transform() -> np.ndarray:
    return np.eye(3, dtype=float)


def _ensure_ip_state(ip: ImageProcessor) -> ImageProcessor:
    wave = np.asarray(ip.fields.get("wave", np.arange(400.0, 701.0, 10.0, dtype=float)), dtype=float)
    ip.fields["wave"] = wave
    ip.fields.setdefault("spectrum", {"wave": wave.copy()})
    ip.fields["spectrum"]["wave"] = wave.copy()
    ip.fields.setdefault("display", display_create("default"))
    ip.fields.setdefault("transform_method", "adaptive")
    ip.fields.setdefault("internal_cs", "xyz")
    ip.fields.setdefault("conversion_method_sensor", "mcc optimized")
    ip.fields.setdefault("illuminant_correction_method", "none")
    ip.fields.setdefault("demosaic_method", "bilinear")
    ip.fields.setdefault("render", {"renderflag": 1, "scale": True})
    ip.fields["render"].setdefault("renderflag", 1)
    ip.fields["render"].setdefault("scale", True)
    ip.fields.setdefault("demosaic", {"method": ip.fields["demosaic_method"]})
    ip.fields.setdefault("sensor_correction", {"method": ip.fields["conversion_method_sensor"]})
    ip.fields.setdefault(
        "illuminant_correction",
        {"method": ip.fields["illuminant_correction_method"]},
    )
    ip.fields["demosaic"]["method"] = ip.fields.get("demosaic_method", ip.fields["demosaic"].get("method", "bilinear"))
    ip.fields["sensor_correction"]["method"] = ip.fields.get(
        "conversion_method_sensor",
        ip.fields["sensor_correction"].get("method", "mcc optimized"),
    )
    ip.fields["illuminant_correction"]["method"] = ip.fields.get(
        "illuminant_correction_method",
        ip.fields["illuminant_correction"].get("method", "none"),
    )
    transforms = list(ip.data.get("transforms", [None, None, None]))
    while len(transforms) < 3:
        transforms.append(None)
    ip.data["transforms"] = transforms[:3]
    return ip


def _ip_transform(ip: ImageProcessor, index: int) -> np.ndarray:
    _ensure_ip_state(ip)
    transform = ip.data["transforms"][index]
    if transform is None:
        return _identity_transform()
    return np.asarray(transform, dtype=float)


def ip_create(
    ip_name: str = "default",
    sensor: Sensor | None = None,
    display: Display | str | None = None,
    l3: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Create an image processor."""

    del l3
    store = _store(asset_store)
    ip = ImageProcessor(name=str(ip_name))
    if sensor is not None:
        ip.fields["wave"] = np.asarray(sensor.fields["wave"], dtype=float)
    else:
        ip.fields["wave"] = np.arange(400.0, 701.0, 10.0, dtype=float)
    ip.fields["spectrum"] = {"wave": np.asarray(ip.fields["wave"], dtype=float).copy()}
    if display is None:
        ip.fields["display"] = display_create("lcdExample.mat", wave=ip.fields["wave"], asset_store=store, session=session)
    elif isinstance(display, str):
        ip.fields["display"] = display_create(display, wave=ip.fields["wave"], asset_store=store, session=session)
    else:
        ip.fields["display"] = track_session_object(session, display)
    ip.fields.update(
        {
            "transform_method": "adaptive",
            "demosaic_method": "bilinear",
            "illuminant_correction_method": "none",
            "internal_cs": "xyz",
            "conversion_method_sensor": "mcc optimized",
            "demosaic": {"method": "bilinear"},
            "sensor_correction": {"method": "mcc optimized"},
            "illuminant_correction": {"method": "none"},
            "render": {"renderflag": 1, "scale": True},
        }
    )
    ip.data["input"] = None if sensor is None else sensor.data.get("dv", sensor.data.get("volts"))
    ip.fields["datamax"] = None if sensor is None else float(sensor.fields["pixel"]["voltage_swing"])
    ip.data["transforms"] = [None, None, None]
    return track_ip_session_state(session, _ensure_ip_state(ip))


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
    sensor_data = sensor.data.get("volts")
    if sensor_data is None:
        sensor_data = sensor.data.get("dv")
    if sensor_data is None:
        raise ValueError("Sensor has no computed volts.")
    if np.asarray(sensor_data).ndim == 3 and not sensor.fields["mosaic"]:
        return np.asarray(sensor_data, dtype=float)
    if np.asarray(sensor_data).ndim == 3:
        return np.asarray(sensor_data, dtype=float)
    if sensor.fields["mosaic"]:
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        rows, cols = np.asarray(sensor_data).shape
        nfilters = int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
        tiled = tile_pattern(pattern, rows, cols)
        planes = np.zeros((rows, cols, nfilters), dtype=float)
        for channel_index in range(nfilters):
            mask = tiled == (channel_index + 1)
            planes[:, :, channel_index][mask] = np.asarray(sensor_data, dtype=float)[mask]
        if nfilters == 1:
            return planes
        return _ie_bilinear(planes, pattern)
    return np.repeat(np.asarray(sensor_data, dtype=float)[..., None], 3, axis=2)


def _sensor_to_internal(
    sensor_space: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    conversion_method = param_format(ip.fields.get("conversion_method_sensor", "mcc optimized"))
    filter_spectra = np.asarray(sensor.fields["filter_spectra"], dtype=float)
    wave = np.asarray(sensor.fields["wave"], dtype=float)

    if conversion_method in {"none", "sensor"}:
        transform = np.eye(sensor_space.shape[2], dtype=float)
        internal = sensor_space.copy()
    elif conversion_method in {"current", "currentmatrix", "manualmatrixentry"}:
        transform = _ip_transform(ip, 0)
        internal = sensor_space @ transform
    elif conversion_method in {"mccoptimized", "mcc", "esseroptimized", "esser"}:
        surfaces = "esser" if "esser" in conversion_method else "mcc"
        transform = sensor_to_target_matrix(
            wave,
            filter_spectra,
            target_space="xyz",
            illuminant="D65",
            surfaces=surfaces,
            asset_store=asset_store,
        )
        internal = sensor_space @ transform
    else:
        raise UnsupportedOptionError("ipCompute", conversion_method)

    internal = np.clip(internal, 0.0, None)
    internal_max = float(np.max(internal))
    if internal_max > 0.0:
        internal = internal / internal_max
    return internal, transform


def _illuminant_white_ratio(
    ip: ImageProcessor,
    channels: int,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    internal_cmf = ip_get(ip, "internal cmf")
    if internal_cmf is None:
        return np.ones(int(channels), dtype=float)

    wave = np.asarray(ip_get(ip, "wave"), dtype=float).reshape(-1)
    target = np.asarray(ie_read_spectra("D65", wave, asset_store=asset_store), dtype=float).reshape(-1)
    white_ratio = np.asarray(internal_cmf, dtype=float).T @ target
    max_value = float(np.max(white_ratio))
    if max_value <= 0.0:
        return np.ones(int(channels), dtype=float)
    return np.asarray(white_ratio / max_value, dtype=float).reshape(-1)


def _gray_world_transform(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    data = np.nan_to_num(np.asarray(internal_image, dtype=float), nan=0.0)
    channels = int(data.shape[2])
    averages = np.mean(data, axis=(0, 1))
    white_ratio = _illuminant_white_ratio(ip, channels, asset_store=asset_store)
    reference = max(float(white_ratio[0]), np.finfo(float).tiny)
    white_ratio = white_ratio / reference
    base_average = max(float(averages[0]), np.finfo(float).tiny)
    scale = np.zeros(channels, dtype=float)
    for channel_index in range(channels):
        average = max(float(averages[channel_index]), np.finfo(float).tiny)
        scale[channel_index] = float(white_ratio[channel_index]) * (base_average / average)
    return np.diag(scale)


def _white_world_transform(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    data = np.nan_to_num(np.asarray(internal_image, dtype=float), nan=0.0)
    channels = int(data.shape[2])
    maxima = np.max(data, axis=(0, 1))
    brightest_channel = int(np.argmax(maxima))
    brightest_plane = data[:, :, brightest_channel]
    max_brightness = max(float(maxima[brightest_channel]), np.finfo(float).tiny)
    criterion = 0.7
    white_ratio = _illuminant_white_ratio(ip, channels, asset_store=asset_store)
    reference = max(float(white_ratio[0]), np.finfo(float).tiny)
    white_ratio = white_ratio / reference

    bright_values = np.zeros(channels, dtype=float)
    mask = brightest_plane >= (criterion * max_brightness)
    for channel_index in range(channels):
        channel_data = data[:, :, channel_index][mask]
        if channel_data.size == 0:
            bright_values[channel_index] = np.finfo(float).tiny
        else:
            bright_values[channel_index] = max(float(np.mean(channel_data)), np.finfo(float).tiny)

    base_value = max(float(bright_values[0]), np.finfo(float).tiny)
    scale = np.zeros(channels, dtype=float)
    for channel_index in range(channels):
        scale[channel_index] = float(white_ratio[channel_index]) * (base_value / float(bright_values[channel_index]))
    return np.diag(scale)


def _illuminant_correct_internal(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    method = param_format(ip.fields.get("illuminant_correction_method", "none"))
    channels = int(np.asarray(internal_image, dtype=float).shape[2])

    if method in {"none"}:
        transform = np.eye(channels, dtype=float)
        return np.asarray(internal_image, dtype=float), transform
    if method in {"grayworld"}:
        transform = _gray_world_transform(internal_image, ip, asset_store=asset_store)
        return image_linear_transform(internal_image, transform), transform
    if method in {"whiteworld"}:
        transform = _white_world_transform(internal_image, ip, asset_store=asset_store)
        return image_linear_transform(internal_image, transform), transform
    if method in {"manualmatrixentry", "manual"}:
        transform = _ip_transform(ip, 1)
        return image_linear_transform(internal_image, transform), transform

    raise UnsupportedOptionError("imageIlluminantCorrection", method)


def _display_render(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    internal_cs = str(ip.fields.get("internal_cs", "xyz"))
    display = ip.fields["display"]
    display_spd = np.asarray(display.fields["spd"], dtype=float)
    conversion_method = param_format(ip.fields.get("conversion_method_sensor", "mcc optimized"))

    if param_format(internal_cs) == "xyz":
        transform = internal_to_display_matrix(
            np.asarray(ip.fields["wave"], dtype=float),
            display_spd,
            internal_cs=internal_cs,
            asset_store=asset_store,
        )
    elif param_format(internal_cs) == "sensor":
        sensor_qe = np.asarray(sensor.fields["filter_spectra"], dtype=float)
        transform = np.linalg.pinv(sensor_qe.T @ display_spd).T
    else:
        raise UnsupportedOptionError("displayRender", internal_cs)

    if conversion_method in {"current", "currentmatrix", "manualmatrixentry", "none"}:
        display_linear = internal_image.copy()
    elif conversion_method in {"sensor", "mccoptimized", "mcc", "esseroptimized", "esser"}:
        display_linear = internal_image @ transform
    else:
        raise UnsupportedOptionError("displayRender", conversion_method)

    display_max = float(np.max(display_linear))
    if bool(ip.fields["render"].get("scale", True)) and display_max > 0.0:
        display_linear = display_linear / display_max * float(sensor_get(sensor, "response ratio"))
    display_linear = np.maximum(display_linear, 0.0)
    return display_linear, transform


def ip_compute(
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    hdr_white: bool = False,
    hdr_level: float = 0.95,
    wgt_blur: float = 2.0,
    network_demosaic: str | None = None,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Compute the default image processing pipeline."""

    del wgt_blur, network_demosaic
    store = _store(asset_store)
    computed = _ensure_ip_state(ip.clone())
    computed.data["input"] = sensor.data.get("dv", sensor.data.get("volts"))
    sensor_space = _sensor_space(sensor)
    internal_image, sensor_transform = _sensor_to_internal(sensor_space, computed, sensor, asset_store=store)
    corrected_internal, illuminant_transform = _illuminant_correct_internal(internal_image, computed, asset_store=store)
    display_linear, display_transform = _display_render(corrected_internal, computed, sensor, asset_store=store)

    if hdr_white:
        max_channel = np.max(display_linear, axis=2, keepdims=True)
        blend = np.clip((max_channel - hdr_level) / max(1e-6, 1.0 - hdr_level), 0.0, 1.0)
        display_linear = display_linear * (1.0 - blend) + blend

    display = computed.fields["display"]
    clamped_display = np.clip(display_linear, 0.0, 1.0)
    display_rgb = invert_gamma_table(clamped_display, np.asarray(display.fields["gamma"], dtype=float))
    srgb = linear_to_srgb(clamped_display)

    computed.fields["sensor_conversion_matrix"] = sensor_transform
    computed.fields["illuminant_correction_matrix"] = illuminant_transform
    computed.fields["ics2display"] = display_transform
    computed.data["transforms"] = [
        np.asarray(sensor_transform, dtype=float),
        np.asarray(computed.fields["illuminant_correction_matrix"], dtype=float),
        np.asarray(display_transform, dtype=float),
    ]
    computed.data["sensorspace"] = sensor_space
    computed.data["xyz"] = internal_image
    computed.data["ics"] = corrected_internal
    computed.data["display_rgb"] = display_rgb
    computed.data["srgb"] = srgb
    computed.data["result"] = display_linear
    return track_ip_session_state(session, computed)


def image_data_xyz(
    ip: ImageProcessor,
    roi_locs: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray | None:
    """Convert display-linear image-processor data to XYZ."""

    ip = _ensure_ip_state(ip)
    if roi_locs is None:
        data = ip_get(ip, "result")
    else:
        from .roi import vc_get_roi_data

        data = vc_get_roi_data(ip, roi_locs, "result")
    if data is None:
        return None

    rgb = np.asarray(data, dtype=float)
    rgb_flag = rgb.ndim != 2
    if rgb_flag:
        rows, cols = rgb.shape[:2]
        rgb = rgb.reshape(-1, rgb.shape[2])

    spd = np.asarray(display_get(ip.fields["display"], "rgb spd"), dtype=float)
    wave = np.asarray(display_get(ip.fields["display"], "wave"), dtype=float)
    energy = rgb @ spd.T
    xyz = xyz_from_energy(energy, wave, asset_store=_store(asset_store))

    if rgb_flag:
        xyz = xyz.reshape(rows, cols, 3)
    return xyz


def ip_get(ip: ImageProcessor, parameter: str, *args: Any) -> Any:
    ip = _ensure_ip_state(ip)
    key = param_format(parameter)
    if key in {"result", "displaylinearrgb", "datadisplay", "displaydata"}:
        return ip.data.get("result")
    if key in {"displayviewingdistance"}:
        return display_get(ip.fields["display"], "viewing distance")
    if key in {"displaydpi"}:
        return display_get(ip.fields["display"], "dpi")

    prefix, remainder = split_prefixed_parameter(parameter, ("display", "l3"))
    if prefix == "display":
        if not remainder:
            return ip.fields["display"]
        return display_get(ip.fields["display"], remainder, *args)
    if prefix == "l3":
        if not remainder:
            return ip.fields.get("l3")
        l3 = ip.fields.get("l3")
        if l3 is None:
            return None
        raise KeyError(f"Unsupported ipGet L3 parameter: {parameter}")

    if key == "type":
        return ip.type
    if key == "name":
        return ip.name
    if key in {"spectrum", "spectrumstructure"}:
        return ip.fields["spectrum"]
    if key in {"wave", "wavelength"}:
        return np.asarray(ip.fields["wave"], dtype=float)
    if key in {"binwidth", "waveresolution"}:
        wave = np.asarray(ip.fields["wave"], dtype=float)
        if wave.size < 2:
            return 1.0
        return float(wave[1] - wave[0])
    if key in {"nwave", "nwaves"}:
        return int(np.asarray(ip.fields["wave"], dtype=float).size)
    if key in {"row", "rows"}:
        input_data = ip.data.get("input")
        return None if input_data is None else int(np.asarray(input_data).shape[0])
    if key in {"col", "cols"}:
        input_data = ip.data.get("input")
        return None if input_data is None else int(np.asarray(input_data).shape[1])
    if key == "inputsize":
        input_data = ip.data.get("input")
        return None if input_data is None else tuple(np.asarray(input_data).shape)
    if key in {"rgbsize", "resultsize", "displaysize", "size"}:
        result = ip.data.get("result")
        return None if result is None else tuple(np.asarray(result).shape)
    if key in {"internalcs", "internalcolorspace"}:
        return ip.fields["internal_cs"]
    if key in {"internalcmf", "internalcolormatchingfunction"}:
        if param_format(ip.fields["internal_cs"]) == "sensor":
            return None
        return xyz_color_matching(np.asarray(ip.fields["wave"], dtype=float), asset_store=_store(None))
    if key in {"illuminantcorrection"}:
        return ip.fields["illuminant_correction"]
    if key in {"illuminantcorrectionmethod"}:
        return ip.fields["illuminant_correction"].get("method", "none")
    if key in {"illuminantcorrectionmatrix", "correctiontransformilluminant", "correctionmatrixilluminant"}:
        return _ip_transform(ip, 1)
    if key in {"demosaic", "demosaicstructure"}:
        return ip.fields["demosaic"]
    if key in {"demosaicmethod"}:
        return ip.fields["demosaic"].get("method", "none")
    if key == "chartparameters":
        return copy.deepcopy(_ip_chart_parameters(ip))
    if key in {"cornerpoints", "chartcornerpoints", "chartcorners"}:
        value = _ip_chart_parameters(ip).get("cornerPoints")
        return None if value is None else np.asarray(value).copy()
    if key == "mcccornerpoints":
        return ip_get(ip, "chart corner points")
    if key in {"chartrects", "chartrectangles"}:
        value = _ip_chart_parameters(ip).get("rects")
        return None if value is None else np.asarray(value).copy()
    if key in {"currentrect", "chartcurrentrect"}:
        value = _ip_chart_parameters(ip).get("currentRect")
        return None if value is None else np.asarray(value).copy()
    if key == "mccrecthandles":
        return _copy_metadata_value(ip.fields.get("mccRectHandles"))
    if key in {"sensorconversion", "conversionsensor"}:
        return ip.fields["sensor_correction"]
    if key in {"sensorconversionmethod", "conversionmethodsensor"}:
        return ip.fields["sensor_correction"].get("method", "none")
    if key in {"sensorconversionmatrix", "conversiontransformsensor", "correctionmatrixsensor"}:
        return _ip_transform(ip, 0)
    if key in {"transformcellarray", "transforms"}:
        return list(ip.data["transforms"])
    if key == "transformmethod":
        return ip.fields["transform_method"]
    if key in {"ics2display", "ics2displaymatrix", "ics2displaytransform", "internalcs2displayspace"}:
        return _ip_transform(ip, 2)
    if key in {"transformcombined", "combinedtransform", "prodt"}:
        return _ip_transform(ip, 0) @ _ip_transform(ip, 1) @ _ip_transform(ip, 2)
    if key in {"render", "renderstructure"}:
        return ip.fields["render"]
    if key in {"renderflag", "displaymode"}:
        return ip.fields["render"].get("renderflag", 1)
    if key in {"renderscale", "scaledisplay", "scaledisplayoutput"}:
        return bool(ip.fields["render"].get("scale", True))
    if key in {"data", "datastructure"}:
        return ip.data
    if key in {"roidata", "dataroi", "roiresult"}:
        if not args:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(ip, args[0], "result")
    if key in {"roixyz", "xyzroi"}:
        return image_data_xyz(ip, args[0] if args else None)
    if key in {"chromaticity", "roichromaticity"}:
        xyz = image_data_xyz(ip, args[0] if args else None)
        return None if xyz is None else chromaticity_xy(np.asarray(xyz, dtype=float))
    if key in {"roichromaticitymean", "roimeanchromaticity"}:
        if not args:
            raise ValueError("ROI required for ipGet(..., 'roi chromaticity mean').")
        chromaticity = ip_get(ip, "chromaticity", args[0])
        return None if chromaticity is None else np.mean(np.asarray(chromaticity, dtype=float), axis=0).reshape(-1)
    if key in {"input", "sensorinput", "sensormosaic"}:
        return ip.data.get("input")
    if key in {"sensorspace", "sensorchannels"}:
        return ip.data.get("sensorspace")
    if key == "nsensorchannels":
        sensor_space = ip.data.get("sensorspace")
        if sensor_space is None:
            return None
        sensor_space_array = np.asarray(sensor_space)
        return 1 if sensor_space_array.ndim < 3 else int(sensor_space_array.shape[2])
    if key in {"maximumsensorvalue", "sensormax", "rgbmax", "datamax"}:
        return ip.fields.get("datamax")
    if key in {"datasrgb", "srgb"}:
        return ip.data.get("srgb")
    if key in {"dataxyz", "xyz"}:
        return ip.data.get("xyz")
    if key in {"dataics", "ics"}:
        return ip.data.get("ics", ip.data.get("xyz"))
    if key == "dataluminance":
        xyz = ip.data.get("xyz")
        return None if xyz is None else np.asarray(xyz, dtype=float)[..., 1]
    if key in {"datawhitepoint", "datawp"}:
        return ip.data.get("wp")
    raise KeyError(f"Unsupported ipGet parameter: {parameter}")


def ip_set(
    ip: ImageProcessor,
    parameter: str,
    value: Any,
    *args: Any,
    session: SessionContext | None = None,
) -> ImageProcessor:
    ip = _ensure_ip_state(ip)
    key = param_format(parameter)
    if key == "displayviewingdistance":
        ip.fields["display"] = display_set(ip.fields["display"], "viewing distance", value)
        return track_ip_session_state(session, ip)
    if key == "displaydpi":
        ip.fields["display"] = display_set(ip.fields["display"], "dpi", value)
        return track_ip_session_state(session, ip)

    prefix, remainder = split_prefixed_parameter(parameter, ("display", "l3"))
    if prefix == "display":
        if not remainder:
            ip.fields["display"] = track_session_object(session, value)
        else:
            ip.fields["display"] = display_set(ip.fields["display"], remainder, value, *args)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if prefix == "l3":
        ip.fields["l3"] = value
        return track_ip_session_state(session, ip)

    if key == "type":
        ip.type = str(value)
        return track_ip_session_state(session, ip)
    if key == "name":
        ip.name = str(value)
        return track_ip_session_state(session, ip)
    if key == "chartparameters":
        chart = _ip_chart_parameters(ip)
        chart.update(dict(value))
        return track_ip_session_state(session, ip)
    if key in {"chartcornerpoints", "cornerpoints", "chartcorners"}:
        chart = _ip_chart_parameters(ip)
        chart["cornerPoints"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key == "mcccornerpoints":
        chart = _ip_chart_parameters(ip)
        chart["cornerPoints"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key in {"chartrects", "chartrectangles"}:
        chart = _ip_chart_parameters(ip)
        chart["rects"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key in {"currentrect", "chartcurrentrect"}:
        chart = _ip_chart_parameters(ip)
        chart["currentRect"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key == "mccrecthandles":
        ip.fields["mccRectHandles"] = _copy_metadata_value(value)
        return track_ip_session_state(session, ip)
    if key in {"spectrum"}:
        ip.fields["spectrum"] = dict(value)
        if "wave" in ip.fields["spectrum"]:
            ip.fields["wave"] = np.asarray(ip.fields["spectrum"]["wave"], dtype=float).reshape(-1)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"wave", "wavelength"}:
        ip.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"internalcs", "internalcolorspace"}:
        ip.fields["internal_cs"] = str(value)
        return track_ip_session_state(session, ip)
    if key in {"ics2display", "ics2displaytransform", "internalcs2displayspace"}:
        ip.data["transforms"][2] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"demosaicstructure", "demosaic"}:
        ip.fields["demosaic"] = dict(value)
        ip.fields["demosaic_method"] = str(ip.fields["demosaic"].get("method", "none"))
        return track_ip_session_state(session, ip)
    if key == "demosaicmethod":
        method = "none" if value in {None, ""} else str(value).lower()
        ip.fields["demosaic_method"] = method
        ip.fields["demosaic"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {"sensorconversion", "conversionsensor"}:
        ip.fields["sensor_correction"] = dict(value)
        ip.fields["conversion_method_sensor"] = str(ip.fields["sensor_correction"].get("method", "none"))
        return track_ip_session_state(session, ip)
    if key in {"sensorconversionmethod", "conversionmethodsensor"}:
        method = "none" if value in {None, ""} else str(value)
        ip.fields["conversion_method_sensor"] = method
        ip.fields["sensor_correction"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {"sensorconversionmatrix", "conversiontransformsensor", "conversionmatrixsensor"}:
        ip.data["transforms"][0] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"illuminantcorrection", "correctionilluminant"}:
        ip.fields["illuminant_correction"] = dict(value)
        ip.fields["illuminant_correction_method"] = str(ip.fields["illuminant_correction"].get("method", "none"))
        return track_ip_session_state(session, ip)
    if key in {"illuminantcorrectionmethod", "correctionmethodilluminant"}:
        method = "none" if value in {None, ""} else str(value).lower()
        ip.fields["illuminant_correction_method"] = method
        ip.fields["illuminant_correction"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {"correctionmatrixilluminant", "illuminantcorrectionmatrix", "correctiontransformilluminant", "illuminantcorrectiontransform"}:
        ip.data["transforms"][1] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"display", "displaystructure"}:
        ip.fields["display"] = track_session_object(session, value)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"data", "datastructure"}:
        ip.data = dict(value)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"input", "sensorinput"}:
        ip.data["input"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"result", "displaylinearrgb"}:
        ip.data["result"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"datawhitepoint", "datawp"}:
        ip.data["wp"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key == "sensorspace":
        ip.data["sensorspace"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key == "quantization":
        ip.data["quantization"] = value
        return track_ip_session_state(session, ip)
    if key in {"nbits", "quantizationnbits"}:
        ip.data.setdefault("quantization", {})
        if not isinstance(ip.data["quantization"], dict):
            ip.data["quantization"] = {"method": ip.data["quantization"]}
        ip.data["quantization"]["bits"] = int(value)
        return track_ip_session_state(session, ip)
    if key == "transforms":
        if args:
            index = int(args[0]) - 1
            ip.data["transforms"][index] = np.asarray(value, dtype=float)
        else:
            transforms = list(value)
            while len(transforms) < 3:
                transforms.append(None)
            ip.data["transforms"] = transforms[:3]
        return track_ip_session_state(session, ip)
    if key == "transformmethod":
        ip.fields["transform_method"] = str(value).lower()
        return track_ip_session_state(session, ip)
    if key in {"datamax", "rgbmax", "sensormax", "maximumsensorvalue", "maximumsensorvoltageswing"}:
        ip.fields["datamax"] = float(value)
        return track_ip_session_state(session, ip)
    if key in {"render", "renderstructure"}:
        ip.fields["render"] = dict(value)
        ip.fields["render"].setdefault("renderflag", 1)
        ip.fields["render"].setdefault("scale", True)
        return track_ip_session_state(session, ip)
    if key in {"renderflag", "displaymode"}:
        normalized = param_format(value)
        mapping = {"rgb": 1, "hdr": 2, "gray": 3}
        ip.fields["render"]["renderflag"] = mapping.get(normalized, int(value) if isinstance(value, (int, np.integer)) else 1)
        return track_ip_session_state(session, ip)
    if key in {"renderscale", "scaledisplay", "scaledisplayoutput"}:
        ip.fields["render"]["scale"] = bool(value)
        return track_ip_session_state(session, ip)
    if key in {"gammadisplay", "rendergamma", "gamma"}:
        ip.fields["render"]["gamma"] = value
        return track_ip_session_state(session, ip)
    if key == "renderdemosaiconly" and bool(value):
        ip = ip_set(ip, "internal cs", "Sensor", session=session)
        ip = ip_set(ip, "conversion method sensor", "None", session=session)
        ip = ip_set(ip, "correction method illuminant", "None", session=session)
        ip = ip_set(ip, "transform method", "current", session=session)
        ip = ip_set(ip, "ics2display transform", _identity_transform(), session=session)
        return track_ip_session_state(session, ip)
    raise KeyError(f"Unsupported ipSet parameter: {parameter}")


imageDataXYZ = image_data_xyz
