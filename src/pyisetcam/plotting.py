"""Headless plot-data wrappers for selected MATLAB plotting APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import UnsupportedOptionError
from .ip import ip_get
from .metrics import xyz_to_lab, xyz_to_luv
from .optics import oi_get
from .scene import scene_get
from .sensor import sensor_get
from .types import ImageProcessor, OpticalImage, Scene, Sensor
from .utils import param_format


def _roi_required(function_name: str, plot_type: str, roi_locs: Any | None) -> Any:
    if roi_locs is None:
        raise ValueError(f"ROI required for {function_name}(..., '{plot_type}').")
    return roi_locs


def _roi_payload(roi_locs: Any) -> dict[str, Any]:
    from .roi import ie_locs2_rect

    roi = np.asarray(roi_locs, dtype=int)
    payload: dict[str, Any] = {"roiLocs": roi.copy()}
    if roi.ndim == 1 and roi.size == 4:
        payload["rect"] = roi.copy()
    elif roi.ndim == 2 and roi.shape[1] == 2:
        payload["rect"] = ie_locs2_rect(roi)
    return payload


def _line_index(function_name: str, plot_type: str, xy: Any | None, orientation: str) -> tuple[int, np.ndarray]:
    if xy is None:
        raise ValueError(f"Line selector required for {function_name}(..., '{plot_type}').")
    xy_array = np.rint(np.asarray(xy, dtype=float)).astype(int).reshape(-1)
    if xy_array.size == 1:
        return int(xy_array[0]), xy_array.copy()
    if xy_array.size != 2:
        raise ValueError("Line selector must be a scalar index or [col, row].")
    index = int(xy_array[1] if orientation == "h" else xy_array[0])
    return index, xy_array.copy()


def _sensor_plot_line_data(sensor: Sensor, line_key: str, xy: Any) -> dict[str, Any]:
    key = param_format(line_key)
    orientation = "h" if "hline" in key else "v"
    data_type = "electrons" if "electrons" in key else "dv" if "dv" in key else "volts"
    line_index, xy_array = _line_index("plotSensor", line_key, xy, orientation)
    profile = sensor_get(sensor, f"{orientation}line {data_type}", line_index)
    if profile is None:
        raise ValueError(f"Sensor has no {data_type} data for {line_key}.")
    return {
        "xy": xy_array,
        "ori": orientation,
        "dataType": data_type,
        "data": [np.asarray(values, dtype=float).copy() for values in profile["data"]],
        "pos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pos"]],
        "pixPos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pixPos"]],
    }


def _sensor_plot_histogram(sensor: Sensor, data_type: str, roi_locs: Any) -> dict[str, Any]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    data = np.asarray(vc_get_roi_data(sensor, roi, data_type), dtype=float)
    payload = _roi_payload(roi)
    payload["data"] = data
    payload["unitType"] = data_type
    return payload


def _ip_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line", xy, orientation)
    data = ip_get(ip, "result")
    if data is None:
        raise ValueError("IP has no result data for line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError("IP result data must be an RGB image for line plotting.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        values = np.asarray(array[line_index - 1, :, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        values = np.asarray(array[:, line_index - 1, :], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, values.shape[0] + 1, dtype=float),
        "values": values.copy(),
    }


def _ip_luminance_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line luminance", xy, orientation)
    data = ip_get(ip, "data luminance")
    if data is None:
        raise ValueError("IP has no XYZ data for luminance line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("IP luminance data must be a 2D image.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        line = np.asarray(array[line_index - 1, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        line = np.asarray(array[:, line_index - 1], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, line.shape[0] + 1, dtype=float),
        "data": line.copy(),
    }


def _ip_plot_color_data(ip: ImageProcessor, roi_locs: Any) -> tuple[np.ndarray, np.ndarray]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    rgb = np.asarray(vc_get_roi_data(ip, roi, "result"), dtype=float)
    xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    return rgb, xyz


def _ip_plot_white_point(ip: ImageProcessor, roi_xyz: np.ndarray) -> np.ndarray:
    white_point = ip_get(ip, "data white point")
    if white_point is not None:
        return np.asarray(white_point, dtype=float).reshape(3)
    return np.mean(np.asarray(roi_xyz, dtype=float), axis=0).reshape(3)


def scene_plot(
    scene: Scene,
    p_type: str = "hlineluminance",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotScene` user-data without opening a figure."""

    key = param_format(p_type)
    if key == "radianceenergyroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        energy = np.mean(np.asarray(scene_get(scene, "roi energy", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "energy": energy}, None
    if key == "radiancephotonsroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        photons = np.mean(np.asarray(scene_get(scene, "roi photons", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "photons": photons}, None
    if key in {"radiancehline", "hlineradiance", "radiancevline", "vlineradiance", "luminancehline", "hlineluminance", "luminancevline", "vlineluminance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        udata = dict(scene_get(scene, p_type, roi, *(args[:1] if args else ()), asset_store=asset_store))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key in {"reflectanceroi", "reflectance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        reflectance = np.mean(np.asarray(scene_get(scene, "roi reflectance", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "reflectance": reflectance}, None
    if key == "luminanceroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        return {"lum": np.asarray(scene_get(scene, "roi luminance", roi, asset_store=asset_store), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"chromaticityroi", "chromaticity"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        data = np.asarray(scene_get(scene, "chromaticity", roi, asset_store=asset_store), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"illuminantenergyroi", "illuminantenergy"}:
        if "roi" in key:
            roi = _roi_required("scenePlot", p_type, roi_locs)
            energy = np.asarray(scene_get(scene, "roi mean illuminant energy", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            energy = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float).reshape(-1)
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "energy": energy,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    if key in {"illuminantphotonsroi", "illuminantphotons"}:
        if "roi" in key:
            roi = _roi_required("scenePlot", p_type, roi_locs)
            photons = np.asarray(scene_get(scene, "roi mean illuminant photons", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            photons = np.asarray(scene_get(scene, "illuminant photons", asset_store=asset_store), dtype=float).reshape(-1)
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "photons": photons,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    raise UnsupportedOptionError("scenePlot", p_type)


def oi_plot(
    oi: OpticalImage,
    p_type: str = "illuminance hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `oiPlot` user-data without opening a figure."""

    key = param_format(p_type)
    if key == "irradiancephotonsroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi photons", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "irradianceenergyroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi energy", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {
        "irradiancehline",
        "hline",
        "hlineirradiance",
        "irradiancevline",
        "vline",
        "vlineirradiance",
        "irradianceenergyhline",
        "hlineenergy",
        "hlineirradianceenergy",
        "irradianceenergyvline",
        "vlineenergy",
        "vlineirradianceenergy",
        "illuminancehline",
        "horizontallineilluminance",
        "hlineilluminance",
        "illuminancevline",
        "vlineilluminance",
    }:
        roi = _roi_required("oiPlot", p_type, roi_locs)
        udata = dict(oi_get(oi, p_type, roi, *(args[:1] if args else ())))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key == "illuminanceroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        return {"illum": np.asarray(oi_get(oi, "roi illuminance", roi), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "chromaticityroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        data = np.asarray(oi_get(oi, "chromaticity", roi), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    raise UnsupportedOptionError("oiPlot", p_type)


def sensor_plot(
    sensor: Sensor,
    p_type: str = "volts hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensor` user-data without opening a figure."""

    del args
    key = param_format(p_type)
    if key in {"electronshline", "hlineelectrons", "electronsvline", "vlineelectrons", "voltshline", "hlinevolts", "voltsvline", "vlinevolts", "dvhline", "hlinedv", "dvvline", "vlinedv"}:
        xy = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_line_data(sensor, key, xy), None
    if key in {"voltshistogram", "voltshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "volts", roi), None
    if key in {"electronshistogram", "electronshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "electrons", roi), None
    if key in {"dvhistogram", "dvhist", "digitalcountshistogram", "digitalcountshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "dv", roi), None
    raise UnsupportedOptionError("plotSensor", p_type)


def ip_plot(
    ip: ImageProcessor,
    p_type: str = "horizontal line",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `ipPlot` user-data without opening a figure."""

    del args, asset_store
    key = param_format(p_type)
    if key in {"horizontalline", "hline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "h", xy), None
    if key in {"verticalline", "vline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "v", xy), None
    if key in {"horizontallineluminance", "hlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "h", xy), None
    if key in {"verticallineluminance", "vlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "v", xy), None
    if key == "chromaticity":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        data = np.asarray(ip_get(ip, "chromaticity", roi), dtype=float)
        xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
        payload = _roi_payload(roi)
        payload["x"] = data[:, 0].copy()
        payload["y"] = data[:, 1].copy()
        payload["XYZ"] = xyz.copy()
        return payload, None
    if key in {"rgbhistogram", "rgb"}:
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        payload["meanRGB"] = np.mean(rgb, axis=0).reshape(-1)
        return payload, None
    if key == "rgb3d":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        return payload, None
    if key == "luminance":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        luminance = np.asarray(xyz[:, 1], dtype=float).reshape(-1)
        payload = _roi_payload(roi)
        payload["luminance"] = luminance.copy()
        payload["meanL"] = float(np.mean(luminance))
        payload["stdLum"] = float(np.std(luminance))
        return payload, None
    if key == "cielab":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        lab = np.asarray(xyz_to_lab(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LAB"] = lab.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLAB"] = np.mean(lab, axis=0).reshape(-1)
        return payload, None
    if key == "cieluv":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        luv = np.asarray(xyz_to_luv(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LUV"] = luv.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLUV"] = np.mean(luv, axis=0).reshape(-1)
        return payload, None
    raise UnsupportedOptionError("ipPlot", p_type)


plotScene = scene_plot
oiPlot = oi_plot
plotSensor = sensor_plot
ipPlot = ip_plot
