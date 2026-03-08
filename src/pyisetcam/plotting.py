"""Headless plot-data wrappers for selected MATLAB plotting APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import UnsupportedOptionError
from .optics import oi_get
from .scene import scene_get
from .types import OpticalImage, Scene
from .utils import param_format


def _roi_required(function_name: str, plot_type: str, roi_locs: Any | None) -> Any:
    if roi_locs is None:
        raise ValueError(f"ROI required for {function_name}(..., '{plot_type}').")
    return roi_locs


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


plotScene = scene_plot
oiPlot = oi_plot
