"""Curated parity case runners."""

from __future__ import annotations

from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_create
from .display import display_create
from .ip import ip_compute, ip_create
from .optics import oi_compute, oi_create, wvf_create
from .scene import scene_adjust_illuminant, scene_create, scene_get
from .sensor import sensor_compute, sensor_create, sensor_create_ideal, sensor_set
from .utils import blackbody


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def run_python_case(case_name: str, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    store = asset_store or AssetStore.default()

    if case_name == "scene_macbeth_default":
        scene = scene_create(asset_store=store)
        return {
            "case_name": case_name,
            "wave": scene_get(scene, "wave"),
            "photons": scene_get(scene, "photons"),
            "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
        }

    if case_name == "scene_illuminant_change":
        scene = scene_create(asset_store=store)
        wave = scene_get(scene, "wave")
        preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=store)
        no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=store)
        return {
            "case_name": case_name,
            "preserve_mean": scene_get(preserve, "mean luminance", asset_store=store),
            "no_preserve_mean": scene_get(no_preserve, "mean luminance", asset_store=store),
            "preserve_photons": scene_get(preserve, "photons"),
            "no_preserve_photons": scene_get(no_preserve, "photons"),
        }

    if case_name == "display_create_lcd_example":
        display = display_create("lcdExample.mat", asset_store=store)
        return {
            "case_name": case_name,
            "wave": display.fields["wave"],
            "spd": display.fields["spd"],
            "gamma": display.fields["gamma"],
        }

    if case_name == "oi_diffraction_limited_default":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        return {"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]}

    if case_name == "oi_wvf_small_scene":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi = oi_compute(oi_create("wvf", wvf_create(aberration_scale=0.5)), scene, crop=True)
        return {"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]}

    if case_name == "sensor_bayer_noiseless":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        return {
            "case_name": case_name,
            "volts": sensor.data["volts"],
            "integration_time": sensor.fields["integration_time"],
        }

    if case_name == "sensor_monochrome_noise_stats":
        scene = scene_create("uniform d65", 32, asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_compute(sensor, oi, seed=0)
        return {"case_name": case_name, **_stats(sensor.data["volts"])}

    if case_name == "ip_default_pipeline":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_compute(sensor_set(sensor_create(asset_store=store), "noise flag", 0), oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)
        return {
            "case_name": case_name,
            "input": ip.data["input"],
            "sensorspace": ip.data["sensorspace"],
            "result": ip.data["result"],
        }

    if case_name == "camera_default_pipeline":
        scene = scene_create(asset_store=store)
        camera = camera_compute(camera_create(asset_store=store), scene, asset_store=store)
        return {
            "case_name": case_name,
            "result": camera.fields["ip"].data["result"],
            "sensor_volts": camera.fields["sensor"].data["volts"],
            "oi_photons": camera.fields["oi"].data["photons"],
        }

    raise KeyError(f"Unknown parity case: {case_name}")
