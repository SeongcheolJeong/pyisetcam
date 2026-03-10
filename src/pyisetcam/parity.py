"""Curated parity case runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_create
from .display import display_create
from .metrics import xyz_from_energy
from .ip import ip_compute, ip_create
from .optics import (
    _cos4th_factor,
    _oi_geometry,
    _pad_scene,
    _radiance_to_irradiance,
    _wvf_psf_stack,
    oi_compute,
    oi_create,
)
from .scene import scene_adjust_illuminant, scene_create, scene_get
from .sensor import sensor_compute, sensor_create, sensor_create_ideal, sensor_set
from .utils import blackbody, energy_to_quanta, quanta_to_energy, unit_frequency_list


@dataclass
class ParityCaseResult:
    payload: dict[str, Any]
    context: dict[str, Any]


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def run_python_case_with_context(
    case_name: str,
    *,
    asset_store: AssetStore | None = None,
) -> ParityCaseResult:
    store = asset_store or AssetStore.default()

    if case_name == "scene_macbeth_default":
        scene = scene_create(asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_checkerboard_small":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_bb_small":
        scene = scene_create("uniform bb", 16, 4500, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "utility_unit_frequency_list":
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "even": unit_frequency_list(50),
                "odd": unit_frequency_list(51),
            },
            context={},
        )

    if case_name == "utility_energy_quanta_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.1, 3.1, wave.size, dtype=float)
        photons = energy_to_quanta(energy, wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "photons": photons,
                "energy_roundtrip": quanta_to_energy(photons, wave),
            },
            context={},
        )

    if case_name == "metrics_xyz_from_energy_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.05, 1.55, wave.size, dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "xyz": xyz_from_energy(energy, wave, asset_store=store),
            },
            context={},
        )

    if case_name == "scene_illuminant_change":
        scene = scene_create(asset_store=store)
        wave = scene_get(scene, "wave")
        preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=store)
        no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "preserve_mean": scene_get(preserve, "mean luminance", asset_store=store),
                "no_preserve_mean": scene_get(no_preserve, "mean luminance", asset_store=store),
                "preserve_photons": scene_get(preserve, "photons"),
                "no_preserve_photons": scene_get(no_preserve, "photons"),
            },
            context={
                "scene": scene,
                "preserve_scene": preserve,
                "no_preserve_scene": no_preserve,
            },
        )

    if case_name == "display_create_lcd_example":
        display = display_create("lcdExample.mat", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": display.fields["wave"],
                "spd": display.fields["spd"],
                "gamma": display.fields["gamma"],
            },
            context={"display": display},
        )

    if case_name == "oi_diffraction_limited_default":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_wvf_small_scene":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi_seed = oi_create("wvf")
        optics = dict(oi_seed.fields["optics"])
        scene_photons = np.asarray(scene.data["photons"], dtype=float)
        wave = np.asarray(scene.fields["wave"], dtype=float)
        _, width_m, _ = _oi_geometry(optics, scene)
        sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
        pre_psf_photons = _radiance_to_irradiance(scene_photons, optics, scene)
        if str(optics.get("offaxis_method", "")).replace(" ", "").lower() == "cos4th":
            pre_psf_photons = pre_psf_photons * _cos4th_factor(
                pre_psf_photons.shape[0],
                pre_psf_photons.shape[1],
                optics,
                scene,
            )[:, :, None]
        pad_pixels = (
            int(np.round(scene_photons.shape[0] / 8.0)),
            int(np.round(scene_photons.shape[1] / 8.0)),
        )
        pre_psf_photons, _, _ = _pad_scene(pre_psf_photons, pad_pixels, "zero")
        psf_stack = _wvf_psf_stack(pre_psf_photons.shape[:2], sample_spacing_m, wave, optics)
        oi = oi_compute(oi_seed, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "pre_psf_photons": pre_psf_photons,
                "psf_stack": psf_stack,
                "photons": oi.data["photons"],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_bayer_noiseless":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "volts": sensor.data["volts"],
                "integration_time": sensor.fields["integration_time"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_monochrome_noise_stats":
        scene = scene_create("uniform d65", 32, asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={"case_name": case_name, **_stats(sensor.data["volts"])},
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "ip_default_pipeline":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_compute(sensor_set(sensor_create(asset_store=store), "noise flag", 0), oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "input": ip.data["input"],
                "sensorspace": ip.data["sensorspace"],
                "result": ip.data["result"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "camera_default_pipeline":
        scene = scene_create(asset_store=store)
        camera = camera_create(asset_store=store)
        camera.fields["sensor"] = sensor_set(camera.fields["sensor"], "noise flag", 0)
        camera = camera_compute(camera, scene, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "result": camera.fields["ip"].data["result"],
                "sensor_volts": camera.fields["sensor"].data["volts"],
                "oi_photons": camera.fields["oi"].data["photons"],
            },
            context={
                "scene": scene,
                "camera": camera,
                "oi": camera.fields["oi"],
                "sensor": camera.fields["sensor"],
                "ip": camera.fields["ip"],
            },
        )

    raise KeyError(f"Unknown parity case: {case_name}")


def run_python_case(case_name: str, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    return run_python_case_with_context(case_name, asset_store=asset_store).payload
