from __future__ import annotations

import numpy as np

from pyisetcam import (
    camera_compute,
    camera_create,
    ip_compute,
    ip_create,
    oi_compute,
    oi_create,
    scene_create,
    sensor_compute,
    sensor_create,
    sensor_set,
)


def test_oi_compute_matches_scene_wave(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    assert oi.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.array_equal(oi.fields["wave"], scene.fields["wave"])


def test_sensor_compute_noiseless(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0)
    sensor = sensor_compute(sensor, oi, seed=0)
    assert sensor.data["volts"].shape == scene.data["photons"].shape[:2]
    assert np.all(sensor.data["volts"] >= 0.0)


def test_ip_compute_default_pipeline(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_compute(sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0), oi, seed=0)
    ip = ip_compute(ip_create(sensor=sensor, asset_store=asset_store), sensor, asset_store=asset_store)
    assert ip.data["result"].shape[:2] == scene.data["photons"].shape[:2]
    assert ip.data["result"].shape[2] == 3
    assert np.all((ip.data["result"] >= 0.0) & (ip.data["result"] <= 1.0))


def test_camera_compute_end_to_end(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)
    result = camera.fields["ip"].data["result"]
    assert result.shape[:2] == scene.data["photons"].shape[:2]
    assert result.shape[2] == 3

