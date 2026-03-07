"""Camera orchestration helpers."""

from __future__ import annotations

from typing import Any

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .ip import ip_compute, ip_create
from .optics import oi_compute, oi_create
from .scene import Scene
from .sensor import sensor_compute, sensor_create, sensor_create_ideal, sensor_get, sensor_set
from .types import Camera, OpticalImage, Sensor
from .utils import param_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def camera_create(camera_type: str = "default", *args: Any, asset_store: AssetStore | None = None) -> Camera:
    """Create a supported camera."""

    del args
    store = _store(asset_store)
    normalized = param_format(camera_type)
    camera = Camera(name=str(camera_type))

    if normalized in {"default"}:
        oi = oi_create()
        sensor = sensor_create(asset_store=store)
    elif normalized in {"ideal"}:
        oi = oi_create()
        sensor = sensor_create_ideal("xyz", asset_store=store)
    elif normalized in {"monochrome"}:
        oi = oi_create()
        sensor = sensor_create("monochrome", asset_store=store)
    elif normalized in {"idealmonochrome"}:
        oi = oi_create()
        sensor = sensor_create_ideal("monochrome", asset_store=store)
    else:
        raise UnsupportedOptionError("cameraCreate", camera_type)

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip_create(sensor=sensor, asset_store=store)
    return camera


def camera_get(camera: Camera, parameter: str) -> Any:
    key = param_format(parameter)
    if key == "type":
        return camera.type
    if key == "name":
        return camera.name
    if key == "oi":
        return camera.fields["oi"]
    if key == "sensor":
        return camera.fields["sensor"]
    if key in {"ip", "vci"}:
        return camera.fields["ip"]
    if key.startswith("ip"):
        return camera.fields["ip"].data.get(key[2:])
    raise KeyError(f"Unsupported cameraGet parameter: {parameter}")


def camera_set(camera: Camera, parameter: str, value: Any) -> Camera:
    key = param_format(parameter)
    if key == "name":
        camera.name = str(value)
        return camera
    if key == "oi":
        camera.fields["oi"] = value
        return camera
    if key == "sensor":
        camera.fields["sensor"] = value
        return camera
    if key in {"ip", "vci"}:
        camera.fields["ip"] = value
        return camera
    raise KeyError(f"Unsupported cameraSet parameter: {parameter}")


def camera_compute(
    camera: Camera,
    p_type: str | Scene = "sensor",
    mode: str = "normal",
    sensor_resize: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Camera:
    """Run the supported camera pipeline."""

    store = _store(asset_store)
    normalized_mode = param_format(mode)
    if normalized_mode != "normal":
        raise UnsupportedOptionError("cameraCompute", mode)

    if isinstance(p_type, Scene):
        scene = p_type
        start_type = "scene"
    else:
        scene = None
        start_type = param_format(p_type)

    oi: OpticalImage = camera.fields["oi"]
    sensor: Sensor = camera.fields["sensor"]
    ip = camera.fields["ip"]

    if start_type == "scene":
        if scene is None:
            raise ValueError("A Scene object is required when starting from scene.")
        if sensor_resize:
            sensor = sensor_set(sensor.clone(), "size", scene.data["photons"].shape[:2])
        oi = oi_compute(oi, scene, crop=True)
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip, sensor, asset_store=store)
    elif start_type == "oi":
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip, sensor, asset_store=store)
    elif start_type == "sensor":
        ip = ip_compute(ip, sensor, asset_store=store)
    else:
        raise UnsupportedOptionError("cameraCompute", str(p_type))

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip
    camera.data["result"] = ip.data.get("result")
    return camera

