"""Camera orchestration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .ip import ip_compute, ip_create, ip_get, ip_set
from .optics import oi_compute, oi_create, oi_get, oi_set
from .session import track_camera_session_state, track_session_object
from .scene import Scene
from .sensor import sensor_compute, sensor_create, sensor_create_ideal, sensor_get, sensor_set, sensor_set_size_to_fov
from .types import Camera, OpticalImage, Sensor, SessionContext
from .utils import param_format, split_prefixed_parameter


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _pixel_get(sensor: Sensor, parameter: str) -> Any:
    pixel = sensor.fields["pixel"]
    key = param_format(parameter)
    mapping = {
        "size": np.asarray(pixel["size_m"], dtype=float),
        "pixelsize": np.asarray(pixel["size_m"], dtype=float),
        "fillfactor": float(pixel["fill_factor"]),
        "conversiongain": float(pixel["conversion_gain_v_per_electron"]),
        "conversiongainvpelectron": float(pixel["conversion_gain_v_per_electron"]),
        "voltageswing": float(pixel["voltage_swing"]),
        "darkvoltage": float(pixel["dark_voltage_v_per_sec"]),
        "darkvoltagevpersec": float(pixel["dark_voltage_v_per_sec"]),
        "readnoise": float(pixel["read_noise_v"]),
        "readnoisev": float(pixel["read_noise_v"]),
        "dsnu": float(pixel["dsnu_sigma_v"]),
        "dsnuv": float(pixel["dsnu_sigma_v"]),
        "prnu": float(pixel["prnu_sigma"]),
    }
    if key in mapping:
        return mapping[key]
    raise KeyError(f"Unsupported camera pixel parameter: {parameter}")


def _pixel_set(sensor: Sensor, parameter: str, value: Any) -> Sensor:
    key = param_format(parameter)
    if key in {"size", "pixelsize"}:
        size = np.asarray(value, dtype=float)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = size
        return sensor
    if key == "fillfactor":
        sensor.fields["pixel"]["fill_factor"] = float(value)
        return sensor
    if key in {"conversiongain", "conversiongainvpelectron"}:
        sensor.fields["pixel"]["conversion_gain_v_per_electron"] = float(value)
        return sensor
    if key == "voltageswing":
        sensor.fields["pixel"]["voltage_swing"] = float(value)
        return sensor
    if key in {"darkvoltage", "darkvoltagevpersec"}:
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = float(value)
        return sensor
    if key in {"readnoise", "readnoisev"}:
        sensor.fields["pixel"]["read_noise_v"] = float(value)
        return sensor
    if key in {"dsnu", "dsnuv"}:
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(value)
        return sensor
    if key == "prnu":
        sensor.fields["pixel"]["prnu_sigma"] = float(value)
        return sensor
    raise KeyError(f"Unsupported camera pixel parameter: {parameter}")


def camera_create(
    camera_type: str = "default",
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Camera:
    """Create a supported camera."""

    store = _store(asset_store)
    normalized = param_format(camera_type)
    camera = Camera(name=str(camera_type))

    if normalized in {"default"}:
        oi = oi_create(session=session)
        sensor = sensor_create(asset_store=store, session=session)
    elif normalized in {"ideal"}:
        oi = oi_create(session=session)
        sensor = sensor_create_ideal("xyz", asset_store=store, session=session)
    elif normalized in {"monochrome"}:
        oi = oi_create(session=session)
        sensor = sensor_create("monochrome", asset_store=store, session=session)
    elif normalized in {"idealmonochrome"}:
        oi = oi_create(session=session)
        sensor = sensor_create_ideal("monochrome", asset_store=store, session=session)
    else:
        try:
            oi = oi_create(session=session)
            sensor = sensor_create(camera_type, *args, asset_store=store, session=session)
        except UnsupportedOptionError as exc:
            raise UnsupportedOptionError("cameraCreate", camera_type) from exc

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip_create(sensor=sensor, asset_store=store, session=session)
    return track_camera_session_state(session, camera)


def camera_get(camera: Camera, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "vcitype":
        ip_name = param_format(str(camera.fields["ip"].name))
        return "l3" if ip_name in {"l3", "l3global"} else "default"

    prefix, remainder = split_prefixed_parameter(parameter, ("oi", "optics", "sensor", "pixel", "ip", "vci", "l3"))
    if prefix == "oi":
        if not remainder:
            return camera.fields["oi"]
        return oi_get(camera.fields["oi"], remainder, *args)
    if prefix == "optics":
        if not remainder:
            return camera.fields["oi"].fields["optics"]
        optics_param = "optics wvf" if remainder == "wvf" else f"optics {remainder}"
        return oi_get(camera.fields["oi"], optics_param, *args)
    if prefix == "sensor":
        if not remainder:
            return camera.fields["sensor"]
        return sensor_get(camera.fields["sensor"], remainder, *args)
    if prefix == "pixel":
        if not remainder:
            return camera.fields["sensor"].fields["pixel"]
        return _pixel_get(camera.fields["sensor"], remainder)
    if prefix in {"ip", "vci"}:
        if not remainder:
            return camera.fields["ip"]
        return ip_get(camera.fields["ip"], remainder, *args)
    if prefix == "l3":
        return camera.fields["ip"].fields.get("l3")

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
    if key == "image":
        return ip_get(camera.fields["ip"], "display data")
    raise KeyError(f"Unsupported cameraGet parameter: {parameter}")


def camera_set(
    camera: Camera,
    parameter: str,
    value: Any,
    *args: Any,
    session: SessionContext | None = None,
) -> Camera:
    prefix, remainder = split_prefixed_parameter(parameter, ("oi", "optics", "sensor", "pixel", "ip", "vci", "l3"))
    if prefix == "oi":
        if not remainder:
            camera.fields["oi"] = track_session_object(session, value)
        else:
            camera.fields["oi"] = oi_set(camera.fields["oi"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix == "optics":
        if not remainder:
            camera.fields["oi"] = oi_set(camera.fields["oi"], "optics", value)
        else:
            optics_param = "optics wvf" if remainder == "wvf" else f"optics {remainder}"
            camera.fields["oi"] = oi_set(camera.fields["oi"], optics_param, value)
        return track_camera_session_state(session, camera)
    if prefix == "sensor":
        if not remainder:
            camera.fields["sensor"] = track_session_object(session, value)
        else:
            camera.fields["sensor"] = sensor_set(camera.fields["sensor"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix == "pixel":
        if not remainder:
            camera.fields["sensor"].fields["pixel"] = dict(value)
        else:
            camera.fields["sensor"] = _pixel_set(camera.fields["sensor"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix in {"ip", "vci"}:
        if not remainder:
            camera.fields["ip"] = value
        else:
            camera.fields["ip"] = ip_set(camera.fields["ip"], remainder, value, *args, session=session)
        return track_camera_session_state(session, camera)
    if prefix == "l3":
        camera.fields["ip"] = ip_set(camera.fields["ip"], "l3", value, session=session)
        return track_camera_session_state(session, camera)

    key = param_format(parameter)
    if key == "name":
        camera.name = str(value)
        return track_camera_session_state(session, camera)
    if key == "type":
        camera.type = str(value)
        return track_camera_session_state(session, camera)
    if key == "oi":
        camera.fields["oi"] = track_session_object(session, value)
        return track_camera_session_state(session, camera)
    if key == "sensor":
        camera.fields["sensor"] = track_session_object(session, value)
        return track_camera_session_state(session, camera)
    if key in {"ip", "vci"}:
        camera.fields["ip"] = value
        return track_camera_session_state(session, camera)
    if key == "l3sensorsize":
        camera.fields["sensor"] = sensor_set(camera.fields["sensor"], "size", value)
        return track_camera_session_state(session, camera)
    if key == "l3sensorfov":
        camera.fields["sensor"] = sensor_set_size_to_fov(camera.fields["sensor"], value, camera.fields["oi"])
        return track_camera_session_state(session, camera)
    raise KeyError(f"Unsupported cameraSet parameter: {parameter}")


def camera_compute(
    camera: Camera,
    p_type: str | Scene = "sensor",
    mode: str = "normal",
    sensor_resize: bool = True,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
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
            scene_hfov = float(scene.fields["fov_deg"])
            scene_vfov = float(scene.fields["vfov_deg"])
            sensor_hfov = float(sensor_get(sensor, "fov horizontal", scene, oi))
            sensor_vfov = float(sensor_get(sensor, "fov vertical", scene, oi))
            if (
                abs((scene_hfov - sensor_hfov) / max(scene_hfov, 1e-12)) > 0.01
                or abs((scene_vfov - sensor_vfov) / max(scene_vfov, 1e-12)) > 0.01
            ):
                sensor = sensor_set_size_to_fov(sensor.clone(), (scene_hfov, scene_vfov), oi)
        oi = oi_compute(oi, scene, session=session)
        sensor = sensor_compute(sensor, oi, session=session)
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    elif start_type == "oi":
        sensor = sensor_compute(sensor, oi, session=session)
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    elif start_type == "sensor":
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    else:
        raise UnsupportedOptionError("cameraCompute", str(p_type))

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip
    camera.data["result"] = ip.data.get("result")
    return track_camera_session_state(session, camera)
