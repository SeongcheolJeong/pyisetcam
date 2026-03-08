"""Core milestone-one Python port of the ISETCam pipeline."""

from .assets import (
    DEFAULT_UPSTREAM_SHA,
    DEFAULT_UPSTREAM_TARBALL_SHA256,
    AssetStore,
    ensure_upstream_snapshot,
)
from .camera import camera_compute, camera_create, camera_get, camera_set
from .display import display_create, display_get, display_set
from .ip import ip_compute, ip_create, ip_get, ip_set
from .optics import oi_compute, oi_create, oi_get, oi_set, wvf_create
from .parity import run_python_case, run_python_case_with_context
from .scene import (
    scene_adjust_illuminant,
    scene_adjust_luminance,
    scene_calculate_luminance,
    scene_clear_data,
    scene_create,
    scene_from_file,
    scene_get,
    scene_set,
)
from .sensor import (
    sensor_compute,
    sensor_create,
    sensor_create_ideal,
    sensor_get,
    sensor_set,
    sensor_set_size_to_fov,
)
from .types import Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor
from .utils import DEFAULT_WAVE, blackbody, param_format

__all__ = [
    "AssetStore",
    "Camera",
    "DEFAULT_UPSTREAM_SHA",
    "DEFAULT_UPSTREAM_TARBALL_SHA256",
    "DEFAULT_WAVE",
    "Display",
    "ImageProcessor",
    "OpticalImage",
    "Scene",
    "Sensor",
    "blackbody",
    "cameraCompute",
    "cameraCreate",
    "cameraGet",
    "cameraSet",
    "camera_compute",
    "camera_create",
    "camera_get",
    "camera_set",
    "displayCreate",
    "displayGet",
    "displaySet",
    "display_create",
    "display_get",
    "display_set",
    "ensure_upstream_snapshot",
    "ipCompute",
    "ipCreate",
    "ipGet",
    "ipSet",
    "ip_compute",
    "ip_create",
    "ip_get",
    "ip_set",
    "oiCompute",
    "oiCreate",
    "oiGet",
    "oiSet",
    "oi_compute",
    "oi_create",
    "oi_get",
    "oi_set",
    "param_format",
    "run_python_case",
    "run_python_case_with_context",
    "sceneAdjustIlluminant",
    "sceneAdjustLuminance",
    "sceneCalculateLuminance",
    "sceneClearData",
    "sceneCreate",
    "sceneFromFile",
    "sceneGet",
    "sceneSet",
    "scene_adjust_illuminant",
    "scene_adjust_luminance",
    "scene_calculate_luminance",
    "scene_clear_data",
    "scene_create",
    "scene_from_file",
    "scene_get",
    "scene_set",
    "sensorCompute",
    "sensorCreate",
    "sensorCreateIdeal",
    "sensorGet",
    "sensorSet",
    "sensorSetSizeToFOV",
    "sensor_compute",
    "sensor_create",
    "sensor_create_ideal",
    "sensor_get",
    "sensor_set",
    "sensor_set_size_to_fov",
    "wvf_create",
]

__version__ = "0.1.0"

# MATLAB-style aliases.
sceneCreate = scene_create
sceneFromFile = scene_from_file
sceneGet = scene_get
sceneSet = scene_set
sceneAdjustIlluminant = scene_adjust_illuminant
sceneAdjustLuminance = scene_adjust_luminance
sceneCalculateLuminance = scene_calculate_luminance
sceneClearData = scene_clear_data

displayCreate = display_create
displayGet = display_get
displaySet = display_set

oiCreate = oi_create
oiCompute = oi_compute
oiGet = oi_get
oiSet = oi_set

sensorCreate = sensor_create
sensorCreateIdeal = sensor_create_ideal
sensorCompute = sensor_compute
sensorGet = sensor_get
sensorSet = sensor_set
sensorSetSizeToFOV = sensor_set_size_to_fov

ipCreate = ip_create
ipCompute = ip_compute
ipGet = ip_get
ipSet = ip_set

cameraCreate = camera_create
cameraCompute = camera_compute
cameraGet = camera_get
cameraSet = camera_set
