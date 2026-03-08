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
from .metrics import (
    comparison_metrics,
    correlated_color_temperature,
    delta_e_ab,
    iePSNR,
    ieXYZ2LAB,
    ieXYZFromEnergy,
    mean_absolute_error,
    mean_relative_error,
    metrics_spd,
    mired_difference,
    peak_signal_to_noise_ratio,
    root_mean_squared_error,
    xyz_from_energy,
    xyz_to_lab,
)
from .optics import (
    optics_ray_trace,
    oi_calculate_illuminance,
    oi_diffuser,
    oi_compute,
    oi_create,
    oi_get,
    oi_set,
    rt_angle_lut,
    rt_block_center,
    rt_choose_block_size,
    rt_di_interp,
    rt_extract_block,
    rt_geometry,
    rt_insert_block,
    rt_otf,
    rt_psf_apply,
    rt_psf_grid,
    rt_psf_interp,
    rt_precompute_psf,
    rt_precompute_psf_apply,
    rt_ri_interp,
    rt_sample_heights,
    rt_synthetic,
    wvf_create,
)
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
    "comparison_metrics",
    "correlated_color_temperature",
    "deltaEab",
    "delta_e_ab",
    "iePSNR",
    "ieXYZ2LAB",
    "ieXYZFromEnergy",
    "ipCompute",
    "ipCreate",
    "ipGet",
    "ipSet",
    "ip_compute",
    "ip_create",
    "ip_get",
    "ip_set",
    "mean_absolute_error",
    "mean_relative_error",
    "metricsSPD",
    "metrics_spd",
    "mired_difference",
    "opticsRayTrace",
    "optics_ray_trace",
    "oiCalculateIlluminance",
    "oiDiffuser",
    "oiCompute",
    "oiCreate",
    "oiGet",
    "oiSet",
    "oi_calculate_illuminance",
    "oi_diffuser",
    "oi_compute",
    "oi_create",
    "oi_get",
    "oi_set",
    "param_format",
    "peak_signal_to_noise_ratio",
    "root_mean_squared_error",
    "rtDIInterp",
    "rtAngleLUT",
    "rtBlockCenter",
    "rtChooseBlockSize",
    "rtExtractBlock",
    "rtGeometry",
    "rtInsertBlock",
    "rtOTF",
    "rtRIInterp",
    "rtPSFGrid",
    "rtPSFApply",
    "rtPrecomputePSF",
    "rtPrecomputePSFApply",
    "rt_di_interp",
    "rt_angle_lut",
    "rt_block_center",
    "rt_choose_block_size",
    "rt_extract_block",
    "rt_geometry",
    "rt_insert_block",
    "rt_otf",
    "rt_psf_grid",
    "rt_psf_apply",
    "rtPSFInterp",
    "rt_psf_interp",
    "rt_precompute_psf",
    "rt_precompute_psf_apply",
    "rtSampleHeights",
    "rt_ri_interp",
    "rt_sample_heights",
    "rtSynthetic",
    "rt_synthetic",
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
    "xyz_from_energy",
    "xyz_to_lab",
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
oiCalculateIlluminance = oi_calculate_illuminance
oiDiffuser = oi_diffuser
opticsRayTrace = optics_ray_trace
rtAngleLUT = rt_angle_lut
rtBlockCenter = rt_block_center
rtChooseBlockSize = rt_choose_block_size
rtDIInterp = rt_di_interp
rtExtractBlock = rt_extract_block
rtGeometry = rt_geometry
rtInsertBlock = rt_insert_block
rtOTF = rt_otf
rtPSFApply = rt_psf_apply
rtPSFGrid = rt_psf_grid
rtPSFInterp = rt_psf_interp
rtPrecomputePSF = rt_precompute_psf
rtPrecomputePSFApply = rt_precompute_psf_apply
rtRIInterp = rt_ri_interp
rtSampleHeights = rt_sample_heights
rtSynthetic = rt_synthetic

sensorCreate = sensor_create
sensorCreateIdeal = sensor_create_ideal
sensorCompute = sensor_compute
sensorGet = sensor_get
sensorSet = sensor_set
sensorSetSizeToFOV = sensor_set_size_to_fov

metricsSPD = metrics_spd
deltaEab = delta_e_ab

ipCreate = ip_create
ipCompute = ip_compute
ipGet = ip_get
ipSet = ip_set

cameraCreate = camera_create
cameraCompute = camera_compute
cameraGet = camera_get
cameraSet = camera_set
