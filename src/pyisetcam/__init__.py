"""Core milestone-one Python port of the ISETCam pipeline."""

from .assets import (
    DEFAULT_UPSTREAM_SHA,
    DEFAULT_UPSTREAM_TARBALL_SHA256,
    AssetStore,
    ensure_upstream_snapshot,
)
from .camera import camera_compute, camera_create, camera_get, camera_set
from .display import display_create, display_get, display_set
from .fileio import vc_export_object, vc_load_object, vc_save_object
from .ip import image_data_xyz, ip_compute, ip_create, ip_get, ip_set
from .metrics import (
    cct_from_uv,
    chromaticity_xy,
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
    xyz_to_luv,
    xyz_to_uv,
)
from .optics import (
    ie_field_height_to_index,
    optics_ray_trace,
    oi_calculate_illuminance,
    oi_diffuser,
    oi_compute,
    oi_create,
    oi_get,
    oi_set,
    optics_psf_to_otf,
    rt_angle_lut,
    rt_block_center,
    rt_choose_block_size,
    rt_di_interp,
    rt_extract_block,
    rt_filtered_block_support,
    rt_geometry,
    rt_insert_block,
    rt_import_data,
    rt_otf,
    rt_file_names,
    rt_psf_apply,
    rt_psf_grid,
    rt_psf_interp,
    rt_precompute_psf,
    rt_precompute_psf_apply,
    rt_ri_interp,
    rt_sample_heights,
    rt_synthetic,
    si_synthetic,
    wvf_aperture,
    wvf_aperture_params,
    wvf_compute,
    wvf_compute_psf,
    wvf_create,
    wvf_defocus_diopters_to_microns,
    wvf_defocus_microns_to_diopters,
    wvf_get,
    wvf_pupil_function,
    wvf_set,
    wvf_to_oi,
    zemax_load,
    zemax_read_header,
)
from .parity import run_python_case, run_python_case_with_context
from .plotting import ip_plot, oi_plot, scene_plot, sensor_plot, sensor_plot_fft, wvf_plot
from .ptable import IEPTable, ie_p_table
from .roi import ie_locs2_rect, ie_rect2_locs, ie_rect2_vertices, ie_roi2_locs, vc_get_roi_data, vc_rect2_locs
from .session import (
    ie_app_get,
    ie_add_object,
    ie_delete_object,
    ie_equivalent_objtype,
    ie_find_object_by_name,
    ie_get_object,
    ie_get_selected_object,
    ie_init_session,
    ie_main_close,
    ie_refresh_window,
    ie_replace_object,
    ie_session_get,
    ie_session_set,
    ie_windows_get,
    ie_windows_set,
    ie_select_object,
    session_add_and_select_object,
    session_add_object,
    session_count_objects,
    session_create,
    session_delete_object,
    session_delete_some_objects,
    session_delete_selected_object,
    session_get_object,
    session_get_object_type,
    session_get_object_with_id,
    session_get_object_names,
    session_get_objects,
    session_get_selected,
    session_get_selected_pair,
    session_get_selected_id,
    session_find_object_by_name,
    session_list_objects,
    session_new_object_name,
    session_new_object_value,
    session_object_id,
    session_replace_and_select_object,
    session_replace_object,
    session_set_objects,
    session_set_selected,
    vc_equivalent_objtype,
    vc_get_figure,
    vc_set_figure_handles,
    vc_select_figure,
)
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
    pixel_snr,
    sensor_snr,
    sensor_compute,
    sensor_create,
    sensor_create_ideal,
    sensor_get,
    sensor_set,
    sensor_set_size_to_fov,
)
from .types import Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import DEFAULT_WAVE, blackbody, ie_parameter_otype, param_format

__all__ = [
    "AssetStore",
    "Camera",
    "DEFAULT_UPSTREAM_SHA",
    "DEFAULT_UPSTREAM_TARBALL_SHA256",
    "DEFAULT_WAVE",
    "Display",
    "IEPTable",
    "ImageProcessor",
    "OpticalImage",
    "Scene",
    "Sensor",
    "SessionContext",
    "blackbody",
    "cameraCompute",
    "cameraCreate",
    "cameraGet",
    "cameraSet",
    "camera_compute",
    "camera_create",
    "camera_get",
    "camera_set",
    "cct",
    "cct_from_uv",
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
    "ieAddObject",
    "ieAppGet",
    "ieDeleteObject",
    "ieEquivalentObjtype",
    "ieFieldHeight2Index",
    "ieFindObjectByName",
    "ieLocs2Rect",
    "ieParameterOtype",
    "iePTable",
    "ieGetObject",
    "ieGetSelectedObject",
    "ieInitSession",
    "imageDataXYZ",
    "image_data_xyz",
    "ieMainClose",
    "ieRefreshWindow",
    "ieRect2Locs",
    "ieRect2Vertices",
    "ieRoi2Locs",
    "ieReplaceObject",
    "ieSessionGet",
    "ieSessionSet",
    "ieWindowsGet",
    "ieWindowsSet",
    "ieSelectObject",
    "ie_add_object",
    "ie_app_get",
    "ie_delete_object",
    "ie_equivalent_objtype",
    "ie_find_object_by_name",
    "ie_locs2_rect",
    "ie_parameter_otype",
    "ie_p_table",
    "ie_field_height_to_index",
    "ie_get_object",
    "ie_get_selected_object",
    "ie_init_session",
    "ie_main_close",
    "ie_refresh_window",
    "ie_rect2_locs",
    "ie_rect2_vertices",
    "ie_roi2_locs",
    "ie_replace_object",
    "ie_session_get",
    "ie_session_set",
    "ie_windows_get",
    "ie_windows_set",
    "ie_select_object",
    "ieXYZ2LAB",
    "ieXYZFromEnergy",
    "ipCompute",
    "ipCreate",
    "ipGet",
    "ipPlot",
    "ipSet",
    "ip_compute",
    "ip_create",
    "ip_get",
    "ip_plot",
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
    "oi_plot",
    "oi_set",
    "optics_psf_to_otf",
    "param_format",
    "peak_signal_to_noise_ratio",
    "root_mean_squared_error",
    "scene_plot",
    "pixel_snr",
    "sensor_snr",
    "sensor_plot",
    "sensor_plot_fft",
    "session_add_and_select_object",
    "session_add_object",
    "session_count_objects",
    "session_create",
    "session_delete_object",
    "session_delete_some_objects",
    "session_delete_selected_object",
    "session_get_object",
    "session_get_object_type",
    "session_get_object_with_id",
    "session_get_object_names",
    "session_get_objects",
    "session_get_selected",
    "session_get_selected_pair",
    "session_get_selected_id",
    "session_find_object_by_name",
    "session_list_objects",
    "session_new_object_name",
    "session_new_object_value",
    "session_object_id",
    "session_replace_and_select_object",
    "session_replace_object",
    "session_set_objects",
    "session_set_selected",
    "vc_equivalent_objtype",
    "rtDIInterp",
    "rtAngleLUT",
    "rtBlockCenter",
    "rtChooseBlockSize",
    "rtExtractBlock",
    "rtFilteredBlockSupport",
    "rtGeometry",
    "rtImportData",
    "rtInsertBlock",
    "rtOTF",
    "rtRIInterp",
    "rtFileNames",
    "rtPSFGrid",
    "rtPSFApply",
    "rtPrecomputePSF",
    "rtPrecomputePSFApply",
    "rt_di_interp",
    "rt_angle_lut",
    "rt_block_center",
    "rt_choose_block_size",
    "rt_extract_block",
    "rt_filtered_block_support",
    "rt_geometry",
    "rt_import_data",
    "rt_insert_block",
    "rt_otf",
    "rt_file_names",
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
    "siSynthetic",
    "si_synthetic",
    "wvf_compute",
    "wvf_compute_psf",
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
    "sensorSNR",
    "sensorSet",
    "sensorSetSizeToFOV",
    "sensor_compute",
    "sensor_create",
    "sensor_create_ideal",
    "sensor_get",
    "pixelSNR",
    "sensor_set",
    "sensor_set_size_to_fov",
    "vcAddAndSelectObject",
    "vcAddObject",
    "vcCountObjects",
    "vcDeleteObject",
    "vcDeleteSomeObjects",
    "vcDeleteSelectedObject",
    "vcEquivalentObjtype",
    "vcExportObject",
    "vcGetObject",
    "vcGetROIData",
    "vcGetObjectType",
    "vcGetObjectNames",
    "vcGetObjects",
    "vcGetSelectedObject",
    "vcGetSelectedObjectID",
    "vcGetFigure",
    "vcSetFigureHandles",
    "vcNewObjectName",
    "vcNewObjectValue",
    "vcReplaceAndSelectObject",
    "vcReplaceObject",
    "vcLoadObject",
    "vcSaveObject",
    "vcSelectFigure",
    "vcSetObjects",
    "vcSetSelectedObject",
    "vc_export_object",
    "vc_get_roi_data",
    "vc_load_object",
    "vc_rect2_locs",
    "vc_save_object",
    "vc_set_figure_handles",
    "wvfCompute",
    "wvfAperture",
    "wvfApertureP",
    "wvfCreate",
    "wvfComputePSF",
    "wvfGet",
    "wvfPupilFunction",
    "wvfSet",
    "wvf2oi",
    "wvfDefocusDioptersToMicrons",
    "wvfDefocusMicronsToDiopters",
    "wvf_aperture",
    "wvf_aperture_params",
    "wvf_create",
    "wvf_compute_psf",
    "wvf_defocus_diopters_to_microns",
    "wvf_defocus_microns_to_diopters",
    "wvf_get",
    "wvf_pupil_function",
    "wvf_set",
    "wvf_to_oi",
    "wvf_plot",
    "chromaticity_xy",
    "plotScene",
    "plotSensor",
    "plotSensorFFT",
    "oiPlot",
    "wvfPlot",
    "xyz_from_energy",
    "xyz_to_lab",
    "xyz_to_luv",
    "xyz_to_uv",
    "xyz2luv",
    "zemaxLoad",
    "zemaxReadHeader",
    "zemax_load",
    "zemax_read_header",
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
opticsPSF2OTF = optics_psf_to_otf
oiPlot = oi_plot
ipPlot = ip_plot
plotScene = scene_plot
plotSensor = sensor_plot
plotSensorFFT = sensor_plot_fft
wvfPlot = wvf_plot
opticsRayTrace = optics_ray_trace
ieFieldHeight2Index = ie_field_height_to_index
rtAngleLUT = rt_angle_lut
rtBlockCenter = rt_block_center
rtChooseBlockSize = rt_choose_block_size
rtDIInterp = rt_di_interp
rtExtractBlock = rt_extract_block
rtFilteredBlockSupport = rt_filtered_block_support
rtGeometry = rt_geometry
rtImportData = rt_import_data
rtInsertBlock = rt_insert_block
rtOTF = rt_otf
rtPSFApply = rt_psf_apply
rtFileNames = rt_file_names
rtPSFGrid = rt_psf_grid
rtPSFInterp = rt_psf_interp
rtPrecomputePSF = rt_precompute_psf
rtPrecomputePSFApply = rt_precompute_psf_apply
rtRIInterp = rt_ri_interp
rtSampleHeights = rt_sample_heights
rtSynthetic = rt_synthetic
siSynthetic = si_synthetic
wvfAperture = wvf_aperture
wvfApertureP = wvf_aperture_params
wvfCompute = wvf_compute
wvfComputePSF = wvf_compute_psf
wvfCreate = wvf_create
wvfGet = wvf_get
wvfPupilFunction = wvf_pupil_function
wvfSet = wvf_set
wvf2oi = wvf_to_oi
wvfDefocusDioptersToMicrons = wvf_defocus_diopters_to_microns
wvfDefocusMicronsToDiopters = wvf_defocus_microns_to_diopters
zemaxLoad = zemax_load
zemaxReadHeader = zemax_read_header

sensorCreate = sensor_create
sensorCreateIdeal = sensor_create_ideal
sensorCompute = sensor_compute
sensorGet = sensor_get
sensorSNR = sensor_snr
pixelSNR = pixel_snr
sensorSet = sensor_set
sensorSetSizeToFOV = sensor_set_size_to_fov

metricsSPD = metrics_spd
deltaEab = delta_e_ab
xyz2luv = xyz_to_luv
cct = cct_from_uv

ieAddObject = ie_add_object
ieAppGet = ie_app_get
ieDeleteObject = ie_delete_object
ieEquivalentObjtype = ie_equivalent_objtype
ieFindObjectByName = ie_find_object_by_name
ieLocs2Rect = ie_locs2_rect
ieParameterOtype = ie_parameter_otype
iePTable = ie_p_table
ieGetObject = ie_get_object
ieGetSelectedObject = ie_get_selected_object
ieInitSession = ie_init_session
ieMainClose = ie_main_close
ieRefreshWindow = ie_refresh_window
ieRect2Locs = ie_rect2_locs
ieRect2Vertices = ie_rect2_vertices
ieRoi2Locs = ie_roi2_locs
ieReplaceObject = ie_replace_object
ieSessionGet = ie_session_get
ieSessionSet = ie_session_set
ieWindowsGet = ie_windows_get
ieWindowsSet = ie_windows_set
ieSelectObject = ie_select_object

ipCreate = ip_create
ipCompute = ip_compute
ipGet = ip_get
ipSet = ip_set
imageDataXYZ = image_data_xyz

cameraCreate = camera_create
cameraCompute = camera_compute
cameraGet = camera_get
cameraSet = camera_set

vcAddAndSelectObject = session_add_and_select_object
vcAddObject = session_add_object
vcCountObjects = session_count_objects
vcDeleteObject = session_delete_object
vcDeleteSomeObjects = session_delete_some_objects
vcDeleteSelectedObject = session_delete_selected_object
vcEquivalentObjtype = vc_equivalent_objtype
vcExportObject = vc_export_object
vcGetObject = session_get_object_with_id
vcGetROIData = vc_get_roi_data
vcGetObjectType = session_get_object_type
vcGetObjectNames = session_get_object_names
vcGetObjects = session_get_objects
vcGetSelectedObject = session_get_selected_pair
vcGetSelectedObjectID = session_get_selected_id
vcGetFigure = vc_get_figure
vcNewObjectName = session_new_object_name
vcNewObjectValue = session_new_object_value
vcRect2Locs = vc_rect2_locs
vcReplaceAndSelectObject = session_replace_and_select_object
vcReplaceObject = session_replace_object
vcLoadObject = vc_load_object
vcSaveObject = vc_save_object
vcSelectFigure = vc_select_figure
vcSetFigureHandles = vc_set_figure_handles
vcSetObjects = session_set_objects
vcSetSelectedObject = session_set_selected
