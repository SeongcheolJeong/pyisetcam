"""Camera orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore
from .display import display_get
from .exceptions import UnsupportedOptionError
from .ip import ip_compute, ip_create, ip_get, ip_set
from .iso import iso12233, iso_find_slanted_bar
from .metrics import delta_e_ab, iso_acutance, xyz_from_energy, xyz_to_lab
from .optics import oi_compute, oi_create, oi_get, oi_set
from .scene import Scene, scene_adjust_luminance, scene_create, scene_set
from .session import track_camera_session_state, track_session_object
from .sensor import (
    _chart_rectangles,
    _chart_roi,
    _macbeth_ideal_linear_rgb,
    sensor_compute,
    sensor_create,
    sensor_create_ideal,
    sensor_get,
    sensor_set,
    sensor_set_size_to_fov,
)
from .types import Camera, ImageProcessor, OpticalImage, Sensor, SessionContext
from .utils import image_increase_image_rgb_size, linear_to_srgb, param_format, rgb_to_xw_format, split_prefixed_parameter, xw_to_rgb_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


@dataclass
class CameraMTFResult:
    """Headless payload returned by camera_mtf()."""

    freq: NDArray[np.float64]
    mtf: NDArray[np.float64]
    nyquistf: float
    lsf: NDArray[np.float64]
    lsfx: NDArray[np.float64]
    mtf50: float
    aliasingPercentage: float
    rect: NDArray[np.int_]
    vci: ImageProcessor
    esf: NDArray[np.float64] | None = None
    fitme: NDArray[np.float64] | None = None
    win: None = None


@dataclass
class CameraVSNRResult:
    """Headless payload returned by camera_vsnr()."""

    lightLevels: NDArray[np.float64]
    vSNR: NDArray[np.float64]
    eTime: NDArray[np.float64]
    rect: NDArray[np.int_]
    ip: list[ImageProcessor]
    oi: OpticalImage
    sensor: Sensor


def _pixel_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
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
    return sensor_get(sensor, f"pixel {parameter}", *args)


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
    return sensor_set(sensor, f"pixel {parameter}", value)


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
        return _pixel_get(camera.fields["sensor"], remainder, *args)
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


def _matlab_round_scalar(value: float) -> int:
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def _whole_chart_corner_points(rows: int, cols: int) -> np.ndarray:
    return np.array(
        [
            [1.0, float(rows)],
            [float(cols), float(rows)],
            [float(cols), 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )


def _chart_rects_data(
    obj: ImageProcessor | Sensor,
    m_locs: Any,
    delta: Any,
    *,
    full_data: bool = False,
    data_type: str = "result",
) -> np.ndarray | list[np.ndarray]:
    from .roi import vc_get_roi_data

    locs = np.asarray(m_locs, dtype=float)
    if locs.ndim != 2 or locs.shape[0] != 2:
        raise ValueError("Chart midpoint locations must be a 2xN array in [row; col] order.")

    patch_data: list[np.ndarray] = []
    for index in range(locs.shape[1]):
        roi_locs, _ = _chart_roi(locs[:, index], delta)
        patch_data.append(np.asarray(vc_get_roi_data(obj, roi_locs, data_type), dtype=float))

    if full_data:
        return patch_data

    n_channels = int(patch_data[0].shape[1]) if patch_data else 0
    mean_values = np.zeros((locs.shape[1], n_channels), dtype=float)
    for index, data in enumerate(patch_data):
        mean_values[index, :] = np.mean(np.asarray(data, dtype=float), axis=0, dtype=float)
    return mean_values


def _display_white_xyz(ip: ImageProcessor, *, asset_store: AssetStore) -> np.ndarray:
    display = ip.fields["display"]
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)
    white_xyz = np.asarray(xyz_from_energy(white_spd, wave, asset_store=asset_store), dtype=float).reshape(-1)
    return 100.0 * (white_xyz / max(float(white_xyz[1]), 1.0e-12))


def _middle_matrix(m: Any, sz: Any) -> np.ndarray:
    data = np.asarray(m, dtype=float)
    target = np.asarray(sz, dtype=float).reshape(-1)
    if target.size == 1:
        target = np.repeat(target, 2)
    half = np.asarray([_matlab_round_scalar(float(value) / 2.0) for value in target[:2]], dtype=int)
    center = np.asarray([_matlab_round_scalar(float(value) / 2.0) for value in data.shape[:2]], dtype=int)

    row_min = max(1, int(center[0] - half[0]))
    row_max = min(int(data.shape[0]), int(center[0] + half[0]))
    col_min = max(1, int(center[1] - half[1]))
    col_max = min(int(data.shape[1]), int(center[1] + half[1]))
    return np.asarray(data[row_min - 1 : row_max, col_min - 1 : col_max, ...], dtype=float)


def _default_vsnr_rect(ip: ImageProcessor) -> np.ndarray:
    size = np.asarray(ip_get(ip, "size"), dtype=int).reshape(-1)
    border = np.asarray([_matlab_round_scalar(0.1 * float(value)) for value in size[:2]], dtype=int)
    return np.array(
        [
            int(border[1]),
            int(border[0]),
            int(size[1] - (2 * border[1])),
            int(size[0] - (2 * border[0])),
        ],
        dtype=int,
    )


def _ip_vsnr(
    ip: ImageProcessor,
    rect: Any,
    *,
    asset_store: AssetStore,
) -> float:
    rect_array = np.asarray(rect, dtype=int).reshape(-1)
    roi_xyz = np.asarray(ip_get(ip, "roixyz", rect_array), dtype=float)
    rows = int(rect_array[3]) + 1
    cols = int(rect_array[2]) + 1
    roi_xyz = xw_to_rgb_format(roi_xyz, rows, cols)

    white_xyz = _display_white_xyz(ip, asset_store=asset_store)
    lab = np.asarray(xyz_to_lab(roi_xyz, white_xyz), dtype=float)
    lab = _middle_matrix(lab, 0.8 * np.asarray(lab.shape[:2], dtype=float))
    channels = lab.reshape(-1, 3)
    variance = np.var(channels, axis=0, ddof=1, dtype=float)
    return 1.0 / max(float(np.sqrt(np.sum(variance, dtype=float))), 1.0e-12)


def _linear_srgb_to_xyz(rgb: Any) -> np.ndarray:
    rgb_array = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    if rgb_array.ndim == 2:
        xw = rgb_array.reshape(-1, 3)
        rows = cols = None
    elif rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        xw, rows, cols, _ = rgb_to_xw_format(rgb_array)
    else:
        raise ValueError("RGB data must be XW Nx3 or RGB-format rows x cols x 3.")

    transform = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    xyz = np.asarray(xw @ transform.T, dtype=float)
    if rows is None or cols is None:
        return xyz
    return xw_to_rgb_format(xyz, rows, cols)


def _macbeth_ideal_xyz(
    wave_nm: Any,
    illuminant_name: str = "D65",
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    from .metrics import xyz_from_energy

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=wave)
    _, illuminant = asset_store.load_illuminant(illuminant_name, wave_nm=wave)
    color_signal = np.asarray(reflectances, dtype=float) * np.asarray(illuminant, dtype=float).reshape(-1, 1)
    macbeth_xyz = np.asarray(xyz_from_energy(color_signal.T, wave, asset_store=asset_store), dtype=float)
    return 100.0 * (macbeth_xyz / max(float(np.max(macbeth_xyz[:, 1])), 1.0e-12))


def _macbeth_color_error(
    ip: ImageProcessor,
    illuminant_name: str = "D65",
    corner_points: Any | None = None,
    *,
    asset_store: AssetStore,
) -> dict[str, Any]:
    if corner_points is None:
        corner_points = ip_get(ip, "chart corner points")
    if corner_points is None or np.asarray(corner_points).size == 0:
        rows, cols = np.asarray(ip_get(ip, "size"), dtype=int)[:2]
        corner_points = _whole_chart_corner_points(int(rows), int(cols))

    corner_points_array = np.asarray(corner_points, dtype=float).reshape(4, 2)
    rects, m_locs, p_size = _chart_rectangles(corner_points_array, 4, 6, 0.3)
    rgb_data = np.asarray(_chart_rects_data(ip, m_locs, float(p_size[0]), full_data=False, data_type="result"), dtype=float)
    macbeth_xyz = np.asarray(_linear_srgb_to_xyz(xw_to_rgb_format(rgb_data, 4, 6)), dtype=float)
    macbeth_xyz_xw, _, _, _ = rgb_to_xw_format(macbeth_xyz)

    ideal_xyz = _macbeth_ideal_xyz(np.asarray(ip_get(ip, "wave"), dtype=float), illuminant_name, asset_store=asset_store)
    white_xyz = np.asarray(macbeth_xyz_xw[3, :], dtype=float).reshape(-1)
    white_ideal_xyz = np.asarray(ideal_xyz[3, :], dtype=float).reshape(-1)
    scale = float(white_ideal_xyz[1]) / max(float(white_xyz[1]), 1.0e-12)
    macbeth_xyz_xw = macbeth_xyz_xw * scale
    macbeth_lab = np.asarray(xyz_to_lab(macbeth_xyz_xw, white_ideal_xyz), dtype=float)
    delta_e = np.asarray(delta_e_ab(macbeth_xyz_xw, ideal_xyz, white_ideal_xyz), dtype=float).reshape(-1)

    return {
        "macbethLAB": macbeth_lab,
        "macbethXYZ": macbeth_xyz_xw,
        "deltaE": delta_e,
        "idealXYZ": np.asarray(ideal_xyz, dtype=float),
        "whiteXYZ": np.asarray(macbeth_xyz_xw[3, :], dtype=float),
        "idealWhiteXYZ": white_ideal_xyz,
        "cornerPoints": corner_points_array,
        "rects": np.asarray(rects, dtype=int),
        "mLocs": np.asarray(m_locs, dtype=int),
        "pSize": np.asarray(p_size, dtype=int),
        "vci": ip,
    }


def macbeth_color_error(
    ip: ImageProcessor,
    illuminant_name: str = "D65",
    corner_points: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ImageProcessor]:
    """Headless MATLAB-style `macbethColorError` wrapper."""

    result = _macbeth_color_error(
        ip,
        illuminant_name,
        corner_points,
        asset_store=_store(asset_store),
    )
    return (
        np.asarray(result["macbethLAB"], dtype=float),
        np.asarray(result["macbethXYZ"], dtype=float),
        np.asarray(result["deltaE"], dtype=float),
        result["vci"],
    )


def macbeth_compare_ideal(
    ip: ImageProcessor,
    m_rgb: Any | None = None,
    illuminant_name: str = "d65",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the headless Macbeth comparison image used by the MATLAB script."""

    store = _store(asset_store)
    current_ip = ip

    patch_size: np.ndarray
    if m_rgb is None or np.asarray(m_rgb).size == 0:
        corner_points = ip_get(current_ip, "chart corner points")
        if corner_points is None or np.asarray(corner_points).size == 0:
            rows, cols = np.asarray(ip_get(current_ip, "size"), dtype=int)[:2]
            corner_points = _whole_chart_corner_points(int(rows), int(cols))
            current_ip = ip_set(current_ip, "chart corner points", corner_points)
        rects, m_locs, patch_size = _chart_rectangles(corner_points, 4, 6, 0.5)
        current_ip = ip_set(current_ip, "chart rectangles", rects)
        m_rgb = _chart_rects_data(current_ip, m_locs, 0.6 * float(patch_size[0]), full_data=False, data_type="result")
    else:
        size = np.asarray(ip_get(current_ip, "size"), dtype=int)
        patch_scalar = _matlab_round_scalar((float(size[0]) / 4.0) * 0.6)
        patch_size = np.array([patch_scalar, patch_scalar], dtype=int)

    patch_rgb = np.asarray(m_rgb, dtype=float)
    if patch_rgb.ndim == 2:
        patch_rgb = xw_to_rgb_format(patch_rgb, 4, 6)
    patch_rgb = patch_rgb / max(float(np.max(patch_rgb)), 1.0e-12)

    ideal_patch = xw_to_rgb_format(
        _macbeth_ideal_linear_rgb(np.asarray(ip_get(current_ip, "wave"), dtype=float), asset_store=store),
        4,
        6,
    )
    ideal_patch = ideal_patch / max(float(np.max(ideal_patch)), 1.0e-12)
    full_ideal_rgb = image_increase_image_rgb_size(ideal_patch, patch_size)
    embedded_rgb = np.asarray(full_ideal_rgb, dtype=float).copy()

    patch_extent = int(np.asarray(patch_size, dtype=int).reshape(-1)[0])
    window = patch_extent + np.array(
        [_matlab_round_scalar(value) for value in np.arange(-patch_extent / 3.0, 1.0, 1.0)],
        dtype=int,
    )
    for row_index in range(4):
        rows = (row_index * patch_extent) + window - 1
        for col_index in range(6):
            cols = (col_index * patch_extent) + window - 1
            embedded_rgb[np.ix_(rows, cols, np.arange(3))] = patch_rgb[row_index, col_index, :]

    return (
        linear_to_srgb(embedded_rgb),
        linear_to_srgb(patch_rgb),
        np.asarray(patch_size, dtype=int).reshape(-1),
    )


def camera_color_accuracy(
    camera: Camera,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[dict[str, Any], Camera]:
    """Run the headless Macbeth color-accuracy workflow from `s_metricsColorAccuracy.m`."""

    store = _store(asset_store)

    scene_distance_m = 1000.0
    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    scene_fov = float(sensor_get(sensor, "fov", scene_distance_m, oi))

    macbeth_scene = scene_create("macbeth d65", asset_store=store, session=session)
    macbeth_scene = scene_adjust_luminance(macbeth_scene, 100.0, asset_store=store)
    macbeth_scene = scene_set(macbeth_scene, "distance", scene_distance_m)
    macbeth_scene = scene_set(macbeth_scene, "fov", scene_fov)
    camera = camera_compute(camera, macbeth_scene, asset_store=store, session=session)

    ip = camera_get(camera, "ip")
    rows, cols = np.asarray(ip_get(ip, "size"), dtype=int)[:2]
    corner_points = _whole_chart_corner_points(int(rows), int(cols))
    ip = ip_set(ip, "chart corner points", corner_points, session=session)
    color_accuracy = _macbeth_color_error(ip, "D65", corner_points, asset_store=store)

    camera = camera_set(camera, "ip", ip, session=session)
    camera = camera_set(camera, "ip chart corner points", corner_points, session=session)
    color_accuracy["vci"] = camera_get(camera, "ip")
    return color_accuracy, camera


def camera_mtf(
    camera: Camera,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraMTFResult:
    """Compute the ISO 12233 camera MTF using the upstream slanted-edge workflow."""

    store = _store(asset_store)

    scene = scene_create("slanted bar", 256, asset_store=store, session=session)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
    scene = scene_set(scene, "fov", 5.0)

    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    sensor = sensor_set(sensor, "fov", 5.0, oi)
    camera = camera_set(camera, "sensor", sensor, session=session)
    camera = camera_compute(camera, scene, asset_store=store, session=session)

    ip = camera_get(camera, "ip")
    result = np.clip(np.asarray(ip_get(ip, "result"), dtype=float), 0.0, None)
    ip = ip_set(ip, "result", result, session=session)
    camera = camera_set(camera, "ip", ip, session=session)

    rect = np.asarray(iso_find_slanted_bar(ip), dtype=int).reshape(-1)
    if rect.size != 4:
        raise ValueError("ISOFindSlantedBar did not return a [col, row, width, height] rect.")
    col_min, row_min, width, height = rect
    bar_image = np.asarray(result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :], dtype=float)
    delta_x = float(camera_get(camera, "pixel width", "mm"))
    mtf = iso12233(bar_image, delta_x=delta_x, plot_options="none")

    return CameraMTFResult(
        freq=np.asarray(mtf.freq, dtype=float),
        mtf=np.asarray(mtf.mtf, dtype=float),
        nyquistf=float(mtf.nyquistf),
        lsf=np.asarray(mtf.lsf, dtype=float),
        lsfx=np.asarray(mtf.lsfx, dtype=float),
        mtf50=float(mtf.mtf50),
        aliasingPercentage=float(mtf.aliasingPercentage),
        rect=np.asarray(rect, dtype=int),
        vci=ip,
        esf=None if mtf.esf is None else np.asarray(mtf.esf, dtype=float),
        fitme=None if mtf.fitme is None else np.asarray(mtf.fitme, dtype=float),
        win=mtf.win,
    )


def camera_acutance(
    camera: Camera,
    plot_flag: bool = True,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> float:
    """Compute ISO acutance from the camera luminance MTF."""

    del plot_flag

    cmtf = camera_mtf(camera, asset_store=asset_store, session=session)
    mtf = np.asarray(cmtf.mtf, dtype=float)
    luminance_mtf = mtf[:, 3] if mtf.ndim == 2 and mtf.shape[1] >= 4 else mtf.reshape(-1)
    oi = camera_get(camera, "oi")
    deg_per_mm = float(camera_get(camera, "sensor h deg per distance", "mm", None, oi))
    cpd = np.asarray(cmtf.freq, dtype=float) / max(deg_per_mm, 1.0e-12)
    return iso_acutance(cpd, luminance_mtf)


def camera_vsnr(
    camera: Camera,
    light_levels: Any | None = None,
    exposure_time: float = 0.01,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraVSNRResult:
    """Run the headless VSNR workflow from `s_metricsVSNR.m`."""

    store = _store(asset_store)
    levels = (
        np.asarray([1.0, 10.0, 100.0], dtype=float)
        if light_levels is None
        else np.asarray(light_levels, dtype=float).reshape(-1)
    )
    vsnr = np.full(levels.shape, np.nan, dtype=float)
    e_time = np.zeros(levels.shape, dtype=float)
    rect: np.ndarray | None = None
    ip_results: list[ImageProcessor] = []

    for index, level in enumerate(levels):
        scene = scene_create("uniform d65", asset_store=store, session=session)
        scene = scene_set(scene, "fov", 5.0)
        scene = scene_adjust_luminance(scene, float(level), asset_store=store)

        working_camera = camera_set(camera.clone(), "sensor exp time", float(exposure_time), session=session)
        working_camera = camera_compute(working_camera, scene, asset_store=store, session=session)

        ip = camera_get(working_camera, "ip")
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        result_max = float(np.max(result)) if result.size else 0.0
        sensor_max = float(ip_get(ip, "sensormax"))

        if (sensor_max - result_max) < (10.0 * np.finfo(float).eps):
            ip_results.append(ip)
            continue

        scaled_ip = ip_set(ip.clone(), "result", result * (sensor_max / max(result_max, 1.0e-12)), session=session)
        if rect is None:
            rect = _default_vsnr_rect(scaled_ip)
        vsnr[index] = _ip_vsnr(scaled_ip, rect, asset_store=store)
        ip_results.append(scaled_ip)

    if rect is None:
        rect = np.zeros(4, dtype=int)

    return CameraVSNRResult(
        lightLevels=levels,
        vSNR=vsnr,
        eTime=e_time,
        rect=np.asarray(rect, dtype=int),
        ip=ip_results,
        oi=camera_get(camera, "oi"),
        sensor=camera_get(camera, "sensor"),
    )


cameraColorAccuracy = camera_color_accuracy
cameraVSNR = camera_vsnr
macbethColorError = macbeth_color_error
macbethCompareIdeal = macbeth_compare_ideal
