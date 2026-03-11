"""Headless MATLAB-style object save/load helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from scipy.io import loadmat, savemat
from tifffile import TiffFile

from .exceptions import UnsupportedOptionError
from .session import ie_add_object, session_add_object, session_replace_object
from .types import BaseISETObject, Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import param_format

_SAVE_KEY_BY_TYPE = {
    "scene": "scene",
    "oi": "opticalimage",
    "opticalimage": "opticalimage",
    "sensor": "isa",
    "isa": "isa",
    "ip": "vcimage",
    "vcimage": "vcimage",
    "display": "display",
    "camera": "camera",
}

_LOAD_KEYS_BY_TYPE = {
    "scene": ("scene",),
    "oi": ("opticalimage",),
    "opticalimage": ("opticalimage",),
    "sensor": ("isa", "sensor"),
    "isa": ("isa", "sensor"),
    "ip": ("vcimage",),
    "vcimage": ("vcimage",),
    "display": ("display",),
    "camera": ("camera",),
}

_OBJECT_CLASS_BY_TYPE = {
    "scene": Scene,
    "oi": OpticalImage,
    "opticalimage": OpticalImage,
    "sensor": Sensor,
    "isa": Sensor,
    "ip": ImageProcessor,
    "vcimage": ImageProcessor,
    "display": Display,
    "camera": Camera,
}


def _default_save_path(obj: BaseISETObject) -> Path:
    name = str(getattr(obj, "name", "") or param_format(obj.type) or "object")
    return Path.cwd() / f"{name}.mat"


def _normalize_save_path(full_name: str | Path | None, obj: BaseISETObject | None = None) -> Path:
    if full_name is None:
        if obj is None:
            raise ValueError("full_name is required when no object is available for a default path.")
        path = _default_save_path(obj)
    else:
        path = Path(full_name)
    if path.suffix.lower() != ".mat":
        path = path.with_suffix(".mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_value(value: Any) -> Any:
    if value is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(value, BaseISETObject):
        return {
            "name": str(value.name),
            "type": str(value.type),
            "metadata": _serialize_value(value.metadata),
            "fields": _serialize_value(value.fields),
            "data": _serialize_value(value.data),
        }
    if isinstance(value, dict):
        return {str(key): _serialize_value(val) for key, val in value.items() if val is not None}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deserialize_value(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {field: _deserialize_value(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, dict):
        return {str(key): _deserialize_value(val) for key, val in value.items() if not str(key).startswith("__")}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.shape == ():
                return _deserialize_value(value.item())
            return [_deserialize_value(item) for item in value.tolist()]
        if value.shape == ():
            return value.item()
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reconstruct_object(value: Any, inferred_type: str | None = None) -> Any:
    if isinstance(value, dict):
        normalized_type = param_format(value.get("type", inferred_type or ""))
        if normalized_type in _OBJECT_CLASS_BY_TYPE and any(key in value for key in ("fields", "data", "metadata")):
            cls = _OBJECT_CLASS_BY_TYPE[normalized_type]
            metadata = _reconstruct_object(value.get("metadata", {}))
            fields = _reconstruct_object(value.get("fields", {}))
            data = _reconstruct_object(value.get("data", {}))
            name = str(value.get("name", normalized_type or cls().name))
            object_type = str(value.get("type", cls().type))
            return cls(name=name, type=object_type, metadata=metadata, fields=fields, data=data)
        return {str(key): _reconstruct_object(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_reconstruct_object(item) for item in value]
    return value


def _save_key_for_object(obj: BaseISETObject) -> str:
    normalized_type = param_format(obj.type)
    key = _SAVE_KEY_BY_TYPE.get(normalized_type)
    if key is None:
        raise ValueError(f"Unsupported object type for vcSaveObject: {obj.type}")
    return key


def _load_payload(path: Path, obj_type: str) -> Any:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    normalized_type = param_format(obj_type or "scene")
    candidates = _LOAD_KEYS_BY_TYPE.get(normalized_type, (normalized_type,))
    for key in candidates:
        if key in data:
            return data[key], normalized_type
    available = [key for key in data if not key.startswith("__")]
    if len(available) == 1:
        return data[available[0]], normalized_type
    raise KeyError(f"No saved object matching type {obj_type} found in {path}.")


def vc_save_object(obj: BaseISETObject, full_name: str | Path | None = None) -> str:
    """Save a core ISET object to a MATLAB `.mat` file."""

    path = _normalize_save_path(full_name, obj)
    save_key = _save_key_for_object(obj)
    savemat(path, {save_key: _serialize_value(obj)}, do_compression=True)
    return str(path)


def vc_export_object(
    obj: BaseISETObject,
    full_name: str | Path | None = None,
    clear_data_flag: bool = False,
) -> str:
    """Export an ISET object, optionally clearing cached data first."""

    export_obj = obj.clone()
    if clear_data_flag:
        export_obj.data = {}
    return vc_save_object(export_obj, full_name)


def vc_load_object(
    obj_type: str = "scene",
    full_name: str | Path | None = None,
    val: int | None = None,
    *,
    session: SessionContext | None = None,
) -> tuple[Any, str]:
    """Load a saved ISET object from a MATLAB `.mat` file."""

    if full_name is None:
        raise ValueError("full_name is required in the headless Python port of vcLoadObject.")
    path = _normalize_save_path(full_name)
    payload, normalized_type = _load_payload(path, obj_type)
    loaded = _reconstruct_object(_deserialize_value(payload), inferred_type=normalized_type)
    if not isinstance(loaded, BaseISETObject):
        cls = _OBJECT_CLASS_BY_TYPE.get(normalized_type)
        if cls is None:
            raise ValueError(f"Unsupported object type for vcLoadObject: {obj_type}")
        loaded = cls(name=path.stem, type=normalized_type, fields=_reconstruct_object(loaded))
    loaded.name = path.stem

    if session is None:
        return loaded, str(path)

    if val is None:
        slot = ie_add_object(session, loaded)
    else:
        if isinstance(loaded, (Camera, ImageProcessor)):
            session_replace_object(session, loaded, int(val), select=True)
        else:
            session_add_object(session, loaded, select=True, object_id=int(val))
        slot = int(val)
    return slot, str(path)


# MATLAB-style aliases.
vcSaveObject = vc_save_object
vcExportObject = vc_export_object
vcLoadObject = vc_load_object


def ie_save_si_data_file(
    psf: Any,
    wave: Any,
    um_per_samp: Any,
    f_name: str | Path | None = None,
) -> str:
    """Write MATLAB-style shift-invariant PSF data for `siSynthetic('custom', ...)`."""

    if psf is None:
        raise ValueError("psf volume required")
    if wave is None:
        raise ValueError("wavelength samples required (nm)")
    if um_per_samp is None:
        raise ValueError("Microns per sample(2-vector) required")

    path = _normalize_save_path(f_name or (Path.cwd() / "siSynthetic.mat"))
    psf_array = np.asarray(psf, dtype=float)
    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    um_per_samp_array = np.asarray(um_per_samp, dtype=float).reshape(-1)
    if um_per_samp_array.size == 1:
        um_per_samp_array = np.repeat(um_per_samp_array, 2)
    if um_per_samp_array.size != 2:
        raise ValueError("umPerSamp must be a scalar or 2-vector.")

    notes = {"timeStamp": datetime.now().isoformat()}
    savemat(
        path,
        {
            "psf": psf_array,
            "wave": wave_array,
            "umPerSamp": um_per_samp_array,
            "notes": notes,
        },
        do_compression=True,
    )
    return str(path)


ieSaveSIDataFile = ie_save_si_data_file


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _decode_tiff_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_tiff_value(value.item())
        return np.asarray([_decode_tiff_value(item) for item in value.tolist()])
    if isinstance(value, Mapping):
        return {str(key): _decode_tiff_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_tiff_value(item) for item in value]
    if isinstance(value, tuple):
        if len(value) == 2 and all(_is_numeric_scalar(item) for item in value):
            numerator = float(value[0])
            denominator = float(value[1])
            return numerator / denominator if denominator != 0 else np.nan
        if len(value) > 2 and len(value) % 2 == 0 and all(_is_numeric_scalar(item) for item in value):
            pairs = np.asarray(value, dtype=float).reshape(-1, 2)
            if np.all(pairs[:, 1] != 0):
                return pairs[:, 0] / pairs[:, 1]
        return tuple(_decode_tiff_value(item) for item in value)
    return value


def _dng_scalar(value: Any) -> Any:
    decoded = _decode_tiff_value(value)
    if isinstance(decoded, np.ndarray) and decoded.shape == (1,):
        return decoded.reshape(-1)[0].item()
    return decoded


def _select_raw_series(tif: TiffFile) -> Any:
    candidates = [series for series in tif.series if len(series.shape) == 2]
    if not candidates:
        raise ValueError("No raw mosaic image found in DNG file.")
    return max(candidates, key=lambda series: int(np.prod(series.shape)))


def _select_rgb_series(tif: TiffFile) -> Any:
    candidates = [series for series in tif.series if len(series.shape) == 3 and series.shape[-1] >= 3]
    if candidates:
        return max(candidates, key=lambda series: int(np.prod(series.shape)))
    if not tif.series:
        raise ValueError("No image series found in DNG file.")
    return tif.series[0]


def _extract_dng_info(tif: TiffFile, path: Path) -> dict[str, Any]:
    preview_page = tif.pages[0]
    raw_series = _select_raw_series(tif)
    raw_page = raw_series.pages[0]

    make_tag = preview_page.tags.get("Make")
    model_tag = preview_page.tags.get("Model")
    orientation_tag = preview_page.tags.get("Orientation")
    exif_tag = preview_page.tags.get("ExifTag")

    subifd_entry: dict[str, Any] = {}
    for tag_name in (
        "BlackLevel",
        "CFARepeatPatternDim",
        "CFAPattern",
        "ActiveArea",
        "DefaultCropOrigin",
        "DefaultCropSize",
    ):
        tag = raw_page.tags.get(tag_name)
        if tag is not None:
            subifd_entry[tag_name] = _decode_tiff_value(tag.value)

    digital_camera = _decode_tiff_value(exif_tag.value if exif_tag is not None else {})
    info: dict[str, Any] = {
        "Filename": str(path),
        "Make": str(_dng_scalar(make_tag.value)) if make_tag is not None else "",
        "Model": str(_dng_scalar(model_tag.value)) if model_tag is not None else "",
        "Orientation": int(_dng_scalar(orientation_tag.value)) if orientation_tag is not None else 1,
        "DigitalCamera": digital_camera,
        "SubIFDs": [subifd_entry],
        "ImageLength": int(raw_series.shape[0]),
        "ImageWidth": int(raw_series.shape[1]),
    }

    if isinstance(digital_camera, dict):
        if "ISOSpeedRatings" in digital_camera:
            info["ISOSpeedRatings"] = int(round(float(_dng_scalar(digital_camera["ISOSpeedRatings"]))))
        if "ExposureTime" in digital_camera:
            info["ExposureTime"] = float(_dng_scalar(digital_camera["ExposureTime"]))
    if "BlackLevel" in subifd_entry:
        info["BlackLevel"] = np.asarray(subifd_entry["BlackLevel"], dtype=float).reshape(-1)
    return info


def ie_dng_simple_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reduced MATLAB-style DNG metadata summary."""

    digital_camera = info.get("DigitalCamera")
    if isinstance(digital_camera, Mapping):
        iso_speed = digital_camera.get("ISOSpeedRatings")
        exposure_time = digital_camera.get("ExposureTime")
        subifds = info.get("SubIFDs", [])
        black_level = None
        if isinstance(subifds, list) and subifds:
            first_subifd = subifds[0]
            if isinstance(first_subifd, Mapping):
                black_level = first_subifd.get("BlackLevel")
        if black_level is None:
            black_level = info.get("BlackLevel")
        orientation = info.get("Orientation", 1)
    else:
        iso_speed = info.get("ISOSpeedRatings")
        exposure_time = info.get("ExposureTime")
        black_level = info.get("BlackLevel")
        orientation = info.get("Orientation", 1)

    black_level_array = np.asarray([] if black_level is None else black_level, dtype=float).reshape(-1)
    return {
        "isoSpeed": int(round(float(_dng_scalar(iso_speed)))) if iso_speed is not None else None,
        "exposureTime": float(_dng_scalar(exposure_time)) if exposure_time is not None else None,
        "blackLevel": black_level_array,
        "orientation": int(_dng_scalar(orientation)),
    }


def ie_dng_read(
    fname: str | Path,
    *args: Any,
    only_info: bool = False,
    simple_info: bool = False,
    rgb: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read a DNG file as raw mosaic or rendered RGB plus MATLAB-style metadata."""

    if len(args) % 2 != 0:
        raise ValueError("ieDNGRead expects key/value pairs.")
    for index in range(0, len(args), 2):
        parameter = param_format(args[index])
        value = args[index + 1]
        if parameter == "onlyinfo":
            only_info = bool(value)
        elif parameter == "simpleinfo":
            simple_info = bool(value)
        elif parameter == "rgb":
            rgb = bool(value)
        else:
            raise UnsupportedOptionError("ieDNGRead", str(args[index]))

    path = Path(fname)
    with TiffFile(path) as tif:
        info = _extract_dng_info(tif, path)
        reduced_info = ie_dng_simple_info(info) if simple_info else info
        if only_info:
            return None, reduced_info
        series = _select_rgb_series(tif) if rgb else _select_raw_series(tif)
        image = np.asarray(series.asarray())
    return image, reduced_info


def _dng_orientation_pattern(orientation: int) -> np.ndarray:
    mapping = {
        1: np.array([[1, 2], [2, 3]], dtype=int),
        3: np.array([[3, 2], [2, 1]], dtype=int),
        6: np.array([[2, 1], [3, 2]], dtype=int),
        8: np.array([[2, 3], [1, 2]], dtype=int),
    }
    if orientation not in mapping:
        raise ValueError(f"Unknown DNG orientation value: {orientation}")
    return mapping[orientation].copy()


def _normalize_sensor_dng_crop(crop: Any, size: tuple[int, int]) -> np.ndarray:
    crop_array = np.asarray(crop, dtype=float).reshape(-1)
    if crop_array.size == 4:
        row, col, height, width = crop_array[:4]
        return np.rint(np.array([col, row, width, height], dtype=float)).astype(int)
    if crop_array.size == 1:
        fraction = float(crop_array[0])
        if not (0.0 < fraction < 1.0):
            raise ValueError(f"Bad crop value {fraction}")
        sensor_size = np.asarray(size, dtype=float)
        middle_position = sensor_size / 2.0
        rowcol = fraction * sensor_size
        row = middle_position[0] - rowcol[0] / 2.0
        col = middle_position[1] - rowcol[1] / 2.0
        height = rowcol[0]
        width = rowcol[1]
        return np.rint(np.array([col, row, width, height], dtype=float)).astype(int)
    raise ValueError(f"Bad crop value {crop}")


def sensor_dng_read(
    fname: str | Path,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[Sensor, dict[str, Any]]:
    """Read a DNG file into an IMX363 sensor with MATLAB-style metadata handling."""

    full_info = True
    crop: Any = None
    if len(args) % 2 != 0:
        raise ValueError("sensorDNGRead expects key/value pairs.")
    for index in range(0, len(args), 2):
        parameter = param_format(args[index])
        value = args[index + 1]
        if parameter == "fullinfo":
            full_info = bool(value)
        elif parameter == "crop":
            crop = value
        else:
            raise UnsupportedOptionError("sensorDNGRead", str(args[index]))

    from .sensor import sensor_create, sensor_crop, sensor_get, sensor_set

    image, info = ie_dng_read(fname)
    if image is None:
        raise ValueError("ieDNGRead returned no image data.")
    simple_info = ie_dng_simple_info(info)
    black_level = int(np.ceil(float(simple_info["blackLevel"][0]))) if simple_info["blackLevel"].size else 0
    exposure_time = float(simple_info["exposureTime"]) if simple_info["exposureTime"] is not None else 0.0
    iso_speed = float(simple_info["isoSpeed"]) if simple_info["isoSpeed"] is not None else 1.0

    clipped_image = np.clip(np.asarray(image, dtype=float), black_level, None)
    sensor = sensor_create("IMX363", None, "isospeed", iso_speed, asset_store=asset_store)
    sensor = sensor_set(sensor, "size", clipped_image.shape[:2])
    sensor = sensor_set(sensor, "exp time", exposure_time)
    sensor = sensor_set(sensor, "black level", black_level)
    sensor = sensor_set(sensor, "name", str(Path(fname)))
    sensor = sensor_set(sensor, "digital values", clipped_image)
    sensor = sensor_set(sensor, "pattern", _dng_orientation_pattern(int(simple_info["orientation"])))

    if crop is not None:
        crop_rect = _normalize_sensor_dng_crop(crop, tuple(int(value) for value in sensor_get(sensor, "size")))
        sensor = sensor_crop(sensor, crop_rect)

    return sensor, info if full_info else simple_info


ieDNGRead = ie_dng_read
ieDNGSimpleInfo = ie_dng_simple_info
sensorDNGRead = sensor_dng_read
