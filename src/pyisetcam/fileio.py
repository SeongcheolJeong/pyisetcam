"""Headless MATLAB-style object save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from scipy.io import loadmat, savemat

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
