"""Optional vcSESSION-style object registry helpers."""

from __future__ import annotations

from typing import TypeVar

from .types import BaseISETObject, Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import param_format

_T = TypeVar("_T", bound=BaseISETObject)

_SESSION_TYPE_ALIASES = {
    "scene": "scene",
    "scenes": "scene",
    "oi": "oi",
    "opticalimage": "oi",
    "opticalimages": "oi",
    "sensor": "sensor",
    "sensors": "sensor",
    "display": "display",
    "displays": "display",
    "ip": "ip",
    "vcimage": "ip",
    "vci": "ip",
    "imageprocessor": "ip",
    "camera": "camera",
    "cameras": "camera",
}


def _session_type_name(value: str) -> str:
    normalized = param_format(value)
    if normalized not in _SESSION_TYPE_ALIASES:
        raise KeyError(f"Unsupported session object type: {value}")
    return _SESSION_TYPE_ALIASES[normalized]


def _object_session_type(obj: BaseISETObject) -> str:
    if isinstance(obj, Scene):
        return "scene"
    if isinstance(obj, OpticalImage):
        return "oi"
    if isinstance(obj, Sensor):
        return "sensor"
    if isinstance(obj, Display):
        return "display"
    if isinstance(obj, ImageProcessor):
        return "ip"
    if isinstance(obj, Camera):
        return "camera"
    return _session_type_name(obj.type)


def session_create(name: str = "vcSESSION") -> SessionContext:
    return SessionContext(name=name)


def session_object_id(obj: BaseISETObject) -> int | None:
    session_id = obj.metadata.get("session_id")
    return int(session_id) if session_id is not None else None


def session_add_object(
    session: SessionContext,
    obj: _T,
    *,
    select: bool = True,
    object_id: int | None = None,
) -> int:
    object_type = _object_session_type(obj)
    bucket = session.objects.setdefault(object_type, {})
    session.next_ids.setdefault(object_type, 1)

    if object_id is None:
        existing_type = obj.metadata.get("session_type")
        existing_id = obj.metadata.get("session_id")
        if existing_type == object_type and existing_id in bucket:
            object_id = int(existing_id)
        else:
            object_id = int(session.next_ids[object_type])
            session.next_ids[object_type] = object_id + 1
    else:
        object_id = int(object_id)
        session.next_ids[object_type] = max(int(session.next_ids[object_type]), object_id + 1)

    bucket[object_id] = obj
    obj.metadata["session_type"] = object_type
    obj.metadata["session_id"] = object_id
    if select:
        session.selected[object_type] = object_id
    return object_id


def track_session_object(session: SessionContext | None, obj: _T, *, select: bool = True) -> _T:
    if session is not None:
        session_add_object(session, obj, select=select)
    return obj


def track_ip_session_state(
    session: SessionContext | None,
    ip: ImageProcessor,
    *,
    select: bool = True,
) -> ImageProcessor:
    if session is None:
        return ip
    display = ip.fields.get("display")
    if isinstance(display, Display):
        ip.fields["display"] = track_session_object(session, display, select=select)
    return track_session_object(session, ip, select=select)


def track_camera_session_state(
    session: SessionContext | None,
    camera: Camera,
    *,
    select: bool = True,
) -> Camera:
    if session is None:
        return camera
    oi = camera.fields.get("oi")
    if isinstance(oi, OpticalImage):
        camera.fields["oi"] = track_session_object(session, oi, select=select)
    sensor = camera.fields.get("sensor")
    if isinstance(sensor, Sensor):
        camera.fields["sensor"] = track_session_object(session, sensor, select=select)
    ip = camera.fields.get("ip")
    if isinstance(ip, ImageProcessor):
        camera.fields["ip"] = track_ip_session_state(session, ip, select=select)
    return track_session_object(session, camera, select=select)


def session_get_object(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> BaseISETObject | None:
    normalized_type = _session_type_name(object_type)
    if object_id is None:
        object_id = session.selected.get(normalized_type)
    if object_id is None:
        return None
    return session.objects.get(normalized_type, {}).get(int(object_id))


def session_get_selected(session: SessionContext, object_type: str) -> BaseISETObject | None:
    return session_get_object(session, object_type)


def session_get_selected_id(session: SessionContext, object_type: str) -> int | None:
    normalized_type = _session_type_name(object_type)
    selected = session.selected.get(normalized_type)
    return int(selected) if selected is not None else None


def session_set_selected(
    session: SessionContext,
    object_type: str,
    value: int | BaseISETObject | None,
) -> None:
    normalized_type = _session_type_name(object_type)
    if value is None:
        session.selected[normalized_type] = None
        return
    if isinstance(value, BaseISETObject):
        session_add_object(session, value, select=True)
        return
    object_id = int(value)
    if object_id not in session.objects.get(normalized_type, {}):
        raise KeyError(f"Session object id {object_id} is not registered for type {normalized_type}.")
    session.selected[normalized_type] = object_id


def session_list_objects(
    session: SessionContext,
    object_type: str | None = None,
) -> dict[int, BaseISETObject] | dict[str, dict[int, BaseISETObject]]:
    if object_type is None:
        return {key: dict(value) for key, value in session.objects.items()}
    normalized_type = _session_type_name(object_type)
    return dict(session.objects.get(normalized_type, {}))


def session_count_objects(session: SessionContext, object_type: str) -> int:
    return len(session_list_objects(session, object_type))


def session_get_object_names(
    session: SessionContext,
    object_type: str = "scene",
    make_unique: bool = False,
) -> list[str]:
    objects = session_list_objects(session, object_type)
    names = [str(obj.name) for _, obj in sorted(objects.items())]
    if make_unique:
        return [f"{index + 1}-{name}" for index, name in enumerate(names)]
    return names


def session_replace_object(
    session: SessionContext,
    obj: _T,
    object_id: int | None = None,
    *,
    select: bool = True,
) -> _T:
    object_type = _object_session_type(obj)
    bucket = session.objects.setdefault(object_type, {})
    if object_id is None:
        existing_id = session_object_id(obj)
        if existing_id is not None and existing_id in bucket:
            object_id = existing_id
        else:
            selected_id = session_get_selected_id(session, object_type)
            object_id = 1 if selected_id is None else selected_id
    object_id = int(object_id)

    if isinstance(obj, Camera):
        obj = track_camera_session_state(session, obj, select=select)
    elif isinstance(obj, ImageProcessor):
        obj = track_ip_session_state(session, obj, select=select)

    bucket[object_id] = obj
    obj.metadata["session_type"] = object_type
    obj.metadata["session_id"] = object_id
    session.next_ids[object_type] = max(int(session.next_ids.get(object_type, 1)), object_id + 1)
    if select:
        session.selected[object_type] = object_id
    return obj


def session_replace_and_select_object(
    session: SessionContext,
    obj: _T,
    object_id: int | None = None,
) -> _T:
    return session_replace_object(session, obj, object_id, select=True)


# MATLAB-style aliases.
vcAddObject = session_add_object
vcCountObjects = session_count_objects
vcGetObjectNames = session_get_object_names
vcGetObject = session_get_object
vcGetObjects = session_list_objects
vcGetSelectedObject = session_get_selected
vcGetSelectedObjectID = session_get_selected_id
vcReplaceAndSelectObject = session_replace_and_select_object
vcReplaceObject = session_replace_object
vcSetSelectedObject = session_set_selected
