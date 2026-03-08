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
    "optics": "oi",
    "sensor": "sensor",
    "sensors": "sensor",
    "isa": "sensor",
    "pixel": "sensor",
    "display": "display",
    "displays": "display",
    "ip": "ip",
    "imgproc": "ip",
    "ipdisplay": "ip",
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


def session_get_selected_pair(
    session: SessionContext,
    object_type: str,
) -> tuple[int | None, BaseISETObject | None]:
    object_id = session_get_selected_id(session, object_type)
    return object_id, session_get_object(session, object_type, object_id)


def session_get_object_with_id(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> tuple[BaseISETObject | None, int | None]:
    if object_id is None:
        object_id = session_get_selected_id(session, object_type)
    return session_get_object(session, object_type, object_id), object_id


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


def session_get_objects(session: SessionContext, object_type: str) -> list[BaseISETObject]:
    objects = session_list_objects(session, object_type)
    return [obj for _, obj in sorted(objects.items())]


def session_count_objects(session: SessionContext, object_type: str) -> int:
    return len(session_get_objects(session, object_type))


def session_get_object_names(
    session: SessionContext,
    object_type: str = "scene",
    make_unique: bool = False,
) -> list[str]:
    names = [str(obj.name) for obj in session_get_objects(session, object_type)]
    if make_unique:
        return [f"{index + 1}-{name}" for index, name in enumerate(names)]
    return names


def session_set_objects(
    session: SessionContext,
    object_type: str,
    objects: dict[int, BaseISETObject] | list[BaseISETObject] | tuple[BaseISETObject, ...],
) -> list[BaseISETObject]:
    normalized_type = _session_type_name(object_type)
    if isinstance(objects, dict):
        ordered_objects = [obj for _, obj in sorted(objects.items())]
    else:
        ordered_objects = list(objects)

    new_bucket: dict[int, BaseISETObject] = {}
    for object_id, obj in enumerate(ordered_objects, start=1):
        if isinstance(obj, Camera):
            obj = track_camera_session_state(session, obj, select=False)
        elif isinstance(obj, ImageProcessor):
            obj = track_ip_session_state(session, obj, select=False)
        obj.metadata["session_type"] = normalized_type
        obj.metadata["session_id"] = object_id
        new_bucket[object_id] = obj

    session.objects[normalized_type] = new_bucket
    session.next_ids[normalized_type] = len(new_bucket) + 1
    selected_id = session.selected.get(normalized_type)
    if selected_id is not None and int(selected_id) not in new_bucket:
        session.selected[normalized_type] = None
    return ordered_objects


def session_new_object_value(
    session: SessionContext,
    object_type: str,
) -> int | tuple[int, int, int]:
    normalized_type = _session_type_name(object_type)
    if normalized_type == "camera":
        return (
            session_count_objects(session, "oi") + 1,
            session_count_objects(session, "sensor") + 1,
            session_count_objects(session, "ip") + 1,
        )
    return session_count_objects(session, normalized_type) + 1


def session_get_object_type(obj: BaseISETObject) -> str:
    object_type = getattr(obj, "type", None)
    if object_type is None:
        raise ValueError("Object does not have a type field.")
    return str(object_type)


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


def _session_reindex_bucket(bucket: dict[int, BaseISETObject]) -> dict[int, BaseISETObject]:
    reindexed: dict[int, BaseISETObject] = {}
    for new_id, obj in enumerate((bucket[key] for key in sorted(bucket)), start=1):
        obj.metadata["session_id"] = new_id
        reindexed[new_id] = obj
    return reindexed


def session_delete_selected_object(session: SessionContext, object_type: str) -> int:
    normalized_type = _session_type_name(object_type)
    selected_id = session_get_selected_id(session, normalized_type)
    if selected_id is None:
        return 0
    return session_delete_object(session, normalized_type, selected_id)


def session_delete_object(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> int:
    normalized_type = _session_type_name(object_type)
    bucket = dict(session.objects.get(normalized_type, {}))
    if object_id is None:
        object_id = session_get_selected_id(session, normalized_type)
    if object_id is None or int(object_id) not in bucket:
        return len(bucket)

    target_id = int(object_id)
    del bucket[target_id]
    if bucket:
        bucket = _session_reindex_bucket(bucket)
        session.objects[normalized_type] = bucket
        session.selected[normalized_type] = max(1, target_id - 1)
    else:
        session.objects[normalized_type] = {}
        session.selected[normalized_type] = None
    session.next_ids[normalized_type] = len(bucket) + 1
    return len(bucket)


# MATLAB-style aliases.
vcAddObject = session_add_object
vcCountObjects = session_count_objects
vcDeleteObject = session_delete_object
vcDeleteSelectedObject = session_delete_selected_object
vcGetObjectType = session_get_object_type
vcGetObjectNames = session_get_object_names
vcGetObject = session_get_object_with_id
vcGetObjects = session_get_objects
vcGetSelectedObject = session_get_selected_pair
vcGetSelectedObjectID = session_get_selected_id
vcNewObjectValue = session_new_object_value
vcReplaceAndSelectObject = session_replace_and_select_object
vcReplaceObject = session_replace_object
vcSetObjects = session_set_objects
vcSetSelectedObject = session_set_selected
