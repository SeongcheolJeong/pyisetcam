from __future__ import annotations

from pyisetcam import (
    camera_compute,
    camera_create,
    camera_set,
    display_create,
    ip_create,
    ip_set,
    oi_create,
    scene_create,
    sensor_create,
    session_add_object,
    session_create,
    session_get_object,
    session_get_selected,
    session_get_selected_id,
    session_object_id,
    session_set_selected,
)


def test_session_add_get_and_select_round_trip(asset_store) -> None:
    session = session_create()
    first_scene = scene_create("uniform ee", 8, asset_store=asset_store)
    second_scene = scene_create("uniform d65", 8, asset_store=asset_store)

    first_id = session_add_object(session, first_scene)
    second_id = session_add_object(session, second_scene, select=False)

    assert first_id == 1
    assert second_id == 2
    assert session_object_id(first_scene) == 1
    assert session_get_selected_id(session, "scene") == 1
    assert session_get_selected(session, "scene") is first_scene
    assert session_get_object(session, "scene", second_id) is second_scene

    session_set_selected(session, "scene", second_id)
    assert session_get_selected(session, "scene") is second_scene


def test_camera_create_registers_session_subobjects(asset_store) -> None:
    session = session_create()
    camera = camera_create(asset_store=asset_store, session=session)

    assert session_get_selected(session, "camera") is camera
    assert session_get_selected(session, "oi") is camera.fields["oi"]
    assert session_get_selected(session, "sensor") is camera.fields["sensor"]
    assert session_get_selected(session, "ip") is camera.fields["ip"]
    assert session_object_id(camera) == 1
    assert session_object_id(camera.fields["sensor"]) == 1
    assert session_object_id(camera.fields["oi"]) == 1
    assert session_object_id(camera.fields["ip"]) == 1


def test_camera_compute_updates_registered_session_objects(asset_store) -> None:
    session = session_create()
    scene = scene_create(asset_store=asset_store, session=session)
    camera = camera_create("mt9v024", "rgbw", asset_store=asset_store, session=session)

    scene_id = session_object_id(scene)
    camera_id = session_object_id(camera)
    oi_id = session_object_id(camera.fields["oi"])
    sensor_id = session_object_id(camera.fields["sensor"])
    ip_id = session_object_id(camera.fields["ip"])

    computed = camera_compute(camera, scene, asset_store=asset_store, session=session)

    assert session_object_id(computed) == camera_id
    assert session_object_id(computed.fields["oi"]) == oi_id
    assert session_object_id(computed.fields["sensor"]) == sensor_id
    assert session_object_id(computed.fields["ip"]) == ip_id
    assert session_get_selected(session, "scene") is scene
    assert session_get_selected_id(session, "scene") == scene_id
    assert session_get_selected(session, "camera") is computed
    assert session_get_selected(session, "oi") is computed.fields["oi"]
    assert session_get_selected(session, "sensor") is computed.fields["sensor"]
    assert session_get_selected(session, "vci") is computed.fields["ip"]
    assert computed.fields["ip"].data["result"].shape[:2] == computed.fields["sensor"].fields["size"]


def test_ip_set_tracks_replacement_display_in_session(asset_store) -> None:
    session = session_create()
    ip = ip_create(asset_store=asset_store, session=session)
    ip_id = session_object_id(ip)
    replacement_display = display_create("default")

    updated = ip_set(ip, "display", replacement_display, session=session)

    assert session_object_id(updated) == ip_id
    assert session_get_selected(session, "ip") is updated
    assert session_get_selected(session, "display") is replacement_display
    assert updated.fields["display"] is replacement_display


def test_camera_set_tracks_replaced_session_subobjects(asset_store) -> None:
    session = session_create()
    camera = camera_create(asset_store=asset_store, session=session)
    camera_id = session_object_id(camera)
    replacement_oi = oi_create("pinhole")
    replacement_sensor = sensor_create("monochrome", asset_store=asset_store)
    replacement_ip = ip_create(sensor=replacement_sensor, asset_store=asset_store)

    camera = camera_set(camera, "oi", replacement_oi, session=session)
    camera = camera_set(camera, "sensor", replacement_sensor, session=session)
    camera = camera_set(camera, "ip", replacement_ip, session=session)

    assert session_object_id(camera) == camera_id
    assert session_get_selected(session, "camera") is camera
    assert session_get_selected(session, "oi") is replacement_oi
    assert session_get_selected(session, "sensor") is replacement_sensor
    assert session_get_selected(session, "ip") is replacement_ip
    assert session_get_selected(session, "display") is replacement_ip.fields["display"]
    assert camera.fields["oi"] is replacement_oi
    assert camera.fields["sensor"] is replacement_sensor
    assert camera.fields["ip"] is replacement_ip
