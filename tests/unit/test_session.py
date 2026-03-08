from __future__ import annotations

from pyisetcam import (
    camera_compute,
    camera_create,
    camera_set,
    display_create,
    ieAddObject,
    ieDeleteObject,
    ieGetObject,
    ieGetSelectedObject,
    ieReplaceObject,
    ieSelectObject,
    ie_add_object,
    ie_delete_object,
    ie_get_object,
    ie_get_selected_object,
    ie_replace_object,
    ie_select_object,
    ip_create,
    ip_set,
    oi_create,
    scene_create,
    sensor_create,
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
    session_new_object_name,
    session_new_object_value,
    session_object_id,
    session_replace_and_select_object,
    session_replace_object,
    session_set_objects,
    session_set_selected,
    vcAddAndSelectObject,
    vcGetObject,
    vcGetObjectType,
    vcGetObjects,
    vcGetSelectedObject,
    vcDeleteSomeObjects,
    vcNewObjectName,
    vcNewObjectValue,
    vcSetObjects,
    vcSetSelectedObject,
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
    assert session_get_selected_pair(session, "scene") == (second_id, second_scene)
    assert session_get_object_with_id(session, "scene", second_id) == (second_scene, second_id)
    assert vcGetSelectedObject(session, "scene") == (second_id, second_scene)
    assert vcGetObject(session, "scene", second_id) == (second_scene, second_id)


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


def test_session_replace_object_preserves_slot_and_updates_names(asset_store) -> None:
    session = session_create()
    first_scene = scene_create("uniform ee", 8, asset_store=asset_store, session=session)
    second_scene = scene_create("uniform d65", 8, asset_store=asset_store, session=session)
    first_id = session_object_id(first_scene)

    replacement = scene_create("checkerboard", 4, 4, asset_store=asset_store)
    replaced = session_replace_object(session, replacement, first_id, select=False)

    assert session_object_id(replaced) == first_id
    assert session_get_object(session, "scene", first_id) is replaced
    assert session_get_selected(session, "scene") is second_scene
    assert session_count_objects(session, "scene") == 2
    assert session_get_object_names(session, "scene") == [replaced.name, second_scene.name]
    assert session_get_object_names(session, "scene", make_unique=True) == [f"1-{replaced.name}", f"2-{second_scene.name}"]


def test_session_replace_and_select_camera_tracks_subobjects(asset_store) -> None:
    session = session_create()
    original = camera_create(asset_store=asset_store, session=session)
    replacement = camera_create("mt9v024", "rgbw", asset_store=asset_store)
    slot_id = session_object_id(original)

    replaced = session_replace_and_select_object(session, replacement, slot_id)

    assert session_object_id(replaced) == slot_id
    assert session_get_selected(session, "camera") is replaced
    assert session_get_selected(session, "oi") is replaced.fields["oi"]
    assert session_get_selected(session, "sensor") is replaced.fields["sensor"]
    assert session_get_selected(session, "ip") is replaced.fields["ip"]
    assert session_get_selected(session, "display") is replaced.fields["ip"].fields["display"]


def test_session_alias_types_and_delete_renumbering(asset_store) -> None:
    session = session_create()
    scene_one = scene_create("uniform ee", 8, asset_store=asset_store, session=session)
    scene_two = scene_create("uniform d65", 8, asset_store=asset_store, session=session)
    scene_three = scene_create("checkerboard", 4, 4, asset_store=asset_store, session=session)
    sensor = sensor_create(asset_store=asset_store, session=session)
    ip = ip_create(sensor=sensor, asset_store=asset_store, session=session)

    assert session_get_selected(session, "isa") is sensor
    assert session_get_selected(session, "imgproc") is ip

    remaining = session_delete_object(session, "scene", 2)

    assert remaining == 2
    assert session_count_objects(session, "scene") == 2
    assert session_get_object(session, "scene", 1) is scene_one
    assert session_get_object(session, "scene", 2) is scene_three
    assert session_object_id(scene_three) == 2
    assert session_get_selected_pair(session, "scene") == (1, scene_one)

    remaining = session_delete_selected_object(session, "scene")

    assert remaining == 1
    assert session_get_selected_pair(session, "scene") == (1, scene_three)


def test_session_add_and_select_tracks_camera_and_aliases(asset_store) -> None:
    session = session_create()
    first_scene = scene_create("uniform ee", 8, asset_store=asset_store)
    camera = camera_create(asset_store=asset_store)

    first_id = session_add_and_select_object(session, "scene", first_scene)
    camera_id = vcAddAndSelectObject(session, camera)

    assert first_id == 1
    assert camera_id == 1
    assert session_get_selected(session, "scene") is first_scene
    assert session_get_selected(session, "camera") is camera
    assert session_get_selected(session, "oi") is camera.fields["oi"]
    assert session_get_selected(session, "sensor") is camera.fields["sensor"]
    assert session_get_selected(session, "ip") is camera.fields["ip"]
    assert session_get_selected(session, "display") is camera.fields["ip"].fields["display"]


def test_session_delete_some_objects_sorts_and_deduplicates(asset_store) -> None:
    session = session_create()
    scene_one = scene_create("uniform ee", 8, asset_store=asset_store, session=session)
    scene_two = scene_create("uniform d65", 8, asset_store=asset_store, session=session)
    scene_three = scene_create("checkerboard", 4, 4, asset_store=asset_store, session=session)
    scene_four = scene_create("slanted bar", 16, asset_store=asset_store, session=session)

    remaining = session_delete_some_objects(session, "scene", [2, 4, 2])

    assert remaining == 2
    assert session_get_object(session, "scene", 1) is scene_one
    assert session_get_object(session, "scene", 2) is scene_three
    assert session_object_id(scene_three) == 2
    assert session_get_selected_pair(session, "scene") == (1, scene_one)

    remaining = vcDeleteSomeObjects(session, "scene", [1])

    assert remaining == 1
    assert session_get_selected_pair(session, "scene") == (1, scene_three)
    assert scene_two is not scene_three
    assert scene_four is not scene_three


def test_session_new_object_name_and_selection_clearing_follow_matlab_style(asset_store) -> None:
    session = session_create()
    scene_create("uniform ee", 8, asset_store=asset_store, session=session)
    scene_create("uniform d65", 8, asset_store=asset_store, session=session)

    assert session_new_object_name(session, "scene") == "scene3"
    assert vcNewObjectName(session, "imgproc") == "ip1"

    session_set_selected(session, "scene", 0)
    assert session_get_selected(session, "scene") is None

    vcSetSelectedObject(session, "scene", 2)
    assert session_get_selected_id(session, "scene") == 2

    vcSetSelectedObject(session, "scene", -1)
    assert session_get_selected(session, "scene") is None


def test_ie_object_wrappers_follow_matlab_style_defaults(asset_store) -> None:
    session = session_create()
    scene = scene_create("uniform ee", 8, asset_store=asset_store)

    new_id = ie_add_object(session, scene)

    assert new_id == 1
    assert ie_get_selected_object(session, "scene") == 1
    assert ieGetSelectedObject(session, "scene", with_object=True) == (1, scene)
    assert ie_get_object(session, "scene") is scene
    assert ieGetObject(session, "scene", 1, with_id=True) == (scene, 1)

    ie_select_object(session, "scene", 0)
    assert session_get_selected(session, "scene") is None

    ieSelectObject(session, "scene", 1)
    assert session_get_selected(session, "scene") is scene


def test_ie_add_object_camera_returns_pipeline_ids(asset_store) -> None:
    session = session_create()
    camera = camera_create(asset_store=asset_store)

    pipeline_ids = ieAddObject(session, camera)

    assert pipeline_ids == (1, 1, 1)
    assert session_object_id(camera) == 1
    assert session_get_selected(session, "camera") is camera
    assert session_get_selected(session, "oi") is camera.fields["oi"]
    assert session_get_selected(session, "sensor") is camera.fields["sensor"]
    assert session_get_selected(session, "ip") is camera.fields["ip"]


def test_ie_get_object_supports_nested_optics_pixel_and_ipdisplay(asset_store) -> None:
    session = session_create()
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)

    session_add_object(session, oi)
    session_add_object(session, sensor)
    session_add_object(session, ip)

    assert ie_get_object(session, "optics") is oi.fields["optics"]
    assert ie_get_object(session, "pixel") is sensor.fields["pixel"]
    assert ie_get_object(session, "ipdisplay") is ip.fields["display"]
    assert ieGetSelectedObject(session, "optics", with_object=True) == (1, oi.fields["optics"])
    assert ieGetObject(session, "pixel", 1, with_id=True) == (sensor.fields["pixel"], 1)


def test_ie_delete_and_replace_object_follow_session_slots(asset_store) -> None:
    session = session_create()
    first_scene = scene_create("uniform ee", 8, asset_store=asset_store)
    second_scene = scene_create("uniform d65", 8, asset_store=asset_store)

    ieAddObject(session, first_scene)
    ieAddObject(session, second_scene)

    replacement = scene_create("checkerboard", 4, 4, asset_store=asset_store)
    replaced = ie_replace_object(session, replacement, 1)

    assert replaced is replacement
    assert session_get_object(session, "scene", 1) is replacement
    assert session_get_selected(session, "scene") is replacement

    remaining = ie_delete_object(session, "scene", 2)
    assert remaining == 1
    assert session_count_objects(session, "scene") == 1

    remaining = ieDeleteObject(session, "scene")
    assert remaining == 0
    assert session_get_selected(session, "scene") is None


def test_session_set_objects_and_new_object_value_follow_matlab_style(asset_store) -> None:
    session = session_create()
    scene_one = scene_create("uniform ee", 8, asset_store=asset_store)
    scene_two = scene_create("checkerboard", 4, 4, asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)
    camera = camera_create(asset_store=asset_store, session=session)

    session_set_objects(session, "scene", [scene_one, scene_two])
    session_add_object(session, sensor)
    session_add_object(session, ip)

    assert session_get_objects(session, "scene") == [scene_one, scene_two]
    assert vcGetObjects(session, "scene") == [scene_one, scene_two]
    assert session_object_id(scene_one) == 1
    assert session_object_id(scene_two) == 2
    assert session_new_object_value(session, "scene") == 3
    assert vcNewObjectValue(session, "scene") == 3
    assert session_new_object_value(session, "camera") == (2, 3, 3)
    assert vcNewObjectValue(session, "camera") == (2, 3, 3)
    assert session_object_id(camera) == 1
    assert session_get_object_type(ip) == "vcimage"
    assert vcGetObjectType(ip) == "vcimage"


def test_vcsetobjects_reindexes_and_clears_invalid_selection(asset_store) -> None:
    session = session_create()
    first = scene_create("uniform ee", 8, asset_store=asset_store, session=session)
    second = scene_create("uniform d65", 8, asset_store=asset_store, session=session)
    replacement = scene_create("checkerboard", 4, 4, asset_store=asset_store)

    session_set_selected(session, "scene", 2)
    vcSetObjects(session, "scene", [replacement])

    assert session_get_objects(session, "scene") == [replacement]
    assert session_object_id(replacement) == 1
    assert session_get_selected(session, "scene") is None
    assert first is not replacement
    assert second is not replacement
