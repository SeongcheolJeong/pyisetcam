from __future__ import annotations

from pyisetcam import (
    camera_compute,
    camera_create,
    camera_set,
    display_create,
    ieAddObject,
    ieAppGet,
    ieDeleteObject,
    ieGetObject,
    ieGetSelectedObject,
    ieInitSession,
    ieMainClose,
    ieReplaceObject,
    ieSessionGet,
    ieSessionSet,
    ieSelectObject,
    ie_add_object,
    ie_app_get,
    ie_delete_object,
    ie_get_object,
    ie_get_selected_object,
    ie_init_session,
    ie_replace_object,
    ie_session_get,
    ie_session_set,
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
    vcGetFigure,
    vcAddAndSelectObject,
    vcGetObject,
    vcGetObjectType,
    vcGetObjects,
    vcGetSelectedObject,
    vcSelectFigure,
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


def test_ie_init_session_and_session_get_defaults() -> None:
    session = ie_init_session()

    assert session.name.startswith("iset-")
    assert ieInitSession(directory=session.directory).directory == session.directory
    assert ie_session_get(session, "name") == session.name
    assert ieSessionGet(session, "dir") == session.directory
    assert ieSessionGet(session, "wait bar") == 0
    assert ieSessionGet(session, "font size") == 12
    assert ieSessionGet(session, "init clear") is False
    assert ieSessionGet(session, "image size threshold") == 1e6


def test_ie_session_set_and_get_metadata_preferences_and_gui(asset_store) -> None:
    session = ie_init_session()
    positions = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8, 0.9],
    ]

    ieSessionSet(session, "version", "1.2.3")
    ieSessionSet(session, "session name", "demo-session")
    ieSessionSet(session, "session dir", "/tmp/demo")
    ieSessionSet(session, "init help", 1)
    ieSessionSet(session, "font size", 14)
    ieSessionSet(session, "wait bar", "on")
    ieSessionSet(session, "window positions", positions)
    ieSessionSet(session, "init clear", "on")
    ieSessionSet(session, "main window", "main-app")
    ieSessionSet(session, "metrics window", "metrics-app", {"event": 1}, {"handles": 2})
    ieSessionSet(session, "graphwin figure", 17)
    ieSessionSet(session, "graphwin handle", "graph-handle")
    ieSessionSet(session, "gpu", True)
    ieSessionSet(session, "image size threshold", 2048)

    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    ieAddObject(session, scene)
    ieSessionSet(session, "scene", 1)

    assert ieSessionGet(session, "version") == "1.2.3"
    assert ieSessionGet(session, "session name") == "demo-session"
    assert ieSessionGet(session, "session dir") == "/tmp/demo"
    assert ieSessionGet(session, "help") is True
    assert ieSessionGet(session, "font size") == 14
    assert ieSessionGet(session, "wait bar") == 1
    assert ieSessionGet(session, "wpos")[:5] == positions[:5]
    assert ieSessionGet(session, "wpos")[5] is None
    assert ieSessionGet(session, "init clear") is True
    assert ieSessionGet(session, "main window") == "main-app"
    assert ieSessionGet(session, "metrics window") == "metrics-app"
    assert ieSessionGet(session, "graphwin figure") == 17
    assert ieSessionGet(session, "graphwin handle") == "graph-handle"
    assert ieSessionGet(session, "gpu computing") is True
    assert ieSessionGet(session, "image size threshold") == 2048.0
    assert ieSessionGet(session, "scene") is scene
    assert ieSessionGet(session, "selected", "scene") == 1
    assert ieSessionGet(session, "nobjects", "scene") == 1
    assert ieSessionGet(session, "names", "scene") == [scene.name]


def test_ie_session_window_aliases_and_ie_app_get(asset_store) -> None:
    session = ie_init_session()
    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)
    display = display_create("default")

    scene_app = {"sceneImage": "scene-axis", "figure1": "scene-figure"}
    oi_app = {"oiImage": "oi-axis", "figure1": "oi-figure"}
    sensor_app = {"imgMain": "sensor-axis", "figure1": "sensor-figure"}
    ip_app = {"ipImage": "ip-axis", "figure1": "ip-figure"}
    display_app = {"displayImage": "display-axis", "figure1": "display-figure"}

    ieAddObject(session, scene)
    session_add_object(session, oi)
    session_add_object(session, sensor)
    session_add_object(session, ip)
    session_add_object(session, display)

    ieSessionSet(session, "scene window", scene_app)
    ieSessionSet(session, "oi window", oi_app)
    ieSessionSet(session, "sensor window", sensor_app)
    ieSessionSet(session, "ip window", ip_app)
    ieSessionSet(session, "display window", display_app)

    assert ieSessionGet(session, "scene figure") is scene_app
    assert ieSessionGet(session, "scene image figures") is scene_app
    assert ieSessionGet(session, "oi figure") is oi_app
    assert ieSessionGet(session, "optical image figures") is oi_app
    assert ieSessionGet(session, "isa window") is sensor_app
    assert ieSessionGet(session, "vcimage figure") is ip_app
    assert ieSessionGet(session, "display window") is display_app
    assert ieSessionGet(session, "scene window handle") == "scene-figure"
    assert ieSessionGet(session, "scene image handle") == "scene-axis"
    assert ieSessionGet(session, "sensorimagehandle") == "sensor-axis"

    assert ie_app_get(session, scene) == (scene_app, "scene-axis")
    assert ieAppGet(session, "oi") == (oi_app, "oi-axis")
    assert ieAppGet(session, "sensor") == (sensor_app, "sensor-axis")
    assert ieAppGet(session, "ip") == (ip_app, "ip-axis")
    assert ieAppGet(session, "display") == (display_app, "display-axis")


def test_ie_app_get_accepts_direct_app_like_objects() -> None:
    session = ie_init_session()
    app = {"current_axes": "free-axis"}

    assert ie_app_get(session, app) == (app, "free-axis")


def test_ie_session_gui_handle_aliases_and_custom_lists() -> None:
    session = ie_init_session()

    oi_state = {"oiImage": "oi-axis", "figure1": "oi-figure"}
    sensor_state = {"imgMain": "sensor-axis", "figure1": "sensor-figure"}
    ip_state = {"ipImage": "ip-axis", "figure1": "ip-figure"}

    ieSessionSet(session, "oi window", "oi-window-h", {"event": 1}, oi_state)
    ieSessionSet(session, "sensor window", "sensor-window-h", {"event": 2}, sensor_state)
    ieSessionSet(session, "ip window", "ip-window-h", {"event": 3}, ip_state)
    ieSessionSet(session, "metrics window", "metrics-window-h", {"event": 4}, {"msg": "handles"})
    ieSessionSet(session, "oicomputelist", ["customOTF", "myCompute"])
    ieSessionSet(session, "sensor gamma", 0.35)

    assert ieSessionGet(session, "oiwindowHandles") == oi_state
    assert ieSessionGet(session, "oi guidata") == oi_state
    assert ieSessionGet(session, "sensorWindowHandles") == sensor_state
    assert ieSessionGet(session, "sensor guidata") == sensor_state
    assert ieSessionGet(session, "vcimagehandles") == ip_state
    assert ieSessionGet(session, "metricshandles") == {"msg": "handles"}
    assert ieSessionGet(session, "oicomputelist") == ["customOTF", "myCompute"]
    assert ieSessionGet(session, "sensor gamma") == 0.35


def test_vc_get_figure_and_vc_select_figure_follow_session_windows(asset_store) -> None:
    session = ie_init_session()
    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)

    scene_app = {"sceneImage": "scene-axis", "figure1": "scene-figure"}
    oi_app = {"oiImage": "oi-axis", "figure1": "oi-figure"}
    sensor_app = {"imgMain": "sensor-axis", "figure1": "sensor-figure"}
    ip_app = {"ipImage": "ip-axis", "figure1": "ip-figure"}

    ieSessionSet(session, "scene window", scene_app)
    ieSessionSet(session, "oi window", oi_app)
    ieSessionSet(session, "sensor window", sensor_app)
    ieSessionSet(session, "ip window", ip_app)
    ieSessionSet(session, "graphwin figure", "graph-figure")

    assert vcGetFigure(session, scene) == (scene_app, "scene-axis")
    assert vcGetFigure(session, oi) == (oi_app, "oi-axis")
    assert vcGetFigure(session, sensor) == (sensor_app, "sensor-axis")
    assert vcGetFigure(session, ip) == (ip_app, "ip-axis")
    assert vcSelectFigure(session, "scene") == "scene-figure"
    assert vcSelectFigure(session, "oi") == "oi-figure"
    assert vcSelectFigure(session, "sensor") == "sensor-figure"
    assert vcSelectFigure(session, "vcimage") == "ip-figure"
    assert vcSelectFigure(session, "graphwin") == "graph-figure"
    assert vcSelectFigure(session, "display", True) is None


def test_vc_select_figure_without_existing_window_raises_when_creation_needed() -> None:
    session = ie_init_session()

    try:
        vcSelectFigure(session, "scene")
    except NotImplementedError as error:
        assert "vcSelectFigure scene window creation" in str(error)
    else:
        raise AssertionError("Expected NotImplementedError for missing scene window.")


def test_ie_main_close_clears_window_slots() -> None:
    session = ie_init_session()
    ieSessionSet(session, "scene window", {"sceneImage": "scene-axis", "figure1": "scene-figure"})
    ieSessionSet(session, "oi window", {"oiImage": "oi-axis", "figure1": "oi-figure"})
    ieSessionSet(session, "sensor window", {"imgMain": "sensor-axis", "figure1": "sensor-figure"})
    ieSessionSet(session, "ip window", {"ipImage": "ip-axis", "figure1": "ip-figure"})
    ieSessionSet(session, "display window", {"displayImage": "display-axis", "figure1": "display-figure"})
    ieSessionSet(session, "metrics window", "metrics-window", {"event": 1}, {"msg": "handles"})

    ieMainClose(session)

    assert ieSessionGet(session, "scene window") is None
    assert ieSessionGet(session, "oi window") is None
    assert ieSessionGet(session, "sensor window") is None
    assert ieSessionGet(session, "ip window") is None
    assert ieSessionGet(session, "display window") is None
    assert ieSessionGet(session, "metrics window") is None

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
