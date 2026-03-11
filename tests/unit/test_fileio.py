from __future__ import annotations

import numpy as np

from pyisetcam import (
    Camera,
    Scene,
    camera_create,
    ieDNGRead,
    ieDNGSimpleInfo,
    ie_dng_read,
    ie_dng_simple_info,
    scene_create,
    sensorDNGRead,
    sensor_crop,
    sensor_dng_read,
    sensor_get,
    session_create,
    session_get_selected,
    vcExportObject,
    vcLoadObject,
    vcSaveObject,
    vc_export_object,
    vc_load_object,
    vc_save_object,
)


def test_vc_save_and_load_object_round_trip_scene(tmp_path, asset_store) -> None:
    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    path = tmp_path / "scene_round_trip.mat"

    saved = vc_save_object(scene, path)
    loaded, full_name = vc_load_object("scene", saved)

    assert full_name == str(path)
    assert path.exists()
    assert isinstance(loaded, Scene)
    assert loaded.name == "scene_round_trip"
    assert loaded.type == "scene"
    assert np.allclose(np.asarray(loaded.fields["wave"], dtype=float), np.asarray(scene.fields["wave"], dtype=float))
    assert np.allclose(np.asarray(loaded.data["photons"], dtype=float), np.asarray(scene.data["photons"], dtype=float))
    assert vcSaveObject(scene, tmp_path / "scene_round_trip_alias.mat").endswith(".mat")


def test_vc_export_object_clear_data_flag_preserves_original(tmp_path, asset_store) -> None:
    scene = scene_create("uniform d65", 8, asset_store=asset_store)
    original_photons = np.asarray(scene.data["photons"], dtype=float).copy()
    path = tmp_path / "scene_export.mat"

    saved = vc_export_object(scene, path, clear_data_flag=True)
    loaded, _ = vc_load_object("scene", saved)

    assert np.allclose(np.asarray(scene.data["photons"], dtype=float), original_photons)
    assert loaded.data == {}
    assert vcExportObject(scene, tmp_path / "scene_export_alias.mat", clear_data_flag=False).endswith(".mat")


def test_vc_load_object_registers_in_session(tmp_path, asset_store) -> None:
    scene = scene_create("checkerboard", 4, 4, asset_store=asset_store)
    path = tmp_path / "session_scene.mat"
    session = session_create()

    vc_save_object(scene, path)
    slot, full_name = vc_load_object("scene", path, session=session)

    assert slot == 1
    assert full_name == str(path)
    loaded = session_get_selected(session, "scene")
    assert isinstance(loaded, Scene)
    assert loaded.name == "session_scene"
    assert np.allclose(np.asarray(loaded.data["photons"], dtype=float), np.asarray(scene.data["photons"], dtype=float))


def test_vc_save_and_load_camera_reconstructs_nested_objects(tmp_path, asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    path = tmp_path / "camera_round_trip.mat"

    vc_save_object(camera, path)
    loaded, _ = vcLoadObject("camera", path)

    assert isinstance(loaded, Camera)
    assert loaded.name == "camera_round_trip"
    assert loaded.fields["oi"].type == "opticalimage"
    assert loaded.fields["sensor"].type == "sensor"
    assert loaded.fields["ip"].type == "vcimage"
    assert loaded.fields["ip"].fields["display"].type == "display"


def test_ie_dng_read_supports_raw_rgb_and_simple_info(asset_store) -> None:
    dng_path = asset_store.resolve("data/images/rawcamera/MCC-centered.dng")

    raw, info = ie_dng_read(dng_path)
    rgb, _ = ie_dng_read(dng_path, "rgb", True)
    none_img, simple_info = ie_dng_read(dng_path, "only info", True, "simple info", True)

    assert raw is not None
    assert raw.shape == (3024, 4032)
    assert raw.dtype == np.uint16
    assert info["Make"] == "Google"
    assert info["Model"] == "Pixel 4a"
    assert info["Orientation"] == 6
    assert int(info["DigitalCamera"]["ISOSpeedRatings"]) == 64
    assert np.allclose(np.asarray(info["BlackLevel"], dtype=float), np.array([1023.0, 1023.0, 1022.0, 1022.0]))

    assert rgb is not None
    assert rgb.shape == (504, 672, 3)
    assert rgb.dtype == np.uint8

    assert none_img is None
    assert simple_info["isoSpeed"] == 64
    assert np.isclose(simple_info["exposureTime"], 35822828 / 1073741824)
    assert simple_info["orientation"] == 6
    assert np.allclose(np.asarray(simple_info["blackLevel"], dtype=float), np.array([1023.0, 1023.0, 1022.0, 1022.0]))

    alias_img, alias_info = ieDNGRead(dng_path, "simple info", True)
    direct_simple = ieDNGSimpleInfo(info)
    assert alias_img is not None
    assert alias_info["isoSpeed"] == direct_simple["isoSpeed"] == ie_dng_simple_info(info)["isoSpeed"]
    assert np.isclose(alias_info["exposureTime"], direct_simple["exposureTime"])
    assert alias_info["orientation"] == direct_simple["orientation"]
    assert np.allclose(alias_info["blackLevel"], direct_simple["blackLevel"])


def test_sensor_dng_read_supports_imx363_raw_tutorial_flow(asset_store) -> None:
    dng_path = asset_store.resolve("data/images/rawcamera/MCC-centered.dng")

    sensor, info = sensor_dng_read(dng_path, asset_store=asset_store)
    cropped, simple_info = sensor_dng_read(
        dng_path,
        "full info",
        False,
        "crop",
        [500, 1000, 2500, 2500],
        asset_store=asset_store,
    )
    fractional, _ = sensorDNGRead(dng_path, "crop", 0.4, asset_store=asset_store)

    assert sensor.name == str(dng_path)
    assert sensor_get(sensor, "size") == (3024, 4032)
    assert np.array_equal(sensor_get(sensor, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert sensor_get(sensor, "black level") == 1023
    assert np.isclose(sensor_get(sensor, "exp time"), 35822828 / 1073741824)
    dv = np.asarray(sensor_get(sensor, "digital values"), dtype=float)
    assert dv.shape == (3024, 4032)
    assert float(dv.min()) >= 1023.0

    manual_crop = sensor_crop(sensor, [1000, 500, 2500, 2500])
    assert sensor_get(cropped, "size") == sensor_get(manual_crop, "size")
    assert np.array_equal(sensor_get(cropped, "metadata crop"), sensor_get(manual_crop, "metadata crop"))
    assert np.array_equal(sensor_get(cropped, "digital values"), sensor_get(manual_crop, "digital values"))

    assert isinstance(info["DigitalCamera"], dict)
    assert simple_info["isoSpeed"] == 64
    assert sensor_get(fractional, "size")[0] < sensor_get(sensor, "size")[0]
    assert sensor_get(fractional, "size")[1] < sensor_get(sensor, "size")[1]
