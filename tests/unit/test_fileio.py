from __future__ import annotations

import numpy as np

from pyisetcam import (
    Camera,
    Scene,
    camera_create,
    scene_create,
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
