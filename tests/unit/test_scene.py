from __future__ import annotations

import imageio.v3 as iio
import numpy as np

from pyisetcam import blackbody, display_create, display_get, scene_adjust_illuminant, scene_create, scene_from_file, scene_get


def test_scene_create_default_macbeth(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    assert photons.shape == (64, 96, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_adjust_illuminant_preserves_mean(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    wave = scene_get(scene, "wave")
    changed = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=asset_store)
    changed_no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=asset_store)
    assert np.isclose(
        scene_get(changed, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=5e-2,
    )
    assert not np.isclose(
        scene_get(changed_no_preserve, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=1e-2,
    )


def test_supported_pattern_scenes(asset_store) -> None:
    checkerboard = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    slanted_bar = scene_create("slanted bar", 64, 0.6, 3.0, asset_store=asset_store)
    assert scene_get(checkerboard, "photons").shape[:2] == (64, 64)
    assert scene_get(slanted_bar, "photons").shape[:2] == (64, 64)


def test_scene_from_file_rgb_array_uses_display_geometry(asset_store) -> None:
    display = display_create("default", asset_store=asset_store)
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )

    scene = scene_from_file(image, "rgb", 50.0, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 2, scene_get(scene, "wave").size)
    assert np.isclose(scene_get(scene, "distance"), display_get(display, "viewing distance"))
    assert np.isclose(scene_get(scene, "fov"), 2.0 * display_get(display, "deg per dot"))
    assert scene_get(scene, "filename") == "numerical"
    assert scene_get(scene, "source type") == "rgb"
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 50.0, rtol=5e-2)


def test_scene_from_file_monochrome_file_path(tmp_path, asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.array(
        [
            [0, 64, 128],
            [255, 192, 32],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "mono.png"
    iio.imwrite(image_path, image)

    scene = scene_from_file(image_path, "monochrome", None, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 3, scene_get(scene, "wave").size)
    assert scene_get(scene, "filename") == str(image_path)
    assert scene.name.startswith("mono - ")
    assert np.all(np.isfinite(photons))
    assert float(np.mean(photons[1, 0, :])) > float(np.mean(photons[0, 0, :]))


def test_scene_from_file_preserves_display_wave_when_not_overridden(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scene = scene_from_file(image, "rgb", None, display, asset_store=asset_store)
    assert np.array_equal(scene_get(scene, "wave"), display_get(display, "wave"))
