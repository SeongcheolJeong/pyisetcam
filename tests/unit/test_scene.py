from __future__ import annotations

import numpy as np

from pyisetcam import blackbody, scene_adjust_illuminant, scene_create, scene_get


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
