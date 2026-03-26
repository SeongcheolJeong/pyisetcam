from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyisetcam import (
    display_create,
    display_get,
    font_create,
    font_get,
    font_set,
    scene_from_font,
    scene_get,
)


def _expected_cached_bitmap(asset_store, relative_path: str) -> np.ndarray:
    path = asset_store.resolve(relative_path)
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    bitmap_plane = 1.0 - np.asarray(data["bmSrc"].dataIndex, dtype=float)
    pad_cols = 3 * int(np.ceil(bitmap_plane.shape[1] / 3.0)) - int(bitmap_plane.shape[1])
    if pad_cols > 0:
        bitmap_plane = np.pad(bitmap_plane, ((0, 0), (pad_cols, 0)), constant_values=1.0)
    bitmap = np.ones((bitmap_plane.shape[0], int(np.ceil(bitmap_plane.shape[1] / 3.0)), 3), dtype=float)
    for channel in range(3):
        bitmap[:, :, channel] = bitmap_plane[:, channel::3]
    return bitmap


def test_font_create_loads_cached_bitmap(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    expected = _expected_cached_bitmap(asset_store, "data/fonts/g-georgia-14-96.mat")

    assert font_get(font, "type") == "font"
    assert font_get(font, "name") == "g-georgia-14-96"
    np.testing.assert_allclose(font_get(font, "bitmap"), expected, rtol=0.0, atol=0.0)


def test_font_get_supports_padded_and_inverted_bitmaps(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)

    bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)
    padded = np.asarray(font_get(font, "padded bitmap", [2, 1], 0), dtype=float)
    inverted = np.asarray(font_get(font, "i bitmap"), dtype=float)

    assert padded.shape == (bitmap.shape[0] + 4, bitmap.shape[1] + 2, bitmap.shape[2])
    np.testing.assert_allclose(inverted, 1.0 - bitmap, rtol=0.0, atol=0.0)
    assert np.all(padded[:2, :, :] == 0.0)


def test_font_create_falls_back_to_headless_rendering(asset_store) -> None:
    font = font_create("A", "Georgia", 17, 97, asset_store=asset_store)
    bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)

    assert font_get(font, "name") == "a-georgia-17-97"
    assert bitmap.ndim == 3
    assert bitmap.shape[2] == 3
    assert bitmap.shape[0] > 0 and bitmap.shape[1] > 0
    assert set(np.unique(bitmap)).issubset({0.0, 1.0})


def test_font_set_rebuilds_bitmap_from_updated_cached_parameters(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    updated = font_set(font, "dpi", 300, asset_store=asset_store)
    expected = _expected_cached_bitmap(asset_store, "data/fonts/g-georgia-14-300.mat")

    assert font_get(updated, "name") == "g-georgia-14-300"
    np.testing.assert_allclose(font_get(updated, "bitmap"), expected, rtol=0.0, atol=0.0)


def test_scene_from_font_matches_display_wave_and_geometry(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    scene = scene_from_font(font, display, None, [2, 3], [1, 2], 1, asset_store=asset_store)

    padded_bitmap = np.asarray(font_get(font, "padded bitmap", [1, 2], 1), dtype=float)
    expected_wave = np.asarray(display_get(display, "wave"), dtype=float)
    expected_size = (padded_bitmap.shape[0] * 2, padded_bitmap.shape[1] * 3)
    expected_fov = float(np.degrees(np.arctan2((0.0254 / 96.0) * padded_bitmap.shape[1], 0.5)))

    assert scene_get(scene, "name") == font_get(font, "name")
    np.testing.assert_allclose(scene_get(scene, "wave"), expected_wave, rtol=0.0, atol=0.0)
    assert scene_get(scene, "size") == expected_size
    assert scene_get(scene, "distance") == 0.5
    assert scene_get(scene, "fov") == expected_fov
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) > 0.0
