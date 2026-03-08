from __future__ import annotations

import numpy as np

from pyisetcam import oiPlot, oi_create, oi_get, oi_set, plotScene, scene_create, scene_get, scene_set


def test_plot_scene_radiance_photons_roi_and_chromaticity(asset_store) -> None:
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    scene = scene_set(scene, "wave", np.array([500.0, 600.0], dtype=float))
    scene = scene_set(
        scene,
        "photons",
        np.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ],
            dtype=float,
        ),
    )
    roi = np.array([1, 1, 1, 1], dtype=int)

    radiance_udata, handle = plotScene(scene, "radiance photons roi", roi, asset_store=asset_store)
    chromaticity_udata, chroma_handle = plotScene(scene, "chromaticity", roi, asset_store=asset_store)

    expected_photons = np.mean(np.asarray(scene_get(scene, "roi photons", roi, asset_store=asset_store), dtype=float), axis=0)
    expected_xy = np.asarray(scene_get(scene, "chromaticity", roi, asset_store=asset_store), dtype=float)

    assert handle is None
    assert chroma_handle is None
    assert np.allclose(radiance_udata["wave"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(radiance_udata["photons"], expected_photons)
    assert np.allclose(chromaticity_udata["x"], expected_xy[:, 0])
    assert np.allclose(chromaticity_udata["y"], expected_xy[:, 1])


def test_plot_scene_illuminant_energy(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)

    udata, handle = plotScene(scene, "illuminant energy")

    assert handle is None
    assert udata["comment"] == "D65.mat"
    assert np.allclose(udata["wave"], np.asarray(scene_get(scene, "wave"), dtype=float))
    assert np.allclose(udata["energy"], np.asarray(scene_get(scene, "illuminant energy"), dtype=float))


def test_oi_plot_roi_and_line_data(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)
    roi = np.array([1, 1, 2, 1], dtype=int)
    line = np.array([1, 1], dtype=int)

    roi_udata, roi_handle = oiPlot(oi, "irradiance photons roi", roi)
    line_udata, line_handle = oiPlot(oi, "illuminance hline", line)

    expected_y = np.mean(np.asarray(oi_get(oi, "roi photons", roi), dtype=float), axis=0)
    expected_line = oi_get(oi, "illuminance hline", line)

    assert roi_handle is None
    assert line_handle is None
    assert np.allclose(roi_udata["x"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(roi_udata["y"], expected_y)
    assert np.array_equal(roi_udata["roiLocs"], roi)
    assert np.allclose(line_udata["pos"], np.asarray(expected_line["pos"], dtype=float))
    assert np.allclose(line_udata["data"], np.asarray(expected_line["data"], dtype=float))
    assert np.array_equal(line_udata["roiLocs"], line)
