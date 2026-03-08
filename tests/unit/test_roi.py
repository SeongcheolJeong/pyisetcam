from __future__ import annotations

import numpy as np

from pyisetcam import ip_create, ip_set, oi_create, oi_set, scene_create, scene_get, sensor_create, sensor_set, vcGetROIData


def test_vc_get_roi_data_scene_rect_returns_xw_illuminant_energy(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)

    roi_data = vcGetROIData(scene, [2, 2, 1, 1], "illuminant energy")

    expected = np.tile(np.asarray(scene_get(scene, "illuminant energy"), dtype=float), (4, 1))
    assert roi_data.shape == expected.shape
    assert np.allclose(roi_data, expected)


def test_vc_get_roi_data_oi_clips_out_of_bounds(asset_store) -> None:
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

    roi_data = vcGetROIData(oi, np.array([[0, 0], [9, 9]]), "photons")

    assert roi_data.shape == (2, 2)
    assert np.allclose(roi_data[0], photons[0, 0, :])
    assert np.allclose(roi_data[1], photons[-1, -1, :])


def test_vc_get_roi_data_sensor_mosaic_returns_nan_planes(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)

    roi_data = vcGetROIData(sensor, np.array([[1, 1], [1, 2], [2, 1], [2, 2]]), "volts")

    expected = np.array(
        [
            [np.nan, 1.0, np.nan],
            [2.0, np.nan, np.nan],
            [np.nan, np.nan, 3.0],
            [np.nan, 4.0, np.nan],
        ],
        dtype=float,
    )
    assert roi_data.shape == expected.shape
    assert np.allclose(roi_data, expected, equal_nan=True)


def test_vc_get_roi_data_ip_result_and_input(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "input", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    ip = ip_set(
        ip,
        "result",
        np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ],
            dtype=float,
        ),
    )

    result_roi = vcGetROIData(ip, np.array([[2, 2]]))
    input_roi = vcGetROIData(ip, np.array([[2, 1]]), "input")

    assert result_roi.shape == (1, 3)
    assert np.allclose(result_roi[0], np.array([1.0, 1.1, 1.2], dtype=float))
    assert input_roi.shape == (1, 1)
    assert np.allclose(input_roi[:, 0], np.array([3.0], dtype=float))
