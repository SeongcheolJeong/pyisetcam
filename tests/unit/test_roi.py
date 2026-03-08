from __future__ import annotations

import numpy as np

from pyisetcam import (
    ieLocs2Rect,
    ieRect2Locs,
    ieRect2Vertices,
    ieRoi2Locs,
    imageDataXYZ,
    ip_create,
    ip_get,
    ip_set,
    oi_create,
    oi_get,
    oi_set,
    scene_create,
    scene_get,
    sensor_create,
    sensor_get,
    sensor_set,
    vcGetROIData,
    vcRect2Locs,
    xyz_from_energy,
)
from pyisetcam.display import display_get


def test_roi_rect_location_helpers_round_trip() -> None:
    rect = np.array([4, 2, 5, 7], dtype=int)

    roi_locs = ieRect2Locs(rect)

    assert np.array_equal(roi_locs[0], np.array([2, 4], dtype=int))
    assert np.array_equal(roi_locs[-1], np.array([9, 9], dtype=int))
    assert np.array_equal(vcRect2Locs(rect), roi_locs)
    assert np.array_equal(ieRoi2Locs(rect), roi_locs)
    assert np.array_equal(ieLocs2Rect(roi_locs), rect)


def test_ie_rect2_vertices_supports_closed_polygon() -> None:
    xv, yv = ieRect2Vertices([3, 5, 2, 4], close_flag=True)

    assert np.array_equal(xv, np.array([3, 3, 5, 5, 3], dtype=int))
    assert np.array_equal(yv, np.array([5, 9, 9, 5, 5], dtype=int))


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


def test_sensor_get_roi_queries_and_means(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.5
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    sensor = sensor_set(sensor, "roi rect", np.array([1, 1, 1, 1], dtype=int))

    roi_locs = sensor_get(sensor, "roi locs")
    roi_volts = sensor_get(sensor, "roi volts")
    roi_electrons = sensor_get(sensor, "roi electrons")
    roi_volts_mean = sensor_get(sensor, "roi volts mean")
    roi_electrons_mean = sensor_get(sensor, "roi electrons mean")

    assert np.array_equal(roi_locs, np.array([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=int))
    assert np.array_equal(sensor_get(sensor, "roi rect"), np.array([1, 1, 1, 1], dtype=int))
    assert np.allclose(
        roi_volts,
        np.array(
            [
                [np.nan, 1.0, np.nan],
                [2.0, np.nan, np.nan],
                [np.nan, np.nan, 3.0],
                [np.nan, 4.0, np.nan],
            ],
            dtype=float,
        ),
        equal_nan=True,
    )
    assert np.allclose(
        roi_electrons,
        np.array(
            [
                [np.nan, 2.0, np.nan],
                [6.0, np.nan, np.nan],
                [np.nan, np.nan, 10.0],
                [np.nan, 14.0, np.nan],
            ],
            dtype=float,
        ),
        equal_nan=True,
    )
    assert np.allclose(roi_volts_mean, np.array([2.0, 2.5, 3.0], dtype=float))
    assert np.allclose(roi_electrons_mean, np.array([6.0, 8.0, 10.0], dtype=float))


def test_sensor_get_electrons_direct_getter(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.25
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[0.5, 1.0], [1.5, 2.0]], dtype=float))

    electrons = sensor_get(sensor, "electrons")

    assert np.allclose(electrons, np.array([[0.0, 4.0], [8.0, 12.0]], dtype=float))


def test_sensor_get_line_profiles(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.5
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))

    hline_volts = sensor_get(sensor, "hline volts", 1)
    vline_electrons = sensor_get(sensor, "vline electrons", 1)
    support = sensor_get(sensor, "spatial support")

    assert set(hline_volts) == {"data", "pos", "pixPos"}
    assert np.allclose(hline_volts["data"][0], np.array([2.0], dtype=float))
    assert np.allclose(hline_volts["data"][1], np.array([1.0], dtype=float))
    assert hline_volts["data"][2].size == 0
    assert np.allclose(hline_volts["pos"][0], np.array([support["x"][1]], dtype=float))
    assert np.allclose(hline_volts["pos"][1], np.array([support["x"][0]], dtype=float))
    assert hline_volts["pos"][2].size == 0
    assert np.allclose(hline_volts["pixPos"][0], hline_volts["pos"][0])
    assert np.allclose(hline_volts["pixPos"][1], hline_volts["pos"][1])
    assert hline_volts["pixPos"][2].size == 0

    assert vline_electrons["data"][0].size == 0
    assert np.allclose(vline_electrons["data"][1], np.array([2.0], dtype=float))
    assert np.allclose(vline_electrons["data"][2], np.array([10.0], dtype=float))
    assert vline_electrons["pos"][0].size == 0
    assert np.allclose(vline_electrons["pos"][1], np.array([support["y"][0]], dtype=float))
    assert np.allclose(vline_electrons["pos"][2], np.array([support["y"][1]], dtype=float))


def test_ip_get_roi_data_and_xyz(asset_store) -> None:
    ip = ip_create(display="default", asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)

    roi_locs = np.array([[1, 1], [2, 2]], dtype=int)
    roi_data = ip_get(ip, "roi data", roi_locs)
    roi_xyz = ip_get(ip, "roi xyz", roi_locs)
    full_xyz = imageDataXYZ(ip)

    spd = np.asarray(display_get(ip.fields["display"], "rgb spd"), dtype=float)
    wave = np.asarray(display_get(ip.fields["display"], "wave"), dtype=float)
    expected_roi_xyz = xyz_from_energy(roi_data @ spd.T, wave, asset_store=asset_store)
    expected_full_xyz = xyz_from_energy(result.reshape(-1, 3) @ spd.T, wave, asset_store=asset_store).reshape(2, 2, 3)

    assert np.allclose(roi_data, np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]], dtype=float))
    assert np.allclose(roi_xyz, expected_roi_xyz)
    assert np.allclose(full_xyz, expected_full_xyz)


def test_scene_get_roi_queries(asset_store) -> None:
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    roi = np.array([1, 1, 1, 1], dtype=int)

    roi_photons = scene_get(scene, "roi photons", roi)
    roi_energy = scene_get(scene, "roi energy", roi)
    roi_reflectance = scene_get(scene, "roi reflectance", roi)
    roi_luminance = scene_get(scene, "roi luminance", roi)

    assert roi_photons.shape[0] == 4
    assert np.allclose(scene_get(scene, "roi mean photons", roi), np.mean(roi_photons, axis=0))
    assert np.allclose(scene_get(scene, "roi mean energy", roi), np.mean(roi_energy, axis=0))
    assert np.allclose(scene_get(scene, "roi mean reflectance", roi), np.mean(roi_reflectance, axis=0))
    assert roi_luminance.shape == (4, 1)
    assert np.allclose(roi_reflectance, np.ones_like(roi_reflectance))


def test_oi_get_roi_queries(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)
    roi = np.array([1, 1, 1, 1], dtype=int)

    roi_photons = oi_get(oi, "roi photons", roi)
    roi_energy = oi_get(oi, "roi energy", roi)

    assert np.allclose(roi_photons, photons.reshape(-1, 2))
    assert np.allclose(oi_get(oi, "roi mean photons", roi), np.mean(roi_photons, axis=0))
    assert np.allclose(oi_get(oi, "roi mean energy", roi), np.mean(roi_energy, axis=0))
