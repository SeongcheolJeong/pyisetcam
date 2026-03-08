from __future__ import annotations

import numpy as np

from pyisetcam import (
    ipPlot,
    ip_create,
    ip_get,
    ip_set,
    oiPlot,
    oi_create,
    oi_get,
    oi_set,
    plotScene,
    plotSensor,
    scene_create,
    scene_get,
    scene_set,
    sensor_create,
    sensor_get,
    sensor_set,
    vc_get_roi_data,
)


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


def test_plot_sensor_line_data(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(
        sensor,
        "dv",
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=float,
        ),
    )

    volts_udata, volts_handle = plotSensor(sensor, "volts hline", np.array([1, 2], dtype=int))
    electrons_udata, electrons_handle = plotSensor(sensor, "electrons hline", np.array([2, 2], dtype=int))
    dv_udata, dv_handle = plotSensor(sensor, "dv vline", np.array([2, 1], dtype=int))

    expected_volts = sensor_get(sensor, "hline volts", 2)
    expected_electrons = sensor_get(sensor, "hline electrons", 2)
    expected_dv = sensor_get(sensor, "vline dv", 2)

    assert volts_handle is None
    assert electrons_handle is None
    assert dv_handle is None
    assert np.array_equal(volts_udata["xy"], np.array([1, 2], dtype=int))
    assert volts_udata["ori"] == "h"
    assert volts_udata["dataType"] == "volts"
    assert np.allclose(volts_udata["data"][0], np.asarray(expected_volts["data"][0], dtype=float))
    assert np.allclose(volts_udata["pos"][0], 1e6 * np.asarray(expected_volts["pos"][0], dtype=float))
    assert np.allclose(electrons_udata["data"][0], np.asarray(expected_electrons["data"][0], dtype=float))
    assert np.allclose(electrons_udata["pixPos"][0], 1e6 * np.asarray(expected_electrons["pixPos"][0], dtype=float))
    assert dv_udata["ori"] == "v"
    assert dv_udata["dataType"] == "dv"
    assert np.allclose(dv_udata["data"][0], np.asarray(expected_dv["data"][0], dtype=float))


def test_ip_plot_line_chromaticity_and_luminance(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0], [3.0, 30.0, 4.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0], [6.0, 60.0, 7.0]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz

    chroma_roi = np.array([1, 1, 1, 1], dtype=int)
    horizontal_xy = np.array([1, 2], dtype=int)
    vertical_xy = np.array([2, 1], dtype=int)

    line_udata, line_handle = ipPlot(ip, "horizontal line", horizontal_xy)
    chroma_udata, chroma_handle = ipPlot(ip, "chromaticity", chroma_roi)
    luminance_udata, luminance_handle = ipPlot(ip, "vertical line luminance", vertical_xy)

    expected_chroma = np.asarray(ip_get(ip, "chromaticity", chroma_roi), dtype=float)
    expected_luminance = np.asarray(ip_get(ip, "data luminance"), dtype=float)[:, 1]

    assert line_handle is None
    assert chroma_handle is None
    assert luminance_handle is None
    assert np.array_equal(line_udata["xy"], horizontal_xy)
    assert line_udata["ori"] == "h"
    assert np.allclose(line_udata["pos"], np.array([1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(line_udata["values"], result[1, :, :])
    assert np.allclose(chroma_udata["x"], expected_chroma[:, 0])
    assert np.allclose(chroma_udata["y"], expected_chroma[:, 1])
    assert np.array_equal(chroma_udata["roiLocs"], chroma_roi)
    assert np.array_equal(luminance_udata["xy"], vertical_xy)
    assert luminance_udata["ori"] == "v"
    assert np.allclose(luminance_udata["pos"], np.array([1.0, 2.0], dtype=float))
    assert np.allclose(luminance_udata["data"], expected_luminance)


def test_plot_sensor_histogram_data(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(
        sensor,
        "dv",
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=float,
        ),
    )
    roi = np.array([1, 1, 1, 1], dtype=int)

    volts_udata, volts_handle = plotSensor(sensor, "volts histogram", roi)
    electrons_udata, electrons_handle = plotSensor(sensor, "electrons hist", roi)
    dv_udata, dv_handle = plotSensor(sensor, "dv hist", roi)

    expected_volts = np.asarray(vc_get_roi_data(sensor, roi, "volts"), dtype=float)
    expected_electrons = np.asarray(vc_get_roi_data(sensor, roi, "electrons"), dtype=float)
    expected_dv = np.asarray(vc_get_roi_data(sensor, roi, "dv"), dtype=float)

    assert volts_handle is None
    assert electrons_handle is None
    assert dv_handle is None
    assert np.array_equal(volts_udata["rect"], roi)
    assert volts_udata["unitType"] == "volts"
    assert np.allclose(volts_udata["data"], expected_volts)
    assert np.allclose(electrons_udata["data"], expected_electrons)
    assert np.allclose(dv_udata["data"], expected_dv)


def test_ip_plot_rgb_histogram_rgb3d_and_luminance(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0], [3.0, 30.0, 4.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0], [6.0, 60.0, 7.0]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz
    roi = np.array([1, 1, 1, 1], dtype=int)

    rgb_hist_udata, rgb_hist_handle = ipPlot(ip, "rgbhistogram", roi)
    rgb3d_udata, rgb3d_handle = ipPlot(ip, "rgb3d", roi)
    luminance_udata, luminance_handle = ipPlot(ip, "luminance", roi)

    expected_rgb = np.asarray(ip_get(ip, "roidata", roi), dtype=float)
    expected_xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    expected_luminance = expected_xyz[:, 1]

    assert rgb_hist_handle is None
    assert rgb3d_handle is None
    assert luminance_handle is None
    assert np.array_equal(rgb_hist_udata["rect"], roi)
    assert np.allclose(rgb_hist_udata["RGB"], expected_rgb)
    assert np.allclose(rgb_hist_udata["meanRGB"], np.mean(expected_rgb, axis=0))
    assert np.allclose(rgb3d_udata["RGB"], expected_rgb)
    assert np.allclose(luminance_udata["luminance"], expected_luminance)
    assert np.isclose(luminance_udata["meanL"], float(np.mean(expected_luminance)))
    assert np.isclose(luminance_udata["stdLum"], float(np.std(expected_luminance)))
