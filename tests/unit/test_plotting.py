from __future__ import annotations

import numpy as np

from pyisetcam import (
    ipPlot,
    ip_create,
    ip_get,
    ip_set,
    pixel_snr,
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
    sensor_snr,
    sensor_set,
    vc_get_roi_data,
    xyz_to_lab,
    xyz_to_luv,
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


def test_ip_plot_cielab_and_cieluv(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0]],
        ],
        dtype=float,
    )
    white_point = np.array([95.047, 100.0, 108.883], dtype=float)
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz
    ip = ip_set(ip, "data white point", white_point)
    roi = np.array([1, 1, 1, 1], dtype=int)

    lab_udata, lab_handle = ipPlot(ip, "cielab", roi)
    luv_udata, luv_handle = ipPlot(ip, "cieluv", roi)

    expected_xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    expected_lab = np.asarray(xyz_to_lab(expected_xyz, white_point), dtype=float)
    expected_luv = np.asarray(xyz_to_luv(expected_xyz, white_point), dtype=float)

    assert lab_handle is None
    assert luv_handle is None
    assert np.array_equal(lab_udata["rect"], roi)
    assert np.allclose(lab_udata["whitePoint"], white_point)
    assert np.allclose(lab_udata["LAB"], expected_lab)
    assert np.allclose(lab_udata["meanLAB"], np.mean(expected_lab, axis=0))
    assert np.allclose(luv_udata["whitePoint"], white_point)
    assert np.allclose(luv_udata["LUV"], expected_luv)
    assert np.allclose(luv_udata["meanLUV"], np.mean(expected_luv, axis=0))


def test_pixel_and_sensor_snr_helpers_and_plot_wrappers(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 2.0e-4
    sensor.fields["pixel"]["voltage_swing"] = 1.2
    sensor.fields["pixel"]["read_noise_v"] = 4.0e-4
    sensor.fields["pixel"]["dsnu_sigma_v"] = 2.0e-4
    sensor.fields["pixel"]["prnu_sigma"] = 0.01
    volts = np.array([1.0e-3, 1.0e-2], dtype=float)

    pixel_curve, pixel_volts, pixel_shot, pixel_read = pixel_snr(sensor, volts)
    sensor_curve, sensor_volts, sensor_shot, sensor_read, sensor_dsnu, sensor_prnu = sensor_snr(sensor, volts)

    conv_gain = 2.0e-4
    read_sd = 4.0e-4 / conv_gain
    dsnu_sd = 2.0e-4 / conv_gain
    prnu_gain_sd = 0.01
    shot_sd = np.sqrt(volts / conv_gain)
    signal_power = (volts / conv_gain) ** 2
    expected_pixel = 10.0 * np.log10(signal_power / (read_sd**2 + shot_sd**2))
    expected_sensor = 10.0 * np.log10(signal_power / (shot_sd**2 + read_sd**2 + dsnu_sd**2 + (prnu_gain_sd * (volts / conv_gain)) ** 2))

    pixel_udata, pixel_handle = plotSensor(sensor, "pixel snr")
    sensor_udata, sensor_handle = plotSensor(sensor, "sensor snr")

    assert pixel_handle is None
    assert sensor_handle is None
    assert np.allclose(pixel_volts, volts)
    assert np.allclose(sensor_volts, volts)
    assert np.allclose(pixel_curve, expected_pixel)
    assert np.allclose(sensor_curve, expected_sensor)
    assert np.allclose(pixel_shot, 10.0 * np.log10(signal_power / (shot_sd**2)))
    assert np.allclose(sensor_shot, 10.0 * np.log10(signal_power / (shot_sd**2)))
    assert np.allclose(pixel_read, 10.0 * np.log10(signal_power / (read_sd**2)))
    assert np.allclose(sensor_read, 10.0 * np.log10(signal_power / (read_sd**2)))
    assert np.allclose(sensor_dsnu, 10.0 * np.log10(signal_power / (dsnu_sd**2)))
    assert np.allclose(sensor_prnu, 10.0 * np.log10(signal_power / ((prnu_gain_sd * (volts / conv_gain)) ** 2)))
    assert np.allclose(pixel_udata["snr"], pixel_snr(sensor)[0])
    assert np.allclose(pixel_udata["volts"], pixel_snr(sensor)[1])
    assert np.allclose(sensor_udata["snr"], sensor_snr(sensor)[0])
    assert np.allclose(sensor_udata["volts"], sensor_snr(sensor)[1])


def test_plot_sensor_spectral_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    color_udata, color_handle = plotSensor(sensor, "color filters")
    qe_udata, qe_handle = plotSensor(sensor, "sensor spectral qe")

    expected_wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    expected_filters = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
    expected_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    expected_names = list(sensor_get(sensor, "filter color letters cell"))

    assert color_handle is None
    assert qe_handle is None
    assert np.allclose(color_udata["x"], expected_wave)
    assert np.allclose(color_udata["y"], expected_filters)
    assert color_udata["filterNames"] == expected_names
    assert color_udata["yLabel"] == "Transmittance"
    assert np.allclose(qe_udata["x"], expected_wave)
    assert np.allclose(qe_udata["y"], expected_qe)
    assert qe_udata["filterNames"] == expected_names
    assert qe_udata["yLabel"] == "Quantum efficiency"


def test_plot_sensor_cfa_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)

    block_udata, block_handle = plotSensor(sensor, "cfa")
    full_udata, full_handle = plotSensor(sensor, "cfafull")

    expected_unit_pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    expected_pattern_colors = np.asarray(sensor_get(sensor, "pattern colors"))

    assert block_handle is None
    assert full_handle is None
    assert block_udata["mode"] == "block"
    assert full_udata["mode"] == "full"
    assert block_udata["scale"] == 8
    assert full_udata["scale"] == 8
    assert np.array_equal(block_udata["unitPattern"], expected_unit_pattern)
    assert np.array_equal(block_udata["pattern"], expected_unit_pattern)
    assert np.array_equal(block_udata["patternColors"], expected_pattern_colors)
    assert np.array_equal(block_udata["unitPatternColors"], expected_pattern_colors)
    assert block_udata["filterNames"] == ["r", "g", "b"]
    assert block_udata["imgSmall"].shape == (2, 2, 3)
    assert block_udata["img"].shape == (16, 16, 3)
    assert np.allclose(block_udata["imgSmall"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][1, 1], np.array([0.0, 1.0, 0.0], dtype=float))

    assert np.array_equal(full_udata["unitPattern"], expected_unit_pattern)
    assert full_udata["imgSmall"].shape == (4, 4, 3)
    assert full_udata["img"].shape == (32, 32, 3)
    assert np.array_equal(full_udata["patternColors"][:2, :2], expected_pattern_colors)
    assert np.array_equal(full_udata["patternColors"][2:, 2:], expected_pattern_colors)
    assert np.allclose(full_udata["imgSmall"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(full_udata["imgSmall"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(full_udata["imgSmall"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))


def test_plot_sensor_true_size_and_cfa_image_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
    )

    true_udata, true_handle = plotSensor(sensor, "true size")
    cfa_image_udata, cfa_image_handle = plotSensor(sensor, "cfa image")

    assert true_handle is None
    assert cfa_image_handle is None
    assert true_udata["dataType"] == "volts"
    assert cfa_image_udata["dataType"] == "volts"
    assert true_udata["img"].shape == (2, 2, 3)
    assert cfa_image_udata["img"].shape == (2, 2, 3)
    assert np.allclose(true_udata["img"][0, 0], np.array([0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(true_udata["img"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(true_udata["img"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(true_udata["img"][1, 1], np.array([0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][1, 1], np.array([0.0, 1.0, 0.0], dtype=float))


def test_plot_sensor_channels_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 16.0
    sensor = sensor_set(sensor, "volts", volts)

    channels_udata, channels_handle = plotSensor(sensor, "channels")

    assert channels_handle is None
    assert channels_udata["dataType"] == "volts"
    assert channels_udata["filterNames"] == ["r", "g", "b"]
    assert len(channels_udata["channelData"]) == 3
    assert len(channels_udata["channelImages"]) == 3
    assert len(channels_udata["masks"]) == 3
    assert channels_udata["pattern"].shape == (4, 4)
    red_mask = channels_udata["masks"][0]
    green_mask = channels_udata["masks"][1]
    blue_mask = channels_udata["masks"][2]
    assert np.array_equal(red_mask, channels_udata["pattern"] == 1)
    assert np.array_equal(green_mask, channels_udata["pattern"] == 2)
    assert np.array_equal(blue_mask, channels_udata["pattern"] == 3)
    assert np.isnan(channels_udata["channelData"][0][0, 0])
    assert np.isclose(channels_udata["channelData"][0][0, 1], volts[0, 1])
    assert np.isclose(channels_udata["channelData"][1][0, 0], volts[0, 0])
    assert np.isclose(channels_udata["channelData"][2][1, 0], volts[1, 0])
    assert np.allclose(channels_udata["channelImages"][0][0, 1, 1:], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][0][0, 1, 0] > 0.0
    assert np.allclose(channels_udata["channelImages"][1][0, 0, [0, 2]], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][1][0, 0, 1] > 0.0
    assert np.allclose(channels_udata["channelImages"][2][1, 0, :2], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][2][1, 0, 2] > 0.0
