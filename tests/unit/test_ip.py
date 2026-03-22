from __future__ import annotations

import imageio.v3 as iio
import numpy as np

from pyisetcam import (
    demosaic,
    displayRender,
    ieInternal2Display,
    imageDataXYZ,
    imageEsserTransform,
    imageIlluminantCorrection,
    imageMCCTransform,
    imageRGB2XYZ,
    imageSensorConversion,
    imageSensorCorrection,
    imageSensorTransform,
    ie_reflectance_samples,
    ipHDRWhite,
    ip_compute,
    ip_create,
    ip_get,
    ip_set,
    ipClearData,
    ipSaveImage,
    sensor_create,
    sensor_get,
    sensor_set,
)
from pyisetcam.color import sensor_to_target_matrix, xyz_color_matching
from pyisetcam.utils import energy_to_quanta


def test_ip_compute_supports_rgb_bayer_demosaic_methods(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    mosaic = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "volts", mosaic)

    expected = {
        "nearest neighbor": np.array(
            [
                [
                    [2.0, 1.0, 7.0],
                    [2.0, 1.0, 7.0],
                    [4.0, 3.0, 9.0],
                    [4.0, 3.0, 9.0],
                    [6.0, 5.0, 11.0],
                    [6.0, 5.0, 11.0],
                ],
                [
                    [2.0, 8.0, 7.0],
                    [2.0, 8.0, 7.0],
                    [4.0, 10.0, 9.0],
                    [4.0, 10.0, 9.0],
                    [6.0, 12.0, 11.0],
                    [6.0, 12.0, 11.0],
                ],
                [
                    [14.0, 13.0, 19.0],
                    [14.0, 13.0, 19.0],
                    [16.0, 15.0, 21.0],
                    [16.0, 15.0, 21.0],
                    [18.0, 17.0, 23.0],
                    [18.0, 17.0, 23.0],
                ],
                [
                    [14.0, 20.0, 19.0],
                    [14.0, 20.0, 19.0],
                    [16.0, 22.0, 21.0],
                    [16.0, 22.0, 21.0],
                    [18.0, 24.0, 23.0],
                    [18.0, 24.0, 23.0],
                ],
                [
                    [26.0, 25.0, 31.0],
                    [26.0, 25.0, 31.0],
                    [28.0, 27.0, 33.0],
                    [28.0, 27.0, 33.0],
                    [30.0, 29.0, 35.0],
                    [30.0, 29.0, 35.0],
                ],
                [
                    [26.0, 32.0, 31.0],
                    [26.0, 32.0, 31.0],
                    [28.0, 34.0, 33.0],
                    [28.0, 34.0, 33.0],
                    [30.0, 36.0, 35.0],
                    [30.0, 36.0, 35.0],
                ],
            ],
            dtype=float,
        ),
        "laplacian": np.array(
            [
                [
                    [0.625, 1.0, 7.5],
                    [2.0, 4.5, 9.875],
                    [1.625, 3.0, 9.75],
                    [4.0, 7.0, 12.125],
                    [3.625, 5.0, 11.5],
                    [6.0, 8.5, 12.875],
                ],
                [
                    [6.125, 4.5, 7.0],
                    [7.5, 8.0, 9.375],
                    [6.875, 6.0, 9.0],
                    [9.25, 10.0, 11.375],
                    [9.125, 8.5, 11.0],
                    [11.5, 12.0, 12.375],
                ],
                [
                    [14.125, 13.0, 13.5],
                    [14.0, 13.5, 14.375],
                    [15.125, 15.0, 15.75],
                    [16.0, 16.0, 16.625],
                    [17.125, 17.0, 17.5],
                    [18.0, 17.5, 17.375],
                ],
                [
                    [19.625, 19.5, 19.0],
                    [19.5, 20.0, 19.875],
                    [20.375, 21.0, 21.0],
                    [21.25, 22.0, 21.875],
                    [22.625, 23.5, 23.0],
                    [23.5, 24.0, 22.875],
                ],
                [
                    [24.625, 25.0, 25.5],
                    [26.0, 28.5, 27.875],
                    [25.625, 27.0, 27.75],
                    [28.0, 31.0, 30.125],
                    [27.625, 29.0, 29.5],
                    [30.0, 32.5, 30.875],
                ],
                [
                    [24.125, 28.5, 31.0],
                    [25.5, 32.0, 33.375],
                    [24.875, 30.0, 33.0],
                    [27.25, 34.0, 35.375],
                    [27.125, 32.5, 35.0],
                    [29.5, 36.0, 36.375],
                ],
            ],
            dtype=float,
        ),
        "adaptive laplacian": np.array(
            [
                [
                    [2.25, 1.0, 6.5],
                    [2.0, 1.0, 7.25],
                    [3.25, 3.0, 9.0],
                    [4.0, 4.0, 9.75],
                    [5.25, 5.0, 10.5],
                    [6.0, 5.0, 10.25],
                ],
                [
                    [8.75, 8.0, 7.0],
                    [8.5, 8.0, 7.75],
                    [9.25, 9.0, 9.0],
                    [10.0, 10.0, 9.75],
                    [11.75, 12.0, 11.0],
                    [12.5, 12.0, 10.75],
                ],
                [
                    [14.25, 13.0, 12.5],
                    [14.0, 13.0, 13.25],
                    [15.25, 15.0, 15.0],
                    [16.0, 16.0, 15.75],
                    [17.25, 17.0, 16.5],
                    [18.0, 17.0, 16.25],
                ],
                [
                    [20.75, 20.0, 19.0],
                    [20.5, 20.0, 19.75],
                    [21.25, 21.0, 21.0],
                    [22.0, 22.0, 21.75],
                    [23.75, 24.0, 23.0],
                    [24.5, 24.0, 22.75],
                ],
                [
                    [26.25, 25.0, 24.5],
                    [26.0, 25.0, 25.25],
                    [27.25, 27.0, 27.0],
                    [28.0, 28.0, 27.75],
                    [29.25, 29.0, 28.5],
                    [30.0, 29.0, 28.25],
                ],
                [
                    [26.75, 32.0, 31.0],
                    [26.5, 32.0, 31.75],
                    [27.25, 33.0, 33.0],
                    [28.0, 34.0, 33.75],
                    [29.75, 36.0, 35.0],
                    [30.5, 36.0, 34.75],
                ],
            ],
            dtype=float,
        ),
    }

    for method, expected_sensor_space in expected.items():
        ip = ip_create(asset_store=asset_store)
        ip = ip_set(ip, "demosaic method", method)
        computed = ip_compute(ip, sensor, asset_store=asset_store)

        assert np.allclose(
            np.asarray(ip_get(computed, "sensorspace"), dtype=float), expected_sensor_space
        )


def test_ip_compute_adaptive_laplacian_falls_back_to_bilinear_for_gbrg(asset_store) -> None:
    sensor = sensor_create("bayer-gbrg", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))

    bilinear = ip_compute(
        ip_set(ip_create(asset_store=asset_store), "demosaic method", "bilinear"),
        sensor,
        asset_store=asset_store,
    )
    adaptive = ip_compute(
        ip_set(ip_create(asset_store=asset_store), "demosaic method", "adaptive laplacian"),
        sensor,
        asset_store=asset_store,
    )

    assert np.allclose(
        np.asarray(ip_get(adaptive, "sensorspace"), dtype=float),
        np.asarray(ip_get(bilinear, "sensorspace"), dtype=float),
    )


def test_demosaic_matches_ip_compute_sensor_space(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))

    for method in ("bilinear", "nearest neighbor", "laplacian", "adaptive laplacian"):
        ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", method)
        computed = ip_compute(ip, sensor, asset_store=asset_store)

        assert np.allclose(
            np.asarray(demosaic(ip, sensor), dtype=float),
            np.asarray(ip_get(computed, "sensorspace"), dtype=float),
        )


def test_demosaic_prefers_ip_input_over_sensor_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    override = np.arange(101, 137, dtype=float).reshape(6, 6)

    reference_sensor = sensor_create("default", asset_store=asset_store)
    reference_sensor = sensor_set(reference_sensor, "volts", override)
    reference_ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", "nearest neighbor")
    reference = ip_compute(reference_ip, reference_sensor, asset_store=asset_store)

    ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", "nearest neighbor")
    ip.data["input"] = override

    assert np.allclose(
        np.asarray(demosaic(ip, sensor), dtype=float),
        np.asarray(ip_get(reference, "sensorspace"), dtype=float),
    )


def test_demosaic_returns_monochrome_sensor_plane(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    volts = np.arange(1, 13, dtype=float).reshape(3, 4)
    sensor = sensor_set(sensor, "volts", volts)
    ip = ip_create(asset_store=asset_store)

    result = np.asarray(demosaic(ip, sensor), dtype=float)

    assert result.shape == (3, 4, 1)
    assert np.allclose(result[:, :, 0], volts)


def test_image_sensor_conversion_matches_direct_formula(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)

    cmf = np.column_stack(
        (
            np.linspace(0.2, 0.7, wave.size),
            np.linspace(0.3, 0.9, wave.size),
            np.linspace(0.8, 0.1, wave.size),
        )
    )
    surfaces = np.column_stack(
        (
            np.linspace(0.1, 0.9, wave.size),
            np.linspace(0.9, 0.2, wave.size),
            np.linspace(0.3, 0.6, wave.size),
        )
    )
    illuminant = np.linspace(1.0, 2.0, wave.size)

    transform, actual, desired, white_cmf = imageSensorConversion(
        sensor,
        cmf,
        surfaces,
        illuminant,
        asset_store=asset_store,
    )

    weighted_surfaces = illuminant.reshape(-1, 1) * surfaces
    expected_actual = spectral_qe.T @ weighted_surfaces
    expected_desired = cmf.T @ weighted_surfaces
    expected_transform = expected_desired @ np.linalg.pinv(expected_actual)
    expected_white_cmf = cmf.T @ illuminant

    assert np.allclose(actual, expected_actual)
    assert np.allclose(desired, expected_desired)
    assert np.allclose(transform, expected_transform)
    assert np.allclose(white_cmf, expected_white_cmf)


def test_image_sensor_correction_matches_ip_compute_sensor_transform(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")

    sensor_space = demosaic(ip, sensor)
    corrected, corrected_ip, sensor_transform = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(corrected, np.asarray(ip_get(computed, "xyz"), dtype=float))
    assert np.allclose(
        sensor_transform,
        np.asarray(ip_get(computed, "conversion transform sensor"), dtype=float),
    )
    assert np.allclose(np.asarray(ip_get(corrected_ip, "xyz"), dtype=float), corrected)


def test_image_sensor_correction_preserves_2d_monochrome_input(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    img = np.arange(1, 13, dtype=float).reshape(3, 4)
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "none")

    corrected, corrected_ip, sensor_transform = imageSensorCorrection(
        img,
        ip,
        sensor,
        asset_store=asset_store,
    )

    assert corrected.shape == img.shape
    assert np.allclose(corrected, img / np.max(img))
    assert sensor_transform.shape == (1, 1)
    assert np.allclose(np.asarray(ip_get(corrected_ip, "sensorspace"), dtype=float)[:, :, 0], img)


def test_image_illuminant_correction_matches_ip_compute_ics(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")
    ip = ip_set(ip, "correction method illuminant", "gray world")

    sensor_space = demosaic(ip, sensor)
    internal, corrected_ip, _ = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    corrected, illuminant_ip, illuminant_transform = imageIlluminantCorrection(
        internal,
        corrected_ip,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(corrected, np.asarray(ip_get(computed, "ics"), dtype=float))
    assert np.allclose(
        illuminant_transform,
        np.asarray(ip_get(computed, "illuminant correction transform"), dtype=float),
    )
    assert np.allclose(np.asarray(ip_get(illuminant_ip, "ics"), dtype=float), corrected)


def test_image_illuminant_correction_manual_matrix(asset_store) -> None:
    img = np.array(
        [
            [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
            [[0.8, 0.1, 0.2], [0.9, 0.3, 0.4]],
        ],
        dtype=float,
    )
    transform = np.diag([1.2, 0.8, 1.1])
    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "manual")
    ip = ip_set(ip, "illuminant correction transform", transform)

    corrected, corrected_ip, returned_transform = imageIlluminantCorrection(
        img,
        ip,
        asset_store=asset_store,
    )

    assert np.allclose(corrected, img @ transform)
    assert np.allclose(returned_transform, transform)
    assert np.allclose(
        np.asarray(ip_get(corrected_ip, "illuminant correction transform"), dtype=float),
        transform,
    )


def test_image_illuminant_correction_preserves_2d_input(asset_store) -> None:
    img = np.arange(1, 13, dtype=float).reshape(3, 4)
    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "none")

    corrected, corrected_ip, illuminant_transform = imageIlluminantCorrection(
        img,
        ip,
        asset_store=asset_store,
    )

    assert corrected.shape == img.shape
    assert np.allclose(corrected, img)
    assert illuminant_transform.shape == (1, 1)
    assert np.allclose(np.asarray(ip_get(corrected_ip, "ics"), dtype=float)[:, :, 0], img)


def test_image_rgb_to_xyz_preserves_rgb_format(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb = np.array(
        [
            [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]],
            [[0.7, 0.2, 0.1], [0.9, 0.8, 0.2]],
        ],
        dtype=float,
    )

    xyz = np.asarray(imageRGB2XYZ(ip, rgb, asset_store=asset_store), dtype=float)

    assert xyz.shape == rgb.shape
    manual = imageRGB2XYZ(ip, rgb.reshape(-1, 3), asset_store=asset_store).reshape(rgb.shape)
    assert np.allclose(xyz, manual)


def test_image_rgb_to_xyz_preserves_xw_format(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb_xw = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5],
            [0.7, 0.2, 0.1],
            [0.9, 0.8, 0.2],
        ],
        dtype=float,
    )

    xyz_xw = np.asarray(imageRGB2XYZ(ip, rgb_xw, asset_store=asset_store), dtype=float)

    assert xyz_xw.shape == rgb_xw.shape
    assert np.allclose(xyz_xw, imageRGB2XYZ(ip, rgb_xw.copy(), asset_store=asset_store))


def test_image_data_xyz_uses_image_rgb_to_xyz_for_result(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb = np.array(
        [
            [[0.25, 0.50, 0.75], [0.10, 0.20, 0.30]],
            [[0.60, 0.40, 0.20], [0.90, 0.70, 0.50]],
        ],
        dtype=float,
    )
    ip.data["result"] = rgb

    xyz_from_result = np.asarray(imageDataXYZ(ip, asset_store=asset_store), dtype=float)
    xyz_direct = np.asarray(imageRGB2XYZ(ip, rgb, asset_store=asset_store), dtype=float)

    assert np.allclose(xyz_from_result, xyz_direct)


def test_display_render_matches_ip_compute_result(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")
    ip = ip_set(ip, "correction method illuminant", "gray world")

    sensor_space = demosaic(ip, sensor)
    internal, corrected_ip, _ = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    ics, corrected_ip, _ = imageIlluminantCorrection(
        internal,
        corrected_ip,
        asset_store=asset_store,
    )
    display_linear, rendered_ip, display_transform = displayRender(
        ics,
        corrected_ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(display_linear, np.asarray(ip_get(computed, "result"), dtype=float))
    assert np.allclose(display_transform, np.asarray(ip_get(computed, "ics2display"), dtype=float))
    assert np.allclose(np.asarray(ip_get(rendered_ip, "result"), dtype=float), display_linear)


def test_display_render_matches_sensor_internal_space_path(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "internal cs", "sensor")
    ip = ip_set(ip, "conversion method sensor", "none")
    ip = ip_set(ip, "correction method illuminant", "none")

    sensor_space = demosaic(ip, sensor)
    display_linear, rendered_ip, display_transform = displayRender(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(display_linear, np.asarray(ip_get(computed, "result"), dtype=float))
    assert np.allclose(display_transform, np.asarray(ip_get(computed, "ics2display"), dtype=float))
    assert np.allclose(np.asarray(ip_get(rendered_ip, "result"), dtype=float), display_linear)


def test_ip_clear_data_clears_computed_payloads(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    cleared = ipClearData(computed)

    assert cleared is not computed
    assert ip_get(cleared, "result") is None
    assert ip_get(cleared, "input") is None
    assert ip_get(cleared, "xyz") is None
    assert ip_get(cleared, "ics") is None
    transforms = ip_get(cleared, "transforms")
    assert len(transforms) == 3
    assert transforms == [None, None, None]
    assert ip_get(cleared, "internal cs") == ip_get(computed, "internal cs")


def test_ip_set_data_empty_matches_ip_clear_data(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    cleared_by_set = ip_set(computed.clone(), "data", [])
    cleared_by_wrapper = ipClearData(computed)

    assert ip_get(cleared_by_set, "result") is None
    assert ip_get(cleared_by_set, "input") is None
    assert ip_get(cleared_by_set, "transforms") == [None, None, None]
    assert ip_get(cleared_by_wrapper, "transforms") == [None, None, None]


def test_ip_save_image_writes_png_and_appends_extension(tmp_path, asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    output = ipSaveImage(computed, tmp_path / "ip-save-image")

    saved_path = tmp_path / "ip-save-image.png"
    assert output == str(saved_path.resolve())
    saved = iio.imread(saved_path)
    srgb = np.asarray(ip_get(computed, "srgb"), dtype=float)
    assert saved.shape == srgb.shape
    assert saved.dtype == np.uint8


def test_ip_save_image_cropborder_crops_black_frame(tmp_path, asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    srgb = np.zeros((6, 7, 3), dtype=float)
    srgb[1:5, 2:6, :] = np.array([0.25, 0.5, 0.75], dtype=float)
    ip.data["srgb"] = srgb

    output = ipSaveImage(ip, tmp_path / "cropped.png", cropborder=True)

    saved = iio.imread(output)
    assert saved.shape == (4, 4, 3)
    assert np.all(saved[0, 0] == np.array([64, 128, 191], dtype=np.uint8))


def test_image_mcc_transform_matches_sensor_to_target_matrix(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )

    transform = imageMCCTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)
    expected = sensor_to_target_matrix(
        wave,
        sensor_qe,
        target_space="xyz",
        illuminant="D65",
        surfaces="mcc",
        asset_store=asset_store,
    )

    assert np.allclose(transform, expected)


def test_image_mcc_transform_accepts_illuminant_vector(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )
    illuminant_energy = np.asarray(
        asset_store.load_illuminant("D65", wave_nm=wave)[1],
        dtype=float,
    )

    from_string = imageMCCTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)
    from_vector = imageMCCTransform(
        sensor_qe,
        target_qe,
        illuminant_energy,
        wave,
        asset_store=asset_store,
    )

    assert np.allclose(from_vector, from_string)


def test_image_sensor_transform_supports_multisurface_default(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )
    illuminant_energy = np.asarray(
        asset_store.load_illuminant("D65", wave_nm=wave)[1],
        dtype=float,
    )
    illuminant_quanta = np.asarray(energy_to_quanta(illuminant_energy, wave), dtype=float)
    reflectances = np.asarray(
        ie_reflectance_samples(None, None, wave, asset_store=asset_store)[0],
        dtype=float,
    )
    weighted_surfaces = reflectances * illuminant_quanta.reshape(-1, 1)
    sensor_response = weighted_surfaces.T @ sensor_qe
    target_response = weighted_surfaces.T @ target_qe
    expected, _, _, _ = np.linalg.lstsq(sensor_response, target_response, rcond=None)

    transform = imageSensorTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)

    assert np.allclose(transform, expected)


def test_image_esser_transform_matches_direct_sensor_transform(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )

    expected = sensor_to_target_matrix(
        wave,
        sensor_qe,
        target_space="xyz",
        illuminant="D65",
        surfaces="esser",
        asset_store=asset_store,
    )
    direct = imageSensorTransform(
        sensor_qe,
        target_qe,
        "D65",
        wave,
        "esser",
        asset_store=asset_store,
    )
    esser = imageEsserTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)

    assert np.allclose(direct, expected)
    assert np.allclose(esser, expected)


def test_ie_internal_to_display_matches_direct_display_matrix(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)

    transform = ieInternal2Display(ip)
    internal_cmf = np.asarray(ip_get(ip, "internal cmf"), dtype=float)
    display_spd = np.asarray(ip_get(ip, "display rgb spd"), dtype=float)
    expected = np.linalg.inv(display_spd.T @ internal_cmf)

    assert np.allclose(transform, expected)


def test_ip_hdr_white_whitens_saturated_pixels_without_blur(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip.data["input"] = np.full((3, 3), 10.0, dtype=float)
    ip.data["result"] = np.full((3, 3, 3), 0.25, dtype=float)

    whitened_ip, weights = ipHDRWhite(
        ip,
        "saturation",
        10.0,
        "hdr level",
        0.5,
        "wgt blur",
        0.0,
    )

    assert np.allclose(weights, np.ones((3, 3), dtype=float))
    assert np.allclose(np.asarray(ip_get(whitened_ip, "result"), dtype=float), 1.0)


def test_ip_hdr_white_uses_input_max_when_saturation_omitted(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip.data["input"] = np.array(
        [
            [0.0, 2.5, 5.0],
            [1.0, 5.0, 2.5],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    ip.data["result"] = np.zeros((3, 3, 3), dtype=float)

    whitened_ip, weights = ipHDRWhite(ip, "hdr level", 0.5, "wgt blur", 0.0)

    expected_weights = np.clip(ip.data["input"] / 5.0 - 0.5, 0.0, 1.0) / 0.5
    assert np.allclose(weights, expected_weights)
    assert np.allclose(np.asarray(ip_get(whitened_ip, "result"), dtype=float), expected_weights[:, :, np.newaxis])
