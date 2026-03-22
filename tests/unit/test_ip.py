from __future__ import annotations

import numpy as np

from pyisetcam import demosaic, ip_compute, ip_create, ip_get, ip_set, sensor_create, sensor_set


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
