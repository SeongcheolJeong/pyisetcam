from __future__ import annotations

import numpy as np

from pyisetcam import (
    DEFAULT_WAVE,
    blackbody,
    comparison_metrics,
    correlated_color_temperature,
    delta_e_ab,
    metrics_spd,
    mired_difference,
    peak_signal_to_noise_ratio,
    xyz_from_energy,
    xyz_to_lab,
    xyz_to_luv,
)


def test_xyz_to_lab_maps_white_point_to_neutral() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    lab = xyz_to_lab(white, white)

    assert np.allclose(lab, np.array([100.0, 0.0, 0.0]), atol=1e-6)


def test_xyz_to_luv_maps_white_point_to_neutral() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    luv = xyz_to_luv(white, white)

    assert np.allclose(luv, np.array([100.0, 0.0, 0.0]), atol=1e-6)


def test_delta_e_ab_is_zero_for_identical_xyz() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    xyz = np.array([20.0, 30.0, 15.0], dtype=float)

    assert np.isclose(delta_e_ab(xyz, xyz, white), 0.0)


def test_metrics_spd_angle_matches_direct_vector_angle() -> None:
    wave = np.array([500.0, 510.0, 520.0], dtype=float)
    spd1 = np.array([1.0, 0.0, 0.0], dtype=float)
    spd2 = np.array([0.0, 1.0, 0.0], dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="angle", wave=wave, return_params=True)

    assert np.isclose(value, 90.0)
    assert params == {}


def test_metrics_spd_cielab_matches_direct_xyz_and_lab_path() -> None:
    spd1 = np.asarray(blackbody(DEFAULT_WAVE, 6500.0, kind="energy"), dtype=float)
    spd2 = np.asarray(blackbody(DEFAULT_WAVE, 5000.0, kind="energy"), dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="cielab", return_params=True)

    xyz1 = xyz_from_energy(spd1 * (100.0 / xyz_from_energy(spd1, DEFAULT_WAVE)[1]), DEFAULT_WAVE)
    xyz2 = xyz_from_energy(spd2 * (100.0 / xyz_from_energy(spd2, DEFAULT_WAVE)[1]), DEFAULT_WAVE)
    white = xyz1 * (100.0 / xyz1[1])
    expected = np.linalg.norm(xyz_to_lab(xyz1, white) - xyz_to_lab(xyz2, white))

    assert np.isclose(value, expected, rtol=1e-8, atol=1e-10)
    assert params["lab1"].shape == (3,)
    assert params["lab2"].shape == (3,)


def test_metrics_spd_mired_returns_estimated_ccts() -> None:
    spd1 = np.asarray(blackbody(DEFAULT_WAVE, 6500.0, kind="energy"), dtype=float)
    spd2 = np.asarray(blackbody(DEFAULT_WAVE, 5000.0, kind="energy"), dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="mired", return_params=True)

    assert value > 0.0
    assert params["cct_k"].shape == (2,)
    assert np.isclose(value, mired_difference(params["cct_k"][0], params["cct_k"][1]))


def test_correlated_color_temperature_tracks_d65_white_point() -> None:
    d65_white = np.array([95.047, 100.0, 108.883], dtype=float)
    cct = correlated_color_temperature(d65_white)

    assert 6000.0 <= cct <= 7000.0


def test_comparison_metrics_report_standard_error_values() -> None:
    reference = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=float)
    actual = np.array([[0.6, 0.5], [0.9, 0.5]], dtype=float)

    metrics = comparison_metrics(reference, actual, data_range=1.0)

    assert np.isclose(metrics["mae"], 0.05)
    assert np.isclose(metrics["rmse"], np.sqrt(0.005))
    assert np.isclose(metrics["mean_rel"], 0.075)
    assert np.isclose(metrics["max_abs"], 0.1)
    assert np.isclose(metrics["psnr"], peak_signal_to_noise_ratio(reference, actual, data_range=1.0))
