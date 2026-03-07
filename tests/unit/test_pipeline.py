from __future__ import annotations

import numpy as np

from pyisetcam import (
    camera_compute,
    camera_create,
    ip_compute,
    ip_create,
    oi_compute,
    oi_create,
    oi_get,
    oi_set,
    run_python_case,
    scene_create,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
)


def test_oi_compute_matches_scene_wave(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    assert oi.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.array_equal(oi.fields["wave"], scene.fields["wave"])


def test_oi_compute_tracks_output_geometry(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    rows, cols = oi.data["photons"].shape[:2]
    assert np.isclose(oi.fields["width_m"], cols * oi.fields["sample_spacing_m"])
    assert np.isclose(oi.fields["height_m"], rows * oi.fields["sample_spacing_m"])


def test_oi_compute_crop_matches_matlab_crop_geometry(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi_uncropped = oi_compute(oi_create(), scene, crop=False)
    oi_cropped = oi_compute(oi_create(), scene, crop=True)

    image_distance = float(oi_uncropped.fields["image_distance_m"])
    focal_length = float(oi_uncropped.fields["optics"]["focal_length_m"])
    expected_scale = image_distance / focal_length

    assert np.isclose(
        oi_cropped.fields["sample_spacing_m"],
        oi_uncropped.fields["sample_spacing_m"] * expected_scale,
    )
    assert oi_cropped.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_get_reports_matlab_style_geometry_vectors(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    rows, cols = oi.data["photons"].shape[:2]
    sample_spacing = oi_get(oi, "sample spacing")
    spatial_resolution = oi_get(oi, "distance per sample")
    angular_resolution = oi_get(oi, "angular resolution")

    assert oi_get(oi, "rows") == rows
    assert oi_get(oi, "cols") == cols
    assert oi_get(oi, "size") == (rows, cols)
    assert np.isclose(sample_spacing[0], oi_get(oi, "width") / cols)
    assert np.isclose(sample_spacing[1], oi_get(oi, "height") / rows)
    assert np.isclose(spatial_resolution[0], oi_get(oi, "height") / rows)
    assert np.isclose(spatial_resolution[1], oi_get(oi, "width") / cols)
    assert angular_resolution.shape == (2,)
    assert np.all(angular_resolution > 0.0)


def test_oi_set_updates_geometry_and_optics_accessors() -> None:
    oi = oi_create()
    oi = oi_set(oi, "photons", np.ones((4, 6, 3), dtype=float))
    oi = oi_set(oi, "focal length", 0.02)
    oi = oi_set(oi, "fov", 5.0)
    oi = oi_set(oi, "compute method", "opticspsf")
    oi = oi_set(oi, "diffuser method", "skip")
    oi = oi_set(oi, "off axis method", "cos4th")

    expected_width = 2.0 * oi_get(oi, "image distance") * np.tan(np.deg2rad(2.5))

    assert np.isclose(oi_get(oi, "width"), expected_width)
    assert np.isclose(oi_get(oi, "sample size"), expected_width / 6.0)
    assert oi_get(oi, "compute method") == "opticspsf"
    assert oi_get(oi, "diffuser method") == "skip"
    assert oi_get(oi, "off axis method") == "cos4th"


def test_oi_get_reports_spatial_and_frequency_support(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    spatial = oi_get(oi, "spatial support linear", "mm")
    mesh = oi_get(oi, "spatial support", "mm")
    angular = oi_get(oi, "angular support", "radians")
    freq = oi_get(oi, "frequency resolution", "mm")
    fsupport = oi_get(oi, "frequency support", "mm")

    rows, cols = oi.data["photons"].shape[:2]
    assert spatial["x"].shape == (cols,)
    assert spatial["y"].shape == (rows,)
    assert mesh.shape == (rows, cols, 2)
    assert angular.shape == (rows, cols, 2)
    assert freq["fx"].shape == (cols,)
    assert freq["fy"].shape == (rows,)
    assert fsupport.shape == (rows, cols, 2)
    assert np.isclose(oi_get(oi, "max frequency resolution", "mm"), max(freq["fx"].max(), freq["fy"].max()))


def test_oi_compute_skip_model_avoids_blur(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_create()
    oi.fields["optics"]["model"] = "skip"
    oi.fields["optics"]["offaxis_method"] = "skip"
    oi = oi_compute(oi, scene, crop=True)

    scene_photons = np.asarray(scene.data["photons"], dtype=float)
    oi_photons = np.asarray(oi.data["photons"], dtype=float)
    scale = oi_photons[0, 0, 0] / scene_photons[0, 0, 0]
    assert np.allclose(oi_photons, scene_photons * scale)


def test_oi_compute_border_padding_matches_corner_photons(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_create()
    oi.fields["optics"]["model"] = "skip"
    oi.fields["optics"]["offaxis_method"] = "skip"
    oi = oi_compute(oi, scene, pad_value="border", crop=False)

    pad_rows, pad_cols = oi.fields["padding_pixels"]
    corner = oi.data["photons"][pad_rows, pad_cols, :]

    assert np.allclose(oi.data["photons"][0, 0, :], corner)
    assert np.allclose(oi.data["photons"][0, -1, :], corner)
    assert np.allclose(oi.data["photons"][-1, 0, :], corner)
    assert np.allclose(oi.data["photons"][-1, -1, :], corner)


def test_wvf_path_preserves_more_checkerboard_contrast_than_diffraction(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi_wvf = oi_compute(oi_create("wvf"), scene, crop=True)
    oi_diffraction = oi_compute(oi_create(), scene, crop=True)

    row = oi_wvf.data["photons"].shape[0] // 2
    dark_col = 8
    band = 0
    dark_wvf = float(oi_wvf.data["photons"][row, dark_col, band])
    dark_diffraction = float(oi_diffraction.data["photons"][row, dark_col, band])

    assert dark_wvf < dark_diffraction


def test_oi_create_wvf_matches_upstream_default_wavefront_metadata() -> None:
    oi = oi_create("wvf")
    wavefront = oi.fields["optics"]["wavefront"]
    assert oi.fields["optics"]["compute_method"] == "opticspsf"
    assert oi.fields["optics"]["model"] == "shiftinvariant"
    assert np.isclose(wavefront["measured_pupil_diameter_mm"], 8.0)
    assert np.isclose(wavefront["measured_wavelength_nm"], 550.0)
    assert wavefront["sample_interval_domain"] == "psf"
    assert wavefront["spatial_samples"] == 201
    assert np.isclose(wavefront["ref_pupil_plane_size_mm"], 16.212)
    assert np.isclose(wavefront["calc_pupil_diameter_mm"], 9.6569e-01)
    assert np.isclose(wavefront["focal_length_m"], 0.003862755099228)
    assert np.isclose(wavefront["f_number"], 4.0)


def test_oi_compute_wvf_uses_custom_aperture(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)
    aperture = np.ones((9, 9), dtype=float)
    aperture[:, :4] = 0.0
    custom_oi = oi_compute(oi_create("wvf"), scene, crop=True, aperture=aperture)

    assert custom_oi.data["photons"].shape == default_oi.data["photons"].shape
    assert not np.allclose(custom_oi.data["photons"], default_oi.data["photons"])


def test_sensor_compute_noiseless(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0)
    sensor = sensor_compute(sensor, oi, seed=0)
    assert sensor.data["volts"].shape == sensor.fields["size"]
    assert np.all(sensor.data["volts"] >= 0.0)


def test_sensor_compute_noiseless_auto_exposure_matches_regression(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0)
    sensor = sensor_compute(sensor, oi, seed=0)
    assert np.isclose(sensor.fields["integration_time"], 0.05778050422668457, rtol=1e-5, atol=1e-8)


def test_sensor_set_integration_time_disables_auto_exposure(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "auto exposure", True)
    sensor = sensor_set(sensor, "integration time", 0.125)
    assert sensor.fields["auto_exposure"] is False
    assert np.isclose(sensor.fields["integration_time"], 0.125)


def test_sensor_get_fov_uses_scene_distance_when_provided(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    oi = oi_create()
    scene = scene_create(asset_store=asset_store)

    default_fov = float(sensor_get(sensor, "fov", None, oi))
    scene_fov = float(sensor_get(sensor, "fov", scene, oi))

    assert scene_fov < default_fov


def test_sensor_noise_flag_one_includes_fpn(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    oi.data["photons"][:] = 0.0

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 1.0)
    sensor = sensor_set(sensor, "noise flag", 1)
    sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
    sensor.fields["pixel"]["read_noise_v"] = 0.0
    sensor.fields["pixel"]["dsnu_sigma_v"] = 0.01
    sensor.fields["pixel"]["prnu_sigma"] = 0.0

    noisy = sensor_compute(sensor, oi, seed=0)
    assert np.any(noisy.data["volts"] > 0.0)


def test_sensor_noise_flag_minus_two_keeps_zero_signal_zero(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    oi.data["photons"][:] = 0.0

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 1.0)
    sensor = sensor_set(sensor, "noise flag", -2)
    sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.1
    sensor.fields["pixel"]["read_noise_v"] = 0.0
    sensor.fields["pixel"]["dsnu_sigma_v"] = 0.01
    sensor.fields["pixel"]["prnu_sigma"] = 0.25

    noisy = sensor_compute(sensor, oi, seed=0)
    assert np.allclose(noisy.data["volts"], 0.0)


def test_ip_compute_default_pipeline(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_compute(sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0), oi, seed=0)
    ip = ip_compute(ip_create(sensor=sensor, asset_store=asset_store), sensor, asset_store=asset_store)
    assert ip.data["result"].shape[:2] == sensor.fields["size"]
    assert ip.data["result"].shape[2] == 3
    assert np.all((ip.data["result"] >= 0.0) & (ip.data["result"] <= 1.0))


def test_camera_compute_end_to_end(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)
    result = camera.fields["ip"].data["result"]
    assert result.shape[:2] == camera.fields["sensor"].fields["size"]
    assert result.shape[2] == 3


def test_camera_compute_skips_resize_when_sensor_fov_is_already_close(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_create(asset_store=asset_store)
    sensor = camera.fields["sensor"].clone()
    oi = camera.fields["oi"]

    target_hfov = float(sensor_get(sensor, "fov horizontal", scene, oi))
    target_vfov = float(sensor_get(sensor, "fov vertical", scene, oi))
    scene.fields["fov_deg"] = target_hfov * 1.005
    scene.fields["vfov_deg"] = target_vfov * 0.995
    original_size = sensor.fields["size"]

    camera.fields["sensor"] = sensor
    camera = camera_compute(camera, scene, asset_store=asset_store)

    assert camera.fields["sensor"].fields["size"] == original_size


def test_camera_parity_case_disables_sensor_noise(asset_store) -> None:
    payload = run_python_case("camera_default_pipeline", asset_store=asset_store)
    assert payload["sensor_volts"].ndim == 2

    scene = scene_create(asset_store=asset_store)
    noiseless_camera = camera_create(asset_store=asset_store)
    noiseless_camera.fields["sensor"] = sensor_set(noiseless_camera.fields["sensor"], "noise flag", 0)
    noiseless_camera = camera_compute(noiseless_camera, scene, asset_store=asset_store)

    assert np.allclose(payload["sensor_volts"], noiseless_camera.fields["sensor"].data["volts"])
