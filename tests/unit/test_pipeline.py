from __future__ import annotations

import numpy as np
import pytest

from pyisetcam import (
    camera_compute,
    camera_create,
    camera_get,
    camera_set,
    ip_get,
    ip_set,
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
    wvf_create,
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


def test_oi_compute_pixel_size_matches_requested_spacing(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True, pixel_size=2e-6)

    assert np.isclose(oi_get(oi, "sample size"), 2e-6)
    assert np.isclose(oi_get(oi, "distance per sample")[1], 2e-6)
    assert np.isclose(oi.fields["requested_pixel_size_m"], 2e-6)


def test_oi_transmittance_scales_and_interpolates(asset_store) -> None:
    wave = np.array([400.0, 550.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 8, wave, asset_store=asset_store)

    baseline_oi = oi_create()
    baseline_oi.fields["optics"]["model"] = "skip"
    baseline_oi.fields["optics"]["offaxis_method"] = "skip"
    baseline = oi_compute(baseline_oi, scene, crop=True)

    scaled_oi = oi_create()
    scaled_oi.fields["optics"]["model"] = "skip"
    scaled_oi.fields["optics"]["offaxis_method"] = "skip"
    scaled_oi = oi_set(scaled_oi, "transmittance wave", wave)
    scaled_oi = oi_set(scaled_oi, "transmittance scale", np.array([0.5, 1.0, 0.25], dtype=float))
    scaled = oi_compute(scaled_oi, scene, crop=True)

    center = (scaled.data["photons"].shape[0] // 2, scaled.data["photons"].shape[1] // 2)
    ratio = scaled.data["photons"][center[0], center[1], :] / baseline.data["photons"][center[0], center[1], :]

    assert np.allclose(ratio, np.array([0.5, 1.0, 0.25], dtype=float))
    assert np.allclose(oi_get(scaled_oi, "transmittance", np.array([475.0, 625.0], dtype=float)), np.array([0.75, 0.625]))
    assert np.array_equal(oi_get(scaled_oi, "transmittance wave"), wave)
    assert oi_get(scaled_oi, "transmittance nwave") == 3


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
    assert wavefront["lca_method"] == "none"
    assert wavefront["compute_sce"] is False
    assert np.array_equal(wavefront["zcoeffs"], np.array([0.0]))
    assert np.array_equal(wavefront["sce_params"]["wave"], wavefront["wave"])
    assert np.allclose(wavefront["sce_params"]["rho"], 0.0)
    assert np.isclose(wavefront["sce_params"]["xo_mm"], 0.0)
    assert np.isclose(wavefront["sce_params"]["yo_mm"], 0.0)


def test_oi_compute_wvf_uses_custom_aperture(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)
    aperture = np.ones((9, 9), dtype=float)
    aperture[:, :4] = 0.0
    custom_oi = oi_compute(oi_create("wvf"), scene, crop=True, aperture=aperture)

    assert custom_oi.data["photons"].shape == default_oi.data["photons"].shape
    assert not np.allclose(custom_oi.data["photons"], default_oi.data["photons"])


def test_oi_compute_wvf_default_aperture_matches_full_open_aperture(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)
    full_aperture = np.ones((9, 9), dtype=float)
    explicit_oi = oi_compute(oi_create("wvf"), scene, crop=True, aperture=full_aperture)

    assert np.allclose(explicit_oi.data["photons"], default_oi.data["photons"])


def test_oi_compute_wvf_zcoeffs_change_wavefront_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)

    custom_wvf = wvf_create(
        wave=scene.fields["wave"],
        focal_length_m=0.003862755099228,
        f_number=4.0,
        calc_pupil_diameter_mm=9.6569e-01,
        zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.25], dtype=float),
    )
    custom_oi = oi_compute(oi_create("wvf", custom_wvf), scene, crop=True)

    row = custom_oi.data["photons"].shape[0] // 2
    dark_col = 8
    band = 0
    dark_default = float(default_oi.data["photons"][row, dark_col, band])
    dark_custom = float(custom_oi.data["photons"][row, dark_col, band])

    assert dark_custom > dark_default


def test_oi_compute_wvf_sce_changes_wavefront_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)

    custom_wvf = wvf_create(
        wave=scene.fields["wave"],
        focal_length_m=0.003862755099228,
        f_number=4.0,
        calc_pupil_diameter_mm=9.6569e-01,
        compute_sce=True,
        sce_params={
            "wave": scene.fields["wave"],
            "rho": np.full(scene.fields["wave"].shape, 200.0, dtype=float),
            "xo_mm": 0.0,
            "yo_mm": 0.0,
        },
    )
    custom_oi = oi_compute(oi_create("wvf", custom_wvf), scene, crop=True)

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


def test_sensor_get_set_supports_n_samples_per_pixel(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "n samples per pixel", 3)

    assert sensor_get(sensor, "n samples per pixel") == 3


def test_sensor_get_reports_matlab_style_geometry_and_cfa_metadata(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    rows, cols = sensor.fields["size"]

    support = sensor_get(sensor, "spatial support", "um")
    pattern_colors = sensor_get(sensor, "pattern colors")

    assert np.isclose(sensor_get(sensor, "height"), rows * pixel_size[0])
    assert np.isclose(sensor_get(sensor, "width", "mm"), cols * pixel_size[1] * 1e3)
    assert np.allclose(sensor_get(sensor, "dimension", "um"), np.array([rows * pixel_size[0], cols * pixel_size[1]]) * 1e6)
    assert np.isclose(sensor_get(sensor, "deltax", "um"), pixel_size[1] * 1e6)
    assert np.isclose(sensor_get(sensor, "deltay"), pixel_size[0])
    assert support["x"].shape == (cols,)
    assert support["y"].shape == (rows,)
    assert np.isclose(support["x"][0], -support["x"][-1])
    assert np.isclose(support["y"][0], -support["y"][-1])
    assert sensor_get(sensor, "unit block rows") == 2
    assert sensor_get(sensor, "unit block cols") == 2
    assert sensor_get(sensor, "cfa size") == (2, 2)
    assert sensor_get(sensor, "filter color letters") == "rgb"
    assert pattern_colors.shape == (2, 2)
    assert np.array_equal(pattern_colors, np.array([["g", "r"], ["b", "g"]], dtype="<U1"))


def test_sensor_set_size_respects_cfa_block_and_clears_cached_data(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.ones((7, 9), dtype=float))

    assert sensor_get(sensor, "size") == (7, 9)

    sensor = sensor_set(sensor, "size", (73, 89))
    assert sensor_get(sensor, "size") == (72, 88)
    assert sensor.data == {}

    sensor = sensor_set(sensor, "rows", 75)
    sensor = sensor_set(sensor, "cols", 91)
    assert sensor_get(sensor, "rows") == 74
    assert sensor_get(sensor, "cols") == 90


def test_sensor_set_etendue_scales_noiseless_response(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    baseline_sensor = sensor_create(asset_store=asset_store)
    baseline_sensor = sensor_set(baseline_sensor, "integration time", 0.01)
    baseline_sensor = sensor_set(baseline_sensor, "noise flag", 0)

    attenuated_sensor = sensor_set(
        baseline_sensor.clone(),
        "sensor etendue",
        np.full(baseline_sensor.fields["size"], 0.5, dtype=float),
    )

    baseline = sensor_compute(baseline_sensor, oi, seed=0)
    attenuated = sensor_compute(attenuated_sensor, oi, seed=0)

    assert np.allclose(attenuated.data["volts"], baseline.data["volts"] * 0.5)
    assert np.allclose(sensor_get(attenuated, "sensor etendue"), 0.5)


def test_sensor_compute_rejects_unsupported_vignetting_modes(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 1.0)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "vignetting", 1)

    with pytest.raises(NotImplementedError):
        sensor_compute(sensor, oi, seed=0)


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


def test_sensor_compute_supersampling_changes_bayer_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    default_sensor = sensor_create(asset_store=asset_store)
    default_sensor = sensor_set(default_sensor, "integration time", 1.0)
    default_sensor = sensor_set(default_sensor, "noise flag", 0)

    supersampled_sensor = sensor_set(default_sensor.clone(), "n samples per pixel", 3)

    default_result = sensor_compute(default_sensor, oi, seed=0)
    supersampled_result = sensor_compute(supersampled_sensor, oi, seed=0)

    assert default_result.data["volts"].shape == supersampled_result.data["volts"].shape
    assert not np.allclose(default_result.data["volts"], supersampled_result.data["volts"])


def test_ip_compute_default_pipeline(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_compute(sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0), oi, seed=0)
    ip = ip_compute(ip_create(sensor=sensor, asset_store=asset_store), sensor, asset_store=asset_store)
    assert ip.data["result"].shape[:2] == sensor.fields["size"]
    assert ip.data["result"].shape[2] == 3
    assert np.all((ip.data["result"] >= 0.0) & (ip.data["result"] <= 1.0))


def test_ip_get_set_support_matlab_style_transforms(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)

    ip = ip_set(ip, "display dpi", 110)
    ip = ip_set(ip, "render flag", "gray")
    ip = ip_set(ip, "scale display", False)
    ip = ip_set(ip, "sensor conversion matrix", np.eye(3))
    ip = ip_set(ip, "illuminant correction matrix", 2.0 * np.eye(3))
    ip = ip_set(ip, "ics2display transform", 3.0 * np.eye(3))

    assert ip_get(ip, "display dpi") == 110
    assert ip_get(ip, "display spd").shape[1] == 3
    assert ip_get(ip, "render flag") == 3
    assert ip_get(ip, "scale display") is False
    assert np.allclose(ip_get(ip, "combined transform"), 6.0 * np.eye(3))


def test_camera_get_set_routes_matlab_style_subobjects(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)

    camera = camera_set(camera, "sensor integration time", 0.125)
    camera = camera_set(camera, "pixel voltage swing", 1.5)
    camera = camera_set(camera, "ip display dpi", 110)
    camera = camera_set(camera, "optics f number", 5.6)

    assert np.isclose(camera_get(camera, "sensor integration time"), 0.125)
    assert np.isclose(camera_get(camera, "pixel voltage swing"), 1.5)
    assert camera_get(camera, "ip display dpi") == 110
    assert np.isclose(camera_get(camera, "optics f number"), 5.6)
    assert camera_get(camera, "vci type") == "default"


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
