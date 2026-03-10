from __future__ import annotations

import imageio.v3 as iio
import numpy as np

from pyisetcam import blackbody, display_create, display_get, scene_adjust_illuminant, scene_create, scene_from_file, scene_get


def test_scene_create_default_macbeth(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    assert photons.shape == (64, 96, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_adjust_illuminant_preserves_mean(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    wave = scene_get(scene, "wave")
    changed = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=asset_store)
    changed_no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=asset_store)
    assert np.isclose(
        scene_get(changed, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=5e-2,
    )
    assert not np.isclose(
        scene_get(changed_no_preserve, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=1e-2,
    )


def test_supported_pattern_scenes(asset_store) -> None:
    checkerboard = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    slanted_bar = scene_create("slanted bar", 64, 0.6, 3.0, asset_store=asset_store)
    freq_orient = scene_create("frequency orientation", asset_store=asset_store)
    harmonic = scene_create("harmonic", asset_store=asset_store)
    sweep = scene_create("sweep frequency", asset_store=asset_store)
    star = scene_create("star pattern", 64, "ee", 6, asset_store=asset_store)
    line = scene_create("line ee", 32, 2, asset_store=asset_store)
    bar = scene_create("bar", 32, 5, asset_store=asset_store)
    point_array = scene_create("point array", 64, 16, "ep", 3, asset_store=asset_store)
    grid_lines = scene_create("grid lines", 64, 16, "ee", 2, asset_store=asset_store)
    white_noise = scene_create("white noise", 32, 20, asset_store=asset_store)
    assert scene_get(checkerboard, "photons").shape[:2] == (64, 64)
    assert scene_get(slanted_bar, "photons").shape[:2] == (64, 64)
    assert scene_get(freq_orient, "photons").shape[:2] == (256, 256)
    assert scene_get(harmonic, "photons").shape[:2] == (65, 65)
    assert scene_get(sweep, "photons").shape[:2] == (128, 128)
    assert scene_get(star, "photons").shape[:2] == (64, 64)
    assert scene_get(line, "photons").shape[:2] == (32, 32)
    assert scene_get(bar, "photons").shape[:2] == (32, 32)
    assert scene_get(point_array, "photons").shape[:2] == (64, 64)
    assert scene_get(grid_lines, "photons").shape[:2] == (64, 64)
    assert scene_get(white_noise, "photons").shape[:2] == (32, 32)
    assert np.isclose(scene_get(freq_orient, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(harmonic, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(sweep, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(star, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(freq_orient, "fov"), 10.0)
    assert np.isclose(scene_get(harmonic, "fov"), 1.0)
    assert np.isclose(scene_get(sweep, "fov"), 10.0)
    assert np.isclose(scene_get(point_array, "fov"), 40.0)
    assert np.isclose(scene_get(grid_lines, "fov"), 40.0)
    assert np.isclose(scene_get(white_noise, "fov"), 1.0)


def test_star_pattern_scene_matches_radial_lines_alias(asset_store) -> None:
    star = scene_create("star pattern", 64, "ee", 6, asset_store=asset_store)
    radial = scene_create("radial lines", 64, "ee", 6, asset_store=asset_store)

    star_photons = scene_get(star, "photons")
    radial_photons = scene_get(radial, "photons")

    assert np.array_equal(star_photons, radial_photons)
    assert star_photons.shape[:2] == (64, 64)
    assert np.isclose(scene_get(star, "fov"), 10.0)
    assert float(star_photons[32, 32, 0]) > float(star_photons[0, 0, 0])


def test_uniform_blackbody_and_monochromatic_scenes(asset_store) -> None:
    bb = scene_create("uniform bb", 16, 4500, asset_store=asset_store)
    mono = scene_create("uniform monochromatic", 550, 12, asset_store=asset_store)

    assert scene_get(bb, "photons").shape == (16, 16, scene_get(bb, "wave").size)
    assert scene_get(mono, "photons").shape == (12, 12, 1)
    assert np.array_equal(scene_get(mono, "wave"), np.array([550.0]))


def test_line_and_bar_patterns_have_centered_bright_features(asset_store) -> None:
    line = scene_create("line ee", 33, 1, asset_store=asset_store)
    bar = scene_create("bar", 33, 3, asset_store=asset_store)

    line_plane = scene_get(line, "photons")[:, :, 0]
    bar_plane = scene_get(bar, "photons")[:, :, 0]
    line_column_energy = np.sum(line_plane, axis=0)
    bar_column_energy = np.sum(bar_plane, axis=0)

    assert int(np.argmax(line_column_energy)) == 16 + 1
    assert np.array_equal(np.sort(np.argsort(bar_column_energy)[-3:]), np.array([15, 16, 17]))


def test_point_array_and_grid_lines_follow_spacing(asset_store) -> None:
    point_array = scene_create("point array", 32, 8, "ep", 1, asset_store=asset_store)
    grid_lines = scene_create("grid lines", 32, 8, "ep", 1, asset_store=asset_store)

    point_plane = scene_get(point_array, "photons")[:, :, 0]
    grid_plane = scene_get(grid_lines, "photons")[:, :, 0]

    point_positions = np.argwhere(point_plane > 0.5)
    assert [int(point_positions[0, 0]), int(point_positions[0, 1])] == [3, 3]
    assert np.all((point_positions[:, 0] - 3) % 8 == 0)
    assert np.all((point_positions[:, 1] - 3) % 8 == 0)

    assert np.all(grid_plane[3::8, :] > 0.5)
    assert np.all(grid_plane[:, 3::8] > 0.5)


def test_frequency_orientation_scene_matches_upstream_parameterization(asset_store) -> None:
    params = {
        "angles": np.linspace(0.0, np.pi / 2.0, 5),
        "freqs": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        "blockSize": 64,
        "contrast": 0.8,
    }
    scene = scene_create("frequency orientation", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (320, 320)
    assert np.max(photons) > np.min(photons)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_freq_orient_scalar_size_matches_tutorial_shape(asset_store) -> None:
    scene = scene_create("freq orient", 512, asset_store=asset_store)
    assert scene_get(scene, "photons").shape[:2] == (512, 512)


def test_harmonic_scene_supports_multiple_components_and_gabor_window(asset_store) -> None:
    params = {
        "freq": np.array([1.0, 5.0]),
        "contrast": np.array([0.2, 0.6]),
        "ph": np.array([0.0, np.pi / 3.0]),
        "ang": np.array([0.0, 0.0]),
        "row": 128,
        "col": 128,
        "GaborFlag": 0.2,
    }
    scene = scene_create("harmonic", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (128, 128)
    assert np.isclose(scene_get(scene, "fov"), 1.0)
    assert photons[64, 64] > photons[0, 0]


def test_sweep_frequency_scene_supports_custom_frequency_and_contrast_profile(asset_store) -> None:
    y_contrast = np.linspace(1.0, 0.25, 64, dtype=float)
    scene = scene_create("sweepFrequency", 64, 12, None, y_contrast, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (64, 64)
    assert np.ptp(photons[0, :]) > np.ptp(photons[-1, :])
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_reflectance_chart_scene_supports_explicit_sample_lists(asset_store) -> None:
    scene = scene_create(
        "reflectance chart",
        8,
        [[1, 2], [1, 2], [1]],
        [
            "MunsellSamples_Vhrel.mat",
            "Food_Vhrel.mat",
            "skin/HyspexSkinReflectance.mat",
        ],
        None,
        True,
        "without replacement",
        asset_store=asset_store,
    )

    photons = scene_get(scene, "photons")
    chart_parameters = scene_get(scene, "chart parameters")

    assert photons.shape == (24, 24, scene_get(scene, "wave").size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.array_equal(chart_parameters["rowcol"], np.array([3, 3]))
    assert chart_parameters["rIdxMap"].shape == (24, 24)
    assert np.array_equal(chart_parameters["sSamples"][0], np.array([1, 2]))
    assert np.array_equal(chart_parameters["sSamples"][1], np.array([1, 2]))
    assert np.array_equal(chart_parameters["sSamples"][2], np.array([1]))


def test_reflectance_chart_scene_supports_struct_parameters_and_absolute_paths(asset_store) -> None:
    params = {
        "pSize": 8,
        "sFiles": [
            asset_store.resolve("data/surfaces/reflectances/MunsellSamples_Vhrel.mat"),
            asset_store.resolve("data/surfaces/reflectances/Food_Vhrel.mat"),
            asset_store.resolve("data/surfaces/reflectances/skin/HyspexSkinReflectance.mat"),
        ],
        "sSamples": [[1], [1], [1]],
        "grayFlag": False,
        "sampling": "all",
    }

    scene = scene_create("reflectance chart", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")
    chart_parameters = scene_get(scene, "chart parameters")

    assert photons.shape == (16, 16, scene_get(scene, "wave").size)
    assert np.array_equal(chart_parameters["rowcol"], np.array([2, 2]))
    assert all(np.array_equal(item, np.array([1])) for item in chart_parameters["sSamples"])
    assert all(path.endswith(".mat") for path in chart_parameters["sFiles"])


def test_scene_from_file_rgb_array_uses_display_geometry(asset_store) -> None:
    display = display_create("default", asset_store=asset_store)
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )

    scene = scene_from_file(image, "rgb", 50.0, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 2, scene_get(scene, "wave").size)
    assert np.isclose(scene_get(scene, "distance"), display_get(display, "viewing distance"))
    assert np.isclose(scene_get(scene, "fov"), 2.0 * display_get(display, "deg per dot"))
    assert scene_get(scene, "filename") == "numerical"
    assert scene_get(scene, "source type") == "rgb"
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 50.0, rtol=5e-2)


def test_scene_from_file_monochrome_file_path(tmp_path, asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.array(
        [
            [0, 64, 128],
            [255, 192, 32],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "mono.png"
    iio.imwrite(image_path, image)

    scene = scene_from_file(image_path, "monochrome", None, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 3, scene_get(scene, "wave").size)
    assert scene_get(scene, "filename") == str(image_path)
    assert scene.name.startswith("mono - ")
    assert np.all(np.isfinite(photons))
    assert float(np.mean(photons[1, 0, :])) > float(np.mean(photons[0, 0, :]))


def test_scene_from_file_preserves_display_wave_when_not_overridden(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scene = scene_from_file(image, "rgb", None, display, asset_store=asset_store)
    assert np.array_equal(scene_get(scene, "wave"), display_get(display, "wave"))
