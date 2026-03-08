from __future__ import annotations

import numpy as np
import pytest
import pyisetcam.optics as optics_module

from pyisetcam.parity import run_python_case_with_context
from pyisetcam import (
    camera_compute,
    camera_create,
    camera_get,
    camera_set,
    ie_field_height_to_index,
    ip_get,
    ip_set,
    ip_compute,
    ip_create,
    optics_ray_trace,
    oi_calculate_illuminance,
    oi_diffuser,
    oi_compute,
    oi_create,
    oi_get,
    oi_set,
    rt_angle_lut,
    rt_block_center,
    rt_choose_block_size,
    rt_di_interp,
    rt_extract_block,
    rt_filtered_block_support,
    rt_file_names,
    rt_geometry,
    rt_import_data,
    rt_insert_block,
    rt_otf,
    rt_psf_apply,
    rt_psf_grid,
    rt_psf_interp,
    rt_precompute_psf,
    rt_precompute_psf_apply,
    rt_ri_interp,
    rt_sample_heights,
    rt_synthetic,
    run_python_case,
    scene_create,
    scene_get,
    scene_set,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
    wvf_create,
    zemax_load,
    zemax_read_header,
)


def _write_mock_zemax_bundle(
    tmp_path,
    *,
    lens_file: str = "CookeLens.ZMX",
    base_lens_file_name: str = "CookeLens",
    wave_assignment: str = "500:100:600",
    base_lens_has_semicolon: bool = True,
    psf_size_assignment: int = 2,
    params_file_name: str = "ISETPARAMS.txt",
    psf_spacing_assignment_mm: float | None = 0.00025,
):
    params_file = tmp_path / params_file_name
    base_lens_line = f"baseLensFileName='{base_lens_file_name}'"
    if base_lens_has_semicolon:
        base_lens_line += ";"
    psf_spacing_line = "" if psf_spacing_assignment_mm is None else f"psfSpacing={psf_spacing_assignment_mm:.7f};\n"
    params_file.write_text(
        "".join(
            [
                f"lensFile='{lens_file}';\n",
                f"psfSize={psf_size_assignment};\n",
                psf_spacing_line,
                f"wave={wave_assignment};\n",
                "imgHeightNum=2;\n",
                "imgHeightMax=1.0;\n",
                "objDist=250.0;\n",
                "mag=-0.1;\n",
                f"{base_lens_line}\n",
                "refWave=550.0;\n",
                "fov=15.0;\n",
                "efl=6.0;\n",
                "fnumber_eff=1.8;\n",
                "fnumber=2.0;\n",
            ]
        ),
        encoding="latin1",
    )
    (tmp_path / "CookeLens_DI_.dat").write_text("0.0 0.05 0.2 0.3\n", encoding="latin1")
    (tmp_path / "CookeLens_RI_.dat").write_text("1.0 0.9 0.8 0.7\n", encoding="latin1")
    kernels = {
        "CookeLens_2D_PSF_Fld1_Wave1.dat": "1 0 0 0\n",
        "CookeLens_2D_PSF_Fld1_Wave2.dat": "0 1 0 0\n",
        "CookeLens_2D_PSF_Fld2_Wave1.dat": "0 0 1 0\n",
        "CookeLens_2D_PSF_Fld2_Wave2.dat": "0 0 0 1\n",
    }
    for name, data in kernels.items():
        (tmp_path / name).write_text(
            "spacing is 0.5000 microns\n"
            "area is 1.0000 microns\n"
            "normalized.\n"
            f"{data}",
            encoding="latin1",
        )
    return params_file


def test_oi_compute_matches_scene_wave(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    assert oi.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.array_equal(oi.fields["wave"], scene.fields["wave"])


def test_scene_get_depth_map_defaults_to_scene_distance(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    depth_map = scene_get(scene, "depth map")

    assert depth_map.shape == scene_get(scene, "size")
    assert np.allclose(depth_map, scene_get(scene, "distance"))
    assert np.allclose(scene_get(scene, "depth range"), np.array([scene_get(scene, "distance"), scene_get(scene, "distance")]))


def test_ie_field_height_to_index_matches_matlab_rules() -> None:
    heights = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

    assert ie_field_height_to_index(heights, 0.6) == 2
    assert ie_field_height_to_index(heights, 0.9) == 3
    assert ie_field_height_to_index(heights, 0.1, bounding=True) == (1, 2)
    assert ie_field_height_to_index(heights, 0.9, bounding=True) == (2, 3)


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


def test_oi_compute_tracks_padded_and_cropped_depth_maps(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    rows, cols = scene_get(scene, "size")
    depth_map = np.linspace(1.0, 1.4, rows * cols, dtype=float).reshape(rows, cols)
    scene = scene_set(scene, "depth map", depth_map)

    oi_uncropped = oi_compute(oi_create(), scene, crop=False)
    uncropped_depth = oi_get(oi_uncropped, "depth map")
    pad_rows, pad_cols = oi_uncropped.fields["padding_pixels"]

    assert uncropped_depth.shape == oi_uncropped.data["photons"].shape[:2]
    assert np.allclose(uncropped_depth[:pad_rows, :], 0.0)
    assert np.allclose(uncropped_depth[:, :pad_cols], 0.0)
    assert np.allclose(uncropped_depth[pad_rows:-pad_rows, pad_cols:-pad_cols], depth_map)

    oi_cropped = oi_compute(oi_create(), scene, crop=True)
    assert np.allclose(oi_get(oi_cropped, "depth map"), depth_map)


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


def test_diffraction_otf_matches_oi_frequency_support(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=False)
    optics = oi.fields["optics"]
    wave = oi_get(oi, "wave")
    sample_spacing = float(oi_get(oi, "sample spacing")[0])

    otf = optics_module._diffraction_otf(oi.data["photons"].shape[:2], sample_spacing, wave, optics, scene)
    frequency_support = oi_get(oi, "frequency support", "m")
    rho = np.sqrt(frequency_support[:, :, 0] ** 2 + frequency_support[:, :, 1] ** 2)
    cutoff = (
        (float(optics["focal_length_m"]) / float(optics["f_number"]) / float(optics["focal_length_m"]))
        / np.maximum(np.asarray(wave, dtype=float) * 1e-9, 1e-12)
    )
    expected = np.zeros_like(otf)
    for index, cutoff_frequency in enumerate(cutoff):
        normalized = rho / max(float(cutoff_frequency), 1e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        current = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        current[normalized >= 1.0] = 0.0
        expected[:, :, index] = np.fft.ifftshift(current)

    assert np.allclose(otf, expected)


def test_oi_get_image_distance_uses_depth_map_when_geometry_is_not_precomputed() -> None:
    oi = oi_create()
    oi = oi_set(oi, "photons", np.ones((4, 6, 3), dtype=float))
    oi = oi_set(oi, "depth map", np.full((4, 6), 1.2, dtype=float))
    oi.fields.pop("image_distance_m", None)
    oi.fields.pop("width_m", None)
    oi.fields.pop("height_m", None)
    oi.fields.pop("sample_spacing_m", None)

    focal_length = oi_get(oi, "focal length")
    expected = 1.0 / ((1.0 / focal_length) - (1.0 / 1.2))

    assert np.isclose(oi_get(oi, "image distance"), expected)


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


def test_oi_create_raytrace_loads_upstream_optics(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    raytrace = oi_get(oi, "raytrace")

    assert oi_get(oi, "model") == "raytrace"
    assert oi_get(oi, "compute method") == ""
    assert np.isclose(oi_get(oi, "focal length"), 0.001999989, rtol=1e-6, atol=1e-9)
    assert np.isclose(oi_get(oi, "fnumber"), 4.999973)
    assert np.isclose(oi_get(oi, "rt object distance"), 2.0)
    assert np.isclose(oi_get(oi, "rtfov"), 38.72116733777534)
    assert oi_get(oi, "raytrace optics name") == "Asphere 2mm"
    assert optics["model"] == "raytrace"
    assert np.isclose(optics["fNumber"], oi_get(oi, "fnumber"))
    assert np.isclose(optics["focalLength"], oi_get(oi, "focal length"))
    assert optics["rayTrace"]["lensFile"].endswith(".ZMX")
    assert raytrace["lensFile"].endswith(".ZMX")
    assert np.isclose(raytrace["objectDistance"], 2000.0)
    assert oi_get(oi, "rtpsffieldheight").shape == (21,)
    assert np.allclose(oi_get(oi, "rtpsffieldheight", "mm"), raytrace["psf"]["fieldHeight"])
    assert np.allclose(oi_get(oi, "rtpsfsamplespacing"), np.array([2.5e-7, 2.5e-7]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.25, 0.25]))
    assert np.array_equal(oi_get(oi, "rtpsfwavelength"), np.array([400.0, 475.0, 550.0, 625.0, 700.0]))
    assert oi_get(oi, "optics rtpsfsize") == oi_get(oi, "rtpsf")["function"].shape
    assert oi_get(oi, "rtpsfsize") == (0, 0)
    assert "fieldHeight" in oi_get(oi, "rtpsf")
    assert "sampleSpacing" in oi_get(oi, "rtpsf")


def test_oi_create_raytrace_exposes_raw_psf_support_axes(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    support_x = np.asarray(oi_get(oi, "rtpsfsupportx", "um"), dtype=float)
    support_y = np.asarray(oi_get(oi, "rtpsfsupporty", "um"), dtype=float).reshape(-1)
    freq_x = np.asarray(oi_get(oi, "rtfreqsupportx", "mm"), dtype=float)
    freq_y = np.asarray(oi_get(oi, "rtfreqsupporty", "mm"), dtype=float).reshape(-1)
    spacing_mm = np.asarray(oi_get(oi, "rtpsfspacing", "mm"), dtype=float)

    assert support_x.shape == (128,)
    assert support_y.shape == (128,)
    assert np.isclose(support_x[63], 0.0)
    assert np.isclose(support_y[63], 0.0)
    assert np.isclose(support_x[0], -63.0 * 0.25)
    assert np.isclose(support_x[-1], 64.0 * 0.25)
    assert np.isclose(freq_x[1] - freq_x[0], 1.0 / (128.0 * spacing_mm[1]))
    assert np.isclose(freq_y[1] - freq_y[0], 1.0 / (128.0 * spacing_mm[0]))


def test_oi_create_raytrace_exposes_raw_geometry_and_relillum_tables(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    geometry = oi_get(oi, "rtgeometry")
    rel_illum = oi_get(oi, "rtrelillum")

    assert oi_get(oi, "rtname") == "Asphere 2mm"
    assert oi_get(oi, "rtopticsprogram") == "Zemax"
    assert oi_get(oi, "rtlensfile").endswith(".ZMX")
    assert np.isclose(oi_get(oi, "rtefl", "mm"), 1.999989, atol=1e-6)
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 4.895375)
    assert np.isclose(oi_get(oi, "rtfnumber"), 4.999973)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.001)
    assert np.isclose(oi_get(oi, "rtrefwave"), 450.0)
    assert np.isclose(oi_get(oi, "rtobjdist", "mm"), 2000.0)
    assert np.isclose(oi_get(oi, "rtmaxfov"), oi_get(oi, "rtfov"))
    assert geometry["function"].shape == (21, 5)
    assert rel_illum["function"].shape == (21, 5)
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), rel_illum["fieldHeight"])
    assert np.allclose(oi_get(oi, "rtgeomfieldheight", "mm"), geometry["fieldHeight"])
    assert np.isclose(oi_get(oi, "rtgeommaxfieldheight", "mm"), np.max(geometry["fieldHeight"]))
    assert np.array_equal(oi_get(oi, "rtriwavelength"), rel_illum["wavelength"])
    assert np.array_equal(oi_get(oi, "rtgeomwavelength"), geometry["wavelength"])
    assert np.allclose(oi_get(oi, "rtgeomfunction"), geometry["function"])
    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0, "mm"), geometry["function"][:, 2])
    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0), geometry["function"][:, 2] / 1e3)
    assert np.allclose(oi_get(oi, "rtrifunction"), rel_illum["function"])


def test_oi_compute_raytrace_applies_lens_shading_and_blur(asset_store) -> None:
    scene = scene_create("uniform ee", 32, asset_store=asset_store)
    raytrace = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    skip = oi_create()
    skip.fields["optics"]["model"] = "skip"
    skip.fields["optics"]["offaxis_method"] = "skip"
    baseline = oi_compute(skip, scene, crop=True)

    band = raytrace.data["photons"].shape[2] // 2
    center = float(raytrace.data["photons"][raytrace.data["photons"].shape[0] // 2, raytrace.data["photons"].shape[1] // 2, band])
    corner = float(raytrace.data["photons"][0, 0, band])

    assert raytrace.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.allclose(oi_get(raytrace, "depth map"), 2.0)
    assert center > corner
    assert not np.allclose(raytrace.data["photons"], baseline.data["photons"])


def test_oi_compute_raytrace_builds_precomputed_psf_structure(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30)
    result = oi_compute(oi, scene, crop=True)

    psf_struct = oi_get(result, "psf struct")
    sampled = oi_get(result, "sampledRTpsf")

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert isinstance(psf_struct, dict)
    assert "sampAngles" in psf_struct
    assert "imgHeight" in psf_struct
    assert "wavelength" in psf_struct
    assert sampled is not None
    assert sampled.dtype == object
    assert sampled.ndim == 3
    assert np.array_equal(oi_get(result, "psfwavelength"), np.array([550.0]))
    assert oi_get(result, "rtpsfsize") == sampled[0, 0, 0].shape
    assert oi_get(result, "optics rtpsfsize") == oi_get(result, "rtpsf")["function"].shape
    assert oi_get(result, "raytrace optics name") == "Asphere 2mm"


def test_oi_compute_raytrace_crop_false_tracks_padding_and_depth_map(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    result = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)

    pad_rows, pad_cols = oi_get(result, "padding pixels")
    photons = np.asarray(result.data["photons"], dtype=float)
    depth_map = np.asarray(oi_get(result, "depth map"), dtype=float)
    base_rows, base_cols = scene.data["photons"].shape[:2]

    assert pad_rows > 0
    assert pad_cols > 0
    assert photons.shape[:2] == (base_rows + 2 * pad_rows, base_cols + 2 * pad_cols)
    assert depth_map.shape == photons.shape[:2]
    assert np.allclose(depth_map[pad_rows:-pad_rows, pad_cols:-pad_cols], 2.0)
    assert np.allclose(depth_map[:pad_rows, :], 0.0)
    assert np.allclose(depth_map[-pad_rows:, :], 0.0)
    assert np.allclose(depth_map[:, :pad_cols], 0.0)
    assert np.allclose(depth_map[:, -pad_cols:], 0.0)


def test_oi_get_set_raytrace_sample_angles_matches_matlab_surface(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    sample_angles = np.arange(0.0, 361.0, 45.0, dtype=float)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf sample angles", sample_angles)

    assert np.array_equal(oi_get(oi, "psf sample angles"), sample_angles)
    assert np.isclose(oi_get(oi, "psf angle step"), 45.0)

    result = oi_compute(oi, scene, crop=True)

    assert np.array_equal(oi_get(result, "psf sample angles"), sample_angles)
    assert np.isclose(oi_get(result, "psf angle step"), 45.0)
    assert np.allclose(oi_get(result, "psf image heights", "mm"), oi_get(result, "psf image heights") * 1e3)


def test_oi_set_psfstruct_normalizes_matlab_style_metadata(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    psf_cells = np.empty((2, 2, 1), dtype=object)
    psf_cells[0, 0, 0] = np.ones((3, 3), dtype=float)
    psf_cells[0, 1, 0] = np.full((3, 3), 2.0, dtype=float)
    psf_cells[1, 0, 0] = np.full((3, 3), 3.0, dtype=float)
    psf_cells[1, 1, 0] = np.full((3, 3), 4.0, dtype=float)
    psf_struct = {
        "psf": psf_cells,
        "sampAngles": np.array([0.0, 180.0], dtype=float),
        "imgHeight": np.array([0.0, 1.5e-3], dtype=float),
        "wavelength": np.array([550.0], dtype=float),
        "opticsName": "Synthetic RT",
    }

    oi = oi_set(oi, "shift variant structure", psf_struct)

    exported = oi_get(oi, "psf struct")
    sampled = oi_get(oi, "sampledRTpsf")
    assert exported["psf"].shape == (2, 2, 1)
    assert sampled.shape == (2, 2, 1)
    assert sampled.dtype == object
    assert sampled[1, 1, 0].shape == (3, 3)
    assert np.array_equal(oi_get(oi, "psf sample angles"), np.array([0.0, 180.0]))
    assert np.isclose(oi_get(oi, "psf angle step"), 180.0)
    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.5e-3]))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.array([0.0, 1.5]))
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([550.0]))
    assert oi_get(oi, "raytrace optics name") == "Synthetic RT"


def test_oi_get_set_raytrace_psf_metadata_before_compute(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "psf image heights", np.array([0.0, 1.0e-3, 2.0e-3], dtype=float))
    oi = oi_set(oi, "psf wavelength", np.array([450.0, 550.0], dtype=float))
    oi = oi_set(oi, "raytrace optics name", "Manual RT")

    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.0e-3, 2.0e-3]))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.array([0.0, 1.0, 2.0]))
    assert oi_get(oi, "psf image heights n") == 3
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([450.0, 550.0]))
    assert oi_get(oi, "psf wavelength n") == 2
    assert oi_get(oi, "raytrace optics name") == "Manual RT"


def test_oi_compute_reuses_matlab_style_psf_struct(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)
    exported = dict(oi_get(baseline, "psf struct"))
    original_cells = np.asarray(exported["psf"], dtype=object)

    delta_cells = np.empty(original_cells.shape, dtype=object)
    for index in np.ndindex(delta_cells.shape):
        kernel = np.zeros_like(np.asarray(original_cells[index], dtype=float))
        kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
        delta_cells[index] = kernel

    exported["psf"] = delta_cells
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf struct", exported)

    assert np.allclose(oi_get(oi, "psf image heights"), np.asarray(exported["imgHeight"], dtype=float))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.asarray(exported["imgHeight"], dtype=float) * 1e3)

    result = oi_compute(oi, scene, crop=True)
    sampled = oi_get(result, "sampledRTpsf")

    assert np.allclose(np.asarray(sampled[0, 0, 0], dtype=float), np.asarray(delta_cells[0, 0, 0], dtype=float))
    assert np.allclose(oi_get(result, "psf image heights"), np.asarray(exported["imgHeight"], dtype=float))
    assert not np.allclose(result.data["photons"], baseline.data["photons"])


def test_oi_set_raw_raytrace_psf_metadata_updates_optics_and_invalidates_cache(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    assert oi_get(oi, "psf struct") is not None

    oi = oi_set(oi, "rtpsfspacing", np.array([0.0005, 0.00075], dtype=float))
    oi = oi_set(oi, "rtpsffieldheight", np.array([0.0, 0.5, 1.0], dtype=float))
    oi = oi_set(oi, "rtpsfwavelength", np.array([500.0, 600.0], dtype=float))

    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "psf image heights").size == 0
    assert oi_get(oi, "psf wavelength").size == 0
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.5, 0.75]))
    assert np.allclose(oi_get(oi, "rtpsffieldheight", "mm"), np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(oi_get(oi, "rtpsfwavelength"), np.array([500.0, 600.0]))


def test_oi_set_raw_raytrace_psf_data_supports_indexed_updates(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    other = np.asarray(oi_get(oi, "rtpsfdata", 0.0, 625.0), dtype=float)
    replacement = np.full_like(np.asarray(oi_get(oi, "rtpsfdata", 0.0, 550.0), dtype=float), 7.0)

    oi = oi_set(oi, "rtpsfdata", replacement, 0.0, 550.0)

    assert np.allclose(oi_get(oi, "rtpsfdata", 0.0, 550.0), replacement)
    assert np.allclose(oi_get(oi, "rtpsfdata", 0.0, 625.0), other)
    assert oi_get(oi, "optics rtpsfsize") == oi_get(oi, "rtpsf")["function"].shape


def test_oi_set_raw_raytrace_geometry_and_relillum_updates_tables(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    geometry_function = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    rel_illum_function = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=float)

    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": geometry_function,
        },
    )
    oi = oi_set(
        oi,
        "rtrelillum",
        {
            "fieldHeight": np.array([0.0, 0.25], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": rel_illum_function,
        },
    )

    assert np.allclose(oi_get(oi, "rtgeomfieldheight", "mm"), np.array([0.0, 0.5]))
    assert np.array_equal(oi_get(oi, "rtgeomwavelength"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "rtgeomfunction", 600.0, "mm"), geometry_function[:, 1])
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), np.array([0.0, 0.25]))
    assert np.array_equal(oi_get(oi, "rtriwavelength"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "rtrifunction"), rel_illum_function)


def test_oi_set_raw_raytrace_geometry_supports_wavelength_index_updates(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    baseline_475 = np.asarray(oi_get(oi, "rtgeomfunction", 475.0, "mm"), dtype=float)
    replacement_550 = np.linspace(0.0, 2.0, baseline_475.size, dtype=float)

    oi = oi_set(oi, "rtgeomfunction", replacement_550, 550.0)

    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0, "mm"), replacement_550)
    assert np.allclose(oi_get(oi, "rtgeomfunction", 475.0, "mm"), baseline_475)


def test_oi_set_raw_raytrace_scalar_metadata_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "rtname", "Custom RT")
    oi = oi_set(oi, "rtopticsprogram", "Code V")
    oi = oi_set(oi, "rtlensfile", "custom.seq")
    oi = oi_set(oi, "rtefff#", 3.2)
    oi = oi_set(oi, "rtfnumber", 3.5)
    oi = oi_set(oi, "rtmag", -0.25)
    oi = oi_set(oi, "rtrefwave", 520.0)
    oi = oi_set(oi, "rtrefobjdist", 1.5)
    oi = oi_set(oi, "rtmaxfov", 25.0)
    oi = oi_set(oi, "rtefl", 0.004)
    oi = oi_set(oi, "rtcomputespacing", 2e-6)

    assert oi_get(oi, "rtname") == "Custom RT"
    assert oi_get(oi, "raytrace optics name") == "Custom RT"
    assert oi_get(oi, "rtopticsprogram") == "Code V"
    assert oi_get(oi, "rtlensfile") == "custom.seq"
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 3.2)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.5)
    assert np.isclose(oi_get(oi, "fnumber"), 3.5)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.25)
    assert np.isclose(oi_get(oi, "rtrefwave"), 520.0)
    assert np.isclose(oi_get(oi, "rtobjdist"), 1.5)
    assert np.isclose(oi_get(oi, "rtobjdist", "mm"), 1500.0)
    assert np.isclose(oi_get(oi, "rtfov"), 25.0)
    assert np.isclose(oi_get(oi, "rtefl"), 0.004)
    assert np.isclose(oi_get(oi, "focal length"), 0.004)
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 2e-6)
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 2.0)


def test_oi_get_set_optics_prefixed_raytrace_parameters(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    geometry_update = np.linspace(0.0, 1.0, oi_get(oi, "rtgeomfieldheight").size, dtype=float)

    assert np.isclose(oi_get(oi, "optics rtfnumber"), oi_get(oi, "rtfnumber"))
    assert np.allclose(oi_get(oi, "optics rtpsfspacing", "um"), oi_get(oi, "rtpsfspacing", "um"))
    assert np.allclose(oi_get(oi, "optics rtgeomfieldheight", "mm"), oi_get(oi, "rtgeomfieldheight", "mm"))

    oi = oi_set(oi, "optics rtrefwave", 530.0)
    oi = oi_set(oi, "optics rtpsfspacing", np.array([0.0004, 0.0006], dtype=float))
    oi = oi_set(oi, "optics rtcomputespacing", 3e-6)
    oi = oi_set(oi, "optics rtgeomfunction", geometry_update, 400.0)

    assert np.isclose(oi_get(oi, "rtrefwave"), 530.0)
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.4, 0.6]))
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 3.0)
    assert np.allclose(oi_get(oi, "rtgeomfunction", 400.0, "mm"), geometry_update)


def test_oi_set_whole_raytrace_struct_normalizes_and_invalidates_cache(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    raytrace = {
        "name": "Whole RT",
        "program": "Code V",
        "lensFile": "whole_rt.seq",
        "effectiveFNumber": 2.8,
        "fNumber": 3.1,
        "referenceWavelength": 600.0,
        "objectDistance": 1500.0,
        "mag": -0.2,
        "effectiveFocalLength": 5.0,
        "maxfov": 12.0,
        "geometry": {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        },
        "relIllum": {
            "fieldHeight": np.array([0.0, 0.25], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[1.0, 0.9], [0.8, 0.7]], dtype=float),
        },
        "psf": {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "sampleSpacing": np.array([0.0004, 0.0006], dtype=float),
            "function": np.ones((3, 3, 2, 2), dtype=float),
        },
        "computation": {"psfSpacing": 4e-6},
    }

    oi = oi_set(oi, "raytrace", raytrace)

    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "psf image heights").size == 0
    assert oi_get(oi, "rtname") == "Whole RT"
    assert oi_get(oi, "rtopticsprogram") == "Code V"
    assert oi_get(oi, "rtlensfile") == "whole_rt.seq"
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 2.8)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.1)
    assert np.isclose(oi_get(oi, "rtrefwave"), 600.0)
    assert np.isclose(oi_get(oi, "rtobjdist"), 1.5)
    assert np.isclose(oi_get(oi, "rtefl"), 0.005)
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), np.array([0.0, 0.25]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.4, 0.6]))
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 4.0)


def test_oi_set_whole_optics_struct_normalizes_raytrace_payload(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    optics = {
        "name": "Whole Optics",
        "fNumber": 2.9,
        "focalLength": 0.006,
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.8, 0.9], dtype=float),
        },
        "rayTrace": {
            "name": "Whole Optics RT",
            "fNumber": 2.9,
            "effectiveFNumber": 2.6,
            "effectiveFocalLength": 6.0,
            "referenceWavelength": 550.0,
            "psf": {
                "fieldHeight": np.array([0.0], dtype=float),
                "wavelength": np.array([550.0], dtype=float),
                "sampleSpacing": np.array([0.0007, 0.0007], dtype=float),
                "function": np.ones((3, 3, 1, 1), dtype=float),
            },
        },
    }

    oi = oi_set(oi, "optics", optics)

    assert oi_get(oi, "model") == "raytrace"
    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "raytrace optics name") == "Whole Optics RT"
    assert np.isclose(oi_get(oi, "fnumber"), 2.9)
    assert np.isclose(oi_get(oi, "focal length"), 0.006)
    assert np.array_equal(oi_get(oi, "transmittance wave"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "transmittance"), np.array([0.85]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.7, 0.7]))


def test_oi_get_optics_roundtrips_matlab_style_raytrace_struct(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["fNumber"] = 3.4
    optics["rayTrace"]["referenceWavelength"] = 610.0

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.4)
    assert np.isclose(oi_get(oi, "rtrefwave"), 610.0)
    assert roundtrip["rayTrace"]["referenceWavelength"] == 610.0
    assert "raytrace" not in roundtrip


def test_oi_compute_raytrace_rotates_psf_with_field_angle(asset_store) -> None:
    wave = np.array([550.0], dtype=float)
    scene = scene_create("uniform ee", 96, wave, asset_store=asset_store)
    scene = scene_set(scene, "fov", 10.0)
    photons = np.zeros_like(scene.data["photons"], dtype=float)
    center = photons.shape[0] // 2
    offset = 18
    photons[center, center + offset, 0] = 1.0
    photons[center - offset, center, 0] = 1.0
    scene = scene_set(scene, "photons", photons)

    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30)
    raytrace = oi_compute(oi, scene, crop=True)
    plane = np.asarray(raytrace.data["photons"][:, :, 0], dtype=float)

    radius = 4
    right_patch = plane[center - radius : center + radius + 1, center + offset - radius : center + offset + radius + 1]
    top_patch = plane[center - offset - radius : center - offset + radius + 1, center - radius : center + radius + 1]
    right_patch = right_patch / max(float(np.sum(right_patch)), 1e-12)
    top_patch = top_patch / max(float(np.sum(top_patch)), 1e-12)

    plain_error = float(np.mean((top_patch - right_patch) ** 2))
    rotated_error = min(float(np.mean((top_patch - np.rot90(right_patch, k)) ** 2)) for k in (1, 3))

    assert rotated_error < plain_error


def test_rt_psf_interp_matches_raw_psf_without_resampling() -> None:
    oi = oi_create("ray trace")
    kernel = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": np.stack((kernel, np.zeros_like(kernel)), axis=2)[:, :, :, None],
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    interp = rt_psf_interp(oi, field_height_m=0.0, wavelength_nm=550.0)

    assert np.allclose(interp, kernel)


def test_rt_psf_interp_interpolates_field_height_and_rotates() -> None:
    oi = oi_create("ray trace")
    psf_stack = np.zeros((5, 5, 2, 1), dtype=float)
    psf_stack[1, 2, 0, 0] = 1.0
    psf_stack[2, 3, 1, 0] = 1.0
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": psf_stack,
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    interp = rt_psf_interp(oi, field_height_m=0.5e-3, wavelength_nm=550.0)
    rotated = rt_psf_interp(oi, field_height_m=0.5e-3, field_angle_deg=90.0, wavelength_nm=550.0)

    expected = 0.5 * psf_stack[:, :, 0, 0] + 0.5 * psf_stack[:, :, 1, 0]
    assert np.allclose(interp, expected)
    assert np.allclose(rotated, np.rot90(expected))


def test_rt_psf_interp_resamples_to_requested_grid() -> None:
    oi = oi_create("ray trace")
    kernel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": kernel[:, :, None, None],
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    grid = np.array([0.0, 1.0e-4], dtype=float)
    x_grid, y_grid = np.meshgrid(grid, grid)
    resampled = rt_psf_interp(oi.fields["optics"], field_height_m=0.0, wavelength_nm=550.0, x_grid_m=x_grid, y_grid_m=y_grid)

    assert resampled.shape == x_grid.shape
    assert np.isclose(resampled[0, 0], kernel[1, 1])
    assert np.isclose(resampled[1, 1], kernel[2, 2])


def test_rt_di_interp_and_rt_ri_interp_use_nearest_wavelength() -> None:
    oi = oi_create("ray trace")
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[1.0, 10.0], [2.0, 20.0]], dtype=float),
        },
    )
    oi = oi_set(
        oi,
        "rtrelillum",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[0.5, 0.9], [0.25, 0.8]], dtype=float),
        },
    )

    assert np.array_equal(rt_di_interp(oi, 540.0), np.array([1.0, 2.0]))
    assert np.array_equal(rt_di_interp(oi.fields["optics"], 580.0), np.array([10.0, 20.0]))
    assert np.array_equal(rt_ri_interp(oi, 520.0), np.array([0.5, 0.25]))
    assert np.array_equal(rt_ri_interp(oi.fields["optics"], 590.0), np.array([0.9, 0.8]))


def test_oi_compute_raytrace_uses_raw_curve_helpers(asset_store, monkeypatch) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)

    di_calls: list[float] = []
    ri_calls: list[float] = []
    original_di = optics_module.rt_di_interp
    original_ri = optics_module.rt_ri_interp

    def record_di(optics: object, wavelength_nm: float) -> np.ndarray:
        di_calls.append(float(wavelength_nm))
        return original_di(optics, wavelength_nm)

    def record_ri(optics: object, wavelength_nm: float) -> np.ndarray:
        ri_calls.append(float(wavelength_nm))
        return original_ri(optics, wavelength_nm)

    monkeypatch.setattr(optics_module, "rt_di_interp", record_di)
    monkeypatch.setattr(optics_module, "rt_ri_interp", record_ri)

    result = oi_compute(oi, scene, crop=True)

    assert result.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert di_calls == [550.0]
    assert ri_calls == [550.0]


def test_rt_sample_heights_matches_matlab_truncation_rule() -> None:
    all_heights = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    data_height = np.array([[0.1, 0.9], [1.4, 1.6]], dtype=float)

    img_height, max_data_height = rt_sample_heights(all_heights, data_height)

    assert np.array_equal(img_height, np.array([0.0, 0.5, 1.0, 2.0]))
    assert np.isclose(max_data_height, 1.6)


def test_rt_block_center_matches_matlab_formula() -> None:
    center = rt_block_center(2, 3, np.array([8, 16], dtype=int))

    assert np.allclose(center, np.array([12.0, 40.0]))


def test_rt_extract_block_returns_matlab_style_indices() -> None:
    plane = np.arange(1, 1 + 6 * 8, dtype=float).reshape(6, 8)

    block, r_list, c_list = rt_extract_block(plane, np.array([2, 3], dtype=int), 2, 2)

    assert np.array_equal(r_list, np.array([3, 4]))
    assert np.array_equal(c_list, np.array([4, 5, 6]))
    assert np.array_equal(block, plane[2:4, 3:6])


def test_rt_insert_block_adds_filtered_data_at_matlab_block_origin() -> None:
    img = np.zeros((8, 10), dtype=float)
    filtered = np.ones((4, 5), dtype=float)

    inserted = rt_insert_block(img, filtered, np.array([2, 3], dtype=int), np.array([1, 1], dtype=int), 2, 2)

    expected = np.zeros_like(img)
    expected[2:6, 3:8] = 1.0
    assert np.array_equal(inserted, expected)


def test_rt_choose_block_size_matches_upstream_formula(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    n_blocks, block_samples, irrad_padding = rt_choose_block_size(scene, oi)

    field_heights = np.asarray(oi_get(oi, "rtgeometryfieldheight", "mm"), dtype=float)
    diagonal_mm = float(oi_get(oi, "diagonal")) * 1e3 / 2.0
    n_heights = int(np.argmin(np.abs(field_heights - diagonal_mm))) + 1
    expected_n_blocks = 4 * n_heights + 1
    expected_block_samples = np.array(
        [
            max(1, int(2 ** np.ceil(np.log2(max(scene_get(scene, "rows") / expected_n_blocks, 1.0))))),
            max(1, int(2 ** np.ceil(np.log2(max(scene_get(scene, "cols") / expected_n_blocks, 1.0))))),
        ],
        dtype=int,
    )
    expected_padding = np.ceil(
        (np.array(
            [
                expected_n_blocks * expected_block_samples[0] - scene_get(scene, "rows"),
                expected_n_blocks * expected_block_samples[1] - scene_get(scene, "cols"),
            ],
            dtype=float,
        ))
        / 2.0
    ).astype(int)

    assert n_blocks == expected_n_blocks
    assert np.array_equal(block_samples, expected_block_samples)
    assert np.array_equal(irrad_padding, expected_padding)


def test_rt_choose_block_size_uses_public_field_height_indexing(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    field_heights = np.asarray(oi_get(oi, "rtgeometryfieldheight", "mm"), dtype=float)
    diagonal_mm = float(oi_get(oi, "diagonal")) * 1e3 / 2.0

    n_blocks, _, _ = rt_choose_block_size(scene, oi)

    assert n_blocks == 4 * int(ie_field_height_to_index(field_heights, diagonal_mm)) + 1


def test_rt_otf_returns_padded_filtered_cube(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    result = rt_otf(scene, stage)
    n_blocks, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2
    expected_shape = (
        int(n_blocks * block_samples[0] + 2 * block_padding[0]),
        int(n_blocks * block_samples[1] + 2 * block_padding[1]),
        1,
    )

    assert result.shape == expected_shape
    assert np.all(result >= 0.0)
    assert np.sum(result) > 0.0


def test_rt_otf_uses_rt_blocks_per_field_height_setting(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    stage = oi_set(stage, "rt blocks per field height", 6)

    result = rt_otf(scene, stage)
    n_blocks, block_samples, _ = rt_choose_block_size(scene, stage, steps_fh=6)
    block_padding = block_samples // 2
    expected_shape = (
        int(n_blocks * block_samples[0] + 2 * block_padding[0]),
        int(n_blocks * block_samples[1] + 2 * block_padding[1]),
        1,
    )

    assert oi_get(stage, "rt blocks per field height") == 6
    assert result.shape == expected_shape


def test_rt_filtered_block_support_matches_block_layout(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    _, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2

    block_x, block_y, mm_row, mm_col = rt_filtered_block_support(stage, block_samples, block_padding)

    assert block_x.ndim == 1
    assert block_y.ndim == 1
    assert block_x.size == int(block_samples[1] + 2 * block_padding[1])
    assert block_y.size == int(block_samples[0] + 2 * block_padding[0])
    assert np.isclose(np.diff(block_x)[0], mm_col)
    assert np.isclose(np.diff(block_y)[0], mm_row)
    assert np.any(np.isclose(block_x, 0.0))
    assert np.any(np.isclose(block_y, 0.0))


def test_rt_otf_filtered_block_support_matches_public_helper(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    _, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2

    block_x, block_y, mm_row, mm_col = rt_filtered_block_support(stage, block_samples, block_padding)
    result = rt_otf(scene, stage)

    expected_rows = int(block_samples[0] + 2 * block_padding[0])
    expected_cols = int(block_samples[1] + 2 * block_padding[1])
    assert block_y.size == expected_rows
    assert block_x.size == expected_cols
    assert mm_row > 0.0
    assert mm_col > 0.0
    assert result.shape[0] >= expected_rows
    assert result.shape[1] >= expected_cols


def test_rt_synthetic_builds_normalized_raytrace_optics(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    optics = rt_synthetic(oi, spread_limits=(3.0, 5.0), xy_ratio=0.3)

    assert optics["model"] == "raytrace"
    assert optics["name"] == "Synthetic Gaussian"
    assert np.array_equal(optics["transmittance"]["wave"], np.array([450.0, 550.0, 650.0]))
    assert optics["raytrace"]["program"] == "Zemax"
    assert optics["raytrace"]["lens_file"] == "Synthetic Gaussian"
    assert optics["raytrace"]["psf"]["function"].shape[0:2] == (128, 128)
    assert optics["raytrace"]["psf"]["function"].shape[2] == 21
    assert optics["raytrace"]["psf"]["function"].shape[3] == 3
    assert np.allclose(np.sum(optics["raytrace"]["psf"]["function"][:, :, 0, 0]), 1.0)


def test_oi_compute_accepts_rt_synthetic_optics(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "optics", rt_synthetic(oi, spread_limits=(2.0, 3.0), xy_ratio=0.5))

    result = oi_compute(oi, scene, crop=True)

    assert result.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert oi_get(result, "raytrace optics name") == "Synthetic Gaussian"
    assert oi_get(result, "sampledRTpsf") is not None


def test_rt_otf_runs_with_rt_synthetic_optics(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "optics", rt_synthetic(oi, spread_limits=(2.0, 4.0), xy_ratio=0.25))
    stage = rt_geometry(oi, scene)

    result = rt_otf(scene, stage)

    assert result.ndim == 3
    assert result.shape[2] == 1
    assert np.sum(result) > 0.0


def test_rt_file_names_matches_zemax_naming(tmp_path) -> None:
    di_name, ri_name, psf_name_list, cra_name = rt_file_names(
        "CookeLens.ZMX",
        np.array([500.0, 600.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        directory=tmp_path,
    )

    assert di_name.endswith("CookeLens_DI_.dat")
    assert ri_name.endswith("CookeLens_RI_.dat")
    assert cra_name.endswith("CookeLens_CRA_.dat")
    assert psf_name_list.shape == (2, 2)
    assert str(psf_name_list[1, 1]).endswith("CookeLens_2D_PSF_Fld2_Wave2.dat")


def test_rt_file_names_normalizes_windows_style_lens_paths(tmp_path) -> None:
    di_name, ri_name, psf_name_list, cra_name = rt_file_names(
        r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX",
        np.array([500.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        directory=tmp_path,
    )

    assert di_name.endswith("CookeLens_DI_.dat")
    assert ri_name.endswith("CookeLens_RI_.dat")
    assert cra_name.endswith("CookeLens_CRA_.dat")
    assert str(psf_name_list[1, 0]).endswith("CookeLens_2D_PSF_Fld2_Wave1.dat")


def test_zemax_read_header_and_load_parse_text_output(tmp_path) -> None:
    psf_file = tmp_path / "Test_2D_PSF_Fld1_Wave1.dat"
    psf_file.write_text(
        "spacing is 0.5000 microns\n"
        "area is 1.0000 microns\n"
        "normalized.\n"
        "1 2 3 4\n",
        encoding="latin1",
    )

    spacing_um, area_um = zemax_read_header(psf_file)
    kernel = zemax_load(psf_file, 2)

    assert np.isclose(spacing_um, 0.5)
    assert np.isclose(area_um, 1.0)
    assert np.array_equal(kernel, np.array([[2.0, 4.0], [1.0, 3.0]], dtype=float))


def test_rt_import_data_builds_usable_raytrace_optics(tmp_path, asset_store) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)
    oi = oi_create("ray trace", imported_optics, asset_store=asset_store)
    scene = scene_create("uniform ee", 16, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi, scene)

    assert optics_file is None
    assert imported_optics["model"] == "raytrace"
    assert imported_optics["raytrace"]["program"] == "zemax"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["relative_illumination"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)
    assert np.allclose(imported_optics["raytrace"]["psf"]["sample_spacing_mm"], np.array([0.0005, 0.0005]))
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 2.5e-7)
    assert stage.data["photons"].shape == scene.data["photons"].shape


def test_rt_import_data_preserves_requested_program_label(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=params_file, rt_program="ZeMaX")

    assert optics_file is None
    assert imported_optics["raytrace"]["program"] == "ZeMaX"


def test_rt_import_data_normalizes_windows_style_base_lens_paths(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        lens_file=r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX",
        base_lens_file_name=r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_multiline_wave_vector_with_continuation(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500 ...\n 600]",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["geometry"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["raytrace"]["psf"]["wavelength_nm"], np.array([500.0, 600.0]))


def test_rt_import_data_parses_column_vector_wave_syntax(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500; 600]",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["relative_illumination"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_transposed_row_vector_wave_syntax(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500 600]'",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["geometry"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_legacy_base_lens_line_without_semicolon(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        base_lens_has_semicolon=False,
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == "CookeLens.ZMX"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)


def test_rt_import_data_rejects_odd_psf_size_from_isetparams(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_size_assignment=3,
    )

    with pytest.raises(ValueError, match="PSF size must be even"):
        rt_import_data(p_file_full=params_file)


def test_rt_import_data_accepts_bundle_directory(tmp_path) -> None:
    _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=tmp_path)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == "CookeLens.ZMX"
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_accepts_legacy_isetparms_filename(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        params_file_name="ISETPARMS.TXT",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=tmp_path)

    assert optics_file is None
    assert params_file.name == "ISETPARMS.TXT"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)


def test_rt_import_data_preserves_existing_optics_fields_and_effective_top_level_state(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "name": "Existing Optics",
        "compute_method": "customrt",
        "aberration_scale": 0.25,
        "offaxis_method": "cos4th",
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.7, 0.8], dtype=float),
        },
        "focal_length_m": 0.123,
        "f_number": 9.9,
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["compute_method"] == "customrt"
    assert np.isclose(imported_optics["aberration_scale"], 0.25)
    assert imported_optics["offaxis_method"] == "cos4th"
    assert np.array_equal(imported_optics["transmittance"]["wave"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["transmittance"]["scale"], np.array([0.7, 0.8]))
    assert np.isclose(imported_optics["focal_length_m"], 0.006)
    assert np.isclose(imported_optics["f_number"], 1.8)
    assert np.isclose(imported_optics["raytrace"]["f_number"], 2.0)
    assert np.isclose(imported_optics["raytrace"]["effective_f_number"], 1.8)


def test_rt_import_data_preserves_existing_compute_spacing_when_bundle_omits_it(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_spacing_assignment_mm=None,
    )
    existing = {
        "name": "Existing Optics",
        "raytrace": {
            "computation": {
                "psf_spacing_m": 7.5e-6,
            },
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 7.5e-6)


def test_oi_create_raytrace_accepts_isetparams_file(tmp_path, asset_store) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    oi = oi_create("ray trace", params_file, asset_store=asset_store)
    exported = oi_get(oi, "optics")

    assert oi.fields["optics"]["model"] == "raytrace"
    assert np.isclose(oi.fields["optics"]["f_number"], 1.8)
    assert oi_get(oi, "rtlensfile") == "CookeLens.ZMX"
    assert np.isclose(oi_get(oi, "fnumber"), 2.0)
    assert np.isclose(exported["fNumber"], 2.0)
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 0.25)
    assert oi.fields["optics"]["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_oi_create_raytrace_accepts_isetparams_directory(tmp_path, asset_store) -> None:
    _write_mock_zemax_bundle(tmp_path)

    oi = oi_create("ray trace", tmp_path, asset_store=asset_store)

    assert oi.fields["optics"]["model"] == "raytrace"
    assert oi_get(oi, "raytraceopticsname") == "CookeLens"


def test_oi_create_raytrace_accepts_legacy_isetparms_directory(tmp_path, asset_store) -> None:
    _write_mock_zemax_bundle(
        tmp_path,
        params_file_name="ISETPARMS.TXT",
    )

    oi = oi_create("ray trace", tmp_path, asset_store=asset_store)

    assert oi.fields["optics"]["model"] == "raytrace"
    assert oi_get(oi, "raytraceopticsname") == "CookeLens"


def test_rt_psf_grid_matches_oi_sample_spacing(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    x_grid, y_grid, sample_spacing = rt_psf_grid(oi, "m")

    assert x_grid.shape == y_grid.shape
    assert x_grid.ndim == 2
    assert sample_spacing.shape == (2,)
    assert np.isclose(np.diff(x_grid[0])[0], sample_spacing[0])
    assert np.isclose(np.diff(y_grid[:, 0])[0], sample_spacing[1])
    assert np.any(np.isclose(x_grid, 0.0))
    assert np.any(np.isclose(y_grid, 0.0))


def test_rt_angle_lut_returns_matlab_style_indices(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 45.0)
    oi = oi_compute(oi, scene, crop=True)

    angle_lut = rt_angle_lut(oi)

    assert angle_lut.shape == (360, 2)
    assert np.all(angle_lut[:, 0] >= 1)
    assert np.all(angle_lut[:, 0] <= len(oi_get(oi, "psf sample angles")) - 1)
    assert np.all(angle_lut[:, 1] >= 0.0)
    assert np.all(angle_lut[:, 1] <= 1.0)


def test_rt_geometry_returns_uncropped_raytrace_stage(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    assert stage.data["photons"].shape == scene.data["photons"].shape
    assert np.allclose(oi_get(stage, "depth map"), 2.0)
    assert oi_get(stage, "padding pixels") == (0, 0)
    assert np.allclose(
        oi_get(stage, "samplespacing"),
        np.array([oi_get(stage, "hspatialresolution"), oi_get(stage, "wspatialresolution")], dtype=float),
    )


def test_rt_precompute_psf_returns_matlab_style_struct(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30.0), scene)
    psf_struct = rt_precompute_psf(stage)

    assert isinstance(psf_struct, dict)
    assert "psf" in psf_struct
    assert "sampAngles" in psf_struct
    assert "imgHeight" in psf_struct
    assert "wavelength" in psf_struct
    assert np.asarray(psf_struct["psf"], dtype=object).dtype == object
    assert np.array_equal(psf_struct["wavelength"], np.array([550.0]))


def test_rt_precompute_psf_apply_matches_oi_compute_uncropped(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    stage = oi_set(stage, "psf struct", rt_precompute_psf(stage))
    applied = rt_precompute_psf_apply(stage)

    assert applied.data["photons"].shape == baseline.data["photons"].shape
    assert applied.fields["padding_pixels"] == baseline.fields["padding_pixels"]
    assert np.allclose(applied.data["photons"], baseline.data["photons"])
    assert np.allclose(oi_get(applied, "depth map"), oi_get(baseline, "depth map"))


def test_rt_psf_apply_matches_rt_precompute_psf_apply(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    legacy = rt_psf_apply(stage)
    cached = rt_precompute_psf_apply(stage)

    assert legacy.data["photons"].shape == cached.data["photons"].shape
    assert legacy.fields["padding_pixels"] == cached.fields["padding_pixels"]
    assert np.allclose(legacy.data["photons"], cached.data["photons"])
    assert np.allclose(oi_get(legacy, "depth map"), oi_get(cached, "depth map"))


def test_rt_psf_apply_uses_explicit_angle_step(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    result = rt_psf_apply(stage, angle_step_deg=30.0)

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert np.array_equal(oi_get(result, "psf sample angles"), np.arange(0.0, 361.0, 30.0))


def test_optics_ray_trace_matches_oi_compute_uncropped(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)
    result = optics_ray_trace(scene, oi_create("ray trace", asset_store=asset_store))

    assert result.data["photons"].shape == baseline.data["photons"].shape
    assert result.fields["padding_pixels"] == baseline.fields["padding_pixels"]
    assert np.allclose(result.data["photons"], baseline.data["photons"])
    assert np.allclose(oi_get(result, "depth map"), oi_get(baseline, "depth map"))
    assert result.fields["illuminance"].shape == result.data["photons"].shape[:2]
    assert np.allclose(result.fields["illuminance"], oi_get(baseline, "illuminance"))
    assert np.isclose(result.fields["mean_illuminance"], oi_get(result, "mean illuminance"))


def test_optics_ray_trace_uses_explicit_angle_step(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    result = optics_ray_trace(scene, oi_create("ray trace", asset_store=asset_store), angle_step_deg=30.0)

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert np.array_equal(oi_get(result, "psf sample angles"), np.arange(0.0, 361.0, 30.0))


def test_oi_calculate_illuminance_updates_cached_fields(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    illuminance, mean_illuminance, mean_comp_illuminance = oi_calculate_illuminance(oi)

    assert np.allclose(illuminance, oi_get(oi, "illuminance"))
    assert np.isclose(mean_illuminance, oi_get(oi, "mean illuminance"))
    assert np.isclose(mean_comp_illuminance, oi_get(oi, "mean comp illuminance"))
    assert np.isclose(mean_comp_illuminance, 0.0)


def test_oi_diffuser_blurs_photons_and_returns_kernel(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    baseline = np.asarray(oi.data["photons"], dtype=float).copy()

    oi, sd, blur_filter = oi_diffuser(oi, 2.0)

    assert np.isclose(float(sd), 2.0)
    assert blur_filter.ndim == 2
    assert np.isclose(np.sum(blur_filter), 1.0)
    assert oi.data["photons"].shape == baseline.shape
    assert not np.allclose(oi.data["photons"], baseline)
    assert np.allclose(oi.fields["illuminance"], oi_get(oi, "illuminance"))


def test_optics_ray_trace_blur_matches_public_oi_diffuser(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    blur_m = 2e-6

    manual = rt_precompute_psf_apply(rt_geometry(oi_create("ray trace", asset_store=asset_store), scene))
    manual, _, _ = oi_diffuser(manual, blur_m * 1e6)

    raytrace_oi = oi_set(oi_create("ray trace", asset_store=asset_store), "diffuser method", "blur")
    raytrace_oi = oi_set(raytrace_oi, "diffuser blur", blur_m)
    wrapped = optics_ray_trace(scene, raytrace_oi)

    assert wrapped.data["photons"].shape == manual.data["photons"].shape
    assert np.allclose(wrapped.data["photons"], manual.data["photons"])
    assert np.allclose(oi_get(wrapped, "illuminance"), oi_get(manual, "illuminance"))


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


def test_sensor_create_rgbw_and_rccc_presets_expose_multichannel_cfas(asset_store) -> None:
    rgbw = sensor_create("rgbw", asset_store=asset_store)
    rccc = sensor_create("rccc", asset_store=asset_store)

    assert sensor_get(rgbw, "nfilters") == 4
    assert sensor_get(rgbw, "filter color letters") == "rgbw"
    assert np.array_equal(sensor_get(rgbw, "pattern colors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert sensor_get(rccc, "nfilters") == 2
    assert sensor_get(rccc, "filter color letters") == "rw"
    assert np.array_equal(sensor_get(rccc, "pattern colors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


def test_sensor_create_vendor_models_load_upstream_rgbw_and_rccc_metadata(asset_store) -> None:
    mt9v024_rgbw = sensor_create("mt9v024", "rgbw", asset_store=asset_store)
    mt9v024_rccc = sensor_create("mt9v024", None, "rccc", asset_store=asset_store)
    ar0132at_rgbw = sensor_create("ar0132at", "rgbw", asset_store=asset_store)
    ar0132at_rccc = sensor_create("ar0132at", None, "rccc", asset_store=asset_store)

    assert mt9v024_rgbw.name == "MTV9V024-RGBW"
    assert mt9v024_rgbw.fields["size"] == (480, 752)
    assert np.allclose(mt9v024_rgbw.fields["pixel"]["size_m"], np.array([6e-6, 6e-6]))
    assert sensor_get(mt9v024_rgbw, "filter color letters") == "rgbw"
    assert np.array_equal(sensor_get(mt9v024_rgbw, "pattern colors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert mt9v024_rccc.name == "MTV9V024-RCCC"
    assert sensor_get(mt9v024_rccc, "filter color letters") == "rw"
    assert np.array_equal(sensor_get(mt9v024_rccc, "pattern colors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))

    assert ar0132at_rgbw.name == "AR0132AT-RGBW"
    assert ar0132at_rgbw.fields["size"] == (960, 1280)
    assert np.allclose(ar0132at_rgbw.fields["pixel"]["size_m"], np.array([3.751e-6, 3.751e-6]))
    assert sensor_get(ar0132at_rgbw, "filter color letters") == "rgbw"
    assert np.array_equal(sensor_get(ar0132at_rgbw, "pattern colors"), np.array([["r", "g"], ["w", "b"]], dtype="<U1"))

    assert ar0132at_rccc.name == "AR0132AT-RCCC"
    assert sensor_get(ar0132at_rccc, "filter color letters") == "rw"
    assert np.array_equal(sensor_get(ar0132at_rccc, "pattern colors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


def test_sensor_compute_supports_rgbw_and_rccc_presets(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    rgbw = sensor_set(sensor_create("rgbw", asset_store=asset_store), "noise flag", 0)
    rgbw = sensor_set(rgbw, "integration time", 0.01)
    rccc = sensor_set(sensor_create("rccc", asset_store=asset_store), "noise flag", 0)
    rccc = sensor_set(rccc, "integration time", 0.01)

    rgbw_result = sensor_compute(rgbw, oi, seed=0)
    rccc_result = sensor_compute(rccc, oi, seed=0)

    assert rgbw_result.data["volts"].shape == rgbw.fields["size"]
    assert rccc_result.data["volts"].shape == rccc.fields["size"]
    assert np.all(rgbw_result.data["volts"] >= 0.0)
    assert np.all(rccc_result.data["volts"] >= 0.0)


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


def test_run_python_case_with_context_returns_pipeline_objects(asset_store) -> None:
    case = run_python_case_with_context("camera_default_pipeline", asset_store=asset_store)

    assert case.payload["result"].shape[:2] == tuple(case.context["sensor"].fields["size"])
    assert np.array_equal(case.payload["oi_photons"], case.context["oi"].data["photons"])
    assert np.array_equal(case.payload["sensor_volts"], case.context["sensor"].data["volts"])
    assert case.context["camera"].fields["ip"] is case.context["ip"]
