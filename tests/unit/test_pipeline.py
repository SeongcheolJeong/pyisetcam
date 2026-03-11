from __future__ import annotations

import numpy as np
import pytest
import pyisetcam.optics as optics_module
from scipy.io import savemat

from pyisetcam.exceptions import UnsupportedOptionError
from pyisetcam.parity import run_python_case_with_context
from pyisetcam.utils import tile_pattern
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
    optics_psf_to_otf,
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
    si_synthetic,
    run_python_case,
    scene_create,
    scene_get,
    scene_set,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
    wvf_compute,
    wvf_create,
    wvf_defocus_diopters_to_microns,
    wvf_defocus_microns_to_diopters,
    wvf_get,
    wvf_set,
    wvf_to_oi,
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


def test_raytrace_struct_uses_normalized_keys_recognizes_blocks_per_field_height() -> None:
    assert optics_module._raytrace_struct_uses_normalized_keys({"blocks_per_field_height": 7}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"blocksPerFieldHeight": 7}) is False


def test_raytrace_struct_uses_normalized_keys_recognizes_scalar_normalized_fields() -> None:
    assert optics_module._raytrace_struct_uses_normalized_keys({"f_number": 3.1}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"magnification": -0.2}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"fNumber": 3.1}) is False
    assert optics_module._raytrace_struct_uses_normalized_keys({"mag": -0.2}) is False


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
    optics["rayTrace"]["blocksPerFieldHeight"] = 7

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.4)
    assert np.isclose(oi_get(oi, "rtrefwave"), 610.0)
    assert oi_get(oi, "rt blocks per field height") == 7
    assert roundtrip["rayTrace"]["referenceWavelength"] == 610.0
    assert roundtrip["rayTrace"]["blocksPerFieldHeight"] == 7
    assert "raytrace" not in roundtrip


def test_oi_set_optics_accepts_normalized_nested_raytrace_scalar_fields(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["raytrace"] = {
        "f_number": 3.7,
        "magnification": -0.42,
    }
    optics.pop("rayTrace", None)

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.7)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.42)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.7)
    assert np.isclose(roundtrip["rayTrace"]["mag"], -0.42)


def test_oi_set_optics_preserves_existing_raytrace_data_for_normalized_partial_update(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    original = oi_get(oi, "optics")
    original_geometry = np.asarray(original["rayTrace"]["geometry"]["function"], dtype=float).copy()
    original_psf_spacing = np.asarray(original["rayTrace"]["psf"]["sampleSpacing"], dtype=float).copy()

    optics = oi_get(oi, "optics")
    optics["raytrace"] = {
        "lens_file": "CustomLens.zmx",
        "computation": {"psf_spacing_m": 8e-6},
    }
    optics.pop("rayTrace", None)

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "rtlensfile") == "CustomLens.zmx"
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 8e-6)
    assert np.array_equal(roundtrip["rayTrace"]["geometry"]["function"], original_geometry)
    assert np.allclose(roundtrip["rayTrace"]["psf"]["sampleSpacing"], original_psf_spacing)
    assert np.isclose(roundtrip["rayTrace"]["referenceWavelength"], original["rayTrace"]["referenceWavelength"])


def test_oi_set_optics_preserves_existing_raytrace_data_for_raw_partial_update(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    original = oi_get(oi, "optics")
    original_geometry = np.asarray(original["rayTrace"]["geometry"]["function"], dtype=float).copy()
    original_psf_spacing = np.asarray(original["rayTrace"]["psf"]["sampleSpacing"], dtype=float).copy()

    optics = oi_get(oi, "optics")
    optics["rayTrace"] = {
        "lensFile": "CustomLens.zmx",
        "computation": {"psfSpacing": 8e-6},
    }

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "rtlensfile") == "CustomLens.zmx"
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 8e-6)
    assert np.array_equal(roundtrip["rayTrace"]["geometry"]["function"], original_geometry)
    assert np.allclose(roundtrip["rayTrace"]["psf"]["sampleSpacing"], original_psf_spacing)
    assert np.isclose(roundtrip["rayTrace"]["referenceWavelength"], original["rayTrace"]["referenceWavelength"])


def test_oi_set_optics_raw_nested_fnumber_overrides_exported_top_level_fnumber(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["rayTrace"] = {
        "fNumber": 3.3,
    }

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.3)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.3)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.3)


def test_oi_set_optics_top_level_focal_length_overrides_exported_nested_effective_focal_length(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["focalLength"] = 0.009

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "focal length"), 0.009)
    assert np.isclose(oi_get(oi, "rteffectivefocallength"), 0.009)
    assert np.isclose(roundtrip["rayTrace"]["effectiveFocalLength"], 9.0)


def test_oi_set_optics_normalized_top_level_f_number_updates_raytrace(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["f_number"] = 3.2

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.2)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.2)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.2)


def test_oi_set_optics_normalized_top_level_focal_length_updates_raytrace(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["focal_length_m"] = 0.008

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "focal length"), 0.008)
    assert np.isclose(oi_get(oi, "rteffectivefocallength"), 0.008)
    assert np.isclose(roundtrip["rayTrace"]["effectiveFocalLength"], 8.0)


def test_oi_set_optics_normalized_top_level_nominal_focal_length_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["nominal_focal_length_m"] = 0.012

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(roundtrip["nominalFocalLength"], 0.012)


def test_oi_set_optics_raw_top_level_nominal_focal_length_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["nominalFocalLength"] = 0.012

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(roundtrip["nominalFocalLength"], 0.012)


def test_oi_set_optics_raw_top_level_offaxis_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["offaxis"] = "skip"

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "offaxis method") == "skip"
    assert roundtrip["offaxis"] == "skip"


def test_oi_set_optics_transmittance_wave_preserves_scale_via_interpolation(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["transmittance"]["wave"] = np.array([450.0, 550.0, 650.0], dtype=float)
    optics["transmittance"]["scale"] = np.array([0.2, 0.5, 0.8], dtype=float)
    oi = oi_set(oi, "optics", optics)

    optics = oi_get(oi, "optics")
    optics["transmittance"]["wave"] = np.array([475.0, 625.0], dtype=float)
    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.array_equal(roundtrip["transmittance"]["wave"], np.array([475.0, 625.0]))
    assert np.allclose(roundtrip["transmittance"]["scale"], np.array([0.275, 0.725]))


def test_oi_set_optics_transmittance_scale_length_mismatch_raises(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["transmittance"]["scale"] = np.array([0.2, 0.5, 0.8], dtype=float)

    with pytest.raises(ValueError, match="Transmittance must match wave dimension."):
        oi_set(oi, "optics", optics)


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


def test_rt_import_data_preserves_existing_raytrace_name_independently(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "name": "Existing Optics",
        "raytrace": {
            "name": "Existing RT Name",
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"


def test_rt_import_data_preserves_raw_matlab_style_optics_fields(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_spacing_assignment_mm=None,
    )
    existing = {
        "name": "Existing Optics",
        "computeMethod": "customrt",
        "aberrationScale": 0.5,
        "offaxis": "cos4th",
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.6, 0.7], dtype=float),
        },
        "rayTrace": {
            "name": "Existing RT Name",
            "computation": {
                "psfSpacing": 9e-6,
            },
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["compute_method"] == "customrt"
    assert np.isclose(imported_optics["aberration_scale"], 0.5)
    assert imported_optics["offaxis_method"] == "cos4th"
    assert np.array_equal(imported_optics["transmittance"]["wave"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["transmittance"]["scale"], np.array([0.6, 0.7]))
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 9e-6)


def test_rt_import_data_uses_existing_raytrace_name_as_top_level_fallback(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "rayTrace": {
            "name": "Existing RT Name",
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing RT Name"
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"


def test_rt_import_data_preserves_existing_blocks_per_field_height(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "rayTrace": {
            "blocksPerFieldHeight": 7,
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["blocks_per_field_height"] == 7


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


def test_oi_create_raytrace_directory_without_bundle_stays_unsupported(tmp_path, asset_store) -> None:
    with pytest.raises(UnsupportedOptionError, match="ray trace optics"):
        oi_create("ray trace", tmp_path, asset_store=asset_store)


def test_oi_create_raytrace_directory_surfaces_malformed_bundle_errors(tmp_path, asset_store) -> None:
    (tmp_path / "ISETPARAMS.txt").write_text("lensFile='CookeLens.ZMX';\n", encoding="latin1")

    with pytest.raises(ValueError, match="Missing Zemax parameter"):
        oi_create("ray trace", tmp_path, asset_store=asset_store)


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


def test_wvf_set_and_get_named_zcoeffs_round_trip() -> None:
    wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
    wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])

    assert np.isclose(wvf_get(wvf, "zcoeffs", "defocus"), 2.0)
    assert np.isclose(wvf_get(wvf, "zcoeffs", "vertical_astigmatism"), 0.5)
    assert np.array_equal(wvf_get(wvf, "wave"), np.array([450.0, 550.0, 650.0], dtype=float))


def test_wvf_defocus_diopter_micron_round_trip() -> None:
    microns = wvf_defocus_diopters_to_microns(1.5, 4.0)
    diopters = wvf_defocus_microns_to_diopters(microns, 4.0)

    assert np.isclose(float(np.asarray(microns).reshape(-1)[0]), 1.5 * (4.0**2) / (16.0 * np.sqrt(3.0)))
    assert np.isclose(float(np.asarray(diopters).reshape(-1)[0]), 1.5)


def test_wvf_compute_returns_psf_and_pupil_function() -> None:
    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
    computed = wvf_compute(wvf)

    assert computed["computed"] is True
    assert computed["psf"].shape == (201, 201, 2)
    assert computed["pupil_function"].shape == (201, 201, 2)
    assert computed["pupil_amplitude"].shape == (201, 201, 2)
    assert computed["pupil_phase"].shape == (201, 201, 2)
    assert np.isclose(float(np.sum(computed["psf"][:, :, 0])), 1.0)
    assert np.isclose(float(wvf_get(computed, "pupil diameter", "mm")), 3.0)
    assert np.asarray(wvf_get(computed, "psf")).shape == (201, 201, 2)
    assert np.asarray(wvf_get(computed, "pupil function")).shape == (201, 201, 2)


def test_wvf_spatial_sampling_getters_match_spatial_model() -> None:
    wvf = wvf_create()
    wvf = wvf_set(wvf, "calc pupil diameter", 7.0 / 4.0, "mm")
    wvf = wvf_set(wvf, "focal length", 7e-3, "m")
    wvf = wvf_compute(wvf)

    wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
    n_pixels = int(wvf_get(wvf, "npixels"))
    pupil_plane_size_mm = float(wvf_get(wvf, "pupil plane size", "mm", wave))
    pupil_sample_spacing_mm = float(wvf_get(wvf, "pupil sample spacing", "mm", wave))
    pupil_positions_mm = np.asarray(wvf_get(wvf, "pupil positions", wave, "mm"), dtype=float)
    pupil_amplitude = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
    pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", wave), dtype=float)

    assert int(wvf_get(wvf, "calc nwave")) == int(np.asarray(wvf_get(wvf, "wave"), dtype=float).size)
    assert np.isclose(float(wvf_get(wvf, "psf sample spacing")), float(wvf_get(wvf, "ref psf sample interval")))
    assert np.isclose(pupil_sample_spacing_mm, pupil_plane_size_mm / n_pixels)
    assert pupil_positions_mm.shape == (n_pixels,)
    assert np.isclose(pupil_positions_mm[1] - pupil_positions_mm[0], pupil_sample_spacing_mm)
    assert pupil_amplitude.shape == (n_pixels, n_pixels)
    assert pupil_phase.shape == (n_pixels, n_pixels)


def test_oi_compute_accepts_wvf_input(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    wvf = wvf_create(wave=scene_get(scene, "wave"))
    wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])

    oi = oi_compute(wvf, scene, crop=True)

    assert oi.fields["optics"]["model"] == "shiftinvariant"
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "defocus")), 2.0)
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")), 0.5)
    assert oi_get(oi, "photons").shape[:2] == scene.data["photons"].shape[:2]
    assert wvf_to_oi(wvf).fields["optics"]["model"] == "shiftinvariant"


def test_oi_set_wvf_prefixed_parameter_rebuilds_oi(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 1.5)
    oi = oi_compute(oi_create("wvf"), scene, crop=True)

    original_wangular = float(oi_get(oi, "wangular"))
    updated = oi_set(oi, "wvf zcoeffs", 1.5, "defocus")

    assert np.isclose(float(oi_get(updated, "wvf zcoeffs", "defocus")), 1.5)
    assert np.isclose(float(oi_get(updated, "wvf pupil diameter", "mm")), float(oi_get(oi, "wvf pupil diameter", "mm")))
    assert np.isclose(float(oi_get(updated, "wangular")), original_wangular)
    assert updated.data == {}

    recomputed = oi_compute(updated, scene, crop=True)
    assert recomputed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_set_optics_wvf_rebuilds_oi_from_wavefront(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 1.5)
    oi = oi_compute(oi_create("wvf"), scene, crop=True)

    wavefront = wvf_set(oi_get(oi, "optics wvf"), "zcoeffs", 0.75, "defocus")
    updated = oi_set(oi, "optics wvf", wavefront)

    assert np.isclose(float(oi_get(updated, "wvf zcoeffs", "defocus")), 0.75)
    assert updated.data == {}

    recomputed = oi_compute(updated, scene, crop=True)
    assert recomputed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_create_psf_builds_default_shift_invariant_psf_optics() -> None:
    oi = oi_create("psf")

    assert oi.fields["optics"]["compute_method"] == "opticsotf"
    assert oi.fields["optics"]["model"] == "shiftinvariant"
    psf_data = oi_get(oi, "psfdata")
    assert psf_data is not None
    assert np.asarray(psf_data["psf"]).ndim == 3
    assert np.asarray(psf_data["wave"]).shape == (31,)
    assert np.allclose(np.asarray(psf_data["umPerSamp"], dtype=float), np.array([0.25, 0.25], dtype=float))
    assert np.asarray(oi.fields["optics"]["otf_data"]).shape == (129, 129, 31)


def test_oi_create_psf_accepts_custom_shift_invariant_psf_data(asset_store) -> None:
    psf = np.zeros((33, 33, 1), dtype=float)
    psf[16, 16, 0] = 1.0
    oi = oi_create(
        "psf",
        {
            "psf": psf,
            "wave": np.array([550.0], dtype=float),
            "umPerSamp": np.array([0.25, 0.25], dtype=float),
        },
    )
    stored = oi_get(oi, "psfdata")
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    computed = oi_compute(oi, scene, crop=True)

    assert np.asarray(stored["psf"]).shape == (33, 33, 1)
    assert np.array_equal(np.asarray(stored["wave"]), np.array([550.0], dtype=float))
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert computed.data["photons"].shape[2] == scene.data["photons"].shape[2]


def test_optics_psf_to_otf_builds_custom_otf_struct(asset_store) -> None:
    otf = optics_psf_to_otf(
        asset_store.resolve("data/optics/flare/flare1.png"),
        1.2e-6,
        np.arange(400.0, 701.0, 10.0, dtype=float),
    )

    assert otf["function"] == "custom"
    assert np.asarray(otf["OTF"]).shape[2] == 31
    assert np.asarray(otf["fx"]).ndim == 1
    assert np.asarray(otf["fy"]).ndim == 1
    assert np.isclose(float(np.abs(np.asarray(otf["OTF"])[0, 0, 0])), 1.0)


def test_oi_set_optics_otfstruct_supports_custom_shift_invariant_otf(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 40.0)
    otf = optics_psf_to_otf(
        asset_store.resolve("data/optics/flare/flare1.png"),
        1.2e-6,
        np.arange(400.0, 701.0, 10.0, dtype=float),
    )
    oi = oi_set(oi_create("shift invariant"), "optics otfstruct", otf)
    computed = oi_compute(oi, scene, crop=True)

    stored = oi_get(oi, "optics otfstruct")
    assert stored is not None
    assert oi.fields["optics"]["compute_method"] == "opticsotf"
    assert np.asarray(stored["OTF"]).shape == np.asarray(otf["OTF"]).shape
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_get_optics_otf_synthesizes_shift_invariant_otf_after_compute(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create("shift invariant"), scene, crop=True)

    otf = oi_get(oi, "optics OTF")

    assert otf is not None
    assert np.asarray(otf).shape == oi.data["photons"].shape
    assert np.isclose(float(np.abs(np.asarray(otf)[0, 0, 0])), 1.0, atol=1e-6)


def test_oi_set_optics_otf_supports_direct_raw_shift_invariant_otf(asset_store) -> None:
    params = {
        "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
        "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
        "blockSize": 16,
        "contrast": 1.0,
    }
    scene = scene_create("frequency orientation", params, asset_store=asset_store)
    scene = scene_set(scene, "fov", 3.0)
    base = oi_compute(oi_create("shift invariant"), scene, crop=True)
    raw_otf = oi_get(base, "optics OTF")
    assert raw_otf is not None

    ideal = oi_set(base, "optics OTF", np.ones_like(np.asarray(raw_otf), dtype=complex))
    stored = oi_get(ideal, "optics OTF")
    computed = oi_compute(ideal, scene, crop=True)

    assert stored is not None
    assert np.asarray(stored).shape == np.asarray(raw_otf).shape
    assert np.allclose(np.asarray(stored), 1.0)
    assert ideal.fields["optics"]["compute_method"] == "opticsotf"
    assert not np.allclose(computed.data["photons"], base.data["photons"])


def test_oi_set_optics_otf_repeats_2d_otf_across_wave(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    base = oi_compute(oi_create("shift invariant"), scene, crop=True)
    raw_otf = oi_get(base, "optics OTF")
    assert raw_otf is not None
    samples = np.asarray(raw_otf).shape[0]
    support = np.linspace(-1.0, 1.0, samples, dtype=float)
    xx, yy = np.meshgrid(support, support)
    gaussian = np.exp(-((xx**2) + (yy**2)) / 0.1)

    oi = oi_set(base, "optics OTF", gaussian)
    stored = np.asarray(oi_get(oi, "optics OTF"))

    assert stored.shape == np.asarray(raw_otf).shape
    assert np.allclose(stored[:, :, 0], gaussian)
    assert np.allclose(stored[:, :, -1], gaussian)


def test_si_synthetic_gaussian_builds_anisotropic_shift_invariant_optics() -> None:
    oi = oi_create("shift invariant")
    wave = np.asarray(oi_get(oi, "wave"), dtype=float)

    optics = si_synthetic("gaussian", oi, wave / wave[0], 2.0)
    updated = oi_set(oi, "optics", optics)
    psf_data = oi_get(updated, "psfdata")
    psf = np.asarray(psf_data["psf"], dtype=float)

    assert updated.fields["optics"]["model"] == "shiftinvariant"
    assert updated.fields["optics"]["compute_method"] == "opticsotf"
    assert psf.shape == (129, 129, wave.size)
    center = psf.shape[0] // 2
    horizontal = np.count_nonzero(psf[center, :, 0] > 1e-8)
    vertical = np.count_nonzero(psf[:, center, 0] > 1e-8)
    assert vertical > horizontal


def test_si_synthetic_lorentzian_applies_to_grid_lines_scene(asset_store) -> None:
    scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)
    oi = oi_create("psf")
    gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)

    optics = si_synthetic("lorentzian", oi, gamma)
    updated = oi_set(oi, "optics", optics)
    computed = oi_compute(updated, scene, crop=True)

    assert updated.fields["optics"]["compute_method"] == "opticsotf"
    assert updated.fields["optics"]["model"] == "shiftinvariant"
    assert np.asarray(oi_get(updated, "psfdata")["psf"]).shape[2] == np.asarray(oi_get(oi, "wave")).size
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_si_lorentzian_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_lorentzian_small", asset_store=asset_store)

    assert payload["photons"].ndim == 3
    assert payload["photons"].shape[2] == np.asarray(payload["wave"]).size


def test_si_synthetic_custom_loads_psf_mat_file(tmp_path, asset_store) -> None:
    psf = np.zeros((129, 129, 1), dtype=float)
    psf[64, 64, 0] = 1.0
    path = tmp_path / "custom_si_psf.mat"
    savemat(
        path,
        {
            "psf": psf,
            "wave": np.array([550.0], dtype=float),
            "umPerSamp": np.array([0.25, 0.25], dtype=float),
        },
    )

    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    optics = si_synthetic("custom", oi_create("shift invariant"), path)
    oi = oi_set(oi_create("shift invariant"), "optics", optics)
    computed = oi_compute(oi, scene, crop=True)

    stored = oi_get(oi, "psfdata")
    assert np.asarray(stored["psf"]).shape == (129, 129, 1)
    assert np.array_equal(np.asarray(stored["wave"]), np.array([550.0], dtype=float))
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]

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
    sensor = sensor_set(sensor, "autoexposure", True)
    sensor = sensor_set(sensor, "integrationtime", 0.125)
    assert sensor.fields["auto_exposure"] is False
    assert np.isclose(sensor.fields["integration_time"], 0.125)


def test_sensor_get_set_supports_n_samples_per_pixel(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "nsamplesperpixel", 3)

    assert sensor_get(sensor, "nsamplesperpixel") == 3


def test_sensor_get_reports_matlab_style_geometry_and_cfa_metadata(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    rows, cols = sensor.fields["size"]

    support = sensor_get(sensor, "spatialsupport", "um")
    cfa = sensor_get(sensor, "cfa")
    pattern = sensor_get(sensor, "pattern")
    cfa_config = sensor_get(sensor, "unitblockconfig")
    pattern_colors = sensor_get(sensor, "patterncolors")

    assert sensor_get(sensor, "rows") == rows
    assert sensor_get(sensor, "cols") == cols
    assert sensor_get(sensor, "size") == (rows, cols)
    assert np.isclose(sensor_get(sensor, "arrayheight"), rows * pixel_size[0])
    assert np.isclose(sensor_get(sensor, "arraywidth", "mm"), cols * pixel_size[1] * 1e3)
    assert np.allclose(sensor_get(sensor, "dimension", "um"), np.array([rows * pixel_size[0], cols * pixel_size[1]]) * 1e6)
    assert np.isclose(sensor_get(sensor, "wspatialresolution", "um"), pixel_size[1] * 1e6)
    assert np.isclose(sensor_get(sensor, "hspatialresolution"), pixel_size[0])
    assert np.isclose(sensor_get(sensor, "deltax", "um"), pixel_size[1] * 1e6)
    assert np.isclose(sensor_get(sensor, "deltay"), pixel_size[0])
    assert support["x"].shape == (cols,)
    assert support["y"].shape == (rows,)
    assert np.isclose(support["x"][0], -support["x"][-1])
    assert np.isclose(support["y"][0], -support["y"][-1])
    assert sensor_get(sensor, "unitblockrows") == 2
    assert sensor_get(sensor, "unitblockcols") == 2
    assert sensor_get(sensor, "cfasize") == (2, 2)
    assert sensor_get(sensor, "cfaname") == "Bayer RGB"
    assert sensor_get(sensor, "filtercolorletters") == "rgb"
    assert np.array_equal(pattern, np.array([[2, 1], [3, 2]], dtype=int))
    assert np.array_equal(cfa["pattern"], pattern)
    assert cfa["unitBlock"]["rows"] == 2
    assert cfa["unitBlock"]["cols"] == 2
    assert np.allclose(cfa["unitBlock"]["config"], cfa_config)
    assert np.allclose(cfa_config, np.array([[0.0, 0.0], [pixel_size[1], 0.0], [0.0, pixel_size[0]], [pixel_size[1], pixel_size[0]]], dtype=float))
    assert pattern_colors.shape == (2, 2)
    assert np.array_equal(pattern_colors, np.array([["g", "r"], ["b", "g"]], dtype="<U1"))


def test_sensor_get_set_supports_matlab_style_spectrum_metadata(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", np.array([400.0, 500.0, 600.0], dtype=float))
    sensor = sensor_set(sensor, "filterspectra", np.array([[0.0], [1.0], [0.0]], dtype=float))
    sensor = sensor_set(sensor, "pixelspectralqe", np.array([0.2, 0.6, 1.0], dtype=float))
    sensor = sensor_set(sensor, "irfilter", np.array([1.0, 0.5, 0.0], dtype=float))

    spectrum = sensor_get(sensor, "sensorspectrum")
    assert np.array_equal(spectrum["wave"], np.array([400.0, 500.0, 600.0], dtype=float))
    assert sensor_get(sensor, "wavelengthresolution") == 100.0
    assert sensor_get(sensor, "nwaves") == 3

    sensor = sensor_set(sensor, "sensorspectrum", {"wave": np.array([450.0, 550.0], dtype=float), "comment": "test spectrum"})

    assert np.array_equal(sensor_get(sensor, "wavelength"), np.array([450.0, 550.0], dtype=float))
    assert sensor_get(sensor, "binwidth") == 100.0
    assert sensor_get(sensor, "numberofwavelengthsamples") == 2
    assert np.allclose(sensor_get(sensor, "filterspectra"), np.array([[0.5], [0.5]], dtype=float))
    assert np.allclose(sensor_get(sensor, "colorfilters"), np.array([[0.5], [0.5]], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixelspectralqe"), np.array([0.4, 0.8], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixelqe"), np.array([0.4, 0.8], dtype=float))
    assert np.allclose(sensor_get(sensor, "infraredfilter"), np.array([0.75, 0.25], dtype=float))
    assert np.allclose(sensor_get(sensor, "irfilter"), np.array([0.75, 0.25], dtype=float))
    assert np.allclose(sensor_get(sensor, "spectralqe"), np.array([[0.15], [0.1]], dtype=float))
    assert np.allclose(sensor_get(sensor, "sensorspectralsr"), sensor_get(sensor, "sensor spectral sr"))
    assert sensor_get(sensor, "sensorspectrum")["comment"] == "test spectrum"


def test_sensor_get_set_supports_raw_color_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", np.array([500.0, 600.0], dtype=float))
    color = {
        "filterSpectra": np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.8]], dtype=float),
        "filterNames": ["red", "green", "blue"],
        "irFilter": np.array([0.75, 0.25], dtype=float),
    }

    sensor = sensor_set(sensor, "color", color)

    exported = sensor_get(sensor, "color")
    assert np.allclose(exported["filterSpectra"], color["filterSpectra"])
    assert exported["filterNames"] == color["filterNames"]
    assert np.allclose(exported["irFilter"], color["irFilter"])
    assert sensor_get(sensor, "filternames") == ["red", "green", "blue"]
    assert sensor_get(sensor, "filternamescellarray") == ["r", "g", "b"]
    assert sensor_get(sensor, "filtercolornamescellarray") == ["r", "g", "b"]
    assert sensor_get(sensor, "filternamescell") == ["r", "g", "b"]


def test_sensor_get_set_supports_pixel_passthrough_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    assert sensor_get(sensor, "pixel name") == "aps"
    assert sensor_get(sensor, "pixel type") == "pixel"

    sensor = sensor_set(sensor, "pixel name", "custom-pixel")
    sensor = sensor_set(sensor, "pixel type", "custom-type")
    sensor = sensor_set(sensor, "fillfactor", 0.5)
    sensor = sensor_set(sensor, "voltsperelectron", 2.0e-4)
    sensor = sensor_set(sensor, "maxvoltage", 1.5)
    sensor = sensor_set(sensor, "darkvoltageperpixel", 2.0e-3)
    sensor = sensor_set(sensor, "readstandarddeviationvolts", 3.0e-3)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 3.0e-6)
    sensor = sensor_set(sensor, "width between pixels", 0.5e-6)
    sensor = sensor_set(sensor, "height between pixels", 0.25e-6)
    sensor = sensor_set(sensor, "pixelspectralqe", np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    pixel_size = np.asarray(sensor_get(sensor, "pixelsize"), dtype=float)
    pd_area = float(sensor_get(sensor, "pdarea"))

    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.5)
    assert sensor_get(sensor, "pixel name") == "custom-pixel"
    assert sensor_get(sensor, "pixel type") == "custom-type"
    assert np.allclose(pixel_size, np.array([3.25e-6, 4.5e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth", "um"), 4.0)
    assert np.isclose(sensor_get(sensor, "pixelheight", "um"), 3.0)
    assert np.isclose(sensor_get(sensor, "pixelwidthmeters"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheightmeters"), 3.0e-6)
    assert np.isclose(sensor_get(sensor, "widthgap", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "heightgap", "um"), 0.25)
    assert np.isclose(sensor_get(sensor, "widthbetweenpixels", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "heightbetweenpixels", "um"), 0.25)
    assert np.allclose(sensor_get(sensor, "xyspacing", "um"), np.array([4.5, 3.25], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelarea"), np.prod(pixel_size))
    assert np.isclose(pd_area, (3.0e-6 * 4.0e-6) * 0.5)
    assert np.allclose(sensor_get(sensor, "pdsize", "um"), np.sqrt(0.5) * np.array([3.0, 4.0], dtype=float))
    assert np.allclose(sensor_get(sensor, "pddimension", "um"), np.sqrt(0.5) * np.array([4.0, 3.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "conversiongain"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "conversiongainvperelectron"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "voltsperelectron"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "voltageswing"), 1.5)
    assert np.isclose(sensor_get(sensor, "maxvoltage"), 1.5)
    assert np.isclose(sensor_get(sensor, "wellcapacity"), 1.5 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "darkvoltage"), 2.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvoltageperpixel"), 2.0e-3)
    assert np.isclose(sensor_get(sensor, "darkelectrons"), 2.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "darkcurrent"), sensor_get(sensor, "darkelectrons") * 1.602177e-19)
    assert np.isclose(sensor_get(sensor, "darkcurrentperpixel"), sensor_get(sensor, "darkcurrent"))
    assert np.isclose(sensor_get(sensor, "darkcurrentdensity"), sensor_get(sensor, "darkcurrent") / pd_area)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readnoiseelectrons"), 3.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationvolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 3.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readnoisemillivolts"), 3.0)
    assert np.allclose(sensor_get(sensor, "pdspectralqe"), sensor_get(sensor, "pixelspectralqe"))
    assert np.allclose(sensor_get(sensor, "pixelqe"), sensor_get(sensor, "pixelspectralqe"))
    assert np.allclose(sensor_get(sensor, "pdspectralsr"), sensor_get(sensor, "pixelspectralsr"))
    assert np.allclose(sensor_get(sensor, "spectralsr"), sensor_get(sensor, "pixelspectralsr"))
    assert np.allclose(sensor_get(sensor, "sr"), sensor_get(sensor, "pixelspectralsr"))

    sensor = sensor_set(sensor, "read noise", 10.0)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 10.0 * 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 10.0)

    sensor = sensor_set(sensor, "readnoisemillivolts", 7.0)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 7.0e-3)

    sensor = sensor_set(sensor, "conversiongainvperelectron", 3.0e-4)
    assert np.isclose(sensor_get(sensor, "conversiongain"), 3.0e-4)

    sensor = sensor_set(sensor, "pdwidth", 2.0e-6)
    sensor = sensor_set(sensor, "pdheight", 1.5e-6)

    assert np.isclose(sensor_get(sensor, "pdwidth", "um"), 2.0)
    assert np.isclose(sensor_get(sensor, "pdheight", "um"), 1.5)
    assert np.isclose(sensor_get(sensor, "fillfactor"), (2.0e-6 * 1.5e-6) / (4.0e-6 * 3.0e-6))

    replacement_pixel = dict(sensor_get(sensor, "pixel"))
    replacement_pixel["fill_factor"] = 0.25
    replacement_pixel["conversion_gain_v_per_electron"] = 1.0e-4
    replacement_pixel["name"] = "replacement-pixel"
    replacement_pixel["type"] = "replacement-type"
    sensor = sensor_set(sensor, "pixel", replacement_pixel)

    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.25)
    assert np.isclose(sensor_get(sensor, "conversiongain"), 1.0e-4)
    assert sensor_get(sensor, "pixel name") == "replacement-pixel"
    assert sensor_get(sensor, "pixel type") == "replacement-type"


def test_sensor_get_set_supports_pixel_optical_and_spectral_metadata(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 0.01)
    sensor = sensor_set(sensor, "layerthicknesses", np.array([1.0e-6, 2.0e-6, 0.5e-6], dtype=float))
    sensor = sensor_set(sensor, "refractiveindices", np.array([1.0, 1.5, 3.4], dtype=float))
    sensor = sensor_set(sensor, "pixelspectrum", {"wave": np.array([450.0, 550.0, 650.0], dtype=float), "comment": "pixel spectrum"})
    sensor = sensor_set(sensor, "quantum efficiency", np.array([0.1, 0.2, 0.3], dtype=float))
    sensor = sensor_set(sensor, "darkvoltageperpixelpersec", 2.0e-3)
    sensor = sensor_set(sensor, "readnoisestdvolts", 1.0e-3)
    sensor = sensor_set(sensor, "voltage swing", 1.2)

    assert np.allclose(sensor_get(sensor, "layerthicknesses", "um"), np.array([1.0, 2.0, 0.5], dtype=float))
    assert np.isclose(sensor_get(sensor, "stackheight", "um"), 3.5)
    assert np.isclose(sensor_get(sensor, "pixeldepth", "um"), 3.5)
    assert np.isclose(sensor_get(sensor, "pixeldepthmeters"), 3.5e-6)
    assert np.allclose(sensor_get(sensor, "refractiveindices"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.allclose(sensor_get(sensor, "refractiveindex"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.allclose(sensor_get(sensor, "n"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.array_equal(sensor_get(sensor, "pixelwavelength"), np.array([450.0, 550.0, 650.0], dtype=float))
    assert sensor_get(sensor, "pixelbinwidth") == 100.0
    assert sensor_get(sensor, "pixelnwave") == 3
    assert sensor_get(sensor, "pixelspectrum")["comment"] == "pixel spectrum"
    assert np.allclose(sensor_get(sensor, "quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixel quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetector quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetector spectral quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))

    expected_pixel_dr = 20.0 * np.log10((1.2 - 2.0e-3 * 0.01) / np.sqrt((2.0e-3 * 0.01) + (1.0e-3**2)))
    assert np.isclose(sensor_get(sensor, "pixeldynamicrange"), expected_pixel_dr)

    sensor = sensor_set(sensor, "n", np.array([1.0, 2.0, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "refractive indices"), np.array([1.0, 2.0, 3.5], dtype=float))

    sensor = sensor_set(sensor, "conversiongainvpelectron", 2.5e-6)
    assert np.isclose(sensor_get(sensor, "conversiongainvpelectron"), 2.5e-6)
    assert np.isclose(sensor_get(sensor, "conversion gain"), 2.5e-6)

    sensor = sensor_set(sensor, "vswing", 1.3)
    assert np.isclose(sensor_get(sensor, "vswing"), 1.3)
    assert np.isclose(sensor_get(sensor, "max voltage"), 1.3)

    sensor = sensor_set(sensor, "darkvolt", 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvolt"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvoltageperpixelpersec"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "voltspersecond"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readnoisestdvolts"), 1.0e-3)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 1.0e-3 / 2.5e-6)


def test_sensor_get_set_supports_photodetector_position_passthrough(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 3.0e-6)
    sensor = sensor_set(sensor, "photodetectorsize", np.array([1.0e-6, 2.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([1.0, 1.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdxpos", "um"), 1.0)
    assert np.isclose(sensor_get(sensor, "pdypos", "um"), 1.0)
    assert np.allclose(sensor_get(sensor, "photodetectorsize", "um"), np.array([1.0, 2.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "photodetectorwidth", "um"), 2.0)
    assert np.isclose(sensor_get(sensor, "photodetectorheight", "um"), 1.0)

    sensor = sensor_set(sensor, "pdxpos", 0.5e-6)
    sensor = sensor_set(sensor, "pdypos", 0.75e-6)

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([0.5, 0.75], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdxpos", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "pdypos", "um"), 0.75)

    sensor = sensor_set(sensor, "photodetectorxposition", 0.25e-6)
    sensor = sensor_set(sensor, "photodetectoryposition", 0.5e-6)

    assert np.isclose(sensor_get(sensor, "photodetectorxposition", "um"), 0.25)
    assert np.isclose(sensor_get(sensor, "photodetectoryposition", "um"), 0.5)

    sensor = sensor_set(sensor, "pdposition", np.array([0.25e-6, 0.5e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([0.25, 0.5], dtype=float))

    with pytest.raises(ValueError, match="photodetector position must keep the photodetector inside the pixel."):
        sensor_set(sensor, "pdposition", np.array([3.0e-6, 2.5e-6], dtype=float))


def test_sensor_set_supports_matlab_width_height_pair_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "widthandheight", np.array([4.0e-6, 3.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 3.0e-6)
    assert np.allclose(sensor_get(sensor, "pixelsize"), np.array([3.0e-6, 4.0e-6], dtype=float))

    sensor = sensor_set(sensor, "widthheight", 5.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 5.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 5.0e-6)

    sensor = sensor_set(sensor, "pdwidthandheight", np.array([2.0e-6, 1.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "photodetectorwidth"), 2.0e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorheight"), 1.0e-6)
    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([1.0e-6, 2.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([2.0e-6, 1.0e-6], dtype=float))

    sensor = sensor_set(sensor, "pdwidthandheight", 1.5e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorwidth"), 1.5e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorheight"), 1.5e-6)
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([1.5e-6, 1.5e-6], dtype=float))


def test_sensor_set_routes_direct_unique_pixel_aliases_without_prefix(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "widthandheight", np.array([4.0e-6, 3.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 3.0e-6)

    sensor = sensor_set(sensor, "pdwidthandheight", np.array([2.0e-6, 1.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdwidth"), 2.0e-6)
    assert np.isclose(sensor_get(sensor, "pdheight"), 1.0e-6)
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([2.0e-6, 1.0e-6], dtype=float))

    initial_fill_factor = float(sensor_get(sensor, "fillfactor"))
    sensor = sensor_set(sensor, "sizeconstantfillfactor", np.array([8.0e-6, 6.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), initial_fill_factor)

    sensor = sensor_set(sensor, "sizekeepfillfactor", np.array([10.0e-6, 8.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), initial_fill_factor)

    sensor = sensor_set(sensor, "dark voltage per pixel per sec", 1.5e-3)
    assert np.isclose(sensor_get(sensor, "dark voltage"), 1.5e-3)


def test_sensor_set_pixel_size_same_fill_factor_scales_photodetector_geometry(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 2.0e-6)
    sensor = sensor_set(sensor, "pd size", np.array([1.0e-6, 2.0e-6], dtype=float))
    sensor = sensor_set(sensor, "pd position", np.array([0.5e-6, 0.25e-6], dtype=float))

    sensor = sensor_set(sensor, "sizesamefillfactor", np.array([4.0e-6, 8.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pixelsize"), np.array([4.0e-6, 8.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([2.0e-6, 4.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "pdposition"), np.array([1.0e-6, 0.5e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.25)

    sensor = sensor_set(sensor, "pixelsize", np.array([8.0e-6, 16.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([2.0e-6, 4.0e-6], dtype=float))
    assert not np.isclose(sensor_get(sensor, "fillfactor"), 0.25)


def test_sensor_get_set_supports_chart_and_metadata_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    corner_points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    rects = np.array([[10.0, 20.0, 5.0, 6.0]], dtype=float)
    current_rect = np.array([7.0, 8.0, 9.0, 10.0], dtype=float)
    rect_handles = ["r1", "r2"]

    sensor = sensor_set(sensor, "chartparameters", {"name": "Macbeth", "nSquares": 24})
    sensor = sensor_set(sensor, "cornerpoints", corner_points)
    sensor = sensor_set(sensor, "chartrects", rects)
    sensor = sensor_set(sensor, "currentrect", current_rect)
    sensor = sensor_set(sensor, "mccrecthandles", rect_handles)
    sensor = sensor_set(sensor, "metadatasensorname", "sensor-a")
    sensor = sensor_set(sensor, "metadatascenename", "scene-a")
    sensor = sensor_set(sensor, "metadataopticsname", "optics-a")
    sensor = sensor_set(sensor, "metadatacrop", np.array([1, 2, 3, 4], dtype=int))

    chart = sensor_get(sensor, "chartparameters")

    assert chart["name"] == "Macbeth"
    assert chart["nSquares"] == 24
    assert np.array_equal(chart["cornerPoints"], corner_points)
    assert np.array_equal(chart["rects"], rects)
    assert np.array_equal(chart["currentRect"], current_rect)
    assert np.array_equal(sensor_get(sensor, "cornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartcornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartcorners"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chart corners"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chart corner points"), corner_points)
    assert np.array_equal(sensor_get(sensor, "mcccornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartrects"), rects)
    assert np.array_equal(sensor_get(sensor, "chartrectangles"), rects)
    assert np.array_equal(sensor_get(sensor, "chart rectangles"), rects)
    assert np.array_equal(sensor_get(sensor, "currentrect"), current_rect)
    assert np.array_equal(sensor_get(sensor, "chartcurrentrect"), current_rect)
    assert np.array_equal(sensor_get(sensor, "current rect"), current_rect)
    assert sensor_get(sensor, "mccrecthandles") == rect_handles
    assert sensor_get(sensor, "metadatasensorname") == "sensor-a"
    assert sensor_get(sensor, "metadatascenename") == "scene-a"
    assert sensor_get(sensor, "metadataopticsname") == "optics-a"
    assert np.array_equal(sensor_get(sensor, "metadatacrop"), np.array([1, 2, 3, 4], dtype=int))

    sensor = sensor_set(sensor, "mcccornerpoints", corner_points + 1.0)
    sensor = sensor_set(sensor, "chartrectangles", rects + 1.0)
    sensor = sensor_set(sensor, "chartcurrentrect", current_rect + 1.0)

    assert np.array_equal(sensor_get(sensor, "cornerpoints"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "chart corner points"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "mcccornerpoints"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "chartrectangles"), rects + 1.0)
    assert np.array_equal(sensor_get(sensor, "chartcurrentrect"), current_rect + 1.0)


def test_sensor_get_set_supports_diffusion_mtf_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    diffusion = {
        "name": "Gaussian",
        "otf": np.array([1.0, 0.8, 0.6], dtype=float),
        "support": np.array([0.0, 0.5, 1.0], dtype=float),
    }

    sensor = sensor_set(sensor, "diffusionmtf", diffusion)

    stored = sensor_get(sensor, "diffusionmtf")
    assert stored is not None
    assert stored["name"] == "Gaussian"
    assert np.array_equal(stored["otf"], diffusion["otf"])
    assert np.array_equal(stored["support"], diffusion["support"])

    stored["otf"][0] = 9.0
    assert np.array_equal(sensor_get(sensor, "diffusionmtf")["otf"], diffusion["otf"])

    sensor = sensor_set(sensor, "diffusionmtf", None)
    assert sensor_get(sensor, "diffusionmtf") is None


def test_sensor_get_set_supports_movement_metadata_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    positions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    frames = np.array([2, 3], dtype=int)

    sensor = sensor_set(sensor, "sensormovement", {"name": "shake", "enabled": True})
    sensor = sensor_set(sensor, "movementpositions", positions)
    sensor = sensor_set(sensor, "framesperpositions", frames)

    movement = sensor_get(sensor, "sensormovement")

    assert movement["name"] == "shake"
    assert movement["enabled"] is True
    assert np.array_equal(movement["pos"], positions)
    assert np.array_equal(sensor_get(sensor, "movement positions"), positions)
    assert np.array_equal(sensor_get(sensor, "sensorpositions"), positions)
    assert np.array_equal(sensor_get(sensor, "sensorpositionsx"), positions[:, 0])
    assert np.array_equal(sensor_get(sensor, "sensorpositionsy"), positions[:, 1])
    assert np.array_equal(sensor_get(sensor, "framesperpositions"), frames)
    assert np.array_equal(sensor_get(sensor, "framesperposition"), frames)
    assert np.array_equal(sensor_get(sensor, "exposuretimesperposition"), frames)
    assert np.array_equal(sensor_get(sensor, "etimeperpos"), frames)


def test_sensor_get_set_supports_legacy_human_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    human = {
        "name": "human",
        "coneType": np.array([[1, 2], [3, 4]], dtype=int),
        "densities": np.array([0.0, 0.6, 0.3, 0.1], dtype=float),
        "xy": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "rSeed": 17,
    }

    sensor = sensor_set(sensor, "human", human)

    stored = sensor_get(sensor, "human")
    assert stored is not None
    assert stored["name"] == "human"
    assert np.array_equal(sensor_get(sensor, "conetype"), human["coneType"])
    assert np.array_equal(sensor_get(sensor, "human cone densities"), human["densities"])
    assert np.array_equal(sensor_get(sensor, "conexy"), human["xy"])
    assert np.array_equal(sensor_get(sensor, "conelocs"), human["xy"])
    assert sensor_get(sensor, "humanrseed") == 17

    stored["coneType"][0, 0] = 9
    assert np.array_equal(sensor_get(sensor, "human")["coneType"], human["coneType"])

    cone_type = np.array([[4, 3], [2, 1]], dtype=int)
    densities = np.array([0.1, 0.5, 0.3, 0.1], dtype=float)
    xy = np.array([[0.5, 0.6]], dtype=float)
    sensor = sensor_set(sensor, "conetype", cone_type)
    sensor = sensor_set(sensor, "humanconedensities", densities)
    sensor = sensor_set(sensor, "conexy", xy)
    sensor = sensor_set(sensor, "humanrseed", 23)

    assert np.array_equal(sensor_get(sensor, "conetype"), cone_type)
    assert np.array_equal(sensor_get(sensor, "humanconetype"), cone_type)
    assert np.array_equal(sensor_get(sensor, "humanconedensities"), densities)
    assert np.array_equal(sensor_get(sensor, "humanconelocs"), xy)
    assert np.array_equal(sensor_get(sensor, "conexy"), xy)
    assert np.array_equal(sensor_get(sensor, "conelocs"), xy)
    assert sensor_get(sensor, "humanrseed") == 23


def test_sensor_get_set_supports_legacy_scene_and_lens_metadata_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "scenename", "scene-b")
    sensor = sensor_set(sensor, "metadatalensname", "lens-b")
    sensor = sensor_set(sensor, "metadatasensorname", "sensor-b")
    sensor = sensor_set(sensor, "metadatacrop", {"rect": [1, 2, 3, 4]})

    assert sensor_get(sensor, "scene_name") == "scene-b"
    assert sensor_get(sensor, "scenename") == "scene-b"
    assert sensor_get(sensor, "metadatascenename") == "scene-b"
    assert sensor_get(sensor, "lens") == "lens-b"
    assert sensor_get(sensor, "metadatalensname") == "lens-b"
    assert sensor_get(sensor, "metadatalens") == "lens-b"
    assert sensor_get(sensor, "metadata optics name") == "lens-b"
    assert sensor_get(sensor, "metadatasensorname") == "sensor-b"
    assert sensor_get(sensor, "metadatacrop") == {"rect": [1, 2, 3, 4]}

    sensor = sensor_set(sensor, "metadatalens", "lens-c")

    assert sensor_get(sensor, "lens") == "lens-c"
    assert sensor_get(sensor, "metadatalensname") == "lens-c"
    assert sensor_get(sensor, "metadatalens") == "lens-c"


def test_sensor_get_set_supports_microlens_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    microlens = {
        "name": "default",
        "type": "microlens",
        "offset": np.array([0.0, 1.0], dtype=float),
        "wavelength": 500.0,
    }
    ml_offset = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float)

    sensor = sensor_set(sensor, "microlens", microlens)
    sensor = sensor_set(sensor, "microlensoffset", ml_offset)

    stored = sensor_get(sensor, "microlens")

    assert stored["name"] == "default"
    assert stored["type"] == "microlens"
    assert np.array_equal(stored["offset"], np.array([0.0, 1.0], dtype=float))
    assert stored["wavelength"] == 500.0
    assert sensor_get(sensor, "ulens")["name"] == "default"
    assert np.array_equal(sensor_get(sensor, "microlensoffset"), ml_offset)
    assert np.array_equal(sensor_get(sensor, "microlensoffsetmicrons"), ml_offset)

    stored["offset"][0] = 9.0
    assert np.array_equal(sensor_get(sensor, "mlens")["offset"], np.array([0.0, 1.0], dtype=float))


def test_sensor_get_set_supports_column_fpn_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    cols = int(sensor_get(sensor, "cols"))
    column_offset = np.linspace(-0.01, 0.01, cols, dtype=float)
    column_gain = np.linspace(0.9, 1.1, cols, dtype=float)

    assert np.array_equal(sensor_get(sensor, "columnfixedpatternnoise"), np.array([0.0, 0.0], dtype=float))
    assert np.array_equal(sensor_get(sensor, "colfpn"), np.array([0.0, 0.0], dtype=float))
    assert sensor_get(sensor, "columnfpnoffset") == 0.0
    assert sensor_get(sensor, "columnfpngain") == 0.0
    assert sensor_get(sensor, "coloffsetfpn") is None
    assert sensor_get(sensor, "colgainfpn") is None

    sensor = sensor_set(sensor, "columnfpnparameters", np.array([0.05, 0.1], dtype=float))
    assert np.array_equal(sensor_get(sensor, "columnfpn"), np.array([0.05, 0.1], dtype=float))

    sensor = sensor_set(sensor, "offsetnoisevalue", 0.015)
    sensor = sensor_set(sensor, "gainnoisevalue", 3.5)
    sensor = sensor_set(sensor, "columnfixedpatternnoise", np.array([0.125, 0.25], dtype=float))
    sensor = sensor_set(sensor, "coloffsetfpnvector", column_offset)
    sensor = sensor_set(sensor, "colgainfpnvector", column_gain)

    assert np.isclose(sensor_get(sensor, "dsnusigma"), 0.015)
    assert np.isclose(sensor_get(sensor, "dsnulevel"), 0.015)
    assert np.isclose(sensor_get(sensor, "sigmaoffsetfpn"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetfpn"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetsd"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetnoisevalue"), 0.015)
    assert np.isclose(sensor_get(sensor, "prnusigma"), 3.5)
    assert np.isclose(sensor_get(sensor, "prnulevel"), 3.5)
    assert np.isclose(sensor_get(sensor, "sigmagainfpn"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainfpn"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainsd"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainnoisevalue"), 3.5)
    assert np.isclose(sensor_get(sensor, "sigmaprnu"), 3.5)
    assert np.allclose(sensor_get(sensor, "fpnparameters"), np.array([0.015, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "fpnoffsetgain"), np.array([0.015, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "fpnoffsetandgain"), np.array([0.015, 3.5], dtype=float))
    assert np.array_equal(sensor_get(sensor, "columnfixedpatternnoise"), np.array([0.125, 0.25], dtype=float))
    assert np.array_equal(sensor_get(sensor, "colfpn"), np.array([0.125, 0.25], dtype=float))
    assert sensor_get(sensor, "columnfpnoffset") == 0.125
    assert sensor_get(sensor, "columnfpngain") == 0.25
    assert sensor_get(sensor, "columndsnu") == 0.125
    assert sensor_get(sensor, "columnprnu") == 0.25
    assert np.array_equal(sensor_get(sensor, "coloffsetfpn"), column_offset)
    assert np.array_equal(sensor_get(sensor, "coloffsetfpnvector"), column_offset)
    assert np.array_equal(sensor_get(sensor, "coloffset"), column_offset)
    assert np.array_equal(sensor_get(sensor, "colgainfpn"), column_gain)
    assert np.array_equal(sensor_get(sensor, "colgainfpnvector"), column_gain)
    assert np.array_equal(sensor_get(sensor, "colgain"), column_gain)

    stored = sensor_get(sensor, "coloffsetfpn")
    assert stored is not None
    stored[0] = 9.0
    assert np.array_equal(sensor_get(sensor, "coloffsetfpn"), column_offset)

    with pytest.raises(ValueError, match="Column FPN"):
        sensor_set(sensor, "columnfixedpatternnoise", np.array([1.0, 2.0, 3.0], dtype=float))
    with pytest.raises(ValueError, match="Bad column offset data"):
        sensor_set(sensor, "coloffsetfpnvector", np.ones(cols - 1, dtype=float))
    with pytest.raises(ValueError, match="Bad column gain data"):
        sensor_set(sensor, "colgainfpnvector", np.ones(cols - 1, dtype=float))


def test_sensor_get_set_supports_fpn_image_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    rows, cols = sensor_get(sensor, "size")
    dsnu_image = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    prnu_image = np.linspace(0.9, 1.1, rows * cols, dtype=float).reshape(rows, cols)

    sensor = sensor_set(sensor, "dsnuimage", dsnu_image)
    sensor = sensor_set(sensor, "prnuimage", prnu_image)

    stored_dsnu = sensor_get(sensor, "dsnuimage")
    stored_prnu = sensor_get(sensor, "prnuimage")

    assert np.array_equal(stored_dsnu, dsnu_image)
    assert np.array_equal(sensor_get(sensor, "offsetfpnimage"), dsnu_image)
    assert np.array_equal(stored_prnu, prnu_image)
    assert np.array_equal(sensor_get(sensor, "gainfpnimage"), prnu_image)

    stored_dsnu[0, 0] = -1.0
    stored_prnu[0, 0] = -1.0
    assert np.array_equal(sensor_get(sensor, "dsnuimage"), dsnu_image)
    assert np.array_equal(sensor_get(sensor, "prnuimage"), prnu_image)

    sensor = sensor_set(sensor, "offsetfpnimage", None)
    sensor = sensor_set(sensor, "gainfpnimage", None)

    assert sensor_get(sensor, "dsnuimage") is None
    assert sensor_get(sensor, "prnuimage") is None


def test_sensor_get_set_supports_consistency_and_compute_method_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "sensorconsistency") is False
    assert sensor_get(sensor, "sensorcomputemethod") is None

    sensor = sensor_set(sensor, "sensorconsistency", True)
    sensor = sensor_set(sensor, "sensorcomputemethod", {"name": "binning", "factor": 2})

    assert sensor_get(sensor, "sensorconsistency") is True
    assert sensor_get(sensor, "sensorcompute") == {"name": "binning", "factor": 2}
    assert sensor_get(sensor, "sensorcomputemethod") == {"name": "binning", "factor": 2}


def test_sensor_get_set_supports_exposure_plane_and_cds_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "nexposures") == 1
    assert sensor_get(sensor, "exposureplane") == 1
    assert sensor_get(sensor, "cds") is False

    sensor.fields["integration_time"] = np.array([0.01, 0.02, 0.03], dtype=float)
    sensor = sensor_set(sensor, "exposureplane", 3.2)
    sensor = sensor_set(sensor, "cds", True)
    sensor = sensor_set(sensor, "autoexp", "off")

    assert sensor_get(sensor, "nexposures") == 3
    assert sensor_get(sensor, "exposureplane") == 3
    assert sensor_get(sensor, "cds") is True
    assert sensor_get(sensor, "correlateddoublesampling") is True
    assert sensor_get(sensor, "autoexposure") is False

    sensor = sensor_set(sensor, "autoexp", "on")

    assert sensor_get(sensor, "automaticexposure") is True
    assert sensor_get(sensor, "integrationtime") == 0.0


def test_sensor_get_set_supports_exposure_method_and_time_summaries(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "exposuretimes", np.array([0.01, 0.02, 0.04], dtype=float))

    assert np.array_equal(sensor_get(sensor, "exptime"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuretimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuretime", "ms"), np.array([10.0, 20.0, 40.0], dtype=float))
    assert np.array_equal(sensor_get(sensor, "expduration"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposureduration"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuredurations"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueintegrationtimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueexptime"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueexptimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.isclose(sensor_get(sensor, "centralexposure"), 0.02)
    assert np.isclose(sensor_get(sensor, "geometricmeanexposuretime"), 0.02)
    assert sensor_get(sensor, "expmethod") == "bracketedExposure"
    assert sensor_get(sensor, "nexposures") == 3

    sensor = sensor_set(sensor, "expmethod", "videoExposure")

    assert sensor_get(sensor, "expmethod") == "videoExposure"

    sensor = sensor_set(sensor, "integrationtime", np.array([[0.01, 0.02], [0.03, 0.04]], dtype=float))
    sensor = sensor_set(sensor, "automaticexposure", "on")

    assert sensor_get(sensor, "automaticexposure") is True
    assert np.array_equal(sensor_get(sensor, "integrationtime"), np.zeros((2, 2), dtype=float))
    assert sensor_get(sensor, "exposuremethod") == "videoExposure"


def test_sensor_compute_rejects_multiple_integration_times(asset_store) -> None:
    scene = scene_create("uniform d65")
    oi = oi_compute(oi_create(), scene)
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "integrationtimes", np.array([0.01, 0.02], dtype=float))

    with pytest.raises(UnsupportedOptionError, match="sensorCompute"):
        sensor_compute(sensor, oi)


def test_sensor_get_set_supports_sampling_and_vignetting_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "pixelsamples") == 1
    assert sensor_get(sensor, "sensorbareetendue") == 0

    sensor = sensor_set(sensor, "spatialsamplesperpixel", 3)
    sensor = sensor_set(sensor, "vignetting", "bare")

    assert sensor_get(sensor, "ngridsamples") == 3
    assert sensor_get(sensor, "nsamplesperpixel") == 3
    assert sensor_get(sensor, "npixelsamplesforcomputing") == 3
    assert sensor_get(sensor, "pixelsamples") == 3
    assert sensor_get(sensor, "vignetting") == "bare"
    assert sensor_get(sensor, "sensorvignetting") == "bare"
    assert sensor_get(sensor, "vignettingflag") == "bare"
    assert sensor_get(sensor, "vignettingname") == "bare"
    assert sensor_get(sensor, "sensorbareetendue") == "bare"
    assert sensor_get(sensor, "nomicrolensetendue") == "bare"


def test_sensor_get_set_supports_noise_seed_reuse_and_response_type(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "reusenoise") is False
    assert sensor_get(sensor, "noiseseed") == 0
    assert sensor_get(sensor, "responsetype") == "linear"

    sensor = sensor_set(sensor, "reusenoise", True)
    sensor = sensor_set(sensor, "noiseseed", 7)
    sensor = sensor_set(sensor, "responsetype", "LOG")

    assert sensor_get(sensor, "reusenoise") is True
    assert sensor_get(sensor, "noiseseed") == 7
    assert sensor_get(sensor, "responsetype") == "log"


def test_sensor_get_supports_response_and_dynamic_range_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "integrationtime", 0.05)
    sensor = sensor_set(sensor, "dsnusigma", 0.002)

    pixel = sensor_get(sensor, "pixel")
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])
    dsnu_sigma = float(pixel["dsnu_sigma_v"])
    voltage_swing = float(pixel["voltage_swing"])
    expected_noise_sd = np.sqrt(
        ((dark_voltage * 0.05) / conversion_gain) * (conversion_gain**2)
        + (read_noise**2)
        + (dsnu_sigma**2)
    )
    expected_dr = 10.0 * np.log10((voltage_swing - (dark_voltage * 0.05)) / expected_noise_sd)

    assert np.isclose(sensor_get(sensor, "dr"), expected_dr)
    assert np.isclose(sensor_get(sensor, "drdb20"), expected_dr)
    assert np.isclose(sensor_get(sensor, "dynamicrange"), expected_dr)
    assert np.isclose(sensor_get(sensor, "sensordynamicrange"), expected_dr)

    sensor = sensor_set(sensor, "integrationtime", 0.0)
    assert sensor_get(sensor, "sensordynamicrange") is None

    sensor = sensor_set(sensor, "volts", np.array([[0.0, 0.5], [0.25, 0.75]], dtype=float))
    assert np.isclose(sensor_get(sensor, "responsedr"), 0.75 / (1.0 / 4096.0))

    sensor = sensor_set(sensor, "noiseflag", 1)
    assert sensor_get(sensor, "noiseflag") == 1
    assert sensor_get(sensor, "shotnoiseflag") == 1


def test_sensor_get_set_supports_black_level_alias(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    nbits = int(sensor_get(sensor, "nbits"))

    sensor = sensor_set(sensor, "blacklevel", 64)

    assert sensor_get(sensor, "blacklevel") == 64.0
    assert sensor_get(sensor, "zerolevel") == 64.0
    assert sensor_get(sensor, "maxdigitalvalue") == float((2**nbits) - 64)

    sensor = sensor_set(sensor, "zerolevel", 32)

    assert sensor_get(sensor, "blacklevel") == 32.0
    assert sensor_get(sensor, "zerolevel") == 32.0


def test_sensor_get_set_supports_digital_value_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    dv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    sensor = sensor_set(sensor, "digitalvalue", dv)

    assert np.array_equal(sensor_get(sensor, "dv"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalue"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalues"), dv)

    sensor = sensor_set(sensor, "digitalvalues", dv + 1.0)

    assert np.array_equal(sensor_get(sensor, "digitalvalue"), dv + 1.0)
    assert np.array_equal(sensor_get(sensor, "digitalvalues"), dv + 1.0)


def test_sensor_get_supports_dv_or_volts_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    dv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    sensor = sensor_set(sensor, "volts", volts)
    assert np.array_equal(sensor_get(sensor, "dvorvolts"), volts)
    assert np.array_equal(sensor_get(sensor, "digitalorvolts"), volts)

    sensor = sensor_set(sensor, "dv", dv)
    assert np.array_equal(sensor_get(sensor, "dvorvolts"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalorvolts"), dv)


def test_sensor_get_supports_response_ratio_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.array([[0.25, 0.5], [0.75, 1.0]], dtype=float))

    expected_volts_ratio = 1.0 / float(sensor.fields["pixel"]["voltage_swing"])
    assert np.isclose(sensor_get(sensor, "responseratio"), expected_volts_ratio)
    assert np.isclose(sensor_get(sensor, "volts2maxratio"), expected_volts_ratio)

    sensor = sensor_set(sensor, "digitalvalue", np.array([[64.0, 128.0], [256.0, 512.0]], dtype=float))
    sensor.data.pop("volts", None)
    expected_dv_ratio = 512.0 / float(2 ** int(sensor.fields["nbits"]))

    assert np.isclose(sensor_get(sensor, "responseratio"), expected_dv_ratio)
    assert np.isclose(sensor_get(sensor, "volts2maxratio"), expected_dv_ratio)


def test_sensor_get_supports_volt_images(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 16.0
    sensor = sensor_set(sensor, "volts", volts)

    plane_images = sensor_get(sensor, "voltimages")

    assert plane_images is not None
    assert plane_images.shape == (4, 4, 3)
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)
    assert np.isnan(plane_images[0, 0, 0])
    assert np.isclose(plane_images[0, 1, 0], volts[0, 1])
    assert np.isclose(plane_images[0, 0, 1], volts[0, 0])
    assert np.isclose(plane_images[1, 0, 2], volts[1, 0])
    assert np.array_equal(~np.isnan(plane_images[:, :, 0]), tiled_pattern == 1)
    assert np.array_equal(~np.isnan(plane_images[:, :, 1]), tiled_pattern == 2)
    assert np.array_equal(~np.isnan(plane_images[:, :, 2]), tiled_pattern == 3)


def test_sensor_get_set_supports_voltage_electron_and_analog_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.full((2, 2), 0.25, dtype=float)

    sensor = sensor_set(sensor, "ag", 2.5)
    sensor = sensor_set(sensor, "ao", 0.05)
    sensor = sensor_set(sensor, "voltage", volts)

    assert np.isclose(sensor_get(sensor, "analoggain"), 2.5)
    assert np.isclose(sensor_get(sensor, "ag"), 2.5)
    assert np.isclose(sensor_get(sensor, "analogoffset"), 0.05)
    assert np.isclose(sensor_get(sensor, "ao"), 0.05)
    assert np.array_equal(sensor_get(sensor, "volts"), volts)
    assert np.array_equal(sensor_get(sensor, "voltage"), volts)
    assert np.array_equal(sensor_get(sensor, "electron"), sensor_get(sensor, "electrons"))


def test_sensor_get_supports_channel_select_for_sensor_data(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(sensor, "ag", 2.0)
    sensor = sensor_set(sensor, "ao", 0.1)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 10.0
    dv = np.arange(1, 17, dtype=float).reshape(4, 4)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)

    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)
    expected_volts = volts[tiled_pattern == 2]
    expected_dv = dv[tiled_pattern == 2]
    expected_electrons = np.asarray(sensor_get(sensor, "electrons"))[tiled_pattern == 2]

    assert np.array_equal(sensor_get(sensor, "volts", 2), expected_volts)
    assert np.array_equal(sensor_get(sensor, "voltage", 2), expected_volts)
    assert np.array_equal(sensor_get(sensor, "dv", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalue", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalues", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "electrons", 2), expected_electrons)
    assert np.array_equal(sensor_get(sensor, "electron", 2), expected_electrons)


def test_sensor_get_supports_electrons_per_area(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(sensor, "ag", 2.0)
    sensor = sensor_set(sensor, "ao", 0.1)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 10.0
    sensor = sensor_set(sensor, "volts", volts)

    pd_area_m2 = float(sensor_get(sensor, "pixel pd area"))
    electrons = np.asarray(sensor_get(sensor, "electrons"), dtype=float)
    expected_m2 = electrons / pd_area_m2
    expected_um2 = expected_m2 / 1e12
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)

    assert np.isclose(sensor_get(sensor, "pixel pd area", "um"), pd_area_m2 * 1e12)
    assert np.array_equal(sensor_get(sensor, "electrons per area"), expected_m2)
    assert np.array_equal(sensor_get(sensor, "electrons per area", "um"), expected_um2)
    assert np.array_equal(sensor_get(sensor, "electrons per area", "um", 2), expected_um2[tiled_pattern == 2])


def test_sensor_get_set_supports_quantization_alias_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    lut = np.array([0.0, 1.0, 2.0], dtype=float)

    sensor = sensor_set(sensor, "nbits", 12)
    sensor = sensor_set(sensor, "quantization", "12 bit")
    sensor = sensor_set(sensor, "quantizatonlut", lut)

    quantization = sensor_get(sensor, "quantization")
    quantization_method = sensor_get(sensor, "quantizationmethod")
    quantization_struct = sensor_get(sensor, "quantizationstructure")

    assert quantization == "12 bit"
    assert quantization_method == "12 bit"
    assert sensor_get(sensor, "nbits") == 12
    assert sensor_get(sensor, "bits") == 12
    assert np.array_equal(sensor_get(sensor, "quantizatonlut"), lut)
    assert np.array_equal(sensor_get(sensor, "quantizationlut"), lut)
    assert quantization_struct["bits"] == 12
    assert quantization_struct["method"] == "12 bit"
    assert np.array_equal(quantization_struct["lut"], lut)
    assert sensor_get(sensor, "maxdigital") == float((2**12) - sensor_get(sensor, "zero level"))
    assert sensor_get(sensor, "maxoutput") == sensor.fields["pixel"]["voltage_swing"]

    sensor = sensor_set(sensor, "quantizationstructure", {"bits": 8, "method": "8 bit", "lut": np.array([0.0, 0.5], dtype=float)})

    assert sensor_get(sensor, "quantizationmethod") == "8 bit"
    assert sensor_get(sensor, "nbits") == 8
    assert sensor_get(sensor, "bits") == 8
    assert np.array_equal(sensor_get(sensor, "lut"), np.array([0.0, 0.5], dtype=float))


def test_sensor_compute_uses_stored_noise_seed_when_seed_omitted(asset_store) -> None:
    scene = scene_create("uniform d65")
    oi = oi_compute(oi_create(), scene)

    sensor_a = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_a = sensor_set(sensor_a, "noise seed", 11)
    sensor_b = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_b = sensor_set(sensor_b, "noise seed", 11)
    sensor_c = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_c = sensor_set(sensor_c, "noise seed", 13)

    result_a = sensor_compute(sensor_a, oi)
    result_b = sensor_compute(sensor_b, oi)
    result_c = sensor_compute(sensor_c, oi)

    assert np.allclose(np.asarray(result_a.data["volts"], dtype=float), np.asarray(result_b.data["volts"], dtype=float))
    assert not np.allclose(np.asarray(result_a.data["volts"], dtype=float), np.asarray(result_c.data["volts"], dtype=float))


def test_sensor_set_cfa_round_trips_matlab_style_struct(asset_store) -> None:
    sensor = sensor_create("rgbw", asset_store=asset_store)
    cfa = sensor_get(sensor, "cfa")
    replacement = {
        "pattern": np.array([[4, 3], [2, 1]], dtype=int),
        "unitBlock": cfa["unitBlock"],
    }

    sensor = sensor_set(sensor, "cfa", replacement)

    assert np.array_equal(sensor_get(sensor, "pattern"), replacement["pattern"])
    assert np.array_equal(sensor_get(sensor, "cfa")["pattern"], replacement["pattern"])
    assert sensor_get(sensor, "cfaname") == "RGBW"


def test_sensor_create_rgbw_and_rccc_presets_expose_multichannel_cfas(asset_store) -> None:
    rgbw = sensor_create("rgbw", asset_store=asset_store)
    rccc = sensor_create("rccc", asset_store=asset_store)

    assert sensor_get(rgbw, "nfilters") == 4
    assert sensor_get(rgbw, "filtercolorletters") == "rgbw"
    assert sensor_get(rgbw, "filtercolorletterscell") == ["r", "g", "b", "w"]
    assert sensor_get(rgbw, "filterplotcolors") == "rgbk"
    assert np.array_equal(sensor_get(rgbw, "patterncolors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert sensor_get(rccc, "nfilters") == 2
    assert sensor_get(rccc, "filtercolorletters") == "rw"
    assert sensor_get(rccc, "filtercolorletterscell") == ["r", "w"]
    assert sensor_get(rccc, "filterplotcolors") == "rk"
    assert np.array_equal(sensor_get(rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


def test_sensor_create_vendor_models_load_upstream_rgbw_and_rccc_metadata(asset_store) -> None:
    mt9v024_rgbw = sensor_create("mt9v024", "rgbw", asset_store=asset_store)
    mt9v024_rccc = sensor_create("mt9v024", None, "rccc", asset_store=asset_store)
    ar0132at_rgbw = sensor_create("ar0132at", "rgbw", asset_store=asset_store)
    ar0132at_rccc = sensor_create("ar0132at", None, "rccc", asset_store=asset_store)

    assert mt9v024_rgbw.name == "MTV9V024-RGBW"
    assert mt9v024_rgbw.fields["size"] == (480, 752)
    assert np.allclose(mt9v024_rgbw.fields["pixel"]["size_m"], np.array([6e-6, 6e-6]))
    assert sensor_get(mt9v024_rgbw, "filtercolorletters") == "rgbw"
    assert np.array_equal(sensor_get(mt9v024_rgbw, "patterncolors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert mt9v024_rccc.name == "MTV9V024-RCCC"
    assert sensor_get(mt9v024_rccc, "filtercolorletters") == "rw"
    assert np.array_equal(sensor_get(mt9v024_rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))

    assert ar0132at_rgbw.name == "AR0132AT-RGBW"
    assert ar0132at_rgbw.fields["size"] == (960, 1280)
    assert np.allclose(ar0132at_rgbw.fields["pixel"]["size_m"], np.array([3.751e-6, 3.751e-6]))
    assert sensor_get(ar0132at_rgbw, "filtercolorletters") == "rgbw"
    assert np.array_equal(sensor_get(ar0132at_rgbw, "patterncolors"), np.array([["r", "g"], ["w", "b"]], dtype="<U1"))

    assert ar0132at_rccc.name == "AR0132AT-RCCC"
    assert sensor_get(ar0132at_rccc, "filtercolorletters") == "rw"
    assert np.array_equal(sensor_get(ar0132at_rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


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


def test_sensor_compute_supports_vendor_rgbw_qe_sampling(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create("mt9v024", "rgbw", asset_store=asset_store), "noise flag", 0)
    sensor = sensor_set(sensor, "integration time", 0.01)

    result = sensor_compute(sensor, oi, seed=0)

    assert result.data["volts"].shape == sensor.fields["size"]
    assert np.all(result.data["volts"] >= 0.0)


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
        "sensoretendue",
        np.full(baseline_sensor.fields["size"], 0.5, dtype=float),
    )

    baseline = sensor_compute(baseline_sensor, oi, seed=0)
    attenuated = sensor_compute(attenuated_sensor, oi, seed=0)

    assert np.allclose(attenuated.data["volts"], baseline.data["volts"] * 0.5)
    assert np.allclose(sensor_get(attenuated, "sensoretendue"), 0.5)


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


def test_camera_create_supports_rgbw_and_rccc_sensor_variants(asset_store) -> None:
    rgbw_camera = camera_create("rgbw", asset_store=asset_store)
    rccc_camera = camera_create("rccc", asset_store=asset_store)

    assert camera_get(rgbw_camera, "sensor filter color letters") == "rgbw"
    assert np.array_equal(
        camera_get(rgbw_camera, "sensor pattern colors"),
        np.array([["r", "g"], ["b", "w"]], dtype="<U1"),
    )
    assert camera_get(rccc_camera, "sensor filter color letters") == "rw"
    assert np.array_equal(
        camera_get(rccc_camera, "sensor pattern colors"),
        np.array([["w", "w"], ["w", "r"]], dtype="<U1"),
    )


def test_camera_create_supports_vendor_sensor_variants(asset_store) -> None:
    mt9v024_rgbw = camera_create("mt9v024", "rgbw", asset_store=asset_store)
    ar0132at_rccc = camera_create("ar0132at", "rccc", asset_store=asset_store)

    assert mt9v024_rgbw.fields["sensor"].name == "MTV9V024-RGBW"
    assert camera_get(mt9v024_rgbw, "sensor filter color letters") == "rgbw"
    assert ar0132at_rccc.fields["sensor"].name == "AR0132AT-RCCC"
    assert camera_get(ar0132at_rccc, "sensor filter color letters") == "rw"


def test_camera_compute_supports_vendor_sensor_variants(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    mt9v024_rgbw = camera_compute(camera_create("mt9v024", "rgbw", asset_store=asset_store), scene, asset_store=asset_store)
    ar0132at_rccc = camera_compute(camera_create("ar0132at", "rccc", asset_store=asset_store), scene, asset_store=asset_store)

    assert mt9v024_rgbw.fields["ip"].data["result"].shape[:2] == mt9v024_rgbw.fields["sensor"].fields["size"]
    assert ar0132at_rccc.fields["ip"].data["result"].shape[:2] == ar0132at_rccc.fields["sensor"].fields["size"]


def test_run_python_case_with_context_returns_pipeline_objects(asset_store) -> None:
    case = run_python_case_with_context("camera_default_pipeline", asset_store=asset_store)

    assert case.payload["result"].shape[:2] == tuple(case.context["sensor"].fields["size"])
    assert np.array_equal(case.payload["oi_photons"], case.context["oi"].data["photons"])
    assert np.array_equal(case.payload["sensor_volts"], case.context["sensor"].data["volts"])
    assert case.context["camera"].fields["ip"] is case.context["ip"]


def test_run_python_case_supports_checkerboard_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_checkerboard_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_uniform_bb_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_bb_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_frequency_orientation_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_frequency_orientation_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_harmonic_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_harmonic_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_sweep_frequency_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_sweep_frequency_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_reflectance_chart_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_reflectance_chart_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (24, 24)
    assert np.array_equal(case.payload["chart_rowcol"], np.array([3, 3]))
    assert case.payload["chart_index_map"].shape == (24, 24)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_star_pattern_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_star_pattern_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_psf_default_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf_default_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert case.context["oi"].fields["optics"]["model"] == "shiftinvariant"
    assert case.context["oi"].fields["optics"]["compute_method"] == "opticsotf"


def test_run_python_case_supports_custom_otf_flare_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_custom_otf_flare_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert case.payload["otf_abs550"].shape == (case.payload["fy"].size, case.payload["fx"].size)
    assert case.payload["interp_otf_abs550"].shape[0] > case.payload["photons"].shape[0]
    assert case.context["oi"].fields["optics"]["compute_method"] == "opticsotf"


def test_run_python_case_supports_optics_psf_to_otf_flare_parity_case(asset_store) -> None:
    case = run_python_case_with_context("optics_psf_to_otf_flare_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["fy"].ndim == 1
    assert case.payload["otf_abs550_row"].shape == case.payload["fx"].shape
    assert case.payload["otf_abs550_center"].shape == (33, 33)
    assert np.max(case.payload["otf_abs550_center"]) > 0.0


def test_run_python_case_supports_wvf_defocus_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_wvf_defocus_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert np.isclose(float(case.payload["defocus"]), 2.0)
    assert np.isclose(float(case.payload["vertical_astigmatism"]), 0.5)


def test_run_python_case_supports_wvf_script_defocus_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_wvf_script_defocus_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert np.isclose(float(case.payload["defocus_zcoeff"]), 1.5)
    assert np.isclose(float(case.payload["pupil_diameter_mm"]), 3.0)
    assert case.payload["f_number"] > 0.0


def test_run_python_case_supports_wvf_spatial_sampling_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_spatial_sampling_small", asset_store=asset_store)

    assert int(case.payload["npixels"]) == 201
    assert int(case.payload["calc_nwave"]) == int(np.asarray(case.payload["wave"], dtype=float).size)
    assert case.payload["psf_xaxis_um"].shape == case.payload["psf_xaxis_data"].shape
    assert case.payload["pupil_positions_mm"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_amp_row"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_phase_row"].shape == (int(case.payload["npixels"]),)
    assert float(case.payload["psf_sample_spacing_arcmin"]) > 0.0


def test_run_python_case_supports_lswavelength_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_lswavelength_diffraction_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["wavelength"].ndim == 1
    assert case.payload["lsWave"].shape == (case.payload["wavelength"].size, case.payload["x"].size)
    assert np.all(case.payload["lsWave"] >= 0.0)


def test_run_python_case_supports_psf550_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf550_diffraction_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 2
    assert case.payload["y"].ndim == 2
    assert case.payload["psf"].shape == case.payload["x"].shape == case.payload["y"].shape
    assert np.all(case.payload["psf"] >= 0.0)


def test_run_python_case_supports_unit_frequency_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_unit_frequency_list", asset_store=asset_store)

    assert np.allclose(case.payload["even"], np.asarray(case.payload["even"], dtype=float))
    assert np.allclose(case.payload["odd"], np.asarray(case.payload["odd"], dtype=float))
    assert case.payload["even"].shape == (50,)
    assert case.payload["odd"].shape == (51,)


def test_run_python_case_supports_energy_quanta_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_energy_quanta_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["energy"].shape == (31,)
    assert case.payload["photons"].shape == (31,)
    assert np.allclose(case.payload["energy_roundtrip"], case.payload["energy"])


def test_run_python_case_supports_energy_quanta_matrix_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_energy_quanta_matrix", asset_store=asset_store)

    assert case.payload["wave"].shape == (3,)
    assert case.payload["energy"].shape == (3, 2)
    assert case.payload["photons"].shape == (3, 2)
    assert np.allclose(case.payload["energy_roundtrip"], case.payload["energy"])


def test_run_python_case_supports_blackbody_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_blackbody_energy_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["temperatures"].shape == (2,)
    assert case.payload["energy"].shape == (31, 2)
    assert np.all(np.asarray(case.payload["energy"], dtype=float) > 0.0)


def test_run_python_case_supports_blackbody_quanta_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_blackbody_quanta_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["temperatures"].shape == (2,)
    assert case.payload["photons"].shape == (31, 2)
    assert np.all(np.asarray(case.payload["photons"], dtype=float) > 0.0)


def test_run_python_case_supports_ie_param_format_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_ie_param_format_string", asset_store=asset_store)

    assert case.payload["original"] == "Exposure Time"
    assert case.payload["formatted"] == "exposuretime"


def test_run_python_case_supports_xyz_from_energy_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_from_energy_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["energy"].shape == (31,)
    assert case.payload["xyz"].shape == (3,)
    assert np.all(np.asarray(case.payload["xyz"], dtype=float) > 0.0)


def test_run_python_case_supports_xyz_to_luv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_luv_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert case.payload["luv"].shape == (3,)
    assert case.payload["luv"][0] > 0.0


def test_run_python_case_supports_xyz_to_lab_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_lab_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert case.payload["lab"].shape == (3,)
    assert case.payload["lab"][0] > 0.0


def test_run_python_case_supports_xyz_to_uv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_uv_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["uv"].shape == (2,)
    assert np.all(np.asarray(case.payload["uv"], dtype=float) > 0.0)


def test_run_python_case_supports_cct_from_uv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_cct_from_uv_1d", asset_store=asset_store)

    assert case.payload["uv"].shape == (2,)
    assert float(case.payload["cct_k"]) > 0.0


def test_run_python_case_supports_delta_e_ab_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_delta_e_ab_1976_1d", asset_store=asset_store)

    assert case.payload["xyz1"].shape == (3,)
    assert case.payload["xyz2"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert float(case.payload["delta_e"]) > 0.0


def test_run_python_case_supports_metrics_spd_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_angle_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (3,)
    assert case.payload["spd1"].shape == (3,)
    assert case.payload["spd2"].shape == (3,)
    assert np.isclose(float(case.payload["angle"]), 90.0)


def test_run_python_case_supports_metrics_spd_cielab_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_cielab_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["spd1"].shape == (31,)
    assert case.payload["spd2"].shape == (31,)
    assert float(case.payload["delta_e"]) > 0.0
    assert case.payload["xyz1"].shape == (3,)
    assert case.payload["xyz2"].shape == (3,)
    assert case.payload["lab1"].shape == (3,)
    assert case.payload["lab2"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)


def test_run_python_case_supports_metrics_spd_mired_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_mired_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["spd1"].shape == (31,)
    assert case.payload["spd2"].shape == (31,)
    assert float(case.payload["mired"]) > 0.0
    assert case.payload["uv"].shape == (2, 2)
    assert case.payload["cct_k"].shape == (2,)
