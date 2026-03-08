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
    scene_get,
    scene_set,
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


def test_scene_get_depth_map_defaults_to_scene_distance(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    depth_map = scene_get(scene, "depth map")

    assert depth_map.shape == scene_get(scene, "size")
    assert np.allclose(depth_map, scene_get(scene, "distance"))
    assert np.allclose(scene_get(scene, "depth range"), np.array([scene_get(scene, "distance"), scene_get(scene, "distance")]))


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
    optics = oi_get(oi, "optics")
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

    assert oi_get(oi, "model") == "raytrace"
    assert oi_get(oi, "compute method") == ""
    assert np.isclose(oi_get(oi, "focal length"), 0.001999989, rtol=1e-6, atol=1e-9)
    assert np.isclose(oi_get(oi, "fnumber"), 4.999973)
    assert np.isclose(oi_get(oi, "rt object distance"), 2.0)
    assert np.isclose(oi_get(oi, "rtfov"), 38.72116733777534)
    assert oi_get(oi, "raytrace optics name") == "Asphere 2mm"
    assert oi_get(oi, "rtpsffieldheight").shape == (21,)
    assert np.allclose(oi_get(oi, "rtpsffieldheight", "mm"), oi_get(oi, "raytrace")["psf"]["field_height_mm"])
    assert np.allclose(oi_get(oi, "rtpsfsamplespacing"), np.array([2.5e-7, 2.5e-7]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.25, 0.25]))
    assert np.array_equal(oi_get(oi, "rtpsfwavelength"), np.array([400.0, 475.0, 550.0, 625.0, 700.0]))


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
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), rel_illum["field_height_mm"])
    assert np.allclose(oi_get(oi, "rtgeomfieldheight", "mm"), geometry["field_height_mm"])
    assert np.isclose(oi_get(oi, "rtgeommaxfieldheight", "mm"), np.max(geometry["field_height_mm"]))
    assert np.array_equal(oi_get(oi, "rtriwavelength"), rel_illum["wavelength_nm"])
    assert np.array_equal(oi_get(oi, "rtgeomwavelength"), geometry["wavelength_nm"])
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
    assert sampled is not None
    assert sampled.ndim == 5
    assert np.array_equal(oi_get(result, "psfwavelength"), np.array([550.0]))
    assert oi_get(result, "rtpsfsize")[0] >= 3
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
    assert np.allclose(oi_get(result, "psf image heights", "m"), oi_get(result, "psf image heights") / 1e3)


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
        "imgHeight": np.array([0.0, 1.5], dtype=float),
        "wavelength": np.array([550.0], dtype=float),
        "opticsName": "Synthetic RT",
    }

    oi = oi_set(oi, "shift variant structure", psf_struct)

    sampled = oi_get(oi, "sampledRTpsf")
    assert sampled.shape == (2, 2, 1, 3, 3)
    assert np.array_equal(oi_get(oi, "psf sample angles"), np.array([0.0, 180.0]))
    assert np.isclose(oi_get(oi, "psf angle step"), 180.0)
    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.5]))
    assert np.allclose(oi_get(oi, "psf image heights", "m"), np.array([0.0, 0.0015]))
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([550.0]))
    assert oi_get(oi, "raytrace optics name") == "Synthetic RT"


def test_oi_get_set_raytrace_psf_metadata_before_compute(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "psf image heights", np.array([0.0, 1.0, 2.0], dtype=float))
    oi = oi_set(oi, "psf wavelength", np.array([450.0, 550.0], dtype=float))
    oi = oi_set(oi, "raytrace optics name", "Manual RT")

    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.0, 2.0]))
    assert oi_get(oi, "psf image heights n") == 3
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([450.0, 550.0]))
    assert oi_get(oi, "psf wavelength n") == 2
    assert oi_get(oi, "raytrace optics name") == "Manual RT"


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

    assert np.isclose(oi_get(oi, "optics rtfnumber"), oi_get(oi, "rtfnumber"))
    assert np.allclose(oi_get(oi, "optics rtpsfspacing", "um"), oi_get(oi, "rtpsfspacing", "um"))
    assert np.allclose(oi_get(oi, "optics rtgeomfieldheight", "mm"), oi_get(oi, "rtgeomfieldheight", "mm"))

    oi = oi_set(oi, "optics rtrefwave", 530.0)
    oi = oi_set(oi, "optics rtpsfspacing", np.array([0.0004, 0.0006], dtype=float))
    oi = oi_set(oi, "optics rtcomputespacing", 3e-6)

    assert np.isclose(oi_get(oi, "rtrefwave"), 530.0)
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.4, 0.6]))
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 3.0)


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
