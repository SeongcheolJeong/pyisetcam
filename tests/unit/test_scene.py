from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import pytest
import pyisetcam.scene as scene_module

from pyisetcam import (
    blackbody,
    buildPyramid,
    build_pyramid,
    display_create,
    display_get,
    exiftool_depth_from_file,
    exiftool_info,
    finalTouch,
    final_touch,
    font_create,
    getPFMraw,
    getpfmraw,
    haarPyramid,
    haar_pyramid,
    imNorm,
    im_norm,
    modulateFlip,
    modulateFlipShift,
    modulate_flip_shift,
    mo_target,
    padReflect,
    padReflectNeg,
    pad_reflect,
    pad_reflect_neg,
    qmfPyramid,
    qmf_pyramid,
    rangeCompressionLum,
    range_compression_lum,
    reconsHaarPyramid,
    reconsPyramid,
    reconsQmfPyramid,
    recons_haar_pyramid,
    recons_pyramid,
    recons_qmf_pyramid,
    scene_adjust_illuminant,
    scene_create,
    scene_from_ddf_file,
    scene_from_file,
    scene_get,
    scene_list,
    scene_set,
    scene_sdr,
    scene_to_file,
)


def test_scene_create_default_macbeth(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    assert photons.shape == (64, 96, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_create_list_alias_matches_public_listing() -> None:
    listing = scene_create("scene list")
    assert isinstance(listing, str)
    assert listing == scene_list()
    assert "letter" in listing
    assert "multispectral" in listing


def test_scene_create_rgb_multispectral_and_monochrome_shells(asset_store) -> None:
    rgb = scene_create("rgb", asset_store=asset_store)
    multispectral = scene_create("hyperspectral", asset_store=asset_store)
    monochrome = scene_create("unispectral", asset_store=asset_store)

    assert scene_get(rgb, "name") == "rgb"
    assert scene_get(multispectral, "name") == "multispectral"
    assert scene_get(monochrome, "name") == "monochrome"
    assert scene_get(rgb, "photons").shape == (1, 1, 31)
    assert scene_get(multispectral, "photons").shape == (1, 1, 31)
    assert scene_get(monochrome, "photons").shape == (1, 1, 1)
    np.testing.assert_array_equal(scene_get(monochrome, "wave"), np.array([550.0]))
    assert scene_get(rgb, "illuminant comment") == "D65.mat"
    assert scene_get(multispectral, "illuminant comment") == "D65.mat"
    assert scene_get(monochrome, "illuminant comment") == "D65.mat"
    assert np.isclose(scene_get(rgb, "mean luminance", asset_store=asset_store), 0.0, atol=1e-12)

    resized = scene_set(rgb.clone(), "photons", np.ones((2, 3, scene_get(rgb, "nwave")), dtype=float))
    assert tuple(scene_get(resized, "size")) == (2, 3)
    assert scene_get(resized, "fov") == scene_get(rgb, "fov")


def test_scene_create_ramp_equal_photon_alias_matches_ramp(asset_store) -> None:
    ramp = scene_create("ramp", 32, 128.0, asset_store=asset_store)
    alias = scene_create("ramp equal photon", 32, 128.0, asset_store=asset_store)

    assert scene_get(alias, "name") == scene_get(ramp, "name")
    np.testing.assert_array_equal(scene_get(alias, "wave"), scene_get(ramp, "wave"))
    np.testing.assert_allclose(np.asarray(scene_get(alias, "photons"), dtype=float), np.asarray(scene_get(ramp, "photons"), dtype=float))
    assert np.isclose(
        scene_get(alias, "mean luminance", asset_store=asset_store),
        scene_get(ramp, "mean luminance", asset_store=asset_store),
        rtol=1e-10,
        atol=1e-10,
    )


def test_ramp_family_dispatch_replays_matlab_default_size(asset_store) -> None:
    ramp_default = scene_create("ramp", asset_store=asset_store)
    ramp_explicit = scene_create("ramp", 256, 256.0, asset_store=asset_store)
    linear_default = scene_create("linear intensity ramp", asset_store=asset_store)
    linear_explicit = scene_create("linear intensity ramp", 256, 256.0, asset_store=asset_store)
    exp_default = scene_create("exponential intensity ramp", asset_store=asset_store)
    exp_explicit = scene_create("exponential intensity ramp", 256, 256.0, asset_store=asset_store)

    np.testing.assert_allclose(np.asarray(scene_get(ramp_default, "photons"), dtype=float), np.asarray(scene_get(ramp_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(scene_get(linear_default, "photons"), dtype=float), np.asarray(scene_get(linear_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(scene_get(exp_default, "photons"), dtype=float), np.asarray(scene_get(exp_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    assert tuple(scene_get(ramp_default, "size")) == (256, 256)
    assert tuple(scene_get(linear_default, "size")) == (256, 256)
    assert tuple(scene_get(exp_default, "size")) == (256, 256)


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


def test_scene_create_supports_macbeth_illuminant_variants(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 30.0, dtype=float)
    ir_wave = np.arange(700.0, 901.0, 40.0, dtype=float)

    d50 = scene_create("macbeth d50", 8, wave, asset_store=asset_store)
    illc = scene_create("macbeth illc", 8, wave, asset_store=asset_store)
    fluor = scene_create("macbeth fluor", 8, wave, asset_store=asset_store)
    custom = scene_create("macbeth custom reflectance", 8, wave, "macbethChart.mat", asset_store=asset_store)
    ee_ir = scene_create("macbeth ee_ir", 8, ir_wave, asset_store=asset_store)

    assert scene_get(d50, "name") == "Macbeth D50"
    assert scene_get(illc, "name") == "Macbeth Ill C"
    assert scene_get(fluor, "name") == "Macbeth Fluorescent"
    assert scene_get(custom, "name") == "Macbeth D65"
    assert scene_get(ee_ir, "name") == "Macbeth IR"
    assert scene_get(d50, "illuminant comment") == "D50.mat"
    assert scene_get(illc, "illuminant comment") == "illuminantC.mat"
    assert scene_get(fluor, "illuminant comment") == "Fluorescent.mat"
    assert scene_get(custom, "illuminant comment") == "D65.mat"
    assert scene_get(ee_ir, "illuminant comment") == "equalEnergy"
    assert scene_get(d50, "photons").shape == (32, 48, wave.size)
    assert scene_get(ee_ir, "photons").shape == (32, 48, ir_wave.size)
    np.testing.assert_array_equal(np.asarray(scene_get(ee_ir, "wave"), dtype=float), ir_wave)
    assert np.isclose(scene_get(d50, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(ee_ir, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert not np.allclose(scene_get(d50, "illuminant energy"), scene_get(custom, "illuminant energy"))
    assert not np.allclose(scene_get(illc, "illuminant energy"), scene_get(custom, "illuminant energy"))
    assert np.allclose(
        np.asarray(scene_get(ee_ir, "illuminant energy"), dtype=float),
        float(np.asarray(scene_get(ee_ir, "illuminant energy"), dtype=float).reshape(-1)[0]),
    )


def test_macbeth_dispatch_accepts_empty_patch_size_placeholder(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    default_placeholder = scene_create("default", None, wave, asset_store=asset_store)
    default_reference = scene_create("default", 16, wave, asset_store=asset_store)
    macbeth_placeholder = scene_create("macbeth", np.array([], dtype=float), wave, asset_store=asset_store)
    macbeth_reference = scene_create("macbeth", 16, wave, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(default_placeholder, "photons"), scene_get(default_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(macbeth_placeholder, "photons"), scene_get(macbeth_reference, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(default_placeholder, "size")) == (64, 96)
    assert tuple(scene_get(macbeth_placeholder, "size")) == (64, 96)
    assert np.array_equal(np.asarray(scene_get(default_placeholder, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(macbeth_placeholder, "wave"), dtype=float), wave)


def test_scene_create_moire_orient_replays_green_target_plane(asset_store) -> None:
    params = {"sceneSize": 96, "f": 1.0 / 1200.0}
    scene = scene_create("moire orient", params, asset_store=asset_store)
    positional = scene_create("moire orient", 96, 1.0 / 1200.0, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    positional_photons = np.asarray(scene_get(positional, "photons"), dtype=float)
    expected = np.clip(np.asarray(mo_target("sinusoidalim", params), dtype=float)[:, :, 1], 1.0e-4, 1.0)
    expected = expected / np.max(expected)
    actual = photons[:, :, 0] / np.max(photons[:, :, 0])

    assert scene_get(scene, "name") == "MOTarget"
    assert scene_get(positional, "name") == "MOTarget"
    assert photons.shape[:2] == (96, 96)
    assert positional_photons.shape[:2] == (96, 96)
    np.testing.assert_allclose(photons[:, :, 0], photons[:, :, -1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(positional_photons, photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_create_letter_branch_reuses_font_pipeline(asset_store) -> None:
    font = font_create("A", "Georgia", 18, asset_store=asset_store)
    scene = scene_create("letter", font, "LCD-Apple", asset_store=asset_store)
    shortcut = scene_create("letter", "A", 18, "Georgia", "LCD-Apple", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    shortcut_photons = np.asarray(scene_get(shortcut, "photons"), dtype=float)

    assert scene_get(scene, "name") == font["name"]
    assert scene_get(shortcut, "name") == font["name"]
    assert photons.ndim == 3
    assert photons.shape[2] == scene_get(scene, "nwave")
    assert float(np.max(photons)) > float(np.min(photons))
    assert scene_get(scene, "fov") > 0.0
    np.testing.assert_allclose(shortcut_photons, photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(scene_get(shortcut, "wave"), dtype=float),
        np.asarray(scene_get(scene, "wave"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )


def test_uniform_family_accepts_empty_size_placeholders(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    uniform_placeholder = scene_create("uniform", [], wave, asset_store=asset_store)
    uniform_reference = scene_create("uniform", 32, wave, asset_store=asset_store)
    d65_placeholder = scene_create("uniform d65", np.array([], dtype=float), wave, asset_store=asset_store)
    d65_reference = scene_create("uniform d65", 32, wave, asset_store=asset_store)
    ep_placeholder = scene_create("uniform ep", [], wave, asset_store=asset_store)
    ep_reference = scene_create("uniform ep", 32, wave, asset_store=asset_store)
    bb_placeholder = scene_create("uniform bb", [], 4500, wave, asset_store=asset_store)
    bb_reference = scene_create("uniform bb", 32, 4500, wave, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(uniform_placeholder, "photons"), scene_get(uniform_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(d65_placeholder, "photons"), scene_get(d65_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(ep_placeholder, "photons"), scene_get(ep_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(bb_placeholder, "photons"), scene_get(bb_reference, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(uniform_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(d65_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(ep_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(bb_placeholder, "size")) == (32, 32)
    np.testing.assert_array_equal(np.asarray(scene_get(bb_placeholder, "wave"), dtype=float), wave)


def test_uniform_monochromatic_dispatch_accepts_size_first_docs_form(asset_store) -> None:
    size_first = scene_create("uniform monochromatic", 12, 550, asset_store=asset_store)
    wave_first = scene_create("uniform monochromatic", 550, 12, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(size_first, "photons"), scene_get(wave_first, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(size_first, "size")) == (12, 12)
    np.testing.assert_array_equal(np.asarray(scene_get(size_first, "wave"), dtype=float), np.array([550.0], dtype=float))
    assert np.isclose(scene_get(size_first, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_from_file_supports_multispectral_mat_files(asset_store) -> None:
    scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/Feng_Office-hdrs.mat"),
        "multispectral",
        200.0,
        asset_store=asset_store,
    )

    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    illuminant = scene_get(scene, "illuminant photons")

    assert photons.shape == (506, 759, 31)
    assert wave.shape == (31,)
    assert illuminant.shape == photons.shape
    assert scene_get(scene, "illuminant format") == "spatial spectral"
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 200.0, rtol=5e-2)


def test_scene_from_ddf_file_matches_scene_from_file_without_embedded_depth(tmp_path, asset_store) -> None:
    image = np.linspace(0.0, 1.0, 4 * 6 * 3, dtype=float).reshape(4, 6, 3)
    image_path = tmp_path / "ddf-source.png"
    iio.imwrite(image_path, np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8))

    expected = scene_from_file(image_path, "rgb", 75.0, "LCD-Apple.mat", asset_store=asset_store)
    actual = scene_from_ddf_file(image_path, "rgb", 75.0, "LCD-Apple.mat", asset_store=asset_store)

    np.testing.assert_array_equal(np.asarray(scene_get(actual, "size"), dtype=int), np.asarray(scene_get(expected, "size"), dtype=int))
    np.testing.assert_allclose(np.asarray(scene_get(actual, "photons"), dtype=float), np.asarray(scene_get(expected, "photons"), dtype=float))
    assert actual.name == expected.name
    assert np.isclose(
        scene_get(actual, "mean luminance", asset_store=asset_store),
        scene_get(expected, "mean luminance", asset_store=asset_store),
        rtol=1e-10,
        atol=1e-10,
    )


def test_exiftool_info_supports_json_format(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"jpg")

    class Result:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    def fake_run(command, check=False, capture_output=True, text=True):  # type: ignore[no-untyped-def]
        del check, capture_output, text
        assert command[:2] == ["/usr/bin/exiftool", "-j"]
        return Result('[{"Orientation":"Rotate 90 CW","DepthMapNear":1.0}]')

    monkeypatch.setattr(scene_module.shutil, "which", lambda name: "/usr/bin/exiftool")
    monkeypatch.setattr(scene_module.subprocess, "run", fake_run)

    info = exiftool_info(image_path, "format", "json")
    assert info["Orientation"] == "Rotate 90 CW"
    assert np.isclose(float(info["DepthMapNear"]), 1.0)


def test_exiftool_depth_from_file_decodes_meter_payload(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"jpg")
    payload_path = tmp_path / "depth.png"
    encoded = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    iio.imwrite(payload_path, encoded)

    monkeypatch.setattr(
        scene_module,
        "exiftool_info",
        lambda *args, **kwargs: {
            "DepthMapUnits": "Meters",
            "DepthMapNear": 1.0,
            "DepthMapFar": 5.0,
            "ImageHeight": 2,
            "ImageWidth": 2,
        },
    )
    monkeypatch.setattr(scene_module, "_exiftool_depth_payload", lambda path: payload_path.read_bytes())

    depth = exiftool_depth_from_file(image_path, "type", "GooglePixel")
    expected = 1.0 + (encoded.astype(float) / 255.0) * 4.0

    np.testing.assert_allclose(depth, expected, rtol=0.0, atol=1e-6)


def test_hdr_helper_wrappers_cover_padding_and_pyramid_reconstruction() -> None:
    small = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    expected_pad = np.array(
        [
            [4.0, 3.0, 4.0, 3.0],
            [2.0, 1.0, 2.0, 1.0],
            [4.0, 3.0, 4.0, 3.0],
            [2.0, 1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    expected_pad_neg = np.array(
        [
            [-4.0, 3.0, 4.0, -3.0],
            [-2.0, 1.0, 2.0, -1.0],
            [-4.0, 3.0, 4.0, -3.0],
            [-2.0, 1.0, 2.0, -1.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(pad_reflect(small, 1), expected_pad)
    np.testing.assert_allclose(padReflect(small, 1), expected_pad)
    np.testing.assert_allclose(pad_reflect_neg(small, 1, 1, 1, 0), expected_pad_neg)
    np.testing.assert_allclose(padReflectNeg(small, 1, 1, 1, 0), expected_pad_neg)

    image = np.arange(1.0, 65.0, dtype=float).reshape(8, 8) / 64.0
    haar = haar_pyramid(image, 2)
    alias_haar = haarPyramid(image, 2)
    built_haar, filt_num = build_pyramid(image, 2, "haar")
    alias_built_haar, alias_filt_num = buildPyramid(image, 2, "haar")

    assert filt_num == 3
    assert alias_filt_num == 3
    np.testing.assert_allclose(alias_haar, haar)
    np.testing.assert_allclose(built_haar, haar)
    np.testing.assert_allclose(alias_built_haar, haar)
    haar_reconstructed = recons_haar_pyramid(haar)
    alias_haar_reconstructed = reconsHaarPyramid(haar)
    dispatch_haar_reconstructed = recons_pyramid(haar, 3, "haar")
    alias_dispatch_haar_reconstructed = reconsPyramid(haar, 3, "haar")
    np.testing.assert_allclose(alias_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dispatch_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(alias_dispatch_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[1:, 1:], image[1:, 1:], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[0, 1:] - image[0, 1:], np.full(7, 1.0 / 16.0), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[1:, 0] - image[1:, 0], np.full(7, 1.0 / 128.0), atol=1e-10, rtol=1e-10)
    assert np.isclose(float(haar_reconstructed[0, 0] - image[0, 0]), 9.0 / 128.0, atol=1e-10, rtol=1e-10)

    qmf = qmf_pyramid(image, 2)
    alias_qmf = qmfPyramid(image, 2)
    built_qmf, qmf_filt_num = build_pyramid(image, 2, "qmf")
    np.testing.assert_allclose(alias_qmf, qmf)
    np.testing.assert_allclose(built_qmf, qmf)
    assert qmf_filt_num == 3
    np.testing.assert_allclose(recons_qmf_pyramid(qmf), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(reconsQmfPyramid(qmf), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(recons_pyramid(qmf, 3, "qmf"), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(reconsPyramid(qmf, 3, "qmf"), image, atol=6e-4, rtol=6e-4)

    with pytest.raises(NotImplementedError):
        build_pyramid(image, 2, "steerable")
    with pytest.raises(NotImplementedError):
        recons_pyramid(haar, 3, "steerable")


def test_hdr_helper_wrappers_cover_normalization_range_touch_filter_and_pfm(tmp_path) -> None:
    image = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=float)
    expected_norm = (image - 1.0) / 4.0
    expected_touch = image + 0.15 * scene_module._hist_equalize_global(np.real(image))
    grayscale = np.linspace(0.1, 1.6, 64, dtype=float).reshape(8, 8)
    expected_range = scene_module._range_compression_lum(grayscale, filt_type="haar", beta=0.6, alpha_a=0.2, ifsharp=0)

    np.testing.assert_allclose(im_norm(image), expected_norm)
    np.testing.assert_allclose(imNorm(image), expected_norm)
    np.testing.assert_allclose(final_touch(image), expected_touch)
    np.testing.assert_allclose(finalTouch(image), expected_touch)
    np.testing.assert_allclose(range_compression_lum(grayscale), expected_range)
    np.testing.assert_allclose(rangeCompressionLum(grayscale), expected_range)
    np.testing.assert_allclose(modulate_flip_shift([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))
    np.testing.assert_allclose(modulateFlip([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))
    np.testing.assert_allclose(modulateFlipShift([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))

    rgb = (np.arange(1.0, 13.0, dtype=np.float32).reshape(2, 2, 3)) / 12.0
    flipped = rgb[::-1, :, :]
    payload = np.concatenate([np.reshape(flipped[:, :, channel].T, -1, order="F") for channel in range(3)]).astype(np.float32)
    pfm_path = tmp_path / "sample.pfm"
    pfm_path.write_bytes(b"P7\n2 2\n1.0\n" + payload.tobytes())

    np.testing.assert_allclose(getpfmraw(pfm_path), rgb, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(getPFMraw(pfm_path), rgb, atol=1e-7, rtol=1e-7)


def test_scene_sdr_prefers_local_cache_for_mat_and_png(tmp_path, asset_store) -> None:
    cached_scene = scene_create("macbeth d65", asset_store=asset_store)
    scene_to_file(tmp_path / "local-scene.mat", cached_scene)

    loaded_scene = scene_sdr(
        "isetcam bitterli",
        "local-scene",
        download_dir=tmp_path,
        asset_store=asset_store,
    )
    assert tuple(scene_get(loaded_scene, "size")) == tuple(scene_get(cached_scene, "size"))
    np.testing.assert_array_equal(np.asarray(scene_get(loaded_scene, "wave"), dtype=float), np.asarray(scene_get(cached_scene, "wave"), dtype=float))
    assert loaded_scene.name == cached_scene.name

    cached_png = (np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 3) % 255
    png_path = tmp_path / "preview.png"
    iio.imwrite(png_path, cached_png)

    loaded_png = scene_sdr("isetcam pharr", "preview.png", download_dir=tmp_path)
    np.testing.assert_array_equal(np.asarray(loaded_png), cached_png)


def test_scene_sdr_rejects_unknown_deposit_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid deposit name"):
        scene_sdr("unknown deposit", "preview.png", download_dir=tmp_path)


def test_supported_pattern_scenes(asset_store) -> None:
    mackay = scene_create("rings rays", asset_store=asset_store)
    checkerboard = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    slanted_bar = scene_create("slanted bar", 64, 0.6, 3.0, asset_store=asset_store)
    freq_orient = scene_create("frequency orientation", asset_store=asset_store)
    harmonic = scene_create("harmonic", asset_store=asset_store)
    sweep = scene_create("sweep frequency", asset_store=asset_store)
    star = scene_create("star pattern", 64, "ee", 6, asset_store=asset_store)
    line = scene_create("line ee", 32, 2, asset_store=asset_store)
    bar = scene_create("bar", 32, 5, asset_store=asset_store)
    point_array = scene_create("point array", 64, 16, "ep", 3, asset_store=asset_store)
    square_array = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    grid_lines = scene_create("grid lines", 64, 16, "ee", 2, asset_store=asset_store)
    white_noise = scene_create("white noise", 32, 20, asset_store=asset_store)
    lstar = scene_create("lstar", [80, 10], 20, 1, asset_store=asset_store)
    uniform_specify = scene_create("uniformEESpecify", 128, np.arange(380.0, 721.0, 10.0, dtype=float), asset_store=asset_store)
    assert scene_get(mackay, "photons").shape[:2] == (256, 256)
    assert scene_get(checkerboard, "photons").shape[:2] == (64, 64)
    assert scene_get(slanted_bar, "photons").shape[:2] == (65, 65)
    assert scene_get(freq_orient, "photons").shape[:2] == (256, 256)
    assert scene_get(harmonic, "photons").shape[:2] == (65, 65)
    assert scene_get(sweep, "photons").shape[:2] == (128, 128)
    assert scene_get(star, "photons").shape[:2] == (64, 64)
    assert scene_get(line, "photons").shape[:2] == (32, 32)
    assert scene_get(bar, "photons").shape[:2] == (32, 32)
    assert scene_get(point_array, "photons").shape[:2] == (64, 64)
    assert scene_get(square_array, "photons").shape[:2] == (64, 64)
    assert scene_get(grid_lines, "photons").shape[:2] == (64, 64)
    assert scene_get(white_noise, "photons").shape[:2] == (32, 32)
    assert scene_get(lstar, "photons").shape[:2] == (80, 200)
    assert scene_get(uniform_specify, "photons").shape[:2] == (128, 128)
    assert np.isclose(scene_get(freq_orient, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(harmonic, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(sweep, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(star, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(mackay, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(square_array, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(lstar, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(uniform_specify, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(freq_orient, "fov"), 10.0)
    assert np.isclose(scene_get(harmonic, "fov"), 1.0)
    assert np.isclose(scene_get(sweep, "fov"), 10.0)
    assert np.isclose(scene_get(mackay, "fov"), 10.0)
    assert np.isclose(scene_get(point_array, "fov"), 40.0)
    assert np.isclose(scene_get(square_array, "fov"), 40.0)
    assert np.isclose(scene_get(grid_lines, "fov"), 40.0)
    assert np.isclose(scene_get(white_noise, "fov"), 1.0)


def test_iso12233_dispatch_accepts_empty_fov_placeholder(asset_store) -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)

    none_placeholder = scene_create("iso12233", 64, 1.33, None, wave, 0.3, asset_store=asset_store)
    empty_placeholder = scene_create("iso12233", 64, 1.33, [], wave, 0.3, asset_store=asset_store)
    explicit_default = scene_create("iso12233", 64, 1.33, 2.0, wave, 0.3, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(none_placeholder, "photons"), scene_get(explicit_default, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(empty_placeholder, "photons"), scene_get(explicit_default, "photons"), rtol=0.0, atol=0.0)
    assert np.isclose(float(scene_get(none_placeholder, "fov")), 2.0)
    assert np.isclose(float(scene_get(empty_placeholder, "fov")), 2.0)


def test_mackay_scene_has_center_mask_and_radial_alias(asset_store) -> None:
    rings = scene_create("rings rays", asset_store=asset_store)
    mackay = scene_create("mackay", asset_store=asset_store)

    rings_plane = scene_get(rings, "photons")[:, :, 0]
    mackay_plane = scene_get(mackay, "photons")[:, :, 0]

    assert np.array_equal(rings_plane, mackay_plane)
    assert rings_plane.shape == (256, 256)
    assert rings_plane[rings_plane.shape[0] // 2, rings_plane.shape[1] // 2] > np.min(rings_plane)


def test_lstar_scene_creates_monotonic_bar_steps(asset_store) -> None:
    scene = scene_create("lstar", [80, 10], 20, 1, asset_store=asset_store)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    bar_means = np.array([np.mean(luminance[:, start : start + 10]) for start in range(0, 200, 10)], dtype=float)

    assert luminance.shape == (80, 200)
    assert np.all(np.diff(bar_means) > 0.0)


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


def test_bar_ee_alias_matches_bar_scene(asset_store) -> None:
    canonical = scene_create("bar", 33, 3, asset_store=asset_store)
    alias = scene_create("bar ee", 33, 3, asset_store=asset_store)

    np.testing.assert_allclose(
        scene_get(alias, "photons"),
        scene_get(canonical, "photons"),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(alias, "size")) == (33, 33)


def test_bar_dispatch_replays_matlab_default_width(asset_store) -> None:
    default_bar = scene_create("bar", asset_store=asset_store)
    explicit_bar = scene_create("bar", 64, 3, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(default_bar, "photons"), dtype=float),
        np.asarray(scene_get(explicit_bar, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(default_bar, "size")) == (64, 64)


def test_squares_alias_matches_square_array_scene(asset_store) -> None:
    canonical = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    alias = scene_create("squares", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)

    np.testing.assert_allclose(
        scene_get(alias, "photons"),
        scene_get(canonical, "photons"),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(alias, "size")) == (64, 64)


def test_disk_array_dispatch_replays_matlab_default_radius(asset_store) -> None:
    default_scene = scene_create("disk array", asset_store=asset_store)
    explicit_scene = scene_create("disk array", 128, 128, np.array([1, 1], dtype=int), asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        np.asarray(scene_get(explicit_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(default_scene, "size")) == (128, 128)


def test_zone_plate_dispatch_accepts_optional_field_of_view_and_wave(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    explicit_fov = scene_create("zone plate", 96, 7.5, wave, asset_store=asset_store)
    placeholder_fov = scene_create("zone plate", 96, [], wave, asset_store=asset_store)
    wave_only = scene_create("zone plate", 96, wave, asset_store=asset_store)

    assert tuple(scene_get(explicit_fov, "size")) == (96, 96)
    assert tuple(scene_get(placeholder_fov, "size")) == (96, 96)
    assert tuple(scene_get(wave_only, "size")) == (96, 96)
    assert np.isclose(scene_get(explicit_fov, "fov"), 7.5, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(placeholder_fov, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(wave_only, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.array_equal(np.asarray(scene_get(explicit_fov, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(placeholder_fov, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(wave_only, "wave"), dtype=float), wave)
    assert np.asarray(scene_get(explicit_fov, "photons"), dtype=float).shape == (96, 96, 3)
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_fov, "photons"), dtype=float),
        np.asarray(scene_get(wave_only, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


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
