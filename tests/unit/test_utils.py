from __future__ import annotations

import numpy as np
import pytest

from pyisetcam import (
    FloydSteinberg,
    HalfToneImage,
    convolvecirc,
    ieCmap,
    ieCropRect,
    ieFindWaveIndex,
    ieLUTDigital,
    ieLUTInvert,
    ieLUTLinear,
    ieParameterOtype,
    ieRadialMatrix,
    ieWave2Index,
    ieXYZFromPhotons,
    imageHparams,
    imageSPD,
    imageSPD2RGB,
    imageInterpolate,
    imageGabor,
    imageMakeMontage,
    imageMontage,
    imageSlantedEdge,
    imageTranslate,
    imageTranspose,
    imagehc2rgb,
    imageBoundingBox,
    imageCentroid,
    imageCircular,
    imageContrast,
    imagescM,
    imagescOPP,
    imagescRGB,
    rgb2dac,
    sceneCreate,
    sceneGet,
    xyz2srgb,
)
from pyisetcam.utils import (
    blackbody,
    convolve_circ,
    energy_to_quanta,
    floyd_steinberg,
    half_tone_image,
    ie_cmap,
    ie_crop_rect,
    ie_fit_line,
    ie_find_wave_index,
    ie_lut_digital,
    ie_lut_invert,
    ie_lut_linear,
    ie_parameter_otype,
    ie_radial_matrix,
    ie_wave2_index,
    image_hparams,
    image_hc2rgb,
    image_interpolate,
    image_gabor,
    image_make_montage,
    image_montage,
    image_slanted_edge,
    image_spd,
    image_spd2rgb,
    image_translate,
    image_transpose,
    imagesc_m,
    imagesc_opp,
    imagesc_rgb,
    interp_spectra,
    image_bounding_box,
    image_centroid,
    image_circular,
    image_contrast,
    param_format,
    quanta_to_energy,
    rgb_to_dac,
    rgb_to_xw_format,
    unit_frequency_list,
)


def test_param_format_string_and_key_value_list() -> None:
    assert param_format("Exposure Time") == "exposuretime"
    assert param_format(["Exposure Time", 1, "Some Flag", True]) == ["exposuretime", 1, "someflag", True]


def test_energy_quanta_round_trip() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array([0.2, 0.5, 0.8])
    quanta = energy_to_quanta(energy, wave)
    restored = quanta_to_energy(quanta, wave)
    assert np.allclose(restored, energy)


def test_energy_quanta_round_trip_supports_wave_first_matrices() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array(
        [
            [0.2, 0.4],
            [0.5, 0.7],
            [0.8, 1.0],
        ],
        dtype=float,
    )
    quanta = energy_to_quanta(energy, wave)
    restored = quanta_to_energy(quanta, wave)
    assert np.allclose(restored, energy)


def test_interp_spectra_supports_descending_source_waves() -> None:
    source_wave = np.array([700.0, 600.0, 500.0, 400.0], dtype=float)
    values = np.array(
        [
            [0.7, 0.2],
            [0.6, 0.3],
            [0.5, 0.4],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    target_wave = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)
    interpolated = interp_spectra(source_wave, values, target_wave)
    expected = values[::-1, :]
    assert np.allclose(interpolated, expected)


def test_blackbody_matlab_scaling() -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    spectra = blackbody(wave, np.array([3000.0, 5000.0]))
    assert spectra.shape == (wave.size, 2)
    assert np.all(spectra > 0.0)
    eq_index = int(np.argmin(np.abs(wave - 550.0)))
    assert np.isclose(spectra[eq_index, 0], spectra[eq_index, 1])


def test_half_tone_image_scales_integer_cells_and_alias_matches() -> None:
    cell = np.array([[1, 2], [3, 4]], dtype=float)
    image = np.array(
        [
            [0.10, 0.50, 0.90],
            [0.20, 0.40, 0.60],
            [0.80, 1.00, 0.10],
        ],
        dtype=float,
    )
    scaled_cell = np.array([[0.125, 0.375], [0.625, 0.875]], dtype=float)
    tiled = np.tile(scaled_cell, (2, 2))[: image.shape[0], : image.shape[1]]
    expected = tiled < image

    result = half_tone_image(cell, image)
    alias = HalfToneImage(cell, image)

    assert result.dtype == np.bool_
    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)


def test_floyd_steinberg_matches_legacy_error_diffusion_and_alias() -> None:
    fs = np.array([[0.0, 0.0, 7.0], [3.0, 5.0, 1.0]], dtype=float) / 16.0
    image = np.array(
        [
            [0.20, 0.40, 0.60, 0.80],
            [0.30, 0.70, 0.50, 0.10],
            [0.90, 0.20, 0.80, 0.40],
        ],
        dtype=float,
    )

    def _reference(fs_kernel: np.ndarray, image_data: np.ndarray) -> np.ndarray:
        rows, cols = image_data.shape
        fs_rows, fs_cols = fs_kernel.shape
        radius = fs_cols // 2
        temp = np.zeros((rows + fs_rows, cols + 2 * radius), dtype=float)
        temp[:rows, radius : radius + cols] = image_data

        def _round_scalar(value: float) -> float:
            return float(np.sign(value) * np.floor(abs(value) + 0.5))

        for row in range(rows):
            for col in range(radius, cols):
                error = float(temp[row, col])
                temp[row, col] = _round_scalar(error)
                error -= float(temp[row, col])
                for dr in range(fs_rows):
                    for dc in range(fs_cols):
                        temp[row + dr, col - radius + dc] += error * float(fs_kernel[dr, dc])

            temp[row : row + fs_rows, cols : cols + radius] += temp[row + 1 : row + fs_rows + 1, :radius]

            for col in range(cols, cols + radius):
                error = float(temp[row, col])
                temp[row, col] = _round_scalar(error)
                error -= float(temp[row, col])
                for dr in range(fs_rows):
                    for dc in range(fs_cols):
                        temp[row + dr, col - radius + dc] += error * float(fs_kernel[dr, dc])

            temp[row + 1 : row + fs_rows + 1, radius : 2 * radius] += temp[
                row : row + fs_rows, cols + radius : cols + 2 * radius
            ]
            temp[:, :radius] = 0.0
            temp[:, cols + radius : cols + 2 * radius] = 0.0

        return temp[:rows, radius : radius + cols]

    expected = _reference(fs, image)
    result = floyd_steinberg(fs, image)
    alias = FloydSteinberg(fs, image)

    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_unit_frequency_list_matches_matlab_even_and_odd() -> None:
    assert np.allclose(unit_frequency_list(4), np.array([-1.0, -0.5, 0.0, 0.5]))
    assert np.allclose(unit_frequency_list(5), np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_ie_fit_line_matches_matlab_one_line_and_multiple_lines() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    y = 2.0 * x + 0.5
    slope, offset = ie_fit_line(x, y)
    assert slope == pytest.approx(2.0)
    assert offset == pytest.approx(0.5)

    x_multi = x.reshape(-1, 1)
    y_multi = np.column_stack((2.0 * x + 0.5, -1.0 * x + 3.0))
    slopes, offsets = ie_fit_line(x_multi, y_multi, "multipleLines")
    assert np.allclose(slopes, np.array([2.0, -1.0], dtype=float))
    assert np.allclose(offsets, np.array([0.5, 3.0], dtype=float))


def test_ie_find_wave_index_exact_and_nearest_match_alias() -> None:
    wave = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)

    exact = ie_find_wave_index(wave, [500.0, 700.0])
    nearest = ieFindWaveIndex(wave, [520.0, 610.0], perfect=False)

    assert np.array_equal(exact, np.array([False, True, False, True]))
    assert np.array_equal(nearest, np.array([False, True, True, False]))


def test_ie_wave2_index_matches_matlab_one_based_and_bounding_pair_alias() -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)

    assert ie_wave2_index(wave, 503.0) == 2
    assert ieWave2Index(wave, 487.0, bounding=True) == (1, 2)
    assert ieWave2Index(wave, 600.0, bounding=True) == (3, 3)


def test_ie_radial_matrix_matches_centered_distance_grid_alias() -> None:
    expected = np.array(
        [
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            [1.0, 0.0, 1.0],
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
        ],
        dtype=float,
    )

    result = ie_radial_matrix(3, 3, 2, 2)
    alias = ieRadialMatrix(3, 3, 2, 2)

    assert np.allclose(result, expected)
    assert np.allclose(alias, expected)


def test_image_bounding_box_and_centroid_match_matlab_pixel_geometry_aliases() -> None:
    support = np.zeros((5, 5), dtype=float)
    support[1:4, 1:4] = 1.0
    weighted = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    box = image_bounding_box(support)
    alias_box = imageBoundingBox(support)
    centroid = image_centroid(weighted)
    alias_centroid = imageCentroid(weighted)

    assert np.allclose(box, np.array([2.0, 2.0, 2.0, 2.0], dtype=float))
    assert np.allclose(alias_box, box)
    assert centroid == (3, 2)
    assert alias_centroid == centroid


def test_image_circular_and_contrast_match_legacy_contract_aliases() -> None:
    image = np.ones((3, 3), dtype=float)
    contrast_input = np.array(
        [
            [[1.0, 2.0], [3.0, 6.0]],
            [[5.0, 10.0], [7.0, 14.0]],
        ],
        dtype=float,
    )
    expected_circular = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    expected_contrast = np.array(
        [
            [[-0.75, -0.75], [-0.25, -0.25]],
            [[0.25, 0.25], [0.75, 0.75]],
        ],
        dtype=float,
    )

    circular = image_circular(image)
    alias_circular = imageCircular(image)
    contrast = image_contrast(contrast_input)
    alias_contrast = imageContrast(contrast_input)

    assert np.array_equal(circular, expected_circular)
    assert np.array_equal(alias_circular, expected_circular)
    assert np.allclose(contrast, expected_contrast)
    assert np.allclose(alias_contrast, expected_contrast)


def test_ie_cmap_matches_rg_bw_and_alias() -> None:
    rg = ie_cmap("rg", 4)
    bw = ieCmap("bw", 3, 2.0)

    assert np.allclose(
        rg,
        np.array(
            [
                [0.0, 1.0, 0.5],
                [1.0 / 3.0, 2.0 / 3.0, 0.5],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
                [1.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
    )
    assert np.allclose(
        bw,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_ie_crop_rect_matches_matlab_fov_geometry_and_alias() -> None:
    oi = {"optics": {"raytrace": {"objectDistance": 2.0}}}
    scenesize = np.array([100, 200], dtype=float)

    crop_rect = ie_crop_rect(oi, scenesize, 20.0, 10.0)
    alias = ieCropRect(oi, scenesize, 20.0, 10.0)

    assert np.array_equal(crop_rect, np.array([52, 26, 98, 49]))
    assert np.array_equal(alias, crop_rect)


def test_ie_lut_digital_invert_linear_match_legacy_tables_and_aliases() -> None:
    dac = np.array([[0, 1], [2, 3]], dtype=float)
    gamma = np.array([0.0, 0.25, 0.5, 1.0], dtype=float)
    rgb = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=float)
    inverse_gamma = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)

    digital = ie_lut_digital(dac, gamma)
    digital_alias = ieLUTDigital(dac, gamma)
    invert = ie_lut_invert(gamma, 4)
    invert_alias = ieLUTInvert(gamma, 4)
    linear = ie_lut_linear(rgb, inverse_gamma)
    linear_alias = ieLUTLinear(rgb, inverse_gamma)

    assert np.allclose(digital, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=float))
    assert np.allclose(digital_alias, digital)
    assert np.allclose(invert, np.array([[1.0], [2.0], [3.0], [3.5]], dtype=float))
    assert np.allclose(invert_alias, invert)
    assert np.allclose(linear, np.array([[0.0, 10.0], [20.0, 30.0]], dtype=float))
    assert np.allclose(linear_alias, linear)


def test_rgb2dac_matches_single_and_three_channel_tables_and_alias() -> None:
    rgb = np.array(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]],
        ],
        dtype=float,
    )
    inv_gamma = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
        dtype=float,
    )

    result = rgb_to_dac(rgb, inv_gamma)
    alias = rgb2dac(rgb, inv_gamma)

    expected = np.array(
        [
            [[0.0, 50.0, 90.0], [10.0, 50.0, 60.0]],
        ],
        dtype=float,
    )
    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)


def test_image_transpose_translate_and_interpolate_match_headless_numeric_contracts() -> None:
    cube = np.arange(1.0, 13.0, dtype=float).reshape(2, 3, 2)
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    transposed = image_transpose(cube)
    transposed_alias = imageTranspose(cube)
    translated = image_translate(image, [1, 1])
    translated_alias = imageTranslate(image, [1, 1])
    interpolated = image_interpolate(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), 4, 4)
    interpolated_alias = imageInterpolate(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), 4, 4)

    assert np.array_equal(transposed[:, :, 0], np.array([[1.0, 7.0], [3.0, 9.0], [5.0, 11.0]], dtype=float))
    assert np.array_equal(transposed_alias, transposed)
    assert np.array_equal(
        translated,
        np.array(
            [
                [5.0, 6.0, 0.0],
                [8.0, 9.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(translated_alias, translated)
    assert interpolated.shape == (4, 4)
    assert interpolated[0, 0] == pytest.approx(1.0)
    assert interpolated[-1, -1] == pytest.approx(4.0)
    assert np.allclose(interpolated_alias, interpolated)


def test_image_hparams_matches_legacy_defaults_and_alias() -> None:
    params = image_hparams()
    alias = imageHparams()

    expected = {
        "freq": 2,
        "contrast": 1,
        "ang": 0,
        "ph": 1.5708,
        "row": 128,
        "col": 128,
        "GaborFlag": 0,
    }
    assert params == expected
    assert alias == expected


def test_image_gabor_matches_legacy_range_shape_and_alias() -> None:
    gabor = image_gabor(frequency=4, phase=0.0, spread=0.25, orientation=np.pi / 4.0, imagesize=6, contrast=1.0)
    alias = imageGabor(frequency=4, phase=0.0, spread=0.25, orientation=np.pi / 4.0, imagesize=6, contrast=1.0)

    assert gabor.shape == (7, 7)
    assert float(np.min(gabor)) >= 0.0
    assert float(np.max(gabor)) <= 1.0
    assert gabor[3, 3] == pytest.approx(1.0)
    assert np.allclose(alias, gabor)


def test_image_make_montage_and_image_montage_match_legacy_tile_layout_aliases() -> None:
    cube = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        ),
        axis=2,
    )

    montage, coords = image_make_montage(cube, [1, 3], n_cols=2, back_val=-1.0)
    fig_h, montage_alias, cb_h = imageMontage(cube, [1, 3], 2)
    headless = image_montage(cube, [1, 3], 2)

    expected = np.array(
        [
            [1.0, 2.0, 9.0, 10.0],
            [3.0, 4.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    assert np.array_equal(montage, expected)
    assert np.array_equal(coords, np.array([[1, 1], [3, 1]], dtype=int))
    assert fig_h is None and cb_h is None
    assert np.array_equal(montage_alias, expected)
    assert headless[0] is None and headless[2] is None
    assert np.array_equal(headless[1], expected)


def test_ie_parameter_otype_handles_direct_prefix_and_unique_params() -> None:
    assert ie_parameter_otype("scene") == ("scene", None)
    assert ie_parameter_otype("oi size") == ("oi", "size")
    assert ie_parameter_otype("pixel/size") == ("pixel", "size")
    assert ie_parameter_otype("pixel width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("pixel pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("widthandheight") == ("pixel", "widthandheight")
    assert ie_parameter_otype("widthheight") == ("pixel", "widthheight")
    assert ie_parameter_otype("pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("pdwidthandheight") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("size same fill factor") == ("pixel", "sizesamefillfactor")
    assert ie_parameter_otype("size constant fill factor") == ("pixel", "sizeconstantfillfactor")
    assert ie_parameter_otype("size keep fill factor") == ("pixel", "sizekeepfillfactor")
    assert ie_parameter_otype("sizesamefillfactor") == ("pixel", "sizesamefillfactor")
    assert ie_parameter_otype("sizeconstantfillfactor") == ("pixel", "sizeconstantfillfactor")
    assert ie_parameter_otype("sizekeepfillfactor") == ("pixel", "sizekeepfillfactor")
    assert ie_parameter_otype("dark voltage per pixel per sec") == ("pixel", "darkvoltageperpixelpersec")
    assert ie_parameter_otype("wvf_zcoeffs") == ("wvf", "zcoeffs")
    assert ie_parameter_otype("display gamma") == ("display", "gamma")
    assert ie_parameter_otype("match oi") == ("sensor", "matchoi")
    assert ie_parameter_otype("dsnu sigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("dsnusigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("dsnulevel") == ("sensor", "dsnulevel")
    assert ie_parameter_otype("sigma offset fpn") == ("sensor", "sigmaoffsetfpn")
    assert ie_parameter_otype("sigmaoffsetfpn") == ("sensor", "sigmaoffsetfpn")
    assert ie_parameter_otype("offset fpn") == ("sensor", "offsetfpn")
    assert ie_parameter_otype("offsetfpn") == ("sensor", "offsetfpn")
    assert ie_parameter_otype("offset") == ("sensor", "offset")
    assert ie_parameter_otype("offsetsd") == ("sensor", "offsetsd")
    assert ie_parameter_otype("offset sd") == ("sensor", "offsetsd")
    assert ie_parameter_otype("offsetnoisevalue") == ("sensor", "offsetnoisevalue")
    assert ie_parameter_otype("offset noise value") == ("sensor", "offsetnoisevalue")
    assert ie_parameter_otype("sigmadsnu") == ("sensor", "sigmadsnu")
    assert ie_parameter_otype("sigma dsnu") == ("sensor", "sigmadsnu")
    assert ie_parameter_otype("sigmagainfpn") == ("sensor", "sigmagainfpn")
    assert ie_parameter_otype("sigma gain fpn") == ("sensor", "sigmagainfpn")
    assert ie_parameter_otype("gainfpn") == ("sensor", "gainfpn")
    assert ie_parameter_otype("gain fpn") == ("sensor", "gainfpn")
    assert ie_parameter_otype("gain") == ("sensor", "gain")
    assert ie_parameter_otype("gainsd") == ("sensor", "gainsd")
    assert ie_parameter_otype("gain sd") == ("sensor", "gainsd")
    assert ie_parameter_otype("gainnoisevalue") == ("sensor", "gainnoisevalue")
    assert ie_parameter_otype("gain noise value") == ("sensor", "gainnoisevalue")
    assert ie_parameter_otype("prnusigma") == ("sensor", "prnusigma")
    assert ie_parameter_otype("prnulevel") == ("sensor", "prnulevel")
    assert ie_parameter_otype("sigmaprnu") == ("sensor", "sigmaprnu")
    assert ie_parameter_otype("sigma prnu") == ("sensor", "sigmaprnu")
    assert ie_parameter_otype("fpn") == ("sensor", "fpn")
    assert ie_parameter_otype("fpnparameters") == ("sensor", "fpnparameters")
    assert ie_parameter_otype("fpnoffsetgain") == ("sensor", "fpnoffsetgain")
    assert ie_parameter_otype("fpnoffsetandgain") == ("sensor", "fpnoffsetandgain")
    assert ie_parameter_otype("fpn offset gain") == ("sensor", "fpnoffsetgain")
    assert ie_parameter_otype("dsnuimage") == ("sensor", "dsnuimage")
    assert ie_parameter_otype("offset fpn image") == ("sensor", "offsetfpnimage")
    assert ie_parameter_otype("offsetfpnimage") == ("sensor", "offsetfpnimage")
    assert ie_parameter_otype("prnuimage") == ("sensor", "prnuimage")
    assert ie_parameter_otype("gain fpn image") == ("sensor", "gainfpnimage")
    assert ie_parameter_otype("gainfpnimage") == ("sensor", "gainfpnimage")
    assert ie_parameter_otype("column fpn parameters") == ("sensor", "columnfpnparameters")
    assert ie_parameter_otype("columnfpnparameters") == ("sensor", "columnfpnparameters")
    assert ie_parameter_otype("columnfpn") == ("sensor", "columnfpn")
    assert ie_parameter_otype("column fixed pattern noise") == ("sensor", "columnfixedpatternnoise")
    assert ie_parameter_otype("columnfixedpatternnoise") == ("sensor", "columnfixedpatternnoise")
    assert ie_parameter_otype("col fpn") == ("sensor", "colfpn")
    assert ie_parameter_otype("colfpn") == ("sensor", "colfpn")
    assert ie_parameter_otype("column fpn offset") == ("sensor", "columnfpnoffset")
    assert ie_parameter_otype("columnfpnoffset") == ("sensor", "columnfpnoffset")
    assert ie_parameter_otype("col fpn offset") == ("sensor", "colfpnoffset")
    assert ie_parameter_otype("colfpnoffset") == ("sensor", "colfpnoffset")
    assert ie_parameter_otype("columndsnu") == ("sensor", "columndsnu")
    assert ie_parameter_otype("col dsnu") == ("sensor", "coldsnu")
    assert ie_parameter_otype("coldsnu") == ("sensor", "coldsnu")
    assert ie_parameter_otype("column fpn gain") == ("sensor", "columnfpngain")
    assert ie_parameter_otype("columnfpngain") == ("sensor", "columnfpngain")
    assert ie_parameter_otype("col fpn gain") == ("sensor", "colfpngain")
    assert ie_parameter_otype("colfpngain") == ("sensor", "colfpngain")
    assert ie_parameter_otype("columnprnu") == ("sensor", "columnprnu")
    assert ie_parameter_otype("col prnu") == ("sensor", "colprnu")
    assert ie_parameter_otype("colprnu") == ("sensor", "colprnu")
    assert ie_parameter_otype("coloffsetfpnvector") == ("sensor", "coloffsetfpnvector")
    assert ie_parameter_otype("col offset fpn") == ("sensor", "coloffsetfpn")
    assert ie_parameter_otype("col offset") == ("sensor", "coloffset")
    assert ie_parameter_otype("colgainfpnvector") == ("sensor", "colgainfpnvector")
    assert ie_parameter_otype("col gain fpn") == ("sensor", "colgainfpn")
    assert ie_parameter_otype("col gain") == ("sensor", "colgain")
    assert ie_parameter_otype("responsedr") == ("sensor", "responsedr")
    assert ie_parameter_otype("drdb20") == ("sensor", "drdb20")
    assert ie_parameter_otype("analoggain") == ("sensor", "analoggain")
    assert ie_parameter_otype("analogoffset") == ("sensor", "analogoffset")
    assert ie_parameter_otype("dynamicrange") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("diffusionmtf") == ("sensor", "diffusionmtf")
    assert ie_parameter_otype("sensordynamicrange") == ("sensor", "sensordynamicrange")
    assert ie_parameter_otype("maxdigitalvalue") == ("sensor", "maxdigitalvalue")
    assert ie_parameter_otype("noiseflag") == ("sensor", "noiseflag")
    assert ie_parameter_otype("integrationtime") == ("sensor", "integrationtime")
    assert ie_parameter_otype("integrationtimes") == ("sensor", "integrationtimes")
    assert ie_parameter_otype("nsamplesperpixel") == ("sensor", "nsamplesperpixel")
    assert ie_parameter_otype("sensorconsistency") == ("sensor", "sensorconsistency")
    assert ie_parameter_otype("sensorcompute") == ("sensor", "sensorcompute")
    assert ie_parameter_otype("sensorcomputemethod") == ("sensor", "sensorcomputemethod")
    assert ie_parameter_otype("nfilters") == ("sensor", "nfilters")
    assert ie_parameter_otype("filternames") == ("sensor", "filternames")
    assert ie_parameter_otype("filtercolorletters") == ("sensor", "filtercolorletters")
    assert ie_parameter_otype("filtercolorletterscell") == ("sensor", "filtercolorletterscell")
    assert ie_parameter_otype("filterplotcolors") == ("sensor", "filterplotcolors")
    assert ie_parameter_otype("filterspectra") == ("sensor", "filterspectra")
    assert ie_parameter_otype("colorfilters") == ("sensor", "colorfilters")
    assert ie_parameter_otype("infraredfilter") == ("sensor", "infraredfilter")
    assert ie_parameter_otype("irfilter") == ("sensor", "irfilter")
    assert ie_parameter_otype("unitblockrows") == ("sensor", "unitblockrows")
    assert ie_parameter_otype("unitblockcols") == ("sensor", "unitblockcols")
    assert ie_parameter_otype("cfasize") == ("sensor", "cfasize")
    assert ie_parameter_otype("spectralqe") == ("sensor", "spectralqe")
    assert ie_parameter_otype("sensorspectralsr") == ("sensor", "sensorspectralsr")
    assert ie_parameter_otype("patterncolors") == ("sensor", "patterncolors")
    assert ie_parameter_otype("pixelspectralqe") == ("pixel", "pixelspectralqe")
    assert ie_parameter_otype("pdspectralqe") == ("pixel", "pdspectralqe")
    assert ie_parameter_otype("pixelqe") == ("pixel", "pixelqe")
    assert ie_parameter_otype("spectralsr") == ("pixel", "spectralsr")
    assert ie_parameter_otype("pdspectralsr") == ("pixel", "pdspectralsr")
    assert ie_parameter_otype("pixelspectralsr") == ("pixel", "pixelspectralsr")
    assert ie_parameter_otype("sr") == ("pixel", "sr")
    assert ie_parameter_otype("dynamic range") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("shotnoiseflag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("shot noise flag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("blacklevel") == ("sensor", "blacklevel")
    assert ie_parameter_otype("black level") == ("sensor", "blacklevel")
    assert ie_parameter_otype("zerolevel") == ("sensor", "zerolevel")
    assert ie_parameter_otype("zero level") == ("sensor", "zerolevel")
    assert ie_parameter_otype("bits") == ("sensor", "bits")
    assert ie_parameter_otype("nbits") == ("sensor", "nbits")
    assert ie_parameter_otype("quantization") == ("sensor", "quantization")
    assert ie_parameter_otype("quantizationmethod") == ("sensor", "quantizationmethod")
    assert ie_parameter_otype("maxdigital") == ("sensor", "maxdigital")
    assert ie_parameter_otype("maxoutput") == ("sensor", "maxoutput")
    assert ie_parameter_otype("roi") == ("sensor", "roi")
    assert ie_parameter_otype("roilocs") == ("sensor", "roilocs")
    assert ie_parameter_otype("roirect") == ("sensor", "roirect")
    assert ie_parameter_otype("voltage") == ("sensor", "voltage")
    assert ie_parameter_otype("scene_name") == ("sensor", "scene_name")
    assert ie_parameter_otype("scenename") == ("sensor", "scenename")
    assert ie_parameter_otype("scene name") == ("sensor", "scenename")
    assert ie_parameter_otype("metadata scene name") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("metadatascenename") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("lens") == ("sensor", "lens")
    assert ie_parameter_otype("metadata lens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata lensname") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadataopticsname") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadata optics name") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadatasensorname") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadata sensor name") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadatacrop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("metadata crop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("sensoretendue") == ("sensor", "sensoretendue")
    assert ie_parameter_otype("microlens") == ("sensor", "microlens")
    assert ie_parameter_otype("ml") == ("sensor", "ml")
    assert ie_parameter_otype("mlens") == ("sensor", "mlens")
    assert ie_parameter_otype("ulens") == ("sensor", "ulens")
    assert ie_parameter_otype("microlensoffset") == ("sensor", "microlensoffset")
    assert ie_parameter_otype("microlens offset") == ("sensor", "microlensoffset")
    assert ie_parameter_otype("mloffset") == ("sensor", "mloffset")
    assert ie_parameter_otype("microlensoffsetmicrons") == ("sensor", "microlensoffsetmicrons")
    assert ie_parameter_otype("microlens offset microns") == ("sensor", "microlensoffsetmicrons")
    assert ie_parameter_otype("vignetting") == ("sensor", "vignetting")
    assert ie_parameter_otype("vignettingflag") == ("sensor", "vignettingflag")
    assert ie_parameter_otype("vignetting flag") == ("sensor", "vignettingflag")
    assert ie_parameter_otype("pixelvignetting") == ("sensor", "pixelvignetting")
    assert ie_parameter_otype("pixel vignetting") == ("sensor", "pixelvignetting")
    assert ie_parameter_otype("sensorvignetting") == ("sensor", "sensorvignetting")
    assert ie_parameter_otype("sensor vignetting") == ("sensor", "vignetting")
    assert ie_parameter_otype("bare etendue") == ("sensor", "bareetendue")
    assert ie_parameter_otype("sensorbareetendue") == ("sensor", "sensorbareetendue")
    assert ie_parameter_otype("sensor bare etendue") == ("sensor", "bareetendue")
    assert ie_parameter_otype("nomicrolensetendue") == ("sensor", "nomicrolensetendue")
    assert ie_parameter_otype("no microlens etendue") == ("sensor", "nomicrolensetendue")
    assert ie_parameter_otype("vignettingname") == ("sensor", "vignettingname")
    assert ie_parameter_otype("vignetting name") == ("sensor", "vignettingname")
    assert ie_parameter_otype("ngridsamples") == ("sensor", "ngridsamples")
    assert ie_parameter_otype("color") == ("sensor", "color")
    assert ie_parameter_otype("digitalvalue") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("digital value") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("digitalvalues") == ("sensor", "digitalvalues")
    assert ie_parameter_otype("digital values") == ("sensor", "digitalvalues")
    assert ie_parameter_otype("electron") == ("sensor", "electron")
    assert ie_parameter_otype("electrons per area") == ("sensor", "electronsperarea")
    assert ie_parameter_otype("filternamescellarray") == ("sensor", "filternamescellarray")
    assert ie_parameter_otype("filter names cell array") == ("sensor", "filternamescellarray")
    assert ie_parameter_otype("filtercolornamescellarray") == ("sensor", "filtercolornamescellarray")
    assert ie_parameter_otype("filter color names cell array") == ("sensor", "filtercolornamescellarray")
    assert ie_parameter_otype("filternamescell") == ("sensor", "filternamescell")
    assert ie_parameter_otype("filter names cell") == ("sensor", "filternamescell")
    assert ie_parameter_otype("lut") == ("sensor", "lut")
    assert ie_parameter_otype("quantizatonlut") == ("sensor", "quantizatonlut")
    assert ie_parameter_otype("quantizationlut") == ("sensor", "quantizationlut")
    assert ie_parameter_otype("quantization lut") == ("sensor", "quantizationlut")
    assert ie_parameter_otype("qmethod") == ("sensor", "qmethod")
    assert ie_parameter_otype("qMethod") == ("sensor", "qmethod")
    assert ie_parameter_otype("quantizationstructure") == ("sensor", "quantizationstructure")
    assert ie_parameter_otype("quantization structure") == ("sensor", "quantizationstructure")
    assert ie_parameter_otype("cfapattern") == ("sensor", "cfapattern")
    assert ie_parameter_otype("pattern") == ("sensor", "pattern")
    assert ie_parameter_otype("cfaname") == ("sensor", "cfaname")
    assert ie_parameter_otype("unitblockconfig") == ("sensor", "unitblockconfig")
    assert ie_parameter_otype("sensorspectrum") == ("sensor", "sensorspectrum")
    assert ie_parameter_otype("wavelengthresolution") == ("sensor", "wavelengthresolution")
    assert ie_parameter_otype("binwidth") == ("sensor", "binwidth")
    assert ie_parameter_otype("numberofwavelengthsamples") == ("sensor", "numberofwavelengthsamples")
    assert ie_parameter_otype("dvorvolts") == ("sensor", "dvorvolts")
    assert ie_parameter_otype("dv or volts") == ("sensor", "dvorvolts")
    assert ie_parameter_otype("digitalorvolts") == ("sensor", "digitalorvolts")
    assert ie_parameter_otype("digital or volts") == ("sensor", "digitalorvolts")
    assert ie_parameter_otype("voltimages") == ("sensor", "voltimages")
    assert ie_parameter_otype("volt images") == ("sensor", "voltimages")
    assert ie_parameter_otype("sensor rows") == ("sensor", "rows")
    assert ie_parameter_otype("sensor cols") == ("sensor", "cols")
    assert ie_parameter_otype("sensor size") == ("sensor", "size")
    assert ie_parameter_otype("sensor dimension") == ("sensor", "dimension")
    assert ie_parameter_otype("sensor arraywidth") == ("sensor", "arraywidth")
    assert ie_parameter_otype("sensor arrayheight") == ("sensor", "arrayheight")
    assert ie_parameter_otype("sensor wspatialresolution") == ("sensor", "wspatialresolution")
    assert ie_parameter_otype("sensor hspatialresolution") == ("sensor", "hspatialresolution")
    assert ie_parameter_otype("sensor deltax") == ("sensor", "deltax")
    assert ie_parameter_otype("sensor deltay") == ("sensor", "deltay")
    assert ie_parameter_otype("sensor volts") == ("sensor", "volts")
    assert ie_parameter_otype("sensor dv") == ("sensor", "dv")
    assert ie_parameter_otype("sensor electrons") == ("sensor", "electrons")
    assert ie_parameter_otype("sensor voltage") == ("sensor", "voltage")
    assert ie_parameter_otype("sensor digitalvalue") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("roivolts") == ("sensor", "roivolts")
    assert ie_parameter_otype("roidata") == ("sensor", "roidata")
    assert ie_parameter_otype("roidatav") == ("sensor", "roidatav")
    assert ie_parameter_otype("roidatavolts") == ("sensor", "roidatavolts")
    assert ie_parameter_otype("roielectrons") == ("sensor", "roielectrons")
    assert ie_parameter_otype("roidatae") == ("sensor", "roidatae")
    assert ie_parameter_otype("roidataelectrons") == ("sensor", "roidataelectrons")
    assert ie_parameter_otype("roidv") == ("sensor", "roidv")
    assert ie_parameter_otype("roidigitalcount") == ("sensor", "roidigitalcount")
    assert ie_parameter_otype("roivoltsmean") == ("sensor", "roivoltsmean")
    assert ie_parameter_otype("roielectronsmean") == ("sensor", "roielectronsmean")
    assert ie_parameter_otype("responseratio") == ("sensor", "responseratio")
    assert ie_parameter_otype("volts2maxratio") == ("sensor", "volts2maxratio")
    assert ie_parameter_otype("sensor chromaticity") == ("sensor", "chromaticity")
    assert ie_parameter_otype("sensor roichromaticitymean") == ("sensor", "roichromaticitymean")
    assert ie_parameter_otype("sensor hlinevolts") == ("sensor", "hlinevolts")
    assert ie_parameter_otype("sensor hlineelectrons") == ("sensor", "hlineelectrons")
    assert ie_parameter_otype("sensor hlinedv") == ("sensor", "hlinedv")
    assert ie_parameter_otype("sensor vlinevolts") == ("sensor", "vlinevolts")
    assert ie_parameter_otype("sensor vlineelectrons") == ("sensor", "vlineelectrons")
    assert ie_parameter_otype("sensor vlinedv") == ("sensor", "vlinedv")
    assert ie_parameter_otype("fill factor") == ("pixel", "fillfactor")
    assert ie_parameter_otype("pixelsize") == ("pixel", "pixelsize")
    assert ie_parameter_otype("pixelwidth") == ("pixel", "pixelwidth")
    assert ie_parameter_otype("pixelheight") == ("pixel", "pixelheight")
    assert ie_parameter_otype("pixelwidthmeters") == ("pixel", "pixelwidthmeters")
    assert ie_parameter_otype("pixelheightmeters") == ("pixel", "pixelheightmeters")
    assert ie_parameter_otype("conversion gain") == ("pixel", "conversiongain")
    assert ie_parameter_otype("conversiongainvpelectron") == ("pixel", "conversiongainvpelectron")
    assert ie_parameter_otype("conversion gain v per electron") == ("pixel", "conversiongainvperelectron")
    assert ie_parameter_otype("vswing") == ("pixel", "vswing")
    assert ie_parameter_otype("wellcapacity") == ("pixel", "wellcapacity")
    assert ie_parameter_otype("well capacity") == ("pixel", "wellcapacity")
    assert ie_parameter_otype("voltsperelectron") == ("pixel", "voltsperelectron")
    assert ie_parameter_otype("volts per electron") == ("pixel", "voltsperelectron")
    assert ie_parameter_otype("saturation voltage") == ("pixel", "saturationvoltage")
    assert ie_parameter_otype("maxvoltage") == ("pixel", "maxvoltage")
    assert ie_parameter_otype("max voltage") == ("pixel", "maxvoltage")
    assert ie_parameter_otype("darkcurrentdensity") == ("pixel", "darkcurrentdensity")
    assert ie_parameter_otype("dark current density") == ("pixel", "darkcurrentdensity")
    assert ie_parameter_otype("dark current per pixel") == ("pixel", "darkcurrentperpixel")
    assert ie_parameter_otype("darkvolt") == ("pixel", "darkvolt")
    assert ie_parameter_otype("darkvolts") == ("pixel", "darkvolts")
    assert ie_parameter_otype("darkvoltageperpixelpersec") == ("pixel", "darkvoltageperpixelpersec")
    assert ie_parameter_otype("darkvoltageperpixel") == ("pixel", "darkvoltageperpixel")
    assert ie_parameter_otype("dark voltage per pixel") == ("pixel", "darkvoltageperpixel")
    assert ie_parameter_otype("darkelectrons") == ("pixel", "darkelectrons")
    assert ie_parameter_otype("voltspersecond") == ("pixel", "voltspersecond")
    assert ie_parameter_otype("volts per second") == ("pixel", "voltspersecond")
    assert ie_parameter_otype("read noise") == ("pixel", "readnoise")
    assert ie_parameter_otype("readnoiseelectrons") == ("pixel", "readnoiseelectrons")
    assert ie_parameter_otype("readstandarddeviationelectrons") == ("pixel", "readstandarddeviationelectrons")
    assert ie_parameter_otype("readnoisevolts") == ("pixel", "readnoisevolts")
    assert ie_parameter_otype("read noise volts") == ("pixel", "readnoisevolts")
    assert ie_parameter_otype("readstandarddeviationvolts") == ("pixel", "readstandarddeviationvolts")
    assert ie_parameter_otype("read standard deviation volts") == ("pixel", "readstandarddeviationvolts")
    assert ie_parameter_otype("read standard deviation electrons") == ("pixel", "readstandarddeviationelectrons")
    assert ie_parameter_otype("readnoisestdvolts") == ("pixel", "readnoisestdvolts")
    assert ie_parameter_otype("read noise std volts") == ("pixel", "readnoisestdvolts")
    assert ie_parameter_otype("readnoisemillivolts") == ("pixel", "readnoisemillivolts")
    assert ie_parameter_otype("read noise millivolts") == ("pixel", "readnoisemillivolts")
    assert ie_parameter_otype("widthgap") == ("pixel", "widthgap")
    assert ie_parameter_otype("width gap") == ("pixel", "widthgap")
    assert ie_parameter_otype("widthbetweenpixels") == ("pixel", "widthbetweenpixels")
    assert ie_parameter_otype("width between pixels") == ("pixel", "widthbetweenpixels")
    assert ie_parameter_otype("heightgap") == ("pixel", "heightgap")
    assert ie_parameter_otype("heightbetweenpixels") == ("pixel", "heightbetweenpixels")
    assert ie_parameter_otype("height between pixels") == ("pixel", "heightbetweenpixels")
    assert ie_parameter_otype("xyspacing") == ("pixel", "xyspacing")
    assert ie_parameter_otype("pixel width meters") == ("pixel", "widthmeters")
    assert ie_parameter_otype("pixel height meters") == ("pixel", "heightmeters")
    assert ie_parameter_otype("pdarea") == ("pixel", "pdarea")
    assert ie_parameter_otype("pd width") == ("pixel", "pdwidth")
    assert ie_parameter_otype("pd dimension") == ("pixel", "pddimension")
    assert ie_parameter_otype("pddimension") == ("pixel", "pddimension")
    assert ie_parameter_otype("pixeldepth") == ("pixel", "pixeldepth")
    assert ie_parameter_otype("layer thicknesses") == ("pixel", "layerthicknesses")
    assert ie_parameter_otype("layerthicknesses") == ("pixel", "layerthicknesses")
    assert ie_parameter_otype("refractive index") == ("pixel", "refractiveindex")
    assert ie_parameter_otype("refractive indices") == ("pixel", "refractiveindices")
    assert ie_parameter_otype("refractiveindices") == ("pixel", "refractiveindices")
    assert ie_parameter_otype("n") == ("pixel", "n")
    assert ie_parameter_otype("stack height") == ("pixel", "stackheight")
    assert ie_parameter_otype("stackheight") == ("pixel", "stackheight")
    assert ie_parameter_otype("pixel depth meters") == ("pixel", "depthmeters")
    assert ie_parameter_otype("pixeldepthmeters") == ("pixel", "pixeldepthmeters")
    assert ie_parameter_otype("pd xpos") == ("pixel", "pdxpos")
    assert ie_parameter_otype("pdxpos") == ("pixel", "pdxpos")
    assert ie_parameter_otype("pdypos") == ("pixel", "pdypos")
    assert ie_parameter_otype("photodetector x position") == ("pixel", "photodetectorxposition")
    assert ie_parameter_otype("photodetectorxposition") == ("pixel", "photodetectorxposition")
    assert ie_parameter_otype("photodetectoryposition") == ("pixel", "photodetectoryposition")
    assert ie_parameter_otype("pd position") == ("pixel", "pdposition")
    assert ie_parameter_otype("pdposition") == ("pixel", "pdposition")
    assert ie_parameter_otype("photodetectorwidth") == ("pixel", "photodetectorwidth")
    assert ie_parameter_otype("photodetectorheight") == ("pixel", "photodetectorheight")
    assert ie_parameter_otype("photodetectorsize") == ("pixel", "photodetectorsize")
    assert ie_parameter_otype("pixelspectrum") == ("pixel", "pixelspectrum")
    assert ie_parameter_otype("pixelwavelength") == ("pixel", "pixelwavelength")
    assert ie_parameter_otype("pixelwavelengthsamples") == ("pixel", "pixelwavelengthsamples")
    assert ie_parameter_otype("pixelbinwidth") == ("pixel", "pixelbinwidth")
    assert ie_parameter_otype("pixelnwave") == ("pixel", "pixelnwave")
    assert ie_parameter_otype("quantum efficiency") == ("pixel", "quantumefficiency")
    assert ie_parameter_otype("pixel quantum efficiency") == ("pixel", "quantumefficiency")
    assert ie_parameter_otype("photodetector quantum efficiency") == ("pixel", "photodetectorquantumefficiency")
    assert ie_parameter_otype("photodetector spectral quantum efficiency") == ("pixel", "photodetectorspectralquantumefficiency")
    assert ie_parameter_otype("pixeldynamicrange") == ("pixel", "pixeldynamicrange")
    assert ie_parameter_otype("diffusion MTF") == ("sensor", "diffusionmtf")
    assert ie_parameter_otype("ag") == ("sensor", "ag")
    assert ie_parameter_otype("ao") == ("sensor", "ao")
    assert ie_parameter_otype("human cone densities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("human cone seed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("mcccornerpoints") == ("sensor", "mcccornerpoints")
    assert ie_parameter_otype("mcc corner points") == ("sensor", "mcccornerpoints")
    assert ie_parameter_otype("mccrecthandles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("mcc rect handles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("consistency") == ("sensor", "consistency")
    assert ie_parameter_otype("sensor consistency") == ("sensor", "consistency")
    assert ie_parameter_otype("sensor compute") == ("sensor", "compute")
    assert ie_parameter_otype("sensor compute method") == ("sensor", "computemethod")
    assert ie_parameter_otype("integration time") == ("sensor", "integrationtime")
    assert ie_parameter_otype("exptime") == ("sensor", "exptime")
    assert ie_parameter_otype("exptimes") == ("sensor", "exptimes")
    assert ie_parameter_otype("exposuretimes") == ("sensor", "exposuretimes")
    assert ie_parameter_otype("exposuretime") == ("sensor", "exposuretime")
    assert ie_parameter_otype("expduration") == ("sensor", "expduration")
    assert ie_parameter_otype("exposureduration") == ("sensor", "exposureduration")
    assert ie_parameter_otype("exposuredurations") == ("sensor", "exposuredurations")
    assert ie_parameter_otype("uniqueintegrationtimes") == ("sensor", "uniqueintegrationtimes")
    assert ie_parameter_otype("uniqueexptime") == ("sensor", "uniqueexptime")
    assert ie_parameter_otype("unique exptimes") == ("sensor", "uniqueexptimes")
    assert ie_parameter_otype("uniqueexptimes") == ("sensor", "uniqueexptimes")
    assert ie_parameter_otype("geometricmeanexposuretime") == ("sensor", "geometricmeanexposuretime")
    assert ie_parameter_otype("central exposure") == ("sensor", "centralexposure")
    assert ie_parameter_otype("centralexposure") == ("sensor", "centralexposure")
    assert ie_parameter_otype("expmethod") == ("sensor", "expmethod")
    assert ie_parameter_otype("exposuremethod") == ("sensor", "exposuremethod")
    assert ie_parameter_otype("exposure method") == ("sensor", "exposuremethod")
    assert ie_parameter_otype("nexposures") == ("sensor", "nexposures")
    assert ie_parameter_otype("n exposures") == ("sensor", "nexposures")
    assert ie_parameter_otype("exposureplane") == ("sensor", "exposureplane")
    assert ie_parameter_otype("exposure plane") == ("sensor", "exposureplane")
    assert ie_parameter_otype("cds") == ("sensor", "cds")
    assert ie_parameter_otype("correlateddoublesampling") == ("sensor", "correlateddoublesampling")
    assert ie_parameter_otype("correlated double sampling") == ("sensor", "correlateddoublesampling")
    assert ie_parameter_otype("autoexp") == ("sensor", "autoexp")
    assert ie_parameter_otype("autoexposure") == ("sensor", "autoexposure")
    assert ie_parameter_otype("auto exposure") == ("sensor", "autoexposure")
    assert ie_parameter_otype("automaticexposure") == ("sensor", "automaticexposure")
    assert ie_parameter_otype("automatic exposure") == ("sensor", "automaticexposure")
    assert ie_parameter_otype("pixelsamples") == ("sensor", "pixelsamples")
    assert ie_parameter_otype("pixel samples") == ("sensor", "pixelsamples")
    assert ie_parameter_otype("npixelsamplesforcomputing") == ("sensor", "npixelsamplesforcomputing")
    assert ie_parameter_otype("n pixel samples for computing") == ("sensor", "npixelsamplesforcomputing")
    assert ie_parameter_otype("spatialsamplesperpixel") == ("sensor", "spatialsamplesperpixel")
    assert ie_parameter_otype("spatial samples per pixel") == ("sensor", "spatialsamplesperpixel")
    assert ie_parameter_otype("reusenoise") == ("sensor", "reusenoise")
    assert ie_parameter_otype("noiseseed") == ("sensor", "noiseseed")
    assert ie_parameter_otype("responsetype") == ("sensor", "responsetype")
    assert ie_parameter_otype("response type") == ("sensor", "responsetype")
    assert ie_parameter_otype("eye movement") == ("sensor", "eyemovement")
    assert ie_parameter_otype("eyemovement") == ("sensor", "eyemovement")
    assert ie_parameter_otype("sensormovement") == ("sensor", "sensormovement")
    assert ie_parameter_otype("movementpositions") == ("sensor", "movementpositions")
    assert ie_parameter_otype("movement positions") == ("sensor", "movementpositions")
    assert ie_parameter_otype("sensorpositions") == ("sensor", "sensorpositions")
    assert ie_parameter_otype("sensor positions") == ("sensor", "positions")
    assert ie_parameter_otype("sensorpositionsx") == ("sensor", "sensorpositionsx")
    assert ie_parameter_otype("sensor positions x") == ("sensor", "positionsx")
    assert ie_parameter_otype("sensorpositionsy") == ("sensor", "sensorpositionsy")
    assert ie_parameter_otype("sensor positions y") == ("sensor", "positionsy")
    assert ie_parameter_otype("framesperposition") == ("sensor", "framesperposition")
    assert ie_parameter_otype("framesperpositions") == ("sensor", "framesperpositions")
    assert ie_parameter_otype("frames per position") == ("sensor", "framesperposition")
    assert ie_parameter_otype("exposuretimesperposition") == ("sensor", "exposuretimesperposition")
    assert ie_parameter_otype("exposure times per position") == ("sensor", "exposuretimesperposition")
    assert ie_parameter_otype("etimeperpos") == ("sensor", "etimeperpos")
    assert ie_parameter_otype("etime per pos") == ("sensor", "etimeperpos")
    assert ie_parameter_otype("human") == ("sensor", "human")
    assert ie_parameter_otype("cone type") == ("sensor", "conetype")
    assert ie_parameter_otype("conetype") == ("sensor", "conetype")
    assert ie_parameter_otype("humanconetype") == ("sensor", "humanconetype")
    assert ie_parameter_otype("human cone type") == ("sensor", "humanconetype")
    assert ie_parameter_otype("densities") == ("sensor", "densities")
    assert ie_parameter_otype("humanconedensities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("cone xy") == ("sensor", "conexy")
    assert ie_parameter_otype("conexy") == ("sensor", "conexy")
    assert ie_parameter_otype("cone locs") == ("sensor", "conelocs")
    assert ie_parameter_otype("conelocs") == ("sensor", "conelocs")
    assert ie_parameter_otype("xy") == ("sensor", "xy")
    assert ie_parameter_otype("humanconelocs") == ("sensor", "humanconelocs")
    assert ie_parameter_otype("human cone locs") == ("sensor", "humanconelocs")
    assert ie_parameter_otype("rseed") == ("sensor", "rseed")
    assert ie_parameter_otype("human rseed") == ("sensor", "humanrseed")
    assert ie_parameter_otype("humanrseed") == ("sensor", "humanrseed")
    assert ie_parameter_otype("humanconeseed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("chart parameters") == ("sensor", "chartparameters")
    assert ie_parameter_otype("chartparameters") == ("sensor", "chartparameters")
    assert ie_parameter_otype("corner points") == ("sensor", "cornerpoints")
    assert ie_parameter_otype("cornerpoints") == ("sensor", "cornerpoints")
    assert ie_parameter_otype("chart corner points") == ("sensor", "chartcornerpoints")
    assert ie_parameter_otype("chartcornerpoints") == ("sensor", "chartcornerpoints")
    assert ie_parameter_otype("chart corners") == ("sensor", "chartcorners")
    assert ie_parameter_otype("chartcorners") == ("sensor", "chartcorners")
    assert ie_parameter_otype("chart rects") == ("sensor", "chartrects")
    assert ie_parameter_otype("chartrects") == ("sensor", "chartrects")
    assert ie_parameter_otype("chart rectangles") == ("sensor", "chartrectangles")
    assert ie_parameter_otype("chartrectangles") == ("sensor", "chartrectangles")
    assert ie_parameter_otype("current rect") == ("sensor", "currentrect")
    assert ie_parameter_otype("currentrect") == ("sensor", "currentrect")
    assert ie_parameter_otype("chart current rect") == ("sensor", "chartcurrentrect")
    assert ie_parameter_otype("chartcurrentrect") == ("sensor", "chartcurrentrect")
    assert ie_parameter_otype("mccrecthandles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("scene name") == ("sensor", "scenename")
    assert ie_parameter_otype("metadata scene name") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("metadatalensname") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadata lens name") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadatalens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata lens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata optics name") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadata sensor name") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadata crop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("fnumber") == ("optics", "fnumber")
    assert ie_parameter_otype("asset light") == ("asset", "assetlight")
    assert ieParameterOtype("ip display") == ("ip", "display")


def test_ie_parameter_otype_returns_empty_type_for_ambiguous_or_unknown_params() -> None:
    assert ie_parameter_otype("size") == ("", "size")
    assert ie_parameter_otype("mystery parameter") == ("", "mysteryparameter")


def test_convolvecirc_and_image_slanted_edge_match_legacy_aliases() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    kernel = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)

    expected_conv = np.array([[9.0, 7.0], [13.0, 11.0]], dtype=float)
    assert np.allclose(convolve_circ(image, kernel), expected_conv)
    assert np.allclose(convolvecirc(image, kernel), expected_conv)

    edge = image_slanted_edge([4, 6], 1.0, 0.25)
    alias = imageSlantedEdge([4, 6], 1.0, 0.25)
    assert edge.shape == (5, 7)
    assert np.allclose(alias, edge)
    assert float(edge[0, -1]) == pytest.approx(0.25)
    assert float(edge[-1, 0]) == pytest.approx(1.0)


def test_imagesc_helpers_match_headless_scaling_and_aliases() -> None:
    rgb = np.array(
        [
            [[-0.5, 0.5, 1.0], [0.25, 0.75, 0.5]],
            [[0.0, 0.1, 0.2], [0.8, 0.6, 0.4]],
        ],
        dtype=float,
    )
    handle, scaled_rgb = imagesc_rgb(rgb, 0.5)
    alias_handle, alias_scaled = imagescRGB(rgb, 0.5)
    assert handle is None
    assert alias_handle is None
    assert scaled_rgb.shape == rgb.shape
    assert np.allclose(alias_scaled, scaled_rgb)
    assert np.all((scaled_rgb >= 0.0) & (scaled_rgb <= 1.0))

    xw, rows, cols, _ = rgb_to_xw_format(np.clip(rgb, 0.0, None))
    _, scaled_xw = imagescRGB(xw, rows, cols, 1.0)
    assert scaled_xw.shape == rgb.shape

    opp = np.dstack(
        (
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
            np.array([[-1.0, 0.0], [1.0, 0.5]], dtype=float),
            np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float),
        )
    )
    res, cmap = imagesc_opp(opp, 0.4, 8)
    alias_res, alias_cmap = imagescOPP(opp, 0.4, 8)
    assert res.shape == opp.shape
    assert cmap.shape == (8, 3, 3)
    assert np.allclose(alias_res, res)
    assert np.allclose(alias_cmap, cmap)

    mono = np.arange(4.0, dtype=float).reshape(2, 2)
    payload = imagesc_m(mono, bar_dir="eastoutside")
    alias_payload = imagescM(mono, bar_dir="eastoutside")
    assert payload is not None
    assert alias_payload is not None
    assert payload["scaled"] is True
    assert payload["colorbar"]["direction"] == "eastoutside"
    assert np.allclose(alias_payload["image"], mono)


def test_image_spd_visible_gray_and_spd2rgb_match_aliases() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    spd = np.zeros((2, 2, wave.size), dtype=float)
    spd[:, :, 10] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    spd[:, :, 20] = np.array([[0.5, 0.25], [0.75, 1.5]], dtype=float)

    visible = image_spd(spd, wave, 1.0, display_flag=-1)
    alias_visible = imageSPD(spd, wave, 1.0, 2, 2, -1)
    expected_xyz = np.asarray(ieXYZFromPhotons(spd, wave), dtype=float)
    expected_xyz = expected_xyz / float(np.max(expected_xyz))
    expected_visible = xyz2srgb(expected_xyz)
    assert np.allclose(visible, expected_visible)
    assert np.allclose(alias_visible, expected_visible)

    gray = image_spd(spd, wave, 1.0, 2, 2, -2)
    expected_gray = np.mean(spd, axis=2)
    expected_gray = expected_gray / float(np.max(expected_gray))
    expected_gray = np.repeat(expected_gray[:, :, np.newaxis], 3, axis=2)
    assert np.allclose(gray, expected_gray)

    spd_xw, _, _, _ = rgb_to_xw_format(spd)
    rgb_xw = image_spd2rgb(spd_xw, wave, 1.0)
    alias_rgb_xw = imageSPD2RGB(spd_xw, wave, 1.0)
    visible_xw, _, _, _ = rgb_to_xw_format(expected_visible)
    assert np.allclose(rgb_xw, visible_xw)
    assert np.allclose(alias_rgb_xw, visible_xw)


def test_image_hc2rgb_returns_waveband_stack_and_overlay(asset_store) -> None:
    scene = sceneCreate("slanted bar", 64, asset_store=asset_store)
    rows = int(sceneGet(scene, "rows"))
    cols = int(sceneGet(scene, "cols"))

    rgb_images, overlay = image_hc2rgb(scene, 3, [10, 10])
    alias_images, alias_overlay = imagehc2rgb(scene, 3, [10, 10])

    assert rgb_images.shape == (rows, cols, 3, 3)
    assert overlay.shape[0] > rows
    assert overlay.shape[1] > cols
    assert np.all((rgb_images >= 0.0) & (rgb_images <= 1.0))
    assert np.all((overlay >= 0.0) & (overlay <= 1.0))
    assert np.allclose(alias_images, rgb_images)
    assert np.allclose(alias_overlay, overlay)
