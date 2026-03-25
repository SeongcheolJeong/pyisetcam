from __future__ import annotations

import numpy as np

from pyisetcam import (
    displayCreate,
    displayGet,
    humanAchromaticOTF,
    humanConeContrast,
    humanConeIsolating,
    humanCore,
    humanLSF,
    humanMacularTransmittance,
    humanOpticalDensity,
    humanOTF,
    humanOTF_ibio,
    humanPupilSize,
    humanSpaceTime,
    ijspeert,
    kellySpaceTime,
    poirsonSpatioChromatic,
    watsonImpulseResponse,
    watsonRGCSpacing,
    westheimerLSF,
)
from pyisetcam.assets import AssetStore
from pyisetcam import oiCreate, oiGet
from pyisetcam.utils import energy_to_quanta


def test_human_pupil_size_models_match_analytic_forms() -> None:
    diameter_ms, area_ms = humanPupilSize(100.0, "ms")
    diameter_dg, area_dg = humanPupilSize(100.0, "dg")
    diameter_sd, area_sd = humanPupilSize(100.0, "sd", 10.0)
    diameter_wy, area_wy = humanPupilSize(100.0, "wy", {"age": 30, "area": 10.0, "eyeNum": 2})

    expected_ms = 4.9 - 3.0 * np.tanh(0.4 * np.log10(100.0) + 1.0)
    expected_dg = 10.0 ** (0.8558 - 0.000401 * (np.log10(100.0) + 8.6) ** 3)
    flux = 100.0 * 10.0
    expected_sd = 7.75 - 5.75 * (flux / 846.0) ** 0.41 / ((flux / 846.0) ** 0.41 + 2.0)
    expected_wy = expected_sd + (30.0 - 28.58) * (0.02132 - 0.009562 * expected_sd)

    assert np.isclose(diameter_ms, expected_ms)
    assert np.isclose(diameter_dg, expected_dg)
    assert np.isclose(diameter_sd, expected_sd)
    assert np.isclose(diameter_wy, expected_wy)
    assert np.isclose(area_ms, np.pi * (diameter_ms / 2.0) ** 2)
    assert np.isclose(area_dg, np.pi * (diameter_dg / 2.0) ** 2)
    assert np.isclose(area_sd, np.pi * (diameter_sd / 2.0) ** 2)
    assert np.isclose(area_wy, np.pi * (diameter_wy / 2.0) ** 2)


def test_human_optical_density_matches_stockman_profiles() -> None:
    fovea = humanOpticalDensity()
    periphery = humanOpticalDensity("stockman periphery", np.arange(400.0, 701.0, 10.0, dtype=float))
    subject = humanOpticalDensity("s1p")

    assert fovea["visfield"] == "fovea"
    assert np.isclose(fovea["macular"], 0.28)
    assert periphery["wave"].shape == (31,)
    assert np.isclose(periphery["LPOD"], 0.38)
    assert np.isclose(periphery["SPOD"], 0.3)
    assert subject["visfield"] == "p"
    assert np.isclose(subject["macular"], 0.0)
    assert np.isclose(subject["LPOD"], (0.4964 / 0.5) * 0.38)


def test_human_cone_contrast_energy_and_quanta_agree() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    background = np.full(wave.shape, 0.25, dtype=float)
    signal = np.column_stack((background * 0.1, background * 0.2))

    contrast_energy = humanConeContrast(signal, background, wave, "energy")
    contrast_quanta = humanConeContrast(
        energy_to_quanta(signal, wave),
        energy_to_quanta(background, wave),
        wave,
        "quanta",
    )

    assert contrast_energy.shape == (3, 2)
    np.testing.assert_allclose(contrast_quanta, contrast_energy)


def test_human_cone_isolating_returns_scaled_isolating_directions() -> None:
    display = displayCreate("LCD-Apple")
    cone_isolating, spd = humanConeIsolating(display)

    wave = np.asarray(displayGet(display, "wave"), dtype=float).reshape(-1)
    _, cones = AssetStore.default().load_spectra("stockman.mat", wave_nm=wave)
    isolation = np.asarray(cones, dtype=float).T @ spd
    off_diagonal = isolation - np.diag(np.diag(isolation))

    assert cone_isolating.shape == (3, 3)
    assert spd.shape == (wave.size, 3)
    np.testing.assert_allclose(np.max(np.abs(cone_isolating), axis=0), 0.5)
    assert np.max(np.abs(off_diagonal)) < 2.0e-4
    assert np.all(np.diag(isolation) > 0.0)


def test_watson_impulse_response_is_normalized() -> None:
    impulse_response, time, t_mtf, frequency = watsonImpulseResponse(np.linspace(0.0, 0.2, 11), 0.25)

    assert np.all(time > 0.0)
    assert np.isclose(np.sum(impulse_response), 1.0)
    assert t_mtf.shape == impulse_response.shape
    assert frequency.shape == impulse_response.shape


def test_kelly_space_time_and_human_space_time_dispatch() -> None:
    spatial = np.array([1.0, 2.0], dtype=float)
    temporal = np.array([4.0, 8.0], dtype=float)

    sensitivity, spatial_grid, temporal_grid = kellySpaceTime(spatial, temporal)
    dispatched, dispatched_spatial, dispatched_temporal = humanSpaceTime("kelly79", spatial, temporal)

    expected = (
        (6.1 + 7.3 * abs(np.log10((4.0 / 1.0) / 3.0)) ** 3)
        * (4.0 / 1.0)
        * (2.0 * np.pi * 1.0) ** 2
        * np.exp(-2.0 * (2.0 * np.pi * 1.0) / (45.9 / ((4.0 / 1.0) + 2.0)))
        / 2.0
    )

    assert sensitivity.shape == (2, 2)
    assert np.isclose(sensitivity[0, 0], expected)
    np.testing.assert_allclose(dispatched, sensitivity)
    np.testing.assert_allclose(dispatched_spatial, spatial_grid)
    np.testing.assert_allclose(dispatched_temporal, temporal_grid)


def test_poirson_spatio_chromatic_and_human_space_time_color_dispatch() -> None:
    lum_1d, rg_1d, by_1d, positions = poirsonSpatioChromatic(120.0, 1)
    lum_2d, rg_2d, by_2d, positions_2d = poirsonSpatioChromatic(120.0, 2)
    dispatched, support, _ = humanSpaceTime("poirsoncolor")

    assert lum_1d.ndim == 1
    assert lum_2d.ndim == 2
    assert np.isclose(np.sum(lum_1d), 1.0)
    assert np.isclose(np.sum(rg_1d), 1.0)
    assert np.isclose(np.sum(by_1d), 1.0)
    assert np.isclose(np.sum(lum_2d), 1.0)
    assert np.isclose(np.sum(rg_2d), 1.0)
    assert np.isclose(np.sum(by_2d), 1.0)
    assert np.isclose(positions[positions.size // 2], 0.0)
    np.testing.assert_allclose(positions, positions_2d)
    assert set(dispatched.keys()) == {"lum", "rg", "by"}
    assert support.ndim == 1


def test_westheimer_lsf_is_symmetric_and_normalized() -> None:
    support = np.arange(-300.0, 301.0, 1.0, dtype=float)
    line_spread = westheimerLSF(support)

    assert np.isclose(np.sum(line_spread), 1.0)
    np.testing.assert_allclose(line_spread, line_spread[::-1])


def test_watson_rgc_spacing_returns_expected_shapes() -> None:
    smf0, r, smf1d = watsonRGCSpacing(4)

    assert smf0.shape == (5, 5)
    assert r.shape == (1000,)
    assert smf1d.shape == (4, 1000)
    assert np.isfinite(smf0[2, 2])
    assert np.all(smf1d > 0.0)


def test_human_achromatic_otf_models_and_core_are_consistent() -> None:
    sample_sf = np.array([0.0, 10.0, 30.0], dtype=float)
    exp_otf = humanAchromaticOTF(sample_sf, "exp")
    dl_otf = humanAchromaticOTF(sample_sf, "dl", 3.0)
    watson_otf = humanAchromaticOTF(sample_sf, "watson", 3.0)
    core, achromatic = humanCore(np.array([500.0, 600.0], dtype=float), sample_sf, 0.0015, 60.0)

    assert np.isclose(exp_otf[0], 1.0)
    assert np.isclose(dl_otf[0], 1.0)
    assert np.isclose(watson_otf[0], 1.0)
    assert core.shape == (2, 3)
    np.testing.assert_allclose(achromatic, exp_otf)


def test_human_otf_and_lsf_return_headless_payloads() -> None:
    freq = np.linspace(-20.0, 20.0, 11, dtype=float)
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    support = np.dstack((fx, fy))

    otf, returned_support, wave = humanOTF(0.0015, 60.0, support, np.array([550.0, 650.0], dtype=float))
    line_spread_mm, x_mm, _ = humanLSF(0.0015, 60.0, "mm", np.array([550.0, 650.0], dtype=float))
    line_spread_um, x_um, _ = humanLSF(0.0015, 60.0, "um", np.array([550.0, 650.0], dtype=float))

    assert otf.shape == (11, 11, 2)
    np.testing.assert_allclose(returned_support, support)
    np.testing.assert_allclose(wave, np.array([550.0, 650.0], dtype=float))
    centered_otf = np.abs(np.fft.ifftshift(otf[:, :, 0]))
    assert np.isclose(centered_otf[centered_otf.shape[0] // 2, centered_otf.shape[1] // 2], 1.0)
    assert line_spread_mm.shape[0] == 2
    np.testing.assert_allclose(x_um / 1000.0, x_mm)
    assert line_spread_um.shape == line_spread_mm.shape


def test_human_otf_ibio_and_macular_transmittance_match_headless_contracts() -> None:
    freq = np.linspace(-20.0, 20.0, 11, dtype=float)
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    support = np.dstack((fx, fy))

    legacy_otf, returned_support, wave = humanOTF(0.0015, 60.0, support, np.array([550.0], dtype=float))
    ibio_otf, returned_support_ibio, wave_ibio = humanOTF_ibio(0.0015, 60.0, support, np.array([550.0], dtype=float))

    assert np.isclose(np.abs(ibio_otf[0, 0, 0]), 1.0)
    np.testing.assert_allclose(returned_support_ibio, returned_support)
    np.testing.assert_allclose(wave_ibio, wave)
    np.testing.assert_allclose(np.fft.ifftshift(np.fft.ifftshift(legacy_otf[:, :, 0])), ibio_otf[:, :, 0])

    oi = humanMacularTransmittance(oiCreate(), 0.35)
    current_wave = np.asarray(oiGet(oi, "wave"), dtype=float).reshape(-1)
    transmittance = np.asarray(oiGet(oi, "transmittance", current_wave), dtype=float).reshape(-1)
    assert transmittance.shape == current_wave.shape
    assert np.all(transmittance <= 1.0)
    assert np.all(transmittance >= 0.0)
    assert float(transmittance[np.argmin(np.abs(current_wave - 460.0))]) < float(
        transmittance[np.argmin(np.abs(current_wave - 650.0))]
    )


def test_ijspeert_returns_mtf_psf_and_lsf() -> None:
    sample_sf = np.arange(0.0, 61.0, 1.0, dtype=float)
    phi = np.deg2rad(np.linspace(0.0, 0.1, 10, dtype=float))

    mtf, psf, lsf = ijspeert(30.0, 3.0, 0.142, sample_sf, phi)

    assert mtf.shape == sample_sf.shape
    assert psf is not None and psf.shape == phi.shape
    assert lsf is not None and lsf.shape == phi.shape
    assert np.isclose(mtf[0], 1.0, atol=1e-6)
