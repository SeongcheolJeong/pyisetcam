"""Curated parity case runners."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Any

import imageio.v3 as iio
import numpy as np

from .assets import AssetStore, ie_read_color_filter, ie_read_spectra
from .camera import camera_compute, camera_create, camera_get, camera_set
from .description import sensor_description
from .display import display_create
from .fileio import ie_save_color_filter, ie_save_si_data_file
from .fileio import sensor_dng_read
from .metrics import cct_from_uv, delta_e_ab, metrics_spd, xyz_from_energy, xyz_to_lab, xyz_to_luv, xyz_to_uv
from .ip import ip_compute, ip_create, ip_get, ip_set
from .optics import (
    _cos4th_factor,
    _oi_geometry,
    _pad_scene,
    _radiance_to_irradiance,
    _shift_invariant_custom_otf,
    _wvf_psf_stack,
    airy_disk,
    optics_build_2d_otf,
    optics_coc,
    optics_defocus_core,
    optics_defocus_displacement,
    optics_dof,
    oi_compute,
    oi_crop,
    oi_create,
    oi_get,
    oi_spatial_resample,
    psf_to_zcoeff_error,
    wvf_compute,
    wvf_aperture,
    wvf_compute_psf,
    wvf_create,
    wvf_defocus_diopters_to_microns,
    wvf_get,
    wvf_load_thibos_virtual_eyes,
    wvf_osa_index_to_zernike_nm,
    wvf_pupil_function,
    wvf_set,
    wvf_to_oi,
    wvf_zernike_nm_to_osa_index,
    optics_psf_to_otf,
    oi_set,
    si_synthetic,
)
from .plotting import ip_plot, oi_plot, sensor_plot, sensor_plot_line, wvf_plot
from .scene import (
    scene_adjust_illuminant,
    scene_adjust_luminance,
    scene_combine,
    scene_create,
    scene_from_file,
    scene_get,
    scene_interpolate_w,
    scene_rotate,
    scene_set,
)
from .sensor import (
    imx490_compute,
    ml_analyze_array_etendue,
    ml_radiance,
    mlens_create,
    mlens_get,
    mlens_set,
    sensor_color_filter,
    sensor_ccm,
    sensor_compute,
    sensor_compute_array,
    sensor_compute_samples,
    sensor_create,
    sensor_create_array,
    sensor_create_ideal,
    sensor_crop,
    sensor_dr,
    sensor_get,
    sensor_set,
)
from .sensor import sensor_snr
from .sensor import sensor_set_size_to_fov
from .sensor import signal_current
from .utils import blackbody, energy_to_quanta, ie_fit_line, ie_mvnrnd, param_format, quanta_to_energy, unit_frequency_list


@dataclass
class ParityCaseResult:
    payload: dict[str, Any]
    context: dict[str, Any]


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def _stats_vector(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.percentile(flat, 5)),
            float(np.percentile(flat, 95)),
        ],
        dtype=float,
    )


def _deterministic_normal_samples(n_rows: int, n_cols: int) -> np.ndarray:
    indices = np.arange(1, n_rows * n_cols + 1, dtype=float).reshape(n_rows, n_cols, order="F")
    u1 = np.mod(indices * 0.7548776662466927, 1.0)
    u2 = np.mod(indices * 0.5698402909980532, 1.0)
    u1 = np.clip(u1, 1e-6, 1.0 - 1e-6)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _channel_normalize(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    return vector / max(float(np.max(np.abs(vector))), 1e-12)


def _find_nearest_two(array: Any, number: float) -> np.ndarray:
    values = np.asarray(array, dtype=float).reshape(-1)
    order = np.argsort(np.abs(values - float(number)), kind="mergesort")
    return values[order[:2]]


def _circle_mask(radius: float, img_size: tuple[int, int]) -> np.ndarray:
    rows, cols = int(img_size[0]), int(img_size[1])
    x, y = np.meshgrid(np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float))
    center_x = cols / 2.0
    center_y = rows / 2.0
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return dist <= float(radius)


def _generate_fringe_psf(zernike_coeffs: Any, *, grid_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    coeffs = np.asarray(zernike_coeffs, dtype=float).reshape(-1)
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, grid_size, dtype=float), np.linspace(-1.0, 1.0, grid_size, dtype=float))
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    zernike_terms = [
        lambda r, t: np.ones_like(r, dtype=float),
        lambda r, t: r * np.cos(t),
        lambda r, t: r * np.sin(t),
        lambda r, t: -1.0 + 2.0 * r**2,
        lambda r, t: r**2 * np.cos(2.0 * t),
        lambda r, t: r**2 * np.sin(2.0 * t),
        lambda r, t: (-2.0 * r + 3.0 * r**3) * np.cos(t),
        lambda r, t: (-2.0 * r + 3.0 * r**3) * np.sin(t),
        lambda r, t: 1.0 - 6.0 * r**2 + 6.0 * r**4,
        lambda r, t: r**3 * np.cos(3.0 * t),
        lambda r, t: r**3 * np.sin(3.0 * t),
        lambda r, t: (-3.0 * r**2 + 4.0 * r**4) * np.cos(2.0 * t),
        lambda r, t: (-3.0 * r**2 + 4.0 * r**4) * np.sin(2.0 * t),
        lambda r, t: (3.0 * r - 12.0 * r**3 + 10.0 * r**5) * np.cos(t),
        lambda r, t: (3.0 * r - 12.0 * r**3 + 10.0 * r**5) * np.sin(t),
    ]

    wavefront = np.zeros_like(rho, dtype=float)
    for index, coefficient in enumerate(coeffs):
        wavefront = wavefront + float(coefficient) * zernike_terms[index](rho, theta)

    aperture_mask = _circle_mask(round(grid_size / 2.0), (grid_size, grid_size)).astype(float)
    pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront)
    amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_phase * aperture_mask)))
    intensity = np.real(amplitude * np.conj(amplitude))
    return intensity, wavefront


def run_python_case_with_context(
    case_name: str,
    *,
    asset_store: AssetStore | None = None,
) -> ParityCaseResult:
    store = asset_store or AssetStore.default()

    if case_name == "scene_macbeth_default":
        scene = scene_create(asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_checkerboard_small":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_bb_small":
        scene = scene_create("uniform bb", 16, 4500, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_frequency_orientation_small":
        params = {
            "angles": np.linspace(0.0, np.pi / 2.0, 4),
            "freqs": np.array([1.0, 2.0, 4.0, 8.0], dtype=float),
            "blockSize": 16,
            "contrast": 0.8,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_harmonic_small":
        params = {
            "freq": np.array([1.0, 5.0], dtype=float),
            "contrast": np.array([0.2, 0.6], dtype=float),
            "ph": np.array([0.0, np.pi / 3.0], dtype=float),
            "ang": np.array([0.0, 0.0], dtype=float),
            "row": 64,
            "col": 64,
            "GaborFlag": 0.2,
        }
        scene = scene_create("harmonic", params, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_sweep_frequency_small":
        scene = scene_create("sweep frequency", 64, 12, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_reflectance_chart_small":
        scene = scene_create(
            "reflectance chart",
            8,
            [[1, 2], [1, 2], [1]],
            [
                store.resolve("data/surfaces/reflectances/MunsellSamples_Vhrel.mat"),
                store.resolve("data/surfaces/reflectances/Food_Vhrel.mat"),
                store.resolve("data/surfaces/reflectances/skin/HyspexSkinReflectance.mat"),
            ],
            None,
            True,
            "without replacement",
            asset_store=store,
        )
        chart_parameters = scene_get(scene, "chart parameters")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
                "chart_rowcol": np.asarray(chart_parameters["rowcol"], dtype=int),
                "chart_index_map": np.asarray(chart_parameters["rIdxMap"], dtype=int),
            },
            context={"scene": scene},
        )

    if case_name == "scene_star_pattern_small":
        scene = scene_create("star pattern", 64, "ee", 6, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "utility_unit_frequency_list":
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "even": unit_frequency_list(50),
                "odd": unit_frequency_list(51),
            },
            context={},
        )

    if case_name == "utility_energy_quanta_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.1, 3.1, wave.size, dtype=float)
        photons = energy_to_quanta(energy, wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "photons": photons,
                "energy_roundtrip": quanta_to_energy(photons, wave),
            },
            context={},
        )

    if case_name == "utility_energy_quanta_matrix":
        wave = np.array([400.0, 500.0, 600.0], dtype=float)
        energy = np.array(
            [
                [0.2, 0.4],
                [0.5, 0.7],
                [0.8, 1.0],
            ],
            dtype=float,
        )
        photons = energy_to_quanta(energy, wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "photons": photons,
                "energy_roundtrip": quanta_to_energy(photons, wave),
            },
            context={},
        )

    if case_name == "utility_blackbody_energy_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        temperatures = np.array([3000.0, 5000.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "temperatures": temperatures,
                "energy": blackbody(wave, temperatures, kind="energy"),
            },
            context={},
        )

    if case_name == "utility_blackbody_quanta_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        temperatures = np.array([3000.0, 5000.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "temperatures": temperatures,
                "photons": blackbody(wave, temperatures, kind="quanta"),
            },
            context={},
        )

    if case_name == "utility_ie_param_format_string":
        original = "Exposure Time"
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "original": original,
                "formatted": param_format(original),
            },
            context={},
        )

    if case_name == "wvf_load_thibos_virtual_eyes_small":
        sample_mean, sample_cov, subject_coeffs = wvf_load_thibos_virtual_eyes(6.0, asset_store=store, full=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pupil_diameter_mm": 6.0,
                "sample_mean": sample_mean,
                "sample_cov": sample_cov,
                "left_eye": np.asarray(subject_coeffs["left_eye"], dtype=float),
                "right_eye": np.asarray(subject_coeffs["right_eye"], dtype=float),
                "both_eyes": np.asarray(subject_coeffs["both_eyes"], dtype=float),
            },
            context={},
        )

    if case_name == "wvf_thibos_model_small":
        measured_pupil_mm = 4.5
        calc_pupil_mm = 3.0
        measured_wavelength_nm = 550.0
        calc_waves = np.arange(450.0, 651.0, 100.0, dtype=float)
        sample_mean, sample_cov, _ = wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=store, full=True)
        example_count = 10
        example_subject_indices = np.arange(0, example_count, 3, dtype=int)
        standard_normal = _deterministic_normal_samples(example_count, sample_mean.size)
        example_coeffs = ie_mvnrnd(sample_mean, sample_cov, standard_normal_samples=standard_normal)

        sample_mean_zcoeffs = np.zeros(65, dtype=float)
        sample_mean_zcoeffs[:13] = np.asarray(sample_mean[:13], dtype=float)

        this_guy = wvf_create()
        this_guy = wvf_set(this_guy, "zcoeffs", sample_mean_zcoeffs)
        this_guy = wvf_set(this_guy, "measured pupil", measured_pupil_mm)
        this_guy = wvf_set(this_guy, "calculated pupil", calc_pupil_mm)
        this_guy = wvf_set(this_guy, "measured wavelength", measured_wavelength_nm)
        this_guy = wvf_set(this_guy, "calc wave", calc_waves)
        this_guy = wvf_compute(this_guy)

        mean_subject_mid_rows: list[np.ndarray] = []
        mean_subject_peaks: list[float] = []
        for wavelength_nm in calc_waves:
            psf = np.asarray(wvf_get(this_guy, "psf", float(wavelength_nm)), dtype=float)
            mean_subject_mid_rows.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
            mean_subject_peaks.append(float(np.max(psf)))

        subject = wvf_create()
        subject = wvf_set(subject, "measured pupil", measured_pupil_mm)
        subject = wvf_set(subject, "calculated pupil", calc_pupil_mm)
        subject = wvf_set(subject, "measured wavelength", measured_wavelength_nm)

        subject_rows_450: list[np.ndarray] = []
        subject_rows_550: list[np.ndarray] = []
        subject_peaks_450: list[float] = []
        subject_peaks_550: list[float] = []
        selected_coeffs: list[np.ndarray] = []
        for subject_index in example_subject_indices:
            subject_zcoeffs = np.zeros(65, dtype=float)
            subject_zcoeffs[:13] = np.asarray(example_coeffs[subject_index, :13], dtype=float)
            selected_coeffs.append(np.asarray(subject_zcoeffs[:13], dtype=float))
            subject = wvf_set(subject, "zcoeffs", subject_zcoeffs)

            subject = wvf_set(subject, "calc wave", 450.0)
            subject = wvf_compute(subject)
            psf_450 = np.asarray(wvf_get(subject, "psf", 450.0), dtype=float)
            subject_rows_450.append(np.asarray(psf_450[psf_450.shape[0] // 2, :], dtype=float))
            subject_peaks_450.append(float(np.max(psf_450)))

            subject = wvf_set(subject, "calc wave", 550.0)
            subject = wvf_compute(subject)
            psf_550 = np.asarray(wvf_get(subject, "psf", 550.0), dtype=float)
            subject_rows_550.append(np.asarray(psf_550[psf_550.shape[0] // 2, :], dtype=float))
            subject_peaks_550.append(float(np.max(psf_550)))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupil_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "measured_wavelength_nm": measured_wavelength_nm,
                "calc_waves_nm": calc_waves,
                "example_coeffs": np.asarray(example_coeffs, dtype=float),
                "mean_subject_psf_mid_rows": np.vstack(mean_subject_mid_rows),
                "mean_subject_psf_peaks": np.asarray(mean_subject_peaks, dtype=float),
                "example_subject_indices": example_subject_indices + 1,
                "example_subject_coeffs": np.vstack(selected_coeffs),
                "example_subject_psf_mid_rows_450": np.vstack(subject_rows_450),
                "example_subject_psf_mid_rows_550": np.vstack(subject_rows_550),
                "example_subject_psf_peaks_450": np.asarray(subject_peaks_450, dtype=float),
                "example_subject_psf_peaks_550": np.asarray(subject_peaks_550, dtype=float),
            },
            context={"wvf": this_guy},
        )

    if case_name == "wvf_pupil_size_human_small":
        measured_pupil_mm = 7.5
        calc_pupil_mm = 3.0
        wavelength_nm = 520.0
        zcoeffs = wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=store)
        wvf = wvf_create(
            "calc wavelengths",
            np.array([wavelength_nm], dtype=float),
            "zcoeffs",
            zcoeffs,
            "measured pupil size",
            measured_pupil_mm,
            "calc pupil size",
            calc_pupil_mm,
            "name",
            "7-pupil",
        )
        wvf = wvf_set(wvf, "lcaMethod", "human")
        wvf = wvf_compute(wvf)
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        middle_row = psf.shape[0] // 2
        measured_wavelength_nm = float(wvf_get(wvf, "measured wavelength", "nm"))
        lca_diopters = 1.8859 - (1.8859 - (0.63346 / (0.001 * measured_wavelength_nm - 0.2141))) - (
            0.63346 / (0.001 * wavelength_nm - 0.2141)
        )
        lca_microns = np.asarray(
            wvf_defocus_diopters_to_microns(-lca_diopters, measured_pupil_mm),
            dtype=float,
        ).reshape(-1)[0]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupil_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "measured_wavelength_nm": measured_wavelength_nm,
                "wave": np.asarray(wvf_get(wvf, "wave"), dtype=float),
                "f_number": float(wvf_get(wvf, "fnumber")),
                "lca_diopters": float(lca_diopters),
                "lca_microns": float(lca_microns),
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": np.asarray(psf[middle_row, :], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_pupil_size_measured_compare_small":
        measured_pupils_mm = np.array([7.5, 6.0, 4.5, 3.0], dtype=float)
        calc_pupil_mm = 3.0
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        wavelength_nm = 550.0
        psf_mid_rows: list[np.ndarray] = []
        psf_peaks: list[float] = []
        psf_sums: list[float] = []
        max_abs_diffs: list[float] = []
        reference_psf: np.ndarray | None = None

        for measured_pupil_mm in measured_pupils_mm:
            zcoeffs = wvf_load_thibos_virtual_eyes(float(measured_pupil_mm), asset_store=store)
            wvf = wvf_create(
                "calc wavelengths",
                wave,
                "zcoeffs",
                zcoeffs,
                "measured pupil size",
                float(measured_pupil_mm),
                "calc pupil size",
                calc_pupil_mm,
                "name",
                f"{measured_pupil_mm:g}-pupil",
            )
            wvf = wvf_set(wvf, "lcaMethod", "human")
            wvf = wvf_compute(wvf)
            psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
            middle_row = psf.shape[0] // 2
            psf_mid_rows.append(np.asarray(psf[middle_row, :], dtype=float))
            psf_peaks.append(float(np.max(psf)))
            psf_sums.append(float(np.sum(psf)))
            if reference_psf is None:
                reference_psf = psf
                max_abs_diffs.append(0.0)
            else:
                max_abs_diffs.append(float(np.max(np.abs(reference_psf - psf))))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupils_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "wave": wave,
                "wavelength_nm": wavelength_nm,
                "psf_sum": np.asarray(psf_sums, dtype=float),
                "psf_peak_550": np.asarray(psf_peaks, dtype=float),
                "max_abs_diff_vs_first_550": np.asarray(max_abs_diffs, dtype=float),
                "psf_mid_row_550": np.asarray(psf_mid_rows, dtype=float),
            },
            context={},
        )

    if case_name == "wvf_psf_spacing_small":
        wavelength_nm = 550.0
        focal_length_mm = 4.0
        f_number = 4.0
        n_pixels = 1024
        psf_spacing_mm = 1e-3

        wvf = wvf_create()
        wvf = wvf_set(wvf, "wave", wavelength_nm)
        wvf = wvf_set(wvf, "focal length", focal_length_mm, "mm")
        wvf = wvf_set(wvf, "calc pupil diameter", focal_length_mm / f_number, "mm")
        wvf = wvf_set(wvf, "spatial samples", n_pixels)
        wvf = wvf_set(wvf, "psf sample spacing", psf_spacing_mm)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wavelength_nm": wavelength_nm,
                "focal_length_mm": focal_length_mm,
                "calc_pupil_diameter_mm": float(wvf_get(wvf, "calc pupil size", "mm")),
                "npixels": int(wvf_get(wvf, "npixels")),
                "field_size_mm": float(wvf_get(wvf, "field size mm", "mm")),
                "pupil_sample_spacing_mm": float(wvf_get(wvf, "pupil sample spacing", "mm", wavelength_nm)),
                "psf_sample_spacing_arcmin": float(wvf_get(wvf, "psf sample spacing")),
                "ref_psf_sample_interval_arcmin": float(wvf_get(wvf, "ref psf sample interval")),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_osa_index_conversion_small":
        indices = np.asarray([0, 1, 2, 5, 15, 20, 35], dtype=int)
        n, m = wvf_osa_index_to_zernike_nm(indices)
        roundtrip = np.asarray(wvf_zernike_nm_to_osa_index(n, m), dtype=int)
        scalar_n, scalar_m = wvf_osa_index_to_zernike_nm(15)
        scalar_roundtrip = int(wvf_zernike_nm_to_osa_index(scalar_n, scalar_m))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "indices": indices,
                "n": np.asarray(n, dtype=int),
                "m": np.asarray(m, dtype=int),
                "roundtrip_indices": roundtrip,
                "scalar_index": 15,
                "scalar_n": int(scalar_n),
                "scalar_m": int(scalar_m),
                "scalar_roundtrip_index": scalar_roundtrip,
            },
            context={},
        )

    if case_name == "metrics_xyz_from_energy_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.05, 1.55, wave.size, dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "xyz": xyz_from_energy(energy, wave, asset_store=store),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_luv_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "white_point": white_point,
                "luv": xyz_to_luv(xyz, white_point),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_lab_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "white_point": white_point,
                "lab": xyz_to_lab(xyz, white_point),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_uv_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "uv": xyz_to_uv(xyz),
            },
            context={},
        )

    if case_name == "optics_airy_disk_small":
        radius_um, image = airy_disk(550.0, 3.0, "units", "um", return_image=True)
        assert image is not None
        image_data = np.asarray(image["data"], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "radius_um": radius_um,
                "diameter_um": airy_disk(550.0, 3.0, "units", "um", "diameter", True),
                "radius_mm": airy_disk(550.0, 3.0, "units", "mm"),
                "radius_deg": airy_disk(700.0, None, "units", "deg", "pupil diameter", 1e-3),
                "radius_rad": airy_disk(700.0, None, "units", "rad", "pupil diameter", 1e-3),
                "image_rows": int(image_data.shape[0]),
                "image_cols": int(image_data.shape[1]),
            },
            context={},
        )

    if case_name == "optics_coc_small":
        base_optics = dict(oi_create(asset_store=store).fields["optics"])
        base_optics["focal_length_m"] = 0.050
        optics_f2 = dict(base_optics)
        optics_f2["f_number"] = 2.0
        optics_f8 = dict(base_optics)
        optics_f8["f_number"] = 8.0
        circ_f2_focus_0_5, x_dist_focus_0_5 = optics_coc(optics_f2, 0.5, "unit", "mm", "n samples", 50)
        circ_f8_focus_0_5, _ = optics_coc(optics_f8, 0.5, "unit", "mm", "n samples", 50)
        circ_f2_focus_3, x_dist_focus_3 = optics_coc(optics_f2, 3.0, "unit", "mm", "n samples", 50)
        circ_f8_focus_3, _ = optics_coc(optics_f8, 3.0, "unit", "mm", "n samples", 50)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "object_distances_m": np.array([0.5, 3.0], dtype=float),
                "f_numbers": np.array([2.0, 8.0], dtype=float),
                "focal_length_m": 0.050,
                "x_dist_focus_0_5_m": np.asarray(x_dist_focus_0_5, dtype=float),
                "circ_f2_focus_0_5_mm": np.asarray(circ_f2_focus_0_5, dtype=float),
                "circ_f8_focus_0_5_mm": np.asarray(circ_f8_focus_0_5, dtype=float),
                "x_dist_focus_3_m": np.asarray(x_dist_focus_3, dtype=float),
                "circ_f2_focus_3_mm": np.asarray(circ_f2_focus_3, dtype=float),
                "circ_f8_focus_3_mm": np.asarray(circ_f8_focus_3, dtype=float),
            },
            context={"optics_f2": optics_f2, "optics_f8": optics_f8},
        )

    if case_name == "metrics_cct_from_uv_1d":
        uv = np.array([0.20029948, 0.31055768], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "uv": uv,
                "cct_k": cct_from_uv(uv, asset_store=store),
            },
            context={},
        )

    if case_name == "metrics_delta_e_ab_1976_1d":
        xyz1 = np.array([20.0, 30.0, 15.0], dtype=float)
        xyz2 = np.array([18.0, 27.0, 16.5], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz1": xyz1,
                "xyz2": xyz2,
                "white_point": white_point,
                "delta_e": delta_e_ab(xyz1, xyz2, white_point),
            },
            context={},
        )

    if case_name == "metrics_spd_angle_1d":
        wave = np.array([500.0, 510.0, 520.0], dtype=float)
        spd1 = np.array([1.0, 0.0, 0.0], dtype=float)
        spd2 = np.array([0.0, 1.0, 0.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "angle": metrics_spd(spd1, spd2, metric="angle", wave=wave),
            },
            context={},
        )

    if case_name == "metrics_spd_cielab_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        spd1 = np.linspace(0.5, 1.7, wave.size, dtype=float)
        spd2 = np.linspace(1.6, 0.4, wave.size, dtype=float)
        value, params = metrics_spd(spd1, spd2, metric="cielab", wave=wave, return_params=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "delta_e": value,
                "xyz1": np.asarray(params["xyz1"], dtype=float),
                "xyz2": np.asarray(params["xyz2"], dtype=float),
                "lab1": np.asarray(params["lab1"], dtype=float),
                "lab2": np.asarray(params["lab2"], dtype=float),
                "white_point": np.asarray(params["white_point"], dtype=float),
            },
            context={},
        )

    if case_name == "metrics_spd_mired_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        spd1 = np.asarray(blackbody(wave, 6500.0, kind="energy"), dtype=float)
        spd2 = np.asarray(blackbody(wave, 5000.0, kind="energy"), dtype=float)
        value, params = metrics_spd(spd1, spd2, metric="mired", wave=wave, asset_store=store, return_params=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "mired": value,
                "uv": np.asarray(params["uv"], dtype=float),
                "cct_k": np.asarray(params["cct_k"], dtype=float),
            },
            context={},
        )

    if case_name == "scene_illuminant_change":
        scene = scene_create(asset_store=store)
        wave = scene_get(scene, "wave")
        preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=store)
        no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "preserve_mean": scene_get(preserve, "mean luminance", asset_store=store),
                "no_preserve_mean": scene_get(no_preserve, "mean luminance", asset_store=store),
                "preserve_photons": scene_get(preserve, "photons"),
                "no_preserve_photons": scene_get(no_preserve, "photons"),
            },
            context={
                "scene": scene,
                "preserve_scene": preserve,
                "no_preserve_scene": no_preserve,
            },
        )

    if case_name == "display_create_lcd_example":
        display = display_create("lcdExample.mat", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": display.fields["wave"],
                "spd": display.fields["spd"],
                "gamma": display.fields["gamma"],
            },
            context={"display": display},
        )

    if case_name == "oi_psf_default_small":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi = oi_compute(oi_create("psf"), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_psf550_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psf", None, 550.0)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfxaxis_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psfxaxis", None, 550.0, "um")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "wave": float(udata["wave"]),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfyaxis_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psfyaxis", None, 550.0, "um")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "wave": float(udata["wave"]),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psf_plot_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["f_number"] = 12.0
        oi.fields["optics"] = optics
        this_wave = 600.0
        units = "um"
        n_samp = 100
        psf_data = dict(oi_get(oi, "psf data", this_wave, units, n_samp))
        xy = np.asarray(psf_data["xy"], dtype=float)
        psf = np.asarray(psf_data["psf"], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": xy[:, :, 0].copy(),
                "y": xy[:, :, 1].copy(),
                "psf": psf,
                "airy_disk_radius_um": float(airy_disk(this_wave, float(oi_get(oi, "fnumber")), "units", units)),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfxaxis_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = 550.0
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi_line = dict(oi_get(oi, "optics psf xaxis", this_wave, "um"))
        wvf_line, _ = wvf_plot(wvf, "psfxaxis", "unit", "um", "wave", this_wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": this_wave,
                "oi_samp": np.asarray(oi_line["samp"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
                "wvf_samp": np.asarray(wvf_line["samp"], dtype=float),
                "wvf_data": np.asarray(wvf_line["data"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_psfyaxis_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = 550.0
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi_line = dict(oi_get(oi, "optics psf yaxis", this_wave, "um"))
        wvf_line, _ = wvf_plot(wvf, "psfyaxis", "unit", "um", "wave", this_wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": this_wave,
                "oi_samp": np.asarray(oi_line["samp"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
                "wvf_samp": np.asarray(wvf_line["samp"], dtype=float),
                "wvf_data": np.asarray(wvf_line["data"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_psf550_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_lswavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_otfwavelength_diffraction_small":
        oi = oi_create("diffraction limited", asset_store=store)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_otfwavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_irradiance_hline_diffraction_lineep_small":
        scene = scene_create("line ep", [128, 128], asset_store=store)
        scene = scene_set(scene, "fov", 0.5)
        oi = oi_create()
        oi = oi_compute(oi, scene)
        roi_locs = np.array([80, 80], dtype=int)
        udata, _ = oi_plot(oi, "irradiance hline", roi_locs)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "roi_locs": roi_locs,
                "pos": np.asarray(udata["pos"], dtype=float),
                "wave": np.asarray(udata["wave"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_wvf_otf_compare_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        wvf_otf = np.asarray(wvf_get(wvf, "otf", 550.0), dtype=complex)
        oi_otf = np.asarray(oi_get(oi, "optics otf"), dtype=complex)
        if oi_otf.ndim == 3:
            oi_otf = oi_otf[:, :, 0]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "oi_otf_abs": np.abs(oi_otf),
                "wvf_otf_abs_shifted": np.abs(wvf_otf),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_si_lorentzian_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_otfwavelength_si_lorentzian_small":
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psf550_si_lorentzian_small":
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_si_pillbox_small":
        scene = scene_create("grid lines", [256, 256], 64, "ee", 3, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        patch_size_mm = float(airy_disk(700.0, float(oi_get(oi, "optics fnumber")), "units", "mm"))
        optics = si_synthetic("pillbox", oi, patch_size_mm)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_gaussian_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.ones(wave.size, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_gaussian_ratio_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_gaussian_psf_point_array_small":
        wave = np.arange(450.0, 651.0, 100.0, dtype=float)
        scene = scene_create("point array", 128, 32, asset_store=store)
        scene = scene_interpolate_w(scene, wave, asset_store=store)
        scene = scene_set(scene, "hfov", 1.0)
        scene = scene_set(scene, "name", "psfPointArray")
        oi = oi_create()
        oi = oi_set(oi, "wave", scene_get(scene, "wave"))
        xy_ratio = np.full(wave.size, 3.0, dtype=float)
        wave_spread = wave / float(wave[0])
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene)
        photons = np.asarray(oi.data["photons"], dtype=float)
        photons_normalized = photons / np.maximum(np.max(photons, axis=(0, 1), keepdims=True), 1e-12)
        center_row = photons_normalized.shape[0] // 2
        center_col = photons_normalized.shape[1] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(oi.fields["wave"], dtype=float),
                "scene_wave": np.asarray(scene.fields["wave"], dtype=float),
                "scene_name": scene.name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "center_row_normalized": np.asarray(photons_normalized[center_row, :, :], dtype=float),
                "center_col_normalized": np.asarray(photons_normalized[:, center_col, :], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_psf550_si_gaussian_ratio_small":
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_illuminance_lines_si_gaussian_ratio_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        xy_middle = np.array([int(np.ceil(size[1] / 2.0)), int(np.ceil(size[0] / 2.0))], dtype=int)
        v_data, _ = oi_plot(oi, "illuminance vline", xy_middle)
        h_data, _ = oi_plot(oi, "illuminance hline", xy_middle)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xy_middle": xy_middle,
                "v_pos": np.asarray(v_data["pos"], dtype=float),
                "v_data": np.asarray(v_data["data"], dtype=float),
                "h_pos": np.asarray(h_data["pos"], dtype=float),
                "h_data": np.asarray(h_data["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_custom_file_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("shift invariant")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        samples = np.arange(129, dtype=float) - 64.0
        xx, yy = np.meshgrid(samples, samples, indexing="xy")
        psf = np.zeros((129, 129, wave.size), dtype=float)
        for idx, wavelength in enumerate(wave):
            sigma = 1.2 + 0.01 * ((float(wavelength) - float(wave[0])) / 10.0)
            plane = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
            psf[:, :, idx] = plane / np.sum(plane)
        with tempfile.TemporaryDirectory(prefix="pyisetcam-si-custom-") as tmpdir:
            path = ie_save_si_data_file(psf, wave, np.array([0.25, 0.25], dtype=float), f"{tmpdir}/custom_si_psf.mat")
            optics = si_synthetic("custom", oi, path)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "input_psf_mid_row_550": psf[64, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_custom_otf_flare_small":
        scene = scene_create("point array", 64, 16, asset_store=store)
        scene = scene_set(scene, "hfov", 40.0)
        otf_struct = optics_psf_to_otf(
            store.resolve("data/optics/flare/flare1.png"),
            1.2e-6,
            np.arange(400.0, 701.0, 10.0, dtype=float),
        )
        oi = oi_create("shift invariant")
        oi = oi_set(oi, "optics otfstruct", otf_struct)
        optics = dict(oi.fields["optics"])
        _, width_m, _ = _oi_geometry(optics, scene)
        sample_spacing_m = float(width_m) / max(int(scene.data["photons"].shape[1]), 1)
        pad_rows = int(np.round(scene.data["photons"].shape[0] / 8.0))
        pad_cols = int(np.round(scene.data["photons"].shape[1] / 8.0))
        custom_otf = _shift_invariant_custom_otf(
            (
                int(scene.data["photons"].shape[0] + (2 * pad_rows)),
                int(scene.data["photons"].shape[1] + (2 * pad_cols)),
            ),
            sample_spacing_m,
            np.asarray(scene.fields["wave"], dtype=float),
            optics,
        )
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "fx": np.asarray(otf_struct["fx"], dtype=float),
                "fy": np.asarray(otf_struct["fy"], dtype=float),
                "otf_abs550": np.abs(np.asarray(otf_struct["OTF"], dtype=complex)[:, :, 15]),
                "interp_otf_abs550": np.abs(np.asarray(custom_otf, dtype=complex)[:, :, 15]),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_psf_to_otf_flare_small":
        otf_struct = optics_psf_to_otf(
            store.resolve("data/optics/flare/flare1.png"),
            1.2e-6,
            np.arange(400.0, 701.0, 10.0, dtype=float),
        )
        otf_plane = np.abs(np.fft.fftshift(np.asarray(otf_struct["OTF"], dtype=complex)[:, :, 15]))
        row_index = (otf_plane.shape[0] // 2)
        center_row = int(np.rint(otf_plane.shape[0] / 2.0)) - 1
        center_col = int(np.rint(otf_plane.shape[1] / 2.0)) - 1
        row = otf_plane[row_index, :]
        center = otf_plane[max(0, center_row - 16) : center_row + 17, max(0, center_col - 16) : center_col + 17]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(otf_struct["fx"], dtype=float),
                "fy": np.asarray(otf_struct["fy"], dtype=float),
                "otf_abs550_row": np.asarray(row, dtype=float),
                "otf_abs550_center": np.asarray(center, dtype=float),
            },
            context={"otf_struct": otf_struct},
        )

    if case_name == "oi_ideal_otf_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 3.0)
        oi = oi_compute(oi_create("shift invariant"), scene, crop=True)
        raw_otf = oi_get(oi, "optics OTF")
        oi = oi_set(oi, "optics OTF", np.ones_like(np.asarray(raw_otf), dtype=complex))
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_wvf_defocus_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        wvf = wvf_create(wave=scene_get(scene, "wave"))
        wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])
        oi = oi_compute(wvf, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "defocus": wvf_get(oi_get(oi, "wvf"), "zcoeffs", "defocus"),
                "vertical_astigmatism": wvf_get(oi_get(oi, "wvf"), "zcoeffs", "vertical_astigmatism"),
            },
            context={"scene": scene, "wvf": wvf, "oi": oi},
        )

    if case_name == "oi_wvf_script_defocus_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        wvf = wvf_create(wave=scene_get(scene, "wave"))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_set(wvf, "zcoeffs", 1.5, "defocus")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "defocus_zcoeff": wvf_get(wvf, "zcoeffs", "defocus"),
                "pupil_diameter_mm": wvf_get(wvf, "pupil diameter", "mm"),
                "f_number": oi_get(oi, "fnumber"),
            },
            context={"scene": scene, "wvf": wvf, "oi": oi},
        )

    if case_name == "optics_defocus_small":
        scene = scene_create("disk array", 256, 32, np.array([2, 2], dtype=int), asset_store=store)
        scene = scene_set(scene, "fov", 0.5)

        wvf = wvf_create(wave=scene_get(scene, "wave"))
        oi = oi_create("wvf", wvf)
        oi = oi_compute(oi, scene)

        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        wave_index_550 = int(np.argmin(np.abs(wave - 550.0)))

        def _mean_photons(current_oi: Any) -> float:
            return float(np.mean(np.asarray(oi_get(current_oi, "photons"), dtype=float)))

        def _center_row_norm_550(current_oi: Any) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            center_row = photons[photons.shape[0] // 2, :, wave_index_550]
            return _channel_normalize(center_row)

        base_mean = _mean_photons(oi)
        base_center_row_550 = _center_row_norm_550(oi)

        oi = oi_set(oi, "wvf zcoeffs", 2.5, "defocus")
        oi = oi_compute(oi, scene)
        defocus_center_row_550 = _center_row_norm_550(oi)
        defocus_coeff = float(oi_get(oi, "wvf", "zcoeffs", "defocus"))

        oi = oi_set(oi, "wvf zcoeffs", 1.0, "vertical_astigmatism")
        oi = oi_compute(oi, scene)
        astig_center_row_550 = _center_row_norm_550(oi)
        astig_coeff = float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism"))

        oi = oi_set(oi, "wvf zcoeffs", 0.0, "vertical_astigmatism")
        oi = oi_compute(oi, scene)

        oi = oi_set(oi, "wvf zcoeffs", 0.0, "defocus")
        oi = oi_compute(oi, scene)
        reset_mean = _mean_photons(oi)
        reset_center_row_550 = _center_row_norm_550(oi)

        current_wvf = oi_get(oi, "wvf")
        pupil_diameter_mm = float(wvf_get(current_wvf, "calc pupil diameter", "mm"))
        updated_wvf = wvf_set(current_wvf, "calc pupil diameter", 2.0 * pupil_diameter_mm, "mm")
        oi = oi_set(oi, "optics wvf", updated_wvf)
        oi = oi_compute(oi, scene)
        large_pupil_center_row_550 = _center_row_norm_550(oi)

        restored_wvf = wvf_set(oi_get(oi, "wvf"), "calc pupil diameter", pupil_diameter_mm, "mm")
        oi = oi_set(oi, "optics wvf", restored_wvf)
        oi = oi_compute(oi, scene)
        final_mean = _mean_photons(oi)
        final_center_row_550 = _center_row_norm_550(oi)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "base_center_row_550_norm": base_center_row_550,
                "defocus_center_row_550_norm": defocus_center_row_550,
                "astig_center_row_550_norm": astig_center_row_550,
                "reset_center_row_550_norm": reset_center_row_550,
                "large_pupil_center_row_550_norm": large_pupil_center_row_550,
                "final_center_row_550_norm": final_center_row_550,
                "defocus_coeff": defocus_coeff,
                "vertical_astigmatism_coeff": astig_coeff,
                "final_defocus_coeff": float(oi_get(oi, "wvf", "zcoeffs", "defocus")),
                "final_vertical_astigmatism_coeff": float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")),
                "pupil_diameter_mm": pupil_diameter_mm,
                "doubled_pupil_diameter_mm": 2.0 * pupil_diameter_mm,
                "initial_reset_ratio": base_mean / max(reset_mean, 1e-12),
                "initial_final_ratio": base_mean / max(final_mean, 1e-12),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_defocus_displacement_small":
        base_diopters = np.arange(50.0, 351.0, 100.0, dtype=float)
        delta_diopters = np.arange(1.0, 16.0, dtype=float)
        displacement_curves_m = np.asarray(
            optics_defocus_displacement(base_diopters[:, None], delta_diopters[None, :]),
            dtype=float,
        )

        ratio_base_diopters = np.arange(50.0, 301.0, 50.0, dtype=float)
        ratio_delta_diopters = ratio_base_diopters / 10.0
        ratio_displacement_m = np.asarray(
            optics_defocus_displacement(ratio_base_diopters, ratio_delta_diopters),
            dtype=float,
        )
        displacement_focal_length_ratio = ratio_displacement_m * ratio_base_diopters

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "base_diopters": base_diopters,
                "delta_diopters": delta_diopters,
                "displacement_curves_m": displacement_curves_m,
                "ratio_base_diopters": ratio_base_diopters,
                "ratio_delta_diopters": ratio_delta_diopters,
                "ratio_displacement_m": ratio_displacement_m,
                "displacement_to_focal_length_ratio": displacement_focal_length_ratio,
            },
            context={},
        )

    if case_name == "optics_dof_small":
        f_number = 2.0
        focal_length_m = 0.100
        object_distance_m = 2.0
        coc_diameter_m = 50e-6

        oi = oi_create()
        optics = dict(oi.fields["optics"])
        optics["f_number"] = f_number
        optics["focal_length_m"] = focal_length_m

        dof_formula_m = float(optics_dof(optics, object_distance_m, coc_diameter_m))
        coc_curve_m, x_dist_m = optics_coc(optics, object_distance_m, "nsamples", 200)
        idx1 = int(np.argmin(np.abs(coc_curve_m[:100] - coc_diameter_m)))
        idx2 = int(np.argmin(np.abs(coc_curve_m[100:] - coc_diameter_m))) + 100
        coc_dof_m = float(x_dist_m[idx2] - x_dist_m[idx1])

        object_distances_m = np.arange(0.5, 20.0 + 1e-12, 0.25, dtype=float)
        f_numbers = np.arange(2.0, 12.0 + 1e-12, 0.25, dtype=float)
        dof_surface_m = np.zeros((object_distances_m.size, f_numbers.size), dtype=float)
        optics_sweep = dict(optics)
        for column_index, sweep_f_number in enumerate(f_numbers):
            optics_sweep["f_number"] = float(sweep_f_number)
            dof_surface_m[:, column_index] = np.asarray(
                optics_dof(optics_sweep, object_distances_m, 20e-6),
                dtype=float,
            )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "f_number": f_number,
                "focal_length_m": focal_length_m,
                "object_distance_m": object_distance_m,
                "coc_diameter_m": coc_diameter_m,
                "dof_formula_m": dof_formula_m,
                "coc_xdist_m": np.asarray(x_dist_m, dtype=float),
                "coc_curve_m": np.asarray(coc_curve_m, dtype=float),
                "coc_idx1": idx1,
                "coc_idx2": idx2,
                "coc_dof_m": coc_dof_m,
                "object_distances_m": object_distances_m,
                "f_numbers": f_numbers,
                "sweep_coc_diameter_m": 20e-6,
                "dof_surface_m": dof_surface_m,
            },
            context={},
        )

    if case_name == "optics_defocus_scene_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        scene_path = store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
        scene = scene_from_file(scene_path, "multispectral", None, None, wave, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        max_sf = float(scene_get(scene, "max freq res", "cpd", asset_store=store))
        n_steps = min(int(np.ceil(max_sf)), 70)
        sample_sf = np.linspace(0.0, max_sf, n_steps, dtype=float)
        scene = scene_adjust_illuminant(scene, "D65.mat", asset_store=store)

        base_oi = oi_create()
        base_optics = dict(oi_get(base_oi, "optics"))
        base_optics["model"] = "shiftinvariant"

        def _build_oi(defocus_diopters: np.ndarray) -> tuple[OpticalImage, np.ndarray]:
            otf_rows, sample_sf_mm = optics_defocus_core(base_optics, sample_sf, defocus_diopters)
            defocused_optics = optics_build_2d_otf(base_optics, otf_rows, sample_sf_mm)
            current_oi = oi_set(oi_create(), "optics", defocused_optics)
            current_oi = oi_compute(current_oi, scene)
            return current_oi, np.asarray(sample_sf_mm, dtype=float)

        optics_wave = np.asarray(base_optics.get("transmittance", {}).get("wave", wave), dtype=float).reshape(-1)

        def _center_row_norm_550(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            center_row = photons[photons.shape[0] // 2, :, wave_index]
            return _channel_normalize(center_row)

        def _peak_550(current_oi: OpticalImage) -> float:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            return float(np.max(photons[:, :, wave_index]))

        defocus5 = np.full(optics_wave.shape, 5.0, dtype=float)
        oi_defocus5, sample_sf_mm = _build_oi(defocus5)

        focused = np.zeros(optics_wave.shape, dtype=float)
        oi_focus, _ = _build_oi(focused)

        focal_length_m = float(base_optics.get("focal_length_m", base_optics.get("focalLength", 0.0)))
        lens_power_diopters = 1.0 / max(focal_length_m, 1e-12)

        delta_distance_10_m = 10e-6
        actual_power_10 = 1.0 / max(focal_length_m - delta_distance_10_m, 1e-12)
        defocus_10_diopters = actual_power_10 - lens_power_diopters
        oi_miss10, _ = _build_oi(np.full(optics_wave.shape, defocus_10_diopters, dtype=float))

        delta_distance_40_m = 40e-6
        actual_power_40 = 1.0 / max(focal_length_m - delta_distance_40_m, 1e-12)
        defocus_40_diopters = actual_power_40 - lens_power_diopters
        oi_miss40, _ = _build_oi(np.full(optics_wave.shape, defocus_40_diopters, dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": optics_wave,
                "max_sf_cpd": max_sf,
                "sample_sf_cpd": sample_sf,
                "sample_sf_mm": sample_sf_mm,
                "defocus_5_diopters": 5.0,
                "defocus_10um_diopters": float(defocus_10_diopters),
                "defocus_40um_diopters": float(defocus_40_diopters),
                "focus_center_row_550_norm": _center_row_norm_550(oi_focus),
                "defocus5_center_row_550_norm": _center_row_norm_550(oi_defocus5),
                "miss10_center_row_550_norm": _center_row_norm_550(oi_miss10),
                "miss40_center_row_550_norm": _center_row_norm_550(oi_miss40),
                "focus_peak_550": _peak_550(oi_focus),
                "defocus5_peak_550": _peak_550(oi_defocus5),
                "miss10_peak_550": _peak_550(oi_miss10),
                "miss40_peak_550": _peak_550(oi_miss40),
            },
            context={"scene": scene, "oi_focus": oi_focus, "oi_defocus5": oi_defocus5, "oi_miss10": oi_miss10, "oi_miss40": oi_miss40},
        )

    if case_name == "wvf_astigmatism_small":
        max_um = 20.0
        base_wvf = wvf_create()
        base_wvf = wvf_set(base_wvf, "lcaMethod", "human")
        base_wvf = wvf_compute(base_wvf)

        z4 = np.arange(-0.5, 1.0, 0.5, dtype=float)
        z5 = np.arange(-0.5, 1.0, 0.5, dtype=float)
        z4_grid, z5_grid = np.meshgrid(z4, z5, indexing="xy")
        zvals = np.column_stack((z4_grid.reshape(-1, order="F"), z5_grid.reshape(-1, order="F")))

        x_support_um: np.ndarray | None = None
        row_profiles: list[np.ndarray] = []
        col_profiles: list[np.ndarray] = []
        centers: list[float] = []
        current = base_wvf
        for pair in zvals:
            current = wvf_set(current, "zcoeffs", np.asarray(pair, dtype=float), ["defocus", "vertical_astigmatism"])
            current = wvf_set(current, "lcaMethod", "human")
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
                "psf normalized",
                "unit",
                "um",
                "wave",
                550.0,
                "plot range",
                max_um,
                "window",
                False,
            )
            psf = np.asarray(udata["z"], dtype=float)
            if x_support_um is None:
                x_support_um = np.asarray(udata["x"], dtype=float)
            row_profiles.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
            col_profiles.append(np.asarray(psf[:, psf.shape[1] // 2], dtype=float))
            centers.append(float(psf[psf.shape[0] // 2, psf.shape[1] // 2]))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "zvals": zvals,
                "x": np.asarray(x_support_um, dtype=float),
                "psf_mid_rows": np.vstack(row_profiles),
                "psf_mid_cols": np.vstack(col_profiles),
                "psf_centers": np.asarray(centers, dtype=float),
            },
            context={"wvf": current},
        )

    if case_name == "wvf_zernike_set_small":
        params = {
            "blockSize": 64,
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        astigmatism_values = np.array([-1.0, 0.0, 1.0], dtype=float)
        defocus_microns = 2.0
        psf_support_um: np.ndarray | None = None
        psf_mid_rows: list[np.ndarray] = []
        oi_center_rows_550: list[np.ndarray] = []
        oi_defocus: list[float] = []
        oi_astigmatism: list[float] = []
        oi_peak_photons_550: list[float] = []

        def _normalize_profile(profile: np.ndarray) -> np.ndarray:
            profile_array = np.asarray(profile, dtype=float)
            return profile_array / max(float(np.max(np.abs(profile_array))), 1e-12)

        current_wvf = wvf_create(wave=scene_get(scene, "wave"))
        last_oi = None
        for astigmatism in astigmatism_values:
            current_wvf = wvf_set(
                current_wvf,
                "zcoeffs",
                np.array([defocus_microns, astigmatism], dtype=float),
                ["defocus", "vertical_astigmatism"],
            )
            current_wvf = wvf_compute(current_wvf)
            udata, _ = wvf_plot(
                current_wvf,
                "psf",
                "unit",
                "um",
                "wave",
                550.0,
                "plot range",
                40.0,
                "window",
                False,
            )
            psf = np.asarray(udata["z"], dtype=float)
            if psf_support_um is None:
                psf_support_um = np.asarray(udata["x"], dtype=float)
            psf_mid_rows.append(_normalize_profile(psf[psf.shape[0] // 2, :]))

            oi = oi_compute(current_wvf, scene)
            last_oi = oi
            oi_wvf = oi_get(oi, "wvf")
            oi_defocus.append(float(wvf_get(oi_wvf, "zcoeffs", "defocus")))
            oi_astigmatism.append(float(wvf_get(oi_wvf, "zcoeffs", "vertical_astigmatism")))

            photons = np.asarray(oi_get(oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            center_row = photons[photons.shape[0] // 2, :, wave_index]
            oi_center_rows_550.append(_normalize_profile(center_row))
            oi_peak_photons_550.append(float(np.max(photons[:, :, wave_index])))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
                "astigmatism_values": astigmatism_values,
                "defocus_microns": defocus_microns,
                "psf_support_um": np.asarray(psf_support_um, dtype=float),
                "psf_mid_rows": np.vstack(psf_mid_rows),
                "oi_defocus_coeffs": np.asarray(oi_defocus, dtype=float),
                "oi_astigmatism_coeffs": np.asarray(oi_astigmatism, dtype=float),
                "oi_center_rows_550": np.vstack(oi_center_rows_550),
                "oi_peak_photons_550": np.asarray(oi_peak_photons_550, dtype=float),
            },
            context={"scene": scene, "wvf": current_wvf, "oi": last_oi},
        )

    if case_name == "wvf_wavefronts_small":
        indices = np.arange(1, 17, dtype=int)
        n_values, m_values = wvf_osa_index_to_zernike_nm(indices)
        x_support_mm: np.ndarray | None = None
        row_profiles: list[np.ndarray] = []
        col_profiles: list[np.ndarray] = []
        peak_abs_values: list[float] = []
        current_wvf = None

        for index in indices:
            current_wvf = wvf_create()
            current_wvf = wvf_set(current_wvf, "npixels", 801)
            current_wvf = wvf_set(current_wvf, "measured pupil size", 2.0)
            current_wvf = wvf_set(current_wvf, "calc pupil size", 2.0)
            current_wvf = wvf_set(current_wvf, "zcoeff", 1.0, int(index))
            current_wvf = wvf_compute(current_wvf)
            udata, _ = wvf_plot(
                current_wvf,
                "image wavefront aberrations",
                "unit",
                "mm",
                "wave",
                550.0,
                "plot range",
                1.0,
                "window",
                False,
            )
            wavefront = np.asarray(udata["z"], dtype=float)
            if x_support_mm is None:
                x_support_mm = np.asarray(udata["x"], dtype=float).reshape(-1)
            peak_abs = max(float(np.max(np.abs(wavefront))), 1e-12)
            normalized_wavefront = wavefront / peak_abs
            row_profiles.append(np.asarray(normalized_wavefront[normalized_wavefront.shape[0] // 2, :], dtype=float))
            col_profiles.append(np.asarray(normalized_wavefront[:, normalized_wavefront.shape[1] // 2], dtype=float))
            peak_abs_values.append(peak_abs)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "indices": indices,
                "n": np.asarray(n_values, dtype=int),
                "m": np.asarray(m_values, dtype=int),
                "x": np.asarray(x_support_mm, dtype=float),
                "wavefront_mid_rows_norm": np.vstack(row_profiles),
                "wavefront_mid_cols_norm": np.vstack(col_profiles),
                "wavefront_peak_abs": np.asarray(peak_abs_values, dtype=float),
                "npixels": 801,
                "measured_pupil_mm": 2.0,
                "calc_pupil_mm": 2.0,
            },
            context={"wvf": current_wvf},
        )

    if case_name == "zernike_interpolation_small":
        raw = store.load_mat("data/optics/zernike_doubleGauss.mat")
        data = raw["data"]
        wavelengths = np.asarray(data.wavelengths, dtype=float).reshape(-1)
        image_heights = np.asarray(data.image_heights, dtype=float).reshape(-1)
        zcoeffs = data.zernikeCoefficients

        image_height_indices = np.arange(1, 22, 4, dtype=int)
        this_wave_index = 3
        test_index = 6
        wavelength_nm = float(wavelengths[this_wave_index - 1])
        image_heights_test = image_heights[image_height_indices - 1]
        zernike_coeff_matrix = np.vstack(
            [
                np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(index)}"), dtype=float).reshape(-1)
                for index in image_height_indices
            ]
        )
        nearest_indices = np.asarray(_find_nearest_two(image_height_indices, test_index), dtype=int)
        test_height = float(image_heights[test_index - 1])
        zernike_gt = np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{test_index}"), dtype=float).reshape(-1)
        zernike_interpolated = np.array(
            [np.interp(test_height, image_heights_test, zernike_coeff_matrix[:, column]) for column in range(zernike_coeff_matrix.shape[1])],
            dtype=float,
        )
        validation = zernike_interpolated - zernike_gt

        psf_interpolated, _ = _generate_fringe_psf(zernike_interpolated)
        psf_gt, _ = _generate_fringe_psf(zernike_gt)
        psf_1, _ = _generate_fringe_psf(np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(nearest_indices[0])}"), dtype=float))
        psf_2, _ = _generate_fringe_psf(np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(nearest_indices[1])}"), dtype=float))
        psf_interp_space = (
            psf_1 * (test_height - float(image_heights[int(nearest_indices[0]) - 1])) / float(nearest_indices[1] - nearest_indices[0])
            + psf_2 * (float(nearest_indices[1]) - test_height) / float(nearest_indices[1] - nearest_indices[0])
        )

        middle_row = psf_gt.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "this_wave_index": this_wave_index,
                "wavelength_nm": wavelength_nm,
                "image_height_indices": image_height_indices,
                "image_heights_test": image_heights_test,
                "test_index": test_index,
                "test_height": test_height,
                "nearest_indices": nearest_indices,
                "zernike_gt": zernike_gt,
                "zernike_interpolated": zernike_interpolated,
                "validation": validation,
                "validation_rmse": float(np.sqrt(np.mean(validation**2))),
                "psf_interpolated_mid_row_norm": _channel_normalize(psf_interpolated[middle_row, :]),
                "psf_gt_mid_row_norm": _channel_normalize(psf_gt[middle_row, :]),
                "psf_interp_space_mid_row_norm": _channel_normalize(psf_interp_space[middle_row, :]),
                "psf_interpolated_peak": float(np.max(psf_interpolated)),
                "psf_gt_peak": float(np.max(psf_gt)),
                "psf_interp_space_peak": float(np.max(psf_interp_space)),
            },
            context={},
        )

    if case_name == "wvf_plot_script_sequence_small":
        wave_550 = 550.0
        wave_460 = 460.0

        wvf = wvf_create()
        wvf = wvf_set(wvf, "wave", wave_550)
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)

        udata_550_um, _ = wvf_plot(wvf, "1d psf", "unit", "um", "wave", wave_550, "window", False)
        udata_550_mm, _ = wvf_plot(wvf, "1d psf", "unit", "mm", "wave", wave_550, "window", False)
        udata_550_norm, _ = wvf_plot(wvf, "1d psf normalized", "unit", "mm", "wave", wave_550, "window", False)

        wvf = wvf_set(wvf, "wave", wave_460)
        wvf = wvf_compute(wvf)

        udata_460_angle, _ = wvf_plot(
            wvf,
            "image psf angle",
            "unit",
            "min",
            "wave",
            wave_460,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf_angle = np.asarray(udata_460_angle["z"], dtype=float)
        udata_460_phase, _ = wvf_plot(
            wvf,
            "image pupil phase",
            "unit",
            "mm",
            "wave",
            wave_460,
            "plot range",
            2.0,
            "window",
            False,
        )
        pupil_phase = np.asarray(udata_460_phase["z"], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave_550_nm": wave_550,
                "wave_460_nm": wave_460,
                "line_550_um_x": np.asarray(udata_550_um["x"], dtype=float),
                "line_550_um_y_norm": _channel_normalize(np.asarray(udata_550_um["y"], dtype=float)),
                "line_550_mm_x": np.asarray(udata_550_mm["x"], dtype=float),
                "line_550_mm_y_norm": _channel_normalize(np.asarray(udata_550_mm["y"], dtype=float)),
                "line_550_mm_norm_y": np.asarray(udata_550_norm["y"], dtype=float),
                "psf_angle_460_x": np.asarray(udata_460_angle["x"], dtype=float),
                "psf_angle_460_mid_row_norm": _channel_normalize(psf_angle[psf_angle.shape[0] // 2, :]),
                "psf_angle_460_center": float(psf_angle[psf_angle.shape[0] // 2, psf_angle.shape[1] // 2]),
                "pupil_phase_460_x": np.asarray(udata_460_phase["x"], dtype=float),
                "pupil_phase_460_mid_row": np.asarray(pupil_phase[pupil_phase.shape[0] // 2, :], dtype=float),
                "pupil_phase_460_center": float(pupil_phase[pupil_phase.shape[0] // 2, pupil_phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_diffraction_small":
        flength_mm = 6.0
        flength_m = flength_mm * 1e-3
        f_number = 3.0
        this_wave = 550.0
        canonical_samples = 41

        def _canonical_profile(values: np.ndarray) -> np.ndarray:
            profile = np.asarray(values, dtype=float)
            if profile.ndim == 1:
                support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
                query = np.linspace(-1.0, 1.0, canonical_samples, dtype=float)
                return np.interp(query, support, profile)
            if profile.ndim == 2:
                return np.vstack([_canonical_profile(row) for row in profile])
            raise ValueError("Expected a vector or row-stacked matrix.")

        wvf = wvf_create()
        wvf = wvf_set(wvf, "calc pupil diameter", flength_mm / f_number)
        wvf = wvf_set(wvf, "focal length", flength_m)
        wvf = wvf_compute(wvf)

        _, _ = wvf_plot(
            wvf,
            "psf",
            "unit",
            "um",
            "wave",
            this_wave,
            "plot range",
            10.0,
            "airy disk",
            True,
            "window",
            False,
        )

        oi = wvf_to_oi(wvf)
        oi_psfx_udata, _ = oi_plot(oi, "psfxaxis", None, this_wave, "um")

        pupil_mm = np.linspace(1.5, 8.0, 4, dtype=float)
        pupil_550_airy: list[float] = []
        current = wvf
        for pupil in pupil_mm:
            current = wvf_set(current, "calc pupil diameter", float(pupil))
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
                "image psf",
                "unit",
                "um",
                "wave",
                this_wave,
                "plot range",
                5.0,
                "airy disk",
                True,
                "window",
                False,
            )
            pupil_550_airy.append(float(airy_disk(this_wave, float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        this_wave = 400.0
        current = wvf_set(current, "calc wave", this_wave)
        pupil_400_airy: list[float] = []
        for pupil in pupil_mm:
            current = wvf_set(current, "calc pupil diameter", float(pupil))
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
                "image psf",
                "unit",
                "um",
                "wave",
                this_wave,
                "plot range",
                5.0,
                "airy disk",
                True,
                "window",
                False,
            )
            pupil_400_airy.append(float(airy_disk(this_wave, float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        current = wvf_set(current, "calc pupil diameter", 3.0)
        current = wvf_set(current, "calc wave", 550.0)
        wavelength_list = np.linspace(400.0, 700.0, 4, dtype=float)
        lca_rows: list[np.ndarray] = []
        lca_airy: list[float] = []
        for wavelength in wavelength_list:
            current = wvf_set(current, "calc wave", float(wavelength))
            current = wvf_set(current, "lcaMethod", "human")
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
                "image psf",
                "unit",
                "um",
                "wave",
                float(wavelength),
                "plot range",
                20.0,
                "airy disk",
                True,
                "window",
                False,
            )
            z = np.asarray(udata["z"], dtype=float)
            lca_rows.append(np.asarray(z[z.shape[0] // 2, :], dtype=float))
            lca_airy.append(float(airy_disk(float(wavelength), float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        base_flength_m = 7e-3
        base_fnumber = 4.0
        diffraction_wvf = wvf_create()
        diffraction_wvf = wvf_set(diffraction_wvf, "calc pupil diameter", (base_flength_m * 1e3) / base_fnumber)
        diffraction_wvf = wvf_set(diffraction_wvf, "focal length", base_flength_m)
        diffraction_wvf = wvf_compute(diffraction_wvf)

        focal_length_sweep_m = np.array([base_flength_m / 2.0, base_flength_m, base_flength_m * 2.0], dtype=float)
        focal_length_um_per_degree = focal_length_sweep_m * 1e6 * (2.0 * np.tan(np.deg2rad(0.5)))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "base_fnumber_ratio_oi_wvf": float(oi_get(oi, "optics fnumber")) / max(float(wvf_get(wvf, "fnumber")), 1e-12),
                "base_airy_diameter_um": float(airy_disk(550.0, f_number, "units", "um", "diameter", True)),
                "base_oi_psfx_data": np.asarray(oi_psfx_udata["data"], dtype=float),
                "pupil_mm": pupil_mm,
                "pupil_550_airy_diameter_um": np.asarray(pupil_550_airy, dtype=float),
                "pupil_400_airy_diameter_um": np.asarray(pupil_400_airy, dtype=float),
                "lca_wavelength_nm": wavelength_list,
                "lca_airy_diameter_um": np.asarray(lca_airy, dtype=float),
                "lca_mid_rows": _canonical_profile(np.vstack(lca_rows)),
                "focal_length_sweep_mm": focal_length_sweep_m * 1e3,
                "focal_length_um_per_degree": np.asarray(focal_length_um_per_degree, dtype=float),
            },
            context={"wvf": diffraction_wvf, "oi": oi},
        )

    if case_name == "wvf_spatial_sampling_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        focal_length_m = 7e-3
        f_number = 4.0
        wvf = wvf_set(wvf, "calc pupil diameter", (focal_length_m * 1e3) / f_number, "mm")
        wvf = wvf_set(wvf, "focal length", focal_length_m, "m")
        wvf = wvf_compute(wvf)
        psf_xaxis = dict(wvf_get(wvf, "psf xaxis", "um", this_wave))
        pupil_amplitude = np.asarray(wvf_get(wvf, "pupil function amplitude", this_wave), dtype=float)
        pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", this_wave), dtype=float)
        middle_row = pupil_amplitude.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wvf_get(wvf, "wave"),
                "npixels": wvf_get(wvf, "npixels"),
                "calc_nwave": wvf_get(wvf, "calc nwave"),
                "psf_sample_spacing_arcmin": wvf_get(wvf, "psf sample spacing"),
                "ref_psf_sample_interval_arcmin": wvf_get(wvf, "ref psf sample interval"),
                "um_per_degree": wvf_get(wvf, "um per degree"),
                "pupil_plane_size_mm": wvf_get(wvf, "pupil plane size", "mm", this_wave),
                "pupil_sample_spacing_mm": wvf_get(wvf, "pupil sample spacing", "mm", this_wave),
                "pupil_positions_mm": wvf_get(wvf, "pupil positions", this_wave, "mm"),
                "psf_xaxis_um": np.asarray(psf_xaxis["samp"], dtype=float),
                "psf_xaxis_data": np.asarray(psf_xaxis["data"], dtype=float),
                "pupil_amp_row": pupil_amplitude[middle_row, :],
                "pupil_phase_row": pupil_phase[middle_row, :],
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_spatial_controls_small":
        this_wave = 550.0
        focal_length_m = 7e-3
        focal_length_mm = focal_length_m * 1e3
        f_number = 4.0

        def _build_wvf() -> dict[str, Any]:
            wavefront = wvf_create(wave=np.array([this_wave], dtype=float))
            wavefront = wvf_set(wavefront, "calc pupil diameter", focal_length_mm / f_number, "mm")
            wavefront = wvf_set(wavefront, "focal length", focal_length_m, "m")
            return wavefront

        def _measure(wavefront: dict[str, Any]) -> dict[str, float]:
            computed = wvf_compute(wavefront)
            um_per_degree = float(wvf_get(computed, "um per degree"))
            return {
                "npixels": float(wvf_get(computed, "npixels")),
                "psf_sample_spacing_arcmin": float(wvf_get(computed, "psf sample spacing")),
                "um_per_degree": um_per_degree,
                "pupil_plane_size_mm": float(wvf_get(computed, "pupil plane size", "mm", this_wave)),
                "pupil_sample_spacing_mm": float(wvf_get(computed, "pupil sample spacing", "mm", this_wave)),
                # Match the legacy MATLAB/Octave wvfGet('focal length', 'm')
                # contract used by the parity baseline, which derives
                # focal length from the returned "um per degree" value.
                "focal_length_m": (um_per_degree / (2.0 * np.tan(np.deg2rad(0.5)))) * 1e-6,
            }

        base = _measure(_build_wvf())

        reduced_pixels = _build_wvf()
        reduced_pixels = wvf_set(reduced_pixels, "npixels", round(base["npixels"] / 4.0))
        reduced_pixels = _measure(reduced_pixels)

        enlarged_pupil_plane = _build_wvf()
        enlarged_pupil_plane = wvf_set(enlarged_pupil_plane, "pupil plane size", base["pupil_plane_size_mm"] * 4.0, "mm")
        enlarged_pupil_plane = _measure(enlarged_pupil_plane)

        reduced_pupil_plane = _build_wvf()
        reduced_pupil_plane = wvf_set(reduced_pupil_plane, "pupil plane size", base["pupil_plane_size_mm"] / 4.0, "mm")
        reduced_pupil_plane = _measure(reduced_pupil_plane)

        focal_half = _build_wvf()
        focal_half = wvf_set(focal_half, "focal length", focal_length_m / 2.0, "m")
        focal_half = _measure(focal_half)

        focal_double = _build_wvf()
        focal_double = wvf_set(focal_double, "focal length", focal_length_m * 2.0, "m")
        focal_double = _measure(focal_double)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.array([this_wave], dtype=float),
                "base_npixels": base["npixels"],
                "base_psf_sample_spacing_arcmin": base["psf_sample_spacing_arcmin"],
                "base_um_per_degree": base["um_per_degree"],
                "base_pupil_plane_size_mm": base["pupil_plane_size_mm"],
                "reduced_pixels_npixels": reduced_pixels["npixels"],
                "reduced_pixels_psf_sample_spacing_arcmin": reduced_pixels["psf_sample_spacing_arcmin"],
                "reduced_pixels_um_per_degree": reduced_pixels["um_per_degree"],
                "pupil_plane_x4_psf_sample_spacing_arcmin": enlarged_pupil_plane["psf_sample_spacing_arcmin"],
                "pupil_plane_x4_um_per_degree": enlarged_pupil_plane["um_per_degree"],
                "pupil_plane_x4_size_mm": enlarged_pupil_plane["pupil_plane_size_mm"],
                "pupil_plane_div4_psf_sample_spacing_arcmin": reduced_pupil_plane["psf_sample_spacing_arcmin"],
                "pupil_plane_div4_um_per_degree": reduced_pupil_plane["um_per_degree"],
                "pupil_plane_div4_size_mm": reduced_pupil_plane["pupil_plane_size_mm"],
                "focal_length_half_m": focal_half["focal_length_m"],
                "focal_length_half_psf_sample_spacing_arcmin": focal_half["psf_sample_spacing_arcmin"],
                "focal_length_half_um_per_degree": focal_half["um_per_degree"],
                "focal_length_double_m": focal_double["focal_length_m"],
                "focal_length_double_psf_sample_spacing_arcmin": focal_double["psf_sample_spacing_arcmin"],
                "focal_length_double_um_per_degree": focal_double["um_per_degree"],
            },
            context={},
        )

    if case_name == "wvf_compute_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 101)
        wvf = wvf_pupil_function(wvf)
        wvf = wvf_compute_psf(wvf, "compute pupil func", False)
        wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        pupil_amp = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
        pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", wave), dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wvf_get(wvf, "wave"),
                "npixels": wvf_get(wvf, "npixels"),
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": psf[middle_row, :],
                "pupil_amp_row": pupil_amp[middle_row, :],
                "pupil_phase_row": pupil_phase[middle_row, :],
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_otf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "2d otf", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_otf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "2d otf normalized", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_angle_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf angle", "unit", "deg", "wave", 550.0, "plot range", 10.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d otf angle normalized",
            "unit",
            "deg",
            "wave",
            550.0,
            "plot range",
            10.0,
            "window",
            False,
        )
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf normalized", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_pupil_amp_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image pupil amp", "unit", "mm", "wave", 550.0, "plot range", 2.0, "window", False)
        amp = np.asarray(udata["z"], dtype=float)
        middle_row = amp.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "amp_mid_row": amp[middle_row, :],
                "amp_center": float(amp[middle_row, amp.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_pupil_amplitude_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d pupil amplitude space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            2.0,
            "window",
            False,
        )
        amp = np.asarray(udata["z"], dtype=float)
        middle_row = amp.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "amp_mid_row": amp[middle_row, :],
                "amp_center": float(amp[middle_row, amp.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_pupil_phase_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image pupil phase", "unit", "mm", "wave", 550.0, "plot range", 2.0, "window", False)
        phase = np.asarray(udata["z"], dtype=float)
        middle_row = phase.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "phase_mid_row": phase[middle_row, :],
                "phase_center": float(phase[middle_row, phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_pupil_phase_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d pupil phase space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            2.0,
            "window",
            False,
        )
        phase = np.asarray(udata["z"], dtype=float)
        middle_row = phase.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "phase_mid_row": phase[middle_row, :],
                "phase_center": float(phase[middle_row, phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_wavefront_aberrations_small":
        wvf = wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.12], dtype=float),
        )
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image wavefront aberrations",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            1.5,
            "window",
            False,
        )
        wavefront = np.asarray(udata["z"], dtype=float)
        middle_row = wavefront.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavefront_mid_row": wavefront[middle_row, :],
                "wavefront_center": float(wavefront[middle_row, wavefront.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_wavefront_aberrations_space_small":
        wvf = wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.12], dtype=float),
        )
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d wavefront aberrations space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            1.5,
            "window",
            False,
        )
        wavefront = np.asarray(udata["z"], dtype=float)
        middle_row = wavefront.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavefront_mid_row": wavefront[middle_row, :],
                "wavefront_center": float(wavefront[middle_row, wavefront.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image psf", "unit", "um", "wave", 550.0, "plot range", 20.0, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf normalized",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "psf", "unit", "mm", "wave", 550.0, "plot range", 0.05, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "psf normalized", "unit", "mm", "wave", 550.0, "plot range", 0.05, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_angle_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image psf angle", "unit", "min", "wave", 460.0, "plot range", 1.0, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf angle normalized",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_psf_angle_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d psf angle",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d psf angle normalized",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf space", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf normalized", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_angle_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d psf angle",
            "unit",
            "min",
            "wave",
            550.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d psf angle normalized",
            "unit",
            "min",
            "wave",
            550.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_xaxis_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf xaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psfxaxis_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf xaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_yaxis_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf yaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psfyaxis_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf yaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_psf2zcoeff_error_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "zcoeffs", 0.2, "defocus")
        wvf = wvf_set(wvf, "zcoeffs", 0.0, "vertical_astigmatism")
        wvf = wvf_compute(wvf)
        this_wave_nm = float(wvf_get(wvf, "wave", 1))
        this_wave_um = float(wvf_get(wvf, "wave", "um", 1))
        pupil_size_mm = float(wvf_get(wvf, "pupil size", "mm"))
        z_pupil_diameter_mm = float(wvf_get(wvf, "z pupil diameter"))
        pupil_plane_size_mm = float(wvf_get(wvf, "pupil plane size", "mm", this_wave_nm))
        n_pixels = int(wvf_get(wvf, "spatial samples"))
        psf_target = np.asarray(wvf_get(wvf, "psf", this_wave_nm), dtype=float)
        query_zcoeffs = np.asarray([0.0, 0.0, 0.0, 0.0, 0.15, 0.02], dtype=float)
        error = psf_to_zcoeff_error(
            query_zcoeffs,
            psf_target,
            pupil_size_mm,
            z_pupil_diameter_mm,
            pupil_plane_size_mm,
            this_wave_um,
            n_pixels,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave_um": this_wave_um,
                "pupil_size_mm": pupil_size_mm,
                "z_pupil_diameter_mm": z_pupil_diameter_mm,
                "pupil_plane_size_mm": pupil_plane_size_mm,
                "n_pixels": n_pixels,
                "query_zcoeffs": query_zcoeffs,
                "error": error,
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_aperture_polygon_clean_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 101)
        aperture, params = wvf_aperture(
            wvf,
            "n sides",
            8,
            "dot mean",
            0,
            "dot sd",
            0,
            "line mean",
            0,
            "line sd",
            0,
            "image rotate",
            0,
        )
        middle_row = aperture.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "image": np.asarray(aperture, dtype=float),
                "mid_row": np.asarray(aperture[middle_row, :], dtype=float),
                "image_sum": float(np.sum(aperture)),
                "nsides": int(params["nsides"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_compute_aperture_polygon_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 101)
        aperture, params = wvf_aperture(
            wvf,
            "n sides",
            8,
            "dot mean",
            0,
            "dot sd",
            0,
            "line mean",
            0,
            "line sd",
            0,
            "image rotate",
            0,
        )
        wvf = wvf_compute(wvf, "aperture", aperture)
        wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        pupil_amp = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": np.asarray(psf[middle_row, :], dtype=float),
                "pupil_amp_row": np.asarray(pupil_amp[middle_row, :], dtype=float),
                "nsides": int(params["nsides"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "oi_lswavelength_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_lswavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_diffraction_limited_default":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_cos4th_small":
        scene = scene_create("uniform d65", 512, asset_store=store)
        scene = scene_set(scene, "fov", 80)

        oi = oi_create("shift invariant", asset_store=store)
        focal_length_default_m = float(oi_get(oi, "optics focal length"))
        oi = oi_compute(oi, scene)
        size_default = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        support_default = dict(oi_get(oi, "spatial support linear", "um"))
        illuminance_default = np.asarray(oi_get(oi, "illuminance"), dtype=float)
        mean_illuminance_default = float(np.mean(illuminance_default))
        center_row = int(np.rint(size_default[1] / 2.0))

        oi = oi_set(oi, "optics focal length", 4.0 * focal_length_default_m)
        focal_length_long_m = float(oi_get(oi, "optics focal length"))
        oi = oi_compute(oi, scene)
        size_long = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        support_long = dict(oi_get(oi, "spatial support linear", "um"))
        illuminance_long = np.asarray(oi_get(oi, "illuminance"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "focal_length_default_m": focal_length_default_m,
                "focal_length_long_m": focal_length_long_m,
                "size_default": size_default,
                "size_long": size_long,
                "center_row": center_row,
                "edge_row": 20,
                "pos_default_um": np.asarray(support_default["x"], dtype=float),
                "center_line_default_lux": np.asarray(illuminance_default[center_row - 1, :], dtype=float),
                "mean_illuminance_default_lux": mean_illuminance_default,
                "pos_long_um": np.asarray(support_long["x"], dtype=float),
                "center_line_long_lux": np.asarray(illuminance_long[center_row - 1, :], dtype=float),
                "edge_line_long_lux": np.asarray(illuminance_long[20 - 1, :], dtype=float),
                "mean_illuminance_long_lux": float(np.mean(illuminance_long)),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_pad_crop_small":
        scene = scene_create("sweep frequency", asset_store=store)
        oi = oi_compute(oi_create(), scene)
        padded_size = np.asarray(oi_get(oi, "size"), dtype=float).reshape(-1)
        original_size = padded_size / 1.25
        offset = (padded_size - original_size) / 2.0
        rect = np.array([offset[1] + 1.0, offset[0] + 1.0, original_size[1] - 1.0, original_size[0] - 1.0], dtype=float)
        oi_cropped = oi_crop(oi, rect)

        sensor_scene_fov = sensor_create(asset_store=store)
        sensor_scene_fov = sensor_set(sensor_scene_fov, "noise flag", 0)
        sensor_scene_fov = sensor_set(sensor_scene_fov, "fov", float(scene_get(scene, "fov")), oi)
        sensor_from_padded = sensor_compute(sensor_scene_fov, oi, seed=0)
        sensor_from_cropped = sensor_compute(sensor_scene_fov, oi_cropped, seed=0)
        padded_volts = np.asarray(sensor_get(sensor_from_padded, "volts"), dtype=float)
        cropped_volts = np.asarray(sensor_get(sensor_from_cropped, "volts"), dtype=float)
        row_index = padded_volts.shape[0] // 2
        sensor_support = sensor_get(sensor_scene_fov, "spatial support", "um")
        normalized_mae = float(np.mean(np.abs(padded_volts - cropped_volts))) / max(float(np.mean(np.abs(cropped_volts))), 1e-12)

        sensor_padded = sensor_set_size_to_fov(sensor_scene_fov.clone(), float(oi_get(oi, "fov")), oi)
        sensor_padded = sensor_compute(sensor_padded, oi, seed=0)
        padded_support = sensor_get(sensor_padded, "spatial support", "um")
        padded_sensor_volts = np.asarray(sensor_get(sensor_padded, "volts"), dtype=float)
        padded_row_index = padded_sensor_volts.shape[0] // 2

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_padded_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "crop_rect": np.rint(rect).astype(int),
                "oi_cropped_size": np.asarray(oi_get(oi_cropped, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "oi_padded_fov_deg": float(oi_get(oi, "fov")),
                "oi_cropped_fov_deg": float(oi_get(oi_cropped, "fov")),
                "sensor_scene_fov_size": np.asarray(sensor_get(sensor_scene_fov, "size"), dtype=int),
                "sensor_scene_fov_pos_um": np.asarray(sensor_support["x"], dtype=float),
                "sensor_scene_fov_padded_row": np.asarray(padded_volts[row_index, :], dtype=float),
                "sensor_scene_fov_cropped_row": np.asarray(cropped_volts[row_index, :], dtype=float),
                "sensor_scene_fov_normalized_mae": normalized_mae,
                "sensor_padded_size": np.asarray(sensor_get(sensor_padded, "size"), dtype=int),
                "sensor_padded_pos_um": np.asarray(padded_support["x"], dtype=float),
                "sensor_padded_row": np.asarray(padded_sensor_volts[padded_row_index, :], dtype=float),
            },
            context={"scene": scene, "oi": oi, "oi_cropped": oi_cropped},
        )

    if case_name == "optics_microlens_small":
        oi = oi_create(asset_store=store)
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "fov", 30.0, oi)

        microlens = mlens_create(asset_store=store)
        chief_ray_angle_default_deg = float(mlens_get(microlens, "chief ray angle"))
        microlens = mlens_set(microlens, "chief ray angle", 10.0)
        chief_ray_angle_set_deg = float(mlens_get(microlens, "chief ray angle"))

        radiance_microlens = mlens_create(asset_store=store)
        radiance_microlens = ml_radiance(radiance_microlens, asset_store=store)
        source_irradiance = np.asarray(mlens_get(radiance_microlens, "source irradiance"), dtype=float)
        pixel_irradiance = np.asarray(mlens_get(radiance_microlens, "pixel irradiance"), dtype=float)
        x_coordinate = np.asarray(mlens_get(radiance_microlens, "x coordinate"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "name": mlens_get(microlens, "name"),
                "type": mlens_get(microlens, "type"),
                "source_fnumber": float(mlens_get(microlens, "source fnumber")),
                "source_diameter_m": float(mlens_get(microlens, "source diameter", "meters")),
                "source_diameter_um": float(mlens_get(microlens, "source diameter", "microns")),
                "ml_fnumber": float(mlens_get(microlens, "ml fnumber")),
                "ml_diameter_m": float(mlens_get(microlens, "ml diameter", "meters")),
                "ml_diameter_um": float(mlens_get(microlens, "ml diameter", "microns")),
                "chief_ray_angle_default_deg": chief_ray_angle_default_deg,
                "chief_ray_angle_set_deg": chief_ray_angle_set_deg,
                "sensor_fov_deg": 30.0,
                "x_coordinate_um": x_coordinate,
                "source_center_row": np.asarray(source_irradiance[source_irradiance.shape[0] // 2, :], dtype=float),
                "pixel_center_row": np.asarray(pixel_irradiance[pixel_irradiance.shape[0] // 2, :], dtype=float),
                "source_irradiance_stats": _stats_vector(source_irradiance),
                "pixel_irradiance_stats": _stats_vector(pixel_irradiance),
                "etendue": float(mlens_get(radiance_microlens, "etendue")),
            },
            context={"oi": oi, "sensor": sensor, "microlens": radiance_microlens},
        )

    if case_name == "oi_wvf_small_scene":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi_seed = oi_create("wvf")
        optics = dict(oi_seed.fields["optics"])
        scene_photons = np.asarray(scene.data["photons"], dtype=float)
        wave = np.asarray(scene.fields["wave"], dtype=float)
        _, width_m, _ = _oi_geometry(optics, scene)
        sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
        pre_psf_photons = _radiance_to_irradiance(scene_photons, optics, scene)
        if str(optics.get("offaxis_method", "")).replace(" ", "").lower() == "cos4th":
            pre_psf_photons = pre_psf_photons * _cos4th_factor(
                pre_psf_photons.shape[0],
                pre_psf_photons.shape[1],
                optics,
                scene,
            )[:, :, None]
        pad_pixels = (
            int(np.round(scene_photons.shape[0] / 8.0)),
            int(np.round(scene_photons.shape[1] / 8.0)),
        )
        pre_psf_photons, _, _ = _pad_scene(pre_psf_photons, pad_pixels, "zero")
        psf_stack = _wvf_psf_stack(pre_psf_photons.shape[:2], sample_spacing_m, wave, optics)
        oi = oi_compute(oi_seed, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "pre_psf_photons": pre_psf_photons,
                "psf_stack": psf_stack,
                "photons": oi.data["photons"],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_bayer_noiseless":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "volts": sensor.data["volts"],
                "integration_time": sensor.fields["integration_time"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_monochrome_noise_stats":
        scene = scene_create("uniform d65", 32, asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={"case_name": case_name, **_stats(sensor.data["volts"])},
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_imx363_crop_small":
        sensor = sensor_create("IMX363", None, "row col", [12, 16], asset_store=store)
        sensor = sensor_set(sensor, "pattern", np.array([[2, 1], [3, 2]], dtype=int))
        sensor = sensor_set(sensor, "wave", np.arange(400.0, 701.0, 10.0, dtype=float))
        dv = np.arange(12 * 16, dtype=float).reshape((12, 16), order="F")
        sensor = sensor_set(sensor, "digital values", dv)
        cropped = sensor_crop(sensor, np.array([2, 3, 7, 5], dtype=float))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "name": str(cropped.name),
                "size": np.asarray(sensor_get(cropped, "size"), dtype=int),
                "metadata_crop": np.asarray(sensor_get(cropped, "metadata crop"), dtype=int),
                "pattern": np.asarray(sensor_get(cropped, "pattern"), dtype=int),
                "digital_values": np.asarray(sensor_get(cropped, "digital values"), dtype=float),
            },
            context={"sensor": cropped},
        )

    if case_name == "sensor_plot_line_volts_space_small":
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "rows", 2)
        sensor = sensor_set(sensor, "cols", 4)
        sensor = sensor_set(
            sensor,
            "volts",
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=float,
            ),
        )
        _, udata = sensor_plot_line(sensor, "h", "volts", "space", np.array([1, 2], dtype=int))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pixPos": np.asarray(udata["pixPos"], dtype=float),
                "pixData": np.asarray(udata["pixData"], dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_signal_current_uniform_small":
        scene = scene_create("uniform ee", 64, asset_store=store)
        scene = scene_set(scene, "fov", 8.0)
        scene = scene_set(scene, "distance", 1.2)
        scene = scene_set(scene, "name", "uniform ee")
        scene = scene_adjust_luminance(scene, 1.0, asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_set(sensor, "exp time", 1.0)
        current = np.asarray(signal_current(oi, sensor), dtype=float)
        start = (current.shape[0] - 40) // 2
        stop = start + 40
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "current_center": current[start:stop, start:stop],
                "mean_current": float(np.mean(current)),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_split_pixel_ovt_saturated_small":
        scene = scene_create("uniform ee", np.array([32, 48], dtype=int), asset_store=store)
        scene = scene_set(scene, "fov", 8.0)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float).copy()
        levels = np.array([1.0, 10.0, 100.0, 1000.0], dtype=float)
        band_width = photons.shape[1] // levels.size
        for index, level in enumerate(levels):
            start = index * band_width
            stop = photons.shape[1] if index == (levels.size - 1) else (index + 1) * band_width
            photons[:, start:stop, :] *= level
        scene.data["photons"] = photons
        oi = oi_create("wvf", asset_store=store)
        oi = oi_compute(oi, scene, "crop", True)

        sensor_array = sensor_create_array(
            "array type",
            "ovt",
            "exp time",
            0.1,
            "size",
            np.array([32, 48], dtype=int),
            "noise flag",
            0,
            asset_store=store,
        )
        combined, captures = sensor_compute_array(sensor_array, oi, "method", "saturated")
        saturated = np.asarray(combined.metadata["saturated"], dtype=bool)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "combined_volts": np.asarray(sensor_get(combined, "volts"), dtype=float),
                "sensor_max_volts": np.array([float(np.max(np.asarray(sensor_get(sensor, "volts"), dtype=float))) for sensor in captures], dtype=float),
                "saturated_counts": np.sum(saturated, axis=(0, 1), dtype=int),
                "sensor_names": np.asarray([str(sensor_get(sensor, "name")) for sensor in captures], dtype=object),
            },
            context={"scene": scene, "oi": oi, "sensor": combined},
        )

    if case_name == "sensor_stacked_pixels_foveon_small":
        scene = scene_create("macbeth d65", 32, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "hfov", 8.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_set(oi, "optics focal length", 3e-3)
        oi = oi_compute(oi, scene)

        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        foveon_filters = np.asarray(ie_read_spectra("Foveon", wave, asset_store=store), dtype=float)

        monochrome_array: list[Any] = []
        for index in range(foveon_filters.shape[1]):
            sensor = sensor_create("monochrome", asset_store=store)
            sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
            sensor = sensor_set(sensor, "exp time", 0.1)
            sensor = sensor_set(sensor, "filter spectra", foveon_filters[:, index])
            sensor = sensor_set(sensor, "name", f"Channel-{index + 1}")
            sensor = sensor_set_size_to_fov(sensor, scene_get(scene, "fov"), oi)
            sensor = sensor_set(sensor, "wave", wave)
            monochrome_array.append(sensor)
        computed_planes = sensor_compute(monochrome_array, oi)
        stacked_volts = np.stack([np.asarray(sensor_get(sensor, "volts"), dtype=float) for sensor in computed_planes], axis=2)

        sensor_foveon = sensor_create(asset_store=store)
        sensor_foveon = sensor_set(sensor_foveon, "name", "foveon")
        sensor_foveon = sensor_set(sensor_foveon, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor_foveon = sensor_set(sensor_foveon, "autoexp", 1)
        sensor_foveon = sensor_set_size_to_fov(sensor_foveon, scene_get(scene, "fov"), oi)
        sensor_foveon = sensor_set(sensor_foveon, "wave", wave)
        sensor_foveon = sensor_set(sensor_foveon, "filter spectra", foveon_filters)
        sensor_foveon = sensor_set(sensor_foveon, "pattern", np.array([[2]], dtype=int))
        sensor_foveon = sensor_set(sensor_foveon, "volts", stacked_volts)

        ip_foveon = ip_compute(ip_create(asset_store=store), sensor_foveon, asset_store=store)
        line_row = min(120, int(np.asarray(sensor_get(sensor_foveon, "size"), dtype=int)[0]))
        foveon_line, _ = ip_plot(ip_foveon, "horizontal line", np.array([1, line_row], dtype=int))

        bayer_filters = np.asarray(ie_read_spectra("NikonD1", wave, asset_store=store), dtype=float)
        sensor_bayer = sensor_create(asset_store=store)
        sensor_bayer = sensor_set(sensor_bayer, "filter spectra", bayer_filters)
        sensor_bayer = sensor_set(sensor_bayer, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor_bayer = sensor_set(sensor_bayer, "autoexp", 1)
        sensor_bayer = sensor_set_size_to_fov(sensor_bayer, scene_get(scene, "fov"), oi)
        sensor_bayer = sensor_compute(sensor_bayer, oi)
        ip_bayer = ip_compute(ip_create(asset_store=store), sensor_bayer, asset_store=store)
        bayer_line, _ = ip_plot(ip_bayer, "horizontal line", np.array([1, line_row], dtype=int))
        bayer_line_values = np.asarray(bayer_line["values"], dtype=float)
        row_start = (stacked_volts.shape[0] - 24) // 2
        col_start = (stacked_volts.shape[1] - 24) // 2
        row_stop = row_start + 24
        col_stop = col_start + 24
        stacked_patch = stacked_volts[row_start:row_stop, col_start:col_stop, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "stacked_center_patch_mean": np.mean(stacked_patch, axis=(0, 1), dtype=float),
                "stacked_center_patch_std": np.std(stacked_patch, axis=(0, 1), dtype=float),
                "stacked_center_patch_p90": np.percentile(stacked_patch, 90.0, axis=(0, 1)),
                "stacked_mean_volts": np.mean(stacked_volts, axis=(0, 1), dtype=float),
                "stacked_std_volts": np.std(stacked_volts, axis=(0, 1), dtype=float),
                "line_row": float(line_row),
                "bayer_line_mean": np.mean(bayer_line_values, axis=0, dtype=float),
                "bayer_line_std": np.std(bayer_line_values, axis=0, dtype=float),
                "bayer_line_p90": np.percentile(bayer_line_values, 90.0, axis=0),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_foveon, "sensor_bayer": sensor_bayer},
        )

    if case_name == "sensor_microlens_etendue_small":
        oi = oi_create(asset_store=store)
        sensor = sensor_create(asset_store=store)
        microlens = mlens_create(sensor, oi, asset_store=store)
        sensor = sensor_set(sensor, "microlens", microlens)
        sensor = sensor_set_size_to_fov(sensor, 4.0, oi)

        no_microlens = ml_analyze_array_etendue(sensor.clone(), "no microlens", asset_store=store)
        centered = ml_analyze_array_etendue(sensor.clone(), "centered", asset_store=store)
        optimal = ml_analyze_array_etendue(sensor.clone(), "optimal", asset_store=store)

        cra_deg = np.asarray(sensor_get(optimal, "cra degrees"), dtype=float)
        ray_angles_deg = np.linspace(0.0, float(np.max(cra_deg)), 10, dtype=float)
        optimal_microlens = sensor_get(optimal, "microlens")
        optimal_offset_curve_um = np.array(
            [
                float(
                    mlens_get(
                        mlens_set(optimal_microlens, "chief ray angle", float(ray_angle_deg)),
                        "optimal offset",
                        optimal,
                        "microns",
                    )
                )
                for ray_angle_deg in ray_angles_deg
            ],
            dtype=float,
        )

        ml_fnumber = float(mlens_get(optimal_microlens, "ml fnumber"))
        half_fnumber_microlens = mlens_set(optimal_microlens, "ml fnumber", 0.5 * ml_fnumber)
        source_f4_microlens = mlens_set(optimal_microlens, "source fnumber", 4.0)
        source_f16_microlens = mlens_set(optimal_microlens, "source fnumber", 16.0)

        display_microlens = mlens_set(mlens_create(asset_store=store), "ml fnumber", 8.0)
        radiance_midlines: list[np.ndarray] = []
        for chief_ray_angle_deg in (-10.0, 0.0, 10.0):
            display_microlens = mlens_set(display_microlens, "chief ray angle", chief_ray_angle_deg)
            display_microlens = ml_radiance(display_microlens, sensor_create(asset_store=store), asset_store=store)
            irradiance = np.asarray(mlens_get(display_microlens, "pixel irradiance"), dtype=float)
            radiance_midlines.append(np.asarray(irradiance[irradiance.shape[0] // 2, :], dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "no_microlens_etendue": np.asarray(sensor_get(no_microlens, "etendue"), dtype=float),
                "centered_etendue": np.asarray(sensor_get(centered, "etendue"), dtype=float),
                "optimal_etendue": np.asarray(sensor_get(optimal, "etendue"), dtype=float),
                "ray_angles_deg": ray_angles_deg,
                "optimal_offset_curve_um": optimal_offset_curve_um,
                "optimal_offsets_default_um": np.asarray(mlens_get(optimal_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_half_fnumber_um": np.asarray(mlens_get(half_fnumber_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_source_f4_um": np.asarray(mlens_get(source_f4_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_source_f16_um": np.asarray(mlens_get(source_f16_microlens, "optimal offsets", optimal), dtype=float),
                "radiance_midline_neg10": radiance_midlines[0],
                "radiance_midline_0": radiance_midlines[1],
                "radiance_midline_10": radiance_midlines[2],
            },
            context={"oi": oi, "sensor": optimal},
        )

    if case_name == "sensor_comparison_small":
        patch_size = 24
        scene_c = scene_create("macbeth d65", patch_size, asset_store=store)
        macbeth_size = np.asarray(scene_get(scene_c, "size"), dtype=int)
        scene_c = scene_set(
            scene_c,
            "resize",
            np.rint(np.array([macbeth_size[0], macbeth_size[1] / 2.0], dtype=float)).astype(int),
        )
        scene_s = scene_create("sweep frequency", int(macbeth_size[0]), float(macbeth_size[0]) / 16.0, asset_store=store)
        scene = scene_combine(scene_c, scene_s, "direction", "horizontal")
        scene = scene_set(scene, "fov", 20.0)
        scene_vfov = float(scene_get(scene, "vfov"))

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 1.2)
        oi = oi_compute(oi, scene)

        sensor_names = ("imx363", "mt9v024", "cyym")
        ip_names = ("imx363", "mt9v024")
        small_sensor_sizes = np.zeros((len(sensor_names), 2), dtype=int)
        small_sensor_mean_volts = np.zeros(len(sensor_names), dtype=float)
        small_sensor_p90_volts = np.zeros(len(sensor_names), dtype=float)
        large_sensor_sizes = np.zeros((len(sensor_names), 2), dtype=int)
        large_sensor_mean_volts = np.zeros(len(sensor_names), dtype=float)
        large_sensor_p90_volts = np.zeros(len(sensor_names), dtype=float)
        small_ip_sizes = np.zeros((len(ip_names), 3), dtype=int)
        large_ip_sizes = np.zeros((len(ip_names), 3), dtype=int)

        def _compute_sensor(sensor_type: str, pixel_size_m: float) -> tuple[Any, np.ndarray]:
            if sensor_type == "mt9v024":
                sensor = sensor_create(sensor_type, None, "rccc", asset_store=store)
            else:
                sensor = sensor_create(sensor_type, asset_store=store)
            sensor = sensor_set(sensor, "pixel size", pixel_size_m)
            sensor = sensor_set(sensor, "hfov", 20.0, oi)
            sensor = sensor_set(sensor, "vfov", scene_vfov)
            sensor = sensor_set(sensor, "auto exposure", True)
            sensor = sensor_compute(sensor, oi)
            return sensor, np.asarray(sensor_get(sensor, "volts"), dtype=float)

        def _compute_ip(sensor_type: str, sensor: Any) -> np.ndarray:
            if sensor_type == "imx363":
                ip = ip_create("imx363 RGB", sensor, asset_store=store)
            elif sensor_type == "mt9v024":
                ip = ip_create("mt9v024 RCCC", sensor, asset_store=store)
                ip = ip_set(ip, "demosaic method", "analog rccc")
            else:
                raise ValueError(f"Unsupported IP comparison sensor: {sensor_type}")
            ip = ip_compute(ip, sensor, asset_store=store)
            return np.asarray(ip_get(ip, "result"), dtype=float)

        for sensor_index, sensor_type in enumerate(sensor_names):
            sensor_small, volts_small = _compute_sensor(sensor_type, 1.5e-6)
            small_sensor_sizes[sensor_index, :] = np.asarray(sensor_get(sensor_small, "size"), dtype=int)
            small_sensor_mean_volts[sensor_index] = float(np.mean(volts_small))
            small_sensor_p90_volts[sensor_index] = float(np.percentile(volts_small, 90.0))

            sensor_large, volts_large = _compute_sensor(sensor_type, 6.0e-6)
            large_sensor_sizes[sensor_index, :] = np.asarray(sensor_get(sensor_large, "size"), dtype=int)
            large_sensor_mean_volts[sensor_index] = float(np.mean(volts_large))
            large_sensor_p90_volts[sensor_index] = float(np.percentile(volts_large, 90.0))

            if sensor_type in ip_names:
                ip_index = ip_names.index(sensor_type)
                result_small = _compute_ip(sensor_type, sensor_small)
                result_large = _compute_ip(sensor_type, sensor_large)
                small_ip_sizes[ip_index, :] = np.asarray(result_small.shape, dtype=int)
                large_ip_sizes[ip_index, :] = np.asarray(result_large.shape, dtype=int)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_vfov": scene_vfov,
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "small_sensor_sizes": small_sensor_sizes,
                "nonimx_small_sensor_mean_volts": small_sensor_mean_volts[1:],
                "nonimx_small_sensor_p90_volts": small_sensor_p90_volts[1:],
                "large_sensor_sizes": large_sensor_sizes,
                "nonimx_large_sensor_mean_volts": large_sensor_mean_volts[1:],
                "nonimx_large_sensor_p90_volts": large_sensor_p90_volts[1:],
                "imx363_mean_ratio_large_small": float(large_sensor_mean_volts[0] / max(small_sensor_mean_volts[0], 1e-12)),
                "imx363_p90_ratio_large_small": float(large_sensor_p90_volts[0] / max(small_sensor_p90_volts[0], 1e-12)),
                "small_ip_sizes": small_ip_sizes,
                "large_ip_sizes": large_ip_sizes,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_noise_samples_small":
        scene = scene_create("slanted bar", 128, asset_store=store)
        scene = scene_set(scene, "fov", 4.0)
        oi = oi_compute(oi_create(asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "exp time", 0.05)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor_nf = sensor_compute(sensor, oi, seed=0)
        volts_nf = np.asarray(sensor_get(sensor_nf, "volts"), dtype=float)

        n_samp = 64
        volt_images = np.asarray(sensor_compute_samples(sensor_nf, n_samp, 2, seed=7), dtype=float)
        noise_images = volt_images - volts_nf[:, :, np.newaxis]
        std_image = np.std(volt_images, axis=2, ddof=1)
        mean_image = np.mean(volt_images, axis=2, dtype=float)
        pair_diff = volt_images[:, :, 0] - volt_images[:, :, 1]
        mean_residual = mean_image - volts_nf

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sample_shape": np.asarray(volt_images.shape, dtype=int),
                "noise_free_mean": float(np.mean(volts_nf)),
                "noise_std_image_stats": np.array(
                    [
                        float(np.mean(std_image)),
                        float(np.percentile(std_image, 10.0)),
                        float(np.percentile(std_image, 50.0)),
                        float(np.percentile(std_image, 90.0)),
                    ],
                    dtype=float,
                ),
                "noise_distribution_stats": np.array(
                    [
                        float(np.std(noise_images, ddof=1)),
                        float(np.percentile(noise_images, 5.0)),
                        float(np.percentile(noise_images, 50.0)),
                        float(np.percentile(noise_images, 95.0)),
                    ],
                    dtype=float,
                ),
                "mean_residual_stats": np.array(
                    [
                        float(np.mean(np.abs(mean_residual))),
                        float(np.percentile(np.abs(mean_residual), 95.0)),
                    ],
                    dtype=float,
                ),
                "pair_diff_stats": np.array(
                    [
                        float(np.std(pair_diff, ddof=1)),
                        float(np.percentile(pair_diff, 5.0)),
                        float(np.percentile(pair_diff, 50.0)),
                        float(np.percentile(pair_diff, 95.0)),
                    ],
                    dtype=float,
                ),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_nf},
        )

    if case_name == "sensor_mcc_small":
        mosaic_path = store.resolve("data/sensor/mccGBRGsensor.tif")
        mosaic = np.asarray(iio.imread(mosaic_path), dtype=float)

        sensor = sensor_create("bayer (gbrg)", asset_store=store)
        sensor = sensor_set(sensor, "name", "Sensor demo")

        minimum = float(np.min(mosaic))
        maximum = float(np.max(mosaic))
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        volts = ((mosaic - minimum) / max(maximum - minimum, 1e-12)) * voltage_swing

        sensor = sensor_set(sensor, "size", np.asarray(volts.shape[:2], dtype=int))
        sensor = sensor_set(sensor, "volts", volts)
        corner_points = np.array([[15, 584], [782, 584], [784, 26], [23, 19]], dtype=float)
        sensor = sensor_set(sensor, "chart corner points", corner_points)
        estimated_ccm, _ = sensor_ccm(sensor, None, None, True, asset_store=store)

        ip_uncorrected = ip_create(asset_store=store)
        ip_uncorrected = ip_set(ip_uncorrected, "name", "No Correction")
        ip_uncorrected = ip_set(ip_uncorrected, "scaledisplay", 1)
        ip_uncorrected = ip_compute(ip_uncorrected, sensor, asset_store=store)
        uncorrected = np.asarray(ip_get(ip_uncorrected, "result"), dtype=float)
        uncorrected_flat = uncorrected.reshape(-1, uncorrected.shape[2])

        fixed_matrix = np.array(
            [
                [0.9205, -0.1402, -0.1289],
                [-0.0148, 0.8763, -0.0132],
                [-0.2516, -0.1567, 0.6987],
            ],
            dtype=float,
        )
        ip_corrected = ip_create(asset_store=store)
        ip_corrected = ip_set(ip_corrected, "name", "CCM Correction")
        ip_corrected = ip_set(ip_corrected, "scaledisplay", 1)
        ip_corrected = ip_set(ip_corrected, "conversion transform sensor", fixed_matrix)
        ip_corrected = ip_set(ip_corrected, "correction transform illuminant", np.eye(3, dtype=float))
        ip_corrected = ip_set(ip_corrected, "ics2Display Transform", np.eye(3, dtype=float))
        ip_corrected = ip_set(ip_corrected, "conversion method sensor", "current matrix")
        ip_corrected = ip_compute(ip_corrected, sensor, asset_store=store)
        corrected = np.asarray(ip_get(ip_corrected, "result"), dtype=float)
        corrected_flat = corrected.reshape(-1, corrected.shape[2])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "mosaic_size": np.asarray(volts.shape[:2], dtype=int),
                "volts_stats": _stats_vector(volts),
                "estimated_ccm": np.asarray(estimated_ccm, dtype=float),
                "uncorrected_mean_rgb_norm": _channel_normalize(np.mean(uncorrected_flat, axis=0, dtype=float)),
                "uncorrected_p95_rgb_norm": _channel_normalize(np.percentile(uncorrected_flat, 95.0, axis=0)),
                "corrected_mean_rgb_norm": _channel_normalize(np.mean(corrected_flat, axis=0, dtype=float)),
                "corrected_p95_rgb_norm": _channel_normalize(np.percentile(corrected_flat, 95.0, axis=0)),
            },
            context={"sensor": sensor, "ip_uncorrected": ip_uncorrected, "ip_corrected": ip_corrected},
        )

    if case_name == "sensor_rolling_shutter_small":
        scene = scene_create("star pattern", 48, "ee", 4, asset_store=store)
        scene = scene_set(scene, "fov", 3.0)
        oi = oi_create(asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor = sensor_set(sensor, "fov", float(scene_get(scene, "fov")) / 2.0, oi)
        sensor = sensor_set(sensor, "exp time", 4.0e-5)
        sensor = sensor_set(sensor, "noise flag", 0)

        sensor_size = np.asarray(sensor_get(sensor, "size"), dtype=int)
        exp_time = float(sensor_get(sensor, "exp time"))
        per_row = 10.0e-6
        rate = 0.3
        n_frames = int(sensor_size[0] + round(exp_time / per_row))
        crop_width = int(sensor_size[1] - 1)
        crop_height = int(sensor_size[0] - 1)

        volt_stack = np.zeros((sensor_size[0], sensor_size[1], n_frames), dtype=float)
        crop_rects = np.zeros((n_frames, 4), dtype=int)
        temporal_mean_volts = np.zeros(n_frames, dtype=float)
        current_sensor = sensor

        for frame_index in range(n_frames):
            rotated_scene = scene_rotate(scene, (frame_index + 1) * rate)
            oi_frame = oi_compute(oi, rotated_scene)
            center_pixel = np.asarray(oi_get(oi_frame, "center pixel"), dtype=float)
            rect = np.rint(
                [
                    center_pixel[1] - crop_width / 2.0,
                    center_pixel[0] - crop_height / 2.0,
                    crop_width,
                    crop_height,
                ],
            ).astype(int)
            crop_rects[frame_index, :] = rect
            oi_cropped = oi_crop(oi_frame, rect)
            current_sensor = sensor_compute(current_sensor, oi_cropped, seed=0)
            volts = np.asarray(sensor_get(current_sensor, "volts"), dtype=float)
            volt_stack[:, :, frame_index] = volts
            temporal_mean_volts[frame_index] = float(np.mean(volts))

        integration_rows = max(int(round(exp_time / per_row)), 1)
        slist = np.arange(integration_rows, dtype=int)
        final = np.zeros((sensor_size[0], sensor_size[1]), dtype=float)
        for row_index in range(sensor_size[0]):
            slist = slist + 1
            row_stack = volt_stack[row_index, :, :]
            final[row_index, :] = np.sum(row_stack[:, slist], axis=1)

        final_sensor = sensor_set(current_sensor.clone(), "volts", final)
        ip = ip_compute(ip_create(asset_store=store), final_sensor, asset_store=store)
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        result_flat = result.reshape(-1, result.shape[2])

        sampled_rows = np.array([0, sensor_size[0] // 2, sensor_size[0] - 1], dtype=int)
        sampled_cols = np.array([0, sensor_size[1] // 2, sensor_size[1] - 1], dtype=int)
        sampled_row_stats = np.vstack([_stats_vector(final[row_index, :]) for row_index in sampled_rows])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_size": sensor_size,
                "n_frames": int(n_frames),
                "crop_size": np.array([crop_height + 1, crop_width + 1], dtype=int),
                "first_crop_rect": crop_rects[0, :],
                "last_crop_rect": crop_rects[-1, :],
                "temporal_mean_volts": temporal_mean_volts,
                "center_pixel_trace": volt_stack[sensor_size[0] // 2, sensor_size[1] // 2, :],
                "final_stats": _stats_vector(final),
                "sampled_rows": sampled_rows + 1,
                "sampled_cols": sampled_cols + 1,
                "sampled_row_stats": sampled_row_stats,
                "result_mean_rgb_norm": _channel_normalize(np.mean(result_flat, axis=0, dtype=float)),
                "result_p95_rgb_norm": _channel_normalize(np.percentile(result_flat, 95.0, axis=0)),
            },
            context={"scene": scene, "sensor": final_sensor, "ip": ip},
        )

    if case_name == "sensor_imx490_uniform_small":
        scene = scene_create("uniform", 256, asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)
        oi = oi_crop(oi, "border")
        oi = oi_spatial_resample(oi, 3.0e-6)

        combined, metadata = imx490_compute(
            oi,
            "method",
            "best snr",
            "exp time",
            0.1,
            "noise flag",
            0,
            asset_store=store,
        )
        captures = list(metadata["sensorArray"])

        capture_mean_electrons = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "electrons"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        capture_mean_volts = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "volts"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        capture_mean_dv = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "dv"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        best_pixel = np.asarray(combined.metadata["bestPixel"], dtype=int)
        best_pixel_counts = np.array([int(np.sum(best_pixel == (index + 1))) for index in range(len(captures))], dtype=int)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "capture_names": np.asarray([str(sensor_get(sensor, "name")) for sensor in captures], dtype=object),
                "capture_mean_electrons": capture_mean_electrons,
                "capture_mean_volts": capture_mean_volts,
                "capture_mean_dv": capture_mean_dv,
                "large_gain_ratio": float(capture_mean_volts[1] / max(capture_mean_volts[0], 1.0e-12)),
                "small_area_ratio": float(capture_mean_electrons[2] / max(capture_mean_electrons[0], 1.0e-12)),
                "combined_volts_stats": _stats_vector(np.asarray(sensor_get(combined, "volts"), dtype=float)),
                "best_pixel_counts": best_pixel_counts,
            },
            context={"scene": scene, "oi": oi, "sensor": combined},
        )

    if case_name == "sensor_hdr_pixel_size_small":
        scene = scene_from_file(
            store.resolve("data/images/multispectral/Feng_Office-hdrs.mat"),
            "multispectral",
            200.0,
            asset_store=store,
        )
        oi = oi_compute(oi_create(asset_store=store), scene)

        pixel_sizes_um = np.array([1.0, 2.0, 4.0], dtype=float)
        dye_size_um = 512.0
        base_sensor = sensor_create("monochrome", asset_store=store)
        base_sensor = sensor_set(base_sensor, "exp time", 0.003)

        sensor_sizes = np.zeros((pixel_sizes_um.size, 2), dtype=int)
        mean_volts = np.zeros(pixel_sizes_um.size, dtype=float)
        p95_volts = np.zeros(pixel_sizes_um.size, dtype=float)
        mean_electrons = np.zeros(pixel_sizes_um.size, dtype=float)
        result_sizes = np.zeros((pixel_sizes_um.size, 3), dtype=int)
        result_mean_gray = np.zeros(pixel_sizes_um.size, dtype=float)
        result_p95_gray = np.zeros(pixel_sizes_um.size, dtype=float)

        for index, pixel_size_um in enumerate(pixel_sizes_um):
            sensor = sensor_set(base_sensor.clone(), "pixel size constant fill factor", np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6)
            sensor = sensor_set(sensor, "rows", int(np.rint(dye_size_um / pixel_size_um)))
            sensor = sensor_set(sensor, "cols", int(np.rint(dye_size_um / pixel_size_um)))
            sensor = sensor_compute(sensor, oi)
            ip = ip_compute(ip_create(asset_store=store), sensor, asset_store=store)
            result = np.asarray(ip_get(ip, "result"), dtype=float)
            gray = result[:, :, 0] if result.ndim == 3 else result

            sensor_sizes[index, :] = np.asarray(sensor_get(sensor, "size"), dtype=int)
            mean_volts[index] = float(np.mean(np.asarray(sensor_get(sensor, "volts"), dtype=float)))
            p95_volts[index] = float(np.percentile(np.asarray(sensor_get(sensor, "volts"), dtype=float), 95.0))
            mean_electrons[index] = float(np.mean(np.asarray(sensor_get(sensor, "electrons"), dtype=float)))
            result_sizes[index, :] = np.asarray(result.shape, dtype=int)
            result_mean_gray[index] = float(np.mean(gray))
            result_p95_gray[index] = float(np.percentile(gray, 95.0))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
                "pixel_sizes_um": pixel_sizes_um,
                "sensor_sizes": sensor_sizes,
                "mean_volts": mean_volts,
                "p95_volts": p95_volts,
                "mean_electrons": mean_electrons,
                "result_sizes": result_sizes,
                "result_mean_gray": result_mean_gray,
                "result_p95_gray": result_p95_gray,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_log_ar0132at_small":
        dynamic_range = float(2**16)
        scene = scene_create("exponential intensity ramp", 256, dynamic_range, asset_store=store)
        scene = scene_set(scene, "fov", 60.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.8)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "response type", "log")
        sensor = sensor_set(sensor, "size", np.array([960, 1280], dtype=int))
        sensor = sensor_set(sensor, "pixel size same fill factor", 3.751e-6)

        color_filter_file = store.resolve("data/sensor/colorfilters/auto/ar0132at.mat")
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, color_filter_file)
        sensor = sensor_set(sensor, "filter spectra", filter_spectra)
        sensor = sensor_set(sensor, "filter names", filter_names)
        sensor = sensor_set(sensor, "pixel read noise volts", 1.0e-3)
        sensor = sensor_set(sensor, "pixel voltage swing", 2.8)
        sensor = sensor_set(sensor, "pixel dark voltage", 1.0e-3)
        sensor = sensor_set(sensor, "pixel conversion gain", 110.0e-6)
        sensor = sensor_set(sensor, "exp time", 0.003)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi)

        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        sampled_cols = np.rint(np.linspace(0.0, volts.shape[1] - 1.0, 33)).astype(int)
        row_15 = np.asarray(volts[14, :], dtype=float)
        row_114 = np.asarray(volts[113, :], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "wave": wave,
                "dr_at_1s": float(sensor_dr(sensor, 1.0)),
                "volts_stats": _stats_vector(volts),
                "sampled_cols": sampled_cols + 1,
                "row15_stats": _stats_vector(row_15),
                "row114_stats": _stats_vector(row_114),
                "row15_profile_norm": _channel_normalize(row_15[sampled_cols]),
                "row114_profile_norm": _channel_normalize(row_114[sampled_cols]),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_aliasing_small":
        fov = 5.0
        sweep_scene = scene_create("sweep frequency", 768, 30.0, asset_store=store)
        sweep_scene = scene_set(sweep_scene, "fov", fov)

        oi = oi_create("diffraction limited", asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.0)
        oi = oi_compute(oi, sweep_scene)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, fov, oi)
        sensor = sensor_set(sensor, "noise flag", 0)

        sensor_small = sensor_set(sensor.clone(), "pixel size constant fill factor", 2.0e-6)
        sensor_small = sensor_compute(sensor_small, oi)
        small_line = sensor_get(sensor_small, "hline electrons", 1)
        small_data = np.asarray(small_line["data"][0], dtype=float)
        small_pos = np.asarray(small_line["pos"][0], dtype=float)

        sensor_large = sensor_set(sensor.clone(), "pixel size constant fill factor", 6.0e-6)
        sensor_large = sensor_set_size_to_fov(sensor_large, fov, oi)
        sensor_large = sensor_compute(sensor_large, oi)
        large_line = sensor_get(sensor_large, "hline electrons", 1)
        large_data = np.asarray(large_line["data"][0], dtype=float)
        large_pos = np.asarray(large_line["pos"][0], dtype=float)

        oi_blur = oi_set(oi.clone(), "optics fnumber", 12.0)
        oi_blur = oi_compute(oi_blur, sweep_scene)
        sensor_blur = sensor_compute(sensor_large.clone(), oi_blur)
        blur_line = sensor_get(sensor_blur, "hline electrons", 1)
        blur_data = np.asarray(blur_line["data"][0], dtype=float)
        blur_pos = np.asarray(blur_line["pos"][0], dtype=float)

        slanted_scene = scene_create("slanted bar", 1024, asset_store=store)
        slanted_scene = scene_set(slanted_scene, "fov", fov)

        oi_slanted_sharp = oi_set(oi_blur.clone(), "optics fnumber", 2.0)
        oi_slanted_sharp = oi_compute(oi_slanted_sharp, slanted_scene)
        sensor_slanted = sensor_set(sensor_large.clone(), "pixel size constant fill factor", 6.0e-6)
        sensor_slanted = sensor_set_size_to_fov(sensor_slanted, fov, oi_slanted_sharp)
        sensor_slanted = sensor_compute(sensor_slanted, oi_slanted_sharp)
        slanted_sharp = np.asarray(sensor_get(sensor_slanted, "electrons"), dtype=float)

        oi_slanted_blur = oi_set(oi_slanted_sharp.clone(), "optics fnumber", 12.0)
        oi_slanted_blur = oi_compute(oi_slanted_blur, slanted_scene)
        sensor_slanted_blur = sensor_compute(sensor_slanted.clone(), oi_slanted_blur)
        slanted_blur = np.asarray(sensor_get(sensor_slanted_blur, "electrons"), dtype=float)
        slanted_sharp = np.flipud(np.fliplr(slanted_sharp))
        slanted_blur = np.flipud(np.fliplr(slanted_blur))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fov_deg": fov,
                "sweep_scene_size": np.asarray(scene_get(sweep_scene, "size"), dtype=int),
                "sweep_oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "small_sensor_size": np.asarray(sensor_get(sensor_small, "size"), dtype=int),
                "large_sensor_size": np.asarray(sensor_get(sensor_large, "size"), dtype=int),
                "small_line_pos": small_pos,
                "small_line_data_norm": _channel_normalize(small_data),
                "large_line_pos": large_pos,
                "large_line_data_norm": _channel_normalize(large_data),
                "blur_line_pos": blur_pos,
                "blur_line_data_norm": _channel_normalize(blur_data),
                "small_line_stats": _stats_vector(small_data),
                "large_line_stats": _stats_vector(large_data),
                "blur_line_stats": _stats_vector(blur_data),
                "slanted_scene_size": np.asarray(scene_get(slanted_scene, "size"), dtype=int),
                "slanted_sensor_size": np.asarray(sensor_get(sensor_slanted, "size"), dtype=int),
                "slanted_sharp_norm": slanted_sharp / max(float(np.max(slanted_sharp)), 1.0e-12),
                "slanted_blur_norm": slanted_blur / max(float(np.max(slanted_blur)), 1.0e-12),
                "slanted_sharp_stats": _stats_vector(slanted_sharp),
                "slanted_blur_stats": _stats_vector(slanted_blur),
            },
            context={"scene": slanted_scene, "oi": oi_slanted_blur, "sensor": sensor_slanted_blur},
        )

    if case_name == "sensor_external_analysis_small":
        dut = sensor_create(asset_store=store)
        dut = sensor_set(dut, "name", "My Sensor")

        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        dut = sensor_set(dut, "wave", wave)
        dut = sensor_set(dut, "colorFilters", ie_read_spectra("RGB.mat", wave, asset_store=store))
        dut = sensor_set(dut, "irFilter", ie_read_spectra("infrared2.mat", wave, asset_store=store))
        dut = sensor_set(dut, "cfapattern", np.array([[2, 1], [3, 2]], dtype=int))
        dut = sensor_set(dut, "size", np.array([144, 176], dtype=int))
        dut = sensor_set(dut, "pixel name", "My Pixel")
        dut = sensor_set(dut, "pixel size constant fill factor", np.array([2.0e-6, 2.0e-6], dtype=float))
        dut = sensor_set(dut, "pixel spectral qe", ie_read_spectra("photodetector.mat", wave, asset_store=store))
        dut = sensor_set(dut, "pixel voltage swing", 1.5)

        volts = np.asarray(store.load_mat("scripts/sensor/dutData.mat")["volts"], dtype=float)
        dut = sensor_set(dut, "volts", volts)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_name": str(sensor_get(dut, "name")),
                "wave": np.asarray(sensor_get(dut, "wave"), dtype=float),
                "filter_spectra": np.asarray(sensor_get(dut, "filter spectra"), dtype=float),
                "ir_filter": np.asarray(sensor_get(dut, "ir filter"), dtype=float),
                "cfa_pattern": np.asarray(sensor_get(dut, "cfapattern"), dtype=int),
                "sensor_size": np.asarray(sensor_get(dut, "size"), dtype=int),
                "pixel_name": str(sensor_get(dut, "pixel name")),
                "pixel_size_m": np.asarray(sensor_get(dut, "pixel size"), dtype=float),
                "pixel_qe": np.asarray(sensor_get(dut, "pixel spectral qe"), dtype=float),
                "pixel_voltage_swing": float(sensor_get(dut, "pixel voltage swing")),
                "volts": np.asarray(sensor_get(dut, "volts"), dtype=float),
                "volts_stats": _stats_vector(np.asarray(sensor_get(dut, "volts"), dtype=float)),
            },
            context={"sensor": dut},
        )

    if case_name == "sensor_filter_transmissivities_small":
        sensor = sensor_create(asset_store=store)
        filters = np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float)
        modified = filters.copy()
        modified[:, 0] = modified[:, 0] * 0.2
        modified[:, 2] = modified[:, 2] * 0.5
        sensor = sensor_set(sensor, "filter transmissivities", modified)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(sensor_get(sensor, "wave"), dtype=float),
                "filters": np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float),
                "spectral_qe": np.asarray(sensor_get(sensor, "spectral qe"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_color_filter_gaussian_roundtrip_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        c_pos = np.arange(400.0, 701.0, 40.0, dtype=float)
        widths = np.full(c_pos.shape, 30.0, dtype=float)
        filters, wave = sensor_color_filter("gaussian", wave, c_pos, widths)
        payload = {
            "wavelength": wave,
            "data": filters,
            "filterNames": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "comment": "Gaussian filters created by parity sensorColorFilter",
            "peakWavelengths": c_pos,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = ie_save_color_filter(payload, f"{tmpdir}/gFiltersDeleteMe.mat")
            read_filters, read_names, file_data = ie_read_color_filter(wave, path, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "created_filters": filters,
                "read_filters": np.asarray(read_filters, dtype=float),
                "filter_names": np.asarray(read_names, dtype=object),
                "comment": str(file_data.get("comment", "")),
                "peak_wavelengths": np.asarray(file_data.get("peakWavelengths", np.array([], dtype=float)), dtype=float).reshape(-1),
            },
            context={},
        )

    if case_name == "sensor_color_filter_asset_nikond100_small":
        wave = np.arange(400.0, 1001.0, 1.0, dtype=float)
        filters, names, file_data = ie_read_color_filter(wave, "NikonD100", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "filters": np.asarray(filters, dtype=float),
                "filter_names": np.asarray(names, dtype=object),
                "comment": str(file_data.get("comment", "")),
            },
            context={},
        )

    if case_name == "sensor_cfa_ycmy_small":
        sensor = sensor_create("ycmy", asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([4, 4], dtype=int))
        sensor = sensor_set(
            sensor,
            "volts",
            np.arange(16, dtype=float).reshape((4, 4), order="F") / 15.0,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "filter_spectra": np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float),
                "rgb": np.asarray(sensor_get(sensor, "rgb"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_cfa_pattern_and_size_rgb_small":
        sensor = sensor_create("rgb", asset_store=store)
        sensor = sensor_set(sensor, "rows", 5)
        sensor = sensor_set(sensor, "cols", 7)
        sensor = sensor_set(sensor, "pattern and size", np.array([[2, 1, 2], [3, 2, 1], [2, 3, 2]], dtype=int))
        sensor = sensor_set(
            sensor,
            "volts",
            np.arange(54, dtype=float).reshape((6, 9), order="F") / 53.0,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "rgb": np.asarray(sensor_get(sensor, "rgb"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_snr_components_small":
        sensor = sensor_create(asset_store=store)
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        read_noise = float(sensor_get(sensor, "pixel read noise volts"))
        volts = np.logspace(np.log10(voltage_swing) - 4.0, np.log10(voltage_swing), 20, dtype=float)

        sensor = sensor_set(sensor, "pixel read noise volts", 3.0 * read_noise)
        sensor = sensor_set(sensor, "gainSD", 2.0)
        sensor = sensor_set(sensor, "offsetSD", voltage_swing * 0.005)

        snr, volts, snr_shot, snr_read, snr_dsnu, snr_prnu = sensor_snr(sensor, volts)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "volts": np.asarray(volts, dtype=float),
                "snr": np.asarray(snr, dtype=float),
                "snr_shot": np.asarray(snr_shot, dtype=float),
                "snr_read": np.asarray(snr_read, dtype=float),
                "snr_dsnu": np.asarray(snr_dsnu, dtype=float),
                "snr_prnu": np.asarray(snr_prnu, dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_counting_photons_small":
        scene = scene_create("uniform equal photon", np.array([128, 128], dtype=int), asset_store=store)
        scene = scene_set(scene, "mean luminance", 10.0)

        oi = oi_create("diffraction limited", asset_store=store)
        roi_rect = np.array([41, 31, 16, 23], dtype=int)
        fnumbers = np.arange(2.0, 17.0, dtype=float)
        total_q = np.zeros(fnumbers.shape, dtype=float)
        aperture_d = np.zeros(fnumbers.shape, dtype=float)
        spectral_irradiance = np.empty(0, dtype=float)

        for idx, f_number in enumerate(fnumbers):
            oi = oi_set(oi, "optics fnumber", float(f_number))
            oi = oi_compute(oi, scene)
            aperture_d[idx] = float(oi_get(oi, "optics aperture diameter", "mm"))
            spectral_irradiance = np.asarray(oi_get(oi, "roi mean photons", roi_rect), dtype=float).reshape(-1)
            total_q[idx] = float(np.sum(spectral_irradiance))

        s_factor = (1.0e-6**2) * 50.0e-3
        snr = (total_q * s_factor) / np.sqrt(np.maximum(total_q * s_factor, 1.0e-12))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(oi_get(oi, "wave"), dtype=float),
                "fnumbers": fnumbers,
                "aperture_d": aperture_d,
                "spectral_irradiance": spectral_irradiance,
                "total_q": total_q,
                "snr": snr,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_poisson_noise_small":
        scene = scene_create("macbeth", asset_store=store)
        scene = scene_set(scene, "fov", 10.0)
        oi = oi_create("diffraction limited", asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create("imx363", asset_store=store)
        sensor = sensor_set(sensor, "row", 256)
        sensor = sensor_set(sensor, "col", 256)
        sensor = sensor_set(sensor, "exp time", 0.016)
        sensor = sensor_compute(sensor, oi, seed=1)

        rect = np.array([96, 156, 24, 28], dtype=int)
        sensor = sensor_set(sensor, "roi", rect)
        dv = np.asarray(sensor_get(sensor, "roi dv", rect), dtype=float)
        finite_dv = dv[np.isfinite(dv)]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "rect": rect,
                "roi_mean_dv": float(np.mean(finite_dv)),
                "roi_std_dv": float(np.std(finite_dv, ddof=1)),
                "roi_percentiles": np.percentile(finite_dv, [10.0, 50.0, 90.0]),
                "sqrt_mean_dv": float(np.sqrt(np.mean(finite_dv))),
                "sensor_mean_dv": float(np.nanmean(np.asarray(sensor_get(sensor, "dv"), dtype=float))),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_estimation_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        macbeth_chart = ie_read_spectra("macbethChart", wave, asset_store=store)
        illuminant_d65 = ie_read_spectra("D65.mat", wave, asset_store=store).reshape(-1)
        sensors = ie_read_spectra("cMatch/camera", wave, asset_store=store)
        cones = ie_read_spectra("SmithPokornyCones", wave, asset_store=store)

        spectral_signals = illuminant_d65[:, None] * macbeth_chart
        rgb_responses = sensors.T @ spectral_signals

        estimate_full = (rgb_responses @ np.linalg.pinv(spectral_signals)).T
        rgb_pred_full = estimate_full.T @ spectral_signals

        sample_indices = np.arange(0, macbeth_chart.shape[1], 5, dtype=int)
        estimate_sparse = (rgb_responses[:, sample_indices] @ np.linalg.pinv(spectral_signals[:, sample_indices])).T
        rgb_pred_sparse = estimate_sparse.T @ spectral_signals

        gray_series = np.arange(3, macbeth_chart.shape[1], 4, dtype=int)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "green_reflectance": macbeth_chart[:, 6],
                "red_reflectance": macbeth_chart[:, 10],
                "gray_reflectance": macbeth_chart[:, 11],
                "illuminant_d65": illuminant_d65,
                "sensors": sensors,
                "cones": cones,
                "rgb_responses_gray": rgb_responses[:, gray_series],
                "estimate_full": estimate_full,
                "rgb_pred_full": rgb_pred_full,
                "estimate_sparse": estimate_sparse,
                "rgb_pred_sparse": rgb_pred_sparse,
            },
            context={},
        )

    if case_name == "sensor_macbeth_daylight_estimate_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        reflectance = ie_read_spectra("macbethChart", wave, asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor_filters = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)

        day_basis_energy = ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=store)
        day_basis_quanta = energy_to_quanta(day_basis_energy, wave)

        true_weights = np.array([1.0, 0.0, 0.0], dtype=float)
        illuminant_photons = day_basis_quanta @ true_weights
        camera_data = sensor_filters.T @ (illuminant_photons[:, None] * reflectance)

        x1 = sensor_filters.T @ (day_basis_quanta[:, [0]] * reflectance)
        x2 = sensor_filters.T @ (day_basis_quanta[:, [1]] * reflectance)
        x3 = sensor_filters.T @ (day_basis_quanta[:, [2]] * reflectance)
        design_matrix = np.column_stack(
            [
                x1.reshape(-1, order="F"),
                x2.reshape(-1, order="F"),
                x3.reshape(-1, order="F"),
            ]
        )
        camera_stacked = camera_data.reshape(-1, order="F")
        normal_matrix = design_matrix.T @ design_matrix
        rhs = design_matrix.T @ camera_stacked
        solved_weights = np.linalg.solve(normal_matrix, rhs)
        estimated_weights = solved_weights / solved_weights[0]
        estimated_illuminant = day_basis_quanta @ estimated_weights

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "reflectance": reflectance,
                "sensor_filters": sensor_filters,
                "day_basis_quanta": day_basis_quanta,
                "true_weights": true_weights,
                "illuminant_photons": illuminant_photons,
                "camera_data": camera_data,
                "design_matrix": design_matrix,
                "camera_stacked": camera_stacked,
                "normal_matrix": normal_matrix,
                "rhs": rhs,
                "estimated_weights": estimated_weights,
                "estimated_illuminant": estimated_illuminant,
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_spectral_radiometer_small":
        scene = scene_create("uniform d65", asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)

        wave = np.arange(400.0, 701.0, 1.0, dtype=float)
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, "radiometer", asset_store=store)
        w_samples = np.array([float(name) for name in filter_names], dtype=float)
        pattern = np.arange(1, filter_spectra.shape[1] + 1, dtype=int).reshape(1, -1)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "wave", wave)
        sensor = sensor_set(sensor, "filter spectra", filter_spectra)
        sensor = sensor_set(sensor, "filter names", filter_names)
        sensor = sensor_set(sensor, "pattern", pattern)
        sensor = sensor_set(sensor, "size", np.array([10, filter_spectra.shape[1]], dtype=int))
        sensor = sensor_set(sensor, "pixel fill factor", 1.0)
        sensor = sensor_set(sensor, "pixel size same fill factor", np.array([1.5e-6, 1.5e-6], dtype=float))
        sensor = sensor_set(sensor, "exposure time", 1.0 / 100.0)

        sensor_noisy = sensor_set(sensor.clone(), "noise flag", -2)
        sensor_noisy = sensor_compute(sensor_noisy, oi)
        electrons_noisy = np.asarray(sensor_get(sensor_noisy, "electrons"), dtype=float)

        sensor_noise_free = sensor_set(sensor.clone(), "noise flag", -1)
        sensor_noise_free = sensor_compute(sensor_noise_free, oi)
        electrons_noise_free = np.asarray(sensor_get(sensor_noise_free, "electrons"), dtype=float)

        noisy_line = electrons_noisy[4, :]
        noise_free_line = electrons_noise_free[4, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "w_samples": w_samples,
                "filter_pattern": pattern,
                "sensor_size": np.asarray(sensor_get(sensor_noise_free, "size"), dtype=int),
                "filter_spectra": np.asarray(sensor_get(sensor_noise_free, "filter spectra"), dtype=float),
                "noise_free_line": noise_free_line,
                "shot_sd_line": np.sqrt(np.maximum(noise_free_line, 0.0)),
                "noisy_line_stats": _stats_vector(noisy_line),
                "noisy_full_stats": _stats_vector(electrons_noisy),
                "noise_free_full_stats": _stats_vector(electrons_noise_free),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_noise_free},
        )

    if case_name == "sensor_spectral_estimation_small":
        scene = scene_create("uniform ee", asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)

        oi = oi_create("default", asset_store=store)
        oi = oi_set(oi, "optics model", "diffraction limited")
        oi = oi_set(oi, "optics fnumber", 0.01)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        sensor = sensor_set(sensor, "auto exposure", True)

        scene = scene_set(scene, "fov", float(sensor_get(sensor, "fov", scene, oi)) * 1.5)

        wave_step = 50.0
        centers = np.arange(wave[0], wave[-1] + wave_step, wave_step, dtype=float)
        width = wave_step / 2.0
        spd = np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / width) ** 2)
        spd *= 1.0e16

        n_filters = int(sensor_get(sensor, "nfilters"))
        responsivity = np.zeros((n_filters, centers.size), dtype=float)
        exposure_times = np.zeros(centers.size, dtype=float)

        for ii in range(centers.size):
            spd_image = np.broadcast_to(spd[:, ii][None, None, :], (32, 32, wave.size)).copy()
            trial_scene = scene_set(scene, "photons", spd_image)
            trial_oi = oi_compute(oi, trial_scene)
            computed = sensor_compute(sensor, trial_oi, False)
            exposure_times[ii] = float(sensor_get(computed, "exposure time"))

            for jj in range(n_filters):
                volts = np.asarray(sensor_get(computed, "volts", jj + 1), dtype=float)
                responsivity[jj, ii] = float(np.mean(volts)) / max(exposure_times[ii], 1e-12)

        weights = responsivity @ np.linalg.pinv(spd.T @ spd)
        estimated_filters = (weights @ spd.T).T
        estimated_peak = float(np.max(estimated_filters))
        if abs(estimated_peak) > 0.0:
            estimated_filters = estimated_filters / estimated_peak

        sensor_filters = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
        sensor_filters = sensor_filters / max(float(np.max(sensor_filters)), 1e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "centers": centers,
                "spd": spd,
                "exposure_times": exposure_times,
                "responsivity": responsivity,
                "weights": weights,
                "estimated_filters": estimated_filters,
                "sensor_filters": sensor_filters,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_exposure_color_small":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)

        filters = np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float)
        filters = filters.copy()
        filters[:, 0] *= 0.2
        filters[:, 2] *= 0.5
        sensor = sensor_set(sensor, "filter transmissivities", filters)
        sensor = sensor_set(sensor, "auto exposure", "on")

        sensor = sensor_compute(sensor, oi)
        exposure_time = float(sensor_get(sensor, "exposure time"))

        ip = ip_create(asset_store=store)
        ip = ip_compute(ip, sensor)
        combined_transform = np.asarray(ip_get(ip, "combined transform"), dtype=float)

        ip = ip_set(ip, "transform method", "current")
        sensor = sensor_set(sensor, "auto exposure", "off")
        sensor = sensor_set(sensor, "exposure time", 3.0 * exposure_time)
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip, sensor)
        result = np.asarray(ip_get(ip, "result"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exposure_time": exposure_time,
                "combined_transform": combined_transform,
                "mean_rgb": np.mean(result, axis=(0, 1)),
                "white_patch_rgb": np.mean(result[28:44, 36:52, :], axis=(0, 1)),
                "result": result,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "sensor_exposure_bracket_small":
        scene = scene_create(asset_store=store)
        scene = scene_set(scene, "fov", 4.0)

        oi = oi_create(asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        integration_times = np.array([0.02, 0.04, 0.08, 0.16, 0.32], dtype=float)
        sensor = sensor_set(sensor, "exp time", integration_times)
        sensor = sensor_set(sensor, "exposure plane", int(np.floor(integration_times.size / 2.0) + 1))
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi, False, seed=0)

        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        center_row = volts[volts.shape[0] // 2, :, :]
        center_col = volts[:, volts.shape[1] // 2, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "integration_times": np.asarray(sensor_get(sensor, "integration time"), dtype=float),
                "exposure_plane": int(sensor_get(sensor, "exposure plane")),
                "n_captures": int(sensor_get(sensor, "n captures")),
                "volts_means": np.mean(volts, axis=(0, 1)),
                "center_pixel": volts[volts.shape[0] // 2, volts.shape[1] // 2, :],
                "center_row_mean": np.mean(center_row, axis=0),
                "center_col_mean": np.mean(center_col, axis=0),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_exposure_cfa_small":
        scene = scene_create(asset_store=store)
        scene = scene_set(scene, "fov", 4.0)

        oi = oi_create(asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        bluish = np.array([[0.04, 0.03], [0.30, 0.02]], dtype=float)
        sensor_b = sensor_compute(sensor_set(sensor.clone(), "exposure duration", bluish), oi, False, seed=0)

        reddish = np.array([[0.04, 0.70], [0.03, 0.02]], dtype=float)
        sensor_r = sensor_compute(sensor_set(sensor.clone(), "exposure duration", reddish), oi, False, seed=0)

        camera = camera_create(asset_store=store)
        camera = camera_set(camera, "sensor noise flag", 0)
        camera = camera_set(camera, "sensor exposure duration", reddish)
        camera = camera_compute(camera, scene, asset_store=store)
        camera_sensor = camera_get(camera, "sensor")
        bluish_volts = np.asarray(sensor_get(sensor_b, "volts"), dtype=float)
        reddish_volts = np.asarray(sensor_get(sensor_r, "volts"), dtype=float)
        camera_volts = np.asarray(sensor_get(camera_sensor, "volts"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "bluish_mean_volts": float(np.mean(bluish_volts)),
                "reddish_mean_volts": float(np.mean(reddish_volts)),
                "reddish_center_pixel": reddish_volts[
                    int(sensor_r.fields["size"][0] // 2), int(sensor_r.fields["size"][1] // 2)
                ],
                "camera_mean_volts": float(np.mean(camera_volts)),
                "camera_center_pixel": camera_volts[
                    int(camera_sensor.fields["size"][0] // 2), int(camera_sensor.fields["size"][1] // 2)
                ],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_r, "camera": camera},
        )

    if case_name == "sensor_dark_voltage_small":
        scene = scene_create("uniform ee", asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        dark_scene = scene_adjust_luminance(scene, 1e-8, asset_store=store)

        oi = oi_compute(oi_create(asset_store=store), dark_scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_set(sensor, "noise seed", 1)
        exp_times = np.logspace(0.0, 1.5, 10)
        n_filters = int(sensor_get(sensor, "nfilters"))

        volt_columns: list[np.ndarray] = []
        for exp_time in exp_times:
            sensor = sensor_set(sensor, "exposureTime", float(exp_time))
            computed = sensor_compute(sensor, oi, 0)
            if n_filters == 3:
                volts = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")
            volt_columns.append(volts)

        volts = np.column_stack(volt_columns)
        mean_volts = np.mean(volts, axis=0)
        dark_voltage_estimate, offset = ie_fit_line(exp_times, mean_volts)
        true_dark_voltage = float(sensor_get(sensor, "dark voltage"))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exp_times": exp_times,
                "mean_volts": mean_volts,
                "dark_voltage_estimate": float(dark_voltage_estimate),
                "offset": float(offset),
                "true_dark_voltage": true_dark_voltage,
            },
            context={"scene": dark_scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_prnu_estimate_small":
        scene = scene_create("uniform ee", asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)

        oi = oi_create("default", asset_store=store)
        oi = oi_set(oi, "optics offaxis method", "skip")

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        sensor = sensor_set(sensor, "noise flag", 2)
        scene = scene_set(scene, "fov", float(sensor_get(sensor, "fov")) * 1.5)
        oi = oi_compute(oi, scene)

        exp_times = np.tile(np.arange(40.0, 62.0, 2.0, dtype=float) / 1000.0, 3)
        sensor = sensor_set(sensor, "dsnu sigma", 0.0)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "pixel read noise volts", 0.0)
        sensor = sensor_set(sensor, "pixel dark voltage", 0.0)

        volt_columns: list[np.ndarray] = []
        n_filters = int(sensor_get(sensor, "nfilters"))
        for ii, exp_time in enumerate(exp_times, start=1):
            trial_sensor = sensor.clone()
            trial_sensor = sensor_set(trial_sensor, "exposure time", float(exp_time))
            computed = sensor_compute(trial_sensor, oi, seed=ii)
            if n_filters == 3:
                volts = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")
            volt_columns.append(volts)

        volts = np.column_stack(volt_columns)
        a = np.column_stack([exp_times, np.ones(exp_times.shape[0], dtype=float)])
        fit = np.linalg.lstsq(a, volts.T, rcond=None)[0]
        slopes = np.asarray(fit[0, :], dtype=float)
        slopes = slopes / max(float(np.mean(slopes)), 1e-12)
        offsets = np.asarray(fit[1, :], dtype=float)
        prnu_estimate = 100.0 * float(np.std(slopes))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exp_times": exp_times,
                "prnu_estimate": prnu_estimate,
                "slope_mean": float(np.mean(slopes)),
                "slope_std": float(np.std(slopes)),
                "offset_mean": float(np.mean(offsets)),
                "offset_std": float(np.std(offsets)),
                "slope_sample": slopes[:8],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_dsnu_estimate_small":
        scene = scene_create("uniform ee", asset_store=store)
        dark_scene = scene_adjust_luminance(scene, 0.1, asset_store=store)

        oi = oi_create("default", asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        dark_scene = scene_set(dark_scene, "fov", float(sensor_get(sensor, "fov")) * 1.5)
        dark_oi = oi_compute(oi, dark_scene)

        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 0.1)
        sensor = sensor_set(sensor, "exposure time", 0.001)
        sensor = sensor_set(sensor, "pixel read noise volts", 0.001)
        sensor = sensor_set(sensor, "noise flag", 2)

        n_filters = int(sensor_get(sensor, "nfilters"))
        n_repeats = 25
        if n_filters == 3:
            n_samp = int(np.prod(sensor_get(sensor, "size")) // 2)
        else:
            n_samp = int(np.prod(sensor_get(sensor, "size")))
        volts = np.zeros((n_samp, n_repeats), dtype=float)
        for ii in range(n_repeats):
            computed = sensor_compute(sensor, dark_oi, seed=ii + 1)
            if n_filters == 3:
                volts[:, ii] = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts[:, ii] = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")

        mirrored = volts[volts > 1.0e-6]
        mirrored = np.concatenate([-mirrored.reshape(-1), mirrored.reshape(-1)])
        mean_offset = np.mean(volts, axis=1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "estimated_dsnu": float(np.std(mirrored)),
                "mean_offset_mean": float(np.mean(mean_offset)),
                "mean_offset_std": float(np.std(mean_offset)),
                "mean_offset_percentiles": np.percentile(mean_offset, [10.0, 50.0, 90.0]),
            },
            context={"scene": dark_scene, "oi": dark_oi, "sensor": sensor},
        )

    if case_name == "sensor_spatial_resolution_small":
        scene = scene_create("sweepFrequency", asset_store=store)
        scene = scene_set(scene, "fov", 1.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_set(oi, "optics focal length", 0.004)
        oi = oi_compute(oi, scene)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi)
        coarse_row = int(round(float(sensor_get(sensor, "rows")) / 2.0))
        coarse_support = sensor_get(sensor, "spatial support", "microns")
        coarse_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)

        oi_row = int(round(float(oi_get(oi, "rows")) / 2.0))
        oi_line, _ = oi_plot(oi, "horizontal line illuminance", np.array([1, oi_row], dtype=int))

        sensor_small = sensor_set(sensor, "pixel size Constant Fill Factor", np.array([2.0e-6, 2.0e-6], dtype=float))
        sensor_small = sensor_compute(sensor_small, oi)
        fine_row = int(round(float(sensor_get(sensor_small, "rows")) / 2.0))
        fine_support = sensor_get(sensor_small, "spatial support", "microns")
        fine_volts = np.asarray(sensor_get(sensor_small, "volts"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "coarse_pixPos": np.asarray(coarse_support["x"], dtype=float),
                "coarse_pixData": np.asarray(coarse_volts[coarse_row - 1, :], dtype=float),
                "fine_pixPos": np.asarray(fine_support["x"], dtype=float),
                "fine_pixData": np.asarray(fine_volts[fine_row - 1, :], dtype=float),
                "oi_pos": np.asarray(oi_line["pos"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_small},
        )

    if case_name == "sensor_fpn_noise_modes_small":
        scene = scene_create("uniform", 512, asset_store=store)
        scene = scene_set(scene, "fov", 8.0)

        oi = oi_compute(oi_create("wvf", asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "match oi", oi)
        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "read noise volts", 0.1)

        xy = np.array([1, 320], dtype=int)
        payload: dict[str, Any] = {"case_name": case_name}
        for noise_flag, label in ((0, "noise0"), (-2, "noiseM2"), (1, "noise1"), (2, "noise2")):
            trial_sensor = sensor.clone()
            trial_sensor = sensor_set(trial_sensor, "noise flag", noise_flag)
            trial_sensor = sensor_set(trial_sensor, "reuse noise", True)
            trial_sensor = sensor_set(trial_sensor, "noise seed", 0)
            computed = sensor_compute(trial_sensor, oi, seed=0)
            udata, _ = sensor_plot(computed, "volts hline", xy, "two lines", True)
            pix_pos = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in udata["pixPos"]])
            pix_data = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in udata["pixData"]])
            pix_color = np.asarray(udata["pixColor"], dtype=int).reshape(-1)
            if noise_flag == 0:
                payload["pixPos"] = pix_pos
                payload["pixColor"] = pix_color
                payload["noise0_pixData"] = pix_data
            else:
                payload[f"{label}_stats"] = _stats_vector(pix_data)

        return ParityCaseResult(
            payload=payload,
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_description_fpn_small":
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "read noise volts", 0.1)
        table, string_table, handle = sensor_description(sensor, show=False, close_window=False)
        rows = {str(row[0]): str(row[1]) for row in string_table.tolist()}
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "title": str(table.title),
                "handle_title": "" if handle is None else str(handle.title),
                "row_count": int(string_table.shape[0]),
                "col_count": int(string_table.shape[1]),
                "read_noise_volts": rows["Read noise (V)"],
                "analog_gain": rows["Analog gain"],
                "exposure_time": rows["Exposure time"],
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_dng_read_crop_small":
        dng_path = store.resolve("data/images/rawcamera/MCC-centered.dng")
        sensor, info = sensor_dng_read(
            dng_path,
            "full info",
            False,
            "crop",
            [500, 1000, 256, 256],
            asset_store=store,
        )
        ip = ip_compute(ip_create(asset_store=store), sensor, asset_store=store)
        result = np.asarray(ip.data["result"], dtype=float)
        result = result / np.maximum(np.max(result, axis=(0, 1), keepdims=True), 1e-12)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "black_level": float(sensor_get(sensor, "black level")),
                "exp_time": float(sensor_get(sensor, "exp time")),
                "iso_speed": float(info["isoSpeed"]),
                "digital_values": np.asarray(sensor_get(sensor, "digital values"), dtype=float),
                "result": result,
            },
            context={"sensor": sensor, "ip": ip},
        )

    if case_name == "ip_default_pipeline":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_compute(sensor_set(sensor_create(asset_store=store), "noise flag", 0), oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "input": ip.data["input"],
                "sensorspace": ip.data["sensorspace"],
                "result": ip.data["result"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "camera_default_pipeline":
        scene = scene_create(asset_store=store)
        camera = camera_create(asset_store=store)
        camera.fields["sensor"] = sensor_set(camera.fields["sensor"], "noise flag", 0)
        camera = camera_compute(camera, scene, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "result": camera.fields["ip"].data["result"],
                "sensor_volts": camera.fields["sensor"].data["volts"],
                "oi_photons": camera.fields["oi"].data["photons"],
            },
            context={
                "scene": scene,
                "camera": camera,
                "oi": camera.fields["oi"],
                "sensor": camera.fields["sensor"],
                "ip": camera.fields["ip"],
            },
        )

    raise KeyError(f"Unknown parity case: {case_name}")


def run_python_case(case_name: str, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    return run_python_case_with_context(case_name, asset_store=asset_store).payload
