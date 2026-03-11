"""Curated parity case runners."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_create
from .display import display_create
from .fileio import ie_save_si_data_file
from .metrics import cct_from_uv, delta_e_ab, metrics_spd, xyz_from_energy, xyz_to_lab, xyz_to_luv, xyz_to_uv
from .ip import ip_compute, ip_create
from .optics import (
    _cos4th_factor,
    _oi_geometry,
    _pad_scene,
    _radiance_to_irradiance,
    _shift_invariant_custom_otf,
    _wvf_psf_stack,
    airy_disk,
    oi_compute,
    oi_create,
    oi_get,
    psf_to_zcoeff_error,
    wvf_compute,
    wvf_aperture,
    wvf_compute_psf,
    wvf_create,
    wvf_get,
    wvf_pupil_function,
    wvf_set,
    wvf_to_oi,
    optics_psf_to_otf,
    oi_set,
    si_synthetic,
)
from .plotting import oi_plot
from .scene import scene_adjust_illuminant, scene_create, scene_get, scene_set
from .sensor import sensor_compute, sensor_create, sensor_create_ideal, sensor_set
from .utils import blackbody, energy_to_quanta, param_format, quanta_to_energy, unit_frequency_list


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

    if case_name == "oi_diffraction_limited_default":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
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
