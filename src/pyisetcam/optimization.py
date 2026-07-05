"""CameraE2E parameter optimization helpers."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from .assets import AssetStore
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario
from .types import Camera, Scene

ObjectiveSpec = Mapping[str, Any] | str | Iterable[Mapping[str, Any]]
ObjectiveCallable = Callable[[Mapping[str, Any]], float]

_PARAMETER_AXIS_CATALOG: dict[str, dict[str, Any]] = {
    "sensor.integration_time": {
        "area": "sensor",
        "unit": "s",
        "readiness_tier": "validated",
        "values": [0.001, 0.002, 0.004],
        "description": "Sensor exposure/integration time for RAW signal and noise tradeoffs.",
    },
    "sensor.analog_gain": {
        "area": "sensor",
        "unit": "x",
        "readiness_tier": "validated",
        "values": [1.0, 2.0, 4.0],
        "description": "Analog gain before digital conversion.",
    },
    "sensor.noise_flag": {
        "area": "sensor",
        "unit": "enum",
        "readiness_tier": "validated",
        "values": [0, 2],
        "description": "Sensor noise model selector.",
    },
    "sensor.pixel_size": {
        "area": "sensor_geometry",
        "unit": "m",
        "readiness_tier": "validated",
        "values": [2.0e-6, 2.8e-6, 3.75e-6],
        "description": (
            "Pixel pitch used by sensor sampling, pixel area, FOV-driven sensor sizing, "
            "and RAW signal/noise tradeoffs."
        ),
    },
    "sensor.pixel_fill_factor": {
        "area": "sensor_geometry",
        "unit": "fraction",
        "readiness_tier": "validated",
        "values": [0.55, 0.75, 0.95],
        "description": "Photodiode fill factor used in pixel-area signal collection.",
    },
    "sensor.n_samples_per_pixel": {
        "area": "sensor_geometry",
        "unit": "samples/pixel",
        "readiness_tier": "validated",
        "values": [1, 3, 5],
        "description": (
            "Sub-pixel spatial sampling for sensor irradiance integration. "
            "This is not a readout-binning factor."
        ),
    },
    "sensor.cfa_pattern": {
        "area": "sensor_spectral",
        "unit": "index matrix",
        "readiness_tier": "validated",
        "values": [
            [[1, 2], [2, 3]],
            [[2, 1], [3, 2]],
        ],
        "description": (
            "CFA pattern-and-size matrix. Filter spectra/names must remain consistent "
            "with the selected pattern indices."
        ),
    },
    "sensor.binning_method": {
        "area": "sensor_readout",
        "unit": "enum",
        "readiness_tier": "proxy",
        "values": ["off", "kodak2008"],
        "description": (
            "Legacy pixel-binning compute wrapper. Treat as a readout proxy until "
            "charge-domain, readout-domain, and ISP-domain binning are separated."
        ),
    },
    "sensor.pixel_read_noise_v": {
        "area": "sensor_noise",
        "unit": "V",
        "readiness_tier": "validated",
        "values": [5.0e-4, 1.0e-3, 2.0e-3],
        "description": "Pixel read-noise voltage used by the sensor noise model.",
    },
    "sensor.pixel_dark_voltage": {
        "area": "sensor_noise",
        "unit": "V/s",
        "readiness_tier": "validated",
        "values": [2.5e-4, 1.0e-3, 4.0e-3],
        "description": "Pixel dark voltage accumulation rate.",
    },
    "sensor.pixel_voltage_swing": {
        "area": "sensor_readout",
        "unit": "V",
        "readiness_tier": "validated",
        "values": [0.6, 1.0, 1.4],
        "description": "Pixel voltage swing/full-well proxy used by clipping and quantization.",
    },
    "sensor.pixel_conversion_gain": {
        "area": "sensor_readout",
        "unit": "V/e-",
        "readiness_tier": "validated",
        "values": [5.0e-5, 1.0e-4, 2.0e-4],
        "description": "Pixel conversion gain used for electron-to-voltage conversion.",
    },
    "optics.fnumber": {
        "area": "optics",
        "unit": "f/#",
        "readiness_tier": "validated",
        "values": [2.0, 2.8, 4.0],
        "description": "Optics f-number for irradiance and diffraction/blur tradeoffs.",
    },
    "optics.focal_length": {
        "area": "optics",
        "unit": "m",
        "readiness_tier": "validated",
        "values": [0.0028, 0.004, 0.006],
        "description": "Effective focal length for FOV, magnification, and ADAS crop tradeoffs.",
    },
    "optics.si_psf_radius_um": {
        "area": "optics_psf",
        "unit": "um",
        "readiness_tier": "proxy",
        "values": [1.0, 2.0, 4.0],
        "description": (
            "Synthetic shift-invariant pillbox PSF radius. This is a blur proxy, "
            "not RayOptics geometric PSF or diffraction/wave-optics sign-off."
        ),
    },
    "optics.psf_angle_step": {
        "area": "optics_psf",
        "unit": "deg",
        "readiness_tier": "proxy",
        "values": [5.0, 10.0, 15.0],
        "description": "RayOptics/geometric PSF angular sampling step.",
    },
    "optics.rt_compute_spacing": {
        "area": "optics_psf",
        "unit": "m",
        "readiness_tier": "proxy",
        "values": [1.0e-6, 2.5e-6, 5.0e-6],
        "description": "Raytrace PSF compute spacing for geometric PSF resampling.",
    },
    "fdtd.mode": {
        "area": "sensor_physics",
        "unit": "enum",
        "readiness_tier": "proxy",
        "values": ["qe", "qe+field", "qe+field+crosstalk"],
        "description": "FDTD optical-response mode. Requires an attached FDTD LUT.",
    },
    "fdtd.crosstalk_strength": {
        "area": "sensor_physics",
        "unit": "x",
        "readiness_tier": "proxy",
        "values": [0.0, 0.5, 1.0],
        "description": "Optical-crosstalk strength applied from an attached FDTD LUT.",
    },
    "tcad.collection_mode": {
        "area": "sensor_physics",
        "unit": "enum",
        "readiness_tier": "calibration_required",
        "values": ["collection", "off"],
        "description": "TCAD collection-response mode. Requires an attached TCAD DB/LUT.",
    },
    "ip.demosaic_method": {
        "area": "isp",
        "unit": "enum",
        "readiness_tier": "validated",
        "values": ["bilinear", "nearest neighbor", "laplacian"],
        "description": "Demosaic method used by the image processor.",
    },
    "hw_isp.ae_apply_delay_frames": {
        "area": "hw_isp",
        "unit": "frames",
        "readiness_tier": "proxy",
        "values": [0, 1, 2],
        "description": "Delayed AE control application in the HW ISP simulator.",
    },
    "hw_isp.awb_apply_delay_frames": {
        "area": "hw_isp",
        "unit": "frames",
        "readiness_tier": "proxy",
        "values": [0, 1, 2],
        "description": "Delayed AWB control application in the HW ISP simulator.",
    },
    "hw_isp.global_latency_factor": {
        "area": "hw_isp",
        "unit": "x",
        "readiness_tier": "proxy",
        "values": [0.8, 1.0, 1.2],
        "description": "Global stage latency scale for system-control sweeps.",
    },
}

_PARAMETER_SPACE_PRESETS: dict[str, tuple[str, ...]] = {
    "exposure": ("sensor.integration_time", "sensor.analog_gain"),
    "optics": ("optics.fnumber",),
    "optics_psf": ("optics.fnumber", "optics.focal_length", "optics.si_psf_radius_um"),
    "raytrace_psf": ("optics.psf_angle_step", "optics.rt_compute_spacing"),
    "sensor_geometry": (
        "sensor.pixel_size",
        "sensor.pixel_fill_factor",
        "sensor.n_samples_per_pixel",
    ),
    "sensor_spectral": ("sensor.cfa_pattern",),
    "sensor_readout": (
        "sensor.integration_time",
        "sensor.analog_gain",
        "sensor.pixel_read_noise_v",
        "sensor.pixel_voltage_swing",
        "sensor.binning_method",
    ),
    "physics_proxy": (
        "fdtd.mode",
        "fdtd.crosstalk_strength",
        "tcad.collection_mode",
    ),
    "isp": ("ip.demosaic_method",),
    "hw_isp_control": (
        "hw_isp.ae_apply_delay_frames",
        "hw_isp.awb_apply_delay_frames",
        "hw_isp.global_latency_factor",
    ),
    "research_smoke": ("sensor.integration_time", "optics.fnumber"),
    "raw_factory": ("sensor.integration_time", "sensor.analog_gain", "optics.fnumber"),
    "adas_camera": (
        "optics.focal_length",
        "optics.fnumber",
        "sensor.pixel_size",
        "sensor.integration_time",
        "sensor.analog_gain",
    ),
}

_SENSOR_OPTIMIZATION_PARAMETERS = {
    "integration_time",
    "integration time",
    "integration",
    "exposure_duration",
    "exposure duration",
    "exposure_time",
    "exposure time",
    "analog_gain",
    "analog gain",
    "gain",
    "noise_flag",
    "noise flag",
    "noise",
    "pixel_size",
    "pixel size",
    "pixel_size_m",
    "pixel size m",
    "pixel_pitch",
    "pixel pitch",
    "pixel_fill_factor",
    "pixel fill factor",
    "fill_factor",
    "fill factor",
    "n_samples_per_pixel",
    "n samples per pixel",
    "samples_per_pixel",
    "samples per pixel",
    "cfa_pattern",
    "cfa pattern",
    "pattern",
    "pattern_and_size",
    "pattern and size",
    "cfa",
    "color_filter_array",
    "color filter array",
    "filter_names",
    "filter names",
    "filter_spectra",
    "filter spectra",
    "color_filters",
    "color filters",
    "pixel_read_noise_v",
    "pixel read noise v",
    "read_noise",
    "read noise",
    "read_noise_v",
    "read noise v",
    "pixel_dark_voltage",
    "pixel dark voltage",
    "dark_voltage",
    "dark voltage",
    "pixel_voltage_swing",
    "pixel voltage swing",
    "voltage_swing",
    "voltage swing",
    "pixel_conversion_gain",
    "pixel conversion gain",
    "conversion_gain",
    "conversion gain",
    "sensor_compute_method",
    "sensor compute method",
    "binning",
    "binning_method",
    "binning method",
    "pixel_binning",
    "pixel binning",
}

_OBJECTIVE_METRIC_CATALOG: dict[str, dict[str, Any]] = {
    "metrics.color.rgb_mean": {
        "area": "color",
        "direction_hint": "maximize",
        "description": "Mean rendered RGB response.",
    },
    "metrics.artifact.raw_std": {
        "area": "artifact",
        "direction_hint": "minimize_or_target",
        "description": "RAW variation/noise proxy from the sensor stage.",
    },
    "metrics.artifact.rgb_std": {
        "area": "artifact",
        "direction_hint": "minimize_or_target",
        "description": "Rendered RGB variation proxy.",
    },
    "metrics.artifact.rgb_clip_fraction": {
        "area": "artifact",
        "direction_hint": "minimize",
        "description": "Fraction of rendered RGB samples clipped to 0 or 1.",
    },
    "metrics.control.frame_count": {
        "area": "control",
        "direction_hint": "target",
        "description": "HW ISP sequence frame count when HW ISP simulation is enabled.",
    },
}


def camerae2e_optimize_parameters(
    base_scenario: Mapping[str, Any] | None = None,
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]] | None = None,
    objective: ObjectiveSpec | ObjectiveCallable | None = None,
    *,
    constraints: Iterable[Mapping[str, Any]] | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    top_k: int = 5,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Optimize CameraE2E camera parameters with deterministic grid search.

    Parameters use dot paths.  Examples:

    - ``sensor.integration_time``
    - ``sensor.analog_gain``
    - ``optics.fnumber``
    - ``ip.demosaic_method``

    The score is always maximized.  Objective specs can maximize, minimize, or
    target any numeric path in the FACA report.
    """

    axes = _normalize_parameter_space(parameter_space or {})
    parameter_validation = camerae2e_parameter_space_validate(
        axes,
        base_scenario=base_scenario,
    )
    _validate_parameter_axis_names(axes, base_scenario)
    objective_specs = _normalize_objective(objective)
    constraint_specs = [dict(item) for item in constraints or []]
    cases: list[dict[str, Any]] = []
    keys = list(axes)
    combinations = [()] if not keys else itertools.product(*(axes[key] for key in keys))
    for index, values in enumerate(combinations):
        scenario = _deep_dict(base_scenario or {})
        scenario.setdefault("name", "camerae2e_parameter_optimization")
        axis_values = {}
        for key, value in zip(keys, values, strict=True):
            _assign_parameter(scenario, key, value)
            axis_values[key] = value
        report = camerae2e_faca_report(
            camerae2e_run_scenario(
                scenario,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=int(seed) + index,
                include_arrays=include_arrays,
            )
        )
        objective_values, objective_utilities, score = _score_report(
            report, objective_specs, objective
        )
        constraint_results = [_evaluate_constraint(report, item) for item in constraint_specs]
        feasible = all(item["pass"] for item in constraint_results)
        if not feasible:
            score = -math.inf
        cases.append(
            {
                "case_index": index,
                "seed": int(seed) + index,
                "parameters": _jsonable(axis_values),
                "score": float(score),
                "feasible": bool(feasible),
                "objective_values": _jsonable(objective_values),
                "objective_utilities": _jsonable(objective_utilities),
                "constraint_results": _jsonable(constraint_results),
                "scenario": _jsonable(report.get("scenario", {})),
                "report": report,
            }
        )

    ranked = sorted(
        cases,
        key=lambda item: (float(item["score"]), -int(item["case_index"])),
        reverse=True,
    )
    feasible_cases = [item for item in ranked if item["feasible"]]
    pareto = _pareto_cases(feasible_cases)
    best = feasible_cases[0] if feasible_cases else None
    return {
        "schema_version": "camerae2e_parameter_optimization_v1",
        "method": "deterministic_grid",
        "seed": int(seed),
        "parameter_space": _jsonable(axes),
        "parameter_space_validation": parameter_validation,
        "objective": _jsonable(_objective_description(objective, objective_specs)),
        "constraints": _jsonable(constraint_specs),
        "case_count": len(cases),
        "feasible_count": len(feasible_cases),
        "pareto_case_count": len(pareto),
        "best_case": _strip_case_report(best),
        "top_cases": [_strip_case_report(item) for item in feasible_cases[: max(int(top_k), 0)]],
        "pareto_front": [_strip_case_report(item) for item in pareto],
        "selected_scenarios": [
            _jsonable(item.get("scenario", {}))
            for item in feasible_cases[: max(int(top_k), 0)]
        ],
        "cases": cases,
    }


def camerae2e_parameter_space_catalog(preset: str | None = None) -> dict[str, Any]:
    """Return supported automated CameraE2E parameter axes and presets."""

    if preset is None:
        return {
            "schema_version": "camerae2e_parameter_space_catalog_v1",
            "axes": _jsonable(_PARAMETER_AXIS_CATALOG),
            "presets": {name: list(paths) for name, paths in _PARAMETER_SPACE_PRESETS.items()},
        }
    key = str(preset).strip().lower()
    if key not in _PARAMETER_SPACE_PRESETS:
        raise ValueError(f"Unknown CameraE2E parameter-space preset: {preset!r}.")
    paths = _PARAMETER_SPACE_PRESETS[key]
    return {
        "schema_version": "camerae2e_parameter_space_catalog_v1",
        "preset": key,
        "axes": {path: _jsonable(_PARAMETER_AXIS_CATALOG[path]) for path in paths},
        "parameter_space": {path: list(_PARAMETER_AXIS_CATALOG[path]["values"]) for path in paths},
    }


def camerae2e_optimization_config_catalog() -> dict[str, Any]:
    """Return all documented CameraE2E optimization configure targets.

    Registered axes are validated smoke-test targets.  Custom path rules
    describe additional dot-paths that the optimizer can assign, but callers
    should still verify parameter lineage and FACA metric movement.
    """

    return {
        "schema_version": "camerae2e_optimization_config_catalog_v1",
        "registered_axis_count": len(_PARAMETER_AXIS_CATALOG),
        "registered_axes": _jsonable(_PARAMETER_AXIS_CATALOG),
        "presets": {name: list(paths) for name, paths in _PARAMETER_SPACE_PRESETS.items()},
        "custom_path_rules": [
            {
                "path_pattern": "sensor.<name>",
                "assignment": "scenario.sensor -> sensor_set(...)",
                "readiness_tier": "validated",
                "allowed_suffixes": sorted(_SENSOR_OPTIMIZATION_PARAMETERS),
                "registered_examples": [
                    "sensor.integration_time",
                    "sensor.analog_gain",
                    "sensor.noise_flag",
                    "sensor.pixel_size",
                    "sensor.pixel_fill_factor",
                    "sensor.cfa_pattern",
                    "sensor.binning_method",
                ],
                "truth_boundary": (
                    "sensor.binning_method routes to a legacy binning compute proxy; "
                    "sensor.n_samples_per_pixel is sub-pixel integration sampling, "
                    "not readout binning."
                ),
            },
            {
                "path_pattern": "hw_isp.<name>",
                "assignment": "scenario.hw_isp control dictionary",
                "readiness_tier": "proxy",
                "allowed_suffixes": "open, simulator-dependent",
                "side_effect": "enabled=True is inserted for HW ISP axes except hw_isp.enabled",
                "registered_examples": [
                    "hw_isp.ae_apply_delay_frames",
                    "hw_isp.awb_apply_delay_frames",
                    "hw_isp.global_latency_factor",
                ],
            },
            {
                "path_pattern": "fdtd.<name>",
                "assignment": "scenario.fdtd attachment options",
                "readiness_tier": "proxy",
                "allowed_suffixes": [
                    "lut",
                    "mode",
                    "field_model",
                    "case",
                    "cra_x_deg",
                    "cra_z_deg",
                    "crosstalk_strength",
                ],
                "requirement": (
                    "FDTD configure axes only change the pipeline when "
                    "base_scenario.fdtd.lut is attached."
                ),
            },
            {
                "path_pattern": "tcad.<name>",
                "assignment": "scenario.tcad attachment options",
                "readiness_tier": "calibration_required",
                "allowed_suffixes": ["db", "collection_mode"],
                "requirement": (
                    "tcad.collection_mode only changes the pipeline when "
                    "base_scenario.tcad.db is attached."
                ),
            },
            {
                "path_pattern": "<camera_set dot path>",
                "assignment": "scenario.parameters/camera_parameters -> camera_set(...)",
                "readiness_tier": "available",
                "examples": [
                    "optics.fnumber",
                    "optics.focal_length",
                    "optics.si_psf_radius_um",
                    "optics.psf_angle_step",
                    "optics.rt_compute_spacing",
                    "pixel.size",
                    "sensor.bits",
                    "ip.demosaic_method",
                ],
                "truth_boundary": (
                    "The optimizer can assign these paths, but actual effect depends "
                    "on camera_set support and should be verified with parameter_lineage. "
                    "optics.si_psf_radius_um is a CameraE2E high-level synthetic PSF proxy, "
                    "not a product lens PSF calibration."
                ),
            },
        ],
        "objective_metrics": _jsonable(_OBJECTIVE_METRIC_CATALOG),
    }


def camerae2e_parameter_space_validate(
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]],
    *,
    base_scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and classify optimization parameter-space paths without running a sweep."""

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        axes = _normalize_parameter_space(parameter_space)
    except ValueError as exc:
        return {
            "schema_version": "camerae2e_parameter_space_validation_v1",
            "ok": False,
            "axis_count": 0,
            "issue_count": 1,
            "warning_count": 0,
            "issues": [{"kind": "parameter_space", "message": str(exc)}],
            "warnings": [],
            "axes": {},
        }
    axis_payload: dict[str, dict[str, Any]] = {}
    for path, values in axes.items():
        axis_issues = _parameter_axis_issues(path, base_scenario)
        issues.extend(axis_issues)
        classification = _parameter_axis_classification(path)
        if classification["status"] == "custom_passthrough":
            warnings.append(
                {
                    "path": path,
                    "kind": "custom_passthrough",
                    "message": (
                        "Axis is assignable through camera_set passthrough but is not "
                        "a registered validated optimization axis."
                    ),
                }
            )
        axis_payload[path] = {
            "values": _jsonable(values),
            "value_count": len(values),
            **classification,
            "issues": axis_issues,
        }
    return {
        "schema_version": "camerae2e_parameter_space_validation_v1",
        "ok": not issues,
        "axis_count": len(axes),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "axes": axis_payload,
    }


def camerae2e_optimize_camera_parameters(
    base_scenario: Mapping[str, Any] | None = None,
    *,
    preset: str = "raw_factory",
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]] | None = None,
    objective: ObjectiveSpec | ObjectiveCallable | None = None,
    constraints: Iterable[Mapping[str, Any]] | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    top_k: int = 5,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Run the automated CameraE2E camera-parameter optimization baseline.

    ``preset`` supplies a validated parameter-space starter set.  Passing
    ``parameter_space`` overrides or extends those axes while retaining the
    same deterministic optimizer, constraint, Pareto, and selected-scenario
    outputs as :func:`camerae2e_optimize_parameters`.
    """

    catalog = camerae2e_parameter_space_catalog(preset)
    axes = dict(catalog["parameter_space"])
    if parameter_space is not None:
        axes.update(_normalize_parameter_space(parameter_space))
    result = camerae2e_optimize_parameters(
        base_scenario,
        axes,
        objective,
        constraints=constraints,
        scene=scene,
        camera=camera,
        asset_store=asset_store,
        seed=seed,
        top_k=top_k,
        include_arrays=include_arrays,
    )
    result["automation"] = {
        "schema_version": "camerae2e_parameter_optimization_automation_v1",
        "preset": str(preset).strip().lower(),
        "axis_count": len(axes),
        "axes": _jsonable(
            {
                path: _PARAMETER_AXIS_CATALOG.get(
                    path,
                    {
                        "readiness_tier": "available",
                        "description": "Caller-provided custom parameter axis.",
                    },
                )
                for path in axes
            }
        ),
    }
    return result


def camerae2e_pareto_front(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact non-dominated feasible cases from an optimization result."""

    if "pareto_front" in result:
        return _jsonable(result["pareto_front"])
    return [_strip_case_report(item) for item in _pareto_cases(list(result.get("cases", [])))]


def camerae2e_optimization_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe optimization report."""

    best = result.get("best_case")
    return {
        "schema_version": "camerae2e_parameter_optimization_report_v1",
        "method": result.get("method"),
        "seed": result.get("seed"),
        "case_count": result.get("case_count", 0),
        "feasible_count": result.get("feasible_count", 0),
        "pareto_case_count": result.get("pareto_case_count", 0),
        "objective": _jsonable(result.get("objective", {})),
        "constraints": _jsonable(result.get("constraints", [])),
        "best_case": _jsonable(best),
        "top_cases": _jsonable(result.get("top_cases", [])),
        "pareto_front": _jsonable(result.get("pareto_front", [])),
        "selected_scenarios": _jsonable(result.get("selected_scenarios", [])),
    }


def _normalize_parameter_space(
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]]
) -> dict[str, list[Any]]:
    axes: dict[str, list[Any]] = {}
    for key, raw in parameter_space.items():
        values = _axis_values(raw)
        if not values:
            raise ValueError(f"Parameter axis {key!r} must contain at least one value.")
        axes[str(key)] = values
    return axes


def _validate_parameter_axis_names(
    axes: Mapping[str, Any], base_scenario: Mapping[str, Any] | None
) -> None:
    for key in axes:
        issues = _parameter_axis_issues(str(key), base_scenario)
        if issues:
            raise ValueError(str(issues[0]["message"]))


def _parameter_axis_issues(
    path: str, base_scenario: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    if "." not in path:
        return []
    prefix, remainder = path.split(".", 1)
    if prefix == "sensor" and remainder not in _SENSOR_OPTIMIZATION_PARAMETERS:
        return [
            {
                "path": path,
                "kind": "unsupported_sensor_axis",
                "message": f"Unsupported sensor optimization parameter: {path!r}.",
            }
        ]
    if prefix == "fdtd" and remainder not in {"lut", "enabled"} and not _has_nested_value(
        base_scenario, "fdtd", "lut"
    ):
        return [
            {
                "path": path,
                "kind": "inactive_fdtd_axis",
                "message": (
                    f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                    "optimizing FDTD options alone would not change the camera pipeline."
                ),
            }
        ]
    if prefix == "tcad" and remainder == "collection_mode" and not _has_nested_value(
        base_scenario, "tcad", "db"
    ):
        return [
            {
                "path": path,
                "kind": "inactive_tcad_axis",
                "message": (
                    f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                    "optimizing the mode alone would not change the camera pipeline."
                ),
            }
        ]
    return []


def _parameter_axis_classification(path: str) -> dict[str, Any]:
    if path in _PARAMETER_AXIS_CATALOG:
        return {
            "status": "registered",
            "assignment": _assignment_for_path(path),
            "readiness_tier": _PARAMETER_AXIS_CATALOG[path]["readiness_tier"],
        }
    if "." not in path:
        return {
            "status": "custom_passthrough",
            "assignment": "scenario.parameters -> camera_set(...)",
            "readiness_tier": "available",
        }
    prefix, remainder = path.split(".", 1)
    if prefix == "sensor" and remainder in _SENSOR_OPTIMIZATION_PARAMETERS:
        return {
            "status": "assignable",
            "assignment": "scenario.sensor -> sensor_set(...)",
            "readiness_tier": "validated",
        }
    if prefix == "hw_isp":
        return {
            "status": "assignable",
            "assignment": "scenario.hw_isp control dictionary",
            "readiness_tier": "proxy",
        }
    if prefix == "fdtd":
        return {
            "status": "assignable",
            "assignment": "scenario.fdtd attachment options",
            "readiness_tier": "proxy",
        }
    if prefix == "tcad":
        return {
            "status": "assignable",
            "assignment": "scenario.tcad attachment options",
            "readiness_tier": "calibration_required",
        }
    return {
        "status": "custom_passthrough",
        "assignment": "scenario.parameters/camera_parameters -> camera_set(...)",
        "readiness_tier": "available",
    }


def _assignment_for_path(path: str) -> str:
    prefix = path.split(".", 1)[0]
    if prefix == "sensor":
        return "scenario.sensor -> sensor_set(...)"
    if prefix == "hw_isp":
        return "scenario.hw_isp control dictionary"
    if prefix in {"fdtd", "tcad"}:
        return f"scenario.{prefix} attachment options"
    return "scenario.parameters/camera_parameters -> camera_set(...)"


def _has_nested_value(payload: Mapping[str, Any] | None, bucket: str, name: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    value = payload.get(bucket)
    return isinstance(value, Mapping) and value.get(name) is not None


def _axis_values(raw: Iterable[Any] | Mapping[str, Any]) -> list[Any]:
    if isinstance(raw, Mapping):
        if "values" in raw:
            return list(raw["values"])
        if "linspace" in raw:
            start, stop, count = raw["linspace"]
            return np.linspace(float(start), float(stop), int(count)).tolist()
        if {"start", "stop", "num"} <= set(raw):
            return np.linspace(float(raw["start"]), float(raw["stop"]), int(raw["num"])).tolist()
        if {"start", "stop", "step"} <= set(raw):
            start = float(raw["start"])
            stop = float(raw["stop"])
            step = float(raw["step"])
            if step == 0:
                raise ValueError("Parameter axis step must be non-zero.")
            count = int(math.floor((stop - start) / step)) + 1
            return [start + index * step for index in range(max(count, 0))]
        raise ValueError("Parameter axis mapping must define values, linspace, or start/stop.")
    return list(raw)


def _normalize_objective(
    objective: ObjectiveSpec | ObjectiveCallable | None,
) -> list[dict[str, Any]]:
    if objective is None:
        return [{"metric": "metrics.color.rgb_mean", "direction": "maximize", "weight": 1.0}]
    if callable(objective):
        return [{"metric": "<callable>", "direction": "maximize", "weight": 1.0}]
    if isinstance(objective, str):
        return [{"metric": objective, "direction": "maximize", "weight": 1.0}]
    if isinstance(objective, Mapping):
        return [dict(objective)]
    return [dict(item) for item in objective]


def _score_report(
    report: Mapping[str, Any],
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
) -> tuple[dict[str, Any], dict[str, float], float]:
    if callable(objective):
        score = float(objective(report))
        return {"<callable>": score}, {"<callable>": score}, score
    values = {}
    utilities = {}
    score = 0.0
    for spec in objective_specs:
        metric = str(spec.get("metric", spec.get("path", "")))
        value = _numeric_path(report, metric)
        weight = float(spec.get("weight", 1.0))
        direction = str(spec.get("direction", "maximize")).lower()
        target = spec.get("target")
        if target is not None:
            component = -abs(value - float(target))
        elif direction in {"min", "minimize", "lower"}:
            component = -value
        elif direction in {"max", "maximize", "higher"}:
            component = value
        else:
            raise ValueError(f"Unsupported objective direction: {direction}")
        values[metric] = value
        utilities[metric] = float(component)
        score += weight * component
    return values, utilities, float(score)


def _evaluate_constraint(report: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    metric = str(spec.get("metric", spec.get("path", "")))
    op = str(spec.get("op", spec.get("operator", "<=")))
    limit = float(spec.get("value", spec.get("limit")))
    value = _numeric_path(report, metric)
    if op in {"<", "lt"}:
        passed = value < limit
    elif op in {"<=", "le"}:
        passed = value <= limit
    elif op in {">", "gt"}:
        passed = value > limit
    elif op in {">=", "ge"}:
        passed = value >= limit
    elif op in {"==", "eq"}:
        passed = math.isclose(value, limit)
    else:
        raise ValueError(f"Unsupported constraint operator: {op}")
    return {"metric": metric, "op": op, "limit": limit, "value": value, "pass": bool(passed)}


def _numeric_path(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Metric path {path!r} is not present in the FACA report.")
    if value is None:
        raise ValueError(f"Metric path {path!r} resolved to None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric path {path!r} resolved to a non-finite value.")
    return number


def _assign_parameter(scenario: dict[str, Any], key: str, value: Any) -> None:
    parts = str(key).split(".", 1)
    if len(parts) == 2 and parts[0] in {"sensor", "fdtd", "tcad", "hw_isp"}:
        bucket = scenario.setdefault(parts[0], {})
        if not isinstance(bucket, dict):
            bucket = {}
            scenario[parts[0]] = bucket
        if parts[0] == "hw_isp" and parts[1] != "enabled":
            bucket.setdefault("enabled", True)
        bucket[parts[1]] = value
        return
    scenario.setdefault("parameters", {})[key] = value


def _strip_case_report(case: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "parameters": _jsonable(case.get("parameters", {})),
        "score": case.get("score"),
        "feasible": case.get("feasible"),
        "objective_values": _jsonable(case.get("objective_values", {})),
        "objective_utilities": _jsonable(case.get("objective_utilities", {})),
        "constraint_results": _jsonable(case.get("constraint_results", [])),
        "scenario": _jsonable(case.get("scenario", {})),
    }


def _pareto_cases(cases: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    feasible = [case for case in cases if case.get("feasible", True)]
    frontier = []
    for candidate in feasible:
        if not _objective_vector(candidate):
            continue
        dominated = False
        for other in feasible:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda item: (float(item.get("score", -math.inf)), -int(item.get("case_index", 0))),
        reverse=True,
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_vector = _objective_vector(left)
    right_vector = _objective_vector(right)
    if set(left_vector) != set(right_vector) or not left_vector:
        return False
    at_least_equal = all(left_vector[key] >= right_vector[key] for key in left_vector)
    strictly_better = any(left_vector[key] > right_vector[key] for key in left_vector)
    return bool(at_least_equal and strictly_better)


def _objective_vector(case: Mapping[str, Any]) -> dict[str, float]:
    values = dict(case.get("objective_utilities", {}))
    vector: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vector[str(key)] = number
    return vector


def _objective_description(
    objective: ObjectiveSpec | ObjectiveCallable | None, specs: list[dict[str, Any]]
) -> Any:
    if callable(objective):
        return {"callable": getattr(objective, "__name__", "<callable>")}
    return specs


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


cameraE2EOptimizeParameters = camerae2e_optimize_parameters  # noqa: N816
cameraE2EOptimizeCameraParameters = camerae2e_optimize_camera_parameters  # noqa: N816
cameraE2EParetoFront = camerae2e_pareto_front  # noqa: N816
cameraE2EOptimizationReport = camerae2e_optimization_report  # noqa: N816
cameraE2EParameterSpaceCatalog = camerae2e_parameter_space_catalog  # noqa: N816
cameraE2EOptimizationConfigCatalog = camerae2e_optimization_config_catalog  # noqa: N816
cameraE2EParameterSpaceValidate = camerae2e_parameter_space_validate  # noqa: N816
