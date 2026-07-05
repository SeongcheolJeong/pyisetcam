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
    "optics.fnumber": {
        "area": "optics",
        "unit": "f/#",
        "readiness_tier": "validated",
        "values": [2.0, 2.8, 4.0],
        "description": "Optics f-number for irradiance and diffraction/blur tradeoffs.",
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
    "isp": ("ip.demosaic_method",),
    "hw_isp_control": (
        "hw_isp.ae_apply_delay_frames",
        "hw_isp.awb_apply_delay_frames",
        "hw_isp.global_latency_factor",
    ),
    "research_smoke": ("sensor.integration_time", "optics.fnumber"),
    "raw_factory": ("sensor.integration_time", "sensor.analog_gain", "optics.fnumber"),
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
    sensor_allowed = {
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
    }
    for key in axes:
        path = str(key)
        if "." not in path:
            continue
        prefix, remainder = path.split(".", 1)
        if prefix == "sensor" and remainder not in sensor_allowed:
            raise ValueError(f"Unsupported sensor optimization parameter: {path!r}.")
        if prefix == "fdtd" and remainder == "mode" and not _has_nested_value(
            base_scenario, "fdtd", "lut"
        ):
            raise ValueError(
                f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                "optimizing the mode alone would not change the camera pipeline."
            )
        if prefix == "tcad" and remainder == "collection_mode" and not _has_nested_value(
            base_scenario, "tcad", "db"
        ):
            raise ValueError(
                f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                "optimizing the mode alone would not change the camera pipeline."
            )


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
