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
cameraE2EParetoFront = camerae2e_pareto_front  # noqa: N816
cameraE2EOptimizationReport = camerae2e_optimization_report  # noqa: N816
