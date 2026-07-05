"""System-level Field/Angle/Color/Artifact/Control Analysis helpers."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_create, camera_get
from .db_catalog import camerae2e_db_lineage, camerae2e_db_summary, camerae2e_db_validate
from .fdtd_sensor import sensor_attach_fdtd_lut
from .hwisp import HWIspConfig, hw_isp_config, hw_isp_simulate_sequence
from .ip import ip_get
from .scene import scene_create, scene_get
from .sensor import sensor_get, sensor_set
from .tcad_sensor import sensor_attach_tcad_lut
from .types import Camera, Scene


def camerae2e_run_scenario(
    scenario: Mapping[str, Any] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Run one CameraE2E FACA scenario and collect stage evidence."""

    store = asset_store or AssetStore.default()
    config = dict(scenario or {})
    scenario_name = str(config.get("name", "camerae2e_faca_scenario"))
    resolved_scene = _resolve_scene(scene if scene is not None else config.get("scene"), store)
    resolved_camera = _resolve_camera(camera if camera is not None else config.get("camera"), store)
    resolved_camera = _apply_physics_and_sensor_overrides(resolved_camera, config)

    hw_payload = dict(config.get("hw_isp", {})) if isinstance(config.get("hw_isp"), Mapping) else {}
    use_hw_isp = bool(hw_payload.get("enabled", False))
    hw_sequence = None
    if use_hw_isp:
        hw_config = _resolve_hw_config(hw_payload.get("config", hw_payload))
        nframes = int(hw_payload.get("nframes", 3))
        hw_sequence = hw_isp_simulate_sequence(
            resolved_camera,
            resolved_scene,
            hw_config,
            nframes=nframes,
            asset_store=store,
        )
        computed_camera = hw_sequence.frames[-1].camera
    else:
        computed_camera = camera_compute(resolved_camera.clone(), resolved_scene, asset_store=store)

    stages = _collect_stages(resolved_scene, computed_camera, include_arrays=include_arrays)
    metrics = _faca_metrics(stages, hw_sequence.aggregate if hw_sequence is not None else None)
    return {
        "schema_version": "camerae2e_faca_scenario_v1",
        "name": scenario_name,
        "seed": int(seed),
        "scenario": _jsonable(_scenario_without_objects(config)),
        "camera": computed_camera,
        "hw_isp_sequence": hw_sequence,
        "stages": stages,
        "metrics": metrics,
        "artifact_lineage": _artifact_lineage_summary(),
    }


def camerae2e_run_sweep(
    base_scenario: Mapping[str, Any] | None = None,
    sweep_axes: Mapping[str, Iterable[Any]] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Run a deterministic Cartesian FACA sweep."""

    axes = {str(key): list(values) for key, values in dict(sweep_axes or {}).items()}
    if not axes:
        result = camerae2e_run_scenario(
            base_scenario,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=seed,
            include_arrays=include_arrays,
        )
        return {
            "schema_version": "camerae2e_faca_sweep_v1",
            "seed": int(seed),
            "axes": {},
            "cases": [camerae2e_faca_report(result)],
        }

    cases = []
    keys = list(axes)
    for index, values in enumerate(itertools.product(*(axes[key] for key in keys))):
        scenario = _deep_dict(base_scenario or {})
        scenario["name"] = str(scenario.get("name", "camerae2e_faca_sweep"))
        axis_payload = {}
        for key, value in zip(keys, values, strict=True):
            _assign_axis_value(scenario, key, value)
            axis_payload[key] = value
        result = camerae2e_run_scenario(
            scenario,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=int(seed) + index,
            include_arrays=include_arrays,
        )
        report = camerae2e_faca_report(result)
        report["axis_values"] = _jsonable(axis_payload)
        cases.append(report)

    return {
        "schema_version": "camerae2e_faca_sweep_v1",
        "seed": int(seed),
        "axes": _jsonable(axes),
        "case_count": len(cases),
        "cases": cases,
    }


def camerae2e_faca_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe FACA report for a scenario or sweep case."""

    if result.get("schema_version") == "camerae2e_faca_sweep_v1":
        return _jsonable(result)
    return {
        "schema_version": "camerae2e_faca_report_v1",
        "name": str(result.get("name", "camerae2e_faca_scenario")),
        "seed": int(result.get("seed", 0)),
        "scenario": _jsonable(result.get("scenario", {})),
        "stage_summaries": {
            str(name): _jsonable({key: value for key, value in payload.items() if key != "array"})
            for name, payload in dict(result.get("stages", {})).items()
        },
        "metrics": _jsonable(result.get("metrics", {})),
        "artifact_lineage": _jsonable(result.get("artifact_lineage", {})),
    }


def _resolve_scene(source: Scene | str | Mapping[str, Any] | None, store: AssetStore) -> Scene:
    if isinstance(source, Scene):
        return source.clone()
    if isinstance(source, Mapping):
        scene_type = str(source.get("type", source.get("name", "uniform ee")))
        args = list(source.get("args", [16]))
        kwargs = dict(source.get("kwargs", {}))
        kwargs.setdefault("asset_store", store)
        return scene_create(scene_type, *args, **kwargs)
    if isinstance(source, str):
        return scene_create(source, asset_store=store)
    return scene_create("uniform ee", 16, asset_store=store)


def _resolve_camera(source: Camera | Mapping[str, Any] | None, store: AssetStore) -> Camera:
    if isinstance(source, Camera):
        return source.clone()
    if isinstance(source, Mapping):
        camera_type = str(source.get("type", source.get("name", "default")))
        args = list(source.get("args", []))
        created = camera_create(camera_type, *args, asset_store=store)
    else:
        created = camera_create(asset_store=store)
    if isinstance(created, list):
        if not created:
            raise ValueError("camera_create returned an empty camera list.")
        return created[0].clone()
    return created.clone()


def _apply_physics_and_sensor_overrides(camera: Camera, config: Mapping[str, Any]) -> Camera:
    updated = camera.clone()
    sensor = camera_get(updated, "sensor").clone()
    sensor_overrides = (
        dict(config.get("sensor", {})) if isinstance(config.get("sensor"), Mapping) else {}
    )
    for key, value in sensor_overrides.items():
        normalized = str(key).replace("_", " ").lower()
        if normalized in {"noise flag", "noise"}:
            sensor = sensor_set(sensor, "noise flag", value)
        elif normalized in {"integration time", "integration"}:
            sensor = sensor_set(sensor, "integration time", value)
        elif normalized in {"exposure duration", "exposure time", "exposure time s"}:
            sensor = sensor_set(sensor, "exposure duration", value)
        elif normalized in {"analog gain", "gain"}:
            sensor = sensor_set(sensor, "analog gain", value)

    fdtd = dict(config.get("fdtd", {})) if isinstance(config.get("fdtd"), Mapping) else {}
    if fdtd.get("lut") is not None:
        fdtd_kwargs = {key: value for key, value in fdtd.items() if key not in {"lut", "enabled"}}
        sensor = sensor_attach_fdtd_lut(sensor, fdtd["lut"], **fdtd_kwargs)

    tcad = dict(config.get("tcad", {})) if isinstance(config.get("tcad"), Mapping) else {}
    if tcad.get("db") is not None:
        tcad_kwargs = {key: value for key, value in tcad.items() if key not in {"db", "enabled"}}
        sensor = sensor_attach_tcad_lut(sensor, tcad["db"], **tcad_kwargs)

    updated.fields["sensor"] = sensor
    return updated


def _resolve_hw_config(raw: Any) -> HWIspConfig:
    if isinstance(raw, HWIspConfig):
        return raw
    if isinstance(raw, Mapping):
        return hw_isp_config(
            **{
                key: value
                for key, value in raw.items()
                if key not in {"enabled", "nframes", "config"}
            }
        )
    return hw_isp_config()


def _collect_stages(
    scene: Scene, camera: Camera, *, include_arrays: bool
) -> dict[str, dict[str, Any]]:
    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    ip = camera_get(camera, "ip")
    return {
        "scene_photons": _stage_payload(
            _safe_get(lambda: scene_get(scene, "photons")), include_arrays=include_arrays
        ),
        "oi_photons": _stage_payload(oi.data.get("photons"), include_arrays=include_arrays),
        "sensor_raw": _stage_payload(_sensor_raw(sensor), include_arrays=include_arrays),
        "sensor_digital": _stage_payload(_sensor_digital(sensor), include_arrays=include_arrays),
        "ip_result": _stage_payload(
            _safe_get(lambda: ip_get(ip, "result")), include_arrays=include_arrays
        ),
        "ip_srgb": _stage_payload(
            _safe_get(lambda: ip_get(ip, "srgb")), include_arrays=include_arrays
        ),
    }


def _sensor_raw(sensor: Any) -> Any:
    for key in ("volts", "dv", "electrons"):
        if key in getattr(sensor, "data", {}):
            return sensor.data[key]
    try:
        return sensor_get(sensor, "volts")
    except Exception:
        return None


def _sensor_digital(sensor: Any) -> Any:
    for key in ("digital_values", "dv"):
        if key in getattr(sensor, "data", {}):
            return sensor.data[key]
    try:
        return sensor_get(sensor, "digital values")
    except Exception:
        return None


def _stage_payload(values: Any, *, include_arrays: bool) -> dict[str, Any]:
    if values is None:
        return {"available": False, "shape": [], "dtype": None}
    array = np.asarray(values)
    payload: dict[str, Any] = {
        "available": bool(array.size),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size:
        finite = np.asarray(array, dtype=float)
        payload.update(
            {
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "std": float(np.nanstd(finite)),
            }
        )
    if include_arrays:
        payload["array"] = array.copy()
    return payload


def _faca_metrics(
    stages: Mapping[str, Mapping[str, Any]], hw_aggregate: Mapping[str, Any] | None
) -> dict[str, Any]:
    raw = dict(stages.get("sensor_raw", {}))
    rgb = dict(stages.get("ip_result", {}))
    rgb_array = np.asarray(rgb.get("array", np.empty(0)), dtype=float)
    raw_array = np.asarray(raw.get("array", np.empty(0)), dtype=float)
    raw_std = float(np.nanstd(raw_array)) if raw_array.size else raw.get("std")
    rgb_std = float(np.nanstd(rgb_array)) if rgb_array.size else rgb.get("std")
    rgb_mean = float(np.nanmean(rgb_array)) if rgb_array.size else rgb.get("mean")
    metrics: dict[str, Any] = {
        "field": {"available": bool(stages.get("oi_photons", {}).get("available", False))},
        "angle": {
            "available": False,
            "note": "CRA/field axes are recorded through attached FDTD scenario parameters.",
        },
        "color": {"rgb_mean": None if rgb_mean is None else float(rgb_mean)},
        "artifact": {
            "raw_std": None if raw_std is None else float(raw_std),
            "rgb_std": None if rgb_std is None else float(rgb_std),
            "rgb_clip_fraction": _clip_fraction(rgb_array) if rgb_array.size else None,
        },
        "control": dict(hw_aggregate or {}),
    }
    return metrics


def _clip_fraction(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean((values <= 0.0) | (values >= 1.0)))


def _artifact_lineage_summary() -> dict[str, Any]:
    validation = camerae2e_db_validate()
    summary = camerae2e_db_summary()
    lineage = {}
    for name in (
        "fdtd_sensor_lut_active",
        "tcad_sensor_db_active",
        "lens_patents_active",
        "hwisp_parameter_profiles",
    ):
        try:
            lineage[name] = camerae2e_db_lineage(name)
        except Exception as exc:
            lineage[name] = {"error": str(exc)}
    return {"summary": summary, "validation": validation, "lineage": lineage}


def _assign_axis_value(scenario: dict[str, Any], key: str, value: Any) -> None:
    parts = str(key).split(".", 1)
    if len(parts) == 2 and parts[0] in {"sensor", "fdtd", "tcad", "hw_isp"}:
        bucket = scenario.setdefault(parts[0], {})
        if not isinstance(bucket, dict):
            bucket = {}
            scenario[parts[0]] = bucket
        bucket[parts[1]] = value
        return
    scenario.setdefault("parameters", {})[key] = value


def _scenario_without_objects(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in config.items():
        if isinstance(value, (Camera, Scene)):
            payload[key] = {"type": type(value).__name__, "name": value.name}
        else:
            payload[key] = value
    return payload


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _safe_get(callback: Any) -> Any:
    try:
        return callback()
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if key not in {"array", "camera", "hw_isp_sequence"}
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


cameraE2ERunScenario = camerae2e_run_scenario  # noqa: N816
cameraE2ERunSweep = camerae2e_run_sweep  # noqa: N816
cameraE2EFACAReport = camerae2e_faca_report  # noqa: N816
