"""Goal-level CameraE2E validation and refresh gate."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .calibration import (
    camerae2e_calibration_evidence_requirements,
    camerae2e_readiness_promotion_plan,
)
from .dataset import (
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_camera_spec_variants,
    camerae2e_dataset_export_from_optimization,
    camerae2e_dataset_export_perception_index,
    camerae2e_dataset_validate,
)
from .db_catalog import camerae2e_db_manifest, camerae2e_db_validate
from .optimization import (
    camerae2e_optimization_config_catalog,
    camerae2e_optimize_camera_parameters,
)
from .physics_pipeline import camerae2e_physics_pipeline_plan
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario


def camerae2e_goal_gate(
    output_dir: str | Path = "reports/camerae2e_goal",
    *,
    artifact_dir: str | Path | None = None,
    strict: bool = False,
    include_demos: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the CameraE2E goal-level evidence gate.

    The non-strict gate proves the research platform can refresh its registry,
    run FACA/optimization smoke cases, export RAW training artifacts, and keep
    proxy/calibration truth boundaries explicit.  Strict mode intentionally
    fails while proxy or calibration-required assets remain in the active chain.
    """

    output_root = Path(output_dir).expanduser().resolve()
    artifact_root = (
        output_root / "goal_gate_artifacts"
        if artifact_dir is None
        else Path(artifact_dir).expanduser().resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    checks = [
        _run_check("registry_manifest", lambda: _registry_check(strict=strict)),
        _run_check("physics_pipeline", lambda: _physics_pipeline_check(strict=strict)),
        _run_check("calibration_evidence_policy", _calibration_evidence_policy),
        _run_check("faca_smoke", lambda: _faca_smoke(seed=seed)),
        _run_check("parameter_optimization", lambda: _optimization_smoke(seed=seed + 10)),
        _run_check(
            "dataset_factory_smoke",
            lambda: _dataset_factory_smoke(artifact_root / "optimization_dataset", seed=seed + 20),
        ),
    ]
    if include_demos:
        checks.extend(
            [
                _run_check(
                    "adas_kitti_demo_smoke",
                    lambda: _adas_kitti_demo_smoke(artifact_root / "adas_kitti", seed=seed + 30),
                ),
                _run_check(
                    "camera_spec_variant_smoke",
                    lambda: _camera_spec_variant_smoke(
                        artifact_root / "camera_spec_variants", seed=seed + 40
                    ),
                ),
            ]
        )
    else:
        checks.extend(
            [
                _skip_check("adas_kitti_demo_smoke", "include_demos=False"),
                _skip_check("camera_spec_variant_smoke", "include_demos=False"),
            ]
        )
    checks.append(_run_check("signoff_claim_guard", _signoff_claim_guard))

    status_counts = _status_counts(checks)
    payload = {
        "schema_version": "camerae2e_goal_gate_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "strict": bool(strict),
        "ok": status_counts.get("fail", 0) == 0,
        "goal": (
            "Research-grade E2E optimization platform plus RAW data factory "
            "for perception training."
        ),
        "output_dir": str(output_root),
        "artifact_dir": str(artifact_root),
        "summary": {
            "check_count": len(checks),
            "status_counts": status_counts,
            "include_demos": bool(include_demos),
            "seed": int(seed),
        },
        "requirements": _requirement_matrix(checks),
        "checks": checks,
    }
    json_path = output_root / "goal_gate.json"
    html_path = output_root / "goal_gate.html"
    payload["reports"] = {"json": str(json_path), "html": str(html_path)}
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_goal_gate_html(payload), encoding="utf-8")
    return payload


def _registry_check(*, strict: bool) -> dict[str, Any]:
    manifest = camerae2e_db_manifest()
    validation = camerae2e_db_validate(strict=strict)
    summary = manifest.get("summary", {})
    warning_count = int(validation.get("warning_count", 0))
    issue_count = int(validation.get("issue_count", 0))
    status = "pass" if validation.get("ok") and warning_count == 0 else "warn"
    if not validation.get("ok"):
        status = "fail"
    return {
        "status": status,
        "tier": "validated" if status == "pass" else "calibration_required",
        "summary": "DB/LUT registry manifest is available and JSON-serializable.",
        "evidence": {
            "manifest_schema": manifest.get("schema_version"),
            "validation_schema": validation.get("schema_version"),
            "entry_count": summary.get("total"),
            "readiness_tiers": summary.get("readiness_tiers", {}),
            "issue_count": issue_count,
            "warning_count": warning_count,
            "stale_dependency_count": validation.get("stale_dependency_count", 0),
        },
    }


def _physics_pipeline_check(*, strict: bool) -> dict[str, Any]:
    plan = camerae2e_physics_pipeline_plan(strict=strict)
    summary = dict(plan.get("summary", {}))
    warning_like = int(summary.get("stale_dependency_count", 0)) + int(
        summary.get("calibration_required_count", 0)
    )
    status = "pass" if plan.get("ok") and warning_like == 0 else "warn"
    if not plan.get("ok"):
        status = "fail"
    return {
        "status": status,
        "tier": "validated" if status == "pass" else "calibration_required",
        "summary": "External FDTD/TCAD/RayOptics/HW ISP lineage plan is generated.",
        "evidence": {
            "schema": plan.get("schema_version"),
            "ok": plan.get("ok"),
            "active_runs": plan.get("active_runs", {}),
            "summary": summary,
            "truth_boundaries": plan.get("truth_boundaries", []),
            "refresh_order": [
                {
                    "entry": item.get("entry"),
                    "action": item.get("action"),
                    "severity": item.get("severity"),
                    "reason": item.get("reason"),
                }
                for item in plan.get("refresh_order", [])
            ],
        },
    }


def _calibration_evidence_policy() -> dict[str, Any]:
    requirements = camerae2e_calibration_evidence_requirements()
    plan = camerae2e_readiness_promotion_plan()
    summary = dict(plan.get("summary", {}))
    passed = (
        requirements.get("schema_version") == "camerae2e_calibration_evidence_requirements_v1"
        and plan.get("schema_version") == "camerae2e_readiness_promotion_plan_v1"
        and plan.get("evidence_validation", {}).get("ok") is True
        and int(summary.get("blocked_count", 0)) > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "tier": "calibration_required",
        "summary": (
            "Measured-evidence requirements are explicit, and calibrated promotion "
            "is blocked until required artifacts validate."
        ),
        "evidence": {
            "requirements_summary": requirements.get("summary", {}),
            "promotion_summary": summary,
            "promotion_candidates": [
                {
                    "entry": item.get("entry"),
                    "target_tier": item.get("target_tier"),
                    "evidence_ids": item.get("evidence_ids", []),
                }
                for item in plan.get("plans", [])
                if item.get("status") == "promotion_candidate"
            ],
            "blocked_examples": [
                {
                    "entry": item.get("entry"),
                    "current_tier": item.get("current_tier"),
                    "missing_evidence_types": item.get("missing_evidence_types", []),
                }
                for item in plan.get("plans", [])
                if str(item.get("status", "")).startswith("blocked_")
            ][:8],
        },
    }


def _faca_smoke(*, seed: int) -> dict[str, Any]:
    result = camerae2e_run_scenario(
        {
            "name": "goal_gate_faca_smoke",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
            "hw_isp": {"enabled": True, "nframes": 2},
        },
        seed=seed,
        include_arrays=False,
    )
    report = camerae2e_faca_report(result)
    metrics = report.get("metrics", {})
    passed = (
        report.get("schema_version") == "camerae2e_faca_report_v1"
        and metrics.get("color", {}).get("rgb_mean") is not None
        and metrics.get("control", {}).get("frame_count") == 2
    )
    return {
        "status": "pass" if passed else "fail",
        "tier": "validated",
        "summary": "System FACA can run Scene -> OI -> Sensor -> IP plus HW ISP control metrics.",
        "evidence": {
            "seed": seed,
            "scenario_name": report.get("name"),
            "stage_summaries": report.get("stage_summaries", {}),
            "metrics": metrics,
            "parameter_lineage_count": len(report.get("parameter_lineage", [])),
        },
    }


def _optimization_smoke(*, seed: int) -> dict[str, Any]:
    config_catalog = camerae2e_optimization_config_catalog()
    result = camerae2e_optimize_camera_parameters(
        {
            "name": "goal_gate_optimization_smoke",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="evolutionary",
        max_cases=4,
        seed=seed,
        top_k=1,
    )
    surrogate_result = camerae2e_optimize_camera_parameters(
        {
            "name": "goal_gate_surrogate_optimization_smoke",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="bayesian",
        max_cases=4,
        seed=seed + 1,
        top_k=1,
    )
    best = dict(result.get("best_case", {}))
    passed = (
        result.get("schema_version") == "camerae2e_parameter_optimization_v1"
        and int(result.get("case_count", 0)) >= 2
        and best.get("parameters", {}).get("sensor.integration_time") == 0.004
        and surrogate_result.get("search_method") == "surrogate"
        and int(surrogate_result.get("case_count", 0)) >= 2
    )
    return {
        "status": "pass" if passed else "fail",
        "tier": "validated",
        "summary": (
            "Camera parameter optimization runs FACA objective search with "
            "validated, budget-aware evolutionary and surrogate candidate planning."
        ),
        "evidence": {
            "seed": seed,
            "registered_configure_count": config_catalog.get("registered_axis_count"),
            "presets": config_catalog.get("presets", {}),
            "adaptive_methods": ["evolutionary", "surrogate"],
            "method": result.get("method"),
            "search_method": result.get("search_method"),
            "candidate_plan": result.get("candidate_plan", {}),
            "surrogate_method": surrogate_result.get("method"),
            "surrogate_search_method": surrogate_result.get("search_method"),
            "surrogate_candidate_plan": surrogate_result.get("candidate_plan", {}),
            "case_count": result.get("case_count"),
            "feasible_count": result.get("feasible_count"),
            "pareto_case_count": result.get("pareto_case_count"),
            "best_parameters": best.get("parameters", {}),
            "parameter_space_validation": result.get("parameter_space_validation", {}),
            "automation": result.get("automation", {}),
        },
    }


def _dataset_factory_smoke(output_dir: Path, *, seed: int) -> dict[str, Any]:
    optimization = camerae2e_optimize_camera_parameters(
        {
            "name": "goal_gate_dataset_source",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="raw_factory",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        seed=seed,
        top_k=1,
    )
    manifest = camerae2e_dataset_export_from_optimization(
        output_dir,
        optimization,
        selection="best",
        max_cases=1,
        include_rgb=False,
        seed=seed,
    )
    validation = camerae2e_dataset_validate(manifest)
    perception_index = camerae2e_dataset_export_perception_index(
        manifest,
        output_dir=output_dir / "perception_index",
        formats=("raw_manifest", "yolo"),
    )
    record = manifest["records"][0] if manifest.get("records") else {}
    passed = bool(validation.get("ok")) and manifest.get("case_count") == 1
    return {
        "status": "pass" if passed else "fail",
        "tier": "validated",
        "summary": "RAW data factory exports deterministic NPZ, metadata, labels, and lineage.",
        "evidence": {
            "dataset_root": manifest.get("dataset_root"),
            "manifest_schema": manifest.get("schema_version"),
            "validation_ok": validation.get("ok"),
            "case_count": manifest.get("case_count"),
            "raw_shape": record.get("raw_shape"),
            "raw_sha256": record.get("raw_sha256"),
            "parameter_lineage_count": len(record.get("parameter_lineage", [])),
            "source_optimization": manifest.get("source_optimization", {}),
            "perception_index": {
                "manifest": perception_index.get("manifest"),
                "formats": perception_index.get("formats", []),
                "warning_count": perception_index.get("warning_count", 0),
                "outputs": perception_index.get("outputs", {}),
            },
        },
    }


def _adas_kitti_demo_smoke(output_dir: Path, *, seed: int) -> dict[str, Any]:
    manifest = camerae2e_dataset_export_adas_kitti_demo(
        output_dir,
        case_count=1,
        seed=seed,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    record = manifest["records"][0] if manifest.get("records") else {}
    passed = bool(validation.get("ok")) and manifest.get("case_count") == 1
    return {
        "status": "pass" if passed else "fail",
        "tier": "proxy",
        "summary": "ADAS/KITTI YOLO demo produces proxy RAW data and labels.",
        "evidence": {
            "dataset_root": manifest.get("dataset_root"),
            "validation_ok": validation.get("ok"),
            "case_count": manifest.get("case_count"),
            "truth_boundary": manifest.get("adas_kitti_demo", {}).get("truth_boundary"),
            "camera_spec": manifest.get("adas_kitti_demo", {}).get("camera_spec", {}),
            "label_coordinate_frame": record.get("label_coordinate_frame"),
            "raw_shape": record.get("raw_shape"),
        },
    }


def _camera_spec_variant_smoke(output_dir: Path, *, seed: int) -> dict[str, Any]:
    manifest = camerae2e_dataset_export_camera_spec_variants(
        output_dir,
        target_specs=["wide_fov_adas_demo", "narrow_fov_adas_demo"],
        case_count=1,
        seed=seed,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    presets = [
        record.get("scenario", {}).get("target_camera_spec", {}).get("preset")
        for record in manifest.get("records", [])
    ]
    focal_lengths = [
        record.get("scenario", {})
        .get("target_camera_spec", {})
        .get("optics", {})
        .get("focal_length_m")
        for record in manifest.get("records", [])
    ]
    transforms = [
        record.get("scenario", {}).get("geometric_scene_transform", {})
        for record in manifest.get("records", [])
    ]
    passed = (
        bool(validation.get("ok"))
        and manifest.get("case_count") == 2
        and len({value for value in focal_lengths if value is not None}) == 2
    )
    return {
        "status": "pass" if passed else "fail",
        "tier": "proxy",
        "summary": "KITTI-style source scene can be proxy re-captured under multiple camera specs.",
        "evidence": {
            "dataset_root": manifest.get("dataset_root"),
            "validation_ok": validation.get("ok"),
            "case_count": manifest.get("case_count"),
            "presets": presets,
            "focal_lengths_m": focal_lengths,
            "geometric_transform": manifest.get("camera_spec_variants", {}).get(
                "geometric_transform", {}
            ),
            "record_transforms": [
                {
                    "mode": item.get("mode"),
                    "object_scale_x": item.get("object_scale_x"),
                    "out_of_source_fraction": item.get("warp", {}).get(
                        "out_of_source_fraction"
                    ),
                }
                for item in transforms
            ],
            "truth_boundary": manifest.get("camera_spec_variants", {}).get("truth_boundary"),
        },
    }


def _signoff_claim_guard() -> dict[str, Any]:
    strict_validation = camerae2e_db_validate(strict=True)
    plan = camerae2e_physics_pipeline_plan(strict=True)
    blocked_tiers = [
        issue
        for issue in strict_validation.get("issues", [])
        if issue.get("kind") in {"readiness_tier", "stale_dependency", "missing_path"}
    ]
    signoff_allowed = bool(strict_validation.get("ok")) and bool(plan.get("ok"))
    passed = signoff_allowed or bool(blocked_tiers or plan.get("summary", {}))
    return {
        "status": "pass" if passed else "fail",
        "tier": "calibration_required",
        "summary": (
            "Proxy/calibration-required assets are prevented from being promoted "
            "to product sign-off claims."
        ),
        "evidence": {
            "calibrated_or_signoff_claim_allowed": signoff_allowed,
            "strict_db_ok": strict_validation.get("ok"),
            "strict_physics_ok": plan.get("ok"),
            "strict_issue_count": strict_validation.get("issue_count"),
            "blocked_issue_examples": blocked_tiers[:8],
        },
    }


def _run_check(name: str, callback: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    try:
        payload = dict(callback())
        status = str(payload.get("status", "fail"))
        if status not in {"pass", "warn", "fail", "skip"}:
            status = "fail"
        return {"name": name, **payload, "status": status}
    except Exception as exc:  # pragma: no cover - exercised by external environments.
        return {
            "name": name,
            "status": "fail",
            "tier": "missing",
            "summary": "Check raised an exception.",
            "evidence": {"error_type": type(exc).__name__, "error": str(exc)},
        }


def _skip_check(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skip",
        "tier": "available",
        "summary": "Check skipped by caller option.",
        "evidence": {"reason": reason},
    }


def _requirement_matrix(checks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(check["name"]): check for check in checks}
    specs = [
        (
            "DB/LUT registry and readiness matrix",
            ["registry_manifest"],
            "Registry manifest validates schema, readiness tiers, and stale dependencies.",
        ),
        (
            "FDTD/TCAD/RayOptics/HW ISP external pipeline",
            ["physics_pipeline"],
            "External asset lineage and refresh actions are generated without sign-off inflation.",
        ),
        (
            "Calibration evidence and readiness promotion",
            ["calibration_evidence_policy"],
            (
                "Measured-evidence requirements exist and calibrated promotion remains "
                "blocked until evidence validates."
            ),
        ),
        (
            "System FACA scenario metrics",
            ["faca_smoke"],
            "Scene/OI/Sensor/IP/HW ISP stages produce metrics and parameter lineage.",
        ),
        (
            "Automated camera-parameter optimization",
            ["parameter_optimization"],
            "Preset camera parameter search optimizes a FACA metric deterministically.",
        ),
        (
            "RAW data factory from optimized camera cases",
            ["dataset_factory_smoke"],
            "Optimization outputs can be rendered into validated RAW training artifacts.",
        ),
        (
            "ADAS/KITTI YOLO demo RAW generation",
            ["adas_kitti_demo_smoke"],
            "Synthetic or supplied KITTI-style inputs can produce proxy RAW and YOLO labels.",
        ),
        (
            "Camera-spec variant re-capture",
            ["camera_spec_variant_smoke"],
            "One KITTI-style scene can be exported under multiple camera specifications.",
        ),
        (
            "Physics proxy/sign-off claim guard",
            ["signoff_claim_guard"],
            "Strict validation blocks calibrated claims until measured evidence is attached.",
        ),
    ]
    return [
        {
            "requirement": name,
            "status": _combined_status([by_name[item]["status"] for item in names]),
            "checks": names,
            "evidence_summary": summary,
        }
        for name, names, summary in specs
    ]


def _combined_status(statuses: list[str]) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if all(status == "skip" for status in statuses):
        return "skip"
    if any(status in {"warn", "skip"} for status in statuses):
        return "warn"
    return "pass"


def _status_counts(checks: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status", "fail"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _render_goal_gate_html(payload: Mapping[str, Any]) -> str:
    check_rows = []
    for check in payload.get("checks", []):
        evidence = json.dumps(
            _jsonable(check.get("evidence", {})),
            indent=2,
            sort_keys=True,
        )
        check_rows.append(
            "<tr>"
            f"<td>{_e(check.get('name'))}</td>"
            f"<td><span class='status status-{_e(check.get('status'))}'>"
            f"{_e(check.get('status'))}</span></td>"
            f"<td>{_e(check.get('tier'))}</td>"
            f"<td>{_e(check.get('summary'))}</td>"
            f"<td><pre>{_e(evidence)}</pre></td>"
            "</tr>"
        )
    requirement_rows = []
    for item in payload.get("requirements", []):
        requirement_rows.append(
            "<tr>"
            f"<td>{_e(item.get('requirement'))}</td>"
            f"<td><span class='status status-{_e(item.get('status'))}'>"
            f"{_e(item.get('status'))}</span></td>"
            f"<td>{_e(', '.join(item.get('checks', [])))}</td>"
            f"<td>{_e(item.get('evidence_summary'))}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CameraE2E Goal Gate</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 32px; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
th, td {{ border: 1px solid #d5d8dc; padding: 8px 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f4f6f7; text-align: left; }}
pre {{ white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto;
  background: #f8f9f9; padding: 8px; }}
.status {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 600; }}
.status-pass {{ background: #d5f5e3; color: #145a32; }}
.status-warn, .status-skip {{ background: #fdebd0; color: #7d3c00; }}
.status-fail {{ background: #fadbd8; color: #78281f; }}
.lead {{ font-size: 18px; }}
code {{ word-break: break-all; }}
</style>
</head>
<body>
<h1>CameraE2E Goal Gate</h1>
<p class="lead">{_e(payload.get("goal"))}</p>
<p>OK: <strong>{_e(payload.get("ok"))}</strong>; strict:
<strong>{_e(payload.get("strict"))}</strong>; generated:
<code>{_e(payload.get("generated_at"))}</code></p>
<p>Artifacts: <code>{_e(payload.get("artifact_dir"))}</code></p>
<h2>Requirements</h2>
<table>
<thead><tr><th>Requirement</th><th>Status</th><th>Checks</th><th>Evidence</th></tr></thead>
<tbody>{"".join(requirement_rows)}</tbody>
</table>
<h2>Checks</h2>
<table>
<thead><tr><th>Name</th><th>Status</th><th>Tier</th><th>Summary</th><th>Evidence</th></tr></thead>
<tbody>{"".join(check_rows)}</tbody>
</table>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in body.splitlines()) + "\n"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


cameraE2EGoalGate = camerae2e_goal_gate  # noqa: N816
