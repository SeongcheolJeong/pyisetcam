"""Generate a machine-readable parity report against stored baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "tests" / "parity" / "cases.yaml"
BASELINES_DIR = REPO_ROOT / "tests" / "parity" / "baselines"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "parity" / "latest.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore, run_python_case


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def _load_reference(case_name: str) -> dict[str, Any]:
    baseline_path = BASELINES_DIR / f"{case_name}.mat"
    return {
        key: value
        for key, value in loadmat(baseline_path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        array = np.squeeze(np.asarray(value))
        if np.iscomplexobj(array):
            array = np.real(array)
        return array
    if np.isscalar(value) and np.iscomplexobj(value):
        return float(np.real(value))
    return value


def _array_metrics(reference: Any, actual: Any) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    difference = np.abs(actual_array - reference_array)
    denominator = np.maximum(np.abs(reference_array), 1e-12)
    relative = difference / denominator
    metrics = {
        "shape": list(reference_array.shape),
        "mae": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(actual_array - reference_array)))),
        "max_abs": float(np.max(difference)),
        "mean_rel": float(np.mean(relative)),
        "max_rel": float(np.max(relative)),
    }
    max_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(relative)), relative.shape))
    metrics["max_index"] = list(max_index)
    metrics["max_index_ref"] = float(reference_array[max_index])
    metrics["max_index_actual"] = float(actual_array[max_index])
    metrics["max_index_abs"] = float(difference[max_index])
    metrics["max_index_rel"] = float(relative[max_index])

    if reference_array.ndim >= 2:
        border_rows = max(1, int(round(reference_array.shape[0] / 10.0)))
        border_cols = max(1, int(round(reference_array.shape[1] / 10.0)))
        spatial_mask = np.zeros(reference_array.shape[:2], dtype=bool)
        spatial_mask[:border_rows, :] = True
        spatial_mask[-border_rows:, :] = True
        spatial_mask[:, :border_cols] = True
        spatial_mask[:, -border_cols:] = True
        if reference_array.ndim > 2:
            expand = (slice(None), slice(None), *([None] * (reference_array.ndim - 2)))
            border_mask = np.broadcast_to(spatial_mask[expand], reference_array.shape)
        else:
            border_mask = spatial_mask
        interior_mask = ~border_mask
        metrics["border_rows"] = border_rows
        metrics["border_cols"] = border_cols
        metrics["edge_mean_rel"] = float(np.mean(relative[border_mask]))
        if np.any(interior_mask):
            metrics["interior_mean_rel"] = float(np.mean(relative[interior_mask]))

    if reference_array.ndim >= 2 and reference_array.shape[0] >= 2 and reference_array.shape[1] >= 2:
        phase_metrics: dict[str, float] = {}
        row_index, col_index = np.indices(reference_array.shape[:2])
        for row_phase in (0, 1):
            for col_phase in (0, 1):
                phase_mask = (row_index % 2 == row_phase) & (col_index % 2 == col_phase)
                if reference_array.ndim > 2:
                    expand = (slice(None), slice(None), *([None] * (reference_array.ndim - 2)))
                    phase_mask = np.broadcast_to(phase_mask[expand], reference_array.shape)
                phase_metrics[f"r{row_phase}c{col_phase}"] = float(np.mean(relative[phase_mask]))
        metrics["phase_2x2_mean_rel"] = phase_metrics

    return metrics


def _compare(reference: Any, actual: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    normalized_reference = _normalize(reference)
    normalized_actual = _normalize(actual)

    if isinstance(normalized_reference, dict):
        keys = sorted(key for key in normalized_reference if not key.startswith("__"))
        fields: dict[str, Any] = {}
        missing = [key for key in keys if key not in normalized_actual]
        if missing:
            return {"pass": False, "missing_keys": missing}
        overall = True
        for key in keys:
            field_report = _compare(
                normalized_reference[key],
                normalized_actual[key],
                rtol=rtol,
                atol=atol,
            )
            fields[key] = field_report
            overall = overall and bool(field_report["pass"])
        return {"pass": overall, "fields": fields}

    if isinstance(normalized_reference, (str, bytes)) or isinstance(normalized_actual, (str, bytes)):
        passed = normalized_actual == normalized_reference
        return {
            "pass": passed,
            "expected": normalized_reference,
            "actual": normalized_actual,
        }

    if np.isscalar(normalized_reference) and np.isscalar(normalized_actual):
        reference_scalar = float(normalized_reference)
        actual_scalar = float(normalized_actual)
        difference = abs(actual_scalar - reference_scalar)
        passed = bool(np.isclose(actual_scalar, reference_scalar, rtol=rtol, atol=atol))
        return {
            "pass": passed,
            "expected": reference_scalar,
            "actual": actual_scalar,
            "abs_diff": difference,
            "rel_diff": float(difference / max(abs(reference_scalar), 1e-12)),
        }

    metrics = _array_metrics(normalized_reference, normalized_actual)
    passed = bool(
        np.allclose(
            np.asarray(normalized_actual, dtype=float),
            np.asarray(normalized_reference, dtype=float),
            rtol=rtol,
            atol=atol,
        )
    )
    return {"pass": passed, **metrics}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_report(*, asset_store: AssetStore | None = None) -> dict[str, Any]:
    store = asset_store or AssetStore.default()
    cases = _case_definitions()
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped = 0

    for case in cases:
        case_name = str(case["name"])
        baseline_path = BASELINES_DIR / f"{case_name}.mat"
        if not baseline_path.exists():
            skipped += 1
            results.append(
                {
                    "name": case_name,
                    "status": "skipped",
                    "reason": f"Missing baseline: {baseline_path}",
                }
            )
            continue

        comparison = _compare(
            _load_reference(case_name),
            run_python_case(case_name, asset_store=store),
            rtol=float(case["rtol"]),
            atol=float(case["atol"]),
        )
        status = "passed" if comparison["pass"] else "failed"
        if comparison["pass"]:
            passed += 1
        else:
            failed += 1
        results.append(
            {
                "name": case_name,
                "status": status,
                "rtol": float(case["rtol"]),
                "atol": float(case["atol"]),
                "comparison": comparison,
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(cases),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON path. Defaults to reports/parity/latest.json",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when any parity case fails.",
    )
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        f"Parity report written to {args.output} "
        f"({summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped)."
    )
    return 1 if args.fail_on_mismatch and int(summary["failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
