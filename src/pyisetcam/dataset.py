"""CameraE2E RAW dataset export helpers."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import imageio.v3 as iio
import numpy as np
import tifffile

from .assets import AssetStore
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario
from .types import Camera, Scene


def camerae2e_dataset_export(
    output_dir: str | Path,
    scenarios: Iterable[Mapping[str, Any]] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    include_rgb: bool = True,
    include_tiff: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Export reproducible CameraE2E RAW/stage data for perception training.

    The v1 writer intentionally does not emit DNG.  It writes raw arrays and
    metadata in simple, inspectable formats so downstream training code can
    choose its own packing and camera-specific metadata policy.
    """

    root = Path(output_dir).expanduser()
    raw_dir = root / "raw"
    rgb_dir = root / "rgb"
    label_dir = root / "labels"
    tiff_dir = root / "raw_tiff"
    for directory in (raw_dir, rgb_dir, label_dir, tiff_dir if include_tiff else None):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    scenario_list = list(scenarios or [{"name": "camerae2e_dataset_case"}])
    label_list = _normalize_labels(labels, len(scenario_list))
    split_list = _normalize_splits(split, len(scenario_list))
    records: list[dict[str, Any]] = []
    metadata_path = root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_stream:
        for index, scenario_config in enumerate(scenario_list):
            case_id = f"case_{index:04d}"
            result = camerae2e_run_scenario(
                scenario_config,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=int(seed) + index,
                include_arrays=True,
            )
            raw_array = _array_from_stage(result, "sensor_raw")
            rgb_array = _array_from_stage(result, "ip_srgb")
            if rgb_array.size == 0:
                rgb_array = _array_from_stage(result, "ip_result")

            raw_path = raw_dir / f"{case_id}.npz"
            sensor_digital_array = _array_from_stage(result, "sensor_digital")
            _write_npz_deterministic(
                raw_path,
                raw=raw_array,
                sensor_digital=sensor_digital_array,
                seed=np.asarray([int(seed) + index], dtype=np.int64),
            )

            rgb_path = None
            if include_rgb and rgb_array.size:
                rgb_path = rgb_dir / f"{case_id}.png"
                iio.imwrite(rgb_path, _uint8_preview(rgb_array))

            tiff_path = None
            if include_tiff and raw_array.size:
                tiff_path = tiff_dir / f"{case_id}.tiff"
                tifffile.imwrite(tiff_path, raw_array.astype(np.float32), photometric="minisblack")

            label_payload = {
                "schema_version": "camerae2e_dataset_labels_v1",
                "case_id": case_id,
                "labels": label_list[index],
                "note": (
                    "Labels are caller-provided or empty; "
                    "CameraE2E does not infer labels in v1."
                ),
            }
            label_path = label_dir / f"{case_id}.json"
            label_path.write_text(
                json.dumps(_jsonable(label_payload), indent=2, sort_keys=True), encoding="utf-8"
            )

            report = camerae2e_faca_report(result)
            record = {
                "case_id": case_id,
                "split": split_list[index],
                "seed": int(seed) + index,
                "scenario_name": report["name"],
                "raw": str(raw_path),
                "rgb": None if rgb_path is None else str(rgb_path),
                "raw_tiff": None if tiff_path is None else str(tiff_path),
                "labels": str(label_path),
                "raw_shape": list(raw_array.shape),
                "raw_dtype": str(raw_array.dtype),
                "rgb_shape": list(rgb_array.shape),
                "rgb_dtype": str(rgb_array.dtype) if rgb_array.size else None,
                "sensor_digital_shape": list(sensor_digital_array.shape),
                "sensor_digital_dtype": str(sensor_digital_array.dtype),
                "raw_sha256": _sha256_file(raw_path),
                "raw_content_sha256": _sha256_array(raw_array),
                "sensor_digital_content_sha256": _sha256_array(sensor_digital_array),
                "rgb_sha256": None if rgb_path is None else _sha256_file(rgb_path),
                "raw_tiff_sha256": None if tiff_path is None else _sha256_file(tiff_path),
                "labels_sha256": _sha256_file(label_path),
                "scenario": report.get("scenario", {}),
                "stage_summaries": report.get("stage_summaries", {}),
                "faca_metrics": report.get("metrics", {}),
            }
            records.append(record)
            metadata_stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    metadata_sha256 = _sha256_file(metadata_path)
    split_counts: dict[str, int] = {}
    for name in split_list:
        split_counts[name] = split_counts.get(name, 0) + 1
    manifest = {
        "schema_version": "camerae2e_dataset_manifest_v1",
        "dataset_root": str(root),
        "seed": int(seed),
        "case_count": len(records),
        "splits": split_counts,
        "format": {
            "raw": "deterministic compressed NumPy .npz with raw and sensor_digital arrays",
            "rgb": "8-bit PNG preview when include_rgb=True",
            "raw_tiff": "float32 TIFF when include_tiff=True",
            "labels": "JSON labels supplied by caller; empty by default",
            "dng": "not emitted in v1",
        },
        "integrity": {
            "hash": "sha256",
            "raw_npz": "deterministic zip container with fixed member timestamps",
            "metadata_jsonl_sha256": metadata_sha256,
        },
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def camerae2e_dataset_export_from_optimization(
    output_dir: str | Path,
    optimization_result: Mapping[str, Any],
    *,
    selection: str = "pareto",
    max_cases: int | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int | None = None,
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    include_rgb: bool = True,
    include_tiff: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Export RAW data from selected optimization cases.

    This bridges the E2E optimizer and RAW data factory: best/top/Pareto
    camera-parameter candidates can be rendered into training-ready RAW
    artifacts without manually copying scenario dictionaries.
    """

    selected_cases = _optimization_cases(optimization_result, selection=selection)
    if max_cases is not None:
        selected_cases = selected_cases[: max(int(max_cases), 0)]
    scenarios = [
        _scenario_from_optimization_case(case, index)
        for index, case in enumerate(selected_cases)
    ]
    effective_seed = int(seed if seed is not None else optimization_result.get("seed", 0))
    effective_split = split if split is not None else str(selection)
    manifest = camerae2e_dataset_export(
        output_dir,
        scenarios,
        scene=scene,
        camera=camera,
        asset_store=asset_store,
        seed=effective_seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        split=effective_split,
    )
    manifest["source_optimization"] = {
        "schema_version": "camerae2e_dataset_source_optimization_v1",
        "selection": str(selection),
        "source_schema_version": optimization_result.get("schema_version"),
        "source_seed": optimization_result.get("seed"),
        "method": optimization_result.get("method"),
        "objective": _jsonable(optimization_result.get("objective", {})),
        "constraints": _jsonable(optimization_result.get("constraints", [])),
        "selected_case_count": len(selected_cases),
        "selected_cases": [_optimization_case_summary(case) for case in selected_cases],
    }
    manifest_path = Path(manifest["dataset_root"]) / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def camerae2e_dataset_validate(
    dataset: str | Path | Mapping[str, Any], *, strict: bool = False
) -> dict[str, Any]:
    """Validate a CameraE2E RAW dataset manifest and referenced artifacts."""

    manifest, manifest_path, root = _load_manifest(dataset)
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    records = list(manifest.get("records", []))
    if manifest.get("schema_version") != "camerae2e_dataset_manifest_v1":
        issues.append(
            {
                "kind": "schema_version",
                "message": "Dataset manifest schema_version is not camerae2e_dataset_manifest_v1.",
            }
        )
    if int(manifest.get("case_count", -1)) != len(records):
        issues.append(
            {
                "kind": "case_count",
                "message": "Dataset manifest case_count does not match records length.",
            }
        )

    metadata_path = root / "metadata.jsonl"
    metadata_rows = _load_metadata_rows(metadata_path, issues)
    metadata_by_case = {str(item.get("case_id")): item for item in metadata_rows}
    if len(metadata_rows) != len(records):
        issues.append(
            {
                "kind": "metadata_count",
                "message": "metadata.jsonl row count does not match records length.",
            }
        )

    for index, record in enumerate(records):
        _validate_record(record, index, root, metadata_by_case, issues, warnings)

    split_counts: dict[str, int] = {}
    for record in records:
        split_name = str(record.get("split", "unspecified"))
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
    if dict(manifest.get("splits", {})) != split_counts:
        issues.append(
            {
                "kind": "split_counts",
                "message": "Manifest split counts do not match record split assignments.",
            }
        )

    if strict and warnings:
        issues.extend({**warning, "strict_promoted": True} for warning in warnings)
    return {
        "schema_version": "camerae2e_dataset_validation_v1",
        "ok": not issues,
        "strict": bool(strict),
        "dataset_root": str(root),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "case_count": len(records),
        "split_counts": split_counts,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


def _optimization_cases(
    optimization_result: Mapping[str, Any], *, selection: str
) -> list[Mapping[str, Any]]:
    key = str(selection).strip().lower()
    if key == "best":
        best = optimization_result.get("best_case")
        return [] if best is None else [dict(best)]
    if key == "top":
        return [dict(item) for item in optimization_result.get("top_cases", [])]
    if key == "pareto":
        return [dict(item) for item in optimization_result.get("pareto_front", [])]
    if key == "selected":
        return [
            {"case_index": index, "scenario": scenario}
            for index, scenario in enumerate(optimization_result.get("selected_scenarios", []))
        ]
    if key == "all":
        return [
            dict(item)
            for item in optimization_result.get("cases", [])
            if item.get("feasible", True)
        ]
    raise ValueError("selection must be one of: best, top, pareto, selected, all.")


def _scenario_from_optimization_case(case: Mapping[str, Any], index: int) -> dict[str, Any]:
    scenario = _deep_dict(dict(case.get("scenario", {})))
    if not scenario:
        scenario = {"parameters": dict(case.get("parameters", {}))}
    base_name = str(scenario.get("name", "camerae2e_optimized_dataset_case"))
    case_index = int(case.get("case_index", index))
    scenario["name"] = f"{base_name}_opt{case_index:04d}"
    scenario.setdefault("optimization_case", _optimization_case_summary(case))
    return scenario


def _optimization_case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "parameters": _jsonable(case.get("parameters", {})),
        "score": case.get("score"),
        "feasible": case.get("feasible", True),
        "objective_values": _jsonable(case.get("objective_values", {})),
        "objective_utilities": _jsonable(case.get("objective_utilities", {})),
        "constraint_results": _jsonable(case.get("constraint_results", [])),
    }


def _normalize_labels(
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None, count: int
) -> list[dict[str, Any]]:
    empty = {"objects": [], "masks": [], "source": "not_provided"}
    if labels is None:
        return [dict(empty) for _ in range(count)]
    if isinstance(labels, Mapping):
        return [dict(labels) for _ in range(count)]
    values = [dict(item) for item in labels]
    if len(values) != count:
        raise ValueError("labels must contain one item per scenario when provided as an iterable.")
    return values


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _normalize_splits(
    split: Mapping[str, float] | Iterable[str] | str | None, count: int
) -> list[str]:
    if count <= 0:
        return []
    if split is None:
        return ["unspecified" for _ in range(count)]
    if isinstance(split, str):
        return [split for _ in range(count)]
    if isinstance(split, Mapping):
        names = [str(key) for key in split]
        weights = np.asarray([float(split[key]) for key in split], dtype=float)
        if len(names) == 0 or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("split mapping must contain non-negative weights with positive sum.")
        normalized = weights / float(np.sum(weights))
        exact = normalized * count
        counts = np.floor(exact).astype(int)
        remainder = count - int(np.sum(counts))
        fractions = exact - counts
        for index in np.argsort(-fractions)[:remainder]:
            counts[int(index)] += 1
        result: list[str] = []
        for name, item_count in zip(names, counts, strict=True):
            result.extend([name] * int(item_count))
        return result[:count]
    values = [str(item) for item in split]
    if len(values) != count:
        raise ValueError("split iterable must contain one split name per scenario.")
    return values


def _array_from_stage(result: Mapping[str, Any], stage_name: str) -> np.ndarray:
    stage = dict(result.get("stages", {}).get(stage_name, {}))
    value = stage.get("array")
    if value is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(value)


def _uint8_preview(array: Any) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.ndim == 3 and values.shape[2] > 3:
        values = values[:, :, :3]
    if values.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(finite)) > 1.0 or float(np.min(finite)) < 0.0:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        finite = (finite - lower) / max(upper - lower, 1.0e-12)
    return np.clip(np.round(finite * 255.0), 0.0, 255.0).astype(np.uint8)


def _write_npz_deterministic(path: Path, **arrays: Any) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_array(array: Any) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), sort_keys=True).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _load_manifest(
    dataset: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None, Path]:
    if isinstance(dataset, Mapping):
        manifest = dict(dataset)
        root = Path(str(manifest.get("dataset_root", "."))).expanduser()
        return manifest, None, root
    path = Path(dataset).expanduser()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(str(manifest.get("dataset_root", manifest_path.parent))).expanduser()
    return manifest, manifest_path, root


def _load_metadata_rows(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        issues.append({"kind": "missing_metadata", "message": f"Missing metadata.jsonl: {path}"})
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    "kind": "metadata_json",
                    "message": f"metadata.jsonl line {line_number} is invalid JSON: {exc}",
                }
            )
    return rows


def _validate_record(
    record: Mapping[str, Any],
    index: int,
    root: Path,
    metadata_by_case: Mapping[str, Mapping[str, Any]],
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    case_id = str(record.get("case_id", f"case_{index:04d}"))
    metadata_row = metadata_by_case.get(case_id)
    if metadata_row is None:
        issues.append({"case_id": case_id, "kind": "missing_metadata_row"})
    elif str(metadata_row.get("raw")) != str(record.get("raw")):
        issues.append({"case_id": case_id, "kind": "metadata_record_mismatch"})

    raw_path = _resolve_dataset_path(root, record.get("raw"))
    if raw_path is None or not raw_path.exists():
        issues.append({"case_id": case_id, "kind": "missing_raw", "path": str(raw_path)})
        return
    _check_file_hash(record, "raw_sha256", raw_path, case_id, issues)
    try:
        with np.load(raw_path, allow_pickle=False) as raw_data:
            raw = np.asarray(raw_data["raw"])
            sensor_digital = np.asarray(raw_data["sensor_digital"])
    except Exception as exc:
        issues.append({"case_id": case_id, "kind": "raw_npz_load", "message": str(exc)})
        return
    if list(raw.shape) != list(record.get("raw_shape", [])):
        issues.append({"case_id": case_id, "kind": "raw_shape"})
    if str(raw.dtype) != str(record.get("raw_dtype", raw.dtype)):
        issues.append({"case_id": case_id, "kind": "raw_dtype"})
    if record.get("raw_content_sha256") != _sha256_array(raw):
        issues.append({"case_id": case_id, "kind": "raw_content_sha256"})
    if record.get("sensor_digital_content_sha256") != _sha256_array(sensor_digital):
        issues.append({"case_id": case_id, "kind": "sensor_digital_content_sha256"})

    for path_key, hash_key, missing_kind in (
        ("rgb", "rgb_sha256", "missing_rgb"),
        ("raw_tiff", "raw_tiff_sha256", "missing_raw_tiff"),
        ("labels", "labels_sha256", "missing_labels"),
    ):
        path_value = record.get(path_key)
        if path_value is None:
            continue
        path = _resolve_dataset_path(root, path_value)
        if path is None or not path.exists():
            issues.append({"case_id": case_id, "kind": missing_kind, "path": str(path)})
            continue
        _check_file_hash(record, hash_key, path, case_id, issues)

    label_path = _resolve_dataset_path(root, record.get("labels"))
    if label_path is not None and label_path.exists():
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        if labels.get("case_id") != case_id:
            issues.append({"case_id": case_id, "kind": "label_case_id"})
        if "labels" not in labels:
            warnings.append({"case_id": case_id, "kind": "label_payload_missing"})


def _check_file_hash(
    record: Mapping[str, Any],
    hash_key: str,
    path: Path,
    case_id: str,
    issues: list[dict[str, Any]],
) -> None:
    expected = record.get(hash_key)
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual != expected:
        issues.append(
            {"case_id": case_id, "kind": hash_key, "expected": expected, "actual": actual}
        )


def _resolve_dataset_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


cameraE2EDatasetExport = camerae2e_dataset_export  # noqa: N816
cameraE2EDatasetExportFromOptimization = camerae2e_dataset_export_from_optimization  # noqa: N816
cameraE2EDatasetValidate = camerae2e_dataset_validate  # noqa: N816
