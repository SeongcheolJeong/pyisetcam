"""CameraE2E RAW dataset export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
            np.savez_compressed(
                raw_path,
                raw=raw_array,
                sensor_digital=_array_from_stage(result, "sensor_digital"),
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
                "seed": int(seed) + index,
                "scenario_name": report["name"],
                "raw": str(raw_path),
                "rgb": None if rgb_path is None else str(rgb_path),
                "raw_tiff": None if tiff_path is None else str(tiff_path),
                "labels": str(label_path),
                "raw_shape": list(raw_array.shape),
                "rgb_shape": list(rgb_array.shape),
                "faca_metrics": report.get("metrics", {}),
            }
            records.append(record)
            metadata_stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    manifest = {
        "schema_version": "camerae2e_dataset_manifest_v1",
        "dataset_root": str(root),
        "seed": int(seed),
        "case_count": len(records),
        "format": {
            "raw": "compressed NumPy .npz with raw and sensor_digital arrays",
            "rgb": "8-bit PNG preview when include_rgb=True",
            "raw_tiff": "float32 TIFF when include_tiff=True",
            "labels": "JSON labels supplied by caller; empty by default",
            "dng": "not emitted in v1",
        },
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


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
