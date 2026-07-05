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
from .scene import scene_from_file, scene_set
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario
from .types import Camera, Scene

_KITTI_CLASS_IDS = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": -1,
}


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

    root = Path(output_dir).expanduser().resolve()
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
                "parameter_lineage": report.get("parameter_lineage", []),
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


def camerae2e_adas_camera_spec(preset: str = "kitti_yolo_demo") -> dict[str, Any]:
    """Return an ADAS camera preset for RAW demo generation.

    The default preset is tied to KITTI-style geometry, not to measured KITTI
    sensor calibration.  It is intended for reproducible ADAS/YOLO RAW factory
    demos and keeps the truth boundary explicit in the returned metadata.
    """

    key = str(preset).strip().lower()
    if key not in {
        "kitti_yolo_demo",
        "kitti_adas_demo",
        "wide_fov_adas_demo",
        "narrow_fov_adas_demo",
    }:
        raise ValueError(
            "ADAS camera spec preset must be one of: "
            "kitti_yolo_demo, wide_fov_adas_demo, narrow_fov_adas_demo."
        )
    image_size = [375, 1242]
    presets = {
        "kitti_yolo_demo": {
            "focal_px": 721.5377,
            "pixel_pitch_m": 3.75e-6,
            "f_number": 1.8,
            "integration_time_s": 0.004,
        },
        "kitti_adas_demo": {
            "focal_px": 721.5377,
            "pixel_pitch_m": 3.75e-6,
            "f_number": 1.8,
            "integration_time_s": 0.004,
        },
        "wide_fov_adas_demo": {
            "focal_px": 520.0,
            "pixel_pitch_m": 3.0e-6,
            "f_number": 2.0,
            "integration_time_s": 0.003,
        },
        "narrow_fov_adas_demo": {
            "focal_px": 1050.0,
            "pixel_pitch_m": 2.8e-6,
            "f_number": 2.4,
            "integration_time_s": 0.006,
        },
    }
    preset_values = presets[key]
    focal_px = float(preset_values["focal_px"])
    pixel_pitch_m = float(preset_values["pixel_pitch_m"])
    focal_length_m = focal_px * pixel_pitch_m
    hfov_deg = float(2.0 * np.rad2deg(np.arctan2(image_size[1] / 2.0, focal_px)))
    vfov_deg = float(2.0 * np.rad2deg(np.arctan2(image_size[0] / 2.0, focal_px)))
    return {
        "schema_version": "camerae2e_adas_camera_spec_v1",
        "preset": key,
        "readiness_tier": "proxy",
        "truth_boundary": (
            "KITTI camera geometry plus public-style ADAS assumptions; "
            "not measured sensor, lens, ISP, or KITTI raw calibration."
        ),
        "dataset_reference": {
            "name": "KITTI object-detection style",
            "label_format": "KITTI text plus YOLO normalized xywh labels",
            "native_image_size_rc": image_size,
            "p2_focal_px": focal_px,
            "principal_point_px": [609.5593, 172.8540],
        },
        "optics": {
            "focal_length_m": focal_length_m,
            "f_number": float(preset_values["f_number"]),
            "hfov_deg": hfov_deg,
            "vfov_deg": vfov_deg,
        },
        "sensor": {
            "pixel_pitch_m": pixel_pitch_m,
            "cfa_pattern": "bayer_rggb",
            "bits": 12,
            "integration_time_s": float(preset_values["integration_time_s"]),
            "analog_gain": 1.0,
            "noise_flag": 2,
        },
        "hw_isp": {
            "enabled": True,
            "nframes": 3,
            "fps": 30.0,
            "line_time_us": 15.2,
            "ae_apply_delay_frames": 2,
            "awb_apply_delay_frames": 2,
        },
        "demo": {
            "image_size_rc": [96, 320],
            "mean_luminance_cd_m2": 80.0,
            "scene_distance_m": 30.0,
        },
    }


def camerae2e_kitti_yolo_labels(
    source: str | Path | Iterable[str] | None = None,
    *,
    image_size: tuple[int, int] | list[int],
    class_ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return caller-supplied or demo KITTI labels with YOLO-normalized boxes."""

    rows, cols = _image_size_rc(image_size)
    mapping = dict(_KITTI_CLASS_IDS if class_ids is None else class_ids)
    if source is None:
        lines = _synthetic_kitti_label_lines(rows, cols)
        source_name = "synthetic_kitti_style"
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        lines = path.read_text(encoding="utf-8").splitlines()
        source_name = str(path)
    else:
        lines = [str(line) for line in source]
        source_name = "inline_kitti_labels"
    objects = [
        _kitti_label_to_object(line, rows=rows, cols=cols, class_ids=mapping)
        for line in lines
        if line.strip()
    ]
    return {
        "schema_version": "camerae2e_kitti_yolo_labels_v1",
        "source": source_name,
        "image_size_rc": [rows, cols],
        "class_map": mapping,
        "objects": objects,
        "masks": [],
    }


def camerae2e_dataset_export_adas_kitti_demo(
    output_dir: str | Path,
    *,
    image_paths: Iterable[str | Path] | None = None,
    label_paths: Iterable[str | Path] | None = None,
    case_count: int = 2,
    spec: Mapping[str, Any] | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_rgb: bool = True,
    include_tiff: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = "demo",
) -> dict[str, Any]:
    """Generate a KITTI/YOLO-style ADAS RAW dataset demo.

    If ``image_paths`` and optional ``label_paths`` are supplied, those KITTI
    RGB frames and labels are used.  Otherwise the function creates a small,
    deterministic KITTI-style road scene with YOLO/KITTI boxes so the RAW
    factory can be exercised without bundling the KITTI dataset.
    """

    store = asset_store or AssetStore.default()
    camera_spec = _jsonable(spec or camerae2e_adas_camera_spec())
    demo_shape = _image_size_rc(camera_spec.get("demo", {}).get("image_size_rc", [96, 320]))
    image_list = [] if image_paths is None else [Path(path).expanduser() for path in image_paths]
    label_list = [] if label_paths is None else [Path(path).expanduser() for path in label_paths]
    if label_list and len(label_list) != len(image_list):
        raise ValueError("label_paths must have the same length as image_paths.")

    scenarios: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    if image_list:
        selected_images = image_list[: max(int(case_count), 0)]
        for index, image_path in enumerate(selected_images):
            image = np.asarray(iio.imread(image_path))
            if image.ndim < 2:
                raise ValueError(f"KITTI demo image must be 2D or 3D: {image_path}")
            rows, cols = int(image.shape[0]), int(image.shape[1])
            label_source = label_list[index] if label_list else None
            scene = _adas_scene_from_rgb(
                image,
                camera_spec,
                store,
                f"kitti_frame_{index:04d}",
            )
            scenarios.append(_adas_scenario(scene, camera_spec, index))
            labels.append(camerae2e_kitti_yolo_labels(label_source, image_size=(rows, cols)))
    else:
        for index in range(max(int(case_count), 0)):
            rgb, label_payload = _synthetic_adas_rgb_and_labels(
                demo_shape[0], demo_shape[1], index=index
            )
            scene = _adas_scene_from_rgb(rgb, camera_spec, store, f"synthetic_kitti_{index:04d}")
            scenarios.append(_adas_scenario(scene, camera_spec, index))
            labels.append(label_payload)

    manifest = camerae2e_dataset_export(
        output_dir,
        scenarios,
        asset_store=store,
        seed=seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        split=split,
    )
    manifest["adas_kitti_demo"] = {
        "schema_version": "camerae2e_adas_kitti_demo_v1",
        "camera_spec": camera_spec,
        "case_count": len(scenarios),
        "input_mode": "synthetic" if not image_list else "kitti_files",
        "note": (
            "This demo applies KITTI-style ADAS geometry and YOLO/KITTI labels "
            "to the CameraE2E RAW factory. It is not KITTI raw sensor ground truth."
        ),
    }
    _rescale_manifest_labels_to_raw(manifest)
    _rewrite_dataset_metadata_and_manifest(manifest)
    return manifest


def camerae2e_dataset_export_camera_spec_variants(
    output_dir: str | Path,
    *,
    image_paths: Iterable[str | Path] | None = None,
    label_paths: Iterable[str | Path] | None = None,
    source_spec: Mapping[str, Any] | None = None,
    target_specs: Iterable[Mapping[str, Any] | str] | None = None,
    case_count: int = 1,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_rgb: bool = True,
    include_tiff: bool = False,
    split: str = "camera_variant",
) -> dict[str, Any]:
    """Export proxy re-captures of one KITTI-style scene under target cameras.

    This is a controlled transformation, not an inverse-physics reconstruction.
    The source RGB frame is interpreted as a display-derived proxy scene, then
    re-rendered through each target camera spec.
    """

    targets = list(
        target_specs
        or ["kitti_yolo_demo", "wide_fov_adas_demo", "narrow_fov_adas_demo"]
    )
    target_payloads = [
        camerae2e_adas_camera_spec(item) if isinstance(item, str) else _jsonable(item)
        for item in targets
    ]
    source_payload = _jsonable(source_spec or camerae2e_adas_camera_spec())
    store = asset_store or AssetStore.default()
    root = Path(output_dir).expanduser().resolve()
    image_list = [] if image_paths is None else [Path(path).expanduser() for path in image_paths]
    label_list = [] if label_paths is None else [Path(path).expanduser() for path in label_paths]
    if label_list and len(label_list) != len(image_list):
        raise ValueError("label_paths must have the same length as image_paths.")

    scenarios: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_descriptions: list[dict[str, Any]] = []
    if image_list:
        selected_images = image_list[: max(int(case_count), 0)]
        source_frames = []
        for index, image_path in enumerate(selected_images):
            image = np.asarray(iio.imread(image_path))
            rows, cols = int(image.shape[0]), int(image.shape[1])
            label_source = label_list[index] if label_list else None
            source_frames.append(
                (
                    image,
                    camerae2e_kitti_yolo_labels(label_source, image_size=(rows, cols)),
                    f"kitti_file_{index:04d}",
                    str(image_path),
                )
            )
    else:
        demo_shape = _image_size_rc(source_payload.get("demo", {}).get("image_size_rc", [96, 320]))
        source_frames = []
        for index in range(max(int(case_count), 0)):
            image, source_labels = _synthetic_adas_rgb_and_labels(
                demo_shape[0], demo_shape[1], index=index
            )
            source_frames.append(
                (image, source_labels, f"synthetic_kitti_{index:04d}", "synthetic")
            )

    for frame_index, (image, source_labels, frame_name, source_path) in enumerate(source_frames):
        source_descriptions.append(
            {
                "frame_index": frame_index,
                "source": source_path,
                "source_label_frame": source_labels.get("image_size_rc"),
            }
        )
        for target_index, target_spec in enumerate(target_payloads):
            scene = _adas_scene_from_rgb(
                image,
                target_spec,
                store,
                f"{frame_name}_{target_spec.get('preset', target_index)}",
            )
            scenario = _adas_scenario(scene, target_spec, len(scenarios))
            scenario["source_camera_spec"] = source_payload
            scenario["target_camera_spec"] = target_spec
            scenarios.append(scenario)
            labels.append(
                _rescale_label_payload(
                    source_labels,
                    _image_size_rc(target_spec["dataset_reference"]["native_image_size_rc"]),
                    coordinate_frame="target_camera_spec_native_image",
                )
            )

    manifest = camerae2e_dataset_export(
        root,
        scenarios,
        asset_store=store,
        seed=seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        split=split,
    )
    manifest["camera_spec_variants"] = {
        "schema_version": "camerae2e_camera_spec_variants_v1",
        "source_spec": source_payload,
        "target_specs": target_payloads,
        "source_frames": source_descriptions,
        "truth_boundary": (
            "RGB-to-scene proxy re-capture. This does not recover true KITTI "
            "spectral radiance, depth, occlusion, lens flare, ISP inverse, or measured raw."
        ),
    }
    _rescale_manifest_labels_to_raw(manifest)
    _rewrite_dataset_metadata_and_manifest(manifest)
    return manifest


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


def _adas_scene_from_rgb(
    rgb: Any, spec: Mapping[str, Any], asset_store: AssetStore, name: str
) -> Scene:
    scene = scene_from_file(
        np.asarray(rgb),
        "rgb",
        float(spec.get("demo", {}).get("mean_luminance_cd_m2", 80.0)),
        "lcdExample.mat",
        asset_store=asset_store,
    )
    scene.name = str(name)
    scene = scene_set(scene, "fov", float(spec.get("optics", {}).get("hfov_deg", 81.4)))
    scene = scene_set(scene, "distance", float(spec.get("demo", {}).get("scene_distance_m", 30.0)))
    return scene


def _adas_scenario(scene: Scene, spec: Mapping[str, Any], index: int) -> dict[str, Any]:
    sensor = dict(spec.get("sensor", {}))
    optics = dict(spec.get("optics", {}))
    hw_isp = dict(spec.get("hw_isp", {}))
    return {
        "name": f"adas_kitti_yolo_demo_{index:04d}",
        "scene": scene,
        "sensor": {
            "noise_flag": sensor.get("noise_flag", 2),
            "integration_time": sensor.get("integration_time_s", 0.004),
            "analog_gain": sensor.get("analog_gain", 1.0),
        },
        "parameters": {
            "pixel.size": [
                sensor.get("pixel_pitch_m", 3.75e-6),
                sensor.get("pixel_pitch_m", 3.75e-6),
            ],
            "sensor.bits": sensor.get("bits", 12),
            "optics.focal_length": optics.get("focal_length_m", 0.0027),
            "optics.fnumber": optics.get("f_number", 1.8),
        },
        "hw_isp": {
            "enabled": bool(hw_isp.get("enabled", True)),
            "nframes": int(hw_isp.get("nframes", 3)),
            "fps": float(hw_isp.get("fps", 30.0)),
            "line_time_us": float(hw_isp.get("line_time_us", 15.2)),
            "ae_apply_delay_frames": int(hw_isp.get("ae_apply_delay_frames", 2)),
            "awb_apply_delay_frames": int(hw_isp.get("awb_apply_delay_frames", 2)),
        },
        "adas_camera_spec": _jsonable(spec),
    }


def _synthetic_adas_rgb_and_labels(
    rows: int, cols: int, *, index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    rr, cc = np.indices((rows, cols))
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    horizon = int(round(rows * 0.42))
    sky_grad = np.clip(0.65 + 0.25 * (1.0 - rr / max(horizon, 1)), 0.0, 1.0)
    road_grad = np.clip(0.18 + 0.25 * (rr - horizon) / max(rows - horizon, 1), 0.0, 1.0)
    sky_mask = rr < horizon
    rgb[..., 0] = np.where(sky_mask, 95 * sky_grad, 90 * road_grad).astype(np.uint8)
    rgb[..., 1] = np.where(sky_mask, 145 * sky_grad, 95 * road_grad).astype(np.uint8)
    rgb[..., 2] = np.where(sky_mask, 210 * sky_grad, 85 * road_grad).astype(np.uint8)

    lane_center = cols / 2.0 + (index - 0.5) * cols * 0.03
    for offset in (-cols * 0.14, cols * 0.14):
        x = (lane_center + offset + (rr - horizon) * offset / max(rows - horizon, 1)).astype(int)
        mask = (rr > horizon) & (np.abs(cc - x) <= 1)
        rgb[mask] = np.array([230, 230, 210], dtype=np.uint8)

    boxes = [
        ("Car", [0.42 * cols, 0.55 * rows, 0.66 * cols, 0.82 * rows], [1.55, 1.70, 4.20]),
        ("Car", [0.18 * cols, 0.50 * rows, 0.31 * cols, 0.66 * rows], [1.50, 1.65, 3.90]),
        ("Pedestrian", [0.72 * cols, 0.49 * rows, 0.77 * cols, 0.73 * rows], [1.75, 0.60, 0.50]),
    ]
    lines = []
    for label, box, dims_hwl in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        color = (
            np.array([170, 35, 30], dtype=np.uint8)
            if label == "Car"
            else np.array([40, 120, 45], dtype=np.uint8)
        )
        rgb[max(y1, 0) : min(y2, rows), max(x1, 0) : min(x2, cols)] = color
        h, w, length = dims_hwl
        lines.append(
            f"{label} 0.00 0 0.00 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} "
            f"{h:.2f} {w:.2f} {length:.2f} 0.00 0.00 20.00 0.00"
        )
    return rgb, camerae2e_kitti_yolo_labels(lines, image_size=(rows, cols))


def _synthetic_kitti_label_lines(rows: int, cols: int) -> list[str]:
    _, labels = _synthetic_adas_rgb_and_labels(rows, cols, index=0)
    return [
        str(item.get("kitti", {}).get("line", ""))
        for item in labels.get("objects", [])
        if item.get("kitti", {}).get("line")
    ]


def _kitti_label_to_object(
    line: str, *, rows: int, cols: int, class_ids: Mapping[str, int]
) -> dict[str, Any]:
    parts = line.strip().split()
    if len(parts) < 8:
        raise ValueError(f"KITTI label line has too few columns: {line!r}")
    label = parts[0]
    bbox = [float(parts[index]) for index in range(4, 8)]
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    yolo = [
        (x1 + width / 2.0) / max(cols, 1),
        (y1 + height / 2.0) / max(rows, 1),
        width / max(cols, 1),
        height / max(rows, 1),
    ]
    return {
        "label": label,
        "class_id": int(class_ids.get(label, -1)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "yolo_xywhn": yolo,
        "kitti": {
            "line": line,
            "truncation": float(parts[1]) if len(parts) > 1 else None,
            "occlusion": int(float(parts[2])) if len(parts) > 2 else None,
            "alpha": float(parts[3]) if len(parts) > 3 else None,
            "dimensions_hwl_m": [float(value) for value in parts[8:11]] if len(parts) >= 11 else [],
            "location_xyz_m": [float(value) for value in parts[11:14]] if len(parts) >= 14 else [],
            "rotation_y": float(parts[14]) if len(parts) >= 15 else None,
        },
    }


def _rescale_manifest_labels_to_raw(manifest: dict[str, Any]) -> None:
    root = Path(str(manifest.get("dataset_root", "."))).expanduser()
    for record in manifest.get("records", []):
        raw_shape = record.get("raw_shape", [])
        if len(raw_shape) < 2:
            continue
        label_path = _resolve_dataset_path(root, record.get("labels"))
        if label_path is None or not label_path.exists():
            continue
        label_json = json.loads(label_path.read_text(encoding="utf-8"))
        label_json["labels"] = _rescale_label_payload(
            label_json.get("labels", {}),
            _image_size_rc(raw_shape[:2]),
            coordinate_frame="exported_raw_shape",
        )
        label_path.write_text(
            json.dumps(_jsonable(label_json), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        record["labels_sha256"] = _sha256_file(label_path)
        record["label_coordinate_frame"] = "exported_raw_shape"


def _rescale_label_payload(
    labels: Mapping[str, Any],
    target_size_rc: tuple[int, int],
    *,
    coordinate_frame: str,
) -> dict[str, Any]:
    payload = _jsonable(labels)
    source_size = _image_size_rc(payload.get("image_size_rc", target_size_rc))
    target_rows, target_cols = target_size_rc
    y_scale = target_rows / max(source_size[0], 1)
    x_scale = target_cols / max(source_size[1], 1)
    objects = []
    for item in payload.get("objects", []):
        obj = dict(item)
        bbox = obj.get("bbox_xyxy")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            scaled = [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]
            width = max(scaled[2] - scaled[0], 0.0)
            height = max(scaled[3] - scaled[1], 0.0)
            obj["bbox_xyxy"] = scaled
            obj["yolo_xywhn"] = [
                (scaled[0] + width / 2.0) / max(target_cols, 1),
                (scaled[1] + height / 2.0) / max(target_rows, 1),
                width / max(target_cols, 1),
                height / max(target_rows, 1),
            ]
        objects.append(obj)
    payload["objects"] = objects
    payload["image_size_rc"] = [target_rows, target_cols]
    payload["coordinate_frame"] = coordinate_frame
    payload["source_coordinate_frame"] = {
        "image_size_rc": [source_size[0], source_size[1]],
        "coordinate_frame": labels.get("coordinate_frame", labels.get("source", "source_image")),
    }
    return payload


def _rewrite_dataset_metadata_and_manifest(manifest: dict[str, Any]) -> None:
    root = Path(str(manifest.get("dataset_root", "."))).expanduser()
    metadata_path = root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_stream:
        for record in manifest.get("records", []):
            metadata_stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
    manifest.setdefault("integrity", {})["metadata_jsonl_sha256"] = _sha256_file(metadata_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _image_size_rc(image_size: tuple[int, int] | list[int] | Any) -> tuple[int, int]:
    values = np.asarray(image_size, dtype=int).reshape(-1)
    if values.size < 2 or int(values[0]) <= 0 or int(values[1]) <= 0:
        raise ValueError("image_size must contain positive row and column counts.")
    return int(values[0]), int(values[1])


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
cameraE2EDatasetExportADASKITTIDemo = camerae2e_dataset_export_adas_kitti_demo  # noqa: N816
cameraE2EDatasetExportCameraSpecVariants = (  # noqa: N816
    camerae2e_dataset_export_camera_spec_variants
)
cameraE2EDatasetExportFromOptimization = camerae2e_dataset_export_from_optimization  # noqa: N816
cameraE2EDatasetValidate = camerae2e_dataset_validate  # noqa: N816
cameraE2EADASCameraSpec = camerae2e_adas_camera_spec  # noqa: N816
cameraE2EKITTIYOLOLabels = camerae2e_kitti_yolo_labels  # noqa: N816
