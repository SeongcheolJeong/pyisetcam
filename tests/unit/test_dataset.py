from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyisetcam import (
    camerae2e_adas_camera_spec,
    camerae2e_dataset_export,
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_camera_spec_variants,
    camerae2e_dataset_export_from_optimization,
    camerae2e_dataset_validate,
    camerae2e_kitti_yolo_labels,
    camerae2e_optimize_parameters,
)


def test_camerae2e_dataset_export_writes_manifest_raw_preview_and_labels(tmp_path: Path) -> None:
    manifest = camerae2e_dataset_export(
        tmp_path,
        [
            {
                "name": "dataset_case",
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
            }
        ],
        seed=5,
        labels={"objects": [{"label": "chart", "bbox_xyxy": [0, 0, 4, 4]}]},
        split="train",
    )
    record = manifest["records"][0]

    assert manifest["schema_version"] == "camerae2e_dataset_manifest_v1"
    assert Path(record["raw"]).exists()
    assert Path(record["rgb"]).exists()
    assert Path(record["labels"]).exists()
    assert record["split"] == "train"
    assert record["raw_sha256"].startswith("sha256:")
    assert record["raw_content_sha256"].startswith("sha256:")
    assert (tmp_path / "metadata.jsonl").exists()

    raw = np.load(record["raw"])
    labels = json.loads(Path(record["labels"]).read_text(encoding="utf-8"))
    validation = camerae2e_dataset_validate(tmp_path)

    assert raw["raw"].shape == tuple(record["raw_shape"])
    assert labels["labels"]["objects"][0]["label"] == "chart"
    assert manifest["format"]["dng"] == "not emitted in v1"
    assert validation["ok"] is True


def test_camerae2e_dataset_export_is_seed_reproducible_and_splits_cases(
    tmp_path: Path,
) -> None:
    scenarios = [
        {"name": f"case_{index}", "scene": {"type": "uniform ee", "args": [8]}}
        for index in range(3)
    ]
    first = camerae2e_dataset_export(
        tmp_path,
        scenarios,
        seed=9,
        split={"train": 2.0, "val": 1.0},
        include_rgb=False,
    )
    second = camerae2e_dataset_export(
        tmp_path,
        scenarios,
        seed=9,
        split={"train": 2.0, "val": 1.0},
        include_rgb=False,
    )

    assert first["splits"] == {"train": 2, "val": 1}
    assert [item["split"] for item in first["records"]] == ["train", "train", "val"]
    assert [item["raw_sha256"] for item in first["records"]] == [
        item["raw_sha256"] for item in second["records"]
    ]
    assert [item["raw_content_sha256"] for item in first["records"]] == [
        item["raw_content_sha256"] for item in second["records"]
    ]
    assert first["integrity"]["metadata_jsonl_sha256"] == second["integrity"][
        "metadata_jsonl_sha256"
    ]


def test_camerae2e_dataset_validate_detects_tampered_raw(tmp_path: Path) -> None:
    manifest = camerae2e_dataset_export(
        tmp_path,
        [{"scene": {"type": "uniform ee", "args": [8]}}],
        seed=2,
        include_rgb=False,
    )
    raw_path = Path(manifest["records"][0]["raw"])
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")

    validation = camerae2e_dataset_validate(manifest)

    assert validation["ok"] is False
    assert any(issue["kind"] == "raw_sha256" for issue in validation["issues"])


def test_camerae2e_dataset_export_from_optimization_uses_pareto_scenarios(
    tmp_path: Path,
) -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "dataset_from_optimizer",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        [
            {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
            {"metric": "metrics.artifact.raw_std", "direction": "minimize"},
        ],
        seed=31,
    )
    manifest = camerae2e_dataset_export_from_optimization(
        tmp_path,
        result,
        selection="pareto",
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)

    assert manifest["case_count"] == result["pareto_case_count"] == 2
    assert manifest["source_optimization"]["selection"] == "pareto"
    assert manifest["source_optimization"]["selected_case_count"] == 2
    assert {record["split"] for record in manifest["records"]} == {"pareto"}
    assert manifest["records"][0]["parameter_lineage"]
    assert any(
        item["path"] == "sensor.integration_time"
        for item in manifest["records"][0]["parameter_lineage"]
    )
    assert validation["ok"] is True


def test_camerae2e_adas_kitti_demo_exports_raw_and_yolo_labels(tmp_path: Path) -> None:
    spec = camerae2e_adas_camera_spec()
    labels = camerae2e_kitti_yolo_labels(
        ["Car 0.00 0 0.00 10 20 50 60 1.5 1.6 4.0 0 0 20 0"],
        image_size=(96, 320),
    )
    manifest = camerae2e_dataset_export_adas_kitti_demo(
        tmp_path,
        case_count=1,
        seed=77,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    record = manifest["records"][0]
    record_labels = json.loads(Path(record["labels"]).read_text(encoding="utf-8"))

    assert spec["preset"] == "kitti_yolo_demo"
    assert labels["objects"][0]["class_id"] == 0
    assert labels["objects"][0]["yolo_xywhn"][2] > 0.0
    assert manifest["adas_kitti_demo"]["input_mode"] == "synthetic"
    assert manifest["adas_kitti_demo"]["camera_spec"]["readiness_tier"] == "proxy"
    assert record["parameter_lineage"]
    assert any(item["path"] == "optics.focal_length" for item in record["parameter_lineage"])
    assert record["label_coordinate_frame"] == "exported_raw_shape"
    assert record_labels["labels"]["objects"][0]["label"] == "Car"
    assert record_labels["labels"]["objects"][0]["yolo_xywhn"][2] > 0.0
    assert record_labels["labels"]["image_size_rc"] == record["raw_shape"][:2]
    assert validation["ok"] is True


def test_camerae2e_camera_spec_variants_recapture_scene_with_target_specs(
    tmp_path: Path,
) -> None:
    manifest = camerae2e_dataset_export_camera_spec_variants(
        tmp_path,
        target_specs=["wide_fov_adas_demo", "narrow_fov_adas_demo"],
        case_count=1,
        seed=88,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    presets = [
        record["scenario"]["target_camera_spec"]["preset"]
        for record in manifest["records"]
    ]
    raw_shapes = [tuple(record["raw_shape"][:2]) for record in manifest["records"]]
    focal_lengths = [
        record["scenario"]["target_camera_spec"]["optics"]["focal_length_m"]
        for record in manifest["records"]
    ]
    first_labels = json.loads(Path(manifest["records"][0]["labels"]).read_text(encoding="utf-8"))

    assert manifest["case_count"] == 2
    assert presets == ["wide_fov_adas_demo", "narrow_fov_adas_demo"]
    assert focal_lengths[0] != focal_lengths[1]
    assert all(
        any(item["path"] == "optics.focal_length" for item in record["parameter_lineage"])
        for record in manifest["records"]
    )
    assert first_labels["labels"]["coordinate_frame"] == "exported_raw_shape"
    assert first_labels["labels"]["image_size_rc"] == list(raw_shapes[0])
    assert "RGB-to-scene proxy" in manifest["camera_spec_variants"]["truth_boundary"]
    assert validation["ok"] is True
