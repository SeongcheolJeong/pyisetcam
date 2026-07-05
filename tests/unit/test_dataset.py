from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyisetcam import camerae2e_dataset_export, camerae2e_dataset_validate


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
