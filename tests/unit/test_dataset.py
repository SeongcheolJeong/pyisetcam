from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyisetcam import camerae2e_dataset_export


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
    )
    record = manifest["records"][0]

    assert manifest["schema_version"] == "camerae2e_dataset_manifest_v1"
    assert Path(record["raw"]).exists()
    assert Path(record["rgb"]).exists()
    assert Path(record["labels"]).exists()
    assert (tmp_path / "metadata.jsonl").exists()

    raw = np.load(record["raw"])
    labels = json.loads(Path(record["labels"]).read_text(encoding="utf-8"))

    assert raw["raw"].shape == tuple(record["raw_shape"])
    assert labels["labels"]["objects"][0]["label"] == "chart"
    assert manifest["format"]["dng"] == "not emitted in v1"
