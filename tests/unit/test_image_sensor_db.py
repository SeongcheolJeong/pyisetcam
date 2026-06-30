from __future__ import annotations

from pathlib import Path

from pyisetcam import (
    image_sensor_db_get,
    image_sensor_db_parameters,
    image_sensor_db_records,
    image_sensor_db_summary,
)


def test_image_sensor_db_records_are_searchable() -> None:
    records = image_sensor_db_records(limit=5)

    assert records
    assert {"sensor_id", "code", "manufacturer", "stack_config_path", "tcad_profile_path"} <= set(records[0])
    assert image_sensor_db_records("sony", limit=1)


def test_image_sensor_db_get_accepts_sensor_id_and_code() -> None:
    first = image_sensor_db_records(limit=1)[0]

    by_id = image_sensor_db_get(first["sensor_id"])
    by_code = image_sensor_db_get(first["code"])

    assert by_id["sensor_id"] == first["sensor_id"]
    assert by_code["code"] == first["code"]


def test_image_sensor_db_parameters_return_existing_paths() -> None:
    first = image_sensor_db_records(limit=1)[0]
    params = image_sensor_db_parameters(first["sensor_id"])

    assert Path(params["catalog_path"]).exists()
    assert Path(params["stack_config_path"]).exists()
    assert Path(params["tcad_profile_path"]).exists()
    assert params["lut_path"] is None or Path(params["lut_path"]).exists()
    assert Path(params["generation_map_path"]).exists()
    assert params["collection_summary_paths"]
    assert Path(params["accuracy_gate_path"]).exists()


def test_image_sensor_db_summary_has_counts() -> None:
    summary = image_sensor_db_summary()

    assert summary["record_count"] >= 100
    assert summary["manufacturer_count"] >= 5
    assert "Sony" in summary["manufacturers"]
