from __future__ import annotations

from pathlib import Path

from pyisetcam import (
    CameraE2EDBEntry,
    camerae2e_db_catalog,
    camerae2e_db_get,
    camerae2e_db_parameters,
    camerae2e_db_search,
    camerae2e_db_summary,
)


def test_camerae2e_db_catalog_has_core_families() -> None:
    entries = camerae2e_db_catalog()

    assert entries
    assert all(isinstance(entry, CameraE2EDBEntry) for entry in entries)
    families = {entry.family for entry in entries}

    assert {"lens", "sensor", "isp", "perception", "assets", "parity"} <= families


def test_camerae2e_db_search_finds_runtime_parameter_sources() -> None:
    lens_rows = camerae2e_db_search("lens", include_missing=False)
    fdtd_rows = camerae2e_db_search(tags=["fdtd"], include_missing=False)
    hwisp_rows = camerae2e_db_search(family="isp", include_missing=False)
    perception_rows = camerae2e_db_search("yolo", include_missing=False)

    assert any(row["name"] == "lens_patents_active" for row in lens_rows)
    assert any(row["name"] == "fdtd_sensor_lut_active" for row in fdtd_rows)
    assert any(row["name"] == "hwisp_parameter_profiles" for row in hwisp_rows)
    assert any(row["name"] == "task_perception_model_profiles" for row in perception_rows)


def test_camerae2e_db_parameters_are_directly_usable_paths() -> None:
    lens_params = camerae2e_db_parameters("lens_patents_active")
    hwisp_params = camerae2e_db_parameters("hwisp_parameter_profiles")
    perception_params = camerae2e_db_parameters("task_perception_model_profiles")

    assert Path(lens_params["db_path"]).exists()
    assert Path(lens_params["psf_dir"]).exists()
    assert isinstance(hwisp_params["profile_names"], list)
    assert hwisp_params["profile_names"]
    assert Path(perception_params["profiles_path"]).exists()
    assert isinstance(perception_params["profile_names"], list)
    assert perception_params["profile_names"]


def test_camerae2e_db_get_and_summary_are_json_friendly() -> None:
    row = camerae2e_db_get("lens_patents_active")
    summary = camerae2e_db_summary()

    assert row["name"] == "lens_patents_active"
    assert isinstance(row["path"], str)
    assert isinstance(row["parameters"]["db_path"], str)
    assert summary["total"] >= 6
    assert "lens_patents_active" in summary["active"]
