"""Image-sensor database selection helpers.

This module indexes the external FDTD ``sensor_db`` package so callers can
search image sensors, select one by stable ID, and pass its generated stack /
TCAD paths into CameraE2E configuration code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .fdtd_sensor import fdtd_sensor_default_lut_path
from .tcad_sensor import tcad_sensor_default_paths


def image_sensor_db_root(root: str | Path | None = None) -> Path:
    """Return the active image-sensor DB root directory."""

    if root is not None:
        return Path(root).expanduser()
    explicit = os.environ.get("PYISETCAM_IMAGE_SENSOR_DB_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    fdtd_root = Path(os.environ.get("PYISETCAM_FDTD_ROOT", "/Users/seongcheoljeong/FDTD")).expanduser()
    return fdtd_root / "sensor_db"


def image_sensor_db_catalog_path(root: str | Path | None = None) -> Path:
    """Return the active image-sensor catalog JSON path."""

    return image_sensor_db_root(root) / "sensor_catalog.json"


def image_sensor_db_load(root: str | Path | None = None) -> dict[str, Any]:
    """Load the image-sensor catalog JSON."""

    path = image_sensor_db_catalog_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Image sensor DB catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def image_sensor_db_records(
    query: str | None = None,
    *,
    manufacturer: str | None = None,
    limit: int | None = None,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return normalized image-sensor records with optional text filters."""

    catalog = image_sensor_db_load(root)
    query_text = "" if query is None else str(query).lower()
    manufacturer_text = "" if manufacturer is None else str(manufacturer).lower()
    records: list[dict[str, Any]] = []
    for raw in catalog.get("records", []):
        record = _normalize_record(raw, image_sensor_db_root(root))
        if manufacturer_text and manufacturer_text not in record["manufacturer"].lower():
            continue
        haystack = " ".join(
            str(record.get(key, ""))
            for key in (
                "sensor_id",
                "code",
                "manufacturer",
                "device_name",
                "title",
                "pixel_architecture",
                "cfa_pattern",
                "illumination",
            )
        ).lower()
        if query_text and query_text not in haystack:
            continue
        records.append(record)
    records.sort(key=lambda item: (item["manufacturer"].lower(), item["device_name"].lower(), item["code"]))
    if limit is not None:
        records = records[: int(limit)]
    return records


def image_sensor_db_get(sensor_id_or_code: str, root: str | Path | None = None) -> dict[str, Any]:
    """Return one normalized image-sensor record by ``sensor_id`` or report code."""

    key = str(sensor_id_or_code).lower()
    for record in image_sensor_db_records(root=root):
        if key in {str(record["sensor_id"]).lower(), str(record["code"]).lower()}:
            return record
    raise KeyError(f"Unknown image sensor DB record: {sensor_id_or_code}")


def image_sensor_db_parameters(sensor_id_or_code: str, root: str | Path | None = None) -> dict[str, Any]:
    """Return path parameters for direct CameraE2E sensor configuration."""

    record = image_sensor_db_get(sensor_id_or_code, root=root)
    defaults = tcad_sensor_default_paths(image_sensor_db_root(root).parent)
    return {
        "sensor_id": record["sensor_id"],
        "code": record["code"],
        "catalog_path": image_sensor_db_catalog_path(root),
        "stack_config_path": Path(record["stack_config_path"]) if record["stack_config_path"] else None,
        "tcad_profile_path": Path(record["tcad_profile_path"]) if record["tcad_profile_path"] else None,
        "lut_path": fdtd_sensor_default_lut_path(),
        "generation_map_path": defaults["generation_map_path"],
        "collection_summary_paths": defaults["collection_summary_paths"],
        "accuracy_gate_path": defaults["accuracy_gate_path"],
    }


def image_sensor_db_summary(root: str | Path | None = None) -> dict[str, Any]:
    """Return compact image-sensor DB summary facts."""

    catalog = image_sensor_db_load(root)
    records = image_sensor_db_records(root=root)
    manufacturers: dict[str, int] = {}
    architectures: dict[str, int] = {}
    for record in records:
        manufacturers[record["manufacturer"]] = manufacturers.get(record["manufacturer"], 0) + 1
        arch = str(record.get("pixel_architecture") or "unknown")
        architectures[arch] = architectures.get(arch, 0) + 1
    return {
        "schema": catalog.get("schema"),
        "generated_at": catalog.get("generated_at"),
        "record_count": len(records),
        "manufacturer_count": len(manufacturers),
        "manufacturers": dict(sorted(manufacturers.items(), key=lambda item: (-item[1], item[0]))),
        "pixel_architectures": dict(sorted(architectures.items(), key=lambda item: (-item[1], item[0]))),
        "catalog_path": str(image_sensor_db_catalog_path(root)),
    }


def _normalize_record(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata = dict(raw.get("metadata", {}))
    specs = dict(raw.get("derived_specs", {}))
    generated = dict(raw.get("generated_files", {}))
    stack_path = generated.get("stack_config")
    tcad_path = generated.get("tcad_profile")
    sensor_id = _sensor_id(raw, stack_path)
    return {
        "sensor_id": sensor_id,
        "code": str(raw.get("code", metadata.get("code", ""))),
        "manufacturer": str(metadata.get("manufacturer", "Unknown")),
        "device_name": str(metadata.get("device_name", "")),
        "title": str(metadata.get("title", raw.get("description_excerpt", ""))),
        "report_type": str(metadata.get("report_type", "")),
        "analysis_year": metadata.get("analysis_year"),
        "pixel_pitch_um": specs.get("pixel_pitch_um"),
        "pixel_architecture": specs.get("pixel_architecture"),
        "cfa_pattern": specs.get("cfa_pattern"),
        "illumination": specs.get("illumination"),
        "microlens_type": specs.get("microlens_type"),
        "optical_stack_height_um": specs.get("optical_stack_height_um"),
        "active_si_thickness_um": specs.get("active_si_thickness_um"),
        "dti_type": specs.get("dti_type"),
        "has_dti": specs.get("has_dti"),
        "has_pdaf": specs.get("has_pdaf"),
        "has_hdr": specs.get("has_hdr"),
        "has_lofic": specs.get("has_lofic"),
        "resolution_mp": specs.get("resolution_mp"),
        "optical_format": specs.get("optical_format"),
        "stack_config_path": str(Path(stack_path).expanduser()) if stack_path else None,
        "tcad_profile_path": str(Path(tcad_path).expanduser()) if tcad_path else None,
        "raw": raw,
        "db_root": str(root),
    }


def _sensor_id(raw: dict[str, Any], stack_path: str | None) -> str:
    if stack_path:
        return Path(stack_path).stem
    metadata = dict(raw.get("metadata", {}))
    parts = [raw.get("code", ""), metadata.get("manufacturer", ""), metadata.get("device_name", "")]
    return _slugify("_".join(str(part) for part in parts if part))


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return "_".join(part for part in cleaned.split())


imageSensorDBRoot = image_sensor_db_root
imageSensorDBCatalogPath = image_sensor_db_catalog_path
imageSensorDBLoad = image_sensor_db_load
imageSensorDBRecords = image_sensor_db_records
imageSensorDBGet = image_sensor_db_get
imageSensorDBParameters = image_sensor_db_parameters
imageSensorDBSummary = image_sensor_db_summary
