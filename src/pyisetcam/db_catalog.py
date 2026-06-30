"""Unified CameraE2E database and parameter catalog.

The catalog is a runtime index over data sources that feed CameraE2E blocks:
lens prescriptions/PSFs, sensor optical/electrical LUTs, HW ISP profiles,
perception model profiles, upstream ISETCam assets, and parity baselines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .assets import DEFAULT_UPSTREAM_SHA
from .fdtd_sensor import fdtd_sensor_default_lut_path, fdtd_sensor_lut_load, fdtd_sensor_lut_summary
from .hwisp_db import hw_isp_profile_names
from .lens_patents import (
    lens_patent_camerae2e_manifest,
    lens_patent_db_summary,
    lens_patent_default_data_dir,
    lens_patent_default_db_path,
    lens_patent_raytrace_psf_manifest,
)
from .task_perception import task_model_profile_names
from .tcad_sensor import tcad_sensor_db_load, tcad_sensor_default_paths, tcad_sensor_validate


@dataclass(frozen=True)
class CameraE2EDBEntry:
    """One CameraE2E database/LUT/model-profile catalog entry."""

    name: str
    family: str
    role: str
    status: str
    path: Path | None
    description: str
    parameter_hint: str
    parameters: dict[str, Any] = field(default_factory=dict)
    env_vars: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "role": self.role,
            "status": self.status,
            "path": None if self.path is None else str(self.path),
            "description": self.description,
            "parameter_hint": self.parameter_hint,
            "parameters": _jsonable(self.parameters),
            "env_vars": list(self.env_vars),
            "tags": list(self.tags),
            "metadata": _jsonable(self.metadata),
        }


def camerae2e_db_catalog(*, include_missing: bool = True) -> list[CameraE2EDBEntry]:
    """Return all known CameraE2E data sources as normalized catalog entries."""

    entries: list[CameraE2EDBEntry] = []
    entries.extend(_lens_entries())
    entries.extend(_sensor_entries())
    entries.extend(_hwisp_entries())
    entries.extend(_perception_entries())
    entries.extend(_upstream_entries())
    entries.extend(_parity_entries())
    if not include_missing:
        entries = [entry for entry in entries if entry.status != "missing"]
    return sorted(entries, key=lambda item: (item.family, item.role, item.name))


def camerae2e_db_search(
    query: str | None = None,
    *,
    family: str | None = None,
    role: str | None = None,
    status: str | None = None,
    tags: Iterable[str] | None = None,
    include_missing: bool = True,
) -> list[dict[str, Any]]:
    """Search CameraE2E DB entries and return JSON-friendly dictionaries."""

    tag_set = {str(tag).lower() for tag in tags or ()}
    query_text = "" if query is None else str(query).lower()
    results: list[CameraE2EDBEntry] = []
    for entry in camerae2e_db_catalog(include_missing=include_missing):
        if family is not None and entry.family != family:
            continue
        if role is not None and entry.role != role:
            continue
        if status is not None and entry.status != status:
            continue
        entry_tags = {tag.lower() for tag in entry.tags}
        if tag_set and not tag_set <= entry_tags:
            continue
        haystack = " ".join(
            [
                entry.name,
                entry.family,
                entry.role,
                entry.status,
                entry.description,
                entry.parameter_hint,
                " ".join(entry.tags),
            ]
        ).lower()
        if query_text and query_text not in haystack:
            continue
        results.append(entry)
    return [entry.to_dict() for entry in results]


def camerae2e_db_get(name: str) -> dict[str, Any]:
    """Return one catalog entry by name as a JSON-friendly dictionary."""

    entry = _entry_by_name(name)
    return entry.to_dict()


def camerae2e_db_parameters(name: str) -> dict[str, Any]:
    """Return parameter values intended for direct CameraE2E API use."""

    return dict(_entry_by_name(name).parameters)


def camerae2e_db_summary() -> dict[str, Any]:
    """Return a compact summary of available CameraE2E DB entries."""

    entries = camerae2e_db_catalog()
    families: dict[str, dict[str, int]] = {}
    statuses: dict[str, int] = {}
    for entry in entries:
        family_counts = families.setdefault(entry.family, {})
        family_counts[entry.status] = family_counts.get(entry.status, 0) + 1
        statuses[entry.status] = statuses.get(entry.status, 0) + 1
    return {
        "total": len(entries),
        "families": families,
        "statuses": statuses,
        "active": [entry.name for entry in entries if entry.status == "active"],
        "available": [entry.name for entry in entries if entry.status in {"active", "available", "fallback"}],
    }


def _entry_by_name(name: str) -> CameraE2EDBEntry:
    key = str(name)
    for entry in camerae2e_db_catalog():
        if entry.name == key:
            return entry
    available = ", ".join(entry.name for entry in camerae2e_db_catalog(include_missing=False))
    raise KeyError(f"Unknown CameraE2E DB entry {name!r}. Available entries: {available}")


def _lens_entries() -> list[CameraE2EDBEntry]:
    entries: list[CameraE2EDBEntry] = []
    active_data_dir = lens_patent_default_data_dir()
    active_db = lens_patent_default_db_path()
    raytrace_dir = active_data_dir / "raytrace_psf"
    highres_dir = active_data_dir / "raytrace_psf_highres"
    lens_meta: dict[str, Any] = {}
    try:
        summary = lens_patent_db_summary(active_db)
        lens_meta["summary"] = summary
    except Exception as exc:  # pragma: no cover - defensive catalog path.
        lens_meta["load_error"] = str(exc)
    try:
        lens_meta["camerae2e_manifest"] = lens_patent_camerae2e_manifest(db_path=active_db)
    except Exception:
        pass
    try:
        lens_meta["raytrace_psf_summary"] = lens_patent_raytrace_psf_manifest(raytrace_dir).get("summary", {})
    except Exception:
        pass
    if (highres_dir / "manifest.json").exists():
        try:
            lens_meta["highres_psf_summary"] = lens_patent_raytrace_psf_manifest(highres_dir).get("summary", {})
        except Exception:
            pass

    entries.append(
        CameraE2EDBEntry(
            name="lens_patents_active",
            family="lens",
            role="optics-prescription-and-psf",
            status="active" if active_db.exists() else "missing",
            path=active_db,
            description="Active lens patent SQLite DB and RayOptics PSF package used by lens_patent_* APIs.",
            parameter_hint="Use db_path for lens_patent_search/get/optics and psf_dir/highres_psf_dir for lens_patent_raytrace_optics.",
            parameters={
                "data_dir": active_data_dir,
                "db_path": active_db,
                "psf_dir": raytrace_dir,
                "highres_psf_dir": highres_dir,
            },
            env_vars=("PYISETCAM_LENS_PATENT_DB", "PYISETCAM_LENS_DB_ROOT", "PYISETCAM_LENS_PATENT_PSF_DIR"),
            tags=("lens", "rayoptics", "psf", "sqlite", "active"),
            metadata=lens_meta,
        )
    )

    bundled_dir = Path(__file__).resolve().parent / "data" / "lens_patents"
    bundled_db = bundled_dir / "lens_patent_simulation_v6.sqlite"
    entries.append(
        CameraE2EDBEntry(
            name="lens_patents_bundled_v6",
            family="lens",
            role="fallback-optics-prescription-and-psf",
            status="active" if active_db == bundled_db else "fallback" if bundled_db.exists() else "missing",
            path=bundled_db,
            description="Bundled v6 Lens DB fallback shipped with pyisetcam package data.",
            parameter_hint="Use when external RayOptics v9 package is unavailable or when a reproducible bundled fallback is needed.",
            parameters={
                "data_dir": bundled_dir,
                "db_path": bundled_db,
                "psf_dir": bundled_dir / "raytrace_psf",
                "highres_psf_dir": bundled_dir / "raytrace_psf_highres",
            },
            env_vars=(),
            tags=("lens", "rayoptics", "fallback", "sqlite"),
            metadata=_lens_db_metadata(bundled_db, bundled_dir),
        )
    )

    rayoptics_v9_dir = Path.home() / "RayOptics" / "CameraE2E_Lens_DB_v9_20260627" / "data" / "lens_patents"
    rayoptics_v9_db = rayoptics_v9_dir / "lens_patent_simulation_v9.sqlite"
    entries.append(
        CameraE2EDBEntry(
            name="rayoptics_lens_db_v9",
            family="lens",
            role="external-optics-prescription-and-psf",
            status="available" if rayoptics_v9_db.exists() else "missing",
            path=rayoptics_v9_db,
            description="External RayOptics v9 Lens DB package prepared for CameraE2E.",
            parameter_hint="Set PYISETCAM_LENS_DB_ROOT to this package root, or pass db_path/psf_dir explicitly.",
            parameters={
                "data_dir": rayoptics_v9_dir,
                "db_path": rayoptics_v9_db,
                "psf_dir": rayoptics_v9_dir / "raytrace_psf",
                "highres_psf_dir": rayoptics_v9_dir / "raytrace_psf_highres",
            },
            env_vars=("PYISETCAM_LENS_DB_ROOT",),
            tags=("lens", "rayoptics", "external", "v9", "sqlite", "psf"),
            metadata={**_lens_db_metadata(rayoptics_v9_db, rayoptics_v9_dir), "selected_as_active": active_db == rayoptics_v9_db},
        )
    )
    return entries


def _lens_db_metadata(db_path: Path, data_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if db_path.exists():
        try:
            metadata["summary"] = lens_patent_db_summary(db_path)
        except Exception as exc:
            metadata["summary_error"] = str(exc)
    for key, manifest_path in {
        "raytrace_psf_summary": data_dir / "raytrace_psf" / "manifest.json",
        "highres_psf_summary": data_dir / "raytrace_psf_highres" / "manifest.json",
    }.items():
        if manifest_path.exists():
            try:
                metadata[key] = json.loads(manifest_path.read_text(encoding="utf-8")).get("summary", {})
            except Exception as exc:
                metadata[f"{key}_error"] = str(exc)
    return metadata


def _sensor_entries() -> list[CameraE2EDBEntry]:
    entries: list[CameraE2EDBEntry] = []
    lut_path = fdtd_sensor_default_lut_path()
    lut_meta: dict[str, Any] = {}
    if lut_path is not None and lut_path.exists():
        try:
            lut = fdtd_sensor_lut_load(lut_path)
            lut_meta["summary"] = fdtd_sensor_lut_summary(lut)
        except Exception as exc:
            lut_meta["load_error"] = str(exc)
    entries.append(
        CameraE2EDBEntry(
            name="fdtd_sensor_lut_active",
            family="sensor",
            role="optical-response-lut",
            status="active" if lut_path is not None and lut_path.exists() else "missing",
            path=lut_path,
            description="Active FDTD-informed sensor optical-response LUT used for QE/RI/crosstalk proxy configuration.",
            parameter_hint="Use lut_path with fdtd_sensor_config(lut_path) and sensor_attach_fdtd_lut.",
            parameters={"lut_path": lut_path, "config": {"lut": lut_path, "enabled": True}},
            env_vars=("PYISETCAM_FDTD_LUT_PATH", "PYISETCAM_FDTD_ROOT"),
            tags=("sensor", "fdtd", "qe", "relative-illumination", "crosstalk", "lut"),
            metadata=lut_meta,
        )
    )

    root = Path(os.environ.get("PYISETCAM_FDTD_ROOT", "/Users/seongcheoljeong/FDTD")).expanduser()
    sensor_catalog = root / "sensor_db" / "sensor_catalog.json"
    stack_dir = root / "sensor_db" / "generated_stack_configs"
    entries.append(
        CameraE2EDBEntry(
            name="fdtd_sensor_stack_catalog",
            family="sensor",
            role="sensor-stack-parameter-catalog",
            status="available" if sensor_catalog.exists() else "missing",
            path=sensor_catalog,
            description="External FDTD sensor stack catalog and generated layer-stack configs.",
            parameter_hint="Use catalog_path to choose a sensor stack, then feed selected FDTD/TCAD artifacts into sensor config.",
            parameters={"catalog_path": sensor_catalog, "stack_config_dir": stack_dir},
            env_vars=("PYISETCAM_FDTD_ROOT",),
            tags=("sensor", "fdtd", "stack", "catalog", "external"),
            metadata={
                "stack_config_count": _count_files(stack_dir, "*.json"),
                "csv_path": root / "sensor_db" / "sensor_catalog.csv",
                "validation_path": root / "sensor_db" / "validation.json",
            },
        )
    )

    defaults = tcad_sensor_default_paths(root)
    validation: dict[str, Any] = {}
    try:
        db = tcad_sensor_db_load(root=root)
        validation = tcad_sensor_validate(db)
    except Exception as exc:
        validation = {"load_error": str(exc)}
    entries.append(
        CameraE2EDBEntry(
            name="tcad_sensor_db_active",
            family="sensor",
            role="electrical-collection-lut",
            status="active" if validation.get("ok") or validation.get("status") else "missing",
            path=Path(defaults["root"]),
            description="Active DEVSIM/TCAD sensor collection DB joining FDTD generation maps with split-PD electrical summaries.",
            parameter_hint="Use these paths with tcad_sensor_db_load(...) or sensor_attach_tcad_lut.",
            parameters=defaults,
            env_vars=("PYISETCAM_FDTD_ROOT",),
            tags=("sensor", "tcad", "devsim", "collection", "split-pd"),
            metadata={"validation": validation},
        )
    )
    return entries


def _hwisp_entries() -> list[CameraE2EDBEntry]:
    root = Path(os.environ.get("PYISETCAM_HWISP_DB", Path(__file__).resolve().parent / "data" / "hwisp")).expanduser()
    names: list[str] = []
    try:
        names = hw_isp_profile_names(root)
    except Exception:
        names = []
    return [
        CameraE2EDBEntry(
            name="hwisp_parameter_profiles",
            family="isp",
            role="timing-and-3a-parameter-profiles",
            status="active" if names else "missing",
            path=root,
            description="HW ISP timing, transport, and 3A profile DB for system-level latency simulation.",
            parameter_hint="Use profile_name with hw_isp_config_from_profile(profile_name).",
            parameters={"db_path": root, "profile_names": names, "default_profile": names[0] if names else None},
            env_vars=("PYISETCAM_HWISP_DB",),
            tags=("isp", "hwisp", "latency", "3a", "profile"),
            metadata={"profile_count": len(names)},
        )
    ]


def _perception_entries() -> list[CameraE2EDBEntry]:
    profiles_path = Path(__file__).resolve().parent / "data" / "task_perception" / "model_profiles.json"
    names = task_model_profile_names()
    cache_dir = Path(
        os.environ.get("PYISETCAM_TASK_MODEL_CACHE", Path.home() / ".cache" / "pyisetcam" / "task_perception" / "yolo")
    ).expanduser()
    return [
        CameraE2EDBEntry(
            name="task_perception_model_profiles",
            family="perception",
            role="model-profile-catalog",
            status="active" if profiles_path.exists() else "missing",
            path=profiles_path,
            description="Task perception model profile DB covering YOLO, tracking, detection, segmentation, pose, OBB, and optional adapters.",
            parameter_hint="Use profile_name with task_model_config_from_profile(profile_name), then task_model_from_config.",
            parameters={"profiles_path": profiles_path, "profile_names": names, "cache_dir": cache_dir},
            env_vars=("PYISETCAM_TASK_MODEL_CACHE",),
            tags=("perception", "yolo", "detection", "segmentation", "tracking", "model-profile"),
            metadata={"profile_count": len(names), "cached_model_count": _count_files(cache_dir, "*.pt")},
        )
    ]


def _upstream_entries() -> list[CameraE2EDBEntry]:
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("PYISETCAM_CACHE_ROOT", repo_root / ".cache")).expanduser()
    override = os.environ.get("PYISETCAM_UPSTREAM_ROOT")
    snapshot_root = Path(override).expanduser() if override else cache_root / "upstream" / "isetcam" / DEFAULT_UPSTREAM_SHA
    return [
        CameraE2EDBEntry(
            name="isetcam_upstream_snapshot",
            family="assets",
            role="upstream-matlab-asset-snapshot",
            status="active" if snapshot_root.exists() else "missing",
            path=snapshot_root,
            description="Pinned upstream ISETCam asset snapshot used by AssetStore for MATLAB-origin data files.",
            parameter_hint="Use AssetStore.default(), or pass PYISETCAM_UPSTREAM_ROOT for an explicit snapshot.",
            parameters={"snapshot_root": snapshot_root, "cache_root": cache_root, "sha": DEFAULT_UPSTREAM_SHA},
            env_vars=("PYISETCAM_CACHE_ROOT", "PYISETCAM_UPSTREAM_ROOT"),
            tags=("assets", "upstream", "isetcam", "matlab", "snapshot"),
            metadata={"sha": DEFAULT_UPSTREAM_SHA},
        )
    ]


def _parity_entries() -> list[CameraE2EDBEntry]:
    repo_root = Path(__file__).resolve().parents[2]
    cases_yaml = repo_root / "tests" / "parity" / "cases.yaml"
    baselines_dir = repo_root / "tests" / "parity" / "baselines"
    latest_json = repo_root / "reports" / "parity" / "latest.json"
    return [
        CameraE2EDBEntry(
            name="matlab_parity_baselines",
            family="parity",
            role="matlab-baseline-evidence",
            status="available" if cases_yaml.exists() and baselines_dir.exists() else "missing",
            path=baselines_dir,
            description="Curated MATLAB baseline .mat DB and parity case registry for regression/evidence reports.",
            parameter_hint="Use cases_yaml and baselines_dir with tests/parity or tools/parity_report.py.",
            parameters={"cases_yaml": cases_yaml, "baselines_dir": baselines_dir, "latest_report": latest_json},
            env_vars=("PYISETCAM_RUN_PARITY",),
            tags=("parity", "matlab", "baseline", "evidence"),
            metadata={"baseline_count": _count_files(baselines_dir, "*.mat"), "latest_report_exists": latest_json.exists()},
        )
    ]


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    return sum(1 for _ in root.glob(pattern))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


cameraE2EDBCatalog = camerae2e_db_catalog
cameraE2EDBSearch = camerae2e_db_search
cameraE2EDBGet = camerae2e_db_get
cameraE2EDBParameters = camerae2e_db_parameters
cameraE2EDBSummary = camerae2e_db_summary
