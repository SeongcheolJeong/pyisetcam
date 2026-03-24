"""Generate a machine-readable migration gap ledger and summary report.

The ledger is written as JSON-compatible YAML so it stays dependency-free while
remaining readable by YAML tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / ".cache" / "upstream" / "isetcam"
CASES_PATH = REPO_ROOT / "tests" / "parity" / "cases.yaml"
LEDGER_PATH = REPO_ROOT / "docs" / "migration-gap-ledger.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "migration-gap" / "latest.json"
README_PATH = REPO_ROOT / "README.md"
IMPLEMENTATION_PLAN_PATH = REPO_ROOT / "docs" / "implementation-plan.md"
SRC_ROOT = REPO_ROOT / "src" / "pyisetcam"
TEST_ROOT = REPO_ROOT / "tests" / "unit"
SUPPORTED_CODE_SUFFIXES = {".m", ".mlapp", ".fig"}
SHORT_SIGNAL_PREFIXES = (
    "oi",
    "ip",
    "lf",
    "rt",
    "wvf",
    "psf",
    "si",
    "ie",
    "sc",
    "ml",
    "vc",
    "hc",
    "xyy",
    "xyz",
    "lms",
)

FAMILY_SURFACE_MAP = {
    "camera": ["pyisetcam.camera"],
    "color": ["pyisetcam.color", "pyisetcam.illuminant"],
    "displays": ["pyisetcam.display"],
    "human": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "imgproc": ["pyisetcam.ip"],
    "imgproc/binning": ["pyisetcam.ip"],
    "imgproc/demosaic": ["pyisetcam.ip"],
    "imgproc/illuminant": ["pyisetcam.ip", "pyisetcam.illuminant"],
    "main": ["pyisetcam.session"],
    "metrics": ["pyisetcam.metrics", "pyisetcam.scielab", "pyisetcam.iso"],
    "metrics/cielab": ["pyisetcam.scielab"],
    "metrics/iso": ["pyisetcam.iso"],
    "metrics/scielab": ["pyisetcam.scielab"],
    "opticalimage": ["pyisetcam.optics"],
    "opticalimage/depth": ["pyisetcam.optics"],
    "opticalimage/optics": ["pyisetcam.optics"],
    "opticalimage/raytrace": ["pyisetcam.optics"],
    "opticalimage/wavefront": ["pyisetcam.optics"],
    "scene": ["pyisetcam.scene"],
    "scene/depth": ["pyisetcam.scene"],
    "scene/illumination": ["pyisetcam.scene", "pyisetcam.illuminant"],
    "scene/imgtargets": ["pyisetcam.scene"],
    "scene/macbeth": ["pyisetcam.scene"],
    "scene/pattern": ["pyisetcam.scene"],
    "scene/reflectance": ["pyisetcam.scene", "pyisetcam.fileio"],
    "scripts/color": ["pyisetcam.color", "pyisetcam.metrics", "pyisetcam.scene"],
    "scripts/display": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    "scripts/human": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "scripts/image": ["pyisetcam.ip", "pyisetcam.fileio"],
    "scripts/metrics": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "scripts/optics": ["pyisetcam.optics"],
    "scripts/scene": ["pyisetcam.scene"],
    "scripts/sensor": ["pyisetcam.sensor", "pyisetcam.camera"],
    "scripts/utility": ["pyisetcam.fileio", "pyisetcam.plotting", "pyisetcam.session"],
    "sensor": ["pyisetcam.sensor"],
    "sensor/binning": ["pyisetcam.sensor"],
    "sensor/cfadesign": ["pyisetcam.sensor"],
    "sensor/microlens": ["pyisetcam.sensor"],
    "sensor/models": ["pyisetcam.sensor"],
    "sensor/pixel": ["pyisetcam.sensor"],
    "sensor/simulation": ["pyisetcam.sensor"],
    "tutorials/camera": ["pyisetcam.camera"],
    "tutorials/color": ["pyisetcam.color", "pyisetcam.metrics"],
    "tutorials/display": ["pyisetcam.display"],
    "tutorials/image": ["pyisetcam.ip", "pyisetcam.fileio"],
    "tutorials/metrics": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "tutorials/oi": ["pyisetcam.optics"],
    "tutorials/optics": ["pyisetcam.optics"],
    "tutorials/scene": ["pyisetcam.scene"],
    "tutorials/sensor": ["pyisetcam.sensor"],
    "utility": ["pyisetcam.utils"],
    "utility/file": ["pyisetcam.fileio"],
    "utility/image": ["pyisetcam.utils", "pyisetcam.fileio"],
    "utility/plots": ["pyisetcam.plotting"],
    "utility/tablebase": ["pyisetcam.ptable"],
}

FAMILY_TEST_MAP = {
    "camera": ["tests/unit/test_pipeline.py"],
    "color": ["tests/unit/test_pipeline.py", "tests/unit/test_metrics.py"],
    "displays": ["tests/unit/test_display.py", "tests/unit/test_pipeline.py"],
    "imgproc": ["tests/unit/test_pipeline.py", "tests/unit/test_ip.py"],
    "main": ["tests/unit/test_session.py"],
    "metrics": ["tests/unit/test_metrics.py", "tests/unit/test_pipeline.py"],
    "opticalimage": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "scene": [
        "tests/unit/test_scene.py",
        "tests/unit/test_pipeline.py",
        "tests/parity/test_parity_harness.py",
    ],
    "scripts": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "sensor": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "tutorials": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "utility/file": ["tests/unit/test_fileio.py"],
    "utility/plots": ["tests/unit/test_plotting.py"],
    "utility/tablebase": ["tests/unit/test_ptable.py"],
}

GUI_KEYWORDS = ("plot", "figure", "window", "handle", "refresh", "show", "app")
HEADLESS_KEYWORDS = GUI_KEYWORDS + ("session", "roi", "vc", "ie")
CORE_PRIORITY_FAMILIES = {
    "camera",
    "color",
    "displays",
    "imgproc",
    "metrics",
    "opticalimage",
    "scene",
    "sensor",
    "scripts/metrics",
    "scripts/optics",
    "scripts/scene",
    "scripts/sensor",
    "tutorials/metrics",
    "tutorials/oi",
    "tutorials/optics",
    "tutorials/scene",
    "tutorials/sensor",
}
MID_PRIORITY_FAMILIES = {
    "human",
    "scripts/display",
    "scripts/human",
    "scripts/image",
    "scripts/utility",
    "tutorials/camera",
    "tutorials/color",
    "tutorials/display",
    "tutorials/image",
    "utility",
    "utility/file",
    "utility/image",
    "utility/plots",
    "utility/tablebase",
}

UPSTREAM_STATUS_OVERRIDES: dict[str, dict[str, Any]] = {
    "imgproc/demosaic/bayerIndices.m": {
        "status": "ported",
        "note": "The low-level MATLAB Bayer-index helper is already covered by the shared internal `_bayer_indices` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/ieBilinear.m": {
        "status": "ported",
        "note": "The low-level MATLAB bilinear demosaic helper is already covered by the shared internal `_ie_bilinear` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/mosaicConverter.m": {
        "status": "ported",
        "note": "The low-level MATLAB Bayer-pattern conversion helper is already covered by the shared internal `_mosaic_converter` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/imageColorBalanceDeprecated.m": {
        "status": "ported",
        "note": "The deprecated MATLAB color-balance gateway is already covered by the Python `image_color_balance(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/faultypixel/FaultyPixelCorrection.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as requiring a rewrite, so it is treated as obsolete rather than actionable headless API debt.",
        "module_hits": [],
    },
    "imgproc/demosaic/lcc1.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked not fully implemented and questions whether the patented method should even remain in the simulator.",
        "module_hits": [],
    },
    "imgproc/demosaic/shtlin.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked not fully implemented and left with TODO placeholders, so it is treated as legacy experimental code rather than supported headless API debt.",
        "module_hits": [],
    },
    "imgproc/demosaic/shtlog.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked as needing a complete rewrite, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispMousePan.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a pure Light Field Toolbox display interaction helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispSetup.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a pure Light Field Toolbox display setup helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispVidCirc.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a display/video viewer helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/imageIlluminantCorrectionDeprecated.m": {
        "status": "ported",
        "note": "The deprecated MATLAB illuminant-correction file is already covered by the Python `image_illuminant_correction(...)` helper.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/lightfield/LFMicrolensGeometry.m": {
        "status": "out_of_scope",
        "note": "The upstream file opens a visualization figure for microlens geometry and is treated as GUI-only rather than a headless compute surface.",
        "module_hits": [],
    },
    "imgproc/ipGet.m": {
        "status": "ported",
        "note": "The direct MATLAB image-processor getter surface is already covered by `ip_get(...)` / `ipGet(...)`.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/ipKeyPress.m": {
        "status": "out_of_scope",
        "note": "GUI keypress handlers remain outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/ipSet.m": {
        "status": "ported",
        "note": "The direct MATLAB image-processor setter surface is already covered by `ip_set(...)` / `ipSet(...)`.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/vcimageKeyPress.m": {
        "status": "out_of_scope",
        "note": "GUI keypress handlers remain outside the headless migration target.",
        "module_hits": [],
    },
    "color/transforms/colorTransformMatrixCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script used to derive static transform matrices, not a reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtBlockPartition.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script-style diagnostic prototype with display side effects rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtImagePSFFieldHeight.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked as not functional yet, so it remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/oiImageInitCustomStrings.m": {
        "status": "out_of_scope",
        "note": "The upstream file only initializes GUI/window label strings and remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/oiExtractMask.m": {
        "status": "out_of_scope",
        "note": "The pinned upstream file is empty, so it is treated as a non-actionable stub rather than a remaining headless API gap.",
        "module_hits": [],
    },
    "opticalimage/depth/oiDepthOverlay.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and directs callers to `oiPlot(..., 'depth map contour', ...)` instead.",
        "module_hits": [],
    },
    "opticalimage/wavefront/RandomDirtyApertureDeprecate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a deprecated Computer Vision Toolbox dirty-aperture generator rather than a reusable headless WVF API surface.",
        "module_hits": [],
    },
    "opticalimage/wavefront/underDevelopment_wavefront/wvfComputeOptimizedConePSF.m": {
        "status": "out_of_scope",
        "note": "The upstream file lives under underDevelopment_wavefront and remains an experimental prototype outside the supported headless target.",
        "module_hits": [],
    },
    "opticalimage/wavefront/underDevelopment_wavefront/wvfComputeOptimizedPSF.m": {
        "status": "out_of_scope",
        "note": "The upstream file lives under underDevelopment_wavefront and remains an experimental prototype outside the supported headless target.",
        "module_hits": [],
    },
    "opticalimage/optics/airyDiskPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly deprecated and only adjusts live MATLAB figure state for an existing PSF plot.",
        "module_hits": [],
    },
    "opticalimage/optics/defocusMTF.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and directs callers to newer defocus/OTF helpers instead.",
        "module_hits": [],
    },
    "opticalimage/optics/psfMovie.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven animation helper, which remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/optics/defocus/opticsReducedSFandW20.m": {
        "status": "out_of_scope",
        "note": "The upstream helper is internally incomplete, references undefined state, and does not expose a usable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/sensorUnitBlock.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked obsolete and does not implement a usable headless API surface.",
        "module_hits": [],
    },
    "sensor/simulation/noise/noiseDarkCurrent.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and does not implement a usable headless API surface.",
        "module_hits": [],
    },
    "sensor/pixel/pixelGUI/pixelGeometryWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a legacy MATLAB figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/pixel/pixelGUI/pixelOEWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a legacy MATLAB figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/microLensWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlOpen.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE window-opening callback and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlRefresh.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUI refresh helper for the microlens window and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlFillMLFromWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file only pulls values from GUIDE controls into a microlens structure and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlFillWindowFromML.m": {
        "status": "out_of_scope",
        "note": "The upstream file only pushes microlens values into GUIDE controls and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlIrradianceImage.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven irradiance-image display helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "scene/pattern/_tests_/sceneFluorescenceChart_test.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a private MATLAB regression script for the fluorescence-chart helper rather than a standalone headless API surface.",
        "module_hits": [],
    },
}


@dataclass(frozen=True)
class InventoryEntry:
    upstream_path: str
    family: str
    kind: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _canonical_signal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _signal_min_length(token: str) -> int:
    normalized = _canonical_signal(token)
    if normalized.startswith(SHORT_SIGNAL_PREFIXES):
        return 4
    return 6


def _json_yaml_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False) + "\n"


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(_read_text(CASES_PATH)).get("cases", [])


def _latest_upstream_snapshot(root: Path) -> Path:
    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not candidates:
        raise FileNotFoundError(f"No upstream snapshots found in {root}")
    return candidates[-1]


def _family_for_relpath(rel_path: Path) -> str:
    parts = [part.lower() for part in rel_path.parts]
    first = parts[0]
    if first in {"scripts", "tutorials", "data"} and len(parts) > 1:
        return f"{first}/{parts[1]}"
    if (
        first in {"opticalimage", "scene", "sensor", "imgproc", "metrics", "utility"}
        and len(parts) > 1
    ):
        second = parts[1]
        if second in {
            "binning",
            "cielab",
            "cfadesign",
            "depth",
            "file",
            "gui",
            "illumination",
            "image",
            "imgtargets",
            "iso",
            "macbeth",
            "metricsgui",
            "microlens",
            "models",
            "openexr",
            "optics",
            "pattern",
            "pixel",
            "plots",
            "raytrace",
            "reflectance",
            "scielab",
            "simulation",
            "tablebase",
            "wavefront",
        }:
            return f"{first}/{second}"
    return first


def _kind_for_relpath(rel_path: Path, family: str) -> str:
    suffix = rel_path.suffix.lower()
    parts = [part.lower() for part in rel_path.parts]
    if family.startswith("data/"):
        return "asset"
    if suffix in {".mlapp", ".fig"}:
        return "gui"
    if "gui" in parts or family.endswith("/gui") or parts[0] == "gui":
        return "gui"
    if parts[0] == "scripts":
        return "script"
    if parts[0] == "tutorials":
        return "tutorial"
    if parts[0] in {"utility", "main", "local", "cp"}:
        return "helper"
    return "api"


def _inventory(snapshot: Path) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        rel_path = path.relative_to(snapshot)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_CODE_SUFFIXES:
            continue
        family = _family_for_relpath(rel_path)
        kind = _kind_for_relpath(rel_path, family)
        entries.append(InventoryEntry(upstream_path=rel_path.as_posix(), family=family, kind=kind))

    data_root = snapshot / "data"
    if data_root.exists():
        for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
            rel = child.relative_to(snapshot)
            entries.append(
                InventoryEntry(
                    upstream_path=f"{rel.as_posix()}/",
                    family=_family_for_relpath(rel),
                    kind="asset",
                )
            )
    return entries


def _family_surface(family: str) -> list[str]:
    for candidate in sorted(FAMILY_SURFACE_MAP, key=len, reverse=True):
        if family == candidate or family.startswith(f"{candidate}/"):
            return FAMILY_SURFACE_MAP[candidate]
    return []


def _family_tests(family: str) -> list[str]:
    for candidate in sorted(FAMILY_TEST_MAP, key=len, reverse=True):
        if family == candidate or family.startswith(f"{candidate}/"):
            return FAMILY_TEST_MAP[candidate]
    return []


def _module_texts() -> dict[str, dict[str, str]]:
    module_texts: dict[str, dict[str, str]] = {}
    for path in sorted(SRC_ROOT.glob("*.py")):
        raw_text = _read_text(path).lower()
        module_texts[path.stem] = {
            "raw": raw_text,
            "canonical": _canonical_signal(raw_text),
        }
    return module_texts


def _module_hits(stem: str, module_texts: dict[str, dict[str, str]]) -> list[str]:
    if len(_canonical_signal(stem)) < _signal_min_length(stem):
        return []
    canonical_stem = _canonical_signal(stem)
    hits = [
        f"pyisetcam.{name}"
        for name, text_bundle in module_texts.items()
        if canonical_stem in text_bundle["canonical"]
    ]
    return hits[:4]


def _combined_test_text() -> str:
    chunks = [_read_text(CASES_PATH).lower()]
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        chunks.append(_read_text(path).lower())
    return "\n".join(chunks)


def _docs_text() -> str:
    return "\n".join((_read_text(README_PATH), _read_text(IMPLEMENTATION_PLAN_PATH))).lower()


def _contains_signal(text: str, token: str) -> bool:
    if len(_canonical_signal(token)) < _signal_min_length(token):
        return False
    if token in text:
        return True
    return _canonical_signal(token) in _canonical_signal(text)


def _status_and_note(
    entry: InventoryEntry,
    *,
    docs_text: str,
    tests_text: str,
    case_text: str,
    module_texts: dict[str, dict[str, str]],
) -> tuple[str, str, list[str]]:
    basename = Path(entry.upstream_path).name.lower()
    stem = Path(entry.upstream_path.rstrip("/")).stem.lower()
    surfaces = _family_surface(entry.family)
    module_hits = _module_hits(stem, module_texts)
    direct_doc_hit = basename in docs_text
    explicit_hit = (
        direct_doc_hit
        or _contains_signal(docs_text, stem)
        or _contains_signal(tests_text, stem)
        or _contains_signal(case_text, stem)
    )
    has_surface = bool(surfaces)

    override = UPSTREAM_STATUS_OVERRIDES.get(entry.upstream_path)
    if override is not None:
        return (
            str(override["status"]),
            str(override["note"]),
            list(override.get("module_hits", [])),
        )

    if entry.kind == "gui":
        return (
            "out_of_scope",
            "GUI/App Designer and figure fidelity remain outside the headless migration target.",
            module_hits,
        )
    if entry.kind == "asset":
        return (
            "partial",
            "Runtime still relies on the upstream asset bundle rather than a fully migrated "
            "Python-native asset pipeline.",
            module_hits,
        )
    if entry.upstream_path.startswith("metrics/ISO/sfrmat4v5/"):
        if stem in {"sfrmat4", "sfrmat4simplified", "clip"}:
            return (
                "ported",
                "The MATLAB-private sfrmat4v5 algorithm helper is already absorbed into the "
                "current headless slanted-edge implementation in `pyisetcam.iso`.",
                ["pyisetcam.iso"],
            )
        return (
            "out_of_scope",
            "The remaining sfrmat4v5 file-I/O and GUI support helpers are not migrated as "
            "standalone headless Python APIs.",
            module_hits,
        )
    if entry.family.startswith("opticalimage/raytrace"):
        if explicit_hit and entry.kind in {"script", "tutorial"}:
            return (
                "parity",
                "Explicit ray-trace script parity exists, but the broader ray-trace family "
                "remains partial.",
                module_hits,
            )
        if module_hits or has_surface:
            return (
                "partial",
                "Ray-trace support exists in Python, but the family is still documented as "
                "partial.",
                module_hits,
            )
        return (
            "missing",
            "No direct Python ray-trace surface was detected for this upstream path.",
            module_hits,
        )
    if explicit_hit:
        if entry.kind in {"script", "tutorial"}:
            return (
                "parity",
                "Curated parity or direct script/tutorial evidence exists for this upstream "
                "workflow.",
                module_hits,
            )
        if entry.kind == "helper" and any(keyword in stem for keyword in HEADLESS_KEYWORDS):
            return (
                "headless_only",
                "The MATLAB workflow is covered through headless/session wrappers instead of "
                "GUI fidelity.",
                module_hits,
            )
        return (
            "parity",
            "The upstream API surface is referenced by the current parity-backed migration "
            "coverage.",
            module_hits,
        )
    if module_hits:
        if (
            entry.kind == "helper"
            or any(keyword in stem for keyword in HEADLESS_KEYWORDS)
            or entry.family.startswith("main")
        ):
            return (
                "headless_only",
                "A headless Python/session wrapper exists, but not full MATLAB GUI parity.",
                module_hits,
            )
        return (
            "ported",
            "A direct Python implementation surface was detected for this upstream path.",
            module_hits,
        )
    if has_surface:
        return (
            "partial",
            "The family has a Python surface, but this specific path lacks explicit parity or "
            "direct implementation evidence.",
            module_hits,
        )
    return ("missing", "No usable Python surface was detected for this upstream path.", module_hits)


def _priority_for(entry: InventoryEntry, status: str) -> str:
    if status in {"parity", "ported", "out_of_scope"}:
        return "P3"
    if status == "headless_only":
        return "P2"
    family = entry.family
    if family in CORE_PRIORITY_FAMILIES or any(
        family.startswith(f"{prefix}/") for prefix in CORE_PRIORITY_FAMILIES
    ):
        return "P0"
    if family in MID_PRIORITY_FAMILIES or any(
        family.startswith(f"{prefix}/") for prefix in MID_PRIORITY_FAMILIES
    ):
        return "P1"
    if "raytrace" in family or family.startswith("main") or family.startswith("utility/"):
        return "P2"
    if entry.kind in {"script", "tutorial"}:
        return "P1"
    return "P2"


def _python_surface(entry: InventoryEntry, module_hits: list[str]) -> str:
    surfaces = _family_surface(entry.family)
    merged: list[str] = []
    for item in surfaces + module_hits:
        if item not in merged:
            merged.append(item)
    return ", ".join(merged)


def _tests_for(entry: InventoryEntry, status: str) -> list[str]:
    tests = list(_family_tests(entry.family))
    if status == "parity":
        if "tests/parity/cases.yaml" not in tests:
            tests.append("tests/parity/cases.yaml")
        if "tests/parity/test_parity_harness.py" not in tests:
            tests.append("tests/parity/test_parity_harness.py")
    return tests


def _status_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["status"] for entry in entries)
    return dict(sorted(counter.items()))


def _kind_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["kind"] for entry in entries)
    return dict(sorted(counter.items()))


def _priority_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["priority"] for entry in entries)
    return dict(sorted(counter.items()))


def _family_summary(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["family"]].append(entry)
    summary: dict[str, dict[str, Any]] = {}
    for family, family_entries in sorted(grouped.items()):
        summary[family] = {
            "total": len(family_entries),
            "by_status": _status_counter(family_entries),
            "by_kind": _kind_counter(family_entries),
            "highest_open_priority": min(
                (
                    entry["priority"]
                    for entry in family_entries
                    if entry["status"] not in {"parity", "ported", "out_of_scope"}
                ),
                default="P3",
            ),
        }
    return summary


def _delta_counts(current: dict[str, int], previous: dict[str, int] | None) -> dict[str, int]:
    previous = previous or {}
    keys = sorted(set(current) | set(previous))
    return {key: int(current.get(key, 0) - previous.get(key, 0)) for key in keys}


def _load_previous_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError:
        return None


def _exception_site_summary() -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for path in sorted(SRC_ROOT.glob("*.py")):
        text = _read_text(path)
        summary[path.stem] = {
            "unsupported_option_error_sites": len(re.findall(r"UnsupportedOptionError\(", text)),
            "not_implemented_error_sites": len(re.findall(r"NotImplementedError\(", text)),
        }
    return dict(sorted(summary.items()))


def _top_open_families(
    family_summary: dict[str, dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for family, summary in family_summary.items():
        open_count = sum(
            count
            for status, count in summary["by_status"].items()
            if status not in {"parity", "ported", "out_of_scope"}
        )
        rows.append(
            {
                "family": family,
                "open_entries": open_count,
                "highest_open_priority": summary["highest_open_priority"],
                "by_status": summary["by_status"],
            }
        )
    rows.sort(
        key=lambda item: (
            item["highest_open_priority"],
            -item["open_entries"],
            item["family"],
        )
    )
    return rows[:limit]


def _ledger_entries(snapshot: Path) -> list[dict[str, Any]]:
    docs_text = _docs_text()
    tests_text = _combined_test_text()
    case_text = _read_text(CASES_PATH).lower()
    module_texts = _module_texts()
    entries: list[dict[str, Any]] = []
    for inventory_entry in _inventory(snapshot):
        status, notes, module_hits = _status_and_note(
            inventory_entry,
            docs_text=docs_text,
            tests_text=tests_text,
            case_text=case_text,
            module_texts=module_texts,
        )
        entries.append(
            {
                "upstream_path": inventory_entry.upstream_path,
                "family": inventory_entry.family,
                "kind": inventory_entry.kind,
                "status": status,
                "python_surface": _python_surface(inventory_entry, module_hits),
                "tests": _tests_for(inventory_entry, status),
                "notes": notes,
                "priority": _priority_for(inventory_entry, status),
            }
        )
    return entries


def _build_outputs(
    snapshot: Path,
    previous_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = _ledger_entries(snapshot)
    family_summary = _family_summary(entries)
    status_counts = _status_counter(entries)
    kind_counts = _kind_counter(entries)
    priority_counts = _priority_counter(entries)
    previous_summary = {} if previous_report is None else previous_report.get("summary", {})
    previous_status_counts = previous_summary.get("status_counts")
    previous_kind_counts = previous_summary.get("kind_counts")
    previous_priority_counts = previous_summary.get("priority_counts")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream_snapshot": snapshot.as_posix(),
        "entry_count": len(entries),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "priority_counts": priority_counts,
        "delta_from_previous": {
            "status_counts": _delta_counts(status_counts, previous_status_counts),
            "kind_counts": _delta_counts(kind_counts, previous_kind_counts),
            "priority_counts": _delta_counts(priority_counts, previous_priority_counts),
        },
        "family_summary": family_summary,
        "top_open_families": _top_open_families(family_summary),
        "exception_sites_by_module": _exception_site_summary(),
    }
    ledger = {
        "generated_at": summary["generated_at"],
        "upstream_snapshot": summary["upstream_snapshot"],
        "summary": {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "generated_at",
                "upstream_snapshot",
            }
        },
        "entries": entries,
    }
    report = {
        "generated_at": summary["generated_at"],
        "upstream_snapshot": summary["upstream_snapshot"],
        "summary": summary,
    }
    return ledger, report


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Migration gap audit")
    print(f"  upstream snapshot: {report['upstream_snapshot']}")
    print(f"  entries: {summary['entry_count']}")
    print(f"  status counts: {summary['status_counts']}")
    print(f"  kind counts: {summary['kind_counts']}")
    print(f"  priority counts: {summary['priority_counts']}")
    print("  top open families:")
    for row in summary["top_open_families"][:10]:
        print(
            f"    - {row['family']}: {row['open_entries']} open, "
            f"highest {row['highest_open_priority']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-snapshot", type=Path, default=None)
    parser.add_argument("--ledger-out", type=Path, default=LEDGER_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-write-ledger", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    snapshot = args.upstream_snapshot or _latest_upstream_snapshot(UPSTREAM_ROOT)
    previous_report = _load_previous_report(args.report_out)
    ledger, report = _build_outputs(snapshot, previous_report)

    if not args.no_write_ledger:
        args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_out.write_text(_json_yaml_dump(ledger), encoding="utf-8")
    if not args.no_write_report:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
