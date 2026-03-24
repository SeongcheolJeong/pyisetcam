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
    "dl",
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
    "imgproc/demosaic/Pocs.m": {
        "status": "ported",
        "note": "The legacy MATLAB POCS demosaic entry point is covered by the Python `pocs(...)` / `Pocs(...)` compatibility wrapper on top of the existing Bayer/CFA helpers.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "imgproc/binning/binAnalog2digital.m": {
        "status": "ported",
        "note": "The legacy MATLAB binning quantizer is already covered by the Python `analog_to_digital(...)` / `analog2digital(...)` compatibility surface.",
        "module_hits": ["pyisetcam.sensor", "pyisetcam.__init__"],
    },
    "imgproc/imageColorBalanceDeprecated.m": {
        "status": "ported",
        "note": "The deprecated MATLAB color-balance gateway is already covered by the Python `image_color_balance(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.ip"],
    },
    "utility/printing/FloydSteinberg.m": {
        "status": "ported",
        "note": "The legacy MATLAB Floyd-Steinberg error-diffusion helper is covered by the Python `floyd_steinberg(...)` / `FloydSteinberg(...)` compatibility wrapper and direct utility regressions.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/printing/HalfToneImage.m": {
        "status": "ported",
        "note": "The legacy MATLAB halftone-cell helper is covered by the Python `half_tone_image(...)` / `HalfToneImage(...)` compatibility wrapper and direct utility regressions.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
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
    "imgproc/openexr/exrreadchannels.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB help shim for a MEX-backed OpenEXR binding, which remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "imgproc/openexr/exrwritechannels.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB help shim for a MEX-backed OpenEXR binding, which remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "imgproc/openexr/make.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a local MATLAB MEX build script for OpenEXR bindings rather than a reusable supported headless API surface.",
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
    "camera/cameraFullReference.m": {
        "status": "ported",
        "note": "The legacy MATLAB full-reference camera benchmark is covered by the Python `camera_full_reference(...)` / `cameraFullReference(...)` wrapper.",
        "module_hits": ["pyisetcam.camera"],
    },
    "camera/cameraPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a plotting gateway stub rather than a reusable supported headless camera API surface.",
        "module_hits": [],
    },
    "camera/cameraMoire.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an exploratory figure-driven moire-analysis workflow with interactive plots rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "camera/cameraVSNR_SL.m": {
        "status": "ported",
        "note": "The legacy MATLAB vSNR wrapper name is covered by the Python `camera_vsnr(...)` / `camera_vsnr_sl(...)` compatibility surface.",
        "module_hits": ["pyisetcam.camera"],
    },
    "camera/cameraWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file only opens MATLAB object windows and remains outside the headless migration target.",
        "module_hits": [],
    },
    "metrics/scielab/ApplyFilters.m": {
        "status": "ported",
        "note": "The legacy MATLAB filter-application helper is covered by the Python `sc_apply_filters(...)` / `ApplyFilters(...)` / `scApplyFilters(...)` compatibility surface.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/scielab/gauss.m": {
        "status": "ported",
        "note": "The direct MATLAB Gaussian-kernel helper is covered by the Python `gauss(...)` compatibility wrapper in `pyisetcam.scielab`.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/scielab/scApplyFilters.m": {
        "status": "ported",
        "note": "The legacy MATLAB S-CIELAB filter-application helper is covered by the Python `sc_apply_filters(...)` / `ApplyFilters(...)` / `scApplyFilters(...)` compatibility surface.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/iePSNR.m": {
        "status": "ported",
        "note": "The direct MATLAB PSNR helper is covered by the Python `peak_signal_to_noise_ratio(...)` / `iePSNR(...)` compatibility surface.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "metrics/ISO/ISOspeedNoise.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching script that expects GUI-managed scene/OI/sensor state rather than a standalone supported headless ISO API.",
        "module_hits": [],
    },
    "utility/tablebase/ieTableGet.m": {
        "status": "ported",
        "note": "The MATLAB metadata-table filter helper is covered by the Python `ie_table_get(...)` / `ieTableGet(...)` compatibility wrapper in `pyisetcam.ptable`.",
        "module_hits": ["pyisetcam.ptable", "pyisetcam.__init__"],
    },
    "color/ieColorTransform.m": {
        "status": "ported",
        "note": "The MATLAB sensor-to-target color transform helper is covered by the Python `ie_color_transform(...)` / `ieColorTransform(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/ieLuminanceFromEnergy.m": {
        "status": "ported",
        "note": "The direct MATLAB luminance-from-energy helper is covered by the Python `luminance_from_energy(...)` / `ieLuminanceFromEnergy(...)` compatibility surface.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/ieLuminanceFromPhotons.m": {
        "status": "ported",
        "note": "The direct MATLAB luminance-from-photons helper is covered by the Python `luminance_from_photons(...)` / `ieLuminanceFromPhotons(...)` compatibility surface.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/transforms/colorTransformMatrixCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script used to derive static transform matrices, not a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/optics/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the optics scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/metrics/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the metrics scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/development/s_sensorIRSimulation.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked `UNDER DEVELOPMENT` and depends on session/window-driven exploratory IR workflows rather than a supported reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/development/s_spectraNatural.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a scratch exploratory notebook-style script with no stable API contract, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/development/s_stereoFundamentals.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an unfinished figure-driven class tutorial marked as still not working, so it is treated as exploratory teaching code rather than actionable headless migration debt.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneOverF1D.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a research-style exploratory synthesis notebook for 1/f intuition with interactive figures rather than a stable reusable MATLAB API surface.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneoverf2D_tinker.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicit tinker script for FFT/cropping experiments and remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneoverf_tinker.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicit tinker script for dead-leaves-style spectrum experiments rather than a supported reusable API surface.",
        "module_hits": [],
    },
    "tutorials/color/t_colorChromaticity.m": {
        "status": "parity",
        "note": "The chromaticity tutorial workflow is already covered by the current headless regression surface through `display_create/get(...)`, `chromaticity_xy(...)`, `xyz_from_energy(...)`, and `scene_create/get(...)` on the default Macbeth scene.",
        "module_hits": ["pyisetcam.color", "pyisetcam.metrics", "pyisetcam.display", "pyisetcam.scene"],
    },
    "tutorials/color/t_colorEnergyQuanta.m": {
        "status": "parity",
        "note": "The energy-versus-quanta tutorial workflow is already covered by the current headless regression surface through `quanta_to_energy(...)`, `energy_to_quanta(...)`, `ie_read_spectra(...)`, `xyz_from_energy(...)`, and `ie_xyz_from_photons(...)`.",
        "module_hits": ["pyisetcam.color", "pyisetcam.metrics"],
    },
    "tutorials/code/t_codeObjects.m": {
        "status": "parity",
        "note": "The deprecated object-database tutorial is covered by the optional `pyisetcam.session` compatibility layer and direct regressions for `ieInitSession(...)`, `ieAddObject(...)`, `ieGetObject(...)`, and `ieReplaceObject(...)`; MATLAB windows remain intentionally headless.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeRendering.m": {
        "status": "parity",
        "note": "The spectral rendering tutorial is covered by the current headless regression surface through `blackbody(...)`, `xyz_from_energy(...)`, `XW2RGBFormat(...)`, `xyz2srgb(...)`, and `imageIncreaseImageRGBSize(...)`.",
        "module_hits": ["pyisetcam.color", "pyisetcam.utils", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeSESSION.m": {
        "status": "parity",
        "note": "The vcSESSION add/get tutorial is covered by the optional `pyisetcam.session` compatibility layer and direct regressions for `ieInitSession(...)`, `ieAddObject(...)`, `ieGetObject(...)`, and selected-object bookkeeping; window display remains intentionally headless.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeStartup.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a deprecated MATLAB path-management note about `startup.m`, not a reusable supported headless API surface.",
        "module_hits": [],
    },
    "tutorials/printing/t_printingHalftone.m": {
        "status": "parity",
        "note": "The printing tutorial is covered by the focused headless halftoning regression exercising `HalfToneImage(...)`, the Bayer-style threshold-cell workflow, and Floyd-Steinberg error diffusion.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "tutorials/camera/t_SystemSimulate.m": {
        "status": "parity",
        "note": "The end-to-end system tutorial is covered by the current headless `scene -> oi -> sensor -> ip` workflow regressions together with the camera-introduction regression and the existing scene/OI/sensor/IP compatibility surface.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor", "pyisetcam.ip", "pyisetcam.camera"],
    },
    "tutorials/camera/t_cameraAntiAliasing.m": {
        "status": "parity",
        "note": "The anti-alias tutorial is covered by the headless frequency-orientation camera regression plus the underlying `oi_diffuser(...)` and `oiBirefringentDiffuser(...)` compatibility coverage, including diffuser replay through `oi_compute(...)`.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor", "pyisetcam.ip"],
    },
    "tutorials/camera/t_cameraIntroduction.m": {
        "status": "parity",
        "note": "The camera-introduction tutorial is covered by the current `camera_create/get/set/compute(...)` regression surface, including direct `cameraGet(..., 'ip data srgb')` access and compute replay starting from `scene`, `oi`, and `sensor`.",
        "module_hits": ["pyisetcam.camera", "pyisetcam.ip"],
    },
    "tutorials/camera/t_cameraNoise.m": {
        "status": "parity",
        "note": "The camera-noise tutorial is covered by the seeded headless camera-noise regression, including `noise flag` control, luminance changes, and the default camera pipeline on an RGB scene file.",
        "module_hits": ["pyisetcam.camera", "pyisetcam.scene", "pyisetcam.display"],
    },
    "tutorials/image/t_ip.m": {
        "status": "parity",
        "note": "The introductory IP tutorial workflow is already covered by the current headless `scene -> oi -> sensor -> ip` regression path, including `MCC Optimized` sensor conversion, `gray world` illuminant correction, adaptive-Laplacian demosaic, and display-backed IP state.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__", "pyisetcam.display"],
    },
    "tutorials/image/t_ipJPEGMonochrome.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a standalone educational JPEG/DCT tutorial built around the legacy `scripts/image/jpegFiles` teaching helpers rather than the ISET object pipeline, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "tutorials/image/t_ipJPEGcolor.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a standalone educational JPEG color-compression tutorial built around the legacy `scripts/image/jpegFiles` teaching helpers rather than the ISET object pipeline, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/display/s_displayCompare.m": {
        "status": "parity",
        "note": "The display-comparison workflow is already covered by the current headless display/scene rendering surface and the existing display-driven `scene_from_file(...)` regression coverage across `OLED-Sony`, `LCD-Apple`, and `CRT-Dell`.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/display/s_displayReflectanceCtemp.m": {
        "status": "parity",
        "note": "The theoretical reflectance-display workflow is already covered by the current `display_reflectance(...)` / `displayReflectance(...)`, `scene_from_file(...)`, and `scene_adjust_illuminant(...)` compatibility surface with direct regression coverage.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/display/s_displaySurfaceReflectance.m": {
        "status": "parity",
        "note": "The MATLAB reflectance-display construction walkthrough is already exercised by the current headless `display_create/get/set(...)` and display-backed `scene_from_file(...)` workflow coverage, so the script is tracked as covered tutorial evidence rather than open API debt.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene"],
    },
    "scripts/utility/s_initSO.m": {
        "status": "out_of_scope",
        "note": "The upstream file only seeds `vcSESSION` with default scene/OI objects through `ieAddObject(...)`, so it remains a session-population script outside the explicit-object headless migration target.",
        "module_hits": [],
    },
    "tutorials/display/t_displayIntroduction.m": {
        "status": "parity",
        "note": "The introductory display tutorial is already covered by the current headless `display_create/get/set(...)` helper surface plus display-backed `scene_from_file(...)` and `scene_adjust_illuminant(...)` regression coverage.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene"],
    },
    "tutorials/display/t_displayRendering.m": {
        "status": "parity",
        "note": "The display-rendering tutorial is already covered by the current headless display rendering/accessor surface together with the multi-display `scene_from_file(...)` rendering workflow coverage used by the existing unit and parity-backed display regressions.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/optics/chromAb/ChromAb.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a shell-driven external-binary workflow around `otf` and `pr_mat`, so it is treated as a legacy launcher rather than a supported headless Python API target.",
        "module_hits": [],
    },
    "scripts/optics/chromAb/makeCAplots.m": {
        "status": "out_of_scope",
        "note": "The upstream file is effectively a plotting/comment stub for the chromatic-aberration workflow and remains outside the headless migration target.",
        "module_hits": [],
    },
    "scripts/sensor/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the sensor scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/sensor/pixel/s_pixelSizeDR.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching script that explores pixel dynamic-range trends rather than a standalone reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/sensor/readrawsensor/s_Raw2ISET.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive file-picker walkthrough layered on top of LoadRawSensorData rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "metrics/ssim/s_metricsSSIM.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching/regression script that drives an interactive camera-to-metrics walkthrough rather than a standalone reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtBlockPartition.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script-style diagnostic prototype with display side effects rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtAngleLUT.m": {
        "status": "ported",
        "note": "The MATLAB angle-lookup helper is covered by the Python `rt_angle_lut(...)` / `rtAngleLUT(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtBlockCenter.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-center helper is covered by the Python `rt_block_center(...)` / `rtBlockCenter(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtChooseBlockSize.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-sizing helper is covered by the Python `rt_choose_block_size(...)` / `rtChooseBlockSize(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtDIInterp.m": {
        "status": "ported",
        "note": "The MATLAB geometric-distortion interpolation helper is covered by the Python `rt_di_interp(...)` / `rtDIInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtExtractBlock.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-extraction helper is covered by the Python `rt_extract_block(...)` / `rtExtractBlock(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtFileNames.m": {
        "status": "ported",
        "note": "The MATLAB Zemax filename helper is covered by the Python `rt_file_names(...)` / `rtFileNames(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtGeometry.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace geometry stage is covered by the Python `rt_geometry(...)` / `rtGeometry(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtImagePSFFieldHeight.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked as not functional yet, so it remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtImageRotate.m": {
        "status": "ported",
        "note": "The MATLAB PSF image-rotation helper is covered by the Python `rt_image_rotate(...)` / `rtImageRotate(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtImportData.m": {
        "status": "ported",
        "note": "The MATLAB Zemax import helper is covered by the Python `rt_import_data(...)` / `rtImportData(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtInsertBlock.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-insertion helper is covered by the Python `rt_insert_block(...)` / `rtInsertBlock(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtOTF.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace OTF builder is covered by the Python `rt_otf(...)` / `rtOTF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFApply.m": {
        "status": "ported",
        "note": "The MATLAB shift-variant PSF application entry point is covered by the Python `rt_psf_apply(...)` / `rtPSFApply(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFEdit.m": {
        "status": "ported",
        "note": "The MATLAB PSF centering/rotation helper is covered by the Python `rt_psf_edit(...)` / `rtPSFEdit(...)` compatibility wrapper; visualization remains intentionally headless.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFGrid.m": {
        "status": "ported",
        "note": "The MATLAB PSF support-grid helper is covered by the Python `rt_psf_grid(...)` / `rtPSFGrid(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFInterp.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace PSF interpolation helper is covered by the Python `rt_psf_interp(...)` / `rtPSFInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFVisualize.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure/movie viewer for ray-trace PSFs and remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive plotting gateway with dialogs and figure side effects rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtPrecomputePSF.m": {
        "status": "ported",
        "note": "The MATLAB precomputed shift-variant PSF builder is covered by the Python `rt_precompute_psf(...)` / `rtPrecomputePSF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPrecomputePSFApply.m": {
        "status": "ported",
        "note": "The MATLAB precomputed shift-variant PSF application helper is covered by the Python `rt_precompute_psf_apply(...)` / `rtPrecomputePSFApply(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtRIInterp.m": {
        "status": "ported",
        "note": "The MATLAB relative-illumination interpolation helper is covered by the Python `rt_ri_interp(...)` / `rtRIInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtRootPath.m": {
        "status": "ported",
        "note": "The MATLAB vendored ray-trace root helper is covered by the Python `rt_root_path(...)` / `rtRootPath(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtSampleHeights.m": {
        "status": "ported",
        "note": "The MATLAB sample-height selection helper is covered by the Python `rt_sample_heights(...)` / `rtSampleHeights(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtSynthetic.m": {
        "status": "ported",
        "note": "The MATLAB synthetic ray-trace optics generator is covered by the Python `rt_synthetic(...)` / `rtSynthetic(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/zemax/zemax2json.m": {
        "status": "out_of_scope",
        "note": "The upstream file depends on a live Windows OpticStudio/.NET connection and external JSON-writing workflow, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/zemax/zemaxLoad.m": {
        "status": "ported",
        "note": "The MATLAB Zemax PSF text-loader is covered by the Python `zemax_load(...)` / `zemaxLoad(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/zemax/zemaxReadHeader.m": {
        "status": "ported",
        "note": "The MATLAB Zemax header parser is covered by the Python `zemax_read_header(...)` / `zemaxReadHeader(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
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
    "opticalimage/wavefront/psf/psfPlotrange.m": {
        "status": "out_of_scope",
        "note": "The upstream file only adjusts figure axis limits and title text for PSF plots, so it remains outside the headless migration target.",
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
    "sensor/human/sensorConePlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven visualization helper for cone mosaics and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/sensor2EXR.m": {
        "status": "out_of_scope",
        "note": "The upstream file depends on MATLAB OpenEXR I/O and a network-training export workflow rather than a supported headless Python API surface.",
        "module_hits": [],
    },
    "sensor/models/s_sensorIMX490Test.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven exploratory IMX490 demo script rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesign.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE figure entry point for interactive CFA editing and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignCallbacks.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE callback/controller layer for the CFA design UI and does not define a reusable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignUI.m": {
        "status": "out_of_scope",
        "note": "The upstream file constructs an interactive GUIDE CFA design window and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignUtilities.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicitly unimplemented example stub for CFA-design scripting rather than a usable supported headless API surface.",
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
    "scene/pattern/sceneFluorescenceChart.m": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot does not vendor the required `fluorescenceSignal` / `fluorescenceWeights` model helpers, so this chart constructor depends on external fluorescence-model code outside the supported ISETCam migration target.",
        "module_hits": [],
    },
    "scene/depth/sceneDepthOverlay.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors and redirects callers to `scenePlot`, so it is treated as obsolete rather than actionable headless API debt.",
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
