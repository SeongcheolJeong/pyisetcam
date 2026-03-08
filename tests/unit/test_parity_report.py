from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_parity_report_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "parity_report.py"
    spec = importlib.util.spec_from_file_location("parity_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_array_metrics_reports_border_and_max_index() -> None:
    parity_report = _load_parity_report_module()

    reference = np.ones((10, 10), dtype=float)
    actual = reference.copy()
    actual[0, 0] = 1.5
    actual[5, 5] = 1.1

    metrics = parity_report._array_metrics(reference, actual)

    assert metrics["max_index"] == [0, 0]
    assert np.isclose(metrics["max_index_ref"], 1.0)
    assert np.isclose(metrics["max_index_actual"], 1.5)
    assert np.isclose(metrics["edge_mean_rel"], 0.5 / 36.0)
    assert np.isclose(metrics["interior_mean_rel"], 0.0015625)


def test_array_metrics_reports_2x2_phase_breakdown() -> None:
    parity_report = _load_parity_report_module()

    reference = np.ones((4, 4), dtype=float)
    actual = reference.copy()
    actual[0::2, 1::2] = 1.4
    actual[1::2, 0::2] = 1.2

    metrics = parity_report._array_metrics(reference, actual)

    assert np.isclose(metrics["phase_2x2_mean_rel"]["r0c1"], 0.4)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r1c0"], 0.2)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r0c0"], 0.0)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r1c1"], 0.0)
