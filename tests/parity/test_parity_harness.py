from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from pyisetcam import run_python_case

if os.environ.get("PYISETCAM_RUN_PARITY") != "1":
    pytestmark = pytest.mark.skip(reason="Set PYISETCAM_RUN_PARITY=1 to execute Octave parity checks.")


CASES_PATH = Path(__file__).with_name("cases.yaml")
BASELINES_DIR = Path(__file__).with_name("baselines")


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            value = np.real(value)
        return np.squeeze(np.asarray(value))
    return value


def _compare(reference: Any, actual: Any, *, rtol: float, atol: float) -> None:
    reference = _normalize(reference)
    actual = _normalize(actual)
    if isinstance(reference, dict):
        for key in reference:
            if key.startswith("__"):
                continue
            if key not in actual:
                raise AssertionError(f"Missing key '{key}' in actual output.")
            _compare(reference[key], actual[key], rtol=rtol, atol=atol)
        return
    if isinstance(reference, (str, bytes)) or isinstance(actual, (str, bytes)):
        assert actual == reference
        return
    if np.isscalar(reference) and np.isscalar(actual):
        assert np.isclose(float(actual), float(reference), rtol=rtol, atol=atol)
        return
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    assert reference_array.shape == actual_array.shape
    assert np.allclose(actual_array, reference_array, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", _case_definitions(), ids=lambda case: case["name"])
def test_parity_case(case, asset_store) -> None:
    baseline_path = BASELINES_DIR / f"{case['name']}.mat"
    if not baseline_path.exists():
        pytest.skip(f"Missing baseline: {baseline_path}")
    reference = {key: value for key, value in loadmat(baseline_path, squeeze_me=True, struct_as_record=False).items() if not key.startswith("__")}
    actual = run_python_case(case["name"], asset_store=asset_store)
    _compare(reference, actual, rtol=float(case["rtol"]), atol=float(case["atol"]))
