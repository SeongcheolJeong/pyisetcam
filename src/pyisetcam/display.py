"""Display creation and accessors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .types import Display
from .utils import interp_spectra, param_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _mat_display_to_display(display_struct: Any) -> Display:
    display = Display(name=str(getattr(display_struct, "name", "display")))
    display.fields["wave"] = np.asarray(getattr(display_struct, "wave"), dtype=float)
    display.fields["spd"] = np.asarray(getattr(display_struct, "spd"), dtype=float)
    display.fields["gamma"] = np.asarray(getattr(display_struct, "gamma"), dtype=float)
    display.fields["dpi"] = float(getattr(display_struct, "dpi", 96.0))
    display.fields["dist"] = float(getattr(display_struct, "dist", 0.5))
    display.fields["is_emissive"] = bool(getattr(display_struct, "isEmissive", True))
    display.fields["image"] = None
    return display


def _display_default() -> Display:
    display = Display(name="default")
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    spd = np.full((wave.size, 3), 1.0 / 700.0, dtype=float)
    gamma = np.repeat(np.linspace(0.0, 1.0, 256, dtype=float)[:, None], 3, axis=1)
    display.fields.update(
        {
            "wave": wave,
            "spd": spd,
            "gamma": gamma,
            "dpi": 96.0,
            "dist": 0.5,
            "is_emissive": True,
            "image": None,
        }
    )
    return display


def display_create(
    display_name: str = "LCD-Apple",
    *args: Any,
    asset_store: AssetStore | None = None,
    wave: np.ndarray | None = None,
) -> Display:
    """Create a supported display."""

    store = _store(asset_store)
    normalized = param_format(display_name)
    if normalized == "default":
        display = _display_default()
    elif normalized == "equalenergy":
        display = _display_default()
        display.name = "equalenergy"
        display.fields["spd"] = np.ones_like(display.fields["spd"], dtype=float) * 1e-3
    else:
        name = display_name
        if not name.lower().endswith(".mat"):
            name = f"{name}.mat"
        try:
            display = _mat_display_to_display(store.load_display_struct(name))
        except Exception as error:  # pragma: no cover - exercised via tests against actual assets
            raise UnsupportedOptionError("displayCreate", display_name) from error

    target_wave = wave
    if target_wave is None and len(args) == 1:
        target_wave = np.asarray(args[0], dtype=float)
    if target_wave is not None:
        target_wave = np.asarray(target_wave, dtype=float).reshape(-1)
        display.fields["spd"] = interp_spectra(
            np.asarray(display.fields["wave"], dtype=float),
            np.asarray(display.fields["spd"], dtype=float),
            target_wave,
        )
        display.fields["wave"] = target_wave
    display.fields.setdefault("image", None)
    return display


def display_get(display: Display, parameter: str) -> Any:
    key = param_format(parameter)
    if key == "type":
        return display.type
    if key == "name":
        return display.name
    if key == "wave":
        return np.asarray(display.fields["wave"], dtype=float)
    if key == "spd":
        return np.asarray(display.fields["spd"], dtype=float)
    if key == "gamma":
        return np.asarray(display.fields["gamma"], dtype=float)
    if key == "dpi":
        return float(display.fields["dpi"])
    if key == "dist":
        return float(display.fields["dist"])
    if key == "image":
        return display.fields.get("image")
    raise KeyError(f"Unsupported displayGet parameter: {parameter}")


def display_set(display: Display, parameter: str, value: Any) -> Display:
    key = param_format(parameter)
    if key == "name":
        display.name = str(value)
        return display
    if key == "wave":
        display.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return display
    if key == "spd":
        display.fields["spd"] = np.asarray(value, dtype=float)
        return display
    if key == "gamma":
        display.fields["gamma"] = np.asarray(value, dtype=float)
        return display
    if key == "image":
        display.fields["image"] = value
        return display
    raise KeyError(f"Unsupported displaySet parameter: {parameter}")

