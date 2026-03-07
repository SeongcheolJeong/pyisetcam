"""Display creation and accessors."""

from __future__ import annotations

from typing import Any

import numpy as np

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .types import Display
from .utils import interp_spectra, invert_gamma_table, param_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _mat_display_to_display(display_struct: Any) -> Display:
    display = Display(name=str(getattr(display_struct, "name", "display")))
    dixel = None
    if hasattr(display_struct, "dixel"):
        dixel_struct = getattr(display_struct, "dixel")
        dixel = {}
        for matlab_key, field_key in {
            "intensitymap": "intensity_map",
            "controlmap": "control_map",
            "nPixels": "n_pixels",
            "renderFunc": "render_function",
        }.items():
            if hasattr(dixel_struct, matlab_key):
                dixel[field_key] = getattr(dixel_struct, matlab_key)
    display.fields["wave"] = np.asarray(getattr(display_struct, "wave"), dtype=float)
    display.fields["spd"] = np.asarray(getattr(display_struct, "spd"), dtype=float)
    display.fields["gamma"] = np.asarray(getattr(display_struct, "gamma"), dtype=float)
    display.fields["dpi"] = float(getattr(display_struct, "dpi", 96.0))
    display.fields["dist"] = float(getattr(display_struct, "dist", 0.5))
    display.fields["is_emissive"] = bool(getattr(display_struct, "isEmissive", True))
    display.fields["ambient_spd"] = np.zeros(display.fields["wave"].shape, dtype=float)
    display.fields["size_m"] = None
    display.fields["dixel"] = dixel
    display.fields["comment"] = None
    display.fields["refresh_rate_hz"] = None
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
            "ambient_spd": np.zeros(wave.shape, dtype=float),
            "size_m": None,
            "dixel": None,
            "comment": None,
            "refresh_rate_hz": None,
            "image": None,
        }
    )
    return display


def _display_ambient(display: Display) -> np.ndarray:
    ambient = display.fields.get("ambient_spd")
    wave = np.asarray(display.fields["wave"], dtype=float)
    if ambient is None:
        return np.zeros(wave.shape, dtype=float)
    return np.asarray(ambient, dtype=float).reshape(wave.shape)


def _display_size_m(display: Display) -> np.ndarray | None:
    size = display.fields.get("size_m")
    if size is not None:
        return np.asarray(size, dtype=float).reshape(2)
    image = display.fields.get("image")
    if image is None:
        return None
    image_array = np.asarray(image)
    if image_array.ndim < 2:
        return None
    meters_per_dot = float(display_get(display, "meters per dot"))
    rows, cols = image_array.shape[:2]
    return np.array([cols * meters_per_dot, rows * meters_per_dot], dtype=float)


def _inverse_gamma_table(gamma_table: np.ndarray, n_steps: int) -> np.ndarray:
    linear_axis = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    linear_rgb = np.repeat(linear_axis, gamma_table.shape[1], axis=1)
    return invert_gamma_table(linear_rgb, gamma_table)


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


def display_get(display: Display, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "type":
        return display.type
    if key == "name":
        return display.name
    if key in {"isemissive"}:
        return bool(display.fields.get("is_emissive", True))
    if key in {"gtable", "dv2intensity", "gamma", "gammatable", "gammatable"}:
        return np.asarray(display.fields["gamma"], dtype=float)
    if key in {"inversegamma", "inversegammatable"}:
        gamma = np.asarray(display_get(display, "gamma table"), dtype=float)
        n_steps = int(args[0]) if args else gamma.shape[0]
        return _inverse_gamma_table(gamma[:, : min(3, gamma.shape[1])], n_steps)
    if key in {"bits", "dacsize"}:
        gamma = np.asarray(display_get(display, "gamma table"), dtype=float)
        return int(round(np.log2(gamma.shape[0])))
    if key in {"nlevels"}:
        return int(2 ** int(display_get(display, "bits")))
    if key in {"levels"}:
        return np.arange(int(display_get(display, "nlevels")), dtype=int)
    if key in {"wave", "wavelength"}:
        return np.asarray(display.fields["wave"], dtype=float)
    if key == "binwidth":
        wave = np.asarray(display.fields["wave"], dtype=float)
        if wave.size < 2:
            return 1.0
        return float(wave[1] - wave[0])
    if key == "nwave":
        return int(np.asarray(display.fields["wave"], dtype=float).size)
    if key == "nprimaries":
        return int(np.asarray(display_get(display, "spd"), dtype=float).shape[1])
    if key in {"spd", "spdprimaries"}:
        spd = np.asarray(display.fields["spd"], dtype=float)
        if spd.shape[0] != int(display_get(display, "nwave")):
            spd = spd.T
        if args:
            wave = np.asarray(args[0], dtype=float).reshape(-1)
            spd = interp_spectra(np.asarray(display.fields["wave"], dtype=float), spd, wave)
        return spd
    if key == "rgbspd":
        spd = np.asarray(display_get(display, "spd", *(args[:1] if args else ())), dtype=float)
        return spd[:, : min(3, spd.shape[1])]
    if key == "whitespd":
        spd = np.asarray(display_get(display, "spd", *(args[:1] if args else ())), dtype=float)
        return np.sum(spd, axis=1)
    if key in {"blackspd", "ambientspd"}:
        ambient = _display_ambient(display)
        if args:
            wave = np.asarray(args[0], dtype=float).reshape(-1)
            ambient = interp_spectra(np.asarray(display.fields["wave"], dtype=float), ambient[:, None], wave).reshape(-1)
        return ambient
    if key in {"dpi", "ppi"}:
        return float(display.fields["dpi"])
    if key in {"metersperdot"}:
        return 0.0254 / max(float(display.fields["dpi"]), 1e-12)
    if key in {"dotspermeter"}:
        return 1.0 / max(float(display_get(display, "meters per dot")), 1e-12)
    if key in {"dotsperdeg"}:
        dist = float(display_get(display, "viewing distance"))
        meters_per_degree = 2.0 * dist * np.tan(np.deg2rad(0.5))
        return float(display_get(display, "dots per meter")) * meters_per_degree
    if key in {"dist", "viewingdistance"}:
        return float(display.fields["dist"])
    if key == "size":
        return _display_size_m(display)
    if key == "dixel":
        return display.fields.get("dixel")
    if key == "pixelsperdixel":
        dixel = display.fields.get("dixel") or {}
        return dixel.get("n_pixels")
    if key == "dixelsize":
        dixel = display.fields.get("dixel") or {}
        intensity = dixel.get("intensity_map")
        return None if intensity is None else intensity.shape[:2]
    if key in {"dixelintensitymap", "dixelimage"}:
        dixel = display.fields.get("dixel") or {}
        return dixel.get("intensity_map")
    if key == "dixelcontrolmap":
        dixel = display.fields.get("dixel") or {}
        return dixel.get("control_map")
    if key == "renderfunction":
        dixel = display.fields.get("dixel") or {}
        return dixel.get("render_function")
    if key == "comment":
        return display.fields.get("comment")
    if key == "refreshrate":
        return display.fields.get("refresh_rate_hz")
    if key == "image":
        return display.fields.get("image")
    raise KeyError(f"Unsupported displayGet parameter: {parameter}")


def display_set(display: Display, parameter: str, value: Any, *args: Any) -> Display:
    key = param_format(parameter)
    if key == "name":
        display.name = str(value)
        return display
    if key == "type":
        display.type = str(value)
        return display
    if key in {"gtable", "dv2intensity", "gamma", "gammatable"}:
        if isinstance(value, str) and param_format(value) == "linear":
            size = int(args[0]) if args else np.asarray(display.fields["gamma"], dtype=float).shape[0]
            axis = np.linspace(0.0, 1.0, size, dtype=float)[:, None]
            value = np.repeat(axis, int(display_get(display, "nprimaries")), axis=1)
        display.fields["gamma"] = np.asarray(value, dtype=float)
        return display
    if key in {"wave", "wavelength"}:
        new_wave = np.asarray(value, dtype=float).reshape(-1)
        if "wave" in display.fields and not np.array_equal(new_wave, np.asarray(display.fields["wave"], dtype=float)):
            display.fields["spd"] = interp_spectra(
                np.asarray(display.fields["wave"], dtype=float),
                np.asarray(display_get(display, "spd"), dtype=float),
                new_wave,
            )
            ambient = _display_ambient(display)
            display.fields["ambient_spd"] = interp_spectra(
                np.asarray(display.fields["wave"], dtype=float),
                ambient[:, None],
                new_wave,
            ).reshape(-1)
        display.fields["wave"] = new_wave
        return display
    if key in {"spd", "spdprimaries"}:
        spd = np.asarray(value, dtype=float)
        if spd.ndim != 2:
            raise ValueError("Display SPD must be a 2D matrix.")
        if spd.shape[0] != int(display_get(display, "nwave")) and spd.shape[1] == int(display_get(display, "nwave")):
            spd = spd.T
        display.fields["spd"] = spd
        return display
    if key in {"dpi", "ppi"}:
        display.fields["dpi"] = float(value)
        return display
    if key == "size":
        display.fields["size_m"] = np.asarray(value, dtype=float).reshape(2)
        return display
    if key in {"viewingdistance", "dist"}:
        display.fields["dist"] = float(value)
        return display
    if key == "refreshrate":
        display.fields["refresh_rate_hz"] = float(value)
        return display
    if key == "dixel":
        display.fields["dixel"] = dict(value)
        return display
    if key in {"dixelintensitymap", "dixelimage"}:
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["intensity_map"] = np.asarray(value, dtype=float)
        return display
    if key == "dixelcontrolmap":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["control_map"] = np.asarray(value, dtype=float)
        return display
    if key == "pixelsperdixel":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["n_pixels"] = value
        return display
    if key == "renderfunction":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["render_function"] = value
        return display
    if key == "comment":
        display.fields["comment"] = str(value)
        return display
    if key in {"rgb", "image"}:
        display.fields["image"] = value
        return display
    if key == "ambientspd":
        display.fields["ambient_spd"] = np.asarray(value, dtype=float).reshape(-1)
        return display
    raise KeyError(f"Unsupported displaySet parameter: {parameter}")
