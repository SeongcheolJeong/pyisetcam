"""Headless font helpers and scene rendering."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .assets import AssetStore
from .display import display_create, display_get, display_set
from .exceptions import MissingAssetError
from .scene import scene_adjust_luminance, scene_create, scene_get, scene_set
from .session import track_session_object
from .types import SessionContext
from .utils import energy_to_quanta, param_format, rgb_to_xw_format, xw_to_rgb_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _font_name(letter: str, family: str, size: int, dpi: int) -> str:
    return str(f"{letter}-{family}-{int(size)}-{int(dpi)}").lower()


def _font_asset_path(font: dict[str, Any]) -> Path:
    return Path("data/fonts") / f"{font_get(font, 'name')}.mat"


def _font_bitmap_from_asset(font: dict[str, Any], store: AssetStore) -> np.ndarray:
    data = store.load_mat(_font_asset_path(font))
    bm_src = data["bmSrc"]
    bitmap_plane = 1.0 - np.asarray(bm_src.dataIndex, dtype=float)
    pad_cols = 3 * int(np.ceil(bitmap_plane.shape[1] / 3.0)) - int(bitmap_plane.shape[1])
    if pad_cols > 0:
        bitmap_plane = np.pad(bitmap_plane, ((0, 0), (pad_cols, 0)), constant_values=1.0)
    bitmap = np.ones((bitmap_plane.shape[0], int(np.ceil(bitmap_plane.shape[1] / 3.0)), 3), dtype=float)
    for channel in range(3):
        bitmap[:, :, channel] = bitmap_plane[:, channel::3]
    return bitmap


def _font_bitmap_from_pillow(font: dict[str, Any]) -> np.ndarray:
    character = str(font_get(font, "character"))
    family = str(font_get(font, "family"))
    size = int(font_get(font, "size"))
    dpi = int(font_get(font, "dpi"))
    scaled_size = max(int(round(size * max(float(dpi) / 72.0, 1.0))), 1)

    pil_font = None
    candidate_names = [
        family,
        f"{family}.ttf",
        f"{family}.otf",
        family.lower(),
        f"{family.lower()}.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidate_names:
        try:
            pil_font = ImageFont.truetype(candidate, scaled_size)
            break
        except OSError:
            continue
    if pil_font is None:
        pil_font = ImageFont.load_default()

    canvas_side = max(scaled_size * 6, 64)
    image = Image.new("L", (canvas_side, canvas_side), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), character, font=pil_font)
    offset_x = max((canvas_side - (bbox[2] - bbox[0])) // 2 - bbox[0], 0)
    offset_y = max((canvas_side - (bbox[3] - bbox[1])) // 2 - bbox[1], 0)
    draw.text((offset_x, offset_y), character, fill=0, font=pil_font)

    bitmap_gray = np.asarray(image, dtype=float)
    occupied = bitmap_gray < 255.0
    if np.any(occupied):
        rows = np.where(np.any(occupied, axis=1))[0]
        cols = np.where(np.any(occupied, axis=0))[0]
        bitmap_gray = bitmap_gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    bitmap = (bitmap_gray > 127.0).astype(float)
    return np.repeat(bitmap[:, :, np.newaxis], 3, axis=2)


def _oversample_pair(value: Any | None) -> tuple[int, int]:
    if value is None:
        return (20, 20)
    samples = np.asarray(value, dtype=int).reshape(-1)
    if samples.size == 1:
        samples = np.repeat(samples, 2)
    if samples.size < 2:
        raise ValueError("Oversample must provide one or two integer values.")
    return (max(int(samples[0]), 1), max(int(samples[1]), 1))


def _display_compute_simple(display: Any, image: np.ndarray, oversample: tuple[int, int]) -> np.ndarray:
    n_primaries = int(display_get(display, "n primaries"))
    if image.shape[2] < n_primaries:
        image = np.pad(image, ((0, 0), (0, 0), (0, n_primaries - image.shape[2])), constant_values=0.0)
    elif image.shape[2] > n_primaries:
        image = image[:, :, :n_primaries]
    image = np.repeat(image, oversample[0], axis=0)
    image = np.repeat(image, oversample[1], axis=1)
    return np.asarray(image, dtype=float)


def font_bitmap_get(
    font: dict[str, Any],
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    if font is None:
        raise ValueError("font required")

    store = _store(asset_store)
    try:
        return _font_bitmap_from_asset(font, store)
    except MissingAssetError:
        return _font_bitmap_from_pillow(font)


def font_create(
    letter: str = "g",
    family: str = "Georgia",
    sz: int = 14,
    dpi: int = 96,
    style: str = "NORMAL",
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    font = {
        "type": "font",
        "name": _font_name(str(letter), str(family), int(sz), int(dpi)),
        "character": str(letter),
        "size": int(sz),
        "family": str(family),
        "style": str(style),
        "dpi": int(dpi),
    }
    font["bitmap"] = font_bitmap_get(font, asset_store=asset_store)
    return font


def font_get(font: dict[str, Any], param: str, *args: Any) -> Any:
    if param is None:
        raise ValueError("Parameter must be defined.")

    key = param_format(param)
    if key == "type":
        return font["type"]
    if key == "name":
        return font["name"]
    if key == "character":
        return font["character"]
    if key == "size":
        return font["size"]
    if key == "family":
        return font["family"]
    if key == "style":
        return font["style"]
    if key == "dpi":
        return font["dpi"]
    if key == "bitmap":
        return np.asarray(font["bitmap"], dtype=float)
    if key == "ibitmap":
        return 1.0 - np.asarray(font_get(font, "bitmap"), dtype=float)
    if key == "paddedbitmap":
        pad_size = np.asarray(args[0] if args else [7, 7], dtype=int).reshape(-1)
        if pad_size.size == 1:
            pad_size = np.repeat(pad_size, 2)
        pad_val = float(args[1]) if len(args) > 1 else 1.0
        bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)
        return np.pad(
            bitmap,
            ((int(pad_size[0]), int(pad_size[0])), (int(pad_size[1]), int(pad_size[1])), (0, 0)),
            constant_values=pad_val,
        )
    if key == "ipaddedbitmap":
        return 1.0 - np.asarray(font_get(font, "padded bitmap", *(args[:2])), dtype=float)
    raise KeyError(f"Unknown parameter: {param}")


def font_set(
    font: dict[str, Any],
    param: str,
    val: Any,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    if param is None:
        raise ValueError("Parameter must be defined.")

    updated = copy.deepcopy(font)
    key = param_format(param)
    if key == "type":
        return updated
    if key == "name":
        updated["name"] = str(val)
        return updated
    if key == "character":
        updated["character"] = str(val)
    elif key == "size":
        updated["size"] = int(val)
    elif key == "family":
        updated["family"] = str(val)
    elif key == "style":
        updated["style"] = str(val)
    elif key == "dpi":
        updated["dpi"] = int(val)
    elif key == "bitmap":
        updated["bitmap"] = np.asarray(val, dtype=float)
        return updated
    else:
        raise KeyError(f"Unknown parameter: {param}")

    rebuilt = font_create(
        updated["character"],
        updated["family"],
        int(updated["size"]),
        int(updated["dpi"]),
        updated.get("style", "NORMAL"),
        asset_store=asset_store,
    )
    return rebuilt


def scene_from_font(
    font: dict[str, Any] | None = None,
    dsp: Any | None = None,
    scene: Any | None = None,
    o_sample: Any | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Any:
    store = _store(asset_store)
    current_font = font_create(asset_store=store) if font is None else copy.deepcopy(font)
    display = display_create("LCD-Apple", asset_store=store) if dsp is None else (
        display_create(dsp, asset_store=store) if isinstance(dsp, str) else dsp
    )

    if scene is None:
        current_scene = scene_create("default", 16, np.asarray(display_get(display, "wave"), dtype=float), asset_store=store)
    else:
        current_scene = scene.clone() if hasattr(scene, "clone") else copy.deepcopy(scene)

    oversample = _oversample_pair(o_sample)
    pad_size = args[0] if len(args) > 0 else None
    pad_val = args[1] if len(args) > 1 else None

    if float(display_get(display, "dpi")) != float(font_get(current_font, "dpi")):
        warnings.warn("Adjusting display dpi to match font", stacklevel=2)
        display = display_set(display, "dpi", float(font_get(current_font, "dpi")))

    padded_bitmap = np.asarray(font_get(current_font, "padded bitmap", pad_size, pad_val), dtype=float)
    display_image = _display_compute_simple(display, padded_bitmap, oversample)
    display_xw, rows, cols, _ = rgb_to_xw_format(display_image)
    spd = np.asarray(display_get(display, "spd"), dtype=float)
    energy = xw_to_rgb_format(display_xw @ spd.T, rows, cols)
    wave = np.asarray(display_get(display, "wave"), dtype=float)
    photons = energy_to_quanta(energy, wave)

    current_scene = scene_set(current_scene, "photons", photons)

    white_point = np.asarray(display_get(display, "white point"), dtype=float).reshape(-1)
    channel_index = min(1, padded_bitmap.shape[2] - 1)
    support_fraction = float(np.mean(padded_bitmap[:, :, channel_index]))
    current_scene = scene_adjust_luminance(current_scene, float(white_point[1]) * support_fraction, asset_store=store)

    distance_m = 0.5
    current_scene = scene_set(current_scene, "distance", distance_m)

    dot_size_m = 0.0254 / max(float(font_get(current_font, "dpi")), 1.0e-12)
    width_m = dot_size_m * float(padded_bitmap.shape[1])
    current_scene = scene_set(current_scene, "fov", float(np.degrees(np.arctan2(width_m, distance_m))))
    current_scene = scene_set(current_scene, "name", str(font_get(current_font, "name")))
    return track_session_object(session, current_scene)


fontBitmapGet = font_bitmap_get
fontCreate = font_create
fontGet = font_get
fontSet = font_set
sceneFromFont = scene_from_font
