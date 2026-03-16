"""Headless Spatial-CIELAB helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.color import deltaE_ciede2000, deltaE_ciede94

from .display import display_create, display_get
from .exceptions import UnsupportedOptionError
from .metrics import xyz_to_lab
from .scene import scene_from_file, scene_get, scene_set
from .types import Display, Scene
from .utils import param_format


def _matlab_round_scalar(value: float) -> int:
    numeric = float(value)
    if numeric >= 0.0:
        return int(np.floor(numeric + 0.5))
    return int(np.ceil(numeric - 0.5))


def _image_linear_transform(image: Any, transform: Any) -> NDArray[np.float64]:
    array = np.asarray(image, dtype=float)
    if array.ndim != 3:
        raise ValueError("image_linear_transform expects a 3D image cube.")
    rows, cols, channels = array.shape
    xw = array.reshape(rows * cols, channels)
    return np.asarray(xw @ np.asarray(transform, dtype=float), dtype=float).reshape(rows, cols, -1)


def _hwhm_to_sd(hwhm: float, dimensions: int = 2) -> float:
    if dimensions == 1:
        return float(hwhm) / (2.0 * np.sqrt(np.log(2.0)))
    if dimensions == 2:
        return float(hwhm) / np.sqrt(2.0 * np.log(2.0))
    raise ValueError(f"Unsupported Gaussian dimensionality {dimensions}.")


def _gauss(hwhm: float, support: int) -> NDArray[np.float64]:
    x = np.arange(1, int(support) + 1, dtype=float) - float(_matlab_round_scalar(int(support) / 2.0))
    sigma = _hwhm_to_sd(hwhm, 1)
    values = np.exp(-np.square(x / (2.0 * sigma)))
    return np.asarray(values / max(float(np.sum(values, dtype=float)), 1.0e-12), dtype=float)


def _gauss2(half_width_y: float, support_y: int, half_width_x: float, support_x: int) -> NDArray[np.float64]:
    x = np.arange(1, int(support_x) + 1, dtype=float) - float(_matlab_round_scalar(int(support_x) / 2.0))
    y = np.arange(1, int(support_y) + 1, dtype=float) - float(_matlab_round_scalar(int(support_y) / 2.0))
    xx, yy = np.meshgrid(x, y)
    sigma_x = _hwhm_to_sd(half_width_x, 2)
    sigma_y = _hwhm_to_sd(half_width_y, 2)
    values = np.exp(-0.5 * (np.square(xx / sigma_x) + np.square(yy / sigma_y)))
    return np.asarray(values / max(float(np.sum(values, dtype=float)), 1.0e-12), dtype=float)


def _sum_gauss(params: NDArray[np.float64], dimension: int) -> NDArray[np.float64]:
    width = int(np.ceil(float(params[0])))
    n_gauss = int((params.size - 1) / 2)
    if dimension == 2:
        kernel = np.zeros((width, width), dtype=float)
    else:
        kernel = np.zeros(width, dtype=float)
    for index in range(n_gauss):
        half_width = float(params[(2 * index) + 1])
        weight = float(params[(2 * index) + 2])
        if dimension == 2:
            kernel += weight * _gauss2(half_width, width, half_width, width)
        else:
            kernel += weight * _gauss(half_width, width)
    return np.asarray(kernel / max(float(np.sum(kernel, dtype=float)), 1.0e-12), dtype=float)


def _color_transform_matrix(matrix_type: str, space_type: int = 10) -> NDArray[np.float64]:
    key = param_format(matrix_type)
    if key == "lms2opp":
        matrix = np.array(
            [
                [0.9900, -0.1060, -0.0940],
                [-0.6690, 0.7420, -0.0270],
                [-0.2120, -0.3540, 0.9110],
            ],
            dtype=float,
        )
        return matrix.T
    if key == "opp2lms":
        matrix = np.array(
            [
                [0.9900, -0.1060, -0.0940],
                [-0.6690, 0.7420, -0.0270],
                [-0.2120, -0.3540, 0.9110],
            ],
            dtype=float,
        )
        return np.linalg.inv(matrix).T
    if key == "hpe2xyz":
        matrix = np.linalg.inv(
            np.array(
                [
                    [0.4002, 0.7076, -0.0808],
                    [-0.2263, 1.1653, 0.0457],
                    [0.0, 0.0, 0.9182],
                ],
                dtype=float,
            )
        )
        return matrix.T
    if key == "xyz2hpe":
        matrix = np.array(
            [
                [0.4002, 0.7076, -0.0808],
                [-0.2263, 1.1653, 0.0457],
                [0.0, 0.0, 0.9182],
            ],
            dtype=float,
        )
        return matrix.T
    if key in {"xyz2opp", "opp2xyz"}:
        if int(space_type) == 2:
            matrix = np.array(
                [
                    [278.7336, 721.8031, -106.5520],
                    [-448.7736, 289.8056, 77.1569],
                    [85.9513, -589.9859, 501.1089],
                ],
                dtype=float,
            ) / 1000.0
        elif int(space_type) == 10:
            matrix = np.array(
                [
                    [288.5613, 659.7617, -130.5654],
                    [-464.8864, 326.2702, 62.4200],
                    [79.8787, -554.7976, 481.4746],
                ],
                dtype=float,
            ) / 1000.0
        else:
            raise ValueError(f"Unsupported XYZ opponent space type {space_type}.")
        if key == "opp2xyz":
            matrix = np.linalg.inv(matrix)
        return matrix.T
    raise UnsupportedOptionError("colorTransformMatrix", matrix_type)


def color_transform_matrix(matrix_type: str, space_type: int = 10) -> NDArray[np.float64]:
    return _color_transform_matrix(matrix_type, space_type)


def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if params is not None:
        for key, value in params.items():
            param_key = param_format(key)
            if param_key == "deltaeversion":
                normalized["deltaEversion"] = value
            elif param_key == "imageformat":
                normalized["imageFormat"] = value
            elif param_key == "sampperdeg":
                normalized["sampPerDeg"] = float(value)
            elif param_key == "filtersize":
                normalized["filterSize"] = float(value)
            elif param_key == "filters":
                normalized["filters"] = value
            elif param_key == "filterversion":
                normalized["filterversion"] = value
            elif param_key == "dimension":
                normalized["dimension"] = int(value)
            elif param_key == "support":
                normalized["support"] = value
            else:
                normalized[str(key)] = value
    normalized.setdefault("deltaEversion", "2000")
    normalized.setdefault("imageFormat", "xyz10")
    normalized.setdefault("sampPerDeg", 224.0)
    normalized.setdefault("filterSize", float(normalized["sampPerDeg"]))
    normalized.setdefault("filters", [])
    normalized.setdefault("filterversion", "distribution")
    normalized.setdefault("dimension", 2)
    return normalized


def sc_params(dpi: float = 120.0, dist: float = 0.5) -> dict[str, Any]:
    pixel_spacing = 0.0254 / max(float(dpi), 1.0e-12)
    deg_per_pixel = np.rad2deg(np.tan(pixel_spacing / max(float(dist), 1.0e-12)))
    n_pixel = max(_matlab_round_scalar(1.0 / max(float(deg_per_pixel), 1.0e-12)), 1)
    return {
        "deltaEversion": "2000",
        "imageFormat": "xyz10",
        "sampPerDeg": float(n_pixel),
        "filterSize": float(n_pixel),
        "filters": [],
    }


def sc_gaussian_parameters(
    samp_per_deg: float,
    params: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    normalized = _normalize_params(params)
    version = param_format(normalized.get("filterversion", "distribution"))
    if version in {"distribution", "johnson"}:
        x1 = np.array([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686], dtype=float)
        x2 = np.array([0.0685, 0.616725, 0.826, 0.383275], dtype=float)
        x3 = np.array([0.0920, 0.567885, 0.6451, 0.432115], dtype=float)
    elif version in {"original", "published1996"}:
        x1 = np.array([0.0283, 0.921, 0.133, 0.105, 4.336, -0.1080], dtype=float)
        x2 = np.array([0.0392, 0.531, 0.494, 0.33], dtype=float)
        x3 = np.array([0.0536, 0.488, 0.386, 0.371], dtype=float)
    elif version == "hires":
        x1 = np.array([0.0283 / 2.0, 0.921, 0.133 / 2.0, 0.105, 4.336 / 2.0, -0.1080], dtype=float)
        x2 = np.array([0.0392 / 2.0, 0.531, 0.494 / 2.0, 0.33], dtype=float)
        x3 = np.array([0.0536 / 2.0, 0.488, 0.386 / 2.0, 0.371], dtype=float)
    else:
        raise UnsupportedOptionError("scGaussianParameters", normalized.get("filterversion", "distribution"))
    x1[[0, 2, 4]] *= float(samp_per_deg)
    x2[[0, 2]] *= float(samp_per_deg)
    x3[[0, 2]] *= float(samp_per_deg)
    return x1, x2, x3


def sc_prepare_filters(
    params: Mapping[str, Any] | None = None,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64], dict[str, Any]]:
    normalized = _normalize_params(params)
    samp_per_deg = float(normalized["sampPerDeg"])
    dimension = int(normalized.get("dimension", 2))
    if dimension not in {1, 2}:
        raise UnsupportedOptionError("scPrepareFilters", f"dimension={dimension}")

    filter_size = int(np.ceil(float(normalized.get("filterSize", samp_per_deg / 2.0))))
    if filter_size % 2 == 0:
        filter_size -= 1
    normalized["filterSize"] = float(filter_size)

    x1, x2, x3 = sc_gaussian_parameters(samp_per_deg, normalized)
    filters = [
        _sum_gauss(np.concatenate(([filter_size], x1)), dimension),
        _sum_gauss(np.concatenate(([filter_size], x2)), dimension),
        _sum_gauss(np.concatenate(([filter_size], x3)), dimension),
    ]
    support = np.arange(1, filter_size + 1, dtype=float)
    support = (support - np.mean(support, dtype=float)) / max(samp_per_deg, 1.0e-12)
    normalized["filters"] = filters
    normalized["support"] = support
    return filters, support, normalized


def _pad_image(image: NDArray[np.float64], row_pad: int, col_pad: int, method: Any) -> NDArray[np.float64]:
    if row_pad == 0 and col_pad == 0:
        return image
    pad_width = ((row_pad // 2, row_pad // 2), (col_pad // 2, col_pad // 2))
    if isinstance(method, str) and param_format(method) == "symmetric":
        return np.pad(image, pad_width, mode="symmetric")
    if method == 1:
        return np.pad(image, pad_width, mode="constant", constant_values=float(np.max(image)))
    return np.pad(image, pad_width, mode="constant", constant_values=float(method))


def sc_apply_filters(
    src_image: Any,
    filters: list[NDArray[np.float64]] | NDArray[np.float64],
    dimension: int = 2,
    img_pad_method: Any = "symmetric",
) -> NDArray[np.float64]:
    if int(dimension) == 1:
        raise UnsupportedOptionError("scApplyFilters", "dimension=1")

    image = np.asarray(src_image, dtype=float)
    if image.ndim != 3:
        raise ValueError("scApplyFilters expects a 3D opponent image.")

    rows, cols, channels = image.shape
    filter_list = [np.asarray(filters, dtype=float)] * channels if not isinstance(filters, list) else filters
    filter_shape = tuple(np.asarray(filter_list[0], dtype=float).shape)
    image_shape = (rows, cols)

    row_image_pad = 0
    col_image_pad = 0
    row_filter_pad = 0
    col_filter_pad = 0
    working_image = image

    if filter_shape != image_shape:
        if filter_shape[0] > image_shape[0] or filter_shape[1] > image_shape[1]:
            row_image_pad = max(0, filter_shape[0] - image_shape[0])
            col_image_pad = max(0, filter_shape[1] - image_shape[1])
            if row_image_pad % 2 == 1:
                working_image = working_image[:-1, :, :]
                row_image_pad += 1
            if col_image_pad % 2 == 1:
                working_image = working_image[:, :-1, :]
                col_image_pad += 1

        if filter_shape[0] < image_shape[0] or filter_shape[1] < image_shape[1]:
            row_filter_pad = max(0, image_shape[0] - filter_shape[0])
            col_filter_pad = max(0, image_shape[1] - filter_shape[1])
            if row_filter_pad % 2 == 1:
                working_image = working_image[:-1, :, :]
                row_filter_pad -= 1
            if col_filter_pad % 2 == 1:
                working_image = working_image[:, :-1, :]
                col_filter_pad -= 1

    dst_image = np.zeros_like(working_image, dtype=float)
    for plane in range(channels):
        this_image = np.asarray(working_image[:, :, plane], dtype=float)
        this_image = _pad_image(this_image, row_image_pad, col_image_pad, img_pad_method)

        this_filter = np.asarray(filter_list[plane], dtype=float)
        if row_filter_pad or col_filter_pad:
            this_filter = np.pad(
                this_filter,
                ((row_filter_pad // 2, row_filter_pad // 2), (col_filter_pad // 2, col_filter_pad // 2)),
                mode="constant",
            )

        filtered = np.fft.ifftshift(
            np.real(
                np.fft.ifft2(
                    np.fft.fft2(np.fft.fftshift(this_image)) * np.fft.fft2(np.fft.fftshift(this_filter))
                )
            )
        )
        dst_image[:, :, plane] = filtered[
            (row_image_pad // 2) : filtered.shape[0] - (row_image_pad // 2),
            (col_image_pad // 2) : filtered.shape[1] - (col_image_pad // 2),
        ]
    return dst_image


def sc_opponent_filter(
    image: Any,
    params: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normalized = _normalize_params(params)
    array = np.asarray(image, dtype=float)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("scOpponentFilter expects an image cube with three color channels.")
    dimension = 2 if (array.shape[0] > 1 and array.shape[1] > 3) else 1

    image_format = param_format(normalized.get("imageFormat", "xyz10"))
    xyz_space_type = 10 if image_format in {"xyz10", "lms10"} else 2
    if image_format.startswith("lms"):
        opponent = _image_linear_transform(array, _color_transform_matrix("lms2opp"))
    elif image_format.startswith("xyz"):
        opponent = _image_linear_transform(array, _color_transform_matrix("xyz2opp", xyz_space_type))
    else:
        raise UnsupportedOptionError("scOpponentFilter", normalized.get("imageFormat", "xyz10"))

    filtered_opponent = sc_apply_filters(opponent, list(normalized["filters"]), dimension)
    filtered_xyz = _image_linear_transform(filtered_opponent, _color_transform_matrix("opp2xyz", xyz_space_type))
    return filtered_xyz, filtered_opponent


def _white_points(white_point: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(white_point, (list, tuple)) and len(white_point) == 2:
        first = np.asarray(white_point[0], dtype=float).reshape(3)
        second = np.asarray(white_point[1], dtype=float).reshape(3)
        return first, second
    white = np.asarray(white_point, dtype=float).reshape(3)
    return white, white.copy()


def sc_compute_difference(
    xyz1: Any,
    xyz2: Any,
    white_point: Any,
    delta_e_version: str = "2000",
) -> NDArray[np.float64]:
    white1, white2 = _white_points(white_point)
    lab1 = xyz_to_lab(np.asarray(xyz1, dtype=float), white1)
    lab2 = xyz_to_lab(np.asarray(xyz2, dtype=float), white2)
    version = param_format(delta_e_version)
    if version in {"1976", "76", "cie1976"}:
        return np.asarray(np.linalg.norm(lab1 - lab2, axis=-1), dtype=float)
    if version in {"1994", "94", "cie1994"}:
        return np.asarray(deltaE_ciede94(lab1, lab2), dtype=float)
    if version in {"2000", "00", "cie2000", "ciede2000"}:
        return np.asarray(deltaE_ciede2000(lab1, lab2), dtype=float)
    raise UnsupportedOptionError("scComputeDifference", delta_e_version)


def _clip_xyz_image(xyz: Any, white_point: Any) -> NDArray[np.float64]:
    array = np.asarray(xyz, dtype=float)
    white = np.asarray(white_point, dtype=float).reshape(1, 1, 3)
    return np.clip(array, 0.0, white)


def sc_compute_scielab(
    xyz: Any,
    white_point: Any,
    params: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normalized = _normalize_params(params)
    filters = normalized.get("filters")
    if not filters:
        filters, support, normalized = sc_prepare_filters(normalized)
        normalized["filters"] = filters
        normalized["support"] = support
    white, _ = _white_points(white_point)
    clipped = _clip_xyz_image(xyz, white)
    filtered_xyz, _ = sc_opponent_filter(clipped, normalized)
    return np.asarray(xyz_to_lab(filtered_xyz, white), dtype=float), white


def scielab(
    image1: Any,
    image2: Any,
    white_point: Any,
    params: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    normalized = _normalize_params(params)
    first = np.asarray(image1, dtype=float)
    second = np.asarray(image2, dtype=float)
    if first.shape != second.shape:
        raise ValueError("image1 and image2 must have the same shape.")

    white1, white2 = _white_points(white_point)
    image_format = param_format(normalized.get("imageFormat", "xyz10"))
    if image_format.startswith("lms"):
        transform = _color_transform_matrix("hpe2xyz")
        first = _image_linear_transform(first, transform)
        second = _image_linear_transform(second, transform)
        white1 = np.asarray(white1.reshape(1, 3) @ transform, dtype=float).reshape(3)
        white2 = np.asarray(white2.reshape(1, 3) @ transform, dtype=float).reshape(3)
        normalized["imageFormat"] = "xyz10"

    if float(np.min(first)) < 0.0 or float(np.min(second)) < 0.0:
        raise ValueError("S-CIELAB requires nonnegative XYZ values.")

    filters = normalized.get("filters")
    if not filters:
        filters, support, normalized = sc_prepare_filters(normalized)
        normalized["filters"] = filters
        normalized["support"] = support

    xyz1, _ = sc_opponent_filter(first, normalized)
    xyz2, _ = sc_opponent_filter(second, normalized)
    delta_e_image = sc_compute_difference(xyz1, xyz2, (white1, white2), str(normalized["deltaEversion"]))
    return np.asarray(delta_e_image, dtype=float), normalized, xyz1, xyz2


def scielab_rgb(
    file1: Any,
    file2: Any,
    disp_cal: str | Display | None = None,
    v_dist: float = 0.38,
    *,
    asset_store: Any | None = None,
) -> tuple[NDArray[np.float64], Scene, Scene, Display]:
    display = display_create("crt.mat" if disp_cal is None else disp_cal, asset_store=asset_store)

    if isinstance(file1, Scene) and isinstance(file2, Scene):
        scene1 = file1
        scene2 = file2
    else:
        scene1 = scene_from_file(file1, "rgb", None, display, asset_store=asset_store)
        scene2 = scene_from_file(file2, "rgb", None, display, asset_store=asset_store)

    scene1 = scene_set(scene1, "distance", float(v_dist))
    scene2 = scene_set(scene2, "distance", float(v_dist))

    size = np.asarray(scene_get(scene1, "size"), dtype=int).reshape(-1)
    image_width = float(size[1]) * float(display_get(display, "meters per dot"))
    fov = np.rad2deg(2.0 * np.arctan2(image_width / 2.0, float(v_dist)))
    scene1 = scene_set(scene1, "fov", float(fov))
    scene2 = scene_set(scene2, "fov", float(fov))

    scene_xyz1 = np.asarray(scene_get(scene1, "xyz", asset_store=asset_store), dtype=float)
    scene_xyz2 = np.asarray(scene_get(scene2, "xyz", asset_store=asset_store), dtype=float)
    white_xyz = np.asarray(display_get(display, "white point"), dtype=float).reshape(3)
    samp_per_deg = 1.0 / max(float(scene_get(scene1, "degrees per sample")), 1.0e-12)
    params = {
        "deltaEversion": "2000",
        "sampPerDeg": float(samp_per_deg),
        "imageFormat": "xyz",
        "filterSize": float(samp_per_deg),
        "filters": [],
        "filterversion": "distribution",
    }
    error_image, _, _, _ = scielab(scene_xyz1, scene_xyz2, white_xyz, params)
    return error_image, scene1, scene2, display
