from __future__ import annotations

import numpy as np

from pyisetcam import display_create, display_get, display_set


def test_display_create_lcd_example(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    assert display_get(display, "spd").shape[1] == 3
    assert display_get(display, "gamma").shape[1] == 3


def test_display_wave_resample(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    display = display_create("lcdExample.mat", wave, asset_store=asset_store)
    assert np.array_equal(display_get(display, "wave"), wave)
    assert display_get(display, "spd").shape == (wave.size, 3)


def test_display_get_reports_matlab_style_derived_values(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)

    inverse_gamma = display_get(display, "inverse gamma", 32)
    gamma = display_get(display, "gamma")
    expected_bits = int(round(np.log2(gamma.shape[0])))

    assert inverse_gamma.shape == (32, 3)
    assert display_get(display, "bits") == expected_bits
    assert display_get(display, "n levels") == 2**expected_bits
    assert np.array_equal(display_get(display, "levels")[:4], np.array([0, 1, 2, 3]))
    assert display_get(display, "n primaries") == 3
    assert display_get(display, "white spd").shape == (display_get(display, "n wave"),)
    assert display_get(display, "black spd").shape == (display_get(display, "n wave"),)
    assert display_get(display, "meters per dot") > 0.0
    assert display_get(display, "dots per deg") > 0.0


def test_display_set_resamples_ambient_and_tracks_dixel_metadata(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    ambient = np.linspace(0.0, 1.0, display_get(display, "n wave"))
    intensity = np.ones((2, 3, 3), dtype=float)
    control = np.zeros((2, 3, 3), dtype=float)

    display = display_set(display, "ambient spd", ambient)
    display = display_set(display, "size", [0.30, 0.20])
    display = display_set(display, "pixels per dixel", [1, 1])
    display = display_set(display, "dixel image", intensity)
    display = display_set(display, "dixel control map", control)
    display = display_set(display, "wave", np.arange(420.0, 681.0, 20.0))

    assert display_get(display, "ambient spd").shape == display_get(display, "wave").shape
    assert np.allclose(display_get(display, "size"), np.array([0.30, 0.20]))
    assert display_get(display, "pixels per dixel") == [1, 1]
    assert display_get(display, "dixel size") == (2, 3)
    assert np.array_equal(display_get(display, "dixel control map"), control)
