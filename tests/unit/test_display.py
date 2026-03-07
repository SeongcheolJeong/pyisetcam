from __future__ import annotations

import numpy as np

from pyisetcam import display_create, display_get


def test_display_create_lcd_example(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    assert display_get(display, "spd").shape[1] == 3
    assert display_get(display, "gamma").shape[1] == 3


def test_display_wave_resample(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    display = display_create("lcdExample.mat", wave, asset_store=asset_store)
    assert np.array_equal(display_get(display, "wave"), wave)
    assert display_get(display, "spd").shape == (wave.size, 3)

