# pyisetcam

`pyisetcam` is a greenfield Python port of the core ISETCam numerical pipeline:

- `scene -> optical image -> sensor -> image processor -> camera`
- pinned to upstream ISETCam commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821`
- validated through a GNU Octave parity harness for curated cases

## Status

This repository implements milestone-one scope:

- explicit-object APIs only, no `vcSESSION`
- core scene/display/oi/sensor/ip/camera objects and functions
- upstream asset fetching into `.cache/upstream/isetcam/<sha>/`
- unit tests and a parity harness scaffold

Out of scope for this milestone:

- GUI/App Designer windows
- ray-trace optics
- OpenEXR/MEX integrations
- facetracker, printing, and most peripheral modules

## Quickstart

Install Miniforge and create the environment:

```bash
conda env create -f environment.yml
conda activate isetcam-py
python -m pip install -e .
python tools/fetch_upstream.py
pytest
```

Run the example pipeline:

```bash
python examples/end_to_end.py
python examples/scene_from_file.py
```

Generate the current machine-readable parity summary:

```bash
python tools/parity_report.py
```

Run the initial metrics helpers:

```python
from pyisetcam import comparison_metrics, metrics_spd

value, params = metrics_spd(spd1, spd2, metric="cielab", wave=wave, return_params=True)
summary = comparison_metrics(reference_image, rendered_image, data_range=1.0)
```

The report includes max-error locations plus edge/interior and 2x2 phase
diagnostics for array outputs, plus case-specific context and
reference-recompute checks for remaining sensor/camera gaps.
For the curated WVF case, the harness also records strict pre-PSF / PSF
stage parity and treats the final Octave delta-PSF convolution as a
documented relaxed `mean_rel` diagnostic, because that path collapses to a
single-precision FFT artifact in Octave.

Regenerate Octave baselines for the curated parity cases:

```bash
python tools/regenerate_parity_baselines.py
```

If the default `octave` wrapper in your environment is broken, point the
tool at a working `octave-cli` binary:

```bash
PYISETCAM_OCTAVE_BIN=/abs/path/to/octave-cli-10.3.0 python tools/regenerate_parity_baselines.py
```

For conda-style Octave installs, the runner now auto-populates
`OCTAVE_HOME`, `OCTAVE_EXEC_HOME`, and `OCTAVE_IMAGE_PATH` from the chosen
binary so baseline export works even when the raw CLI crashes at startup.

## Layout

- `src/pyisetcam/`: package source
- `tools/fetch_upstream.py`: fetch/extract the pinned upstream snapshot
- `tools/octave/`: Octave parity export helpers
- `tests/unit/`: deterministic unit tests
- `tests/parity/`: curated parity case definitions and baseline comparator
- `docs/migration.md`: MATLAB-to-Python name mapping

## Current Expansion

Post-milestone expansion now covers broader scene support and early sensor
preset growth. `scene_from_file` / `sceneFromFile` supports RGB and
monochrome inputs backed by emissive display calibration, the classic test
scene catalog has been expanded, and `sensor_create` now includes generic
`rgbw` / `rccc` presets plus upstream-backed `mt9v024` and `ar0132at`
RGBW/RCCC variants. The first metrics/validation slice is also in place:
`metrics_spd`, CIELAB helpers, and generic MAE/RMSE/PSNR utilities.
Initial ray-trace optics support is also in place: `oi_create('ray trace')`
loads the pinned upstream Zemax-derived optics asset, and `oi_compute`
applies upstream-backed geometric distortion, relative illumination, and a
radially interpolated PSF approximation. Multispectral / EXR /
reflective-display cases, full angle-dependent ray-trace PSF interpolation,
and the rest of the vendor sensor catalog remain explicitly out of scope
until they are ported deliberately.
