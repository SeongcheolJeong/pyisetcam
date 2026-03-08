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
applies upstream-backed geometric distortion, relative illumination, and
angle-aware precomputed ray-trace PSFs with cached reuse on repeated
computes, plus MATLAB-style uncropped padding/output bookkeeping and
`oiGet/oiSet` support for ray-trace PSF metadata, including MATLAB-style
`psf struct` normalization, sample angles, image heights, wavelengths, and
optics-name accessors, plus unit-aware raw ray-trace PSF spacing / field
height getters and derived support / frequency axes. Raw ray-trace geometry
and relative-illumination tables are also exposed now, including scalar
ray-trace metadata, unit-aware field-height getters, and MATLAB-style raw
table/function accessors. Raw scalar ray-trace metadata now round-trips
through `oiGet/oiSet` as well, including object-distance aliases and the
legacy `rtcomputespacing` field. MATLAB-style `oiGet/oiSet('optics ...')`
delegation now routes through the same compatibility surface too, and
whole-struct `optics` / `raytrace` replacement now normalizes MATLAB-style
payloads while invalidating stale cached PSFs. Raw ray-trace table fidelity
now also includes MATLAB-style `rtpsfsize` reporting of the underlying 4-D
PSF table, indexed `oiSet` updates for `rtpsfdata` and `rtgeomfunction`,
MATLAB-style raw struct export for `raytrace`, `rtpsf`, `rtgeometry`, and
`rtrelillum`, and MATLAB-style whole-optics export via `oiGet('optics')`
that round-trips through `oiSet('optics', ...)`. The precomputed ray-trace
cache now also exports as MATLAB-style `psfStruct` / `sampledRTpsf`, and
plain `rtpsfsize` now tracks the precomputed PSF kernel size separately
from raw `optics rtpsfsize`. The upstream `rtPSFInterp` helper is also
ported now as `rt_psf_interp` / `rtPSFInterp` for direct interpolation of
raw ray-trace PSFs by field height, angle, and wavelength. The upstream
`rtDIInterp` / `rtRIInterp` helpers are also ported as `rt_di_interp` /
`rtDIInterp` and `rt_ri_interp` / `rtRIInterp`, and the ray-trace geometry
path now routes through those helper ports. The MATLAB helper surface now
also includes `rt_sample_heights` / `rtSampleHeights`, `rt_psf_grid` /
`rtPSFGrid`, and `rt_angle_lut` / `rtAngleLUT`. The staged ray-trace
pipeline is now partially exposed as well through `rt_geometry` /
`rtGeometry`, `rt_precompute_psf` / `rtPrecomputePSF`, and
`rt_precompute_psf_apply` / `rtPrecomputePSFApply`. The legacy
`rt_psf_apply` / `rtPSFApply` name is now exposed too and routes through
that validated cached-PSF path. The MATLAB-style top-level wrapper
`optics_ray_trace` / `opticsRayTrace` is now exposed as the staged
end-to-end ray-trace entry point as well, including cached OI illuminance
bookkeeping. The supporting OI helpers `oi_calculate_illuminance` /
`oiCalculateIlluminance` and `oi_diffuser` / `oiDiffuser` are now public
too, and `opticsRayTrace` routes its blur/illuminance path through them.
The `rtOTF` groundwork is also starting to land through public helper ports
for `rt_block_center` / `rtBlockCenter`, `rt_extract_block` /
`rtExtractBlock`, `rt_insert_block` / `rtInsertBlock`, and
`rt_choose_block_size` / `rtChooseBlockSize`. An initial public
`rt_otf` / `rtOTF` wrapper is now available as the block-wise ray-trace
OTF path, together with MATLAB-style `rtBlocksPerFieldHeight` control.
The underlying filtered-block support helper is now public too as
`rt_filtered_block_support` / `rtFilteredBlockSupport`.
`rt_synthetic` / `rtSynthetic` is now available as a synthetic ray-trace
optics generator for testing and controlled local experiments.
Multispectral / EXR / reflective-display cases and the rest of the vendor
sensor catalog remain explicitly out of scope until they are ported
deliberately.
