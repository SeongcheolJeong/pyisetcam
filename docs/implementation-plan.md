# ISETCam MATLAB-to-Python Migration Plan

## Current Status
- Milestone one is complete in the repository: the curated Octave parity suite is green and the handoff docs/examples are in place.
- Current execution focus has moved into post-milestone expansion on broader scene support, sensor presets, metrics, and ray-trace optics.
- `scene_from_file` / `sceneFromFile` is implemented for RGB and monochrome emissive-display inputs.
- Additional MATLAB-style scene patterns are now implemented: `uniform bb`, `uniform monochromatic`, `line ee`, `line ep`, `bar`, `point array`, `grid lines`, and `white noise`.
- Sensor preset expansion now includes generic `rgbw` / `rccc` creation and upstream-backed `mt9v024` / `ar0132at` RGBW and RCCC model variants.
- Metrics expansion now includes initial `metricsSPD`-style spectral comparison plus generic MAE/RMSE/PSNR helpers.
- Initial ray-trace optics support is now in place: `oi_create('ray trace')` loads upstream ray-trace optics assets, and `oi_compute` applies geometric distortion, relative illumination, angle-aware precomputed PSFs with cached reuse from the pinned MATLAB data, MATLAB-style uncropped padding/output bookkeeping, and the core ray-trace PSF metadata `oiGet/oiSet` surface, including MATLAB-style whole-optics export via `oiGet('optics')`, MATLAB-style raw-struct export for `raytrace` / `rtpsf` / `rtgeometry` / `rtrelillum`, MATLAB-style `psfStruct` / `sampledRTpsf` export for the precomputed cache, MATLAB-style `psf struct` normalization, split semantics between precomputed `rtpsfsize` and raw `optics rtpsfsize`, unit-aware raw PSF spacing / field-height getters, derived support / frequency axes, raw geometry / relative-illumination table accessors, raw scalar ray-trace metadata, the legacy `rtcomputespacing` field, MATLAB-style `oiGet/oiSet('optics ...')` delegation, whole-struct `optics` / `raytrace` replacement that normalizes MATLAB-style payloads while invalidating stale cached PSFs, indexed raw-table updates for `rtpsfdata` and `rtgeomfunction`, direct `rt_psf_interp` / `rtPSFInterp` support, direct `rt_di_interp` / `rtDIInterp` plus `rt_ri_interp` / `rtRIInterp` helper ports that now also back the ray-trace geometry compute path, and public helper ports for `rtSampleHeights`, `rtPSFGrid`, and `rtAngleLUT`.

## Summary
- Build a new Python library, `pyisetcam`, that ports the numerical `scene -> optical image -> sensor -> image processor -> camera` pipeline from ISETCam and validates it against GNU Octave runs of the upstream MATLAB code.
- Pin the migration baseline to upstream commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821` so the port is reproducible and not tied to a moving `main` branch.
- Treat milestone one as a hard-scope core port. The upstream repo is large, asset-heavy, and mixed with GUI/App Designer, global session state, MEX/OpenEXR, and facetracker code that should not block the first Python deliverable.
- Grounding sources: [ISETCam repo](https://github.com/ISET/isetcam), [pinned commit](https://github.com/ISET/isetcam/tree/412b9f9bdb3262f2552b96f0e769b5ad6cdff821), [README](https://raw.githubusercontent.com/ISET/isetcam/412b9f9bdb3262f2552b96f0e769b5ad6cdff821/README.md), [LICENSE](https://raw.githubusercontent.com/ISET/isetcam/412b9f9bdb3262f2552b96f0e769b5ad6cdff821/LICENSE), [Miniforge](https://github.com/conda-forge/miniforge), [conda-forge Octave](https://anaconda.org/conda-forge/octave).
- Local constraints to plan around: the workspace is empty and not a git repo, macOS Command Line Tools are missing, `/usr/bin/python3` and `/usr/bin/git` are installer shims, and `octave` is not installed.

## Scope
- In scope for milestone one: `sceneCreate`, `sceneAdjustIlluminant`, `sceneGet/Set`, `displayCreate`, `oiCreate`, `oiCompute`, `oiGet/Set`, `sensorCreate`, `sensorCompute`, `sensorGet/Set`, `ipCreate`, `ipCompute`, `ipGet/Set`, `cameraCreate`, `cameraCompute`, `cameraGet/Set`, plus shared color, spectral, units, and asset-loading helpers.
- Supported option set in milestone one: default Macbeth scenes plus `uniform d65`, `uniform ee`, `checkerboard`, `slanted bar`; diffraction-limited and wavefront-based shift-invariant optics; sensor types `default/bayer-grbg`, `bayer-rggb`, `monochrome`, and `ideal`; default IP pipeline with bilinear demosaic and display rendering; default, monochrome, and ideal camera constructors.
- Explicitly out of scope in milestone one: `vcSESSION` globals and implicit object lookup, GUIs/`.mlapp`/`.fig`, ray-trace optics, OpenEXR MEX, facetracker, metrics, printing, peripheral human/display tools, and most vendor-specific sensor presets.
- Any call outside that support matrix must raise `NotImplementedError` with the original MATLAB function name and requested option in the error text.

## Repository And Toolchain
- Initialize the empty workspace as a new git repo with a `src/` layout.
- Install Miniforge first, then create one environment named `isetcam-py` containing `python=3.12`, `octave`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `imageio`, `h5py`, `pooch`, `pytest`, `pytest-cov`, `ruff`, and `mypy`.
- Do not rely on system `python3`, system `git`, or a system Octave install; the current machine state is not usable for this project.
- Add `tools/fetch_upstream.py` that downloads the pinned ISETCam tarball into `.cache/upstream/isetcam/<sha>/` and verifies SHA256. Do not commit the full upstream MATLAB repo into the new Python repo.
- Add `tools/octave/run_case.m` and `tools/octave/export_case.m` to execute reference cases in Octave and save canonical outputs with `save('-mat7-binary', ...)`.

## Important Changes Or Additions To Public APIs / Interfaces / Types
- Package import path: `pyisetcam`.
- Canonical Python API uses snake_case: `scene_create`, `oi_compute`, `sensor_compute`, `ip_compute`, `camera_compute`. Re-export thin MATLAB-style aliases: `sceneCreate = scene_create`, `oiCompute = oi_compute`, and so on.
- Core types are flexible dataclasses, not raw dicts: `Scene`, `OpticalImage`, `Sensor`, `ImageProcessor`, `Display`, and `Camera`. Each has `name`, `type`, `metadata`, and a mutable `fields`/`data` store so MATLAB-style extensibility is preserved.
- Add shared `param_format()` that mirrors `ieParamFormat`: lowercase keys and strip spaces from keys only.
- All compute APIs require explicit object arguments. Any MATLAB behavior that would default to `vcSESSION` is unsupported in milestone one.
- Internal array convention is `(rows, cols, wave)` with wavelengths in nanometers and spatial units stored in meters. Use `float64` throughout milestone one to minimize parity drift.
- Asset resolution goes through an `AssetStore` abstraction that loads upstream `.mat`, `.json`, `.dat`, and image assets from `.cache/upstream/...` and never hardcodes user-specific paths.

## Implementation Phases
1. Bootstrap the repo, packaging, lint/test config, Miniforge environment instructions, and upstream asset fetcher. Exit gate: clean environment creation and successful download of the pinned upstream snapshot.
2. Port shared foundations: dataclasses, `param_format`, unit conversion helpers, spectral interpolation, `.mat`/JSON/image loaders, and error classes. Exit gate: deterministic unit tests pass without Octave.
3. Port scene, display, and color basics: `scene_create`, `scene_get/set`, `scene_adjust_illuminant`, luminance helpers, `display_create`, and the spectral/color transforms needed by the selected parity cases. Exit gate: parity on Macbeth/default, uniform scenes, and illuminant adjustment.
4. Port optics and optical image: `oi_create`, `oi_get/set`, diffraction-limited OTF/PSF path, shift-invariant wavefront path, padding, cropping, and resampling. Defer ray-trace optics. Exit gate: parity on default scene -> OI and one small wavefront case.
5. Port the sensor stack: essential pixel helpers, supported CFA patterns, `sensor_set_size_to_fov`, exposure handling, `sensor_compute`, noise flag `0` and `2`, analog gain/offset, clipping, and quantization. Exit gate: parity on noiseless volts and seeded statistical checks for noisy output.
6. Port image processing and camera orchestration: `ip_create`, `ip_compute`, bilinear demosaic, default adaptive transform path, display rendering, and `camera_create/compute/get/set`. Exit gate: end-to-end camera parity on curated cases.
7. Add docs and examples: one migration guide mapping MATLAB names to Python names, one end-to-end notebook or script, and a machine-readable parity report in `reports/parity/`. Exit gate: a new engineer can run the curated cases without opening MATLAB.

## Test Cases And Scenarios
- `scene_macbeth_default`: `sceneCreate()` equivalent, compare wave axis, photons shape, mean luminance, and selected pixel spectra.
- `scene_illuminant_change`: apply a 3000 K blackbody illuminant with `preserveMean=True` and `False`; compare luminance preservation and spectral ratios.
- `display_create_lcd_example`: load and resample a display calibration; compare `spd`, `gamma`, and wave grid.
- `oi_diffraction_limited_default`: default scene through `oiCreate()/oiCompute()`, compare irradiance cube, padding behavior, and FOV-derived geometry.
- `oi_wvf_small_scene`: small deterministic wavefront case, compare PSF/OTF-derived irradiance numerically with relaxed FFT tolerance.
- `sensor_bayer_noiseless`: default Bayer sensor with `noise flag = 0`, compare `volts` and exposure-derived scaling.
- `sensor_monochrome_noise_stats`: monochrome sensor with fixed seed and `noise flag = 2`, compare mean, std, and percentile bands, not bitwise samples.
- `ip_default_pipeline`: bilinear demosaic plus default rendering, compare `input`, `sensorspace`, and final RGB image with PSNR and MAE thresholds.
- `camera_default_pipeline`: `cameraCreate('default')` plus `cameraCompute(scene)`, compare final `ip.data.result` and intermediate stage metadata.
- Tolerances: deterministic scalar/vector math `rtol=1e-5`, `atol=1e-8`; FFT/convolution optics `rtol=1e-4`, `atol=1e-6`; final RGB `MAE <= 1e-3` and `PSNR >= 60 dB`; noise cases validated statistically within 1% on mean/std and 2% on percentile bands.

## Validation Harness
- Maintain parity case definitions in `tests/parity/cases.yaml`.
- Each case runs twice: Octave executes the pinned upstream MATLAB functions and writes canonical outputs to `tests/parity/baselines/<case>.mat`; Python computes the same case and compares normalized outputs.
- Normalize before compare: squeeze singleton dims, convert MATLAB column vectors to 1D arrays, harmonize dtypes, and label units explicitly in the exported baseline payload.
- Do not use `oct2py`; invoke Octave as a subprocess so the harness behaves the same locally and in future CI.
- Keep Octave as a validation-only dependency. The Python library itself must run without Octave once baselines exist.

## Assumptions And Defaults
- Upstream baseline remains fixed at `412b9f9bdb3262f2552b96f0e769b5ad6cdff821` until milestone one ships.
- Package name is `pyisetcam`; the new repo is greenfield and does not mirror the MATLAB tree one-for-one.
- The Python port preserves MATLAB semantics first, not a fully redesigned Pythonic OO model.
- `vcSESSION` compatibility is deferred; omitted-object calls are errors in milestone one.
- Octave is the reference executor even if some GUI or MEX-heavy MATLAB features are unsupported there; those features are already out of scope.
- Upstream assets stay external in `.cache/upstream`; do not bulk-copy them into the Python package during milestone one.
- Load standard `.mat` files with `scipy.io.loadmat` and use `h5py` only as a fallback for v7.3/HDF5 assets.
- If parity exposes a numerically unstable MATLAB path, reproduce the documented outputs of the curated cases first rather than widening scope to match every historical edge case immediately.

## Post-Milestone Expansion Order
- Add more scene patterns and `sceneFromFile`.
- Expand sensor presets to RGBW, RCCC, and selected vendor models actually used in scripts.
- Port metrics and validation utilities.
- Expand ray-trace optics support from the current upstream-backed geometry / angle-aware PSF slice to fuller MATLAB fidelity beyond the newly landed whole-optics/raw-struct/psfStruct export, padding/output bookkeeping, raw-table setter semantics, whole-struct optics replacement semantics, and the core PSF metadata `oiGet/oiSet` surface.
- Introduce an optional `SessionContext` compatibility layer and only then consider GUI replacement.
