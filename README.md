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
RGBW/RCCC variants. `camera_create(...)` now forwards those expanded
sensor presets too, so camera constructors can directly target `rgbw`,
`rccc`, `mt9v024`, and `ar0132at` sensor variants. The first
metrics/validation slice is also in place:
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
The shared MATLAB field-height lookup rule is now exposed too as
`ie_field_height_to_index` / `ieFieldHeight2Index`, and the ray-trace
block-size logic now uses that public helper directly.
The initial Zemax text-import surface is now ported too through
`rt_file_names` / `rtFileNames`, `zemax_read_header` / `zemaxReadHeader`,
`zemax_load` / `zemaxLoad`, and a limited `rt_import_data` /
`rtImportData` path for standard ISETPARAMS + DI/RI/PSF bundles. That
bundle format is now accepted directly by both `rt_import_data(...)` and
`oi_create('ray trace', ...)`, either as the `ISETPARAMS.txt` /
`ISETPARMS.TXT` file itself or as a containing directory, and the
importer now normalizes Windows-style `lensFile` / `baseLensFileName`
paths the way the upstream MATLAB path does. The parameter parser now
also accepts multiline MATLAB vector syntax,
including `...` continuations, bracketed row/column wavelength lists, and
transpose-style vectors such as `[500 600]'`, plus legacy single-line
assignments without trailing semicolons like the older Zemax macros emit
for `baseLensFileName`.
When importing into an existing optics object, `rt_import_data` now also
preserves the existing name/transmittance/compute settings while updating
the top-level ray-trace focal-length / f-number state to the imported
effective values, and it now enforces the upstream `psfSize` evenness
check from the parameter file before loading PSF data. Imported Zemax
bundles now also carry `psfSpacing` through to the ray-trace computation
metadata, so `rtcomputespacing` is populated on imported optics, and an
existing computation spacing is preserved when a bundle omits that field.
Imported ray-trace optics now also retain the requested `rtProgram`
label instead of forcing the lower-case default. Directory-based bundle
loading now also preserves real import failures for malformed Zemax
bundles instead of collapsing them into an unsupported-option error.
Raw MATLAB-style optics structs now also preserve top-level control
fields like `computeMethod`, `offaxis`, and `aberrationScale` through the
import path. When importing into an existing optics object, a distinct
existing ray-trace name is now preserved independently from the top-level
optics name, and a raw `rayTrace.name` now seeds the top-level optics
name when no explicit top-level name is present. Existing
`blocks_per_field_height` settings are now preserved across Zemax
imports too, and raw MATLAB-style ray-trace export now carries
`blocksPerFieldHeight` for round-tripping through `oi_get('optics')`.
Normalized ray-trace key detection now also recognizes
`blocks_per_field_height` directly, plus scalar normalized fields like
`f_number` and `magnification`, keeping raw/normalized struct handling
aligned for those cases too. Partial normalized nested `raytrace`
updates now also merge into the current ray-trace state instead of
dropping existing geometry / PSF tables, and the same preservation now
applies to partial MATLAB-style nested `rayTrace` updates. Raw nested
`rayTrace.fNumber` updates now also override the exported top-level
`fNumber` correctly instead of being clobbered during normalization, and
top-level `focalLength` edits now likewise override the stale exported
nested `rayTrace.effectiveFocalLength`. Whole-optics updates now also
honor normalized top-level aliases like `f_number`, `focal_length_m`,
and `nominal_focal_length_m` in that same export-and-edit workflow, and
raw exported `nominalFocalLength` and `offaxis` now round-trip too.
Whole-optics edits that change only `transmittance.wave` now preserve the
existing scale by interpolation instead of resetting it, and explicit
whole-optics `transmittance.scale` edits now reject wrong-length vectors
the same way the dedicated setter does.
An initial optional `SessionContext` compatibility layer is now in place
too, including session-aware core create/compute entry points and
MATLAB-style helpers for registering and retrieving the current
scene/OI/sensor/IP/camera objects without changing the explicit-object
default workflow. Session-aware `camera_set(...)` and `ip_set(...)`
replacement flows now keep that registry coherent when camera, sensor,
OI, IP, or display objects are swapped out after creation, and the
session layer now also exposes MATLAB-style generic replacement and
listing helpers such as `vcReplaceObject`, `vcReplaceAndSelectObject`,
`vcGetObjects`, `vcGetObjectNames`, and `vcCountObjects`, plus
MATLAB-style pair-return wrappers for `vcGetSelectedObject` /
`vcGetObject`, delete helpers that renumber session slots the MATLAB
way, bulk object-list replacement through `vcSetObjects`, slot-allocation
queries through `vcNewObjectValue`, add-and-select helpers through
`vcAddAndSelectObject`, name generation through `vcNewObjectName`,
batch deletion through `vcDeleteSomeObjects`, nonpositive selected-slot
clearing through `vcSetSelectedObject`, and type reporting through
`vcGetObjectType`. The session layer now also exposes the matching
`ieAddObject` / `ieDeleteObject` / `ieGetObject` /
`ieGetSelectedObject` / `ieReplaceObject` / `ieSelectObject`
wrappers, including nested `optics`, `pixel`, and `ipdisplay`
getter semantics, plus a minimal `ieInitSession` /
`ieSessionGet` / `ieSessionSet` port for session metadata,
preferences, GUI slots, selected-object queries, and
image-size/GPU session flags. Window/figure aliases and
`ieAppGet` are now supported too for the scene/OI/sensor/IP/display
app lookup path, together with the concrete GUI-side aliases that
upstream still reads such as `scene image handle`,
`sensorimagehandle`, `oiwindowHandles`, `vcimagehandles`,
`metricshandles`, `oicomputelist`, and `sensor gamma`. The
session GUI layer now also exposes `vcGetFigure`, `vcSelectFigure`,
and a minimal `ieMainClose` cleanup wrapper over the stored session
windows. Stored window geometry/state compatibility is now exposed too
through `ieWindowsGet` / `ieWindowsSet`, and the adjacent lightweight
GUI/session wrappers `vcEquivalentObjtype`, `vcSetFigureHandles`, and
`ieRefreshWindow` are now available on top of that session state.
Name-based lookup is now covered too through `ieEquivalentObjtype` and
`ieFindObjectByName`.
The shared MATLAB-style programming helper `ieParameterOtype` is now
ported too for object-type inference from parameter strings.
Headless `.mat` object persistence is now available through
`vcSaveObject`, `vcExportObject`, and `vcLoadObject` when explicit file
paths are provided.
`iePTable` is now ported in a headless form too, returning structured
parameter-table rows for the core object types without depending on a GUI
table widget.
`vcGetROIData` is now ported in a headless form too for scene/OI/sensor/IP
objects, including MATLAB-style rect-to-location conversion, clipped ROI
bounds, XW-style row output, and NaN-filled sensor mosaic planes.
The adjacent headless ROI conversion helpers are now public too:
`ieRect2Locs`, `vcRect2Locs`, `ieRoi2Locs`, `ieLocs2Rect`, and
`ieRect2Vertices`.
The ROI-driven getter surface is now partially ported too: sensor ROI
state plus `sensorGet(..., 'roi volts/electrons/dv')`,
`sensorGet(..., 'roi volts mean')`, `sensorGet(..., 'roi electrons mean')`,
and `ipGet(..., 'roidata'/'roixyz')`, backed by headless
`imageDataXYZ`.
Scene and OI ROI getters are now covered too through
`sceneGet(..., 'roi photons/energy/luminance/reflectance')`,
`sceneGet(..., 'roi mean photons/energy/reflectance')`, and
`oiGet(..., 'roi photons/energy')` plus their ROI-mean variants.
The adjacent sensor compatibility surface now covers direct
`sensorGet(..., 'electrons')` as well as MATLAB-style line-profile
queries through `sensorGet(..., 'hline volts/electrons/dv')` and
`sensorGet(..., 'vline volts/electrons/dv')`, returning per-filter `data`,
`pos`, and `pixPos` vectors.
Sensor ROI/plot compatibility now also includes
`sensorGet(..., 'chromaticity')` and
`sensorGet(..., 'roi chromaticity mean')`, backed by the existing
headless demosaic path.
Scene/OI/IP ROI plotting helpers now expose the matching chromaticity and
summary getters too: `sceneGet(..., 'chromaticity'/'roi chromaticity mean')`,
`sceneGet(..., 'roi mean luminance')`, `oiGet(..., 'chromaticity'/'roi chromaticity mean')`,
`oiGet(..., 'roi illuminance'/'roi mean illuminance')`, and
`ipGet(..., 'chromaticity'/'roi chromaticity mean')`.
The adjacent headless line-profile surface is now covered too:
`sceneGet(..., 'radiance hline/vline')`, `sceneGet(..., 'luminance hline/vline')`,
`sceneGet(..., 'spatial support linear')`, `oiGet(..., 'irradiance hline/vline')`,
`oiGet(..., 'irradiance energy hline/vline')`, and
`oiGet(..., 'illuminance hline/vline')`.
The remaining scene illuminant getter surface around that plotting path is
now exposed too: `sceneGet(..., 'illuminant comment')`,
`sceneGet(..., 'roi illuminant photons/energy')`,
`sceneGet(..., 'roi mean illuminant photons/energy')`, and
`sceneGet(..., 'illuminant hline/vline photons/energy')`.
Minimal headless `plotScene` / `oiPlot` wrappers now sit on top of those
getters too. The supported subset returns MATLAB-style `uData` payloads
and `None` instead of opening figures, which makes the plot APIs usable in
tests and scripts.
That same headless wrapper layer now also includes `plotSensor` for
volts/electrons/dv line plots and ROI histograms, and `ipPlot` for RGB
line data, chromaticity, ROI RGB summaries (`rgbhistogram` / `rgb3d`),
luminance summaries, and ROI `cielab` / `cieluv` payloads. The shared
metrics layer now exposes `xyz_to_luv` / `xyz2luv` alongside the existing
XYZ, LAB, and chromaticity helpers, and the sensor layer now exposes
`pixel_snr` / `pixelSNR` plus `sensor_snr` / `sensorSNR` with matching
headless `plotSensor('pixel snr'/'sensor snr')` payloads. The current
headless `plotSensor` subset also includes spectral wrappers for
`'color filters'` and `'sensor spectral qe'` on top of the stored sensor
spectral bundle, plus headless CFA views for `'cfa'` / `'cfa block'` and
`'cfa full'` with MATLAB-style unit-block/full-array tiling and default
display scaling. The adjacent headless sensor image-view slice is now in
place too through `plotSensor('cfa image')`, `plotSensor('true size')`,
and `plotSensor('channels')`, returning image and per-channel payloads
with MATLAB-style sensor-name metadata instead of opening GUI windows.
The explicit sensor render surface now also includes `sensorGet/Set(...,
'gamma'/'scale max')`, shared `sensorGet('rgb', ...)` rendering with
upstream-style `max output` / `max digital value` scaling semantics, and
headless true-size / CFA-image payloads report the gamma / scale-max
settings they use.
The adjacent raw CFA compatibility surface now also includes
`sensorGet/Set('cfa')`, `sensorGet('cfa pattern')`, `sensorGet('cfa name')`,
and `sensorGet('unit block config')` with MATLAB-style unit-block metadata.
The neighboring spectral metadata surface now also includes
`sensorGet/Set('spectrum')`, wavelength aliases, `sensorGet('bin width')`,
and MATLAB-style wave updates that keep stored filter spectra, pixel QE,
and IR-filter samples interpolated onto the new wavelength grid.
The adjacent raw chart/metadata block is now covered too through
`sensorGet/Set('chart parameters')`, `sensorGet/Set('chart corner points')`,
`sensorGet/Set('chart rectangles')`, `sensorGet/Set('current rect')`, and
the nearby `metadata sensor/scene/optics name` plus `metadata crop`
round-trip fields, along with the adjacent sensor movement surface through
`sensorGet/Set('sensor movement'/'eye movement')`,
`sensorGet/Set('movement positions'/'sensor positions')`, and
`sensorGet/Set('frames per position'/'exposure times per position')`. The
neighboring legacy human-cone storage surface now round-trips too through
`sensorGet/Set('human')`, `sensorGet/Set('cone type')`,
`sensorGet/Set('densities')`, `sensorGet/Set('xy')`, and the
`human cone seed` / `rseed` aliases. The
neighboring legacy MCC aliases now round-trip too through
`sensorGet/Set('mcc corner points')`, which reuses the chart corner-point
field, plus raw `sensorGet/Set('mcc rect handles')` storage. The
neighboring noise-summary aliases now also include
`sensorGet/Set('black level')`, mapped onto the same stored zero-level
state used by `zero level` / `zero`. The neighboring sensor data aliases
now also include `sensorGet/Set('digital value')` and
`sensorGet/Set('digital values')`, routed through the existing `dv`
storage path, and `sensorGet('dv or volts'/'digital or volts')` now uses
the same preferred-DV fallback behavior as upstream. The adjacent
headless sensor data-view surface now also includes
`sensorGet('volt images')`, returning per-filter plane stacks with
`NaN`-filled empty locations like MATLAB `plane2rgb(...)`. The same
adjacent voltage/response block now also includes the upstream short-form
aliases `sensorGet/Set('voltage')`, `sensorGet('electron')`, and
`sensorGet/Set('ag'/'ao')` for voltage, electron, analog-gain, and
analog-offset access. The neighboring sensor model surface now also includes raw
`sensorGet/Set('diffusion MTF')` storage for legacy compatibility. The
neighboring legacy scene/lens aliases now round-trip too through
`sensorGet/Set('scene_name')`, `sensorGet/Set('lens')`, and
`sensorGet('metadata lensname'/'metadata lens')`. The adjacent raw
microlens storage surface now round-trips as well through
`sensorGet/Set('microlens'/'ml')` and
`sensorGet/Set('microlens offset'/'mloffset')`. The remaining scalar hooks
in that upstream block now round-trip too through
`sensorGet/Set('consistency'/'sensor consistency')` and
`sensorGet/Set('sensor compute'/'sensor compute method')`. The adjacent
exposure-control surface now includes `sensorGet('n exposures')`,
`sensorGet/Set('exposure plane')`, `sensorGet/Set('cds')`, and MATLAB-style
`auto exposure` string handling for `'on'` / `'off'`. The neighboring
exposure-summary surface now also includes `sensorGet/Set('exposure method')`,
array-capable `integration time` / `exptimes` storage, `unique exptimes`,
and `central exposure`, with `sensorCompute` explicitly rejecting multi-
exposure arrays until bracketed/CFA capture compute is ported. The nearby
sampling and vignetting aliases now also round-trip through
`sensorGet/Set('pixel samples'/'n pixel samples for computing')` and
`sensorGet/Set('sensor vignetting'/'sensor bare etendue'/'no microlens etendue')`.
The adjacent noise-control storage surface now also includes
`sensorGet/Set('reuse noise')`, `sensorGet/Set('noise seed')`, and
`sensorGet/Set('response type')`, with stored noise seeds now feeding
`sensorCompute(...)` whenever no explicit seed argument is passed. The
neighboring column fixed-pattern-noise storage surface now also
round-trips through `sensorGet/Set('column fpn')`,
`sensorGet('column dsnu'/'column prnu')`, and
`sensorGet/Set('col offset fpn vector'/'col gain fpn vector')`. The same
adjacent response/noise summary block now also includes
`sensorGet('response dr')`, `sensorGet('dr'/'dynamic range'/'sensor dynamic range')`,
and the upstream `shot noise flag` getter alias.
The same headless plotting surface now
also covers `plotSensor('etendue')`, returning MATLAB-style support and
relative-illumination payloads without opening a mesh plot window. Sensor
spectral plotting now also includes `plotSensor('ir filter')`,
`plotSensor('pixel spectral qe')`, `plotSensor('pixel spectral sr')`, and
`plotSensor('sensor spectral sr')`, backed by explicit `ir filter` and
`pixel spectral qe` state on the sensor model plus combined sensor
responsivity derived from the stored QE/filter bundle. The same headless
sensor plotting layer now also includes deterministic noise-map payloads
for `plotSensor('shot noise')`, `plotSensor('dsnu')`, and
`plotSensor('prnu')`. A matching headless `plotSensorFFT(...)` wrapper is
now available too for monochrome horizontal/vertical line FFT payloads,
and `plotSensor('chromaticity')` now returns MATLAB-style `rg` and
spectrum-locus payloads for sensor ROI plots without opening a figure.
Line-profile plots also now support MATLAB-style adjacent-line aggregation
through `plotSensor(..., 'two lines', True)` on the supported sensor
horizontal/vertical line cases, and headless `plotSensor(...)` now also
accepts MATLAB-style `capture` selection for multi-capture sensor stacks.
The adjacent line/FFT metadata is tighter too: single-line sensor plots now
return plot labels and filter-color metadata, and `plotSensorFFT(...)` now
accepts the same capture selection path for multi-capture stacks.
The remaining histogram/chromaticity wrappers now return the matching
headless plot labels too, including histogram axis labels/filter colors and
the MATLAB-style `rg` chromaticity axis/title strings.
The spectral and etendue wrappers now return the same core plot metadata as
well, including spectral wavelength-axis labels plus MATLAB-style window
names and the etendue axis labels/name string.
The headless pixel/sensor SNR wrappers now expose the matching plot labels
too, including signal/SNR axis labels, title strings, and legend entries
for the returned component curves.
Multispectral / EXR / reflective-display cases and the rest of the vendor
sensor catalog remain explicitly out of scope until they are ported
deliberately.
