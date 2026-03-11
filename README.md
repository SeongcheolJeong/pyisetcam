# pyisetcam

`pyisetcam` is a greenfield Python port of the core ISETCam numerical pipeline:

- `scene -> optical image -> sensor -> image processor -> camera`
- pinned to upstream ISETCam commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821`
- validated through a GNU Octave parity harness for curated cases
- with the broad-parity expansion now started by adding post-core scene-family cases, utility-helper cases for `unitFrequencyList`, `Energy2Quanta/Quanta2Energy` in vector and matrix form, `blackbody` energy/quanta, and `ieParamFormat`, plus initial metrics-family cases for `ieXYZFromEnergy`, `xyz2luv`, `ieXYZ2LAB`, `xyz2uv`, `cct`, `deltaEab` (1976), and `metricsSPD` angle/CIELAB/mired

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
scene catalog has been expanded with script-driven patterns such as
`harmonic`, `frequency orientation` / `freq orient`, and
`sweep frequency`, `star pattern`, plus script-driven `reflectance chart` support with
explicit sample-list recreation, and `sensor_create` now includes generic
`rgbw` / `rccc` presets plus upstream-backed `mt9v024` and `ar0132at`
RGBW/RCCC variants. `camera_create(...)` now forwards those expanded
sensor presets too, so camera constructors can directly target `rgbw`,
`rccc`, `mt9v024`, and `ar0132at` sensor variants. The first
metrics/validation slice is also in place:
`metrics_spd`, CIELAB helpers, and generic MAE/RMSE/PSNR utilities.
The optics expansion now also includes script-driven
`oi_create('psf')` / `oiCreate('psf')` support for shift-invariant PSF optics,
including a MATLAB-style default synthetic PSF constructor and explicit
`psf` / `wave` / `umPerSamp` custom PSF data.
The same shift-invariant optics surface now also includes
`si_synthetic` / `siSynthetic` for the script-driven synthetic PSF
families from `s_opticsSIExamples.m`, including MATLAB-style Gaussian,
Lorentzian, pillbox, and custom-PSF workflows that return
shift-invariant optics ready for `oiSet(..., 'optics', optics)`.
The same shift-invariant optics surface now also includes
`optics_psf_to_otf` / `opticsPSF2OTF` and
`oi_set(..., 'optics otfstruct', ...)` for MATLAB-style custom OTF
injection from flare PSFs, with a curated Octave parity case on the
upstream flare image; that case compares `photons` with a scale-invariant
rule because the upstream Octave path preserves image shape but returns a
constant-factor magnitude offset.
The adjacent custom shift-invariant file workflow now also includes
`ie_save_si_data_file` / `ieSaveSIDataFile` plus direct
`siSynthetic('custom', file)` parity, with the case comparing the saved
input PSF row directly and treating the downstream OI `photons` field
scale-invariantly for the same stable custom-OTF magnitude offset.
The same raw OTF workflow now also includes direct
`oi_get(..., 'optics OTF')` / `oi_set(..., 'optics OTF', ...)` support
for MATLAB-style shift-invariant OTF access and replacement, with a
curated Octave parity case that replaces the stored OTF with an ideal
all-ones OTF and compares the resulting image using a normalized-MAE rule
that stays stable in dark regions.
The script-driven synthetic-PSF workflow is also parity-covered by a
Lorentzian `siSynthetic` case that compares the final irradiance cube
scale-invariantly, because the upstream Octave path preserves image shape
while retaining a stable magnitude offset.
The adjacent wavefront helper surface now also includes
`wvf_set` / `wvfSet`, `wvf_get` / `wvfGet`,
`wvf_compute` / `wvfCompute`,
`wvf_defocus_diopters_to_microns` /
`wvfDefocusDioptersToMicrons`, `wvf_to_oi` / `wvf2oi`, direct
`oi_compute(wvf, scene)` support, MATLAB-style `oi_get(..., 'wvf ...')`,
and rebuilding `oi_set(..., 'wvf ...')` / `oi_set(..., 'optics wvf', ...)`
for script-driven WVF defocus/Zernike workflows.
That same explicit wavefront surface now also includes
`wvf_pupil_function` / `wvfPupilFunction` and
`wvf_compute_psf` / `wvfComputePSF`, including the MATLAB-style
stored-aperture workflow and direct parity coverage on the standalone
pupil-function/PSF path.
That same explicit WVF helper surface now also includes
`psf_to_zcoeff_error` / `psf2zcoeff`, plus unit-aware/index-aware
`wvfGet(..., 'wave', unit, idx)`, unit-aware
`wvfGet(..., 'measured wavelength', unit)`, and MATLAB-style
`wvfGet(..., 'z pupil diameter')` support for the upstream
`s_opticsPSF2Zcoeffs.m` workflow.
The same WVF backend now also includes
`wvf_aperture` / `wvfAperture` plus `wvf_aperture_params` / `wvfApertureP`
for script-driven synthetic aperture generation, including deterministic
clean polygon support used by the upstream flare scripts and direct
Octave parity on that clean-aperture path.
That same explicit WVF compute surface now also supports the MATLAB
key/value workflow `wvfCompute(wvf, 'aperture', aperture)`, with direct
parity on the script-driven polygon-aperture compute path used by the
upstream flare scripts.
The same script-driven optics surface now also includes headless
`oiPlot(..., 'psf'/'psf550'/'psfxaxis'/'psfyaxis')` support, backed by
computed `oiGet(..., 'psf data'/'psf xaxis'/'psf yaxis')` access for
diffraction-limited, custom-PSF, and computed WVF shift-invariant optics.
The adjacent diffraction-limited optics plotting surface now also includes
headless `oiPlot(..., 'ls wavelength'/'lswavelength'/'otf wavelength'/'mtf wavelength')`
support for the script-driven line-spread and OTF-by-wavelength workflows
used in the upstream diffraction PSF tutorials.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf', [], 550)` parity coverage for the upstream
`s_opticsDLPsf.m` workflow.
The script-driven custom optics surface now also includes direct
`opticsPSF2OTF(...)` parity coverage for the upstream flare-image
workflow from `s_opticsPSF2OTF.m`.
The adjacent wavefront plotting surface now also includes headless
`wvfPlot(...)` support for script-driven PSF, 1D PSF, pupil amplitude,
pupil phase, wavefront-aberration, PSF-angle, and OTF views, backed by
wavelength-aware `wvfGet(...)` support for `psf`, `pupil function`,
`wavefront aberrations`, `psf spatial samples`, `psf angular samples`,
`pupil spatial samples`, `1d psf`, and OTF support.
That same WVF plotting surface now also has direct Octave parity on the
`wvfPlot(..., '2d otf', ...)` workflow from `s_wvfPlot.m`, including the
returned OTF support axis and center-row magnitude data.
The same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil amp', ...)`, including the returned
spatial support axis and pupil-amplitude center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil phase', ...)`, including the
returned spatial support axis and pupil-phase center row.
That same optics helper surface now also includes `airy_disk` /
`airyDisk` plus headless airy-disk overlay payloads for `wvfPlot(...)`
and `oiPlot(...)`, with direct Octave parity on the scalar/image helper
contract used by the upstream diffraction plotting workflows.
That same script-driven WVF surface now also includes the spatial getter
family used by `s_wvfSpatial.m`, including `calc nwave`, `psf sample spacing`,
`ref psf sample interval`, `pupil sample spacing`, `pupil positions`,
`pupil function amplitude`, and `pupil function phase`.
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
analog-offset access, and the same `volts` / `dv` / `electrons` getters
now accept a channel index to return MATLAB-style per-filter vectors. The
same adjacent sensor-response geometry surface now also includes
`sensorGet('pixel pd area')` and `sensorGet('electrons per area', unit, channel)`
with MATLAB-style area-unit scaling and optional channel extraction. The
same neighboring quantization block now also includes `sensorGet/Set('bits')`,
`sensorGet/Set('qMethod')`, `sensorGet/Set('lut')`, and
`sensorGet/Set('quantization structure')` while preserving the existing
string-returning `sensorGet('quantization')` path used locally. The same
adjacent raw color/filter block now also includes `sensorGet/Set('color')`
plus the MATLAB cell-style aliases `filter names cell array`,
`filter color names cell array`, and `filter names cell`. The same
sensor-facing pixel passthrough now also routes supported pixel electrical
and geometric names such as `fill factor`, `pd area`, `conversion gain`,
`well capacity`, `dark current`, and `read noise volts` through the
stored pixel model, including whole-`pixel` replacement via
`sensorSet('pixel', ...)`. The same passthrough now also covers pixel spatial
geometry such as `pixel width` / `pixel height`, `width gap` / `height gap`,
`pd width` / `pd height`, `pd size`, `pd dimension`, and `pixel area`.
The neighboring pixel metadata surface now also covers `pixel name`,
`pixel type`, `pixel layer thicknesses`, `pixel refractive indices`,
`pixel spectrum` / `pixel wavelength`, `pixel bin width` / `pixel nwave`,
and MATLAB-style `pixel dr`.
The same pixel passthrough now also covers `pd xpos`, `pd ypos`, and
`pd position`, and that metadata now feeds the internal photodetector array
placement instead of being stored as a dead field.
The neighboring pixel electrical alias surface now also matches MATLAB for
`volts per electron`, `saturation voltage` / `max voltage`,
`dark current per pixel`, `dark voltage per pixel`, `volts per second`,
and the `read noise` family, including the important setter rule that
`sensorSet(..., 'read noise', value)` is interpreted in electrons rather
than volts.
The adjacent MATLAB pair-setter surface now also matches upstream ordering
for `pixel width and height` and `pixel pd width and height`, so those
aliases accept `[width, height]` inputs without changing the existing local
`pixel size` / `pd size` semantics.
The neighboring long-form pixel alias surface now also covers
`photodetector x position` / `photodetector y position` and the remaining
QE aliases such as `pixel quantum efficiency`,
`photodetector quantum efficiency`, and
`photodetector spectral quantum efficiency`.
The same pixel geometry alias layer now also covers the MATLAB gap-spacing
names `width between pixels` and `height between pixels`.
The direct unprefixed sensor-routing path now also recognizes the remaining
unique pixel setters such as `width and height`, `pd width and height`,
`size same fill factor`, and `dark voltage per pixel per sec`, so those
MATLAB-style calls no longer require an explicit `pixel` prefix.
The same adjacent unit/electrical alias surface now also covers
`pixel width meters`, `pixel height meters`, `pixel depth meters`, and the
direct `conversion gain v per electron` form.
The neighboring normalized electrical shorthand surface now also supports
direct sensor-level `conversiongainvpelectron`, `vswing`, `darkvolt`, and
`darkvolts` access without requiring the explicit `pixel` prefix.
The adjacent QE alias surface now also supports direct sensor-level
`quantum efficiency` access without requiring the explicit `pixel` prefix.
The neighboring FPN setter surface now also supports the upstream
`offset noise value` and `gain noise value` aliases.
The same FPN block now also has parser-level support for the broader
MATLAB alias family around DSNU/PRNU and column FPN names such as
`sigma offset fpn`, `gain fpn`, `column fixed pattern noise`, and
`col gain`.
That same FPN parser surface now also recognizes the setter-only
`column fpn parameters` alias.
The neighboring sensor metadata block now also has parser-level support
for `scene_name`, `metadata scene name`, `lens`, `metadata lensname`,
`metadata optics name`, `metadata sensor name`, and `metadata crop`.
The adjacent microlens block now also has parser-level support for
`ml`, `mlens`, `ulens`, `microlens offset`, `mloffset`, and
`microlens offset microns`.
The adjacent consistency block now also has parser-level routing for
`consistency`, `sensor consistency`, and `sensor compute`.
The neighboring exposure block now also has parser-level routing for
`integration time`, `exptimes`, `unique exptimes`, `central exposure`,
`exposure method`, `n exposures`, `exposure plane`, `correlated double sampling`,
`auto exposure`, and `automatic exposure`.
That same exposure coverage now also exercises the exact no-space aliases
`integrationtime`, `uniqueexptimes`, `centralexposure`, `exposuremethod`,
`nexposures`, `exposureplane`, `correlateddoublesampling`, `autoexp`,
`autoexposure`, and `automaticexposure`.
The neighboring sampling/noise-control block now also has parser-level routing for
`pixel samples`, `n pixel samples for computing`, `spatial samples per pixel`,
and `response type`.
That same sampling/noise-control coverage now also exercises the exact no-space aliases
`pixelsamples`, `npixelsamplesforcomputing`, `spatialsamplesperpixel`,
`reusenoise`, `noiseseed`, and `responsetype`.
The adjacent movement block now also has parser-level routing for
`eye movement`, `frames per position`, `exposure times per position`,
and `etime per pos`.
The same movement alias surface now also has focused parser/runtime coverage for
`movement positions` plus the prefixed `sensor positions` and `sensor positions x/y` forms.
That same movement coverage now also exercises the exact no-space aliases
`eyemovement`, `movementpositions`, `sensorpositions`, `sensorpositionsx`,
`sensorpositionsy`, `framesperposition`, `exposuretimesperposition`, and `etimeperpos`.
The neighboring legacy human block now also has parser-level routing for
`human`, `cone type`, `densities`, `xy`, and `rseed`.
That same block now also routes the legacy location aliases `cone xy` and
`cone locs`.
That same legacy human surface now also has focused parser/runtime coverage for
the long-form aliases `human cone type` and `human cone locs`.
The same test coverage now also exercises `human cone densities` and
`human cone seed` directly on the runtime path.
That same runtime/parser coverage now also exercises the exact legacy alias
`humanrseed`, and the same runtime coverage now also exercises the exact
location aliases `conexy` and `conelocs`.
The same legacy human coverage now also exercises the exact no-space aliases
`humanconetype`, `humanconedensities`, and `humanconelocs`, with parser
coverage also including `humanconeseed`.
The adjacent chart block now also has parser-level routing for
`chart parameters`, `corner points`, `chart corners`, `chart rects`,
`chart rectangles`, `current rect`, and `chart current rect`.
That same chart surface now also has focused parser/runtime coverage for the
natural long-form setter alias `chart corner points`.
The same runtime coverage now also exercises the long-form setter path for
`chart rectangles` and `current rect`, plus the adjacent `mcc corner points`
setter alias.
That same chart coverage now also exercises the exact no-space aliases
`chartparameters`, `cornerpoints`, `chartcorners`, `chartrects`,
`currentrect`, and `mccrecthandles`.
The same runtime coverage now also exercises the exact long no-space chart
aliases `chartcornerpoints`, `chartrectangles`, and `chartcurrentrect`.
The neighboring metadata block now also has focused parser/runtime coverage for
`scene name`, `metadata scene name`, `metadata lens name`, `metadata lens`,
`metadata optics name`, `metadata sensor name`, and `metadata crop`.
That same runtime coverage now also exercises the exact aliases `scene_name`,
`scenename`, `metadatalensname`, `metadatalens`, `metadatasensorname`, and
`metadatacrop`.
The adjacent chart/metadata coverage now also exercises the exact no-space MCC
and metadata aliases `mcccornerpoints`, `mccrecthandles`,
`metadatascenename`, `metadataopticsname`, `metadatasensorname`, and
`metadatacrop`.
The neighboring pixel material alias surface now also supports direct
sensor-level `refractive index` and `n` access without requiring the
explicit `pixel` prefix.
The adjacent setter semantics now also match MATLAB for
`pixel size same fill factor` / `pixel size constant fill factor`, scaling the
photodetector geometry with the pixel size instead of treating that alias like
a plain pixel-size replacement.
The neighboring sensor model surface now also includes raw
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
The same vignetting block now also has parser-level support for
`vignetting flag`, `pixel vignetting`, and `vignetting name`.
That same vignetting coverage now also exercises the exact no-space aliases
`vignettingflag`, `pixelvignetting`, `sensorvignetting`,
`sensorbareetendue`, `nomicrolensetendue`, and `vignettingname`.
The adjacent sensor-data coverage now also exercises the exact no-space
aliases `digitalvalue`, `digitalvalues`, `dvorvolts`, `digitalorvolts`,
`voltimages`, `voltage`, `electron`, `ag`, and `ao`.
The neighboring quantization and color/filter coverage now also
exercises the exact no-space aliases `qmethod`,
`quantizationstructure`, `quantizationlut`, `filternamescellarray`,
`filtercolornamescellarray`, and `filternamescell`.
The adjacent CFA and spectral-metadata coverage now also exercises the
exact no-space aliases `cfapattern`, `cfaname`, `unitblockconfig`,
`sensorspectrum`, `wavelengthresolution`, `binwidth`, and
`numberofwavelengthsamples`.
The neighboring filter/response metadata coverage now also exercises the
exact no-space aliases `nfilters`, `filtercolorletters`,
`filtercolorletterscell`, `filterplotcolors`, `patterncolors`,
`spectralqe`, and `sensorspectralsr`.
The adjacent spectral-filter and pixel-response coverage now also
exercises the exact no-space aliases `filterspectra`, `colorfilters`,
`infraredfilter`, `irfilter`, `pixelspectralqe`, `pdspectralqe`,
`pixelqe`, `spectralsr`, `pdspectralsr`, `pixelspectralsr`, and `sr`.
The neighboring geometry/block metadata coverage now also exercises the
exact no-space aliases `unitblockrows`, `unitblockcols`, and `cfasize`,
while the runtime surface now also uses the exact sensor getter form
`spatialsupport`.
The adjacent control/summary coverage now also exercises the exact
no-space aliases `autoexposure`, `integrationtime`,
`integrationtimes`, `nsamplesperpixel`, `sensorconsistency`,
`sensorcompute`, `sensorcomputemethod`, `dynamicrange`,
`sensordynamicrange`, `maxdigitalvalue`, `analoggain`, and
`analogoffset`.
The neighboring pixel passthrough coverage now also exercises the exact
no-space aliases `fillfactor`, `pixelsize`, `pixelwidth`,
`pixelheight`, `pixelwidthmeters`, `pixelheightmeters`, `widthgap`,
`heightgap`, `widthbetweenpixels`, `heightbetweenpixels`,
`xyspacing`, `pdarea`, `voltsperelectron`, `wellcapacity`,
`maxvoltage`, `darkvoltageperpixel`, `darkelectrons`,
`darkcurrentdensity`, `readnoisevolts`, `readnoiseelectrons`,
`readstandarddeviationvolts`, `readnoisemillivolts`,
`pixelspectrum`, `pixelwavelength`, `pixelbinwidth`, `pixelnwave`,
`layerthicknesses`, `refractiveindices`, `stackheight`,
`pixeldepthmeters`, `pdposition`, `pdxpos`, `pdypos`,
`photodetectorwidth`, `photodetectorheight`,
`photodetectorsize`, `photodetectorxposition`,
`photodetectoryposition`, `pdwidthandheight`,
`pixeldynamicrange`, `darkvoltageperpixelpersec`,
`voltspersecond`, `readnoisestdvolts`, and
`readstandarddeviationelectrons`, `widthandheight`,
`widthheight`, `sizeconstantfillfactor`,
`sizekeepfillfactor`, `sizesamefillfactor`, and
`pddimension`. The neighboring direct sensor geometry/data runtime
coverage now also exercises `rows`, `cols`, `size`, `dimension`,
`arraywidth`, `arrayheight`, `wspatialresolution`,
`hspatialresolution`, `deltax`, and `deltay`; parser coverage for this
block remains on the unambiguous prefixed forms like `sensor rows`
because the bare names intentionally stay object-generic in
`ie_parameter_otype(...)`. The adjacent direct sensor data-access
coverage now also explicitly exercises `volts`, `voltage`, `dv`,
`digitalvalue`, `digitalvalues`, `electrons`, `electron`,
`dvorvolts`, `digitalorvolts`, `voltimages`, and channel-select access
through the exact no-space aliases. The adjacent ROI/response block now
also explicitly exercises `roivolts`, `roielectrons`, `roidv`,
`roivoltsmean`, `roielectronsmean`, `responseratio`, and
`volts2maxratio`. That same ROI/response block now also exercises the
exact ROI-data synonym family `roidata`, `roidatav`, `roidatavolts`,
`roidatae`, `roidataelectrons`, and `roidigitalcount`, plus the exact
line-profile aliases `hlinevolts`, `hlineelectrons`, `hlinedv`,
`vlinevolts`, `vlineelectrons`, and `vlinedv`. That same sensor ROI and
line-profile block now also exercises the exact getter form
`roichromaticitymean`, with prefixed parser coverage for
`sensor chromaticity` and `sensor roichromaticitymean`. The adjacent ROI
geometry block now also exercises the exact forms `roi`, `roilocs`, and
`roirect`. The neighboring microlens/vignetting/etendue block now also
exercises the exact forms `microlens`, `microlensoffset`,
`microlensoffsetmicrons`, `vignetting`, `ngridsamples`, and
`sensoretendue`. The adjacent quantization block now also exercises the
exact forms `quantization`, `quantizationmethod`, `nbits`,
`quantizatonlut`, `maxdigital`, and `maxoutput`. The neighboring
CFA/filter metadata block now also exercises the exact forms `pattern`,
`filternames`, and `diffusionmtf`. The adjacent FPN/noise-control block
now also exercises the exact forms `dsnusigma`, `offsetnoisevalue`,
`prnusigma`, `gainnoisevalue`, `fpnparameters`,
`columnfixedpatternnoise`, `coloffsetfpnvector`,
`colgainfpnvector`, and `noiseflag`, and the same neighboring FPN-image
surface now also supports exact `dsnuimage`, `offsetfpnimage`,
`prnuimage`, and `gainfpnimage` access, including clearing those stored
images with `None`. The neighboring exposure-summary
block now also exercises the exact forms `exptime`, `exposuretimes`,
`exposuretime`, `expduration`, `exposureduration`,
`exposuredurations`, `uniqueintegrationtimes`, `uniqueexptime`,
`uniqueexptimes`, `geometricmeanexposuretime`, `expmethod`, and `cds`.
The adjacent movement and legacy-human block now also exercises the
exact forms `sensormovement`, `framesperpositions`, `conetype`,
`conexy`, and `conelocs`.
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
That same noise-summary coverage now also exercises the exact no-space aliases
`columnfpnparameters`, `columnfpn`, `columnfpnoffset`, `columnfpngain`,
`coloffsetfpn`, `coloffset`, `colgainfpn`, `colgain`, `responsedr`,
`drdb20`, `shotnoiseflag`, `blacklevel`, and `zerolevel`.
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
