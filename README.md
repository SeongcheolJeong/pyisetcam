# pyisetcam

`pyisetcam` is a greenfield Python port of the core ISETCam numerical pipeline:

- `scene -> optical image -> sensor -> image processor -> camera`
- pinned to upstream ISETCam commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821`
- validated through a GNU Octave parity harness for curated cases
- with the broad-parity expansion now started by adding post-core scene-family cases, utility-helper cases for `unitFrequencyList`, `Energy2Quanta/Quanta2Energy` in vector and matrix form, `blackbody` energy/quanta, and `ieParamFormat`, plus initial metrics-family cases for `ieXYZFromEnergy`, `xyz2luv`, `ieXYZ2LAB`, `xyz2uv`, `cct`, `deltaEab` (1976), and `metricsSPD` angle/CIELAB/mired
- with Phase 3 scene-script coverage now started by adding MATLAB-style `spd_to_cct` / `spd2cct` plus curated Octave parity on the `s_sceneCCT.m` blackbody-CCT workflow
- and now also MATLAB-style `daylight(...)` plus curated Octave parity on the `s_sceneDaylight.m` daylight-SPD and daylight-basis workflow
- and now also headless `illuminantCreate/Get/Set` plus curated Octave parity on the `s_sceneIlluminant.m` illuminant-structure workflow
- and now also headless `sceneIlluminantSS(...)`, spatial-spectral `sceneAdjustIlluminant(...)`, and `sceneCreate('macbeth tungsten')` coverage plus curated Octave parity on the `s_sceneIlluminantMixtures.m` mixed-illuminant workflow
- and now also curated Octave parity on the `s_sceneIlluminantSpace.m` spatial-spectral illuminant workflow, covering the row-temperature gradient and the column-harmonic illuminant modulation path
- and now also headless `scenePlot(..., 'illuminant image')`, matching MATLAB's spectral-to-uniform-image fallback plus spatial-spectral illuminant rendering through returned `srgb` payloads without opening a figure window
- and now also `sceneGet(..., 'xyz')` plus MATLAB-style `RGB2XWFormat` / `XW2RGBFormat` coverage with curated Octave parity on the `s_sceneXYZilluminantTransforms.m` illuminant-transfer workflow, locking down normalized XYZ balance plus the fitted transfer matrices and relative error trend
- and now also curated Octave parity on the `s_sceneFromRGB.m` display-calibrated RGB-image workflow, covering `displayCreate('LCD-Apple.mat')`, the display white point, `sceneFromFile(..., 'rgb', ..., 'LCD-Apple.mat')`, the 6500 K illuminant swap, and the beak ROI reflectance contract
- and now also curated Octave parity on the `s_sceneFromMultispectral.m` multispectral-file workflow, covering `sceneFromFile(..., 'multispectral', ..., 400:10:700)` on `StuffedAnimals_tungsten-hdrs`, the loaded scene geometry and mean luminance, and normalized mean/center scene spectra
- and now also headless `sceneGet(..., 'rgb')` plus curated Octave parity on the `s_sceneFromRGBvsMultispectral.m` roundtrip workflow, covering the `StuffedAnimals_tungsten-hdrs` multispectral scene rendered to RGB and reconstructed through `LCD-Apple.mat`, with preserved geometry/luminance, preserved 6500 K illuminant chromaticity, and strong per-channel RGB/XYZ roundtrip correlations
- and now also headless `ie_reflectance_samples(...)` / `ieReflectanceSamples(...)` plus curated Octave parity on the `s_sceneReflectanceSamples.m` workflow, covering no-replacement reflectance sampling, exact sample-list replay, and the deterministic normalized-mean / singular-value statistics on explicit reflectance lists
- and now also headless `hc_basis(...)` / `hcBasis(...)` plus curated Octave parity on the `s_sceneReflectanceChartBasisFunctions.m` workflow, covering canonical reflectance-chart basis extraction at `bType=0.999`, `0.95`, and `5`, with parity on the selected basis counts, variance explained, sign-canonicalized basis functions, and coefficient summary statistics
- and now also headless `scene_reflectance_chart(...)` / `sceneReflectanceChart(...)` plus curated Octave parity on the `s_sceneReflectanceCharts.m` workflow, covering the default Natural-100 chart, the explicit Munsell/Food/Dupont/Hyspex no-replacement chart, the D65 illuminant swap, the gray-strip variant, and exact replay of the replica chart from returned stored samples
- and now also curated Octave parity on the `s_sceneChangeIlluminant.m` workflow, covering the default Macbeth-to-tungsten swap together with the StuffedAnimals multispectral scene converted to equal-energy and Horizon illuminants, with parity on preserved mean luminance, normalized illuminant spectra, and rendered-RGB mean summaries
- and now also curated Octave parity on the `s_sceneDataExtractionAndPlotting.m` workflow, covering the default `macbethd65` scene line/ROI extraction path, the returned luminance-hline support and profile, the attached illuminant-energy spectrum, the yellow-patch radiance and reflectance ROI summaries, and exact agreement between `scenePlot(...)` ROI spectra and manual `vcGetROIData(...)` means
- and now also curated Octave parity on the `s_sceneMonochrome.m` workflow, covering `displayCreate('crt')`, `sceneFromFile('cameraman.tif', 'monochrome', 100, 'crt')`, the source display-white illuminant and monochrome scene spectral summaries, and the D65-adjusted unispectral scene plus rendered-RGB mean contract
- and now also `scene_plot(..., 'illuminant energy roi')` support without an explicit ROI, matching upstream `scenePlot(..., [], ...)` fallback behavior, plus curated Octave parity on the `s_sceneSlantedBar.m` workflow, covering the equal-photons and D65 slanted-bar illuminant spectra together with the two slanted-bar constructions’ geometry, mean luminance, and normalized center-row/center-column luminance profiles
- and now also curated Octave parity on the `s_sceneHarmonics.m` workflow, covering the four exact harmonic constructions from the script: the basic 1 cpi pattern, the two-frequency sum with phase offset, the crossed-orientation 2-and-5 cpi pair, and the symmetric 5-and-5 cpi pair, with parity on scene geometry, preserved mean luminance, and normalized center-row/center-column luminance profiles
- and now also headless `sceneCreate('zone plate', ...)` coverage plus curated Octave parity on the `s_sceneZonePlate.m` workflow, covering the equal-photon zone-plate generator, its default `4 deg` field of view, and the clipped hyperspectral photon cube
- and now also headless `sceneCreate('dead leaves', ...)` coverage plus curated Octave parity on the dead-leaves scene family, covering the OLED-Sony display-rendered grayscale target, default `10 deg` field of view, and deterministic dead-leaves photon cube under a fixed sample stream
- and now also corrected `sceneCreate('bar', ...)` mean-luminance normalization plus curated Octave parity on the equal-photon bar scene family, covering the centered three-pixel bar geometry and `100 cd/m^2` luminance contract
- and now also corrected the shared `sceneCreate('line ...', ...)` mean-luminance normalization plus curated Octave parity on `sceneCreate('line ee', ...)`, `sceneCreate('line ep', ...)`, and `sceneCreate('lined65', ...)`, covering the equal-energy, equal-photon, and D65 vertical-line geometry with the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform monochromatic', ...)`, covering the single-wave `550 nm` narrow-band scene, its `12x12` geometry, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform d65', ...)`, covering the `24x24` D65 uniform scene, its flat spatial layout with non-flat D65 spectrum, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('empty')`, covering the cleared Macbeth-D65 geometry, the preserved illuminant spectrum, the all-zero photon cube, and the `0 cd/m^2` mean-luminance contract
- and now also curated Octave parity on `sceneCreate('lstar', ...)`, covering the `80x200` stepped L* chart, its monotonically increasing 20-bar luminance sequence, the full photon cube, and the shared `100 cd/m^2` mean-luminance contract
- and now also corrected `sceneCreate('hdr')` mean-luminance normalization plus curated Octave parity on the default `384x384` HDR-lights constructor, covering the normalized mean spectrum, sampled normalized luminance rows through the circle/line/square bands, and the returned `100 cd/m^2` contract
- and now also curated Octave parity on `sceneCreate('macbethtungsten')`, covering the `64x96` Macbeth chart under tungsten illumination together with the returned `100 cd/m^2` photon-cube contract
- and now also curated Octave parity on `sceneCreate('uniform', ...)` / `sceneCreate('uniformEE', ...)`, covering the direct `24x24` equal-energy constructor alias, its flat spatial layout with wavelength-varying photon spectrum, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform equal photon', ...)`, covering the `24x24` equal-photon uniform scene, its flat spectral-spatial photon cube, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniformbb', ...)`, covering the compact blackbody alias and its exact equivalence to `sceneCreate('uniform bb', 16, 4500, ...)`
- and now also Python `sceneCreate('uniformblackbody', ...)` alias coverage, with curated Octave parity against the canonical upstream `sceneCreate('uniformbb', 16, 4500, ...)` constructor and exact equivalence to `sceneCreate('uniform bb', 16, 4500, ...)`
- and now also added the missing upstream `sceneCreate('uniformephoton')` alias in Python, with curated Octave parity on the compact equal-photon alias and exact equivalence to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on Python `sceneCreate('uniformep', ...)`, using the canonical upstream `sceneCreate('uniformephoton', ...)` constructor and locking exact equivalence back to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on Python `sceneCreate('uniformequalphoton', ...)` and `sceneCreate('uniformequalphotons', ...)`, using the canonical upstream singular `sceneCreate('uniformequalphoton', ...)` constructor and locking exact equivalence back to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on `sceneCreate('line', ...)`, covering the direct D65 line-scene alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('lined65', ...)`
- and now also curated Octave parity on Python `sceneCreate('lineequalphoton', ...)`, using the upstream `sceneCreate('lineequalphoton', ...)` constructor and locking exact equivalence back to `sceneCreate('line ep', ...)`
- and now also curated Octave parity on `sceneCreate('lineee', ...)`, covering the compact equal-energy line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('line ee', ...)`
- and now also curated Octave parity on `sceneCreate('impulse1dd65', ...)`, covering the impulse-style D65 line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('lined65', ...)`
- and now also curated Octave parity on `sceneCreate('impulse1dee', ...)`, covering the impulse-style equal-energy line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('line ee', ...)`
- and now also curated Octave parity on `sceneCreate('sinusoid', ...)`, covering the direct harmonic-scene alias, the shared `64x64` multi-frequency photon cube, and its exact equivalence to `sceneCreate('harmonic', ...)`
- and now also curated Octave parity on `sceneCreate('uniformEESpecify', ...)`, covering the explicit `380:10:720 nm` equal-energy uniform scene, its `128x128` geometry, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('exponential intensity ramp', ...)`, covering the `64x64` exponential ramp with dynamic range `256`, exact row-uniformity, monotonic column growth, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('linear intensity ramp', ...)`, covering the `64x64` linear ramp with dynamic range `256`, per-row monotonic growth, the top-row full-scale ratio, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('rings rays', ...)`, covering the `64x64` Mackay-style radial scene at frequency `8`, its row/column center symmetry, and the same `100 cd/m^2` luminance contract
- and now also corrected shared `sceneCreate('point array', ...)` / `sceneCreate('grid lines', ...)` luminance normalization, with curated Octave parity on `sceneCreate('point array', ...)` covering the `64x64` equal-photon point lattice with `16 px` spacing, its exact `4x4` bright-point grid, and the sparse scene geometry used by downstream optics and sensor scripts, and on `sceneCreate('grid lines', ...)` covering the matching `64x64` equal-photon distortion grid with `16 px` spacing, exact `4x4` full-bright line support, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('white noise', ...)`, covering the deterministic `128x128` D65 white-noise scene at `20%` contrast through its `1 deg` FOV, photon-cube geometry, normalized spatial distribution summaries, and normalized mean spectrum rather than brittle raw random pixels
- and now also corrected `sceneCreate('disk array', ...)` mean-luminance normalization plus curated Octave parity on the deterministic `64x64` equal-photon `2x2` disk lattice with radius `8`, covering its exact four connected components and centroids, the two-level photon cube, and the same `100 cd/m^2` luminance contract
- and now also implemented `sceneCreate('square array', ...)` plus curated Octave parity on the deterministic `64x64` equal-photon `2x2` square lattice with side `8`, covering its exact four connected components and centroids, the two-level photon cube, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on the `s_surfaceMunsell.m` workflow, covering explicit `munsell.mat` loading from `data/surfaces/charts`, the illuminant-C wavelength/support vector, the reshaped XYZ-to-sRGB chip rendering, chromaticity and LAB summary bounds, and the first 45 Munsell hue/value/angle records printed by the script
- and now also curated Octave parity on the `s_sceneDemo.m` workflow, covering the `macbethd65` scene creation path, luminance/photons/wave extraction with the script’s max-photon and mean-spectrum invariants, the FOV update to `20 deg`, and the `freq orient pattern` bottom-row luminance plus `550 nm` radiance-line summaries
- and now also curated Octave parity on the `s_sceneExamples.m` workflow, covering the built-in example scene catalog across `rings rays`, frequency-orientation, harmonic, checkerboard, line, slanted bar, grid, point-array, Macbeth tungsten, `uniformEESpecify`, `lstar`, and exponential-ramp scenes, with deterministic center-row/center-column luminance summaries and geometry-only coverage for the default random reflectance chart
- and now also curated Octave parity on the `s_sceneRoi.m` workflow, covering ROI photons/energy/illuminant/reflectance extraction on the default scene, the returned ROI-mean spectra, and exact manual-versus-direct reflectance agreement
- and now also curated Octave parity on the `s_sceneRotate.m` workflow, covering the default star-pattern rotation movie through selected rotated-frame sizes, luminance summaries, and canonicalized center-row/center-column luminance profiles
- and now also curated Octave parity on the `s_sceneWavelength.m` workflow, covering default-scene wavelength resampling through `sceneSet(..., 'wave', ...)`, preserved geometry/luminance at 10 nm, 5 nm, and 2 nm narrowband supports, and the normalized mean/center spectral radiance trends across those three scene states
- and now also headless `ie_save_multispectral_image(...)` / `ieSaveMultiSpectralImage(...)` plus curated Octave parity on the `s_sceneHCCompress.m` workflow, covering `hcBasis(...)`-driven 95% and 99% basis compression of `StuffedAnimals_tungsten-hdrs`, the save/reload roundtrip through basis-coded multispectral MAT files, preserved scene geometry, and the reconstructed 5 nm mean/center spectral trends
- and now also headless `image_increase_image_rgb_size(...)` / `imageIncreaseImageRGBSize(...)` plus curated Octave parity on the `s_sceneIncreaseSize.m` workflow, covering exact pixel-replication resizing by `[2 3]`, `[1 2]`, and `[3 1]`, the preserved mean luminance / normalized mean scene SPD contract across those scene states, and exact replay back to the previous photon cube when striding by the requested enlargement factors
- and now also headless `scene_show_image(...)` / `sceneShowImage(...)`, Haar-path `hdr_render(...)` / `hdrRender(...)`, and MATLAB-style column-vector illuminant adjustment for the `s_sceneRender.m` workflow, with curated Octave parity on the D75-adjusted StuffedAnimals render plus the Feng Office and StuffedAnimals HDR-companded render summaries, center-pixel RGB values, and canonicalized center-row luminance profiles
- and now also MATLAB-style upstream RGB asset lookup in `scene_from_file(..., 'rgb', ...)` plus curated Octave parity on the `t_sceneRGB2Radiance.m` workflow, covering `sceneFromFile('macbeth.tif', 'rgb', ..., display)` across `OLED-Sony`, `LCD-Apple`, and `CRT-Dell`, together with the three display white points, primary chromaticities, scene luminance/SPD summaries, and display-dependent rendered RGB means
- and now also headless `macbeth_read_reflectance(...)` / `macbethReadReflectance(...)`, `xyz_to_srgb(...)` / `xyz2srgb(...)`, and `image_flip(...)` / `imageFlip(...)` plus curated Octave parity on the `t_sceneSurfaceModels.m` tutorial, covering Macbeth SVD basis extraction, low-rank reflectance reconstruction error, the normalized D65 render path, and the rendered 4x6 chart RGB summaries for the 1-D, 2-D, 3-D, 4-D, and full-basis cases
- and now also curated Octave parity on `s_reflectanceBasis.m`, covering the upstream two-file reflectance aggregation path, the 8-D SVD basis subspace, the first four sign-canonicalized basis vectors, and the reduced-dimensional reconstruction RMSE/statistics
- and now also curated Octave parity on `s_colorIlluminantTransforms.m`, covering the blackbody `3500:500:8000 K` transform bank on the default reflectance chart plus the two fixed-matrix cosine-similarity tables from the script
- and now also curated Octave parity on `s_chromaticSpatialChart.m`, covering the synthesized row-chromatic / column-frequency RGB chart, its white-border construction, the `sceneFromFile(..., 'rgb', 100, 'LCD-Apple')` conversion, and the resulting scene luminance/spectrum center-line summaries
- and now also curated Octave parity on `s_colorConstancy.m`, covering the stuffed-animals and uniform-D65 blackbody sweeps across the script's 15 reciprocal-temperature samples together with the mean-luminance and normalized RGB render summaries for each frame
- and now also headless `srgb_to_color_temp(...)` / `srgb2colortemp(...)` plus curated Octave parity on `s_rgbColorTemperature.m`, covering the exact Macbeth tungsten/D65 camera pipeline, the returned coarse temperature buckets, the shared `2500:500:10500 K` chromaticity lookup table, and the normalized rendered-RGB means used by the script
- and now also headless `scene_reflectance_chart(...)` / `sceneReflectanceChart(...)` plus `srgb_parameters(...)` / `adobergb_parameters(...)`, with curated Octave parity on `s_srgbGamut.m`, covering the natural/synthetic reflectance charts, the sRGB and Adobe RGB gamut polygons, and the D65/tungsten chromaticity clouds computed from the script's explicit surface sets
- and now also Phase 4 metrics coverage has started with new `iso.py`, headless `ISOFindSlantedBar(...)` / `ieCXcorr(...)` / `edge_to_mtf(...)`, curated Octave parity on `s_metricsEdge2MTF.m`, and a MATLAB-faithful odd-sized `sceneCreate('slanted bar', ...)` generator that matches the upstream edge orientation and geometry
- and now also headless `ISO12233(...)` / `ieISO12233(...)` in `iso.py`, with curated Octave parity on `s_metricsMTFSlantedBar.m` covering the direct RGB slanted-bar path, the `ieISO12233(ip, sensor, ...)` sensor-space path, the monochrome direct path, and their stable ESF/LSF/MTF50/aliasing contracts under case-scoped parity tolerances
- and now also curated Octave parity on `s_metricsMTFPixelSize.m`, covering the fixed-die-size monochrome slanted-bar sweep across 2/3/5/9 um pixels, the per-pixel-size sensor geometry and ROI replay, Nyquist and MTF50 trends, and the normalized luminance MTF profiles under case-scoped parity tolerances
- and now also headless `pixelVperLuxSec(...)` / `pixelSNRluxsec(...)` in `sensor.py`, with curated Octave parity on `s_metricsSNRPixelSizeLuxsec.m` covering the monochrome 2/4/6/9/10 um pixel sweep, its lux-sec sensitivity curves, and the returned SNR/read-noise/shot-noise summaries
- and now also curated Octave parity on `s_metricsMTFSlantedBarInfrared.m`, including `ieReadColorFilter(..., 'IRBlocking')` compatibility, the NikonD200IR multispectral slanted-bar setup, the fixed-ROI direct ISO12233 branch, and the IR-blocked auto-ROI `ieISO12233(...)` line-spread/MTF branch
- and now also headless `cameraMTF(...)` / `cameraAcutance(...)` plus `cpiqCSF(...)` / `ISOAcutance(...)`, with curated Octave parity on `s_metricsAcutance.m` covering the default camera slanted-edge workflow, the luminance MTF and CPIQ weighting profiles, the sensor degrees-per-distance conversion, and the final acutance scalar
- and now also headless `cameraColorAccuracy(...)` plus `macbethCompareIdeal(...)`, with curated Octave parity on `s_metricsColorAccuracy.m` covering the default Macbeth-camera workflow, normalized white-point recovery, per-patch Delta E / LAB results, and the returned sRGB Macbeth comparison patches
- and now also headless `macbethColorError(...)`, with curated Octave parity on `s_metricsMacbethDeltaE.m` covering the default scene/OI/sensor/IP current-matrix workflow, recovered sensor CCM, normalized white-point recovery, per-patch Delta E / LAB outputs, and processed-image RGB summary statistics
- and now also curated Octave parity on `s_metricsSPD.m`, covering the daylight `4000:500:7000 K` comparison sweep for both the D4000-reference and fixed-D65-white-point branches, including the returned angle, CIELAB Delta E, and mired curves
- and now also headless `cameraVSNR(...)`, with curated Octave parity on `s_metricsVSNR.m` covering the default camera uniform-field sweep, the stable ROI geometry, normalized VSNR / reciprocal-Delta-E curves across the three light levels, and processed-image RGB summary statistics
- and now also headless `scielabRGB(...)` / `scielab(...)` in a new `scielab.py`, plus MATLAB-style display `rgb2xyz` / `white point` support, with curated Octave parity on `t_metricsScielab.m` covering the `hats.jpg` versus `hatsC.jpg` LCD-Apple tutorial workflow, returned scene geometry, display white point, and the stable S-CIELAB error-map summaries
- and now also curated Octave parity on `s_rgb2scielab.m`, covering the `crt.mat` S-CIELAB script path for `hats.jpg` versus `hatsC.jpg`, including the mean Delta E, the mean/percent above the script’s `Delta E > 2` threshold, scene geometry, display white point, and a normalized error-profile slice
- and now also headless `dac2rgb(...)` / `imageLinearTransform(...)`, with curated Octave parity on `s_scielabExample.m` covering both the script’s `scielabRGB(...)` branch and its explicit `dac2rgb -> displayGet('rgb2xyz') -> imageLinearTransform -> scielab(...)` walkthrough on `hats.jpg` versus `hatsC.jpg`, including the returned scene geometry, white point, mean Delta E summaries, and canonicalized filter/error profiles
- and now also headless `scPrepareFilters(...)` / `sc_prepare_filters(...)`, with curated Octave parity on `s_scielabFilters.m` covering the 101-sample point-spread filters, the 512-sample MTF branch, and the `distribution` / `original` / `hires` Gaussian-parameter variants through normalized center-row and peak summaries
- and now also headless `sceneAdd(...)` / `scene_add(...)` plus `sceneGet(..., 'illuminant xyz')`, with curated Octave parity on `s_scielabMTF.m` covering the harmonic `1:32 cpd` sweep, the `remove spatial mean` scene-add path, the returned illuminant XYZ/SPD contract, and the CIELAB versus S-CIELAB Delta E curves
- and now also curated Octave parity on `s_scielabPatches.m`, covering the 7x7 uniform-patch illuminant-perturbation sweep, the returned illuminant XYZ/SPD setup for `sceneCreate('uniform')`, and the paired CIELAB / S-CIELAB Delta E surfaces together with the script’s quantized S-CIELAB summary vector
- and now also curated Octave parity on `s_scielabMasking.m`, covering the fixed `4 cpd` / `0.8` harmonic-mask setup, the `remove spatial mean` target-combination path, the doubled white-point convention from the script, and the returned CIELAB / S-CIELAB target-contrast curves
- and now also headless `colorTransformMatrix(...)` / `scOpponentFilter(...)` / `scComputeSCIELAB(...)` plus `srgb2xyz(...)`, with curated Octave parity on `s_scielabTutorial.m` covering the multispectral Stuffed Animals scene through OI/sensor/IP, the Gray World rendered RGB handoff, the 50 samples-per-degree SCIELAB filter setup, the opponent/filtered-image summaries, and the final SCIELAB LAB output
- and now also curated Octave parity on `s_scielabHarmonicExperiments.m`, covering the sweep-frequency scene through diffuser-blurred OI, gray-world IP rendering, the script’s three opponent-channel scaling branches, and the resulting padded S-CIELAB error-surface summaries

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

Generate the current machine-readable migration gap ledger and audit summary:

```bash
python tools/audit_migration_gap.py
```

This writes a JSON-compatible YAML ledger to
`docs/migration-gap-ledger.yaml` and a summary snapshot to
`reports/migration-gap/latest.json`.

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
The same sensor/tutorial wave now also includes headless raw DNG support
through `ie_dng_read` / `ieDNGRead`, `ie_dng_simple_info` /
`ieDNGSimpleInfo`, and `sensor_dng_read` / `sensorDNGRead`, covering the
checked-in Pixel 4a `MCC-centered.dng` workflow from
`t_sensorReadRaw.m` and upstream-backed `sensorCreate('IMX363')` crop
flows. The raw DNG parity case now matches decoded `digital_values`
exactly and checks the rendered RGB result with a bounded normalized-MAE
comparator to absorb the remaining legacy display-gain mismatch between
the Python and Octave render paths. That same tutorial path now also includes headless
`sensor_plot_line` / `sensorPlotLine` coverage for the
`t_sensorSpatialResolution.m` line-profile workflow, including the exact
`sensorSet(..., 'pixel size Constant Fill Factor', ...)` tutorial path and
curated parity on the coarse/fine sensor-vs-OI line data, and headless
`sensor_description` / `sensorDescription` coverage for the
`t_sensorFPN.m` summary path, with curated parity on the returned summary
table fields for the fixed-noise setup used by that tutorial, plus direct
curated parity on the four `noise flag` modes and two-line volts plot used
by the rest of that tutorial, plus ROI-DV summary parity for
`s_sensorPoissonNoise.m`. The same sensor tutorial wave now also
includes `signal_current` / `signalCurrent` coverage for the
`t_sensorInputRefer.m` current-to-electrons workflow, plus
`ie_read_spectra` / `ieReadSpectra` coverage for the core spectral-loading
path used by `t_sensorEstimation.m`, together with curated parity on the
Macbeth/D65/scanner pseudoinverse estimation workflow,
plus direct curated parity on the daylight-basis Macbeth illuminant solve
from `s_sensorMacbethDaylightEstimate.m`,
plus `sensor_color_filter` / `sensorColorFilter`,
`ie_read_color_filter` / `ieReadColorFilter`, and
`ie_save_color_filter` / `ieSaveColorFilter` coverage for the Gaussian
color-filter roundtrip workflow from `s_sensorGaussianFilter.m`,
plus direct curated parity on calibrated `ieReadColorFilter('NikonD100')`
asset loading from `s_sensorPlotColorFilters.m`,
plus direct curated parity on the simulated filter-responsivity and
spectral-filter recovery workflow from `s_sensorSpectralEstimation.m`,
plus direct curated parity on the CFA construction workflow from
`s_sensorCFA.m`, including `sensorCreate('ycmy')` and
`sensorSet(..., 'pattern and size', pattern)` behavior,
plus
`filter transmissivities` and overexposed `ipCompute` color-shift coverage
from `t_sensorExposureColor.m`, and `ieFitLine`-driven dark-voltage
estimation coverage for `s_sensorAnalyzeDarkVoltage.m`, plus slope-based
PRNU estimation coverage for `s_sensorSpatialNoisePRNU.m`, plus DSNU
estimation coverage for `s_sensorSpatialNoiseDSNU.m`, plus the
`s_sensorCountingPhotons.m` workflow with MATLAB-style
`sceneCreate('uniform equal photon', [rows cols])`,
`sceneSet(..., 'mean luminance', ...)`, and
`oiGet(..., 'optics aperture diameter', unit)` support, plus headless
`sensor_formats` / `sensorFormats` and `ie_n_to_megapixel` /
`ieN2MegaPixel` coverage for `s_sensorSizeResolution.m`, plus direct
curated parity on the `s_sensorCFA.m` zebra-scene workflow through the
default Bayer, explicit Bayer, YCMY, custom 3x3 RGB, RGBW, and quad-CFA
branches, including MATLAB-style
`cameraSet(..., 'pixel size constant fill factor', ...)` routing, exact
pattern/size metadata, and normalized sensor-RGB summaries, plus direct
curated parity on the cropped CFA point-image workflow from
`s_sensorCFAPointSpread.m`, comparing the 12x12 CFA-image crops,
rendered RGB summaries, and canonical center-line profiles across the
`f/#` sweep, plus direct
curated parity for the `s_sensorSNR.m` sensor-noise component curves
returned by `sensorSNR(...)`, plus direct curated parity for the
`s_sensorExposureBracket.m` stacked multi-exposure sensor workflow, plus
direct curated parity for the CFA-matched exposure-duration workflow from
`s_sensorExposureCFA.m`, plus headless `sensor_create_array` /
`sensorCreateArray`, `sensor_create_split_pixel` / `sensorCreateSplitPixel`,
and `sensor_compute_array` / `sensorComputeArray` coverage for the
OVT split-pixel saturated-combine workflow that underlies
`s_sensorSplitPixel.m`. That same sensor wave now also includes
sensor-array `sensor_compute(...)` support together with
`ie_read_spectra('Foveon'/'NikonD1')` sensor-color-filter asset loading,
covering the stacked monochrome-plane and triple-well image-processing
workflow from `s_sensorStackedPixels.m` with curated parity on the stable
stacked-plane and Bayer-comparison summaries. That same Phase 1 sensor
wave now also includes direct `mlensCreate` / `mlensSet` / `mlensGet` /
`mlRadiance` / `mlAnalyzeArrayEtendue` coverage for the array-efficiency
and optimal-offset workflow from `s_sensorMicrolens.m`, with curated
Octave parity on the no-microlens/centered/optimal etendue maps, the
optimal-offset curve and offset maps, and the single-pixel irradiance
midlines used in the script’s chief-ray-angle illustration. That same
sensor wave now also includes MATLAB-style `sceneSet(..., 'resize', ...)`,
`sceneCombine(...)`, and `sensorSet(..., 'hfov'/'vfov', ...)` coverage for
the mixed-sensor comparison workflow from `s_sensorComparison.m`, with
curated Octave parity on the combined-scene geometry plus the IMX363,
MT9V024 RCCC, and CYYM small-pixel/large-pixel response summaries. That
same sensor wave now also includes headless `sensorComputeSamples(...)`
for the repeated-capture workflow from `s_sensorNoise.m`, with curated
Octave parity on stable sample-stack noise statistics instead of raw
RNG-matched captures. That same sensor compatibility surface now also
includes headless `sensor_clear_data(...)` / `sensorClearData(...)`,
matching the direct MATLAB computed-payload reset path for sensor
objects, plus headless `sensor_show_image(...)` / `sensorShowImage(...)`
as a direct wrapper over the existing sensor RGB render path using the
MATLAB default `dv or volts` selection, plus headless
`sensor_save_image(...)` / `sensorSaveImage(...)` for the direct MATLAB
PNG-export helper on top of the same rendered sensor RGB path, plus
headless `sensor_show_cfa(...)` / `sensorShowCFA(...)` for the direct
MATLAB CFA-pattern render helper on top of the same sensor rendering
surface, plus headless `sensor_image_color_array(...)` /
`sensorImageColorArray(...)` for the legacy CFA-letter to color-order map
helper used by the older sensor CFA utilities, plus headless
`sensor_show_cfa_weights(...)` / `sensorShowCFAWeights(...)` for the
weighted CFA-visualization helper used by the same legacy sensor CFA
utilities, plus headless `sensor_color_order(...)` / `sensorColorOrder(...)`
and `sensor_determine_cfa(...)` / `sensorDetermineCFA(...)` for the legacy
CFA color-hint and tiled-pattern helpers used across the older sensor CFA
utilities. That same Phase 1 sensor
wave now also includes
public `sceneRotate(...)` / `oiCrop(...)` support for the deterministic
rolling-shutter assembly path from `s_sensorRollingShutter.m`, with
curated Octave parity on the per-frame mean-voltage trace, final
row summaries, and normalized final RGB summaries. That same Phase 1
sensor wave now also includes `sensorCreate('imx490-large')`,
`sensorCreate('imx490-small')`, `oiCrop(..., 'border')`,
`oiSpatialResample(...)`, and headless `imx490Compute(...)` coverage for
the HDR four-capture combine workflow from `s_sensorIMX490.m`, with
curated Octave parity on the deterministic uniform-field best-SNR path,
including capture names, capture mean electrons/volts/DV, combined-volts
summaries, and selected pixel-source counts. That same Phase 1 sensor
wave now also includes `sceneFromFile(..., 'multispectral', ...)`
support for basis-coded HDR scene assets plus the mixed pixel-size
monochrome workflow from `s_sensorHDR_PixelSize.m`, with curated Octave
parity on the Feng Office HDR scene path, comparing the three sensor
sizes, mean/p95 volts, mean electrons, and grayscale output summaries
for the 1/2/4 micron pixel-size sweep. That same Phase 1 sensor wave now
also includes `sceneCreate('linear intensity ramp')`,
`sceneCreate('exponential intensity ramp')`, public `sensor_dr` /
`sensorDR`, and headless log-response sensor compute support for
`s_sensorLogAR0132AT.m`, with curated Octave parity on the stable
noise-free log-response line-shape workflow and the script’s 1-second
dynamic-range check. That same Phase 1 sensor wave now also includes the
full `s_sensorAliasing.m` workflow, with curated Octave parity on the
noise-free sweep-frequency aliasing line profiles for the fine-pixel,
coarse-pixel, and blurred-lens cases plus canonicalized slanted-bar
sensor captures for the sharp and anti-aliased large-pixel paths. That
same Phase 1 sensor wave now also includes the asset-driven DUT setup
workflow from `s_sensorExternalAnalysis.m`, with curated Octave parity on
the configured wave/filter/QE payload, CFA pattern, pixel metadata, and
imported `dutData.mat` voltage summaries. That same Phase 1 sensor wave
now also includes the spectral-radiometer workflow from
`s_spectralRadiometer.m`, with curated Octave parity on the radiometer
filter geometry, deterministic noise-free electron line, theoretical
shot-noise curve, and stable photon-noise summary statistics.
Phase 2 optics coverage now also includes direct `optics_coc` /
`opticsCoC` support for the thin-lens circle-of-confusion workflow from
`s_opticsCoC.m`, with curated Octave parity on the 50 mm F/2 and F/8
diameter-versus-distance sweeps at 0.5 m and 3 m focus.
That same Phase 2 optics wave now also includes the wide-field
`s_opticsCos4th.m` vignetting workflow, with curated Octave parity on the
default and 4x-focal-length illuminance hline profiles and their mean
illuminance summaries for a uniform D65 scene at 80 degrees FOV.
That same optics wave now also includes MATLAB-style
`scene_interpolate_w` / `sceneInterpolateW` behavior through
`sceneSet(..., 'wave', ...)`, plus curated parity for the
`s_opticsGaussianPSF.m` point-array workflow on the normalized blurred
center row and column profiles across 450 nm, 550 nm, and 650 nm.
That same Phase 2 optics wave now also includes the diffraction-limited
`s_opticsPSFPlot.m` workflow, with curated parity on the 600 nm, `f/#=12`,
`nSamp=100` PSF surface and Airy-disk radius returned by the underlying
headless PSF data contract.
That same optics wave now also includes the `s_opticsPadCrop.m` workflow,
with curated parity on the padded OI size, MATLAB crop rectangle, cropped
OI geometry, and the deterministic sensor line-profile outputs for both
the scene-FOV crop path and the expanded padded-FOV path.
That same Phase 2 optics wave now also includes the `s_opticsMicrolens.m`
workflow, with curated parity on the microlens getter/setter contract,
default 30-degree sensor FOV setup, and the radiance-generated source and
pixel irradiance center-line outputs plus etendue.
That same Phase 2 optics wave now also includes the
`s_opticsFlare.m` workflow, with upstream-style `sceneCreate('hdr')`
support via the HDR-lights scene family, deterministic seeded flare
apertures for the Python parity path, and curated Octave parity on the
dirty-aperture summaries, normalized 550 nm PSF rows, blur-width
statistics, and HDR/point-scene OI geometry returned by the script.
That same Phase 2 optics wave now also includes the
`s_opticsFlare2.m` workflow, with curated Octave parity on the stable
dirty-aperture script contract: aperture sums, point-scene blur-width
summaries, HDR-scene geometry, and normalized HDR irradiance ratios
across the six-sided, five-sided, and defocused branches.
That same Phase 2 wavefront block now also includes `ie_mvnrnd` /
`ieMvnrnd` together with the `s_wvfThibosModel.m` workflow, with curated
Octave parity on deterministic virtual-eye coefficient draws, the
sample-mean subject PSFs across 450/550/650 nm, and the selected
example-subject PSF center rows at 450 nm and 550 nm.
That same Phase 2 wavefront block now also includes the
`s_wvfZernikeSet.m` workflow, with curated Octave parity on the
defocus-plus-vertical-astigmatism loop, comparing the returned Zernike
coefficients, 550 nm PSF center rows, and uncropped 550 nm OI center
rows for `A = [-1, 0, 1]` at `D = 2` microns.
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
That same Lorentzian `siSynthetic(...)` workflow now also has direct
Octave parity on `oiPlot(..., 'psf', 550)`, comparing the returned PSF
surface and spatial support grids on the shift-invariant optics object
before image formation.
That same synthetic-PSF surface now also has direct Octave parity on the
pillbox `siSynthetic(...)` workflow from `s_opticsSIExamples.m`,
including the returned wave axis and the 550 nm input PSF center row.
That same synthetic-PSF surface now also has direct Octave parity on the
anisotropic Gaussian `siSynthetic(...)` workflow from
`s_opticsSIExamples.m`, including the returned wave axis plus the 550 nm
input PSF center row.
That same anisotropic Gaussian workflow now also has direct Octave parity
on `oiPlot(..., 'psf', 550)`, comparing the returned PSF surface and
spatial support grids at 550 nm.
That same anisotropic Gaussian workflow now also has direct Octave parity
on the `oiPlot(..., 'illuminance hline'/'illuminance vline', ..., 'nofigure')`
comparison used in `s_opticsSIExamples.m`, comparing the returned
horizontal and vertical illuminance line profiles through the image center.
The adjacent wavefront helper surface now also includes
`wvf_set` / `wvfSet`, `wvf_get` / `wvfGet`,
`wvf_compute` / `wvfCompute`,
`wvf_defocus_diopters_to_microns` /
`wvfDefocusDioptersToMicrons`, `wvf_to_oi` / `wvf2oi`, direct
`oi_compute(wvf, scene)` support, MATLAB-style `oi_get(..., 'wvf ...')`,
and rebuilding `oi_set(..., 'wvf ...')` / `oi_set(..., 'optics wvf', ...)`
for script-driven WVF defocus/Zernike workflows.
That same WVF-to-OI bridge now also has direct Octave parity on the
`s_wvfOI.m` PSF-slice workflow, comparing both
`wvfGet(..., 'psf xaxis', ...)` / `wvfGet(..., 'psf yaxis', ...)` against
`oiGet(wvf2oi(wvf), 'optics psf xaxis'/'optics psf yaxis', ...)` at
550 nm.
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
and now includes `wvf_load_thibos_virtual_eyes` /
`wvfLoadThibosVirtualEyes` for the upstream pupil-size scripts, with the
MATLAB default mean-vector return plus an explicit full-output path for
covariance and measured-subject coefficient matrices.
That same pupil-size workflow now also supports MATLAB key/value
`wvfCreate(...)` construction together with `lcaMethod='human'`, with
direct Octave parity on the Thibos pupil-size compute path.
That same pupil-size workflow now also has direct Octave parity on the
measured-vs-calculated pupil comparison path from `s_wvfPupilSize.m`,
holding the calculated pupil fixed while varying the measured Thibos
coefficient set.
The same WVF surface now also supports the `s_wvfPSFSpacing.m` numeric
path through `wvfSet(..., 'psf sample spacing', ...)`, with direct Octave
parity on the resulting field-size and sampling relationship.
That same script-driven WVF surface now also has direct Octave parity on
the defocus/vertical-astigmatism sweep from `s_wvfAstigmatism.m`,
comparing the returned normalized PSF center rows and columns at 550 nm
across the full 3x3 Zernike grid.
That same script-driven WVF surface now also has direct Octave parity on
the diffraction workflow from `s_wvfDiffraction.m`, comparing the initial
Airy-disk match and `wvf2oi` f-number handoff, the pupil-size sweeps at
550 nm and 400 nm, the human-LCA wavelength sweep, and the focal-length
`um per degree` scaling.
That same script-driven WVF helper surface now also includes
`wvfOSAIndexToZernikeNM` / `wvfZernikeNMToOSAIndex`, with direct Octave
parity on the OSA index round-trip used by `s_wvfWavefronts.m`.
That same script-driven WVF surface now also has direct Octave parity on
the 16-coefficient OSA-index sweep from `s_wvfWavefronts.m`, comparing
the returned spatial support axis, normalized wavefront center rows and
columns, and peak absolute wavefront amplitude for indices 1:16 at a
2 mm pupil.
That same Phase 2 wavefront block now also has direct Octave parity on
`s_zernikeInterpolation.m`, comparing the interpolated-vs-ground-truth
Zernike coefficient vector at the target field height and the normalized
center-row PSFs for Zernike-space interpolation, ground truth, and the
script's direct space-interpolation path.
That same script-driven WVF plotting block now also has direct Octave
parity on the end-to-end sequence in `s_wvfPlot.m`, comparing the 550 nm
mixed-unit 1D PSF payloads and the recalculated 460 nm PSF-angle /
pupil-phase payloads after `wvfSet(..., 'wave', 460)` and `wvfCompute(...)`.
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
That same focus/DoF optics block now also has direct Octave parity on
`s_opticsDefocus.m`, comparing the blur/astigmatism/pupil-diameter WVF
sequence through mean-photon conservation, recovered Zernike coefficients,
pupil-diameter changes, and normalized 550 nm center-row irradiance slices.
That same focus/DoF optics block now also includes public
`optics_defocus_displacement` / `opticsDefocusDisplacement`, with direct
Octave parity on `s_opticsDefocusDisplacement.m` for the base-power sweep,
image-plane displacement curves, and constant displacement-to-focal-length
ratio check.
That same focus/DoF optics block now also has direct Octave parity on
`s_opticsDoF.m`, comparing the 100 mm F/2 thin-lens DOF formula result,
the CoC-derived DOF crossing estimate, and the object-distance by f-number
DOF sweep surface.
That same focus/DoF optics block now also includes public
`optics_depth_defocus` / `opticsDepthDefocus`, with direct Octave parity
on `s_opticsDepthDefocus.m` for the default-lens object-distance sweep,
the focal-plane and 1.1x-focal-length defocus curves, the in-focus object
distance for the shifted image plane, and the pupil-scaled Hopkins `w20`
surface.
That same focus/DoF optics block now also includes public
`optics_defocus_core` / `opticsDefocusCore` and
`optics_build_2d_otf` / `opticsBuild2Dotf`, with direct Octave parity on
`s_opticsDefocusScene.m` for the multispectral D65-adjusted defocus-image
workflow, the derived 10 um and 40 um sensor-plane defocus values, and
the normalized 550 nm center-row blur states for the focused and defocused
OI outputs.
That same focus/DoF optics block now also includes direct Octave parity on
`s_opticsDefocusWVF.m`, comparing the explicit diffraction-limited and
1.5-diopter WVF branches with the alternate `oiCreate('wvf')` update path
through their returned f-numbers, defocus coefficients, and normalized
550 nm PSF / OI center rows.
The adjacent diffraction-limited optics plotting surface now also includes
headless `oiPlot(..., 'ls wavelength'/'lswavelength'/'otf wavelength'/'mtf wavelength')`
support for the script-driven line-spread and OTF-by-wavelength workflows
used in the upstream diffraction PSF tutorials.
That same line-spread plotting surface now also has direct Octave parity on
the WVF-backed `oiPlot(..., 'ls wavelength')` workflow, comparing the
returned spatial axis, wavelength vector, and line-spread-by-wavelength
surface after `wvf2oi`.
That same OTF-by-wavelength plotting surface now also has direct Octave
parity on the diffraction-limited `oiPlot(..., 'otf wavelength')`
workflow, comparing the returned frequency axis, wavelength vector, and
OTF-by-wavelength surface.
That same WVF-backed OI plotting surface now also has direct Octave parity
on `oiPlot(..., 'otf wavelength')`, comparing the returned frequency axis,
wavelength vector, and OTF-by-wavelength surface after `wvf2oi`.
That same diffraction image-formation plotting surface now also has
direct Octave parity on `oiPlot(..., 'irradiance hline', [80 80])` from
`s_opticsImageFormation.m`, comparing the returned spatial axis,
wavelength vector, and irradiance mesh data for the diffraction-limited
line scene.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf', [], 550)` parity coverage for the upstream
`s_opticsDLPsf.m` workflow.
That same diffraction plotting surface now also includes the script-level
`s_opticsDiffraction.m` workflow, with direct Octave parity on the
default-to-`f/12` diffraction-limited point-array path, including the
unit-aware `oiGet(..., 'optics focal length'/'optics pupil diameter',
...)` parameter checks, the returned `psf 550` surface, the
`ls wavelength` payload, and stable 550 nm image-blur width summaries.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf xaxis', [], 550, 'um')` parity coverage for the
upstream diffraction and WVF spatial scripts.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf yaxis', [], 550, 'um')` parity coverage for the
upstream diffraction and WVF spatial scripts.
The script-driven custom optics surface now also includes direct
`opticsPSF2OTF(...)` parity coverage for the upstream flare-image
workflow from `s_opticsPSF2OTF.m`.
The adjacent `s_wvfOI.m` bridge is now covered on the direct OTF path
too: `wvf2oi(...)`-backed OIs can synthesize `oiGet(..., 'optics otf')`
from the embedded wavefront, and curated parity checks that against
`wvfGet(..., 'otf')` after the same `ifftshift` relationship used by the
MATLAB script.
The adjacent wavefront plotting surface now also includes headless
`wvfPlot(...)` support for script-driven PSF, 1D PSF, pupil amplitude,
pupil phase, wavefront-aberration, PSF-angle, and OTF views, backed by
wavelength-aware `wvfGet(...)` support for `psf`, `pupil function`,
`wavefront aberrations`, `psf spatial samples`, `psf angular samples`,
`pupil spatial samples`, `1d psf`, and OTF support.
That same WVF plotting surface now also has direct Octave parity on the
`wvfPlot(..., '2d otf', ...)` workflow from `s_wvfPlot.m`, including the
returned OTF support axis and center-row magnitude data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '2d otf normalized', ...)`, including the returned OTF
support axis and normalized center-row magnitude data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf', ...)`, including the returned cropped OTF support
axis and OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf normalized', ...)`, including the returned cropped
OTF support axis and normalized OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf angle', ...)`, including the returned cropped
frequency support axis and OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf angle normalized', ...)`, including the returned
cropped frequency support axis and normalized OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d psf space', ...)`, including the returned spatial
support axis and 1D PSF line data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '2d wavefront aberrations space', ...)`, including the
returned spatial support axis and wavefront-aberration center row.
The shift-invariant optics script coverage now also has direct Octave
parity on the circular Gaussian `siSynthetic('gaussian', ..., xyRatio=1)`
workflow from `s_opticsSIExamples.m`.
The same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil amp', ...)`, including the returned
spatial support axis and pupil-amplitude center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d pupil amplitude space', ...)`, including the
returned spatial support axis and pupil-amplitude center row for the
exact legacy alias.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil phase', ...)`, including the
returned spatial support axis and pupil-phase center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d pupil phase space', ...)`, including the
returned spatial support axis and pupil-phase center row for the exact
legacy plot name used by the upstream Zernike tutorial.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image wavefront aberrations', ...)`, including
the returned spatial support axis and wavefront-aberration center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf', ...)`, including the returned
spatial support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf', ..., 'airy disk', true)`, including
the returned spatial support axis, PSF center row, and Airy-disk radius.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf normalized', ...)`, including the
returned spatial support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf', 'unit', 'mm', ...)`, including the
returned cropped spatial support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf normalized', 'unit', 'mm', ...)`, including
the returned cropped spatial support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf xaxis', ..., 'airy disk', true)`, including
the full returned support/data payload and Airy-disk radius from
`s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf xaxis', ...)`, including the full returned
support/data payload from `s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf yaxis', ..., 'airy disk', true)`, including
the full returned support/data payload and Airy-disk radius from
`s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf yaxis', ...)`, including the full returned
support/data payload from `s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf angle', ...)`, including the returned
angular support axis and PSF center row.
That same script-driven WVF plotting surface now also has Octave-backed
parity on the normalized `image psf angle` numerical contract,
including the returned angular support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d psf angle', ...)`, including the returned
angular support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d psf angle normalized', ...)`, including the
returned angular support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf', ...)`, including the returned cropped
spatial support axis and raw PSF line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf normalized', ...)`, including the
returned cropped spatial support axis and normalized line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf angle', ...)`, including the returned
cropped angular support axis and raw line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf angle normalized', ...)`, including the
returned cropped angular support axis and normalized line slice.
That same optics helper surface now also includes `airy_disk` /
`airyDisk` plus headless airy-disk overlay payloads for `wvfPlot(...)`
and `oiPlot(...)`, with direct Octave parity on the scalar/image helper
contract used by the upstream diffraction plotting workflows.
That same script-driven WVF surface now also includes the spatial getter
family used by `s_wvfSpatial.m`, including `calc nwave`, `psf sample spacing`,
`ref psf sample interval`, `pupil sample spacing`, `pupil positions`,
`pupil function amplitude`, and `pupil function phase`.
That same script-driven WVF surface now also has direct Octave parity on
the scalar control workflow from `s_wvfSpatial.m`, comparing how `npixels`,
`pupil plane size`, and `focal length` change PSF sample spacing and
`um per degree`.
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
That same ray-trace block now also has direct Octave parity on
`s_opticsRTSynthetic.m`, comparing the synthetic geometry and relative
illumination curves, normalized 550 nm center/edge PSF slices, and the
resulting point-array OI summary outputs from the synthetic increasing-Gaussian
ray-trace workflow.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTGridLines.m`, comparing the staged `rtGeometry` /
`rtPrecomputePSF` / `rtPrecomputePSFApply` path against automated
ray-trace and matched diffraction-limited outputs through the returned
gridline image sizes, sampled PSF support metadata, and normalized 550 nm
center-row image slices for the wide-field and reduced-FOV branches.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTPSF.m`, comparing the stored sampled ray-trace PSF metadata,
normalized 550 nm center/edge PSF slices from the computed `sampledRTpsf`
cache, and the rendered point-array ray-trace versus diffraction-limited
image summaries used by the upstream PSF comparison workflow.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTPSFView.m`, comparing the stored sampled ray-trace PSF view
metadata, the normalized 550 nm field-height and angle-sweep PSF rows,
their stable blur-width summaries, and the canonicalized center/edge
`rtPlot(..., 'psf', ...)` row profiles used by the upstream PSF-view
workflow.
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
Missing session windows now fall back to lightweight headless
placeholders so MATLAB-style `vcSelectFigure(...)` and
`ieRefreshWindow(...)` calls do not fail just because no GUI app has been
created.
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
The adjacent image-processing demosaic surface now also supports
MATLAB-style `ieNearestNeighbor`, `Laplacian`, and
`AdaptiveLaplacian` routing through
`ipSet(..., 'demosaic method', ...)` for RGB Bayer sensors, with the
same bilinear fallback that upstream `Demosaic.m` uses for the
unsupported `gbrg` adaptive-Laplacian branch.
That same headless image-processing surface now also exposes
`demosaic(...)` / `Demosaic(...)`, matching the tutorial-visible
`Demosaic(ip, sensor)` entry point and preferring the current
`ip.data['input']` mosaic when it is present.
That same image-processing surface now also exposes
`image_sensor_conversion(...)` / `imageSensorConversion(...)` and
`image_sensor_correction(...)` / `imageSensorCorrection(...)`, matching
the direct MATLAB sensor-to-ICS entry points on top of the existing IP
sensor-conversion path.
That same image-processing surface now also exposes
`image_illuminant_correction(...)` / `imageIlluminantCorrection(...)`,
matching the direct MATLAB ICS white-balance entry point on top of the
existing IP illuminant-correction path.
That same image-processing surface now also exposes
`image_rgb_to_xyz(...)` / `imageRGB2XYZ(...)`, matching the direct MATLAB
display-linear RGB-to-XYZ entry point used by Macbeth and display-plot
helpers while preserving RGB or XW data format.
That same image-processing surface now also exposes
`display_render(...)` / `displayRender(...)`, matching the direct MATLAB
ICS-to-linear-display entry point on top of the existing IP display-render
stage.
That same image-processing surface now also exposes
`ip_clear_data(...)` / `ipClearData(...)`, matching the direct MATLAB
computed-payload reset path and the `ipSet(..., 'data', [])` behavior.
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
The script-driven sensor slice now also covers the headless
`s_sensorMCC.m` workflow: `sensorCreate('bayer (gbrg)')`, headless
`sensorCCM(...)` chart fitting from stored corner points, and the fixed
`ipSet(..., 'conversion method sensor', 'current matrix')` path now all
run end to end with an Octave-backed parity case on the stable CCM and
corrected-render summaries. The adjacent `s_sensorRollingShutter.m`
workflow is now also covered headlessly through `sceneRotate(...)`,
`oiCrop(...)`, and a deterministic rolling-shutter assembly parity case
that compares the temporal sensor trace plus the final assembled render.
