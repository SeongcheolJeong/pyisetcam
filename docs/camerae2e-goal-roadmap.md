# CameraE2E Goal Roadmap

이 문서는 `camerae2e-technical-overview-images.pptx`가 설명한 목표와 현재 저장소 구현을 대조해, Research-grade E2E 최적화 플랫폼과 Perception 학습용 RAW 데이터 팩토리까지 가는 실행 기준을 정리한다.

중요한 경계는 명확하다. 현재 목표는 제품 sign-off가 아니다. FDTD, TCAD, RayOptics, HW ISP profile은 CameraE2E 파이프라인에 연결되어 있지만, measured calibration 없이 `calibrated` 또는 sign-off claim으로 승격하지 않는다.

## Readiness Tier

| Tier | 의미 |
|---|---|
| `missing` | 알려진 asset/API지만 현재 workspace에서 찾을 수 없음 |
| `available` | 로드 가능하지만 강한 검증 claim은 없음 |
| `proxy` | 연구 탐색에는 사용 가능하지만 물리/시스템 proxy가 포함됨 |
| `validated` | 저장소 검증 gate를 통과한 research-use 기능 |
| `calibration_required` | framework는 연결됐지만 정량 정확도에는 측정 calibration이 필요함 |
| `calibrated` | measured calibration evidence가 붙어 정량 분석에 쓸 수 있음 |

## Capability Matrix

| Area | Current tier | 구현된 것 | 남은 일 |
|---|---:|---|---|
| Scene | `validated` | spectral scene, chart scene, RGB/multispectral import, illuminant control | 실제 scene/illumination capture calibration |
| Optics / RayOptics | `proxy` | Lens DB 검색, RayOptics geometric PSF, CameraE2E optics 실행 | diffraction/wave-optics sign-off, flare/ghost/coating, 제조 오차 |
| Image Sensor | `proxy` | CFA/pixel/exposure/noise/RAW, image-sensor selector DB | per-sensor calibrated process deck, measured n,k, CAD/GDS, sensor-specific LUT |
| FDTD optical LUT | `proxy` | QE/field/crosstalk proxy LUT, physics sanity checks | full convergence, localized crosstalk, measured optical stack |
| TCAD / DEVSIM | `calibration_required` | generation-map ingestion, split-PD current proxy, accuracy gate | active FDTD/TCAD lineage closure, carrier calibration, dark/noise/lag/full-well |
| HW ISP | `proxy` | rolling shutter, stage latency, queue, DMA, delayed AE/AWB | board/vendor trace calibration, AF/HDR/TNR detail |
| Metrics | `validated` | MTF, ISO12233, Delta E, SCIELAB, VSNR, SQRI | product-specific weighting and pass/fail gates |
| Optimization | `validated` | dot-path camera parameter grid search, preset parameter-space catalog, FACA objective scoring, hard constraints, Pareto front, selected scenarios, parameter-lineage evidence | Bayesian/evolutionary search, hardware-in-loop calibration |
| Perception | `available` | task adapters, detection/segmentation/classification/pose/tracking metrics, robustness sweep | training loop, dataset-specific model calibration |
| RAW data factory | `validated` | manifest, metadata JSONL, deterministic RAW NPZ, split, checksum, labels JSON, validation, ADAS/KITTI YOLO demo export, proxy camera-spec variant re-capture | DNG writer, automatic label synthesis |
| DB/LUT registry | `validated` / `calibration_required` | manifest, readiness tier, provenance, dependency lineage, stale detection, calibration evidence manifest, readiness promotion plan | actual measured evidence attachment and calibrated promotion |
| External pipeline | `calibration_required` | FDTD, TCAD, RayOptics, HW ISP assets discoverable in one registry | refresh orchestration and calibrated end-to-end asset generation |

## Implemented API Surface

DB/LUT registry:

- `camerae2e_db_manifest()`
- `camerae2e_db_validate(strict=False)`
- `camerae2e_db_lineage(name)`
- `camerae2e_physics_pipeline_plan(strict=False)`
- `camerae2e_goal_gate(...)`
- `camerae2e_calibration_evidence_requirements(...)`
- `camerae2e_calibration_evidence_manifest(...)`
- `camerae2e_calibration_evidence_validate(...)`
- `camerae2e_readiness_promotion_plan(...)`

`camerae2e_goal_gate(...)` is the top-level research-platform evidence gate.
It regenerates a machine-readable pass/warn/fail matrix over registry,
physics-pipeline lineage, calibration evidence policy, FACA smoke,
camera-parameter optimization, RAW dataset export, ADAS/KITTI YOLO demo export,
camera-spec variant re-capture, and the strict sign-off claim guard. Non-strict
mode is the normal research gate. Strict mode is expected to fail while proxy or
calibration-required assets remain active.

The calibration evidence APIs define the measured artifacts required before a
registry entry can be promoted to `calibrated`. For example HW ISP promotion
requires board latency traces, hardware counter traces, and 3A telemetry traces;
sensor promotion requires measured optical/electrical evidence and lineage
closure. Without those artifacts the promotion plan remains blocked, which is
the intended behavior.

Validate a calibration evidence bundle with:

```bash
python tools/validate_camerae2e_calibration_evidence.py path/to/manifest.json
```

Default output:

- `reports/camerae2e_goal/calibration_evidence_validation.json`

System FACA:

- `camerae2e_run_scenario(...)`
- `camerae2e_run_sweep(...)`
- `camerae2e_faca_report(...)`

Optimization:

- `camerae2e_optimize_parameters(...)`
- `camerae2e_optimize_camera_parameters(...)`
- `camerae2e_parameter_space_catalog(...)`
- `camerae2e_optimization_config_catalog(...)`
- `camerae2e_parameter_space_validate(...)`
- `camerae2e_optimization_report(...)`
- `camerae2e_pareto_front(...)`

The first optimizer is deterministic grid search over dot-path camera
parameters such as `sensor.integration_time`, `sensor.analog_gain`,
`optics.fnumber`, `ip.demosaic_method`, and HW ISP control-delay parameters.
`camerae2e_parameter_space_catalog(...)` exposes validated preset search spaces
such as `exposure`, `raw_factory`, `isp`, and `hw_isp_control`.
`camerae2e_optimization_config_catalog(...)` lists every registered configure
axis, custom dot-path assignment rule, and supported FACA objective metric path.
`camerae2e_parameter_space_validate(...)` classifies a caller parameter space as
`registered`, `assignable`, `custom_passthrough`, or blocked before any expensive
sweep is launched. This prevents false optimization runs where an axis is
syntactically accepted but does not affect the camera pipeline.
`camerae2e_optimize_camera_parameters(...)` runs those presets directly while
still allowing caller overrides. The optimizer maximizes objective paths from
the FACA report, supports hard metric constraints, reports the feasible Pareto
front, and emits selected scenario configs that can be passed into the RAW data
factory. FACA and dataset records include parameter-lineage entries with
requested/before/after values so a RAW export can be traced back to the actual
camera parameters applied. This is the reproducible baseline for later
Bayesian/evolutionary optimizers.

RAW data factory:

- `camerae2e_dataset_export(...)`
- `camerae2e_dataset_export_adas_kitti_demo(...)`
- `camerae2e_dataset_export_camera_spec_variants(...)`
- `camerae2e_adas_camera_spec(...)`
- `camerae2e_kitti_yolo_labels(...)`
- `camerae2e_dataset_export_from_optimization(...)`
- `camerae2e_dataset_validate(...)`

ADAS/KITTI demo export is intentionally labeled `proxy`: it applies KITTI-style
object-detection geometry and YOLO/KITTI label metadata to the RAW factory, but
does not claim KITTI raw sensor ground truth or measured ADAS camera
calibration. Regenerate the demo with:

```bash
python tools/render_adas_kitti_raw_demo.py
```

For camera-spec transformation, use:

```bash
python tools/render_adas_kitti_raw_demo.py --variants \
  --output-dir outputs/adas-kitti-camera-variants-demo
```

This performs a proxy re-capture of the same KITTI-style RGB scene through
target camera specs such as `wide_fov_adas_demo` and `narrow_fov_adas_demo`.
The default `geometric_transform="pinhole_crop"` applies a focal-ratio
center crop/resize approximation to the source RGB and remaps/clips labels in
the same coordinate transform. Narrow FoV therefore behaves like a zoomed crop;
wide FoV exposes source-outside regions that are explicitly filled and counted
in metadata. It is useful for controlled robustness experiments, but it does not
recover true KITTI spectral radiance, depth, occlusion, lens flare, ISP inverse,
or measured RAW.

## External Pipeline Policy

FDTD and TCAD artifacts must be checked as a connected lineage, not as isolated files. If the active FDTD LUT is from one run and the TCAD generation map is from another, the registry marks the TCAD entry with `stale_dependency`. This does not block research runs by default, but it blocks strict validation.

RayOptics PSF assets are kept at `proxy` tier because they are geometric ray histograms. Diffraction and wavefront analysis remain separate CameraE2E optics paths and should be compared explicitly when a design decision depends on diffraction.

HW ISP seed profiles are system-simulation inputs. They become calibrated only when replaced or fitted with board traces, hardware counters, BSP timing, or equivalent measured evidence.

## Report Outputs

Regenerate the current goal-readiness reports with:

```bash
python tools/render_camerae2e_asset_registry_report.py
```

Outputs:

- `reports/camerae2e_goal/readiness.json`
- `reports/camerae2e_goal/readiness.html`
- `reports/camerae2e_goal/asset_registry.json`
- `reports/camerae2e_goal/asset_registry.html`

Run the goal-level evidence gate with:

```bash
python tools/run_camerae2e_goal_gate.py
```

Outputs:

- `reports/camerae2e_goal/goal_gate.json`
- `reports/camerae2e_goal/goal_gate.html`
- smoke RAW artifacts under `outputs/camerae2e-goal-gate-smoke/`

Validate external physics pipeline lineage with:

```bash
python tools/validate_camerae2e_physics_pipeline.py
```

The validation payload includes a `plan` section with refresh/calibration
actions, and the registry report command also writes:

- `reports/camerae2e_goal/physics_pipeline_plan.json`
