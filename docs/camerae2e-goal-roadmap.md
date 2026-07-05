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
| Optimization | `validated` | dot-path camera parameter grid search, FACA objective scoring, hard constraints, Pareto front, selected scenarios | Bayesian/evolutionary search, hardware-in-loop calibration |
| Perception | `available` | task adapters, detection/segmentation/classification/pose/tracking metrics, robustness sweep | training loop, dataset-specific model calibration |
| RAW data factory | `validated` | manifest, metadata JSONL, deterministic RAW NPZ, split, checksum, labels JSON, validation | DNG writer, automatic label synthesis |
| DB/LUT registry | `validated` / `calibration_required` | manifest, readiness tier, provenance, dependency lineage, stale detection | measured evidence ingestion and calibrated promotion |
| External pipeline | `calibration_required` | FDTD, TCAD, RayOptics, HW ISP assets discoverable in one registry | refresh orchestration and calibrated end-to-end asset generation |

## Implemented API Surface

DB/LUT registry:

- `camerae2e_db_manifest()`
- `camerae2e_db_validate(strict=False)`
- `camerae2e_db_lineage(name)`
- `camerae2e_physics_pipeline_plan(strict=False)`

System FACA:

- `camerae2e_run_scenario(...)`
- `camerae2e_run_sweep(...)`
- `camerae2e_faca_report(...)`

Optimization:

- `camerae2e_optimize_parameters(...)`
- `camerae2e_optimization_report(...)`
- `camerae2e_pareto_front(...)`

The first optimizer is deterministic grid search over dot-path camera
parameters such as `sensor.integration_time`, `sensor.analog_gain`,
`optics.fnumber`, and `ip.demosaic_method`. It maximizes objective paths from
the FACA report, supports hard metric constraints, reports the feasible Pareto
front, and emits selected scenario configs that can be passed into the RAW data
factory. This is the reproducible baseline for later Bayesian/evolutionary
optimizers.

RAW data factory:

- `camerae2e_dataset_export(...)`
- `camerae2e_dataset_validate(...)`

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

Validate external physics pipeline lineage with:

```bash
python tools/validate_camerae2e_physics_pipeline.py
```

The validation payload includes a `plan` section with refresh/calibration
actions, and the registry report command also writes:

- `reports/camerae2e_goal/physics_pipeline_plan.json`
