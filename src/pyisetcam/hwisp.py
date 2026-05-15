"""Hardware ISP latency simulation layer.

The simulator keeps the existing image pipeline as the source of truth for pixel
values and adds deterministic timing metadata for system verification.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_get
from .scene import scene_create
from .sensor import sensor_get, sensor_set
from .types import Camera, ImageProcessor, OpticalImage, Scene, Sensor


@dataclass(frozen=True)
class HWIspSensorTiming:
    fps: float = 30.0
    line_time_us: float = 15.2
    active_lines: int = 1080
    hidden_lines_top: int = 16
    hidden_lines_bottom: int = 20
    exposure_time_us: float = 8000.0
    rolling_shutter: bool = True


@dataclass(frozen=True)
class HWIspStage:
    name: str
    domain: str
    buffering: str
    window_lines: int = 1
    stage_latency_cycles: float = 0.0
    clock_mhz: float = 400.0
    pixels_per_cycle: float = 2.0
    latency_factor: float = 1.0
    enabled: bool = True
    history_frames: int = 0


@dataclass(frozen=True)
class HWIspControlPath:
    ae_enabled: bool = True
    awb_enabled: bool = True
    stats_ready_at: str = "frame_end"
    ae_apply_delay_frames: int = 2
    awb_apply_delay_frames: int = 2


@dataclass(frozen=True)
class HWIspTransport:
    request_queue_depth: int = 4
    max_buffers: int = 6
    dma_submit_us: float = 120.0
    dma_complete_us: float = 320.0
    app_processing_us: float = 500.0
    jitter_us_std: float = 0.0
    drop_policy: str = "stall"


@dataclass(frozen=True)
class HWIspConfig:
    sensor_timing: HWIspSensorTiming = field(default_factory=HWIspSensorTiming)
    stages: tuple[HWIspStage, ...] = field(default_factory=tuple)
    control_path: HWIspControlPath = field(default_factory=HWIspControlPath)
    transport: HWIspTransport = field(default_factory=HWIspTransport)
    global_latency_factor: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class HWIspStageSpan:
    name: str
    domain: str
    buffering: str
    start_us: float
    end_us: float
    latency_us: float
    line_buffer_delay_us: float
    cycle_latency_us: float
    stalled_us: float = 0.0


@dataclass(frozen=True)
class HWIspFrameTimeline:
    frame_id: int
    timestamps_us: dict[str, float]
    stages: tuple[HWIspStageSpan, ...]
    queue_stall_us: float = 0.0
    dropped: bool = False

    def row_start_us(self, row: int) -> float:
        line_time = float(self.timestamps_us["line_time_us"])
        hidden_top = float(self.timestamps_us["hidden_lines_top"])
        readout_start = float(self.timestamps_us["readout_start"])
        return readout_start + (hidden_top + int(row)) * line_time


@dataclass(frozen=True)
class HWIspFrameResult:
    camera: Camera
    oi: OpticalImage
    sensor: Sensor
    ip: ImageProcessor
    timeline: HWIspFrameTimeline
    controls_applied: dict[str, Any]


@dataclass(frozen=True)
class HWIspSequenceResult:
    frames: tuple[HWIspFrameResult, ...]
    aggregate: dict[str, float]


def _default_stages() -> tuple[HWIspStage, ...]:
    return (
        HWIspStage("blc", "bayer", "stream", window_lines=1, stage_latency_cycles=24),
        HWIspStage("dpc", "bayer", "line", window_lines=3, stage_latency_cycles=80),
        HWIspStage("demosaic", "bayer_to_rgb", "line", window_lines=5, stage_latency_cycles=220),
        HWIspStage("ccm_gamma", "rgb", "stream", window_lines=1, stage_latency_cycles=48),
        HWIspStage(
            "tnr",
            "yuv",
            "frame",
            window_lines=1,
            stage_latency_cycles=0,
            enabled=False,
            history_frames=2,
        ),
    )


def _replace_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    payload = asdict(instance)
    payload.update(updates)
    return type(instance)(**payload)


def _normalize_stage(stage: HWIspStage | dict[str, Any]) -> HWIspStage:
    if isinstance(stage, HWIspStage):
        return stage
    return HWIspStage(**dict(stage))


def hw_isp_config(**overrides: Any) -> HWIspConfig:
    """Create an HW ISP timing configuration with optional nested overrides."""

    config = HWIspConfig(stages=_default_stages())
    payload: dict[str, Any] = {
        "sensor_timing": config.sensor_timing,
        "stages": config.stages,
        "control_path": config.control_path,
        "transport": config.transport,
        "global_latency_factor": config.global_latency_factor,
        "seed": config.seed,
    }

    if "sensor_timing" in overrides:
        value = overrides.pop("sensor_timing")
        payload["sensor_timing"] = value if isinstance(value, HWIspSensorTiming) else HWIspSensorTiming(**dict(value))
    if "control_path" in overrides:
        value = overrides.pop("control_path")
        payload["control_path"] = value if isinstance(value, HWIspControlPath) else HWIspControlPath(**dict(value))
    if "transport" in overrides:
        value = overrides.pop("transport")
        payload["transport"] = value if isinstance(value, HWIspTransport) else HWIspTransport(**dict(value))
    if "stages" in overrides:
        payload["stages"] = tuple(_normalize_stage(stage) for stage in overrides.pop("stages"))

    nested_sensor: dict[str, Any] = {}
    nested_control: dict[str, Any] = {}
    nested_transport: dict[str, Any] = {}
    for key in list(overrides):
        if hasattr(HWIspSensorTiming, key):
            nested_sensor[key] = overrides.pop(key)
        elif hasattr(HWIspControlPath, key):
            nested_control[key] = overrides.pop(key)
        elif hasattr(HWIspTransport, key):
            nested_transport[key] = overrides.pop(key)

    if nested_sensor:
        payload["sensor_timing"] = _replace_dataclass(payload["sensor_timing"], nested_sensor)
    if nested_control:
        payload["control_path"] = _replace_dataclass(payload["control_path"], nested_control)
    if nested_transport:
        payload["transport"] = _replace_dataclass(payload["transport"], nested_transport)

    payload.update(overrides)
    return HWIspConfig(**payload)


def _deterministic_jitter(config: HWIspConfig, frame_id: int) -> float:
    std = float(config.transport.jitter_us_std)
    if std <= 0.0:
        return 0.0
    generator = np.random.default_rng(int(config.seed) + int(frame_id))
    return float(generator.normal(0.0, std))


def _frame_visibility_history(previous_state: Any) -> list[float]:
    if previous_state is None:
        return []
    if isinstance(previous_state, HWIspSequenceResult):
        return [float(frame.timeline.timestamps_us["app_visible"]) for frame in previous_state.frames]
    if isinstance(previous_state, HWIspFrameResult):
        return [float(previous_state.timeline.timestamps_us["app_visible"])]
    if isinstance(previous_state, dict):
        values = previous_state.get("app_visible_times_us", [])
        return [float(value) for value in values]
    return []


def _queue_adjusted_request(
    requested_us: float,
    *,
    previous_state: Any,
    config: HWIspConfig,
) -> tuple[float, float, bool]:
    history = _frame_visibility_history(previous_state)
    if not history:
        return requested_us, 0.0, False

    depth = max(1, min(int(config.transport.request_queue_depth), int(config.transport.max_buffers)))
    if len(history) < depth:
        return requested_us, 0.0, False

    earliest_release = float(history[-depth])
    if requested_us >= earliest_release:
        return requested_us, 0.0, False

    if str(config.transport.drop_policy).lower() == "drop_newest":
        return requested_us, 0.0, True
    return earliest_release, earliest_release - requested_us, False


def _infer_sensor_shape(sensor: Sensor, timing: HWIspSensorTiming) -> tuple[int, int]:
    try:
        size = np.asarray(sensor_get(sensor, "size"), dtype=int).reshape(-1)
    except Exception:
        size = np.asarray([], dtype=int)
    rows = int(size[0]) if size.size >= 1 and int(size[0]) > 0 else int(timing.active_lines)
    cols = int(size[1]) if size.size >= 2 and int(size[1]) > 0 else rows
    return rows, cols


def _stage_cycle_latency_us(stage: HWIspStage, config: HWIspConfig) -> float:
    clock = max(float(stage.clock_mhz), 1.0e-12)
    return (
        float(stage.stage_latency_cycles)
        / clock
        * float(stage.latency_factor)
        * float(config.global_latency_factor)
    )


def _build_timeline(
    *,
    frame_id: int,
    sensor: Sensor,
    config: HWIspConfig,
    previous_state: Any,
) -> HWIspFrameTimeline:
    timing = config.sensor_timing
    frame_interval_us = 1.0e6 / max(float(timing.fps), 1.0e-12)
    requested_us = frame_id * frame_interval_us + _deterministic_jitter(config, frame_id)
    t_request, queue_stall_us, dropped = _queue_adjusted_request(
        requested_us,
        previous_state=previous_state,
        config=config,
    )

    exposure_start = t_request
    exposure_mid = exposure_start + float(timing.exposure_time_us) / 2.0
    readout_start = exposure_start + float(timing.exposure_time_us)
    active_lines, active_cols = _infer_sensor_shape(sensor, timing)
    readout_end = readout_start + (
        int(timing.hidden_lines_top) + active_lines + int(timing.hidden_lines_bottom)
    ) * float(timing.line_time_us)
    stats_ready = readout_end

    prev_start = readout_start + int(timing.hidden_lines_top) * float(timing.line_time_us)
    prev_end = readout_end - int(timing.hidden_lines_bottom) * float(timing.line_time_us)
    spans: list[HWIspStageSpan] = []
    for stage in config.stages:
        if not stage.enabled:
            continue
        buffering = str(stage.buffering).lower()
        cycle_latency = _stage_cycle_latency_us(stage, config)
        if buffering == "frame":
            pixel_count = max(active_lines * active_cols, 1)
            throughput_us = pixel_count / (
                max(float(stage.clock_mhz), 1.0e-12) * max(float(stage.pixels_per_cycle), 1.0e-12)
            )
            history_us = max(int(stage.history_frames), 0) * frame_interval_us
            start = prev_end + cycle_latency
            end = start + throughput_us * float(stage.latency_factor) * float(config.global_latency_factor) + history_us
            line_delay = 0.0
        else:
            line_delay = (
                max(int(stage.window_lines) - 1, 0)
                * float(timing.line_time_us)
                * float(stage.latency_factor)
                * float(config.global_latency_factor)
            )
            start = prev_start + line_delay + cycle_latency
            end = prev_end + line_delay + cycle_latency

        spans.append(
            HWIspStageSpan(
                name=str(stage.name),
                domain=str(stage.domain),
                buffering=buffering,
                start_us=float(start),
                end_us=float(end),
                latency_us=float(end - start),
                line_buffer_delay_us=float(line_delay),
                cycle_latency_us=float(cycle_latency),
            )
        )
        prev_start = start
        prev_end = end

    isp_start = spans[0].start_us if spans else readout_end
    isp_done = spans[-1].end_us if spans else readout_end
    transport_scale = float(config.global_latency_factor)
    dma_done = isp_done + (
        float(config.transport.dma_submit_us) + float(config.transport.dma_complete_us)
    ) * transport_scale
    app_visible = dma_done + float(config.transport.app_processing_us) * transport_scale

    timestamps = {
        "request": float(t_request),
        "requested": float(requested_us),
        "exposure_start": float(exposure_start),
        "exposure_mid": float(exposure_mid),
        "readout_start": float(readout_start),
        "readout_end": float(readout_end),
        "stats_ready": float(stats_ready),
        "isp_start": float(isp_start),
        "isp_done": float(isp_done),
        "dma_done": float(dma_done),
        "app_visible": float(app_visible),
        "frame_interval_us": float(frame_interval_us),
        "line_time_us": float(timing.line_time_us),
        "hidden_lines_top": float(timing.hidden_lines_top),
        "hidden_lines_bottom": float(timing.hidden_lines_bottom),
    }
    return HWIspFrameTimeline(
        frame_id=int(frame_id),
        timestamps_us=timestamps,
        stages=tuple(spans),
        queue_stall_us=float(queue_stall_us),
        dropped=bool(dropped),
    )


def _control_payload(frame_id: int, timeline: HWIspFrameTimeline, config: HWIspConfig) -> dict[str, Any]:
    control = config.control_path
    ae_source = frame_id - int(control.ae_apply_delay_frames) if control.ae_enabled else None
    awb_source = frame_id - int(control.awb_apply_delay_frames) if control.awb_enabled else None
    ae_applied = ae_source if ae_source is not None and ae_source >= 0 else None
    awb_applied = awb_source if awb_source is not None and awb_source >= 0 else None
    return {
        "ae_enabled": bool(control.ae_enabled),
        "awb_enabled": bool(control.awb_enabled),
        "ae_stats_frame": ae_applied,
        "awb_stats_frame": awb_applied,
        "ae_apply_delay_frames": int(control.ae_apply_delay_frames),
        "awb_apply_delay_frames": int(control.awb_apply_delay_frames),
        "stats_ready_us": float(timeline.timestamps_us["stats_ready"]),
        "warmup": ae_applied is None or awb_applied is None,
    }


def _attach_metadata(
    camera: Camera,
    timeline: HWIspFrameTimeline,
    controls: dict[str, Any],
    config: HWIspConfig,
) -> Camera:
    camera.metadata["hw_isp"] = {
        "timeline": _timeline_to_dict(timeline),
        "controls_applied": copy.deepcopy(controls),
        "config": _config_to_dict(config),
    }
    ip = camera.fields.get("ip")
    if ip is not None:
        ip.metadata["hw_isp"] = copy.deepcopy(camera.metadata["hw_isp"])
    return camera


def hw_isp_simulate_frame(
    camera: Camera,
    scene: Scene | str,
    config: HWIspConfig | None = None,
    frame_id: int = 0,
    previous_state: Any = None,
    *,
    asset_store: AssetStore | None = None,
) -> HWIspFrameResult:
    """Run the normal camera pipeline and attach deterministic HW ISP timing."""

    store = asset_store or AssetStore.default()
    resolved_config = config or hw_isp_config()
    working_scene = scene_create(scene, asset_store=store) if isinstance(scene, str) else scene.clone()
    working_camera = camera_compute(camera.clone(), working_scene, asset_store=store)
    sensor = camera_get(working_camera, "sensor")
    timeline = _build_timeline(
        frame_id=int(frame_id),
        sensor=sensor,
        config=resolved_config,
        previous_state=previous_state,
    )
    controls = _control_payload(int(frame_id), timeline, resolved_config)
    working_camera = _attach_metadata(working_camera, timeline, controls, resolved_config)
    return HWIspFrameResult(
        camera=working_camera,
        oi=camera_get(working_camera, "oi"),
        sensor=sensor,
        ip=camera_get(working_camera, "ip"),
        timeline=timeline,
        controls_applied=controls,
    )


def _broadcast_frames(scenes: Any, nframes: int | None) -> list[Any]:
    if isinstance(scenes, (list, tuple)):
        scene_list = list(scenes)
    else:
        scene_list = [scenes]
    total = int(nframes) if nframes is not None else len(scene_list)
    total = max(total, 1)
    if len(scene_list) == 1 and total > 1:
        scene_list = scene_list * total
    if len(scene_list) != total:
        raise ValueError("scenes must contain one item or exactly nframes items.")
    return scene_list


def hw_isp_simulate_sequence(
    camera: Camera,
    scenes: Any,
    config: HWIspConfig | None = None,
    exposure_times: Any = None,
    nframes: int | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> HWIspSequenceResult:
    """Simulate a sequence and preserve one timeline per computed frame."""

    store = asset_store or AssetStore.default()
    resolved_config = config or hw_isp_config()
    scene_list = _broadcast_frames(scenes, nframes)
    if exposure_times is None:
        exposure_list = [resolved_config.sensor_timing.exposure_time_us] * len(scene_list)
    else:
        values = np.asarray(exposure_times, dtype=float).reshape(-1).tolist()
        exposure_list = values * len(scene_list) if len(values) == 1 else values
    if len(exposure_list) != len(scene_list):
        raise ValueError("exposure_times must contain one value or one value per frame.")

    frames: list[HWIspFrameResult] = []
    visibility_history: list[float] = []
    current_camera = camera.clone()
    for frame_id, (scene_item, exposure_us) in enumerate(zip(scene_list, exposure_list, strict=True)):
        frame_config = HWIspConfig(
            sensor_timing=_replace_dataclass(
                resolved_config.sensor_timing,
                {"exposure_time_us": float(exposure_us)},
            ),
            stages=resolved_config.stages,
            control_path=resolved_config.control_path,
            transport=resolved_config.transport,
            global_latency_factor=resolved_config.global_latency_factor,
            seed=resolved_config.seed,
        )
        current_camera.fields["sensor"] = sensor_set(
            current_camera.fields["sensor"],
            "exposure duration",
            float(exposure_us) * 1.0e-6,
        )
        frame_result = hw_isp_simulate_frame(
            current_camera,
            scene_item,
            frame_config,
            frame_id=frame_id,
            previous_state={"app_visible_times_us": visibility_history},
            asset_store=store,
        )
        frames.append(frame_result)
        visibility_history.append(float(frame_result.timeline.timestamps_us["app_visible"]))
        current_camera = frame_result.camera

    aggregate = hw_isp_latency_summary(HWIspSequenceResult(tuple(frames), aggregate={}))
    return HWIspSequenceResult(frames=tuple(frames), aggregate=aggregate)


def hw_isp_timeline_table(result: HWIspFrameResult | HWIspSequenceResult) -> list[dict[str, float | int | str]]:
    """Return frame and stage timing rows suitable for reports."""

    frames = result.frames if isinstance(result, HWIspSequenceResult) else (result,)
    rows: list[dict[str, float | int | str]] = []
    for frame in frames:
        timestamps = frame.timeline.timestamps_us
        rows.append(
            {
                "frame_id": int(frame.timeline.frame_id),
                "type": "frame",
                "name": "frame",
                "start_us": float(timestamps["request"]),
                "end_us": float(timestamps["app_visible"]),
                "duration_us": float(timestamps["app_visible"] - timestamps["request"]),
                "queue_stall_us": float(frame.timeline.queue_stall_us),
            }
        )
        for stage in frame.timeline.stages:
            rows.append(
                {
                    "frame_id": int(frame.timeline.frame_id),
                    "type": "stage",
                    "name": stage.name,
                    "start_us": float(stage.start_us),
                    "end_us": float(stage.end_us),
                    "duration_us": float(stage.end_us - stage.start_us),
                    "queue_stall_us": 0.0,
                }
            )
    return rows


def hw_isp_latency_summary(result: HWIspFrameResult | HWIspSequenceResult) -> dict[str, float]:
    frames = result.frames if isinstance(result, HWIspSequenceResult) else (result,)
    if not frames:
        return {"frame_count": 0.0}
    e2e = np.asarray(
        [
            frame.timeline.timestamps_us["app_visible"] - frame.timeline.timestamps_us["request"]
            for frame in frames
        ],
        dtype=float,
    )
    queue = np.asarray([frame.timeline.queue_stall_us for frame in frames], dtype=float)
    return {
        "frame_count": float(len(frames)),
        "e2e_latency_mean_us": float(np.mean(e2e)),
        "e2e_latency_max_us": float(np.max(e2e)),
        "e2e_latency_min_us": float(np.min(e2e)),
        "queue_stall_total_us": float(np.sum(queue)),
        "queue_stall_max_us": float(np.max(queue)),
    }


def _timeline_to_dict(timeline: HWIspFrameTimeline) -> dict[str, Any]:
    return {
        "frame_id": int(timeline.frame_id),
        "timestamps_us": dict(timeline.timestamps_us),
        "stages": [asdict(stage) for stage in timeline.stages],
        "queue_stall_us": float(timeline.queue_stall_us),
        "dropped": bool(timeline.dropped),
    }


def _config_to_dict(config: HWIspConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["stages"] = [asdict(stage) for stage in config.stages]
    return payload


def _frame_to_dict(frame: HWIspFrameResult) -> dict[str, Any]:
    return {
        "timeline": _timeline_to_dict(frame.timeline),
        "controls_applied": copy.deepcopy(frame.controls_applied),
        "latency_summary": hw_isp_latency_summary(frame),
    }


def hw_isp_export_json(result: HWIspFrameResult | HWIspSequenceResult, path: str | Path) -> Path:
    """Write a stable JSON timing summary."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, HWIspSequenceResult):
        payload = {
            "type": "sequence",
            "aggregate": dict(result.aggregate),
            "frames": [_frame_to_dict(frame) for frame in result.frames],
        }
    else:
        payload = {"type": "frame", **_frame_to_dict(result)}
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


hwISPConfig = hw_isp_config
hwISPSimulateFrame = hw_isp_simulate_frame
hwISPSimulateSequence = hw_isp_simulate_sequence
hwISPTimelineTable = hw_isp_timeline_table
hwISPLatencySummary = hw_isp_latency_summary
hwISPExportJSON = hw_isp_export_json
