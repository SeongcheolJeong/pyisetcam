from __future__ import annotations

import json

import numpy as np

from pyisetcam import (
    HWIspStage,
    camera_compute,
    camera_create,
    camera_get,
    hwISPConfig,
    hw_isp_config,
    hw_isp_export_json,
    hw_isp_latency_summary,
    hw_isp_simulate_frame,
    hw_isp_simulate_sequence,
    hw_isp_timeline_table,
    ip_get,
    scene_create,
)
from tools.render_hwisp_timeline_report import render as render_hwisp_timeline_report


def _small_scene(asset_store):
    return scene_create("uniform ee", 8, asset_store=asset_store)


def test_default_config_and_alias_are_deterministic() -> None:
    config = hw_isp_config(fps=60.0, seed=7)
    alias_config = hwISPConfig(fps=60.0, seed=7)

    assert config.sensor_timing.fps == 60.0
    assert alias_config == config
    assert config.stages[0].name == "blc"
    assert config.transport.request_queue_depth > 0


def test_single_frame_timestamps_are_monotonic(asset_store) -> None:
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), _small_scene(asset_store), asset_store=asset_store)
    timestamps = result.timeline.timestamps_us

    assert timestamps["request"] <= timestamps["exposure_start"]
    assert timestamps["exposure_start"] < timestamps["exposure_mid"] < timestamps["readout_start"]
    assert timestamps["readout_start"] < timestamps["readout_end"] <= timestamps["stats_ready"]
    assert timestamps["readout_start"] <= timestamps["isp_start"] < timestamps["isp_done"]
    assert timestamps["isp_done"] < timestamps["dma_done"] < timestamps["app_visible"]
    assert "hw_isp" in result.ip.metadata
    assert "hw_isp" in result.camera.metadata


def test_rolling_shutter_row_timing_formula(asset_store) -> None:
    config = hw_isp_config(line_time_us=10.0, hidden_lines_top=3, exposure_time_us=1000.0)
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        asset_store=asset_store,
    )
    timeline = result.timeline

    assert timeline.row_start_us(0) == timeline.timestamps_us["readout_start"] + 30.0
    assert timeline.row_start_us(5) == timeline.timestamps_us["readout_start"] + 80.0


def test_line_buffer_stage_starts_before_frame_readout_end(asset_store) -> None:
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), _small_scene(asset_store), asset_store=asset_store)
    readout_end = result.timeline.timestamps_us["readout_end"]
    line_stages = [stage for stage in result.timeline.stages if stage.buffering in {"stream", "line"}]

    assert line_stages
    assert all(stage.start_us < readout_end for stage in line_stages)


def test_frame_buffer_stage_waits_for_previous_stage(asset_store) -> None:
    config = hw_isp_config(
        stages=[
            HWIspStage("stream_a", "bayer", "stream", window_lines=1, stage_latency_cycles=10),
            HWIspStage("frame_b", "rgb", "frame", stage_latency_cycles=10),
        ]
    )
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        asset_store=asset_store,
    )
    stream_stage, frame_stage = result.timeline.stages

    assert frame_stage.start_us >= stream_stage.end_us


def test_latency_factors_change_modeled_timing(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    scene = _small_scene(asset_store)
    nominal = hw_isp_simulate_frame(camera, scene, hw_isp_config(), asset_store=asset_store)
    slow = hw_isp_simulate_frame(camera, scene, hw_isp_config(global_latency_factor=2.0), asset_store=asset_store)

    nominal_isp = nominal.timeline.timestamps_us["isp_done"] - nominal.timeline.timestamps_us["isp_start"]
    slow_isp = slow.timeline.timestamps_us["isp_done"] - slow.timeline.timestamps_us["isp_start"]
    assert slow_isp > nominal_isp
    assert slow.timeline.timestamps_us["readout_end"] == nominal.timeline.timestamps_us["readout_end"]


def test_stage_latency_factor_affects_that_stage_and_downstream(asset_store) -> None:
    base_stages = [
        HWIspStage("first", "bayer", "stream", stage_latency_cycles=10),
        HWIspStage("second", "rgb", "stream", stage_latency_cycles=10),
    ]
    slow_stages = [
        HWIspStage("first", "bayer", "stream", stage_latency_cycles=10),
        HWIspStage("second", "rgb", "stream", stage_latency_cycles=10, latency_factor=3.0),
    ]
    baseline = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(stages=base_stages),
        asset_store=asset_store,
    )
    custom = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(stages=slow_stages),
        asset_store=asset_store,
    )

    assert custom.timeline.stages[1].cycle_latency_us > custom.timeline.stages[0].cycle_latency_us
    assert custom.timeline.timestamps_us["app_visible"] > baseline.timeline.timestamps_us["app_visible"]


def test_control_delay_maps_stats_frames(asset_store) -> None:
    config = hw_isp_config(control_path={"ae_apply_delay_frames": 2, "awb_apply_delay_frames": 2})
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        nframes=4,
        asset_store=asset_store,
    )

    assert sequence.frames[0].controls_applied["warmup"] is True
    assert sequence.frames[1].controls_applied["warmup"] is True
    assert sequence.frames[2].controls_applied["ae_stats_frame"] == 0
    assert sequence.frames[3].controls_applied["awb_stats_frame"] == 1


def test_queue_depth_creates_stall(asset_store) -> None:
    config = hw_isp_config(
        fps=1000.0,
        transport={"request_queue_depth": 1, "max_buffers": 1, "app_processing_us": 5000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        nframes=3,
        asset_store=asset_store,
    )

    assert sequence.frames[1].timeline.queue_stall_us > 0.0
    assert sequence.aggregate["queue_stall_total_us"] > 0.0


def test_export_json_and_timeline_table(asset_store, tmp_path) -> None:
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(),
        nframes=2,
        asset_store=asset_store,
    )
    table = hw_isp_timeline_table(sequence)
    output = hw_isp_export_json(sequence, tmp_path / "hwisp.json")
    payload = json.loads(output.read_text())

    assert output.exists()
    assert payload["type"] == "sequence"
    assert len(payload["frames"]) == 2
    assert any(row["type"] == "stage" for row in table)
    assert hw_isp_latency_summary(sequence)["frame_count"] == 2.0


def test_hw_isp_frame_preserves_camera_compute_image(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    scene = _small_scene(asset_store)
    normal = camera_compute(camera.clone(), scene, asset_store=asset_store)
    simulated = hw_isp_simulate_frame(camera, scene, hw_isp_config(), asset_store=asset_store)

    np.testing.assert_allclose(
        ip_get(camera_get(normal, "ip"), "result"),
        ip_get(simulated.ip, "result"),
    )


def test_hw_isp_report_renderer_writes_html(asset_store, tmp_path) -> None:
    outputs = render_hwisp_timeline_report(tmp_path / "hwisp", nframes=3)

    assert outputs["html"].exists()
    assert outputs["details_html"].exists()
    assert outputs["summary"].exists()
    assert outputs["frame_timeline"].exists()
    html = outputs["html"].read_text(encoding="utf-8")
    details = outputs["details_html"].read_text(encoding="utf-8")
    assert "HW ISP Simulation Report" in html
    assert "frame_timeline.png" in html
    assert "HW ISP Frame Details" in details
    assert "Stage Timing" in details
