# HW ISP Simulation

`pyisetcam` includes a hardware ISP simulation layer for system verification. The normal `Scene -> OpticalImage -> Sensor -> ImageProcessor` computation still produces the image, and the HW ISP layer adds deterministic timing metadata plus optional delayed AE/AWB control feedback on top.

This is useful when image parity is not enough. System verification often needs to know when a frame was exposed, when rolling-shutter readout finished, when ISP stages completed, which AE/AWB controls were active, and when the frame became visible to the app.

## Quick Start

```python
from pyisetcam import camera_create, scene_create
from pyisetcam.hwisp import hw_isp_config, hw_isp_simulate_sequence, hw_isp_latency_summary

scene = scene_create("uniform ee", 8)
camera = camera_create()

config = hw_isp_config(
    fps=30.0,
    line_time_us=15.2,
    exposure_time_us=8000.0,
    transport={"request_queue_depth": 2, "max_buffers": 3},
)

sequence = hw_isp_simulate_sequence(camera, scene, config, nframes=8)
print(hw_isp_latency_summary(sequence))
```

The computed image values are still produced by `camera_compute`, `sensor_compute`, and `ip_compute`. Timing data is attached under:

```python
frame = sequence.frames[0]
frame.camera.metadata["hw_isp"]
frame.ip.metadata["hw_isp"]
```

## Timing Model

The default model records these timestamps in microseconds:

- `request`
- `exposure_start`
- `exposure_mid`
- `readout_start`
- `readout_end`
- `stats_ready`
- `isp_start`
- `isp_done`
- `dma_done`
- `app_visible`

Rolling-shutter row timing follows:

```text
t_row_start(row) = t_readout_start + (hidden_lines_top + row) * line_time_us
```

ISP stages support `stream`, `line`, and `frame` buffering. Streaming and line-buffer stages can start before full-frame readout completes. Frame-buffer stages wait for the previous stage to complete the full frame.

## Latency Factors

Use `global_latency_factor` to scale ISP and transport delays together:

```python
slow = hw_isp_config(global_latency_factor=2.0)
```

Use `HWIspStage.latency_factor` to scale one block:

```python
from pyisetcam.hwisp import HWIspStage, hw_isp_config

config = hw_isp_config(
    stages=[
        HWIspStage("blc", "bayer", "stream", stage_latency_cycles=24),
        HWIspStage("demosaic", "bayer_to_rgb", "line", window_lines=5, stage_latency_cycles=220, latency_factor=2.0),
    ]
)
```

## 3A And Queue Delay

By default, AE/AWB are modeled as delayed control metadata only, so existing image values are unchanged. If `ae_apply_delay_frames=2`, statistics from frame `N` become active on frame `N + 2`. Early frames are marked as warmup.

Enable image-affecting AE/AWB with:

```python
config = hw_isp_config(
    control_path={
        "apply_to_image": True,
        "ae_apply_delay_frames": 2,
        "awb_apply_delay_frames": 2,
        "target_luma": 0.18,
    }
)
```

When enabled, AE applies delayed `exposure_time_us` and `analog_gain` to the sensor. AWB applies delayed gray-world RGB gains through a manual IP illuminant-correction matrix. AF is not part of v1.

The queue model uses `request_queue_depth` and `max_buffers`. When the pipeline is slower than the frame interval, the simulator records `queue_stall_us`. By default it stalls rather than dropping frames.

## Report

Generate the default timeline report:

```bash
python tools/render_hwisp_timeline_report.py
```

Outputs are written under `reports/hwisp/`:

- `index.html`
- `frame_details.html`
- `timeline_report.md`
- `timeline_summary.json`
- `three_a_summary.json`
- `frame_timeline.png`
- `stage_latency.png`
- `e2e_latency.png`
- `ae_convergence.png`
- `awb_convergence.png`
- `three_a_thumbnails.png`

Open `reports/hwisp/index.html` for the browser-facing dashboard. Use `frame_details.html` when you need the full per-frame and per-stage timestamp tables. The dashboard includes a deterministic 12-frame Macbeth-style 3A scenario with a brightness step and a warm-illuminant step.

## Relationship To Parity

The HW ISP simulation is not MATLAB parity. It is a new system timing layer for verification. MATLAB parity remains responsible for image behavior, while this layer explains timing, control delay, and pipeline visibility.
