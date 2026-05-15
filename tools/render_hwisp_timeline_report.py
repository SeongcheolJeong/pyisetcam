"""Render a hardware ISP latency timeline report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore, camera_create, scene_create
from pyisetcam.hwisp import (
    hw_isp_config,
    hw_isp_export_json,
    hw_isp_latency_summary,
    hw_isp_simulate_sequence,
    hw_isp_timeline_table,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "hwisp"


def _render_frame_timeline(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    stage_rows = [row for row in rows if row["type"] == "stage"]
    frame_ids = sorted({int(row["frame_id"]) for row in stage_rows})
    stage_names = list(dict.fromkeys(str(row["name"]) for row in stage_rows))
    y_lookup = {(frame_id, stage): index for index, (frame_id, stage) in enumerate((f, s) for f in frame_ids for s in stage_names)}

    fig_height = max(4.0, 0.35 * max(len(y_lookup), 1))
    fig, axis = plt.subplots(figsize=(11.5, fig_height), constrained_layout=True)
    for row in stage_rows:
        frame_id = int(row["frame_id"])
        name = str(row["name"])
        y = y_lookup[(frame_id, name)]
        start = float(row["start_us"]) / 1000.0
        duration = float(row["duration_us"]) / 1000.0
        axis.broken_barh([(start, duration)], (y - 0.38, 0.76), facecolors=f"C{stage_names.index(name) % 10}")
    axis.set_yticks(list(y_lookup.values()))
    axis.set_yticklabels([f"F{frame_id} {stage}" for frame_id in frame_ids for stage in stage_names])
    axis.set_xlabel("Time (ms)")
    axis.set_title("HW ISP Stage Timeline")
    axis.grid(axis="x", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_stage_latency(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    stage_rows = [row for row in rows if row["type"] == "stage"]
    stage_names = list(dict.fromkeys(str(row["name"]) for row in stage_rows))
    means = []
    for name in stage_names:
        durations = [float(row["duration_us"]) / 1000.0 for row in stage_rows if str(row["name"]) == name]
        means.append(float(np.mean(durations)))

    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.bar(stage_names, means, color="#4477aa")
    axis.set_ylabel("Mean latency (ms)")
    axis.set_title("Mean ISP Stage Latency")
    axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_e2e_latency(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    frame_rows = [row for row in rows if row["type"] == "frame"]
    frame_ids = [int(row["frame_id"]) for row in frame_rows]
    e2e_ms = [float(row["duration_us"]) / 1000.0 for row in frame_rows]
    stalls_ms = [float(row["queue_stall_us"]) / 1000.0 for row in frame_rows]

    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.plot(frame_ids, e2e_ms, marker="o", label="E2E latency")
    axis.bar(frame_ids, stalls_ms, alpha=0.35, label="Queue stall")
    axis.set_xlabel("Frame")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Frame Latency")
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render(output_dir: Path = DEFAULT_OUTPUT_DIR, *, nframes: int = 8) -> dict[str, Path]:
    store = AssetStore.default()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = hw_isp_config(
        sensor_timing={"fps": 100.0, "line_time_us": 15.2, "exposure_time_us": 8000.0},
        transport={"request_queue_depth": 1, "max_buffers": 1, "dma_submit_us": 120.0, "dma_complete_us": 320.0, "app_processing_us": 500.0},
        seed=42,
    )
    scene = scene_create("uniform ee", 8, asset_store=store)
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=store),
        scene,
        config,
        nframes=nframes,
        asset_store=store,
    )
    rows = hw_isp_timeline_table(sequence)

    frame_timeline = output_dir / "frame_timeline.png"
    stage_latency = output_dir / "stage_latency.png"
    e2e_latency = output_dir / "e2e_latency.png"
    summary_json = output_dir / "timeline_summary.json"
    report_md = output_dir / "timeline_report.md"

    _render_frame_timeline(rows, frame_timeline)
    _render_stage_latency(rows, stage_latency)
    _render_e2e_latency(rows, e2e_latency)
    hw_isp_export_json(sequence, summary_json)

    summary = hw_isp_latency_summary(sequence)
    report = [
        "# HW ISP Timeline Report",
        "",
        "## Summary",
        f"- Frames: `{int(summary['frame_count'])}`",
        f"- Mean E2E latency: `{summary['e2e_latency_mean_us'] / 1000.0:.3f} ms`",
        f"- Max E2E latency: `{summary['e2e_latency_max_us'] / 1000.0:.3f} ms`",
        f"- Total queue stall: `{summary['queue_stall_total_us'] / 1000.0:.3f} ms`",
        "",
        "## Figures",
        f"![frame timeline]({frame_timeline})",
        f"![stage latency]({stage_latency})",
        f"![e2e latency]({e2e_latency})",
        "",
        "## Regenerate",
        "- `python tools/render_hwisp_timeline_report.py`",
        f"- JSON: [{summary_json}]({summary_json})",
    ]
    report_md.write_text("\n".join(report) + "\n")
    return {
        "report": report_md,
        "summary": summary_json,
        "frame_timeline": frame_timeline,
        "stage_latency": stage_latency,
        "e2e_latency": e2e_latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nframes", type=int, default=8)
    args = parser.parse_args()
    outputs = render(args.output_dir, nframes=args.nframes)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
