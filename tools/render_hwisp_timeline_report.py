"""Render a hardware ISP latency timeline report."""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
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


def _ms(value_us: float) -> str:
    return f"{float(value_us) / 1000.0:.3f}"


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #16202a;
      --muted: #5d6b78;
      --line: #d9e0e7;
      --card: #ffffff;
      --bg: #f4f7fa;
      --accent: #215f9a;
      --warn: #9a5a21;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0 0 14px;
    }}
    h1 {{
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin-top: 34px;
      font-size: 22px;
    }}
    a {{
      color: var(--accent);
    }}
    .lead {{
      max-width: 900px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin: 26px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .value {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
      font-weight: 760;
      letter-spacing: -0.03em;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    figure {{
      margin: 0;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    figcaption {{
      padding: 12px 14px 14px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #eaf0f6;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 0;
    }}
    .links a {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      text-decoration: none;
    }}
    .note {{
      border-left: 4px solid var(--warn);
      background: #fff7ea;
      padding: 12px 14px;
      margin: 18px 0;
    }}
    .scroll {{
      overflow-x: auto;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _figure_html(path: Path, title: str, caption: str) -> str:
    return (
        "<figure>"
        f'<img src="{escape(path.name)}" alt="{escape(title)}">'
        f"<figcaption><strong>{escape(title)}</strong><br>{escape(caption)}</figcaption>"
        "</figure>"
    )


def _summary_cards(summary: dict[str, float]) -> str:
    cards = [
        ("Frames", f"{int(summary['frame_count'])}"),
        ("Mean E2E", f"{_ms(summary['e2e_latency_mean_us'])} ms"),
        ("Max E2E", f"{_ms(summary['e2e_latency_max_us'])} ms"),
        ("Total Queue Stall", f"{_ms(summary['queue_stall_total_us'])} ms"),
        ("Max Queue Stall", f"{_ms(summary['queue_stall_max_us'])} ms"),
    ]
    return "\n".join(
        f'<div class="card"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span></div>'
        for label, value in cards
    )


def _frame_table_html(sequence) -> str:
    rows = []
    for frame in sequence.frames:
        ts = frame.timeline.timestamps_us
        controls = frame.controls_applied
        rows.append(
            "<tr>"
            f"<td>{frame.timeline.frame_id}</td>"
            f"<td>{_ms(ts['requested'])}</td>"
            f"<td>{_ms(ts['request'])}</td>"
            f"<td>{_ms(ts['readout_end'])}</td>"
            f"<td>{_ms(ts['isp_done'])}</td>"
            f"<td>{_ms(ts['app_visible'])}</td>"
            f"<td>{_ms(ts['app_visible'] - ts['request'])}</td>"
            f"<td>{_ms(frame.timeline.queue_stall_us)}</td>"
            f"<td>{escape(str(controls['warmup']))}</td>"
            f"<td>{escape(str(controls['ae_stats_frame']))}</td>"
            f"<td>{escape(str(controls['awb_stats_frame']))}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Frame</th><th>Requested ms</th><th>Actual request ms</th>"
        "<th>Readout end ms</th><th>ISP done ms</th><th>App visible ms</th>"
        "<th>E2E ms</th><th>Queue stall ms</th><th>Warmup</th>"
        "<th>AE stats frame</th><th>AWB stats frame</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def _stage_table_html(sequence) -> str:
    rows = []
    for frame in sequence.frames:
        for stage in frame.timeline.stages:
            rows.append(
                "<tr>"
                f"<td>{frame.timeline.frame_id}</td>"
                f"<td>{escape(stage.name)}</td>"
                f"<td>{escape(stage.domain)}</td>"
                f"<td>{escape(stage.buffering)}</td>"
                f"<td>{_ms(stage.start_us)}</td>"
                f"<td>{_ms(stage.end_us)}</td>"
                f"<td>{_ms(stage.latency_us)}</td>"
                f"<td>{_ms(stage.line_buffer_delay_us)}</td>"
                f"<td>{_ms(stage.cycle_latency_us)}</td>"
                "</tr>"
            )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Frame</th><th>Stage</th><th>Domain</th><th>Buffering</th>"
        "<th>Start ms</th><th>End ms</th><th>Latency ms</th>"
        "<th>Line-buffer delay ms</th><th>Cycle latency ms</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def _write_html_reports(
    *,
    sequence,
    summary: dict[str, float],
    report_html: Path,
    details_html: Path,
    frame_timeline: Path,
    stage_latency: Path,
    e2e_latency: Path,
    report_md: Path,
    summary_json: Path,
) -> None:
    dashboard_body = f"""
<h1>HW ISP Simulation Report</h1>
<p class="lead">This report visualizes deterministic timing metadata layered on top of the normal pyisetcam image pipeline. Pixel values still come from the existing camera pipeline; this report focuses on exposure, readout, ISP stage, queue, DMA, and app-visible timing.</p>
<div class="cards">
{_summary_cards(summary)}
</div>
<div class="note">Queue stall is reported separately from per-frame E2E latency. E2E latency is measured from the actual request time after stall to app visibility.</div>
<div class="links">
  <a href="{escape(details_html.name)}">Frame and stage details</a>
  <a href="{escape(summary_json.name)}">Machine-readable JSON</a>
  <a href="{escape(report_md.name)}">Markdown report</a>
</div>
<h2>Timing Figures</h2>
<div class="grid">
  {_figure_html(frame_timeline, "Frame timeline", "Per-frame ISP stage spans in wall-clock time.")}
  {_figure_html(stage_latency, "Stage latency", "Mean modeled latency by HW ISP block.")}
  {_figure_html(e2e_latency, "E2E latency and queue stall", "Application-visible latency and queue back-pressure per frame.")}
</div>
<h2>Frame Summary</h2>
{_frame_table_html(sequence)}
"""
    details_body = f"""
<h1>HW ISP Frame Details</h1>
<p class="lead">Detailed timestamps for every simulated frame and every enabled ISP stage. All times are milliseconds from the start of the simulation.</p>
<div class="links">
  <a href="{escape(report_html.name)}">Back to dashboard</a>
  <a href="{escape(summary_json.name)}">Machine-readable JSON</a>
</div>
<h2>Frame Timing</h2>
{_frame_table_html(sequence)}
<h2>Stage Timing</h2>
{_stage_table_html(sequence)}
"""
    report_html.write_text(_html_page("HW ISP Simulation Report", dashboard_body), encoding="utf-8")
    details_html.write_text(_html_page("HW ISP Frame Details", details_body), encoding="utf-8")


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
    report_html = output_dir / "index.html"
    details_html = output_dir / "frame_details.html"

    _render_frame_timeline(rows, frame_timeline)
    _render_stage_latency(rows, stage_latency)
    _render_e2e_latency(rows, e2e_latency)
    hw_isp_export_json(sequence, summary_json)

    summary = hw_isp_latency_summary(sequence)
    _write_html_reports(
        sequence=sequence,
        summary=summary,
        report_html=report_html,
        details_html=details_html,
        frame_timeline=frame_timeline,
        stage_latency=stage_latency,
        e2e_latency=e2e_latency,
        report_md=report_md,
        summary_json=summary_json,
    )
    report = [
        "# HW ISP Timeline Report",
        "",
        "## Summary",
        f"- Frames: `{int(summary['frame_count'])}`",
        f"- Mean E2E latency: `{summary['e2e_latency_mean_us'] / 1000.0:.3f} ms`",
        f"- Max E2E latency: `{summary['e2e_latency_max_us'] / 1000.0:.3f} ms`",
        f"- Total queue stall: `{summary['queue_stall_total_us'] / 1000.0:.3f} ms`",
        f"- HTML dashboard: [{report_html}]({report_html})",
        f"- HTML details: [{details_html}]({details_html})",
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
        "html": report_html,
        "details_html": details_html,
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
