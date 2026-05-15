from __future__ import annotations

import json

from pyisetcam import (
    HWIspConfig,
    camera_create,
    hw_isp_config_from_profile,
    hw_isp_parameter_db,
    hw_isp_profile,
    hw_isp_profile_names,
    hw_isp_simulate_frame,
    scene_create,
)
from tools.collect_hwisp_parameter_db import collect


def test_builtin_hwisp_profiles_are_loadable() -> None:
    names = hw_isp_profile_names()
    profiles = hw_isp_parameter_db()

    assert "generic_1080p_30fps" in names
    assert "rpi_vc4_imx219_public_seed" in names
    assert profiles["generic_1080p_30fps"].config.sensor_timing.fps == 30.0
    assert profiles["rpi_vc4_imx219_public_seed"].calibration["black_level"]["value"] == 4096


def test_hwisp_config_from_profile_accepts_nested_overrides() -> None:
    config = hw_isp_config_from_profile(
        "rpi_vc4_imx219_public_seed",
        sensor_timing={"fps": 60.0, "line_time_us": 7.6},
        control_path={"apply_to_image": True, "target_luma": 0.2},
        transport={"request_queue_depth": 2},
    )

    assert isinstance(config, HWIspConfig)
    assert config.sensor_timing.fps == 60.0
    assert config.sensor_timing.line_time_us == 7.6
    assert config.control_path.apply_to_image is True
    assert config.control_path.target_luma == 0.2
    assert config.transport.request_queue_depth == 2
    assert config.stages


def test_hwisp_profile_config_runs_simulation(asset_store) -> None:
    config = hw_isp_config_from_profile("generic_1080p_30fps")
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        scene_create("uniform ee", 8, asset_store=asset_store),
        config,
        asset_store=asset_store,
    )

    assert result.timeline.timestamps_us["app_visible"] > result.timeline.timestamps_us["request"]
    assert result.controls_applied["produced_stats"]["ae"]["stats_grid"] == [8, 8]


def test_collector_writes_normalized_profile_from_libcamera_json(tmp_path) -> None:
    source = tmp_path / "imx219.json"
    source.write_text(
        json.dumps(
            {
                "version": 2.0,
                "target": "bcm2835",
                "algorithms": [
                    {"rpi.black_level": {"black_level": 4096}},
                    {
                        "rpi.lux": {
                            "reference_shutter_speed": 27685,
                            "reference_gain": 1.0,
                            "reference_lux": 998,
                            "reference_Y": 12744,
                        }
                    },
                    {"rpi.noise": {"reference_constant": 0, "reference_slope": 3.67}},
                    {"rpi.geq": {"offset": 204, "slope": 0.01633}},
                    {"rpi.agc": {"y_target": [0, 0.16, 1000, 0.165]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "profiles"

    written = collect([source], output_dir, overwrite=True)
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["calibration"]["black_level"]["value"] == 4096
    assert payload["config"]["control_path"]["target_luma"] == 0.16

    profile = hw_isp_profile(payload["name"], db_path=output_dir)
    assert profile.calibration["noise"]["reference_slope"] == 3.67
