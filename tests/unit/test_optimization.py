from __future__ import annotations

from pyisetcam import (
    camerae2e_faca_report,
    camerae2e_optimization_config_catalog,
    camerae2e_optimization_report,
    camerae2e_optimize_camera_parameters,
    camerae2e_optimize_parameters,
    camerae2e_parameter_space_catalog,
    camerae2e_parameter_space_validate,
    camerae2e_pareto_front,
    camerae2e_run_scenario,
)


def test_camerae2e_scenario_applies_generic_camera_parameters() -> None:
    low = camerae2e_faca_report(
        camerae2e_run_scenario(
            {
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
                "parameters": {"sensor.integration_time": 0.001},
            },
            include_arrays=False,
        )
    )
    high = camerae2e_faca_report(
        camerae2e_run_scenario(
            {
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
                "parameters": {"sensor.integration_time": 0.004},
            },
            include_arrays=False,
        )
    )

    assert high["metrics"]["color"]["rgb_mean"] > low["metrics"]["color"]["rgb_mean"]
    assert high["parameter_lineage"][0]["path"] == "sensor.integration_time"
    assert high["parameter_lineage"][0]["status"] == "applied"


def test_camerae2e_parameter_space_catalog_exposes_presets() -> None:
    catalog = camerae2e_parameter_space_catalog()
    raw_factory = camerae2e_parameter_space_catalog("raw_factory")

    assert catalog["schema_version"] == "camerae2e_parameter_space_catalog_v1"
    assert "sensor.integration_time" in catalog["axes"]
    assert "sensor_geometry" in catalog["presets"]
    assert "raw_factory" in catalog["presets"]
    assert set(raw_factory["parameter_space"]) == {
        "sensor.integration_time",
        "sensor.analog_gain",
        "optics.fnumber",
    }


def test_camerae2e_optimization_config_catalog_lists_configure_targets() -> None:
    catalog = camerae2e_optimization_config_catalog()

    assert catalog["schema_version"] == "camerae2e_optimization_config_catalog_v1"
    assert catalog["registered_axis_count"] >= 20
    assert "sensor.integration_time" in catalog["registered_axes"]
    assert "sensor.pixel_size" in catalog["registered_axes"]
    assert "sensor.cfa_pattern" in catalog["registered_axes"]
    assert "optics.si_psf_radius_um" in catalog["registered_axes"]
    assert "fdtd.crosstalk_strength" in catalog["registered_axes"]
    assert "hw_isp.global_latency_factor" in catalog["registered_axes"]
    assert "metrics.artifact.raw_std" in catalog["objective_metrics"]
    assert any(
        rule["path_pattern"] == "<camera_set dot path>"
        for rule in catalog["custom_path_rules"]
    )
    sensor_rule = next(
        rule for rule in catalog["custom_path_rules"] if rule["path_pattern"] == "sensor.<name>"
    )
    assert "exposure_time" in sensor_rule["allowed_suffixes"]
    assert "pixel_size" in sensor_rule["allowed_suffixes"]
    assert "binning_method" in sensor_rule["allowed_suffixes"]


def test_camerae2e_parameter_space_validate_classifies_axes() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.integration_time": [0.001, 0.004],
            "optics.focal_length": [0.002, 0.004],
            "hw_isp.ae_apply_delay_frames": [0, 2],
            "optics.transmittance_scale": [[1, 1, 1]],
        }
    )

    assert validation["ok"] is True
    assert validation["axes"]["sensor.integration_time"]["status"] == "registered"
    assert validation["axes"]["hw_isp.ae_apply_delay_frames"]["status"] == "registered"
    assert validation["axes"]["optics.focal_length"]["status"] == "registered"
    assert validation["axes"]["optics.transmittance_scale"]["status"] == "custom_passthrough"
    assert validation["warning_count"] == 1


def test_camerae2e_parameter_space_validate_reports_ineffective_axes() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.not_a_real_parameter": [1],
            "fdtd.mode": ["qe"],
            "tcad.collection_mode": ["collection"],
        }
    )
    fdtd_ready = camerae2e_parameter_space_validate(
        {"fdtd.mode": ["qe"]},
        base_scenario={"fdtd": {"lut": "unit_lut.json"}},
    )

    assert validation["ok"] is False
    assert {issue["kind"] for issue in validation["issues"]} == {
        "unsupported_sensor_axis",
        "inactive_fdtd_axis",
        "inactive_tcad_axis",
    }
    assert fdtd_ready["ok"] is True


def test_camerae2e_parameter_space_validate_reports_invalid_values() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.pixel_size": [0.0],
            "sensor.pixel_fill_factor": [1.5],
            "sensor.n_samples_per_pixel": [2],
            "sensor.cfa_pattern": [[[1.2, 2], [2, 3]]],
            "sensor.binning_method": ["unsupported"],
            "optics.si_psf_radius_um": [-1.0],
        }
    )

    assert validation["ok"] is False
    issue_kinds = {issue["kind"] for issue in validation["issues"]}
    assert {
        "invalid_pixel_size",
        "invalid_fill_factor",
        "unsupported_samples_per_pixel",
        "invalid_cfa_pattern",
        "unsupported_enum_value",
        "invalid_positive_value",
    } <= issue_kinds
    assert validation["axes"]["sensor.n_samples_per_pixel"]["value_issues"][0][
        "kind"
    ] == "unsupported_samples_per_pixel"


def test_camerae2e_parameter_space_validate_reports_value_warnings() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.cfa_pattern": [[[1, 4], [2, 3]]],
            "fdtd.crosstalk_strength": [1.5],
        },
        base_scenario={"fdtd": {"lut": "unit_lut.json"}},
    )

    assert validation["ok"] is True
    assert {warning["kind"] for warning in validation["warnings"]} == {
        "cfa_filter_dependency",
        "extrapolating_crosstalk_strength",
    }


def test_camerae2e_optimize_parameters_rejects_invalid_values_before_run() -> None:
    try:
        camerae2e_optimize_parameters(
            {"scene": {"type": "uniform ee", "args": [8]}},
            {"sensor.n_samples_per_pixel": [2]},
        )
    except ValueError as exc:
        assert "positive odd integer" in str(exc)
    else:
        raise AssertionError("Expected invalid value to fail before optimization.")


def test_camerae2e_scenario_applies_extended_configure_axes() -> None:
    result = camerae2e_run_scenario(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {
                "noise_flag": 0,
                "pixel_size": 3.75e-6,
                "pixel_fill_factor": 0.55,
                "cfa_pattern": [[1, 2], [2, 3]],
                "binning_method": "kodak2008",
            },
            "parameters": {"optics.si_psf_radius_um": 2.0},
        },
        include_arrays=False,
    )
    lineage = result["parameter_lineage"]

    assert any(item["path"] == "sensor.pixel_size" for item in lineage)
    assert any(item["path"] == "sensor.pixel_fill_factor" for item in lineage)
    assert any(item["path"] == "sensor.cfa_pattern" for item in lineage)
    assert any(item["path"] == "sensor.binning_method" for item in lineage)
    assert any(item["path"] == "optics.si_psf_radius_um" for item in lineage)
    assert result["camera"].fields["sensor"].fields["sensor_compute_method"] == {
        "name": "binning",
        "method": "kodak2008",
    }


def test_camerae2e_optimize_parameters_selects_best_grid_case() -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {
            "sensor.integration_time": [0.001, 0.004],
            "optics.fnumber": [2.8, 4.0],
        },
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        seed=100,
        top_k=2,
    )
    report = camerae2e_optimization_report(result)

    assert result["schema_version"] == "camerae2e_parameter_optimization_v1"
    assert result["case_count"] == 4
    assert result["feasible_count"] == 4
    assert result["pareto_case_count"] == 1
    assert report["best_case"]["parameters"]["sensor.integration_time"] == 0.004
    assert result["parameter_space_validation"]["ok"] is True
    assert report["selected_scenarios"][0]["sensor"]["integration_time"] == 0.004
    assert len(report["top_cases"]) == 2


def test_camerae2e_optimize_camera_parameters_uses_automation_preset() -> None:
    result = camerae2e_optimize_camera_parameters(
        {
            "name": "unit_auto_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        seed=50,
        top_k=1,
    )

    assert result["automation"]["preset"] == "exposure"
    assert result["best_case"]["parameters"]["sensor.integration_time"] == 0.004
    assert "sensor.analog_gain" in result["automation"]["axes"]


def test_camerae2e_optimize_parameters_supports_constraints() -> None:
    result = camerae2e_optimize_parameters(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        constraints=[
            {"metric": "metrics.artifact.raw_std", "op": "<=", "value": 0.002},
        ],
        seed=10,
    )

    assert result["feasible_count"] == 1
    assert result["best_case"]["parameters"]["sensor.integration_time"] == 0.001


def test_camerae2e_optimize_parameters_reports_pareto_front_for_tradeoffs() -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_pareto_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        [
            {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
            {"metric": "metrics.artifact.raw_std", "direction": "minimize"},
        ],
        seed=30,
    )
    pareto = camerae2e_pareto_front(result)
    parameter_values = {case["parameters"]["sensor.integration_time"] for case in pareto}

    assert result["pareto_case_count"] == 2
    assert parameter_values == {0.001, 0.004}
    assert all("scenario" in case for case in pareto)
    assert all("objective_utilities" in case for case in pareto)


def test_camerae2e_optimize_parameters_rejects_ineffective_sensor_axis() -> None:
    try:
        camerae2e_optimize_parameters(
            {"scene": {"type": "uniform ee", "args": [8]}},
            {"sensor.not_a_real_parameter": [1, 2]},
        )
    except ValueError as exc:
        assert "Unsupported sensor optimization parameter" in str(exc)
    else:
        raise AssertionError("Expected unsupported sensor axis to fail before optimization.")
