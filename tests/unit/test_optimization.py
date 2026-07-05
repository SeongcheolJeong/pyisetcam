from __future__ import annotations

from pyisetcam import (
    camerae2e_faca_report,
    camerae2e_optimization_report,
    camerae2e_optimize_parameters,
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
    assert report["selected_scenarios"][0]["sensor"]["integration_time"] == 0.004
    assert len(report["top_cases"]) == 2


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
