"""Evaluation protocol metadata, descriptive stats, and scientific gates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from quadruped_rl_genesis.simulation.terrain import (
    normalize_terrain_mode,
    rough_terrain_config,
)


def _describe(values: list[float]) -> dict[str, float]:
    """Compute a compact descriptive-statistics summary for numeric values.

    Args:
        values (list[float]): Numeric samples to summarize.

    Returns:
        dict[str, float]: Mean, standard deviation, min, max, median, and
            interquartile statistics. Empty inputs yield all-zero values.
    """
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _max_episode_length(experiment_config: dict[str, Any]) -> int:
    """Convert the configured episode duration into simulation steps.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        int: Maximum number of control steps in one episode.
    """
    simulator_dt = float(experiment_config["environment"]["simulator"]["dt"])
    episode_length_s = float(
        experiment_config["environment"]["termination"]["episode_length_s"]
    )

    return max(1, math.ceil(episode_length_s / simulator_dt))


def _selection_rule_description(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Build selection rule description including task_score_mode.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        dict[str, Any]: Selection rule with task_score expression and eligibility.
    """
    eval_cfg = (experiment_config or {}).get("evaluation", {})
    mode = (
        str(eval_cfg.get("task_score_mode", "article"))
        if isinstance(eval_cfg, dict)
        else "article"
    )
    if mode == "exploration":
        task_score_expr = (
            "80*success_rate + 12*mean_arc_progress_speed - 6*mean_cross_track_error "
            "- 8*reverse_motion_ratio - 6*lateral_motion_ratio - 6*airborne_ratio "
            "- 0.12*mean_final_goal_distance - 0.10*mean_final_goal_yaw_error_deg "
            "- 6*fall_rate - 4*curve_deviation_rate - 3*stagnation_rate "
            "- 2*timeout_rate - 4*mean_stop_speed_at_goal"
        )
    else:
        task_score_expr = (
            "120*success_rate + 16*mean_arc_progress_speed - 8*mean_cross_track_error "
            "- 10*reverse_motion_ratio - 8*lateral_motion_ratio - 8*airborne_ratio "
            "- 0.18*mean_final_goal_distance - 0.18*mean_final_goal_yaw_error_deg "
            "- 10*fall_rate - 6*curve_deviation_rate - 4*stagnation_rate "
            "- 4*timeout_rate - 6*mean_stop_speed_at_goal"
        )
    gates_cfg = (
        eval_cfg.get("scientific_gates", {})
        if isinstance(eval_cfg.get("scientific_gates", {}), dict)
        else {}
    )
    return {
        "task_score_mode": mode,
        "task_score": task_score_expr,
        "eligibility": (
            "scientific_gates_passed and "
            "(success_rate > 0 or mean_arc_progress_speed > 0)"
        ),
        "fallback": (
            "highest mean_arc_progress_speed, then lowest mean_cross_track_error"
        ),
        "scientific_gates": gates_cfg,
    }


def _scientific_gates(
    article_metrics: dict[str, Any],
    experiment_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate scientific acceptance gates from experiment configuration.

    Args:
        article_metrics (dict[str, Any]): Aggregated article-style metric scalars.
        experiment_config (dict[str, Any] | None): Full experiment YAML payload;
            reads ``evaluation.scientific_gates``.

    Returns:
        dict[str, Any]: ``enabled``, ``all_passed``, and per-metric ``checks`` when
            gates are configured; otherwise ``enabled: False`` and ``all_passed: True``.
    """
    eval_cfg = (experiment_config or {}).get("evaluation", {})
    cfg = (
        eval_cfg.get("scientific_gates", {})
        if isinstance(eval_cfg.get("scientific_gates", {}), dict)
        else {}
    )
    enabled = bool(cfg.get("enabled", False))
    checks: dict[str, dict[str, Any]] = {}
    if not enabled:
        return {"enabled": False, "all_passed": True, "checks": checks}

    gate_defs = [
        ("success_rate", "min", "success_rate", float, "success_rate_min"),
        (
            "mean_final_goal_distance",
            "max",
            "mean_final_goal_distance",
            float,
            "max_final_distance_m",
        ),
        (
            "mean_final_goal_yaw_error_deg",
            "max",
            "mean_final_goal_yaw_error_deg",
            float,
            "max_final_heading_error_deg",
        ),
        (
            "mean_stop_speed_at_goal",
            "max",
            "mean_stop_speed_at_goal",
            float,
            "max_final_planar_speed_mps",
        ),
        (
            "mean_final_yaw_rate_deg_s",
            "max",
            "mean_final_yaw_rate_deg_s",
            float,
            "max_final_yaw_rate_deg_s",
        ),
        ("fall_rate", "max", "fall_rate", float, "max_fall_rate"),
        ("timeout_rate", "max", "timeout_rate", float, "max_timeout_rate"),
        (
            "curve_deviation_rate",
            "max",
            "curve_deviation_rate",
            float,
            "max_curve_deviation_rate",
        ),
        (
            "stagnation_rate",
            "max",
            "stagnation_rate",
            float,
            "max_stagnation_rate",
        ),
        (
            "reverse_motion_ratio",
            "max",
            "reverse_motion_ratio",
            float,
            "max_reverse_motion_ratio",
        ),
        (
            "lateral_motion_ratio",
            "max",
            "lateral_motion_ratio",
            float,
            "max_lateral_motion_ratio",
        ),
        (
            "airborne_ratio",
            "max",
            "airborne_ratio",
            float,
            "max_airborne_ratio",
        ),
    ]

    for label, mode, metric_key, caster, cfg_key in gate_defs:
        if cfg_key not in cfg:
            continue
        value = caster(article_metrics.get(metric_key, 0.0))
        threshold = caster(cfg[cfg_key])
        passed = value >= threshold if mode == "min" else value <= threshold
        checks[label] = {
            "mode": mode,
            "metric": metric_key,
            "value": value,
            "threshold": threshold,
            "passed": bool(passed),
        }

    all_passed = all(check["passed"] for check in checks.values()) if checks else True
    return {"enabled": True, "all_passed": bool(all_passed), "checks": checks}


def build_evaluation_protocol(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Build a reproducible description of the evaluation protocol.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        dict[str, Any]: Evaluation protocol metadata included in summaries and
            persisted reports.
    """
    environment = experiment_config["environment"]
    rewards = environment["rewards"]
    bezier = environment.get("bezier", {})
    dense_scales = rewards["dense_scales"]
    terminal_scales = rewards["terminal_scales"]
    normalization = experiment_config.get("training", {}).get("normalization", {})

    return {
        "control_mode": environment["control"]["mode"],
        "simulate_action_latency": bool(
            environment["control"].get("simulate_action_latency", False)
        ),
        "terrain_enabled": bool(environment["terrain"]["enabled"]),
        "terrain_mode": normalize_terrain_mode(environment["terrain"].get("mode")),
        "terrain_seed": environment["terrain"].get("seed"),
        "terrain_resolution": list(environment["terrain"]["resolution"]),
        "terrain_height_range": list(environment["terrain"]["height_range"]),
        "terrain_curriculum": dict(environment["terrain"].get("curriculum", {})),
        "terrain_generator": dict(environment["terrain"].get("generator", {})),
        "terrain_rough": dict(rough_terrain_config(environment["terrain"])),
        "goal_distance_range_m": [
            float(environment["goal"]["min_distance_m"]),
            float(environment["goal"]["max_distance_m"]),
        ],
        "reward_scales": dict(dense_scales),
        "terminal_reward_scales": dict(terminal_scales),
        "base_height_target": float(rewards.get("base_height_target", 0.0)),
        "base_height_tolerance": float(rewards.get("base_height_tolerance", 0.0)),
        "bezier": dict(bezier),
        "normalization": dict(normalization),
        "fall_base_height_threshold": float(
            environment["termination"]["fall_base_height_threshold"]
        ),
        "roll_deg_threshold": float(environment["termination"]["roll_deg"]),
        "pitch_deg_threshold": float(environment["termination"]["pitch_deg"]),
        "episode_length_s": float(environment["termination"]["episode_length_s"]),
        "max_episode_length_steps": _max_episode_length(experiment_config),
        "success_position_radius_m": float(bezier.get("success_radius_m", 0.0)),
        "success_heading_tolerance_deg": float(bezier.get("success_yaw_deg", 0.0)),
        "sensor_profile": {
            "imu": bool(environment["sensors"]["imu"]["enabled"]),
            "lidar": bool(environment["sensors"]["lidar"]["enabled"]),
            "feet": bool(environment["sensors"]["feet"]["enabled"]),
            "camera": bool(environment["sensors"]["camera"]["enabled"]),
        },
        "selection_rule": _selection_rule_description(experiment_config),
    }
