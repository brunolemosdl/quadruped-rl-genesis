"""Build monitored and optionally normalized Genesis vector environments."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import VecMonitor

from quadruped_rl_genesis.environments.normalization import wrap_with_vecnormalize
from quadruped_rl_genesis.environments.vector import GenesisEnv
from quadruped_rl_genesis.services.logger import get_logger

LOGGER = get_logger(__name__)


def _monitor_keys(experiment_config: dict) -> tuple[str, ...]:
    """Return the scalar info keys preserved by the ``VecMonitor`` wrapper.

    Args:
        experiment_config (dict): Resolved experiment configuration.

    Returns:
        tuple[str, ...]: Info keys that should be copied into the episode
            summaries recorded by Stable-Baselines3.
    """
    rewards_cfg = experiment_config["environment"]["rewards"]
    dense_scale_keys = rewards_cfg["dense_scales"]
    terminal_scale_keys = rewards_cfg["terminal_scales"]
    reward_keys = sorted(
        {
            *(f"rew_{name}" for name in dense_scale_keys),
            *(f"rew_{name}" for name in terminal_scale_keys),
        }
    )
    metric_keys = (
        "metric_success",
        "metric_time_to_goal_s",
        "metric_final_goal_distance",
        "metric_final_goal_yaw_error_deg",
        "metric_planar_speed",
        "metric_mean_cross_track_error",
        "metric_mean_arc_progress_speed",
        "metric_mean_curve_progress_gate",
        "metric_mean_local_slope_deg",
        "metric_mean_forward_slope_deg",
        "metric_arc_progress_fraction",
        "metric_reverse_motion_ratio",
        "metric_lateral_motion_ratio",
        "metric_airborne_ratio",
        "metric_trot_contact_score",
        "metric_mean_stop_speed_at_goal",
        "metric_final_planar_speed",
        "metric_final_yaw_rate_deg_s",
        "metric_turn_in_place_fraction",
        "metric_impact",
        "metric_foot_airtime",
        "metric_curve_deviation",
        "metric_stagnation",
        "metric_timeout",
        "metric_terrain_stage",
        "metric_terrain_step_height_m",
        "metric_terrain_terrace_width_m",
        "metric_terrain_edge_smoothing",
        "metric_terrain_global_height_scale",
        "metric_terrain_local_irregularity_m",
        "metric_terrain_roughness_residual_m",
        "metric_final_local_slope_deg",
        "metric_final_forward_slope_deg",
    )
    return ("termination_reason", *reward_keys, *metric_keys)


def build_vector_env(
    experiment_config: dict,
    num_envs: int,
    show_viewer: bool = False,
    add_camera: bool = False,
    monitor: bool = True,
    fast_viz: bool = False,
    viewer_help_text: bool = True,
    disable_reward_curriculum: bool = False,
    disable_terrain_curriculum: bool | None = None,
    normalize: bool | None = None,
    vecnormalize_path: str | Path | None = None,
    for_training: bool = True,
    norm_reward: bool | None = None,
):
    """Create the vectorized Genesis environment used by project workflows.

    Args:
        experiment_config (dict): Resolved experiment configuration.
        num_envs (int): Number of parallel environments.
        show_viewer (bool, optional): Whether to open the Genesis viewer.
        add_camera (bool, optional): Whether to attach a render camera.
        monitor (bool, optional): Whether to wrap the environment with
            ``VecMonitor``.
        fast_viz (bool, optional): Whether to use lighter visualization
            settings.
        viewer_help_text (bool, optional): Whether to show default viewer
            keyboard instructions. Set False when using a custom overlay.
        disable_reward_curriculum (bool, optional): If True, all reward terms
            use full configured scales (no curriculum stage zeroing). Use for
            ``visualize`` and standalone ``evaluate`` where ``global_step`` is
            not the training counter.
        disable_terrain_curriculum (bool | None, optional): If True, terrain
            resets use the configured evaluation/final stage instead of the
            training curriculum. Defaults to ``not for_training`` when omitted.
        normalize (bool | None, optional): Override for experiment-level
            normalization enablement.
        vecnormalize_path (str | Path | None, optional): Existing stats path to
            load when wrapping with VecNormalize.
        for_training (bool, optional): Whether the env will be used for
            training updates. Forwarded to VecNormalize.
        norm_reward (bool | None, optional): Optional override for reward
            normalization.

    Returns:
        Any: ``GenesisEnv`` instance, optionally wrapped in ``VecMonitor``.
    """
    camera_config = (
        experiment_config.get("environment", {}).get("sensors", {}).get("camera", {})
    )
    if disable_terrain_curriculum is None:
        disable_terrain_curriculum = not bool(for_training)

    if add_camera and not bool(camera_config.get("enabled", True)):
        LOGGER.warning(
            "Video recording requested while environment.sensors.camera.enabled=false. "
            "A visualization camera will still be created for capture."
        )

    env = GenesisEnv(
        experiment_config=experiment_config,
        num_envs=num_envs,
        show_viewer=show_viewer,
        add_camera=add_camera,
        fast_viz=fast_viz,
        viewer_help_text=viewer_help_text,
        disable_reward_curriculum=disable_reward_curriculum,
        disable_terrain_curriculum=bool(disable_terrain_curriculum),
    )

    if monitor:
        env = VecMonitor(env, info_keywords=_monitor_keys(experiment_config))

    if normalize is None or bool(normalize):
        env = wrap_with_vecnormalize(
            env,
            experiment_config=experiment_config,
            for_training=for_training,
            vecnormalize_path=vecnormalize_path,
            norm_reward=norm_reward,
        )

    return env
