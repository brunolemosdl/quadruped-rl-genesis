"""Dense reward terms for Bezier-commanded quadruped navigation.

Registered callables in ``REWARD_FUNCTIONS`` take the active task instance and
return a per-environment ``torch.Tensor`` with shape ``[num_envs]`` (or a boolean
mask promoted later) unless the docstring states otherwise.
"""

from __future__ import annotations

import math

import torch

_EPS = 1.0e-6


def _smoothstep_torch(t: torch.Tensor) -> torch.Tensor:
    """Apply a cubic smoothstep after clamping inputs to ``[0, 1]``.

    Args:
        t (torch.Tensor): Raw interpolation parameter.

    Returns:
        torch.Tensor: ``3t^2 - 2t^3`` element-wise with the same shape as ``t``.
    """
    x = t.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _moving_mask(task) -> torch.Tensor:
    """Return a mask of environments moving faster than the configured threshold.

    Args:
        task: Navigation task with ``reward_config`` and body velocities.

    Returns:
        torch.Tensor: Boolean mask ``[num_envs]``.
    """
    threshold = float(task.reward_config.get("moving_threshold_mps", 0.15))
    planar_speed = torch.linalg.vector_norm(task.base_lin_vel_body[:, :2], dim=1)

    return planar_speed > threshold


def _within_approach_zone(task) -> torch.Tensor:
    """Return the planner-defined approach-zone mask.

    Args:
        task: Navigation task with ``within_approach_zone`` buffer.

    Returns:
        torch.Tensor: Boolean mask ``[num_envs]``.
    """
    return task.within_approach_zone


def goal_projected_speed(task) -> torch.Tensor:
    """Return projected body-frame speed along the commanded forward axis.

    Args:
        task: Navigation task with ``cmd_vel`` and ``base_lin_vel_body``.

    Returns:
        torch.Tensor: Scalar speed per env ``[num_envs]``.
    """
    command_dir = task.cmd_vel[:, :2]
    command_norm = torch.linalg.vector_norm(command_dir, dim=1, keepdim=True).clamp(
        min=_EPS
    )
    command_dir = command_dir / command_norm

    return torch.sum(command_dir * task.base_lin_vel_body[:, :2], dim=1)


def turn_in_place_mask(task) -> torch.Tensor:
    """Detect turning in place while commanded forward speed is low.

    Args:
        task: Navigation task with IMU gyro, velocities, and reward thresholds.

    Returns:
        torch.Tensor: Boolean mask ``[num_envs]``.
    """
    planar_speed = torch.linalg.vector_norm(task.base_lin_vel_body[:, :2], dim=1)
    yaw_rate = torch.abs(task.imu_gyro[:, 2])
    cmd_speed = torch.abs(task.cmd_vel[:, 0])
    speed_ok = planar_speed < float(
        task.reward_config.get("turn_in_place_speed_threshold", 0.15)
    )
    cmd_ok = cmd_speed < float(
        task.reward_config.get("turn_in_place_cmd_threshold", 0.1)
    )
    yaw_ok = yaw_rate > float(
        task.reward_config.get(
            "turn_in_place_yaw_rate_rad_s",
            math.radians(20.0),
        )
    )

    return speed_ok & cmd_ok & yaw_ok


def reward_tracking_lin_vel(task) -> torch.Tensor:
    """Gaussian shaping reward for tracking commanded planar body velocity.

    Args:
        task: Navigation task with ``cmd_vel`` and ``base_lin_vel_body``.

    Returns:
        torch.Tensor: Reward in ``(0, 1]`` per environment.
    """
    tracking_sigma = float(task.reward_config.get("tracking_sigma", 0.25))
    error = torch.sum(
        (task.cmd_vel[:, :2] - task.base_lin_vel_body[:, :2]) ** 2,
        dim=1,
    )

    return torch.exp(-error / max(tracking_sigma, _EPS))


def reward_tracking_ang_vel(task) -> torch.Tensor:
    """Gaussian shaping reward for tracking commanded yaw rate.

    Args:
        task: Navigation task with ``cmd_vel`` and IMU gyro Z.

    Returns:
        torch.Tensor: Reward in ``(0, 1]`` per environment.
    """
    tracking_sigma = float(task.reward_config.get("tracking_sigma", 0.25))
    error = (task.cmd_vel[:, 2] - task.imu_gyro[:, 2]) ** 2

    return torch.exp(-error / max(tracking_sigma, _EPS))


def reward_curve_progress(task) -> torch.Tensor:
    """Progress reward along the Bezier arc with corridor and heading gates.

    Args:
        task: Navigation task with arc progress, cross-track error, and gates.

    Returns:
        torch.Tensor: Shaped progress reward ``[num_envs]``; also writes gate buffers.
    """
    progress_speed = torch.clamp(task.arc_progress / max(task.dt, _EPS), -0.5, 1.5)
    corridor_reward_m = float(task.bezier_config.get("corridor_reward_m", 0.75))
    corridor_scale = torch.clamp(
        torch.abs(task.cross_track_error) / max(corridor_reward_m, _EPS),
        min=0.0,
    )
    corridor_gate = 1.0 / (1.0 + corridor_scale**2)

    heading_cos = torch.cos(task.heading_error_to_tangent)
    heading_min_cos = float(
        task.reward_config.get("curve_progress_heading_soft_min_cos", -0.20)
    )
    heading_full_cos = float(
        task.reward_config.get("curve_progress_heading_full_cos", 0.85)
    )
    heading_span = max(heading_full_cos - heading_min_cos, _EPS)
    heading_gate = _smoothstep_torch((heading_cos - heading_min_cos) / heading_span)
    min_gate = float(task.reward_config.get("curve_progress_heading_min_gate", 0.20))
    gate = corridor_gate * (min_gate + (1.0 - min_gate) * heading_gate)

    task.curve_progress_corridor_gate[:] = corridor_gate
    task.curve_progress_heading_gate[:] = heading_gate
    task.curve_progress_gate[:] = gate

    return progress_speed * gate


def reward_cross_track(task) -> torch.Tensor:
    """Return squared cross-track error relative to the Bezier corridor.

    Args:
        task: Navigation task with ``cross_track_error``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return task.cross_track_error**2


def reward_heading_alignment(task) -> torch.Tensor:
    """Cosine-shaped reward for heading alignment with the local curve tangent.

    Args:
        task: Navigation task with ``heading_error_to_tangent``.

    Returns:
        torch.Tensor: Reward in ``[0, 1]`` per environment.
    """
    return 0.5 * (1.0 + torch.cos(task.heading_error_to_tangent))


def reward_reverse_vel(task) -> torch.Tensor:
    """Penalty for backward body-frame longitudinal velocity.

    Args:
        task: Navigation task with ``base_lin_vel_body``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return torch.clamp(-task.base_lin_vel_body[:, 0], min=0.0)


def reward_lateral_vel(task) -> torch.Tensor:
    """Squared lateral body-frame velocity penalty.

    Args:
        task: Navigation task with ``base_lin_vel_body``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return task.base_lin_vel_body[:, 1] ** 2


def reward_upward_vel(task) -> torch.Tensor:
    """Penalty for upward world-frame vertical velocity with slope relief.

    Args:
        task: Navigation task with world velocities and climbability factor.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    penalty = torch.clamp(task.base_lin_vel_world[:, 2], min=0.0)
    relief = float(task.reward_config.get("upward_vel_slope_relief", 0.08))
    return penalty * (1.0 - relief * task.climbable_slope_factor)


def reward_flight_phase(task) -> torch.Tensor:
    """Penalty for flight phases while the robot is commanded to move.

    Args:
        task: Navigation task with foot contact flags.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    contact_count = task.foot_contact.float().sum(dim=1)
    moving = _moving_mask(task).float()
    airborne = (contact_count == 0).float()
    single_contact = (contact_count == 1).float()

    return (airborne + 0.5 * single_contact) * moving


def reward_orientation(task) -> torch.Tensor:
    """Penalty for base tilt using projected gravity with slope relief.

    Args:
        task: Navigation task with ``projected_gravity``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    penalty = torch.sum(task.projected_gravity[:, :2] ** 2, dim=1)
    relief = float(task.reward_config.get("orientation_slope_relief", 0.35))
    return penalty * (1.0 - relief * task.climbable_slope_factor)


def reward_base_height(task) -> torch.Tensor:
    """Penalty for deviation from nominal height above terrain.

    Args:
        task: Navigation task with base pose, terrain height, and nominal height.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    height_above_terrain = task.base_pos[:, 2] - task.terrain_height_at_robot
    target_height = task.nominal_base_height
    tolerance = float(task.reward_config.get("base_height_tolerance", 0.0))
    excess = torch.clamp(
        torch.abs(height_above_terrain - target_height) - tolerance,
        min=0.0,
    )

    penalty = excess**2
    relief = float(task.reward_config.get("base_height_slope_relief", 0.45))
    return penalty * (1.0 - relief * task.climbable_slope_factor)


def reward_lin_vel_z(task) -> torch.Tensor:
    """Squared world-frame vertical linear velocity with slope relief.

    Args:
        task: Navigation task with ``base_lin_vel_world``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    penalty = task.base_lin_vel_world[:, 2] ** 2
    relief = float(task.reward_config.get("lin_vel_z_slope_relief", 0.08))
    return penalty * (1.0 - relief * task.climbable_slope_factor)


def reward_ang_vel_xy(task) -> torch.Tensor:
    """Squared roll/pitch angular velocity penalty.

    Args:
        task: Navigation task with IMU gyro X/Y.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return torch.sum(task.imu_gyro[:, :2] ** 2, dim=1)


def reward_action_rate(task) -> torch.Tensor:
    """Squared L2 penalty on action deltas versus the previous step.

    Args:
        task: Navigation task with ``actions`` and ``last_actions``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return torch.sum((task.actions - task.last_actions) ** 2, dim=1)


def reward_joint_torque(task) -> torch.Tensor:
    """Squared L2 penalty on motor torques.

    Args:
        task: Navigation task with ``joint_torques``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return torch.sum(task.joint_torques**2, dim=1)


def reward_dof_acc(task) -> torch.Tensor:
    """Squared L2 penalty on joint accelerations.

    Args:
        task: Navigation task with ``dof_acc``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    return torch.sum(task.dof_acc**2, dim=1)


def reward_similar_to_default(task) -> torch.Tensor:
    """L1 penalty on joint deviation from the default standing pose.

    Args:
        task: Navigation task with ``dof_pos`` and ``default_dof_pos``.

    Returns:
        torch.Tensor: Non-negative penalty ``[num_envs]``.
    """
    penalty = torch.sum(torch.abs(task.dof_pos - task.default_dof_pos), dim=1)
    relief = float(task.reward_config.get("similar_to_default_slope_relief", 0.12))
    return penalty * (1.0 - relief * task.climbable_slope_factor)


def reward_feet_air_time(task) -> torch.Tensor:
    """Positive shaping term from capped foot airtime while moving.

    Args:
        task: Navigation task with ``foot_airtime_term`` buffer.

    Returns:
        torch.Tensor: Per-environment reward ``[num_envs]``.
    """
    return task.foot_airtime_term


def reward_trot_contact(task) -> torch.Tensor:
    """Heuristic score favoring diagonal foot contact over lateral pairs.

    Args:
        task: Navigation task with ``foot_contact``.

    Returns:
        torch.Tensor: Shaped score ``[num_envs]``, zero when not moving.
    """
    contacts = task.foot_contact.float()
    fl = contacts[:, 0]
    fr = contacts[:, 1]
    rl = contacts[:, 2]
    rr = contacts[:, 3]
    diag_match = 1.0 - 0.5 * (torch.abs(fl - rr) + torch.abs(fr - rl))
    lateral_match = 0.5 * (torch.abs(fl - fr) + torch.abs(rl - rr))

    return (
        torch.clamp(diag_match - lateral_match, -1.0, 1.0) * _moving_mask(task).float()
    )


def reward_goal_heading_track(task) -> torch.Tensor:
    """Gaussian reward on goal yaw error gated by the approach zone.

    Args:
        task: Navigation task with ``goal_yaw_error``.

    Returns:
        torch.Tensor: Reward ``[num_envs]``.
    """
    goal_yaw_sigma = float(task.reward_config.get("goal_yaw_sigma", 0.15))
    base_reward = torch.exp(-(task.goal_yaw_error**2) / max(goal_yaw_sigma, _EPS))

    return base_reward * _within_approach_zone(task).float()


def reward_hold_still(task) -> torch.Tensor:
    """Reward settling to low planar speed and yaw rate near the goal.

    Args:
        task: Navigation task with body velocities and approach-zone mask.

    Returns:
        torch.Tensor: Reward ``[num_envs]``.
    """
    stop_lin_sigma = float(task.reward_config.get("stop_lin_sigma", 0.05))
    stop_yaw_sigma = float(task.reward_config.get("stop_yaw_sigma", 0.08))
    lin_err = torch.sum(task.base_lin_vel_body[:, :2] ** 2, dim=1)
    yaw_err = task.imu_gyro[:, 2] ** 2
    base_reward = torch.exp(
        -lin_err / max(stop_lin_sigma, _EPS) - yaw_err / max(stop_yaw_sigma, _EPS)
    )

    return base_reward * _within_approach_zone(task).float()


REWARD_FUNCTIONS = {
    "tracking_lin_vel": reward_tracking_lin_vel,
    "tracking_ang_vel": reward_tracking_ang_vel,
    "curve_progress": reward_curve_progress,
    "cross_track": reward_cross_track,
    "heading_alignment": reward_heading_alignment,
    "reverse_vel": reward_reverse_vel,
    "lateral_vel": reward_lateral_vel,
    "upward_vel": reward_upward_vel,
    "flight_phase": reward_flight_phase,
    "orientation": reward_orientation,
    "base_height": reward_base_height,
    "lin_vel_z": reward_lin_vel_z,
    "ang_vel_xy": reward_ang_vel_xy,
    "action_rate": reward_action_rate,
    "joint_torque": reward_joint_torque,
    "dof_acc": reward_dof_acc,
    "similar_to_default": reward_similar_to_default,
    "feet_air_time": reward_feet_air_time,
    "trot_contact": reward_trot_contact,
    "goal_heading_track": reward_goal_heading_track,
    "hold_still": reward_hold_still,
}
