"""Episode termination conditions and terminal rewards."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
import torch


@dataclass
class Termination:
    """Structured terminal flags computed after each control step.

    Attributes:
        timeout (torch.Tensor): Episode reached max length.
        fall (torch.Tensor): Base height below fall threshold.
        posture (torch.Tensor): Roll/pitch sustained beyond patience.
        curve_deviation (torch.Tensor): Left Bezier corridor.
        stagnation (torch.Tensor): No progress along the arc for too long.
        success (torch.Tensor): Held inside success set for required steps.
    """

    timeout: torch.Tensor
    fall: torch.Tensor
    posture: torch.Tensor
    curve_deviation: torch.Tensor
    stagnation: torch.Tensor
    success: torch.Tensor


def compute_termination(task) -> tuple[torch.Tensor, Termination]:
    """Compute per-environment ``done`` mask and structured termination flags.

    Args:
        task: Navigation task with buffers and Bezier/termination configuration.

    Returns:
        tuple[torch.Tensor, Termination]: Boolean ``done`` vector and a
            :class:`Termination` bundle for reward shaping.
    """
    timeout = task.episode_length_buf >= task.max_episode_length
    relative_height = task.base_pos[:, 2] - task.terrain_height_at_robot
    fall = relative_height < float(
        task.termination_config["fall_base_height_threshold"]
    )

    soft_roll_deg = float(
        task.termination_config.get(
            "posture_soft_roll_deg",
            task.termination_config["roll_deg"] * 0.8,
        )
    )
    soft_pitch_deg = float(
        task.termination_config.get(
            "posture_soft_pitch_deg",
            task.termination_config["pitch_deg"] * 0.8,
        )
    )
    hard_roll_deg = float(task.termination_config["roll_deg"])
    hard_pitch_deg = float(task.termination_config["pitch_deg"])

    soft_exceeded = (torch.abs(task.base_euler_deg[:, 0]) > soft_roll_deg) | (
        torch.abs(task.base_euler_deg[:, 1]) > soft_pitch_deg
    )
    hard_exceeded = (torch.abs(task.base_euler_deg[:, 0]) > hard_roll_deg) | (
        torch.abs(task.base_euler_deg[:, 1]) > hard_pitch_deg
    )
    posture_patience = int(task.termination_config["bad_posture_patience"])
    hard_patience = int(task.termination_config.get("critical_posture_patience", 3))

    task.bad_posture_counter = torch.where(
        soft_exceeded,
        task.bad_posture_counter + 1,
        torch.clamp(task.bad_posture_counter - 1, min=0),
    )
    task.critical_posture_counter = torch.where(
        hard_exceeded,
        task.critical_posture_counter + 1,
        torch.zeros_like(task.critical_posture_counter),
    )
    posture = (task.bad_posture_counter >= posture_patience) | (
        task.critical_posture_counter >= hard_patience
    )

    curve_deviation = task.deviation_mask

    stagnation_min_remaining_m = float(
        task.bezier_config.get("stagnation_min_remaining_m", 0.75)
    )
    stagnation_progress_eps = float(
        task.bezier_config.get("stagnation_progress_eps", 0.005)
    )
    stagnation_consistent_progress_eps = float(
        task.bezier_config.get(
            "stagnation_consistent_progress_eps",
            stagnation_progress_eps * 0.5,
        )
    )
    stagnation_steps_limit = int(task.bezier_config.get("stagnation_steps_limit", 75))
    stagnation_residual_speed_mps = float(
        task.bezier_config.get("stagnation_residual_speed_mps", 0.10)
    )
    stagnation_uphill_slope_deg = float(
        task.bezier_config.get("stagnation_uphill_slope_deg", 2.0)
    )
    stagnation_recovery_steps = int(
        task.bezier_config.get("stagnation_recovery_steps", 1)
    )
    planar_speed = torch.linalg.vector_norm(task.base_lin_vel_body[:, :2], dim=1)
    stagnating = (task.remaining_distance > stagnation_min_remaining_m) & (
        task.arc_progress < stagnation_progress_eps
    )
    tolerated_stagnation = (
        (planar_speed >= stagnation_residual_speed_mps)
        | (task.local_forward_slope_deg >= stagnation_uphill_slope_deg)
        | (task.arc_progress >= stagnation_consistent_progress_eps)
    )
    task.stagnation_counter = torch.where(
        stagnating & ~tolerated_stagnation,
        task.stagnation_counter + 1,
        torch.clamp(task.stagnation_counter - stagnation_recovery_steps, min=0),
    )
    stagnation = task.stagnation_counter >= stagnation_steps_limit

    success_radius_m = float(task.bezier_config["success_radius_m"])
    success_yaw_rad = float(task.success_yaw_rad)
    success_speed_mps = float(task.bezier_config["success_speed_mps"])
    success_yaw_rate_rad_s = float(task.bezier_config["success_yaw_rate_rad_s"])
    hold_steps = int(task.bezier_config["hold_steps"])
    planar_speed = torch.linalg.vector_norm(task.base_lin_vel_body[:, :2], dim=1)
    settled = (
        (task.goal_distance <= success_radius_m)
        & (torch.abs(task.goal_yaw_error) <= success_yaw_rad)
        & (planar_speed <= success_speed_mps)
        & (torch.abs(task.imu_gyro[:, 2]) <= success_yaw_rate_rad_s)
    )
    task.success_hold_counter = torch.where(
        settled,
        task.success_hold_counter + 1,
        torch.zeros_like(task.success_hold_counter),
    )
    success = task.success_hold_counter >= hold_steps

    done = timeout | fall | posture | curve_deviation | stagnation | success

    return done, Termination(
        timeout=timeout,
        fall=fall,
        posture=posture,
        curve_deviation=curve_deviation,
        stagnation=stagnation,
        success=success,
    )


def terminal_reward_terms(task, termination: Termination) -> dict[str, torch.Tensor]:
    """Compute terminal-only bonuses and penalties from structured flags.

    Args:
        task: Navigation task exposing ``terminal_reward_scales``.
        termination (Termination): Flags from :func:`compute_termination`.

    Returns:
        dict[str, torch.Tensor]: Sparse terminal reward tensors keyed by term name.
    """
    terminal_scales = getattr(task, "terminal_reward_scales", {})
    success_bonus = termination.success.to(gs.tc_float) * float(
        terminal_scales.get("success", 0.0)
    )
    fall_penalty = (termination.fall | termination.posture).to(gs.tc_float) * float(
        terminal_scales.get("fall", 0.0)
    )
    curve_deviation_penalty = termination.curve_deviation.to(gs.tc_float) * float(
        terminal_scales.get("curve_deviation", 0.0)
    )
    stagnation_penalty = termination.stagnation.to(gs.tc_float) * float(
        terminal_scales.get("stagnation", 0.0)
    )

    return {
        "success": success_bonus,
        "fall": fall_penalty,
        "curve_deviation": curve_deviation_penalty,
        "stagnation": stagnation_penalty,
    }
