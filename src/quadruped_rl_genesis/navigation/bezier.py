"""Batched cubic Bezier planner used by the navigation task."""

from __future__ import annotations

from typing import Any

import torch

_EPS = 1.0e-6


def _normalize_xy(values: torch.Tensor) -> torch.Tensor:
    """Normalize 2D vectors row-wise with a small epsilon floor.

    Args:
        values (torch.Tensor): Shape ``[..., 2]`` planar vectors.

    Returns:
        torch.Tensor: Unit vectors with the same shape as ``values``.
    """
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp(min=_EPS)

    return values / norms


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles into the ``[-pi, pi]`` interval.

    Args:
        angle (torch.Tensor): Angles in radians.

    Returns:
        torch.Tensor: Wrapped angles with the same shape as ``angle``.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class BezierPlanner:
    """Fixed-per-episode cubic Bezier planner batched over vector environments.

    Samples the spline at uniform parameter values, caches arc length, and exposes
    :meth:`step` for closed-loop command generation.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        config: dict[str, Any],
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create one Bezier planner shared by all environments.

        Args:
            num_envs (int): Number of vectorized environments.
            config (dict[str, Any]): Planner configuration block.
            device (torch.device | str): Torch device used by the planner.
            dtype (torch.dtype): Floating-point dtype for internal buffers. Defaults
                to ``torch.float32``.
        """
        self.num_envs = int(num_envs)
        self.config = config
        self.device = device
        self.dtype = dtype
        self.num_samples = int(config.get("num_samples", 96))

        self.control_points = torch.zeros(
            (self.num_envs, 4, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self.samples_xy = torch.zeros(
            (self.num_envs, self.num_samples, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self.samples_tangent = torch.zeros_like(self.samples_xy)
        self.samples_curvature = torch.zeros(
            (self.num_envs, self.num_samples),
            device=self.device,
            dtype=self.dtype,
        )
        self.samples_s = torch.zeros(
            (self.num_envs, self.num_samples),
            device=self.device,
            dtype=self.dtype,
        )
        self.segment_length = torch.zeros(
            (self.num_envs, self.num_samples - 1),
            device=self.device,
            dtype=self.dtype,
        )
        self.curve_length = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=self.dtype,
        )
        self.prev_arc_s = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=self.dtype,
        )
        self.goal_yaw_target = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=self.dtype,
        )
        self._t_values = torch.linspace(
            0.0,
            1.0,
            self.num_samples,
            device=self.device,
            dtype=self.dtype,
        )

    def _sample_curve(self, control_points: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Sample discrete geometry along cubic control polygons.

        Args:
            control_points (torch.Tensor): Shape ``[N, 4, 2]`` P0..P3 for each env.

        Returns:
            tuple[torch.Tensor, ...]: ``points``, ``tangent``, ``curvature``,
                cumulative ``curve_s``, and per-segment lengths.
        """
        p0 = control_points[:, 0].unsqueeze(1)
        p1 = control_points[:, 1].unsqueeze(1)
        p2 = control_points[:, 2].unsqueeze(1)
        p3 = control_points[:, 3].unsqueeze(1)

        t = self._t_values.view(1, -1, 1)
        one_minus_t = 1.0 - t

        points = (
            (one_minus_t**3) * p0
            + 3.0 * (one_minus_t**2) * t * p1
            + 3.0 * one_minus_t * (t**2) * p2
            + (t**3) * p3
        )
        first_derivative = 3.0 * (
            (one_minus_t**2) * (p1 - p0)
            + 2.0 * one_minus_t * t * (p2 - p1)
            + (t**2) * (p3 - p2)
        )
        second_derivative = 6.0 * (
            one_minus_t * (p2 - 2.0 * p1 + p0) + t * (p3 - 2.0 * p2 + p1)
        )

        tangent = first_derivative / torch.linalg.vector_norm(
            first_derivative,
            dim=2,
            keepdim=True,
        ).clamp(min=_EPS)

        cross = (
            first_derivative[..., 0] * second_derivative[..., 1]
            - first_derivative[..., 1] * second_derivative[..., 0]
        )
        curvature = torch.abs(cross) / (
            torch.linalg.vector_norm(first_derivative, dim=2).clamp(min=_EPS) ** 3
        )

        segments = points[:, 1:, :] - points[:, :-1, :]
        segment_length = torch.linalg.vector_norm(segments, dim=2).clamp(min=_EPS)
        curve_s = torch.zeros(
            (control_points.shape[0], self.num_samples),
            device=self.device,
            dtype=self.dtype,
        )
        curve_s[:, 1:] = torch.cumsum(segment_length, dim=1)

        return points, tangent, curvature, curve_s, segment_length

    def reset_envs(
        self,
        env_ids: torch.Tensor,
        robot_xy: torch.Tensor,
        robot_forward_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> None:
        """Generate a fresh fixed Bezier curve for the selected environments.

        Args:
            env_ids (torch.Tensor): Active environment indices.
            robot_xy (torch.Tensor): World XY base positions ``[num_envs, 2]``.
            robot_forward_xy (torch.Tensor): Body-forward projections ``[num_envs, 2]``.
            goal_xy (torch.Tensor): World XY goal positions ``[num_envs, 2]``.
        """
        if env_ids.numel() == 0:
            return

        start_xy = robot_xy[env_ids]
        goal_xy_selected = goal_xy[env_ids]
        robot_forward_xy_selected = _normalize_xy(robot_forward_xy[env_ids])
        goal_dir = _normalize_xy(goal_xy_selected - start_xy)

        handle_ratio = float(self.config.get("handle_ratio", 0.33))
        handle_min_m = float(self.config.get("handle_min_m", 0.5))
        handle_max_m = float(self.config.get("handle_max_m", 4.0))
        distance = torch.linalg.vector_norm(goal_xy_selected - start_xy, dim=1)
        handle_len = torch.clamp(
            handle_ratio * distance,
            min=handle_min_m,
            max=handle_max_m,
        )

        control_points = torch.zeros(
            (env_ids.numel(), 4, 2),
            device=self.device,
            dtype=self.dtype,
        )
        control_points[:, 0] = start_xy
        control_points[:, 1] = (
            start_xy + handle_len.unsqueeze(1) * robot_forward_xy_selected
        )
        control_points[:, 2] = goal_xy_selected - handle_len.unsqueeze(1) * goal_dir
        control_points[:, 3] = goal_xy_selected

        points, tangent, curvature, curve_s, segment_length = self._sample_curve(
            control_points
        )
        self.control_points[env_ids] = control_points
        self.samples_xy[env_ids] = points
        self.samples_tangent[env_ids] = tangent
        self.samples_curvature[env_ids] = curvature
        self.samples_s[env_ids] = curve_s
        self.segment_length[env_ids] = segment_length
        self.curve_length[env_ids] = curve_s[:, -1]
        self.prev_arc_s[env_ids] = 0.0
        final_tangent = tangent[:, -1]
        self.goal_yaw_target[env_ids] = torch.atan2(
            final_tangent[:, 1],
            final_tangent[:, 0],
        )

    def _interpolate_samples(
        self,
        s_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate pose along the curve at the queried arc lengths.

        Args:
            s_query (torch.Tensor): Arc length per environment, shape ``[num_envs]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: World XY point, unit
                tangent, and scalar curvature per env.
        """
        upper_indices = torch.sum(
            self.samples_s <= s_query.unsqueeze(1),
            dim=1,
        ).clamp(min=1, max=self.num_samples - 1)
        lower_indices = upper_indices - 1
        batch_indices = torch.arange(self.num_envs, device=self.device)

        lower_s = self.samples_s[batch_indices, lower_indices]
        upper_s = self.samples_s[batch_indices, upper_indices]
        interp = (s_query - lower_s) / (upper_s - lower_s).clamp(min=_EPS)

        lower_points = self.samples_xy[batch_indices, lower_indices]
        upper_points = self.samples_xy[batch_indices, upper_indices]
        points = lower_points + interp.unsqueeze(1) * (upper_points - lower_points)

        lower_tangent = self.samples_tangent[batch_indices, lower_indices]
        upper_tangent = self.samples_tangent[batch_indices, upper_indices]
        tangent = _normalize_xy(
            lower_tangent + interp.unsqueeze(1) * (upper_tangent - lower_tangent)
        )

        lower_curvature = self.samples_curvature[batch_indices, lower_indices]
        upper_curvature = self.samples_curvature[batch_indices, upper_indices]
        curvature = lower_curvature + interp * (upper_curvature - lower_curvature)

        return points, tangent, curvature

    def step(
        self, robot_xy: torch.Tensor, robot_yaw: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project each robot onto the curve and compute planner outputs.

        Args:
            robot_xy (torch.Tensor): World XY positions ``[num_envs, 2]``.
            robot_yaw (torch.Tensor): World yaw per env ``[num_envs]``.

        Returns:
            dict[str, torch.Tensor]: Command velocity, arc metrics, distances, masks,
                and lookahead tangent fields consumed by the task.
        """
        segment_start = self.samples_xy[:, :-1, :]
        segment_end = self.samples_xy[:, 1:, :]
        segment_vec = segment_end - segment_start
        segment_len_sq = torch.sum(segment_vec * segment_vec, dim=2).clamp(min=_EPS)

        rel_to_start = robot_xy.unsqueeze(1) - segment_start
        proj_u = torch.sum(rel_to_start * segment_vec, dim=2) / segment_len_sq
        proj_u = torch.clamp(proj_u, 0.0, 1.0)
        closest_points = segment_start + proj_u.unsqueeze(2) * segment_vec
        rel_to_closest = robot_xy.unsqueeze(1) - closest_points
        distance_sq = torch.sum(rel_to_closest * rel_to_closest, dim=2)
        closest_segment = torch.argmin(distance_sq, dim=1)

        batch_indices = torch.arange(self.num_envs, device=self.device)
        closest_u = proj_u[batch_indices, closest_segment]
        closest_point = closest_points[batch_indices, closest_segment]
        closest_segment_vec = segment_vec[batch_indices, closest_segment]
        closest_segment_len = self.segment_length[batch_indices, closest_segment]
        closest_segment_dir = closest_segment_vec / closest_segment_len.unsqueeze(
            1
        ).clamp(min=_EPS)
        closest_offset = robot_xy - closest_point
        cross_track_error = (
            closest_segment_dir[:, 0] * closest_offset[:, 1]
            - closest_segment_dir[:, 1] * closest_offset[:, 0]
        )

        base_s = self.samples_s[batch_indices, closest_segment]
        arc_s = base_s + closest_u * closest_segment_len
        arc_progress = arc_s - self.prev_arc_s
        self.prev_arc_s = arc_s

        remaining_distance = torch.clamp(self.curve_length - arc_s, min=0.0)
        lookahead_m = float(self.config.get("lookahead_m", 1.0))
        lookahead_s = torch.clamp(arc_s + lookahead_m, min=0.0)
        lookahead_s = torch.minimum(lookahead_s, self.curve_length)
        _lookahead_point, lookahead_tangent, lookahead_curvature = (
            self._interpolate_samples(lookahead_s)
        )

        tangent_yaw = torch.atan2(lookahead_tangent[:, 1], lookahead_tangent[:, 0])
        heading_error_to_tangent = _wrap_angle(tangent_yaw - robot_yaw)
        goal_yaw_error = _wrap_angle(self.goal_yaw_target - robot_yaw)
        goal_distance = torch.linalg.vector_norm(
            self.control_points[:, 3] - robot_xy,
            dim=1,
        )

        target_speed_mps = float(self.config.get("target_speed_mps", 1.0))
        decel_radius_m = float(self.config.get("decel_radius_m", 2.0))
        stop_radius_m = float(self.config.get("stop_radius_m", 0.35))
        a_lat_max_mps2 = float(self.config.get("a_lat_max_mps2", 1.5))
        curvature_eps = float(self.config.get("curvature_eps", 1.0e-4))
        yaw_kp = float(self.config.get("yaw_kp", 2.0))
        yaw_rate_max = float(self.config.get("yaw_rate_max", 1.5))

        goal_scale = torch.clamp(
            remaining_distance / max(decel_radius_m, _EPS), 0.0, 1.0
        )
        stop_scale = torch.clamp(goal_distance / max(stop_radius_m, _EPS), 0.0, 1.0)
        v_goal = target_speed_mps * goal_scale * stop_scale
        v_curve = torch.sqrt(
            torch.full_like(lookahead_curvature, a_lat_max_mps2)
            / torch.clamp(lookahead_curvature.abs(), min=curvature_eps)
        )
        cmd_vx = torch.clamp(
            torch.minimum(v_goal, v_curve),
            min=0.0,
            max=target_speed_mps,
        )
        cmd_yaw_rate = torch.clamp(
            yaw_kp * heading_error_to_tangent,
            min=-yaw_rate_max,
            max=yaw_rate_max,
        )
        cmd_vel = torch.stack(
            (
                cmd_vx,
                torch.zeros_like(cmd_vx),
                cmd_yaw_rate,
            ),
            dim=1,
        )

        approach_radius_m = float(self.config.get("approach_radius_m", 1.0))
        success_radius_m = float(self.config.get("success_radius_m", 0.25))
        max_deviation_m = float(self.config.get("max_deviation_m", 2.5))

        return {
            "cmd_vel": cmd_vel,
            "arc_s": arc_s,
            "arc_progress": arc_progress,
            "remaining_distance": remaining_distance,
            "goal_distance": goal_distance,
            "cross_track_error": cross_track_error,
            "heading_error_to_tangent": heading_error_to_tangent,
            "goal_yaw_target": self.goal_yaw_target.clone(),
            "goal_yaw_error": goal_yaw_error,
            "curve_length": self.curve_length.clone(),
            "deviation_mask": torch.abs(cross_track_error) > max_deviation_m,
            "within_approach_zone": goal_distance <= approach_radius_m,
            "within_success_zone": goal_distance <= success_radius_m,
            "lookahead_tangent_xy": lookahead_tangent,
            "lookahead_curvature": lookahead_curvature,
        }
