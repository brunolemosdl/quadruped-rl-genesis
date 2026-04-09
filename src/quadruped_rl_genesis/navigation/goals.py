"""Goal sampling and per-episode goal tracker resets."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from quadruped_rl_genesis.simulation.terrain import sample_goal_xy, sample_height_torch


def _effective_goal_distance_range(task) -> tuple[float, float]:
    """Resolve goal distance bounds from static config or active curriculum stage.

    Args:
        task: Navigation task exposing ``goal_config`` and ``global_step``.

    Returns:
        tuple[float, float]: ``(min_distance_m, max_distance_m)`` for sampling.
    """
    goal_cfg = getattr(task, "goal_config", {}) or {}
    curriculum = goal_cfg.get("curriculum", {})
    if not curriculum.get("enabled", False):
        return (
            float(goal_cfg.get("min_distance_m", 8.0)),
            float(goal_cfg.get("max_distance_m", 24.0)),
        )

    stage_steps = list(
        curriculum.get("stage_steps", [0, 100_000, 300_000, 600_000, 1_000_000])
    )
    global_step = int(getattr(task, "global_step", 0))
    current_stage = 0
    for stage_index, step_threshold in enumerate(stage_steps):
        if global_step >= step_threshold:
            current_stage = stage_index

    stage_cfg = curriculum.get(f"stage_{current_stage + 1}", {})
    return (
        float(stage_cfg.get("min_distance_m", goal_cfg.get("min_distance_m", 8.0))),
        float(stage_cfg.get("max_distance_m", goal_cfg.get("max_distance_m", 24.0))),
    )


def sample_goal_pose(task, env_ids: torch.Tensor) -> None:
    """Sample goal positions on the terrain for the given environments.

    Args:
        task: Navigation task with terrain heightmap, RNG, and goal buffers.
        env_ids (torch.Tensor): 1-D indices of environments to update in-place.
    """
    if env_ids.numel() == 0:
        return

    min_dist, max_dist = _effective_goal_distance_range(task)
    goal_xy_list = []
    for env_id in env_ids.tolist():
        spawn_xy = (
            float(task.spawn_pos[env_id, 0].item()),
            float(task.spawn_pos[env_id, 1].item()),
        )
        goal_xy_list.append(
            sample_goal_xy(
                count=1,
                size=tuple(task.terrain_config["size"]),
                margin=float(task.goal_config["spawn_margin_m"]),
                min_distance=min_dist,
                max_distance=max_dist,
                spawn_xy=spawn_xy,
                rng=task.goal_rng,
            )[0]
        )
    goal_xy = np.stack(goal_xy_list, axis=0).astype(np.float32)
    goal_xy_tensor = torch.from_numpy(goal_xy).to(
        device=task.device,
        dtype=gs.tc_float,
    )
    goal_z = sample_height_torch(
        task.terrain_heightmap_tensor,
        goal_xy_tensor,
        size=tuple(task.terrain_config["size"]),
    )

    task.goal_pos[env_ids, :2] = goal_xy_tensor
    task.goal_pos[env_ids, 2] = goal_z


def reset_goal_trackers(task, env_ids: torch.Tensor) -> None:
    """Reset goal-distance trackers after sampling a new goal.

    Args:
        task: Navigation task with ``initial_goal_distance`` and
            ``previous_goal_distance`` buffers.
        env_ids (torch.Tensor): 1-D indices of environments to reset.
    """
    if env_ids.numel() == 0:
        return

    goal_delta = task.goal_pos[env_ids, :2] - task.spawn_pos[env_ids, :2]
    initial_distance = torch.linalg.vector_norm(goal_delta, dim=1)
    task.initial_goal_distance[env_ids] = initial_distance
    task.previous_goal_distance[env_ids] = initial_distance
