"""Terrain curriculum resolution for procedural ``irregular`` heightmaps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from quadruped_rl_genesis.simulation.terrain.config import _resolve_terrace_settings
from quadruped_rl_genesis.simulation.terrain.modes import (
    TERRAIN_MODE_ROUGH,
    normalize_terrain_mode,
)

_CURRICULUM_LAYOUT_DIRECTIONS = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, 1.0),
)


def _goal_curriculum_stage_max_distance(
    goal_config: Mapping[str, Any] | None,
    stage_index: int,
) -> float:
    """Return the configured maximum goal distance for one goal-curriculum stage.

    Args:
        goal_config (Mapping[str, Any] | None): Goal block from experiment YAML.
        stage_index (int): Zero-based stage index (``stage_{k+1}`` keys in YAML).

    Returns:
        float: ``max_distance_m`` for the stage, or the global default when unset.
    """
    if goal_config is None:
        return 0.0

    curriculum = goal_config.get("curriculum", {})
    if isinstance(curriculum, Mapping):
        stage_cfg = curriculum.get(f"stage_{stage_index + 1}", {})
        if isinstance(stage_cfg, Mapping) and "max_distance_m" in stage_cfg:
            return float(stage_cfg["max_distance_m"])

    return float(goal_config.get("max_distance_m", 0.0))


def resolve_terrain_curriculum_spec(
    terrain_config: Mapping[str, Any] | None,
    goal_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve terrain curriculum for ``terrain.mode: irregular`` only.

    Applies when a procedural heightmap is built (noise, terraces, patches). Not used
    for ``terrain.mode: rough`` (subterrain grid), which must keep ``terrain.curriculum.enabled``
    false. When curriculum is enabled, the final stage defines the global heightfield;
    earlier stages are blended as easier local regions; episode resets pick spawns per stage.
    """
    terrain_cfg = terrain_config or {}
    curriculum_cfg_early = terrain_cfg.get("curriculum", {})
    if (
        normalize_terrain_mode(terrain_cfg.get("mode")) == TERRAIN_MODE_ROUGH
        and isinstance(curriculum_cfg_early, Mapping)
        and curriculum_cfg_early.get("enabled", False)
    ):
        raise ValueError(
            "terrain.curriculum.enabled is incompatible with "
            "terrain.mode=rough; disable terrain.curriculum."
        )
    generator_cfg = terrain_cfg.get("generator", {})
    terrace_defaults = _resolve_terrace_settings(
        generator_cfg.get("terrace", {}) if isinstance(generator_cfg, Mapping) else None
    )
    curriculum_cfg = terrain_cfg.get("curriculum", {})
    if not isinstance(curriculum_cfg, Mapping) or not curriculum_cfg.get(
        "enabled", False
    ):
        return {
            "enabled": False,
            "progression": "steps",
            "stage_steps": [0],
            "evaluation_stage_index": 0,
            "stages": [
                {
                    "index": 0,
                    "name": "stage_1",
                    "step_height_m": float(terrace_defaults["step_height_m"]),
                    "terrace_width_m": float(terrace_defaults["terrace_width_m"]),
                    "edge_smoothing": float(terrace_defaults["edge_smoothing"]),
                    "global_height_scale": 1.0,
                    "local_irregularity_m": 0.0,
                    "roughness_residual_m": 0.0,
                    "spawn_xy": (0.0, 0.0),
                    "patch_radius_m": 0.0,
                    "blend_radius_m": 0.0,
                }
            ],
        }

    stage_keys = sorted(
        key
        for key in curriculum_cfg
        if key.startswith("stage_") and key != "stage_steps"
    )
    if not stage_keys:
        raise ValueError(
            "terrain.curriculum.enabled=true requires at least one stage_N block."
        )

    terrain_size = tuple(
        float(value) for value in terrain_cfg.get("size", (50.0, 50.0))
    )
    half_width = terrain_size[0] * 0.5
    half_length = terrain_size[1] * 0.5
    outer_margin = max(float(curriculum_cfg.get("outer_margin_m", 2.0)), 0.0)
    patch_padding = max(float(curriculum_cfg.get("patch_padding_m", 1.5)), 0.0)
    global_stage_index = len(stage_keys) - 1
    progression = str(curriculum_cfg.get("progression", "steps"))
    stage_steps = list(curriculum_cfg.get("stage_steps", [0] * len(stage_keys)))
    evaluation_stage = int(curriculum_cfg.get("evaluation_stage", len(stage_keys))) - 1
    evaluation_stage = int(np.clip(evaluation_stage, 0, len(stage_keys) - 1))

    stages: list[dict[str, Any]] = []
    for stage_index, stage_key in enumerate(stage_keys):
        stage_cfg_raw = curriculum_cfg.get(stage_key, {})
        stage_cfg = stage_cfg_raw if isinstance(stage_cfg_raw, Mapping) else {}
        terrace_width_m = float(
            stage_cfg.get("terrace_width_m", terrace_defaults["terrace_width_m"])
        )
        terrace_width_m = max(terrace_width_m, 0.25)
        goal_max_distance = _goal_curriculum_stage_max_distance(
            goal_config, stage_index
        )
        patch_radius_m = 0.0
        spawn_xy = (0.0, 0.0)
        if stage_index != global_stage_index:
            patch_radius_m = max(
                goal_max_distance + max(terrace_width_m * 1.5, 2.0),
                terrace_width_m * 3.0,
            )
            direction_x, direction_y = _CURRICULUM_LAYOUT_DIRECTIONS[
                stage_index % len(_CURRICULUM_LAYOUT_DIRECTIONS)
            ]
            center_x = direction_x * max(
                half_width - patch_radius_m - outer_margin, 0.0
            )
            center_y = direction_y * max(
                half_length - patch_radius_m - outer_margin, 0.0
            )
            spawn_xy = (float(center_x), float(center_y))
        blend_radius_m = (
            patch_radius_m + max(terrace_width_m * 1.25, 2.0)
            if patch_radius_m > 0.0
            else 0.0
        )
        stages.append(
            {
                "index": stage_index,
                "name": stage_key,
                "step_height_m": max(
                    float(
                        stage_cfg.get(
                            "step_height_m", terrace_defaults["step_height_m"]
                        )
                    ),
                    0.0,
                ),
                "terrace_width_m": terrace_width_m,
                "edge_smoothing": float(
                    np.clip(
                        stage_cfg.get(
                            "edge_smoothing", terrace_defaults["edge_smoothing"]
                        ),
                        0.0,
                        0.45,
                    )
                ),
                "global_height_scale": float(
                    np.clip(stage_cfg.get("global_height_scale", 1.0), 0.05, 1.0)
                ),
                "local_irregularity_m": max(
                    float(stage_cfg.get("local_irregularity_m", 0.0)), 0.0
                ),
                "roughness_residual_m": max(
                    float(stage_cfg.get("roughness_residual_m", 0.0)), 0.0
                ),
                "spawn_xy": spawn_xy,
                "patch_radius_m": patch_radius_m,
                "blend_radius_m": blend_radius_m,
                "goal_max_distance_m": goal_max_distance,
                "patch_padding_m": patch_padding,
            }
        )

    return {
        "enabled": True,
        "progression": progression,
        "stage_steps": stage_steps,
        "evaluation_stage_index": evaluation_stage,
        "stages": stages,
    }
