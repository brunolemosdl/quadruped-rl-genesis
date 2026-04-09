"""Terrain heightmaps and sampling for ``terrain.mode`` ``irregular`` | ``rough``.

``irregular`` uses a procedural heightfield (noise, terraces, optional terrain
curriculum). ``rough`` uses Genesis ``Terrain`` from a subterrain grid; settings
live under ``terrain.rough`` (legacy key ``terrain.genesis_subterrain`` is still
read). Local height queries and curriculum resolution live in this package.
"""

from __future__ import annotations

from quadruped_rl_genesis.simulation.terrain.config import (
    DEFAULT_TERRACE_GENERATOR,
    DEFAULT_TERRAIN_GENERATOR,
)
from quadruped_rl_genesis.simulation.terrain.curriculum import (
    resolve_terrain_curriculum_spec,
)
from quadruped_rl_genesis.simulation.terrain.heightmap import (
    generate_random_terrain_heightmap,
)
from quadruped_rl_genesis.simulation.terrain.modes import (
    TERRAIN_MODE_IRREGULAR,
    TERRAIN_MODE_ROUGH,
    build_rough_terrain_morph_kwargs,
    normalize_terrain_mode,
    rough_terrain_config,
    validate_rough_terrain_extent,
)
from quadruped_rl_genesis.simulation.terrain.sampling import (
    estimate_heightmap_slopes_torch,
    sample_goal_xy,
    sample_height_numpy,
    sample_height_torch,
    terrain_limits,
)

__all__ = [
    "DEFAULT_TERRACE_GENERATOR",
    "DEFAULT_TERRAIN_GENERATOR",
    "TERRAIN_MODE_IRREGULAR",
    "TERRAIN_MODE_ROUGH",
    "build_rough_terrain_morph_kwargs",
    "estimate_heightmap_slopes_torch",
    "generate_random_terrain_heightmap",
    "normalize_terrain_mode",
    "resolve_terrain_curriculum_spec",
    "rough_terrain_config",
    "sample_goal_xy",
    "sample_height_numpy",
    "sample_height_torch",
    "terrain_limits",
    "validate_rough_terrain_extent",
]
