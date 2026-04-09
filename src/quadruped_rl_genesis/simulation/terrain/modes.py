"""Terrain mode strings and Genesis rough-terrain morph integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quadruped_rl_genesis.simulation.terrain.config import DEFAULT_TERRAIN_GENERATOR

TERRAIN_MODE_IRREGULAR = "irregular"
TERRAIN_MODE_ROUGH = "rough"

_TERRAIN_MODE_ALIASES: dict[str, str] = {
    "procedural": TERRAIN_MODE_IRREGULAR,
    "genesis_subterrain": TERRAIN_MODE_ROUGH,
}


def normalize_terrain_mode(mode: object | None) -> str:
    """Return canonical ``terrain.mode``: ``irregular`` (procedural heightmap) or ``rough`` (subterrain grid).

    Aliases for older configs: ``procedural`` → ``irregular``; ``genesis_subterrain``
    → ``rough`` (as a *mode string* only). Omitted or ``None`` → ``irregular``.
    """
    if mode is None:
        return TERRAIN_MODE_IRREGULAR
    key = str(mode).strip().lower()
    return _TERRAIN_MODE_ALIASES.get(key, str(mode).strip())


def rough_terrain_config(terrain_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``terrain.rough`` settings dict for ``terrain.mode: rough``.

    Falls back to legacy ``terrain.genesis_subterrain`` when ``rough`` is absent.
    """
    if not terrain_config:
        return {}
    block = terrain_config.get("rough")
    if isinstance(block, Mapping) and block:
        return block
    legacy = terrain_config.get("genesis_subterrain")
    return legacy if isinstance(legacy, Mapping) else {}


def validate_rough_terrain_extent(terrain_config: Mapping[str, Any]) -> None:
    """Check ``terrain.size`` matches the subterrain grid extent for ``terrain.mode: rough``.

    Physical width and length are ``n_subterrains * subterrain_size`` (per axis).
    """
    sub = rough_terrain_config(terrain_config)
    if not isinstance(sub, Mapping) or not sub:
        raise ValueError(
            "terrain.mode=rough requires a non-empty terrain.rough mapping "
            "(or legacy terrain.genesis_subterrain)."
        )
    n_sub = sub.get("n_subterrains")
    st_size = sub.get("subterrain_size")
    if n_sub is None or st_size is None:
        raise ValueError("terrain.rough must define n_subterrains and subterrain_size.")
    nx, ny = int(n_sub[0]), int(n_sub[1])
    sx, sy = float(st_size[0]), float(st_size[1])
    expected = (nx * sx, ny * sy)
    cfg = tuple(float(x) for x in terrain_config.get("size", (0.0, 0.0)))
    tol = 1.0e-2
    if abs(cfg[0] - expected[0]) > tol or abs(cfg[1] - expected[1]) > tol:
        raise ValueError(
            f"terrain.size {cfg} must equal n_subterrains * subterrain_size = {expected}."
        )


def build_rough_terrain_morph_kwargs(
    terrain_config: Mapping[str, Any],
    *,
    terrain_pos: tuple[float, float, float],
    default_uv_scale: float,
) -> dict[str, Any]:
    """Build Genesis ``Terrain`` morph keyword arguments for rough subterrain grids.

    Args:
        terrain_config (Mapping[str, Any]): Experiment terrain YAML block.
        terrain_pos (tuple[float, float, float]): World position passed to Genesis.
        default_uv_scale (float): Fallback UV scale when omitted in config.

    Returns:
        dict[str, Any]: Keyword arguments for ``gs.morphs.Terrain``.

    Raises:
        ValueError: When the rough block is missing or lacks required keys.
    """
    sub = rough_terrain_config(terrain_config)
    if not isinstance(sub, Mapping) or not sub:
        raise ValueError(
            "terrain.mode=rough requires a non-empty terrain.rough block "
            "(or legacy terrain.genesis_subterrain)."
        )
    required = (
        "n_subterrains",
        "subterrain_size",
        "horizontal_scale",
        "vertical_scale",
    )
    for key in required:
        if key not in sub:
            raise ValueError(f"terrain.rough must set {key}.")
    kwargs: dict[str, Any] = {
        "pos": terrain_pos,
        "uv_scale": float(sub.get("uv_scale", default_uv_scale)),
        "randomize": bool(sub.get("randomize", False)),
        "n_subterrains": tuple(int(x) for x in sub["n_subterrains"]),
        "subterrain_size": tuple(float(x) for x in sub["subterrain_size"]),
        "horizontal_scale": float(sub["horizontal_scale"]),
        "vertical_scale": float(sub["vertical_scale"]),
    }
    if sub.get("name"):
        kwargs["name"] = str(sub["name"])
    if "subterrain_types" in sub:
        kwargs["subterrain_types"] = sub["subterrain_types"]
    if sub.get("subterrain_parameters") is not None:
        kwargs["subterrain_parameters"] = sub["subterrain_parameters"]
    return kwargs


# Re-export for callers that expect constants next to mode helpers.
__all__ = [
    "DEFAULT_TERRAIN_GENERATOR",
    "TERRAIN_MODE_IRREGULAR",
    "TERRAIN_MODE_ROUGH",
    "build_rough_terrain_morph_kwargs",
    "normalize_terrain_mode",
    "rough_terrain_config",
    "validate_rough_terrain_extent",
]
