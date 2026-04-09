"""Default terrain generator settings and terrace resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_TERRAIN_GENERATOR = {
    "macro_weight": 1.0,
    "ridge_weight": 0.45,
    "detail_weight": 0.18,
    "feature_weight": 0.32,
    "swale_weight": 0.2,
    "warp_strength": 0.08,
    "feature_density": 1.0,
    "depression_ratio": 0.38,
}

DEFAULT_TERRACE_GENERATOR = {
    "enabled": False,
    "profile": "contour",
    "step_height_m": 0.04,
    "terrace_width_m": 2.5,
    "edge_smoothing": 0.18,
    "sharp_edges": True,
    "detail_suppress": 0.72,
    "plateau_flatten": 0.92,
    "terrace_noise_suppress": 0.94,
    "warp_strength": 0.08,
    "local_noise_weight": 0.08,
    "activation_slope_deg_min": 4.0,
    "activation_slope_deg_max": 14.0,
}


def _resolve_generator_settings(
    generator_config: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Merge YAML generator scalars into the default procedural noise profile.

    Args:
        generator_config (Mapping[str, Any] | None): ``terrain.generator`` mapping.

    Returns:
        dict[str, float]: Clamped numeric settings used by heightmap synthesis.
    """
    settings = DEFAULT_TERRAIN_GENERATOR.copy()

    if generator_config is not None:
        for key, value in generator_config.items():
            if key in settings and isinstance(value, (int, float)):
                settings[key] = float(value)

    settings["ridge_weight"] = max(0.0, settings["ridge_weight"])
    settings["detail_weight"] = max(0.0, settings["detail_weight"])
    settings["feature_weight"] = max(0.0, settings["feature_weight"])
    settings["swale_weight"] = max(0.0, settings["swale_weight"])
    settings["warp_strength"] = float(np.clip(settings["warp_strength"], 0.0, 0.3))
    settings["feature_density"] = max(0.25, settings["feature_density"])
    settings["depression_ratio"] = float(
        np.clip(settings["depression_ratio"], 0.0, 1.0)
    )

    return settings


def _resolve_terrace_settings(
    terrace_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge terrace YAML into defaults and normalize profile names.

    Args:
        terrace_config (Mapping[str, Any] | None): ``terrain.generator.terrace`` block.

    Returns:
        dict[str, Any]: Terrace settings with validated ``profile`` and booleans.
    """
    settings: dict[str, Any] = dict(DEFAULT_TERRACE_GENERATOR)

    if terrace_config is not None:
        for key, value in terrace_config.items():
            if key not in settings:
                continue
            settings[key] = value

    settings["enabled"] = bool(settings["enabled"])
    settings["profile"] = str(settings.get("profile", "radial")).lower()
    if settings["profile"] not in {"contour", "radial", "axis_x"}:
        settings["profile"] = "contour"
    settings["step_height_m"] = max(float(settings["step_height_m"]), 0.0)
    settings["terrace_width_m"] = max(float(settings["terrace_width_m"]), 0.25)
    settings["edge_smoothing"] = float(np.clip(settings["edge_smoothing"], 0.0, 0.45))
    settings["sharp_edges"] = bool(settings.get("sharp_edges", False))
    settings["detail_suppress"] = float(
        np.clip(float(settings.get("detail_suppress", 0.0)), 0.0, 1.0)
    )
    settings["plateau_flatten"] = float(
        np.clip(float(settings.get("plateau_flatten", 0.0)), 0.0, 1.0)
    )
    settings["terrace_noise_suppress"] = float(
        np.clip(float(settings.get("terrace_noise_suppress", 0.0)), 0.0, 1.0)
    )
    settings["warp_strength"] = float(np.clip(settings["warp_strength"], 0.0, 0.35))
    settings["local_noise_weight"] = max(float(settings["local_noise_weight"]), 0.0)
    settings["activation_slope_deg_min"] = max(
        float(settings.get("activation_slope_deg_min", 4.0)),
        0.0,
    )
    settings["activation_slope_deg_max"] = max(
        float(settings.get("activation_slope_deg_max", 14.0)),
        settings["activation_slope_deg_min"] + 1.0,
    )

    return settings
