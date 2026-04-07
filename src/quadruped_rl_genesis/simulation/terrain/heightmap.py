"""Procedural irregular heightmap generation (noise, terraces, curriculum patches)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from quadruped_rl_genesis.simulation.terrain.config import (
    _resolve_generator_settings,
    _resolve_terrace_settings,
)
from quadruped_rl_genesis.simulation.terrain.fields import (
    _normalize_centered,
    _normalize_unit,
    _sample_bilinear,
    _smooth_field,
    _smoothstep,
)
from quadruped_rl_genesis.simulation.terrain.sampling import sample_height_numpy


def _fractal_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base_sigma: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    *,
    ridged: bool = False,
) -> np.ndarray:
    """Generate normalized fractal noise by summing smoothed octaves.

    Args:
        shape (tuple[int, int]): Output field shape.
        rng (np.random.Generator): Random generator used for octave sampling.
        base_sigma (float): Initial smoothing scale for the first octave.
        octaves (int): Number of noise octaves.
        persistence (float): Amplitude decay factor between octaves.
        lacunarity (float): Frequency growth factor between octaves.
        ridged (bool, optional): Whether to transform each octave into ridged
            noise before accumulation.

    Returns:
        np.ndarray: Centered fractal noise field in ``float32`` format.
    """
    field = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    sigma = float(base_sigma)

    for _ in range(max(1, octaves)):
        octave = rng.standard_normal(shape).astype(np.float32)
        octave = _smooth_field(octave, sigma=sigma)
        octave = _normalize_centered(octave)

        if ridged:
            octave = 1.0 - np.abs(octave)
            octave = _normalize_centered(octave)

        field += amplitude * octave
        amplitude *= persistence
        sigma = max(0.75, sigma / lacunarity)

    return _normalize_centered(field)


def _generate_local_features(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rng: np.random.Generator,
    *,
    feature_count: int,
    width: float,
    length: float,
    depression_ratio: float,
) -> np.ndarray:
    """Generate localized hill and depression features over a terrain grid.

    Args:
        grid_x (np.ndarray): X coordinate grid.
        grid_y (np.ndarray): Y coordinate grid.
        rng (np.random.Generator): Random generator used for feature placement.
        feature_count (int): Number of local features to accumulate.
        width (float): Terrain width in meters.
        length (float): Terrain length in meters.
        depression_ratio (float): Probability that a feature becomes a
            depression instead of an elevation.

    Returns:
        np.ndarray: Centered local-feature field.
    """
    features = np.zeros_like(grid_x, dtype=np.float32)

    for _ in range(max(1, feature_count)):
        center_x = rng.uniform(-width / 2, width / 2)
        center_y = rng.uniform(-length / 2, length / 2)
        radius_x = rng.uniform(width * 0.04, width * 0.14)
        radius_y = rng.uniform(length * 0.04, length * 0.14)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        shape_power = rng.uniform(0.8, 1.4)
        amplitude = rng.uniform(0.35, 1.0)
        if rng.random() < depression_ratio:
            amplitude *= -rng.uniform(0.7, 1.2)

        dx = grid_x - center_x
        dy = grid_y - center_y
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        local_x = cos_angle * dx + sin_angle * dy
        local_y = -sin_angle * dx + cos_angle * dy

        radial = (np.abs(local_x) / (radius_x + 1e-6)) ** (2.0 * shape_power) + (
            np.abs(local_y) / (radius_y + 1e-6)
        ) ** (2.0 * shape_power)
        feature = np.exp(-radial).astype(np.float32)

        if rng.random() < 0.55:
            skirt = np.exp(
                -0.5
                * (
                    (local_x / (radius_x * 1.8 + 1e-6)) ** 2
                    + (local_y / (radius_y * 1.8 + 1e-6)) ** 2
                )
            ).astype(np.float32)
            feature = feature - 0.25 * skirt

        tilt = rng.uniform(-0.35, 0.35)
        feature *= 1.0 + tilt * np.tanh(local_x / (radius_x + 1e-6))
        features += amplitude * feature

    return _normalize_centered(features)


def _generate_swales(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rng: np.random.Generator,
    *,
    swale_count: int,
    width: float,
    length: float,
) -> np.ndarray:
    """Generate elongated swale-like depressions over a terrain grid.

    Args:
        grid_x (np.ndarray): X coordinate grid.
        grid_y (np.ndarray): Y coordinate grid.
        rng (np.random.Generator): Random generator used for swale placement.
        swale_count (int): Number of swales to accumulate.
        width (float): Terrain width in meters.
        length (float): Terrain length in meters.

    Returns:
        np.ndarray: Centered swale field.
    """
    swales = np.zeros_like(grid_x, dtype=np.float32)
    span = max(width, length)

    for _ in range(max(1, swale_count)):
        center_x = rng.uniform(-width * 0.25, width * 0.25)
        center_y = rng.uniform(-length * 0.25, length * 0.25)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        reach = rng.uniform(span * 0.22, span * 0.45)
        half_width = rng.uniform(span * 0.015, span * 0.04)
        curvature = rng.uniform(-0.5, 0.5)
        depth = rng.uniform(0.3, 0.9)

        dx = grid_x - center_x
        dy = grid_y - center_y
        along = np.cos(angle) * dx + np.sin(angle) * dy
        across = -np.sin(angle) * dx + np.cos(angle) * dy
        centerline = curvature * (along / (reach + 1e-6)) ** 2 * reach * 0.35
        distance = across - centerline

        trench = np.exp(
            -0.5
            * ((distance / (half_width + 1e-6)) ** 2 + (along / (reach + 1e-6)) ** 2)
        ).astype(np.float32)
        shoulders = np.exp(
            -0.5
            * (
                (distance / (half_width * 2.7 + 1e-6)) ** 2
                + (along / (reach * 1.2 + 1e-6)) ** 2
            )
        ).astype(np.float32)
        swales += depth * (0.22 * shoulders - trench)

    return _normalize_centered(swales)


def _generate_terrain_base_unit_field(
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    size: tuple[float, float],
    resolution: tuple[int, int],
    num_functions: int,
    rng: np.random.Generator,
    settings: Mapping[str, float],
) -> np.ndarray:
    """Generate a normalized rough base height field shared across curriculum stages.

    Args:
        grid_x (np.ndarray): World X coordinate grid ``[Wx, Ly]``.
        grid_y (np.ndarray): World Y coordinate grid ``[Wx, Ly]``.
        size (tuple[float, float]): Terrain extent in meters.
        resolution (tuple[int, int]): Heightmap resolution ``(width, length)``.
        num_functions (int): Complexity driver for fractal octave counts.
        rng (np.random.Generator): NumPy RNG for procedural noise.
        settings (Mapping[str, float]): Generator weights such as ``feature_density``.

    Returns:
        np.ndarray: Centered unit-ish height field ``float32`` matching ``resolution``.
    """
    width, length = size
    width_res, length_res = resolution
    shape = (width_res, length_res)
    complexity = max(4, int(num_functions))
    max_resolution = float(max(shape))
    macro_octaves = min(6, 2 + complexity // 2)
    detail_octaves = min(6, 2 + complexity // 2)
    feature_count = round(complexity * 2.5 * settings["feature_density"])
    swale_count = max(1, complexity // 3)

    macro = _fractal_noise(
        shape,
        rng,
        base_sigma=max_resolution / 5.5,
        octaves=macro_octaves,
        persistence=0.62,
        lacunarity=1.85,
    )
    ridged = _fractal_noise(
        shape,
        rng,
        base_sigma=max_resolution / 7.5,
        octaves=max(3, macro_octaves - 1),
        persistence=0.58,
        lacunarity=1.9,
        ridged=True,
    )
    detail = _fractal_noise(
        shape,
        rng,
        base_sigma=max_resolution / 18.0,
        octaves=detail_octaves,
        persistence=0.5,
        lacunarity=1.7,
    )
    warp_x = _fractal_noise(
        shape,
        rng,
        base_sigma=max_resolution / 10.0,
        octaves=3,
        persistence=0.55,
        lacunarity=1.9,
    )
    warp_y = _fractal_noise(
        shape,
        rng,
        base_sigma=max_resolution / 10.0,
        octaves=3,
        persistence=0.55,
        lacunarity=1.9,
    )

    pixel_x = np.broadcast_to(
        np.arange(width_res, dtype=np.float32)[:, None], shape
    ).copy()
    pixel_y = np.broadcast_to(
        np.arange(length_res, dtype=np.float32)[None, :], shape
    ).copy()
    warp_pixels = settings["warp_strength"] * min(width_res, length_res)

    sample_x = np.clip(pixel_x + warp_pixels * warp_x, 0.0, width_res - 1.001)
    sample_y = np.clip(pixel_y + warp_pixels * warp_y, 0.0, length_res - 1.001)
    warped_macro = _sample_bilinear(macro, sample_x, sample_y)
    warped_ridged = _sample_bilinear(ridged, sample_x, sample_y)
    warped_detail = _sample_bilinear(
        detail,
        np.clip(pixel_x + 1.35 * warp_pixels * warp_x, 0.0, width_res - 1.001),
        np.clip(pixel_y + 1.35 * warp_pixels * warp_y, 0.0, length_res - 1.001),
    )

    local_features = _generate_local_features(
        grid_x,
        grid_y,
        rng,
        feature_count=feature_count,
        width=width,
        length=length,
        depression_ratio=settings["depression_ratio"],
    )
    swales = _generate_swales(
        grid_x,
        grid_y,
        rng,
        swale_count=swale_count,
        width=width,
        length=length,
    )

    roughness_mask = _normalize_unit(
        np.abs(warped_ridged) + 0.5 * np.abs(local_features)
    )
    warped_detail *= 0.3 + 0.7 * roughness_mask

    heightmap = settings["macro_weight"] * warped_macro
    heightmap += settings["ridge_weight"] * warped_ridged
    heightmap += settings["detail_weight"] * warped_detail
    heightmap += settings["feature_weight"] * local_features
    heightmap += settings["swale_weight"] * swales
    heightmap = _smooth_field(heightmap, sigma=max(0.65, max_resolution / 140.0))

    return _normalize_centered(heightmap)


def _generate_terrace_field(
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rng: np.random.Generator,
    settings: Mapping[str, Any],
    center_xy: tuple[float, float],
    resolution: tuple[int, int],
) -> np.ndarray:
    """Generate a soft staircase height field around a curriculum patch center.

    Args:
        grid_x (np.ndarray): World X grid ``[Wx, Ly]``.
        grid_y (np.ndarray): World Y grid ``[Wx, Ly]``.
        rng (np.random.Generator): RNG for optional radial warp noise.
        settings (Mapping[str, Any]): Terrace YAML block (``step_height_m``, etc.).
        center_xy (tuple[float, float]): Patch center in world meters.
        resolution (tuple[int, int]): Grid resolution.

    Returns:
        np.ndarray: Terrace height contribution ``float32``, or zeros if disabled.
    """
    if (
        not bool(settings.get("enabled", False))
        or float(settings["step_height_m"]) <= 0.0
    ):
        return np.zeros_like(grid_x, dtype=np.float32)

    center_x, center_y = center_xy
    local_x = grid_x - center_x
    local_y = grid_y - center_y

    if str(settings.get("profile", "radial")) == "axis_x":
        coordinate = np.abs(local_x)
    else:
        coordinate = np.sqrt(local_x**2 + local_y**2)

    warp_strength = float(settings.get("warp_strength", 0.0))
    if warp_strength > 0.0:
        warp = _fractal_noise(
            resolution,
            rng,
            base_sigma=max(float(max(resolution)) / 8.0, 1.5),
            octaves=3,
            persistence=0.55,
            lacunarity=1.85,
        )
        coordinate = np.maximum(
            coordinate + warp_strength * float(settings["terrace_width_m"]) * warp,
            0.0,
        )

    terrace_width_m = max(float(settings["terrace_width_m"]), 0.25)
    level = coordinate / terrace_width_m
    whole_level = np.floor(level)
    fractional = level - whole_level
    edge_smoothing = float(np.clip(settings["edge_smoothing"], 0.0, 0.45))
    if edge_smoothing <= 1.0e-6:
        transition = np.zeros_like(fractional, dtype=np.float32)
    else:
        transition_start = 1.0 - edge_smoothing
        transition = _smoothstep(
            (fractional - transition_start) / max(edge_smoothing, 1.0e-6)
        )

    terraces = (whole_level + transition) * float(settings["step_height_m"])

    local_noise_weight = float(settings.get("local_noise_weight", 0.0))
    if local_noise_weight > 0.0:
        local_noise = _fractal_noise(
            resolution,
            rng,
            base_sigma=max(float(max(resolution)) / 20.0, 1.0),
            octaves=2,
            persistence=0.55,
            lacunarity=1.9,
        )
        terraces += local_noise_weight * float(settings["step_height_m"]) * local_noise

    return terraces.astype(np.float32)


def _surface_slope_deg(
    surface: np.ndarray,
    *,
    size: tuple[float, float],
) -> np.ndarray:
    """Estimate slope magnitude in degrees from finite differences on a height grid.

    Args:
        surface (np.ndarray): Height samples indexed like the simulation grid.
        size (tuple[float, float]): Physical terrain width and length in meters.

    Returns:
        np.ndarray: Slope magnitude in degrees, same shape as ``surface``.
    """
    width, length = size
    dx = width / max(surface.shape[0] - 1, 1)
    dy = length / max(surface.shape[1] - 1, 1)
    grad_x, grad_y = np.gradient(surface.astype(np.float32), dx, dy, edge_order=1)
    gradient_norm = np.sqrt(grad_x**2 + grad_y**2)
    return np.rad2deg(np.arctan(gradient_norm)).astype(np.float32)


def _terrace_quantize_surface(
    surface: np.ndarray,
    *,
    step_height_m: float,
    edge_smoothing: float,
    sharp_edges: bool = False,
) -> np.ndarray:
    """Quantize a height surface into terrace treads with optional smooth risers.

    Args:
        surface (np.ndarray): Carrier height field.
        step_height_m (float): Vertical spacing between terrace levels.
        edge_smoothing (float): Fractional band used for smoothstep transitions.
        sharp_edges (bool): When ``True``, use hard ``floor`` quantization.

    Returns:
        np.ndarray: Quantized ``float32`` surface.
    """
    if step_height_m <= 0.0:
        return surface.astype(np.float32, copy=False)

    level = surface / step_height_m
    if sharp_edges or edge_smoothing <= 1.0e-6:
        return (np.floor(level) * step_height_m).astype(np.float32)

    whole_level = np.floor(level)
    fractional = level - whole_level
    edge_smoothing = float(np.clip(edge_smoothing, 0.0, 0.45))
    transition_start = 1.0 - edge_smoothing
    transition = _smoothstep(
        (fractional - transition_start) / max(edge_smoothing, 1.0e-6)
    )

    return ((whole_level + transition) * step_height_m).astype(np.float32)


def _apply_contour_terraces(
    base_surface: np.ndarray,
    *,
    size: tuple[float, float],
    resolution: tuple[int, int],
    rng: np.random.Generator,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Apply terraces along the underlying rough-terrain contours.

    Terraces are derived from a smoothed ``carrier_surface`` (contour-aligned).
    ``detail_suppress`` reduces base roughness where terraces apply; ``sharp_edges``
    hard-quantizes the carrier. ``plateau_flatten`` snaps heights toward exact
    multiples of ``step_height_m`` under ``activation`` so treads are flat and
    risers are as steep as the heightfield grid allows. Terrace-local noise is
    scaled down by ``terrace_noise_suppress`` where ``activation`` is high.
    """
    if (
        not bool(settings.get("enabled", False))
        or float(settings["step_height_m"]) <= 0.0
    ):
        return base_surface.astype(np.float32, copy=False)

    width, length = size
    pixel_size_m = min(
        width / max(resolution[0] - 1, 1),
        length / max(resolution[1] - 1, 1),
    )
    carrier_sigma_px = max(
        float(settings["terrace_width_m"]) / max(pixel_size_m, 1.0e-6) * 0.35,
        0.85,
    )
    carrier_surface = _smooth_field(base_surface, sigma=carrier_sigma_px)
    slope_deg = _surface_slope_deg(carrier_surface, size=size)
    activation = _smoothstep(
        (slope_deg - float(settings["activation_slope_deg_min"]))
        / max(
            float(settings["activation_slope_deg_max"])
            - float(settings["activation_slope_deg_min"]),
            1.0e-6,
        )
    )

    sharp_edges = bool(settings.get("sharp_edges", False))
    terraced_carrier = _terrace_quantize_surface(
        carrier_surface,
        step_height_m=float(settings["step_height_m"]),
        edge_smoothing=float(settings["edge_smoothing"]),
        sharp_edges=sharp_edges,
    )
    terrace_delta = terraced_carrier - carrier_surface
    detail = base_surface - carrier_surface
    detail_suppress = float(settings.get("detail_suppress", 0.0))
    terrace_adjust = terrace_delta - detail_suppress * detail
    contribution = activation * terrace_adjust
    terraced_surface = base_surface + contribution

    step_h = float(settings["step_height_m"])
    plateau_flatten = float(settings.get("plateau_flatten", 0.0))
    if plateau_flatten > 1.0e-6 and step_h > 1.0e-9:
        level = np.floor(terraced_surface / step_h)
        flat_z = level * step_h
        blend = activation * plateau_flatten
        terraced_surface = terraced_surface + blend * (flat_z - terraced_surface)

    if float(settings.get("local_noise_weight", 0.0)) > 0.0:
        local_noise = _fractal_noise(
            resolution,
            rng,
            base_sigma=max(float(max(resolution)) / 20.0, 1.0),
            octaves=2,
            persistence=0.55,
            lacunarity=1.9,
        )
        tn_sup = float(settings.get("terrace_noise_suppress", 0.0))
        noise_w = (1.0 - activation * np.clip(tn_sup, 0.0, 1.0)).astype(np.float32)
        terraced_surface += (
            float(settings["local_noise_weight"]) * step_h * local_noise * noise_w
        )

    return terraced_surface.astype(np.float32)


def _stage_surface(
    *,
    stage: dict[str, Any],
    size: tuple[float, float],
    resolution: tuple[int, int],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    base_surface: np.ndarray,
    rough_unit: np.ndarray,
    local_unit: np.ndarray,
    terrace_defaults: Mapping[str, Any],
    rng: np.random.Generator,
    reference_surface: np.ndarray | None = None,
) -> np.ndarray:
    """Blend one curriculum stage recipe into the shared rough carrier.

    Args:
        stage (dict[str, Any]): Stage dict with spawn, scales, and terrace params.
        size (tuple[float, float]): Terrain extent in meters.
        resolution (tuple[int, int]): Grid resolution.
        grid_x (np.ndarray): World coordinate grid X.
        grid_y (np.ndarray): World coordinate grid Y.
        base_surface (np.ndarray): Shared rough base height.
        rough_unit (np.ndarray): Normalized rough basis field.
        local_unit (np.ndarray): Local irregularity basis field.
        terrace_defaults (Mapping[str, Any]): Default terrace generator settings.
        rng (np.random.Generator): RNG for terrace noise.
        reference_surface (np.ndarray | None): Optional surface for height anchoring.

    Returns:
        np.ndarray: Stage heightmap ``float32`` aligned at the spawn location.
    """
    center_xy = tuple(float(value) for value in stage["spawn_xy"])
    global_scale = float(stage.get("global_height_scale", 1.0))
    stage_base = base_surface.astype(np.float32, copy=False) * global_scale
    terrace_profile = str(
        stage.get("terrace_profile", terrace_defaults.get("profile", "contour"))
    ).lower()
    if terrace_profile == "contour":
        surface = _apply_contour_terraces(
            stage_base,
            size=size,
            resolution=resolution,
            rng=rng,
            settings={
                **terrace_defaults,
                "step_height_m": float(stage["step_height_m"]),
                "terrace_width_m": float(stage["terrace_width_m"]),
                "edge_smoothing": float(stage["edge_smoothing"]),
            },
        )
    else:
        surface = stage_base + _generate_terrace_field(
            grid_x=grid_x,
            grid_y=grid_y,
            rng=rng,
            settings={
                **terrace_defaults,
                "step_height_m": float(stage["step_height_m"]),
                "terrace_width_m": float(stage["terrace_width_m"]),
                "edge_smoothing": float(stage["edge_smoothing"]),
                "profile": terrace_profile,
            },
            center_xy=center_xy,
            resolution=resolution,
        )
    surface += float(stage.get("roughness_residual_m", 0.0)) * rough_unit
    surface += float(stage.get("local_irregularity_m", 0.0)) * local_unit
    reference_height = 0.0
    if reference_surface is not None:
        reference_height = float(
            sample_height_numpy(
                reference_surface,
                np.array([[center_xy[0], center_xy[1]]], dtype=np.float32),
                size=size,
            )[0]
        )
    stage_height = float(
        sample_height_numpy(
            surface,
            np.array([[center_xy[0], center_xy[1]]], dtype=np.float32),
            size=size,
        )[0]
    )
    surface -= stage_height - reference_height

    return surface.astype(np.float32)


def _patch_blend_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    rng: np.random.Generator,
    resolution: tuple[int, int],
    center_xy: tuple[float, float],
    patch_radius_m: float,
    blend_radius_m: float,
) -> np.ndarray:
    """Build a smooth, noise-perturbed radial mask for curriculum patch blending.

    Args:
        grid_x (np.ndarray): World X grid.
        grid_y (np.ndarray): World Y grid.
        rng (np.random.Generator): RNG for mask boundary noise.
        resolution (tuple[int, int]): Grid resolution.
        center_xy (tuple[float, float]): Patch center in meters.
        patch_radius_m (float): Full-strength inner radius.
        blend_radius_m (float): Outer radius where the mask reaches zero.

    Returns:
        np.ndarray: Weights in ``[0, 1]`` with shape matching ``grid_x``.
    """
    if patch_radius_m <= 0.0 or blend_radius_m <= patch_radius_m:
        return np.zeros_like(grid_x, dtype=np.float32)

    center_x, center_y = center_xy
    distance = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    mask_noise = _fractal_noise(
        resolution,
        rng,
        base_sigma=max(float(max(resolution)) / 10.0, 2.0),
        octaves=3,
        persistence=0.55,
        lacunarity=1.85,
    )
    mask_noise = _smooth_field(
        mask_noise, sigma=max(float(max(resolution)) / 36.0, 1.0)
    )
    patch_radius = patch_radius_m * np.clip(1.0 + 0.35 * mask_noise, 0.55, 1.55)
    blend_radius = blend_radius_m * np.clip(1.0 + 0.25 * mask_noise, 0.7, 1.45)
    mask = np.zeros_like(distance, dtype=np.float32)
    inside = distance <= patch_radius
    mask[inside] = 1.0
    transition = (distance - patch_radius) / np.maximum(
        blend_radius - patch_radius,
        1.0e-6,
    )
    band = (distance > patch_radius) & (distance < blend_radius)
    mask[band] = 1.0 - _smoothstep(transition[band])
    return mask


def generate_random_terrain_heightmap(
    size: tuple[float, float],
    resolution: tuple[int, int],
    height_range: tuple[float, float],
    flat_radius: float,
    num_functions: int = 8,
    seed: int | None = None,
    generator_config: Mapping[str, Any] | None = None,
    terrain_curriculum: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Generate a procedural heightmap for irregular terrain (curriculum optional).

    Args:
        size (tuple[float, float]): Terrain width and length in meters.
        resolution (tuple[int, int]): Discrete grid resolution.
        height_range (tuple[float, float]): Final min/max height in meters.
        flat_radius (float): Inner flat region radius for the base field.
        num_functions (int): Base-field complexity knob.
        seed (int | None): RNG seed for reproducibility.
        generator_config (Mapping[str, Any] | None): ``terrain.generator`` YAML block.
        terrain_curriculum (Mapping[str, Any] | None): Optional curriculum spec.

    Returns:
        np.ndarray: Final heightmap ``float32`` with shape ``resolution``.
    """
    rng = np.random.default_rng(seed)
    settings = _resolve_generator_settings(generator_config)
    terrace_defaults = _resolve_terrace_settings(
        generator_config.get("terrace", {})
        if isinstance(generator_config, Mapping)
        else None
    )

    min_height, max_height = height_range
    width, length = size
    width_res, length_res = resolution

    x = np.linspace(-width / 2, width / 2, width_res, dtype=np.float32)
    y = np.linspace(-length / 2, length / 2, length_res, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    rough_unit = _generate_terrain_base_unit_field(
        grid_x=grid_x,
        grid_y=grid_y,
        size=size,
        resolution=resolution,
        num_functions=num_functions,
        rng=rng,
        settings=settings,
    )
    local_unit = _fractal_noise(
        resolution,
        rng,
        base_sigma=max(float(max(resolution)) / 24.0, 1.0),
        octaves=2,
        persistence=0.52,
        lacunarity=1.9,
    )
    base_surface = _normalize_unit(rough_unit)
    base_surface = base_surface * (max_height - min_height) + min_height

    if isinstance(terrain_curriculum, Mapping) and terrain_curriculum.get(
        "enabled", False
    ):
        stages = list(terrain_curriculum.get("stages", []))
        if not stages:
            raise ValueError(
                "Resolved terrain curriculum is enabled but contains no stages."
            )

        global_stage = stages[-1]
        heightmap = _stage_surface(
            stage=global_stage,
            size=size,
            resolution=resolution,
            grid_x=grid_x,
            grid_y=grid_y,
            base_surface=base_surface,
            rough_unit=rough_unit,
            local_unit=local_unit,
            terrace_defaults=terrace_defaults,
            rng=rng,
        )
        for stage in stages[:-1]:
            patch_mask = _patch_blend_mask(
                grid_x,
                grid_y,
                rng=rng,
                resolution=resolution,
                center_xy=tuple(stage["spawn_xy"]),
                patch_radius_m=float(stage["patch_radius_m"]),
                blend_radius_m=float(stage["blend_radius_m"]),
            )
            if not np.any(patch_mask > 0.0):
                continue

            patch_surface = _stage_surface(
                stage=stage,
                size=size,
                resolution=resolution,
                grid_x=grid_x,
                grid_y=grid_y,
                base_surface=base_surface,
                rough_unit=rough_unit,
                local_unit=local_unit,
                terrace_defaults=terrace_defaults,
                rng=rng,
                reference_surface=heightmap,
            )
            heightmap = patch_mask * patch_surface + (1.0 - patch_mask) * heightmap

        heightmap = heightmap.astype(np.float32)
    elif terrace_defaults["enabled"]:
        surface = {
            "step_height_m": float(terrace_defaults["step_height_m"]),
            "terrace_width_m": float(terrace_defaults["terrace_width_m"]),
            "edge_smoothing": float(terrace_defaults["edge_smoothing"]),
            "global_height_scale": 1.0,
            "local_irregularity_m": float(
                terrace_defaults["local_noise_weight"]
                * terrace_defaults["step_height_m"]
            ),
            "roughness_residual_m": 0.08 * (max_height - min_height),
            "spawn_xy": (0.0, 0.0),
        }
        heightmap = _stage_surface(
            stage=surface,
            size=size,
            resolution=resolution,
            grid_x=grid_x,
            grid_y=grid_y,
            base_surface=base_surface,
            rough_unit=rough_unit,
            local_unit=local_unit,
            terrace_defaults=terrace_defaults,
            rng=rng,
        )
    else:
        heightmap = base_surface.copy()

        if flat_radius > 0.0:
            distance = np.sqrt(grid_x**2 + grid_y**2)
            inner_radius = flat_radius * 0.7
            transition_span = max(flat_radius - inner_radius, 1.0e-6)
            transition = np.clip(
                (distance - inner_radius) / transition_span,
                0.0,
                1.0,
            )
            blend = 0.5 - 0.5 * np.cos(np.pi * transition)
            heightmap *= blend.astype(np.float32)

    center_height = float(heightmap[width_res // 2, length_res // 2])
    heightmap = (heightmap - center_height).astype(np.float32)
    heightmap = np.clip(heightmap, min_height, max_height).astype(np.float32)

    return heightmap
