"""VecNormalize helpers shared by train, eval, and visualization pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.vec_env import VecNormalize


def get_normalization_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Return the normalization config block from an experiment config.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        dict[str, Any]: Normalization config, or ``{}`` when omitted.
    """
    training_config = experiment_config.get("training", {})
    normalization_config = training_config.get("normalization", {})

    return normalization_config if isinstance(normalization_config, dict) else {}


def normalization_enabled(experiment_config: dict[str, Any]) -> bool:
    """Check whether VecNormalize should wrap the environment.

    Args:
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        bool: ``True`` when normalization is enabled for the experiment.
    """
    normalization_config = get_normalization_config(experiment_config)

    return bool(normalization_config.get("enabled", False))


def wrap_with_vecnormalize(
    env,
    *,
    experiment_config: dict[str, Any],
    for_training: bool,
    vecnormalize_path: str | Path | None = None,
    norm_reward: bool | None = None,
):
    """Wrap one vectorized environment with VecNormalize.

    Args:
        env: Vectorized environment to wrap.
        experiment_config (dict[str, Any]): Resolved experiment configuration.
        for_training (bool): Whether the wrapped environment will be used for
            training updates.
        vecnormalize_path (str | Path | None, optional): Existing stats file to
            load before using the environment.
        norm_reward (bool | None, optional): Optional override for reward
            normalization.

    Returns:
        Any: ``env`` wrapped in ``VecNormalize`` when enabled, otherwise the
            original environment.

    Raises:
        FileNotFoundError: If normalization is enabled for eval/visualization
            and a requested stats path does not exist.
    """
    normalization_config = get_normalization_config(experiment_config)
    if not bool(normalization_config.get("enabled", False)):
        return env

    stats_path = Path(vecnormalize_path) if vecnormalize_path is not None else None
    resolved_norm_reward = (
        bool(norm_reward)
        if norm_reward is not None
        else bool(normalization_config.get("norm_reward", True) and for_training)
    )

    if stats_path is not None:
        if not stats_path.exists():
            if not for_training:
                raise FileNotFoundError(
                    f"VecNormalize stats file was not found at {stats_path}."
                )
        else:
            normalized_env = VecNormalize.load(str(stats_path), env)
            normalized_env.training = bool(for_training)
            normalized_env.norm_reward = bool(resolved_norm_reward)
            return normalized_env

    normalized_env = VecNormalize(
        env,
        training=bool(for_training),
        norm_obs=bool(normalization_config.get("norm_obs", True)),
        norm_reward=bool(resolved_norm_reward),
        clip_obs=float(normalization_config.get("clip_obs", 10.0)),
        clip_reward=float(normalization_config.get("clip_reward", 10.0)),
        gamma=float(normalization_config.get("gamma", 0.99)),
        epsilon=float(normalization_config.get("epsilon", 1.0e-8)),
    )
    normalized_env.training = bool(for_training)
    normalized_env.norm_reward = bool(resolved_norm_reward)

    return normalized_env
