"""Episode aggregation, random-policy baseline, and policy evaluation loops."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .protocol import (
    _describe,
    _max_episode_length,
    build_evaluation_protocol,
)
from .scoring import (
    _ARTICLE_SCORE_KEYS,
    enrich_task_metrics,
)


def _extract_episode_record(
    info: dict[str, Any],
    episode_index: int,
    max_episode_length: int,
) -> dict[str, Any]:
    """Extract one normalized episode record from SB3/Gym info dictionaries.

    Args:
        info (dict[str, Any]): Episode info dictionary emitted by the
            environment.
        episode_index (int): Sequential evaluation episode index.
        max_episode_length (int): Configured maximum number of steps in an
            episode.

    Returns:
        dict[str, Any]: Normalized episode record with scalar reward terms,
            task metrics, and termination metadata.
    """
    episode_payload = info.get("episode", {})
    reward_terms: dict[str, float] = {}
    task_metrics: dict[str, float] = {}

    for key, value in episode_payload.items():
        if key.startswith("rew_"):
            reward_terms[key.removeprefix("rew_")] = float(value)
        elif key.startswith("metric_"):
            task_metrics[key.removeprefix("metric_")] = float(value)

    for key, value in info.items():
        if key.startswith("rew_") and key.removeprefix("rew_") not in reward_terms:
            reward_terms[key.removeprefix("rew_")] = float(value)
        elif (
            key.startswith("metric_")
            and key.removeprefix("metric_") not in task_metrics
        ):
            task_metrics[key.removeprefix("metric_")] = float(value)

    length = int(episode_payload.get("l", 0))
    termination_reason = str(
        episode_payload.get("termination_reason", info.get("termination_reason", ""))
    )

    return {
        "episode_index": episode_index,
        "reward": float(episode_payload.get("r", 0.0)),
        "length": length,
        "survival_fraction": float(length / max(max_episode_length, 1)),
        "termination_reason": termination_reason or "unknown",
        "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
        "reward_terms": reward_terms,
        "task_metrics": task_metrics,
        "stagnation_snapshot": info.get(
            "stagnation_snapshot",
            episode_payload.get("stagnation_snapshot"),
        ),
    }


def _aggregate_episode_records(
    episode_records: list[dict[str, Any]],
    experiment_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-episode records into project-level evaluation metrics.

    Args:
        episode_records (list[dict[str, Any]]): Episode records collected during
            evaluation.
        experiment_config (dict | None): Experiment config for task_score mode.

    Returns:
        dict[str, Any]: Aggregated metrics enriched with task-selection fields.
    """
    rewards = [float(record["reward"]) for record in episode_records]
    lengths = [float(record["length"]) for record in episode_records]
    survivals = [float(record["survival_fraction"]) for record in episode_records]

    termination_counts = Counter(
        str(record["termination_reason"]) for record in episode_records
    )
    total_episodes = max(len(episode_records), 1)
    termination_rates = {
        reason: float(count / total_episodes)
        for reason, count in sorted(termination_counts.items())
    }

    reward_term_values: dict[str, list[float]] = defaultdict(list)
    task_metric_values: dict[str, list[float]] = defaultdict(list)
    stagnation_snapshots: list[dict[str, Any]] = []

    for record in episode_records:
        for name, value in record["reward_terms"].items():
            reward_term_values[name].append(float(value))

        for name, value in record["task_metrics"].items():
            task_metric_values[name].append(float(value))

        snapshot = record.get("stagnation_snapshot")
        if snapshot is not None:
            stagnation_snapshots.append(dict(snapshot))

    reward_term_stats = {
        name: _describe(values) for name, values in sorted(reward_term_values.items())
    }
    task_metric_stats = {
        name: _describe(values) for name, values in sorted(task_metric_values.items())
    }

    article_metrics = {
        "mean_episode_length": _describe(lengths)["mean"],
        "mean_survival_fraction": _describe(survivals)["mean"],
        "success_rate": float(termination_counts.get("success", 0) / total_episodes),
        "timeout_rate": float(termination_counts.get("timeout", 0) / total_episodes),
        "fall_rate": float(termination_counts.get("fall", 0) / total_episodes),
        "posture_rate": float(termination_counts.get("posture", 0) / total_episodes),
        "curve_deviation_rate": float(
            termination_counts.get("curve_deviation", 0) / total_episodes
        ),
        "stagnation_rate": float(
            termination_counts.get("stagnation", 0) / total_episodes
        ),
    }

    for name, stats in task_metric_stats.items():
        if name.startswith("mean_") or name.endswith(("_rate", "_ratio")):
            article_metrics[name] = stats["mean"]
        else:
            article_metrics[f"mean_{name}"] = stats["mean"]

    for key in _ARTICLE_SCORE_KEYS:
        if key not in article_metrics:
            article_metrics[key] = 0.0

    return enrich_task_metrics(
        {
            "mean_reward": _describe(rewards)["mean"],
            "std_reward": _describe(rewards)["std"],
            "reward": _describe(rewards),
            "episode_length": _describe(lengths),
            "survival_fraction": _describe(survivals),
            "termination_counts": dict(sorted(termination_counts.items())),
            "termination_rates": termination_rates,
            "reward_terms": reward_term_stats,
            "task_metrics": task_metric_stats,
            "article_metrics": article_metrics,
            "stagnation_snapshots": stagnation_snapshots,
            "episode_records": episode_records,
        },
        experiment_config=experiment_config,
    )


class RandomPolicy:
    """Uniform random actions in the environment's action space (baseline).

    Implements the same ``predict`` signature as Stable-Baselines3 models so it
    can be passed to :func:`evaluate_policy_metrics` and
    :func:`evaluate_policy_variants`.
    """

    def __init__(self, action_space: Any, rng: np.random.Generator) -> None:
        """Bind the gym action space and NumPy RNG used for sampling.

        Args:
            action_space (Any): Gymnasium / Gym action space with ``low`` and ``high``.
            rng (np.random.Generator): Generator used for uniform action draws.
        """
        self.action_space = action_space
        self.rng = rng

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        state: Any = None,
        episode_start: Any = None,
    ) -> tuple[np.ndarray, None]:
        """Return a batch of i.i.d. uniform actions (SB3-compatible API).

        Args:
            observations (Any): Ignored; batch size is inferred from shape.
            deterministic (bool): Ignored; sampling is always stochastic here.
            state (Any): Unused SB3 RNN state slot.
            episode_start (Any): Unused SB3 episode-start flag.

        Returns:
            tuple[np.ndarray, None]: Actions shaped ``(batch, *action_space.shape)`` and
                ``None`` for the unused state.
        """
        del deterministic, state, episode_start
        batch_size = int(np.asarray(observations).shape[0])
        space = self.action_space
        low = np.broadcast_to(
            np.asarray(space.low, dtype=np.float64),
            space.shape,
        )
        high = np.broadcast_to(
            np.asarray(space.high, dtype=np.float64),
            space.shape,
        )
        shape = (batch_size, *tuple(space.shape))
        actions = self.rng.uniform(low, high, size=shape).astype(np.float32)
        return actions, None


def evaluate_policy_metrics(
    model: Any,
    env: Any,
    *,
    n_eval_episodes: int,
    deterministic: bool,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one policy mode and return aggregate navigation metrics.

    Args:
        model (Any): Policy or SB3 model implementing ``predict``.
        env (Any): Vectorized environment used for evaluation.
        n_eval_episodes (int): Number of completed episodes to collect.
        deterministic (bool): Whether ``model.predict`` should act
            deterministically.
        experiment_config (dict[str, Any]): Resolved experiment configuration.

    Returns:
        dict[str, Any]: Aggregated evaluation metrics for the requested policy
            mode.
    """
    max_episode_length = _max_episode_length(experiment_config)
    observations = env.reset()
    episode_records: list[dict[str, Any]] = []

    while len(episode_records) < n_eval_episodes:
        actions, _state = model.predict(observations, deterministic=deterministic)
        observations, _rewards, dones, infos = env.step(actions)

        for env_index, done in enumerate(dones):
            if not bool(done):
                continue

            if "episode" not in infos[env_index]:
                continue

            episode_records.append(
                _extract_episode_record(
                    infos[env_index],
                    episode_index=len(episode_records),
                    max_episode_length=max_episode_length,
                )
            )
            if len(episode_records) >= n_eval_episodes:
                break

    aggregated = _aggregate_episode_records(
        episode_records, experiment_config=experiment_config
    )
    aggregated["episodes"] = int(n_eval_episodes)
    aggregated["evaluation_protocol"] = build_evaluation_protocol(experiment_config)
    aggregated["deterministic"] = bool(deterministic)

    return aggregated


def evaluate_policy_variants(
    model: Any,
    env: Any,
    *,
    n_eval_episodes: int,
    experiment_config: dict[str, Any],
    include_stochastic: bool,
    deterministic_eval: bool = True,
) -> dict[str, Any]:
    """Evaluate deterministic and optional stochastic variants of one policy.

    Args:
        model (Any): Policy or SB3 model implementing ``predict``.
        env (Any): Vectorized environment used for evaluation.
        n_eval_episodes (int): Number of completed episodes per evaluation pass.
        experiment_config (dict[str, Any]): Resolved experiment configuration.
        include_stochastic (bool): Whether to run an additional stochastic pass.
        deterministic_eval (bool, optional): Deterministic flag for the primary
            evaluation pass.

    Returns:
        dict[str, Any]: Combined evaluation bundle with deterministic metrics and
            optional stochastic metrics.
    """
    metrics_deterministic = evaluate_policy_metrics(
        model=model,
        env=env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic_eval,
        experiment_config=experiment_config,
    )

    metrics_stochastic = None
    if include_stochastic:
        metrics_stochastic = evaluate_policy_metrics(
            model=model,
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
            experiment_config=experiment_config,
        )

    return {
        "episodes": int(n_eval_episodes),
        "mean_reward": float(metrics_deterministic["mean_reward"]),
        "std_reward": float(metrics_deterministic["std_reward"]),
        "task_score": float(metrics_deterministic["task_score"]),
        "selection_score": float(metrics_deterministic["selection_score"]),
        "task_eligible": bool(metrics_deterministic["task_eligible"]),
        "metrics_deterministic": metrics_deterministic,
        "metrics_stochastic": metrics_stochastic,
        "evaluation_protocol": build_evaluation_protocol(experiment_config),
    }
