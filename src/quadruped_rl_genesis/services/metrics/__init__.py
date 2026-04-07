"""Policy evaluation bundles, summaries, and task-aware metric comparisons."""

from __future__ import annotations

from typing import Any

from .protocol import build_evaluation_protocol
from .runs import (
    RandomPolicy,
    evaluate_policy_metrics,
    evaluate_policy_variants,
)
from .scoring import (
    enrich_task_metrics,
    is_better_evaluation_bundle,
    is_better_task_metrics,
)

__all__ = [
    "RandomPolicy",
    "build_evaluation_protocol",
    "enrich_task_metrics",
    "evaluate_policy_metrics",
    "evaluate_policy_variants",
    "is_better_evaluation_bundle",
    "is_better_task_metrics",
    "summarize_evaluation_bundle",
    "summarize_metrics",
]


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, serialization-friendly summary of one metric payload.

    Args:
        metrics (dict[str, Any]): Full aggregated metrics dictionary.

    Returns:
        dict[str, Any]: Reduced summary keeping the main scalar metrics and
            article-aligned values.
    """
    article_metrics = metrics.get("article_metrics", {})

    return {
        "mean_reward": float(metrics.get("mean_reward", 0.0)),
        "std_reward": float(metrics.get("std_reward", 0.0)),
        "task_score": float(metrics.get("task_score", 0.0)),
        "selection_score": float(metrics.get("selection_score", 0.0)),
        "task_eligible": bool(metrics.get("task_eligible", False)),
        "scientific_gates_passed": bool(metrics.get("scientific_gates_passed", True)),
        "scientific_gates": dict(metrics.get("scientific_gates", {})),
        "article_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in article_metrics.items()
        },
        "termination_rates": dict(metrics.get("termination_rates", {})),
        "reward_terms": dict(metrics.get("reward_terms", {})),
        "task_metrics": dict(metrics.get("task_metrics", {})),
        "stagnation_snapshots": list(metrics.get("stagnation_snapshots", []))[:5],
    }


def summarize_evaluation_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Create a compact summary of an evaluation bundle.

    Args:
        bundle (dict[str, Any]): Full evaluation bundle returned by
            ``evaluate_policy_variants``.

    Returns:
        dict[str, Any]: Reduced evaluation summary suitable for logs and JSON
            reports.
    """
    summary = {
        "mean_reward": float(bundle["mean_reward"]),
        "std_reward": float(bundle["std_reward"]),
        "task_score": float(bundle["task_score"]),
        "selection_score": float(bundle["selection_score"]),
        "task_eligible": bool(bundle["task_eligible"]),
        "metrics_deterministic": summarize_metrics(bundle["metrics_deterministic"]),
        "evaluation_protocol": dict(bundle["evaluation_protocol"]),
    }

    if bundle.get("metrics_stochastic") is not None:
        summary["metrics_stochastic"] = summarize_metrics(bundle["metrics_stochastic"])

    return summary
