"""Task scoring, enrichment, and selection comparisons for evaluation metrics."""

from __future__ import annotations

from typing import Any

from .protocol import _scientific_gates

_ARTICLE_SCORE_KEYS = (
    "success_rate",
    "timeout_rate",
    "fall_rate",
    "curve_deviation_rate",
    "stagnation_rate",
    "mean_final_goal_distance",
    "mean_final_goal_yaw_error_deg",
    "mean_stop_speed_at_goal",
    "mean_final_yaw_rate_deg_s",
    "mean_cross_track_error",
    "mean_arc_progress_speed",
    "reverse_motion_ratio",
    "lateral_motion_ratio",
    "airborne_ratio",
    "trot_contact_score",
)


def _task_score_article(article_metrics: dict[str, Any]) -> float:
    """Score using article-aligned weights for paper reproducibility.

    Args:
        article_metrics (dict[str, Any]): Aggregated evaluation metrics.

    Returns:
        float: Scalar task score.
    """
    return (
        120.0 * float(article_metrics.get("success_rate", 0.0))
        + 16.0 * float(article_metrics.get("mean_arc_progress_speed", 0.0))
        - 8.0 * float(article_metrics.get("mean_cross_track_error", 0.0))
        - 10.0 * float(article_metrics.get("reverse_motion_ratio", 0.0))
        - 8.0 * float(article_metrics.get("lateral_motion_ratio", 0.0))
        - 8.0 * float(article_metrics.get("airborne_ratio", 0.0))
        - 0.18 * float(article_metrics.get("mean_final_goal_distance", 0.0))
        - 0.18 * float(article_metrics.get("mean_final_goal_yaw_error_deg", 0.0))
        - 10.0 * float(article_metrics.get("fall_rate", 0.0))
        - 6.0 * float(article_metrics.get("curve_deviation_rate", 0.0))
        - 4.0 * float(article_metrics.get("stagnation_rate", 0.0))
        - 4.0 * float(article_metrics.get("timeout_rate", 0.0))
        - 6.0 * float(article_metrics.get("mean_stop_speed_at_goal", 0.0))
        + 2.0 * float(article_metrics.get("trot_contact_score", 0.0))
    )


def _task_score_exploration(article_metrics: dict[str, Any]) -> float:
    """Score favoring locomotion learning; rewards movement, less punitive on falls.

    Prioritizes learning to walk over avoiding falls. Encourages exploration.

    Args:
        article_metrics (dict[str, Any]): Aggregated evaluation metrics.

    Returns:
        float: Scalar task score.
    """
    return (
        80.0 * float(article_metrics.get("success_rate", 0.0))
        + 12.0 * float(article_metrics.get("mean_arc_progress_speed", 0.0))
        - 6.0 * float(article_metrics.get("mean_cross_track_error", 0.0))
        - 8.0 * float(article_metrics.get("reverse_motion_ratio", 0.0))
        - 6.0 * float(article_metrics.get("lateral_motion_ratio", 0.0))
        - 6.0 * float(article_metrics.get("airborne_ratio", 0.0))
        - 0.12 * float(article_metrics.get("mean_final_goal_distance", 0.0))
        - 0.10 * float(article_metrics.get("mean_final_goal_yaw_error_deg", 0.0))
        - 6.0 * float(article_metrics.get("fall_rate", 0.0))
        - 4.0 * float(article_metrics.get("curve_deviation_rate", 0.0))
        - 3.0 * float(article_metrics.get("stagnation_rate", 0.0))
        - 2.0 * float(article_metrics.get("timeout_rate", 0.0))
        - 4.0 * float(article_metrics.get("mean_stop_speed_at_goal", 0.0))
        + 3.0 * float(article_metrics.get("trot_contact_score", 0.0))
    )


def _task_score_from_article_metrics(
    article_metrics: dict[str, Any],
    mode: str = "article",
) -> float:
    """Score task performance using the configured weighted metric.

    Modes:
        article: Paper-aligned weights. Best for final evaluation.
        exploration: Favors learning to walk; rewards movement, less punitive on falls.

    Args:
        article_metrics (dict[str, Any]): Aggregated evaluation metrics.
        mode (str): "article" or "exploration".

    Returns:
        float: Scalar task score.
    """
    if mode == "exploration":
        return _task_score_exploration(article_metrics)

    return _task_score_article(article_metrics)


def _task_eligible_from_article_metrics(article_metrics: dict[str, Any]) -> bool:
    """Check whether metrics qualify for the primary selection rule.

    Eligible = at least one success, or some positive arc-progress speed.

    Args:
        article_metrics (dict[str, Any]): Aggregated evaluation metrics in the
            article-friendly format.

    Returns:
        bool: ``True`` when the run is considered eligible for primary ranking.
    """
    success_rate = float(article_metrics.get("success_rate", 0.0))
    arc_progress_speed = float(article_metrics.get("mean_arc_progress_speed", 0.0))
    cross_track = float(article_metrics.get("mean_cross_track_error", 0.0))

    return success_rate > 0.0 or (
        arc_progress_speed > 0.0 and cross_track < float("inf")
    )


def enrich_task_metrics(
    metrics: dict[str, Any],
    experiment_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Augment raw metrics with selection-oriented task scoring fields.

    Args:
        metrics (dict[str, Any]): Aggregated evaluation metrics.
        experiment_config (dict | None): Experiment config for task_score mode.

    Returns:
        dict[str, Any]: Copy of ``metrics`` enriched with task eligibility,
            score, selection score, and fallback ranking score.
    """
    eval_cfg = (experiment_config or {}).get("evaluation", {})
    if isinstance(eval_cfg, dict):
        task_score_mode = str(eval_cfg.get("task_score_mode", "article"))
    else:
        task_score_mode = "article"

    article_metrics = dict(metrics.get("article_metrics", {}))
    task_score = _task_score_from_article_metrics(article_metrics, mode=task_score_mode)
    scientific_gates = _scientific_gates(
        article_metrics, experiment_config=experiment_config
    )
    gates_passed = bool(scientific_gates["all_passed"])
    task_eligible = (
        _task_eligible_from_article_metrics(article_metrics) and gates_passed
    )
    fallback_score = (
        10.0 * float(article_metrics.get("mean_arc_progress_speed", 0.0))
        - 2.0 * float(article_metrics.get("mean_cross_track_error", 0.0))
        - float(article_metrics.get("reverse_motion_ratio", 0.0))
        - float(article_metrics.get("lateral_motion_ratio", 0.0))
        - float(article_metrics.get("airborne_ratio", 0.0))
    )
    selection_score = task_score if task_eligible else -1000.0 + fallback_score
    if not gates_passed:
        selection_score -= 1000.0

    enriched = dict(metrics)
    enriched["article_metrics"] = article_metrics
    enriched["task_score"] = float(task_score)
    enriched["task_eligible"] = bool(task_eligible)
    enriched["selection_score"] = float(selection_score)
    enriched["selection_fallback_score"] = float(fallback_score)
    enriched["scientific_gates"] = scientific_gates
    enriched["scientific_gates_passed"] = bool(gates_passed)

    return enriched


def is_better_task_metrics(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
) -> bool:
    """Compare two aggregated metric payloads using the task selection rule.

    Args:
        candidate_metrics (dict[str, Any]): Metrics for the new candidate.
        incumbent_metrics (dict[str, Any] | None): Metrics for the current best
            candidate, or ``None`` when there is no incumbent.

    Returns:
        bool: ``True`` when the candidate should replace the incumbent.
    """
    if incumbent_metrics is None:
        return True

    candidate_article = candidate_metrics.get("article_metrics", {})
    incumbent_article = incumbent_metrics.get("article_metrics", {})
    candidate_rank = (
        1 if bool(candidate_metrics.get("task_eligible", False)) else 0,
        1 if bool(candidate_metrics.get("scientific_gates_passed", False)) else 0,
        float(candidate_article.get("success_rate", 0.0)),
        float(candidate_article.get("mean_arc_progress_speed", 0.0)),
        -float(candidate_article.get("mean_cross_track_error", 0.0)),
        -float(candidate_article.get("reverse_motion_ratio", 0.0)),
        -float(candidate_article.get("lateral_motion_ratio", 0.0)),
        -float(candidate_article.get("airborne_ratio", 0.0)),
        -float(candidate_article.get("mean_final_goal_distance", 0.0)),
        -float(candidate_article.get("mean_final_goal_yaw_error_deg", 0.0)),
        float(candidate_metrics.get("selection_score", float("-inf"))),
        float(candidate_metrics.get("mean_reward", float("-inf"))),
    )
    incumbent_rank = (
        1 if bool(incumbent_metrics.get("task_eligible", False)) else 0,
        1 if bool(incumbent_metrics.get("scientific_gates_passed", False)) else 0,
        float(incumbent_article.get("success_rate", 0.0)),
        float(incumbent_article.get("mean_arc_progress_speed", 0.0)),
        -float(incumbent_article.get("mean_cross_track_error", 0.0)),
        -float(incumbent_article.get("reverse_motion_ratio", 0.0)),
        -float(incumbent_article.get("lateral_motion_ratio", 0.0)),
        -float(incumbent_article.get("airborne_ratio", 0.0)),
        -float(incumbent_article.get("mean_final_goal_distance", 0.0)),
        -float(incumbent_article.get("mean_final_goal_yaw_error_deg", 0.0)),
        float(incumbent_metrics.get("selection_score", float("-inf"))),
        float(incumbent_metrics.get("mean_reward", float("-inf"))),
    )

    return candidate_rank > incumbent_rank


def is_better_evaluation_bundle(
    candidate_bundle: dict[str, Any],
    incumbent_bundle: dict[str, Any] | None,
) -> bool:
    """Compare two evaluation bundles using their deterministic task metrics.

    Args:
        candidate_bundle (dict[str, Any]): Candidate evaluation bundle.
        incumbent_bundle (dict[str, Any] | None): Current best bundle.

    Returns:
        bool: ``True`` when the candidate bundle is preferred.
    """
    if incumbent_bundle is None:
        return True

    return is_better_task_metrics(
        candidate_bundle["metrics_deterministic"],
        incumbent_bundle["metrics_deterministic"],
    )
