"""Monitor layout resolution and JSON snapshot loading for live metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quadruped_rl_genesis.services.artifacts import ArtifactStore
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_json


def resolve_monitor_layout(
    settings: AppSettings,
    *,
    output_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Path | None]:
    """Resolve the filesystem roots used by the monitor workflows.

    Args:
        settings (AppSettings): Global application settings.
        output_root (str | Path | None, optional): Benchmark output root that
            contains ``artifacts/`` and ``reports/``.
        artifacts_root (str | Path | None, optional): Explicit artifacts root
            override.

    Returns:
        dict[str, Path | None]: Mapping with resolved artifacts and optional
            benchmark report locations.

    Raises:
        ValueError: If both ``output_root`` and ``artifacts_root`` are given.
    """
    if output_root is not None and artifacts_root is not None:
        raise ValueError(
            "Use either output_root or artifacts_root when monitoring, not both."
        )

    if output_root is not None:
        benchmark_root = _resolve_path(settings.project_root, output_root)
        return {
            "artifacts_root": benchmark_root / "artifacts",
            "benchmark_root": benchmark_root,
            "benchmark_summary_path": benchmark_root
            / "reports"
            / "benchmark_summary.json",
        }

    if artifacts_root is not None:
        resolved_artifacts_root = _resolve_path(settings.project_root, artifacts_root)
        return {
            "artifacts_root": resolved_artifacts_root,
            "benchmark_root": None,
            "benchmark_summary_path": None,
        }

    return {
        "artifacts_root": settings.artifacts_root,
        "benchmark_root": None,
        "benchmark_summary_path": None,
    }


def build_monitor_snapshot(
    *,
    settings: AppSettings,
    output_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    experiment_name: str | None = None,
    algorithm: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Collect a lightweight status snapshot from the artifact tree.

    Args:
        settings (AppSettings): Global application settings.
        output_root (str | Path | None, optional): Benchmark output root.
        artifacts_root (str | Path | None, optional): Explicit artifacts root.
        experiment_name (str | None, optional): Optional experiment filter.
        algorithm (str | None, optional): Optional algorithm filter.
        limit (int, optional): Maximum number of experiment rows to keep.

    Returns:
        dict[str, Any]: Snapshot ready for serialization or formatted output.
    """
    layout = resolve_monitor_layout(
        settings,
        output_root=output_root,
        artifacts_root=artifacts_root,
    )
    resolved_artifacts_root = layout["artifacts_root"]
    assert resolved_artifacts_root is not None

    rows: list[dict[str, Any]] = []
    experiments_root = resolved_artifacts_root / "experiments"

    if experiments_root.exists():
        for experiment_dir in sorted(
            path for path in experiments_root.iterdir() if path.is_dir()
        ):
            if experiment_name is not None and experiment_dir.name != experiment_name:
                continue

            for algorithm_dir in sorted(
                path for path in experiment_dir.iterdir() if path.is_dir()
            ):
                if algorithm is not None and algorithm_dir.name != algorithm:
                    continue

                rows.append(
                    _build_run_row(
                        artifacts_root=resolved_artifacts_root,
                        experiment_name=experiment_dir.name,
                        algorithm=algorithm_dir.name,
                        algorithm_root=algorithm_dir,
                    )
                )

    snapshot = {
        "artifacts_root": str(resolved_artifacts_root),
        "benchmark_root": (
            str(layout["benchmark_root"])
            if layout["benchmark_root"] is not None
            else None
        ),
        "benchmark_summary": _read_benchmark_summary(layout["benchmark_summary_path"]),
        "runs": rows[: max(limit, 1)],
    }

    return snapshot


def format_monitor_snapshot(snapshot: dict[str, Any]) -> str:
    """Render a monitor snapshot into terminal-friendly text.

    Args:
        snapshot (dict[str, Any]): Snapshot produced by
            :func:`build_monitor_snapshot`.

    Returns:
        str: Human-readable terminal summary.
    """
    lines = [f"Artifacts root: {snapshot['artifacts_root']}"]

    benchmark_summary = snapshot.get("benchmark_summary")
    if benchmark_summary is not None:
        lines.append(
            "Benchmark summary: "
            f"tuning={benchmark_summary['tuning_count']} "
            f"training={benchmark_summary['training_count']} "
            f"evaluation={benchmark_summary['evaluation_count']} "
            f"videos={benchmark_summary['video_count']}"
        )
        lines.append(f"Benchmark report: {benchmark_summary['path']}")

    runs = snapshot.get("runs", [])
    if not runs:
        lines.append("No experiment artifacts found for the requested filters.")
        return "\n".join(lines)

    for row in runs:
        lines.append("")
        lines.append(f"[{row['experiment_name']}/{row['algorithm']}]")
        lines.append(f"latest run: {row['run_id'] or 'n/a'}")
        if row["run_root"] is not None:
            lines.append(f"run root: {row['run_root']}")
        if row["tensorboard_logdir"] is not None:
            lines.append(f"tensorboard: {row['tensorboard_logdir']}")

        live_eval = row.get("live_evaluation")
        if live_eval is not None:
            lines.append(
                "last eval: "
                f"step={_format_int(live_eval.get('timesteps'))} "
                f"reward={_format_float(live_eval.get('mean_reward'))} "
                f"task={_format_float(live_eval.get('task_score'))} "
                f"selection={_format_float(live_eval.get('selection_score'))}"
            )

        training_summary = row.get("training_summary")
        if training_summary is not None:
            lines.append(
                "train summary: "
                f"best_task={_format_float(training_summary.get('best_task_score'))} "
                f"best_selection={_format_float(training_summary.get('best_selection_score'))} "
                f"final_task={_format_float(training_summary.get('final_task_score'))} "
                f"final_selection={_format_float(training_summary.get('final_selection_score'))}"
            )

        final_evaluation = row.get("final_evaluation")
        if final_evaluation is not None:
            lines.append(
                "final eval: "
                f"episodes={_format_int(final_evaluation.get('episodes'))} "
                f"reward={_format_float(final_evaluation.get('mean_reward'))} "
                f"task={_format_float(final_evaluation.get('task_score'))} "
                f"selection={_format_float(final_evaluation.get('selection_score'))}"
            )

        optuna_summary = row.get("optuna_summary")
        if optuna_summary is not None:
            lines.append(
                "optuna summary: "
                f"completed={_format_int(optuna_summary.get('completed_trials'))}/"
                f"{_format_int(optuna_summary.get('target_trials'))} "
                f"best_selection={_format_float(optuna_summary.get('best_selection_score'))}"
            )

        current_trial = row.get("current_trial")
        if current_trial is not None:
            lines.append(
                "optuna current: "
                f"status={current_trial.get('status', 'n/a')} "
                f"trial={_format_int(current_trial.get('trial_index'))}/"
                f"{_format_int(current_trial.get('total_trials'))} "
                f"eval={_format_int(current_trial.get('eval_index'))} "
                f"step={_format_int(current_trial.get('timesteps'))} "
                f"selection={_format_float(current_trial.get('selection_score'))} "
                f"best={_format_float(current_trial.get('best_selection_score'))}"
            )

    return "\n".join(lines)


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(value).expanduser()
    if not path.is_absolute():
        path = base / path

    return path.resolve()


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _latest_run_root(algorithm_root: Path) -> Path | None:
    run_dirs = sorted(
        (
            path
            for path in algorithm_root.iterdir()
            if path.is_dir() and ArtifactStore.parse_run_id(path.name) is not None
        ),
        key=lambda path: path.name,
    )
    if not run_dirs:
        return None

    return run_dirs[-1]


def _read_optuna_current_trial(optuna_root: Path) -> dict[str, Any] | None:
    phases_root = optuna_root / "phases"
    if not phases_root.exists():
        return None

    candidates = sorted(
        (path for path in phases_root.glob("*/current_trial.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        payload = _safe_read_json(candidate)
        if payload is not None:
            return payload

    return None


def _build_run_row(
    *,
    artifacts_root: Path,
    experiment_name: str,
    algorithm: str,
    algorithm_root: Path,
) -> dict[str, Any]:
    latest_run_root = _latest_run_root(algorithm_root)

    live_evaluation = None
    training_summary = None
    final_evaluation = None
    tensorboard_logdir = None
    best_model_path = None
    final_model_path = None

    if latest_run_root is not None:
        tensorboard_logdir = latest_run_root / "logs" / "tensorboard"
        live_evaluation = _read_latest_evaluation(latest_run_root)
        training_summary = _read_training_summary(latest_run_root)
        final_evaluation = _read_final_evaluation(latest_run_root)
        best_candidate = latest_run_root / "models" / "best_model" / "best_model.zip"
        final_candidate = latest_run_root / "models" / "final_model.zip"
        best_model_path = str(best_candidate) if best_candidate.exists() else None
        final_model_path = str(final_candidate) if final_candidate.exists() else None

    optuna_root = artifacts_root / "optuna" / experiment_name / algorithm

    return {
        "experiment_name": experiment_name,
        "algorithm": algorithm,
        "run_id": latest_run_root.name if latest_run_root is not None else None,
        "run_root": str(latest_run_root) if latest_run_root is not None else None,
        "tensorboard_logdir": (
            str(tensorboard_logdir) if tensorboard_logdir is not None else None
        ),
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "live_evaluation": live_evaluation,
        "training_summary": training_summary,
        "final_evaluation": final_evaluation,
        "optuna_summary": _safe_read_json(optuna_root / "study_summary.json"),
        "current_trial": _read_optuna_current_trial(optuna_root),
    }


def _read_benchmark_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    payload = _safe_read_json(path)
    if payload is None:
        return None

    return {
        "path": str(path),
        "tuning_count": len(payload.get("tuning", [])),
        "training_count": len(payload.get("training", [])),
        "evaluation_count": len(payload.get("evaluation", [])),
        "video_count": len(payload.get("videos", [])),
    }


def _read_latest_evaluation(run_root: Path) -> dict[str, Any] | None:
    payload = _safe_read_json(run_root / "eval" / "evaluations_summary.json")
    if payload is None:
        return None

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        return None

    last_evaluation = evaluations[-1]
    if not isinstance(last_evaluation, dict):
        return None

    return last_evaluation


def _read_training_summary(run_root: Path) -> dict[str, Any] | None:
    payload = _safe_read_json(run_root / "eval" / "training_summary.json")
    if payload is None:
        return None

    return {
        "best_task_score": payload.get("best_task_score"),
        "best_selection_score": payload.get("best_selection_score"),
        "final_task_score": payload.get("final_task_score"),
        "final_selection_score": payload.get("final_selection_score"),
    }


def _read_final_evaluation(run_root: Path) -> dict[str, Any] | None:
    for candidate in (
        run_root / "eval" / "evaluation_best.json",
        run_root / "eval" / "evaluation_final.json",
    ):
        payload = _safe_read_json(candidate)
        if payload is None:
            continue

        return {
            "episodes": payload.get("episodes"),
            "mean_reward": payload.get("mean_reward"),
            "task_score": payload.get("task_score"),
            "selection_score": payload.get("selection_score"),
        }

    return None


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_int(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"
