"""Benchmark runs comparing policies or hyperparameter configurations."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from quadruped_rl_genesis.config.loader import load_optuna_config
from quadruped_rl_genesis.operations.common import (
    configure_headless_runtime_env,
    derive_output_settings,
)
from quadruped_rl_genesis.pipeline.evaluate import run_evaluation
from quadruped_rl_genesis.pipeline.train import run_training
from quadruped_rl_genesis.pipeline.tune import run_tuning
from quadruped_rl_genesis.services.artifacts import ArtifactStore
from quadruped_rl_genesis.services.logger import configure_logging, get_logger
from quadruped_rl_genesis.services.platform import collect_runtime_context
from quadruped_rl_genesis.services.runtime import shutdown_genesis
from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_json, write_json

LOGGER = get_logger(__name__)


def _aggregate_evaluation_results(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate evaluation metrics by experiment, algorithm, and hyperparameter source.

    Args:
        evaluations (list[dict[str, Any]]): Per-run evaluation payloads (must include
            keys such as ``experiment_name``, ``algorithm``, ``hyperparams_source``).

    Returns:
        dict[str, Any]: ``by_group`` summaries and ``comparison`` cohort slices used in
            the benchmark report.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in evaluations:
        key = (
            str(item.get("experiment_name", "")),
            str(item.get("algorithm", "")),
            str(item.get("hyperparams_source", "unknown")),
        )
        grouped.setdefault(key, []).append(item)

    by_group: dict[str, Any] = {}
    for (experiment_name, algorithm, source), rows in grouped.items():
        gate_pass_rate = sum(
            1 for row in rows if bool(row.get("scientific_gates_passed", True))
        ) / max(len(rows), 1)
        task_scores = [float(row.get("task_score", 0.0)) for row in rows]
        selection_scores = [float(row.get("selection_score", 0.0)) for row in rows]
        article = [
            row.get("metrics_deterministic", {}).get("article_metrics", {})
            for row in rows
        ]
        success_rates = [float(m.get("success_rate", 0.0)) for m in article]
        mean_final_distances = [
            float(m.get("mean_final_goal_distance", 0.0)) for m in article
        ]
        mean_final_heading_errors = [
            float(m.get("mean_final_goal_yaw_error_deg", 0.0)) for m in article
        ]
        mean_arc_progress_speeds = [
            float(m.get("mean_arc_progress_speed", 0.0)) for m in article
        ]
        mean_cross_track_errors = [
            float(m.get("mean_cross_track_error", 0.0)) for m in article
        ]
        by_group[f"{experiment_name}:{algorithm}:{source}"] = {
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "hyperparams_source": source,
            "runs": len(rows),
            "scientific_gate_pass_rate": float(gate_pass_rate),
            "mean_task_score": float(sum(task_scores) / max(len(task_scores), 1)),
            "mean_selection_score": float(
                sum(selection_scores) / max(len(selection_scores), 1)
            ),
            "mean_success_rate": float(sum(success_rates) / max(len(success_rates), 1)),
            "mean_final_goal_distance": float(
                sum(mean_final_distances) / max(len(mean_final_distances), 1)
            ),
            "mean_final_goal_yaw_error_deg": float(
                sum(mean_final_heading_errors) / max(len(mean_final_heading_errors), 1)
            ),
            "mean_arc_progress_speed": float(
                sum(mean_arc_progress_speeds) / max(len(mean_arc_progress_speeds), 1)
            ),
            "mean_cross_track_error": float(
                sum(mean_cross_track_errors) / max(len(mean_cross_track_errors), 1)
            ),
        }

    baseline_candidates = [
        payload
        for payload in by_group.values()
        if payload["hyperparams_source"] == "default"
    ]
    tuned_candidates = [
        payload
        for payload in by_group.values()
        if payload["hyperparams_source"] == "optuna"
    ]
    return {
        "by_group": by_group,
        "comparison": {
            "baseline_default": baseline_candidates,
            "tuned_optuna": tuned_candidates,
        },
    }


def _release_gpu_memory() -> None:
    """Release GPU and runtime memory between benchmark steps to reduce VRAM buildup.

    Shuts down Genesis, runs garbage collection, and clears the CUDA cache
    when available to avoid VRAM accumulation across sequential runs.
    """
    shutdown_genesis()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _get_completed_training_runs(
    artifacts_root: Path,
    experiments: list[str],
    algorithms: list[str],
    seeds: list[int],
    hyperparams_source: str,
) -> dict[tuple[str, str, int], Path]:
    """Find run roots for (experiment, algorithm, seed) that already have a completed training.

    Returns:
        dict mapping (experiment_name, algorithm, seed) to the run_root Path
        of the most recent completed run (by run_id name).
    """
    result: dict[tuple[str, str, int], Path] = {}
    for exp in experiments:
        for algo in algorithms:
            algo_dir = artifacts_root / "experiments" / exp / algo
            if not algo_dir.exists():
                continue
            for run_dir in sorted(
                algo_dir.iterdir(), key=lambda p: p.name, reverse=True
            ):
                if not run_dir.is_dir():
                    continue
                parsed_run_id = ArtifactStore.parse_run_id(run_dir.name)
                if parsed_run_id is None:
                    continue
                seed_val = int(parsed_run_id["seed"])
                if seed_val not in seeds:
                    continue
                if parsed_run_id["hyperparams_source"] != hyperparams_source:
                    continue
                key = (exp, algo, seed_val)
                if key in result:
                    continue
                if (run_dir / "eval" / "training_summary.json").exists() or (
                    run_dir / "models" / "final_model.zip"
                ).exists():
                    result[key] = run_dir
    return result


def _ensure_output_root(output_root: str | Path) -> Path:
    """Resolve and create the benchmark output directory structure.

    Args:
        output_root (str | Path): Requested benchmark output root.

    Returns:
        Path: Absolute benchmark output directory with ``logs`` and ``reports``
            subdirectories created.
    """
    resolved = Path(output_root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "logs").mkdir(parents=True, exist_ok=True)
    (resolved / "reports").mkdir(parents=True, exist_ok=True)

    return resolved


def _index_training_models(
    training_results: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, str]]:
    """Index benchmark training outputs by experiment, algorithm, and seed.

    Args:
        training_results (list[dict[str, Any]]): Training summaries returned by
            ``run_training``.

    Returns:
        dict[tuple[str, str, int], dict[str, str]]: Mapping to the preferred
            saved model path for each training run.
    """
    index: dict[tuple[str, str, int], dict[str, str]] = {}

    for summary in training_results:
        experiment_name = str(summary["experiment_name"])
        algorithm = str(summary["algorithm"])
        seed = int(summary["seed"])
        best_model_path = summary.get("best_model_path")
        final_model_path = summary.get("final_model_path")

        if best_model_path:
            index[(experiment_name, algorithm, seed)] = {
                "model_kind": "best",
                "model_path": str(best_model_path),
            }
        elif final_model_path:
            index[(experiment_name, algorithm, seed)] = {
                "model_kind": "final",
                "model_path": str(final_model_path),
            }

    return index


def run_benchmark(
    settings: AppSettings,
    *,
    output_root: str | Path,
    experiments: list[str],
    algorithms: list[str],
    seeds: list[int],
    device: str,
    genesis_device: str,
    algorithm_device: str,
    hyperparams_source: str,
    tune_seed: int,
    eval_seed: int,
    final_eval_episodes: int,
    tune_base_source: str = "default",
    skip_tune: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    record_videos: bool = False,
    video_episodes: int = 1,
    video_max_steps: int = 1200,
    video_seed: int | None = None,
    video_fast_viz: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full benchmark across tuning, training, evaluation, and videos.

    When ``resume`` is True, training runs that already have a completed run
    (same experiment, algorithm, seed) under the benchmark artifacts root are
    skipped and their existing summary is loaded. This allows re-running the
    benchmark after a pod restart or interrupt without redoing finished work.

    After each step (tuning, training, evaluation, video), GPU memory is
    released (Genesis shutdown + gc + CUDA cache) to reduce VRAM buildup.

    Args:
        settings (AppSettings): Base application settings.
        output_root (str | Path): Directory used to store benchmark outputs.
        experiments (list[str]): Experiment names to execute.
        algorithms (list[str]): Algorithms included in the benchmark.
        seeds (list[int]): Training seeds for repeated runs.
        device (str): Default device (used when genesis/algorithm not set).
        genesis_device (str): Device for Genesis simulation.
        algorithm_device (str): Device for RL algorithm.
        hyperparams_source (str): Hyperparameter source used by training and
            evaluation steps.
        tune_base_source (str, optional): Named base hyperparameter source used
            during Optuna studies before trial overrides are applied.
        tune_seed (int): Seed for Optuna studies.
        eval_seed (int): Seed for final evaluation runs.
        final_eval_episodes (int): Number of episodes used in final evaluation.
        skip_tune (bool, optional): Whether to skip Optuna tuning.
        skip_train (bool, optional): Whether to skip training runs.
        skip_eval (bool, optional): Whether to skip evaluation runs.
        record_videos (bool, optional): Whether to record visualization videos.
        video_episodes (int, optional): Number of episodes per recorded video.
        video_max_steps (int, optional): Maximum steps per video episode.
        video_seed (int | None, optional): Optional seed for video generation.
        video_fast_viz (bool, optional): Whether to enable lightweight
            visualization settings during video capture (defaults to ``False``).
        resume (bool, optional): If True, skip training runs that already have
            a completed run for the same (experiment, algorithm, seed).

    Returns:
        dict[str, Any]: Benchmark summary with profile, tuning, training,
            evaluation, and video results.
    """
    output_root_path = _ensure_output_root(output_root)
    benchmark_settings = derive_output_settings(settings, output_root_path)
    benchmark_settings.artifacts_root.mkdir(parents=True, exist_ok=True)

    configure_headless_runtime_env(benchmark_settings.project_root)
    configure_logging(output_root_path / "logs" / "benchmark.log", settings.log_level)

    os.environ["QUADRUPED_RL_GENESIS_ARTIFACTS_ROOT"] = str(
        benchmark_settings.artifacts_root
    )
    os.environ["QUADRUPED_RL_GENESIS_OPTUNA_STORAGE"] = (
        benchmark_settings.optuna_storage
    )
    try:
        optuna_cfg = load_optuna_config(settings).get("optuna", {})
    except FileNotFoundError:
        optuna_cfg = {}
    phase_cfg = optuna_cfg.get("phases", {})
    tuning_phases = (
        list(phase_cfg.get("sequence", []))
        if bool(phase_cfg.get("enabled", False))
        else ["single"]
    )

    profile = {
        "output_root": str(output_root_path),
        "artifacts_root": str(benchmark_settings.artifacts_root),
        "optuna_storage": benchmark_settings.optuna_storage,
        "device": device,
        "genesis_device": genesis_device,
        "algorithm_device": algorithm_device,
        "hyperparams_source": hyperparams_source,
        "tune_base_source": tune_base_source,
        "experiments": experiments,
        "algorithms": algorithms,
        "seeds": seeds,
        "tune_seed": int(tune_seed),
        "eval_seed": int(eval_seed),
        "final_eval_episodes": int(final_eval_episodes),
        "skip_tune": bool(skip_tune),
        "skip_train": bool(skip_train),
        "skip_eval": bool(skip_eval),
        "record_videos": bool(record_videos),
        "resume": bool(resume),
        "tuning_mode": "two_phase" if tuning_phases != ["single"] else "single_phase",
        "tuning_phases": tuning_phases,
        "scientific_gate_policy": "selection_and_ranking_require_gate_pass",
        "comparison_targets": {
            "baseline_hyperparams_source": "default",
            "tuned_hyperparams_source": "optuna",
        },
        "runtime_context": collect_runtime_context(),
    }
    profile_path = output_root_path / "reports" / "benchmark_profile.json"
    write_json(profile_path, profile)

    LOGGER.info(
        "Starting benchmark | output_root=%s experiments=%s algorithms=%s seeds=%s",
        output_root_path,
        " ".join(experiments),
        " ".join(algorithms),
        " ".join(str(seed) for seed in seeds),
    )

    results: dict[str, Any] = {
        "profile_path": str(profile_path),
        "tuning": [],
        "training": [],
        "evaluation": [],
        "videos": [],
    }

    if hyperparams_source == "optuna" and not skip_tune:
        tune_tasks = [(exp, alg) for exp in experiments for alg in algorithms]
        for experiment_name, algorithm in tqdm(
            tune_tasks, desc="Optuna tuning", unit="run"
        ):
            LOGGER.info(
                "Running Optuna | experiment=%s algorithm=%s",
                experiment_name,
                algorithm,
            )
            results["tuning"].append(
                run_tuning(
                    settings=benchmark_settings,
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    seed=tune_seed,
                    genesis_device=genesis_device,
                    algorithm_device=algorithm_device,
                    base_hyperparams_source=tune_base_source,
                )
            )
            _release_gpu_memory()
    else:
        LOGGER.info(
            "Skipping Optuna | hyperparams_source=%s skip_tune=%s",
            hyperparams_source,
            skip_tune,
        )

    completed_training = (
        _get_completed_training_runs(
            benchmark_settings.artifacts_root,
            experiments,
            algorithms,
            seeds,
            hyperparams_source,
        )
        if resume
        else {}
    )

    if not skip_train:
        train_tasks = [
            (exp, alg, s) for exp in experiments for alg in algorithms for s in seeds
        ]
        for experiment_name, algorithm, seed in tqdm(
            train_tasks, desc="Training", unit="run"
        ):
            key = (experiment_name, algorithm, seed)
            if resume and key in completed_training:
                run_root = completed_training[key]
                summary_path = run_root / "eval" / "training_summary.json"
                if summary_path.exists():
                    summary = read_json(summary_path)
                    results["training"].append(summary)
                else:
                    final_path = run_root / "models" / "final_model.zip"
                    best_path = run_root / "models" / "best_model" / "best_model.zip"
                    results["training"].append(
                        {
                            "experiment_name": experiment_name,
                            "algorithm": algorithm,
                            "hyperparams_source": hyperparams_source,
                            "seed": seed,
                            "run_id": run_root.name,
                            "run_root": str(run_root),
                            "final_model_path": str(final_path),
                            "best_model_path": (
                                str(best_path) if best_path.exists() else None
                            ),
                        }
                    )

                LOGGER.info(
                    "Resumed (skipped) training | experiment=%s algorithm=%s seed=%s run_root=%s",
                    experiment_name,
                    algorithm,
                    seed,
                    run_root.name,
                )

                continue

            LOGGER.info(
                "Running training | experiment=%s algorithm=%s seed=%s",
                experiment_name,
                algorithm,
                seed,
            )

            results["training"].append(
                run_training(
                    settings=benchmark_settings,
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    hyperparams_source=hyperparams_source,
                    seed=seed,
                    genesis_device=genesis_device,
                    algorithm_device=algorithm_device,
                )
            )

            _release_gpu_memory()
    else:
        LOGGER.info("Skipping training by request.")

    if not skip_eval:
        training_models = _index_training_models(results["training"])

        eval_tasks = [
            (exp, alg, s) for exp in experiments for alg in algorithms for s in seeds
        ]
        for experiment_name, algorithm, seed in tqdm(
            eval_tasks, desc="Evaluation", unit="run"
        ):
            trained_model = training_models.get((experiment_name, algorithm, seed))
            LOGGER.info(
                "Running evaluation | experiment=%s algorithm=%s seed=%s",
                experiment_name,
                algorithm,
                seed,
            )

            results["evaluation"].append(
                run_evaluation(
                    settings=benchmark_settings,
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    hyperparams_source=hyperparams_source,
                    seed=eval_seed,
                    genesis_device=genesis_device,
                    algorithm_device=algorithm_device,
                    model_kind=(
                        str(trained_model["model_kind"])
                        if trained_model is not None
                        else "best"
                    ),
                    model_path=(
                        str(trained_model["model_path"])
                        if trained_model is not None
                        else None
                    ),
                    run_id=None,
                    train_seed=seed if trained_model is None else None,
                    episodes=final_eval_episodes,
                )
            )

            _release_gpu_memory()
    else:
        LOGGER.info("Skipping final evaluation by request.")

    if record_videos:
        from quadruped_rl_genesis.pipeline.visualize import run_visualization

        resolved_video_seed = video_seed if video_seed is not None else eval_seed
        video_tasks = [(exp, alg) for exp in experiments for alg in algorithms]
        for experiment_name, algorithm in tqdm(video_tasks, desc="Videos", unit="run"):
            LOGGER.info(
                "Recording visualization | experiment=%s algorithm=%s",
                experiment_name,
                algorithm,
            )

            results["videos"].append(
                run_visualization(
                    settings=benchmark_settings,
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    hyperparams_source=hyperparams_source,
                    seed=resolved_video_seed,
                    genesis_device=genesis_device,
                    algorithm_device=algorithm_device,
                    model_kind="best",
                    model_path=None,
                    run_id=None,
                    train_seed=None,
                    episodes=video_episodes,
                    max_steps=video_max_steps,
                    record_video=True,
                    no_model=False,
                    fast_viz=video_fast_viz,
                    tag=f"{experiment_name}_{algorithm}_benchmark",
                )
            )

            _release_gpu_memory()
    else:
        LOGGER.info("Skipping visualization videos.")

    results["evaluation_aggregate"] = _aggregate_evaluation_results(
        results["evaluation"]
    )
    results["profile"] = profile
    summary_path = output_root_path / "reports" / "benchmark_summary.json"
    write_json(summary_path, results)
    LOGGER.info("Benchmark finished. Summary saved to %s", summary_path)

    return results
