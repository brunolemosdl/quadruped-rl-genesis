"""SB3 callbacks for task-aware evaluation and checkpoint selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from quadruped_rl_genesis.services.metrics import (
    evaluate_policy_variants,
    is_better_evaluation_bundle,
    summarize_evaluation_bundle,
)
from quadruped_rl_genesis.utils.io import to_json_compatible, write_json


class TaskEvalCallback(BaseCallback):
    """Evaluate checkpoints using task-aware selection instead of reward only.

    Uses task-specific metrics (success, path progress, projected speed) to
    select the best model instead of relying solely on episode return.
    """

    def __init__(
        self,
        *,
        eval_env,
        experiment_config: dict[str, Any],
        algorithm: str,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic_eval: bool,
        include_stochastic_eval: bool,
        best_model_save_path: Path | None,
        log_dir: Path,
        max_no_improvement_evals: int | None = None,
        min_evals: int = 0,
        verbose: int = 0,
    ) -> None:
        """Create a task-aware evaluation callback.

        Args:
            eval_env: Evaluation environment shared across callback runs.
            experiment_config (dict[str, Any]): Resolved experiment
                configuration.
            algorithm (str): Algorithm name.
            eval_freq (int): Evaluation frequency in SB3 callback calls.
            n_eval_episodes (int): Number of evaluation episodes per pass.
            deterministic_eval (bool): Whether the primary evaluation pass is
                deterministic.
            include_stochastic_eval (bool): Whether to also evaluate a
                stochastic policy mode.
            best_model_save_path (Path | None): Directory where the best model
                should be saved.
            log_dir (Path): Directory used for callback logs and summaries.
            max_no_improvement_evals (int | None, optional): Early-stopping
                patience in evaluation passes.
            min_evals (int, optional): Minimum number of evaluations before
                early stopping can trigger.
            verbose (int, optional): SB3 callback verbosity.
        """
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.experiment_config = experiment_config
        self.algorithm = algorithm
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)
        self.include_stochastic_eval = bool(include_stochastic_eval)
        self.best_model_save_path = (
            Path(best_model_save_path) if best_model_save_path is not None else None
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_no_improvement_evals = (
            None if max_no_improvement_evals is None else int(max_no_improvement_evals)
        )
        self.min_evals = int(min_evals)

        self.npz_path = self.log_dir / "evaluations.npz"
        self.history_path = self.log_dir / "evaluations_summary.json"

        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []
        self.evaluations_task_scores: list[float] = []
        self.evaluations_selection_scores: list[float] = []
        self.evaluations_task_eligible: list[bool] = []
        self.evaluation_summaries: list[dict[str, Any]] = []

        self.last_evaluation_bundle: dict[str, Any] | None = None
        self.best_evaluation_bundle: dict[str, Any] | None = None
        self.last_mean_reward = float("-inf")
        self.best_mean_reward = float("-inf")
        self.last_task_score = float("-inf")
        self.best_task_score = float("-inf")
        self.last_selection_score = float("-inf")
        self.best_selection_score = float("-inf")

        self._eval_count = 0
        self._no_improvement_evals = 0

    def _on_step(self) -> bool:
        """Run evaluation when the configured callback frequency is reached.

        On non-evaluation steps the callback immediately returns ``True`` so
        training can proceed without extra work.

        Returns:
            bool: ``True`` to continue training, ``False`` to request stopping.
        """
        if self.n_calls % self.eval_freq != 0:
            return True

        return self.evaluate_current(timesteps=int(self.num_timesteps))

    def record_initial_evaluation(self, model) -> dict[str, Any]:
        """Evaluate the initial model state before any training step.

        Args:
            model: SB3 model to evaluate.

        Returns:
            dict[str, Any]: Last recorded evaluation bundle.
        """
        self.model = model
        self.num_timesteps = 0
        self.n_calls = 0
        self.evaluate_current(timesteps=0)

        return self.last_evaluation_bundle or {}

    def evaluate_current(self, *, timesteps: int) -> bool:
        """Evaluate the current model, update logs, and check early stopping.

        Args:
            timesteps (int): Training timestep associated with this evaluation.

        Returns:
            bool: ``True`` to continue training, ``False`` when early stopping
                should stop the run.
        """
        training_env = self.model.get_env() if self.model is not None else None
        if training_env is not None:
            try:
                sync_envs_normalization(training_env, self.eval_env)
            except Exception:
                pass

        bundle = evaluate_policy_variants(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            experiment_config=self.experiment_config,
            include_stochastic=self.include_stochastic_eval,
            deterministic_eval=self.deterministic_eval,
        )
        improved = self._record_evaluation(bundle=bundle, timesteps=timesteps)
        self._after_evaluation(
            bundle=bundle,
            timesteps=timesteps,
            eval_index=self._eval_count,
            improved=improved,
        )
        return not (
            self.max_no_improvement_evals is not None
            and self._eval_count >= self.min_evals
            and self._no_improvement_evals >= self.max_no_improvement_evals
        )

    def _record_evaluation(self, *, bundle: dict[str, Any], timesteps: int) -> bool:
        """Record one evaluation bundle and update best-model tracking.

        Args:
            bundle (dict[str, Any]): Evaluation bundle for the current model.
            timesteps (int): Training timestep associated with this evaluation.

        Returns:
            bool: ``True`` when the bundle improves over the current best.
        """
        deterministic_metrics = bundle["metrics_deterministic"]
        episode_records = deterministic_metrics["episode_records"]
        rewards = [float(record["reward"]) for record in episode_records]
        lengths = [int(record["length"]) for record in episode_records]

        self._eval_count += 1
        self.last_evaluation_bundle = bundle
        self.last_mean_reward = float(bundle["mean_reward"])
        self.last_task_score = float(bundle["task_score"])
        self.last_selection_score = float(bundle["selection_score"])

        self.evaluations_timesteps.append(int(timesteps))
        self.evaluations_results.append(rewards)
        self.evaluations_length.append(lengths)
        self.evaluations_task_scores.append(self.last_task_score)
        self.evaluations_selection_scores.append(self.last_selection_score)
        self.evaluations_task_eligible.append(bool(bundle["task_eligible"]))
        article_metrics = deterministic_metrics.get("article_metrics", {})
        self.evaluation_summaries.append(
            {
                "timesteps": int(timesteps),
                "episode_length": float(
                    article_metrics.get("mean_episode_length", 0.0)
                ),
                "mean_planar_speed": float(
                    article_metrics.get("mean_planar_speed", 0.0)
                ),
                **summarize_evaluation_bundle(bundle),
            }
        )
        self._persist_logs()

        improved = is_better_evaluation_bundle(bundle, self.best_evaluation_bundle)
        if improved:
            self.best_evaluation_bundle = bundle
            self.best_mean_reward = self.last_mean_reward
            self.best_task_score = self.last_task_score
            self.best_selection_score = self.last_selection_score
            self._no_improvement_evals = 0
            self._save_best_model()
        else:
            self._no_improvement_evals += 1

        return improved

    def _persist_logs(self) -> None:
        """Persist callback evaluation history to NPZ and JSON files.

        The NPZ file keeps array-friendly data, while the JSON summary is easier
        to inspect manually.
        """
        np.savez(
            self.npz_path,
            timesteps=np.asarray(self.evaluations_timesteps, dtype=np.int64),
            results=np.asarray(self.evaluations_results, dtype=np.float32),
            ep_lengths=np.asarray(self.evaluations_length, dtype=np.int64),
            task_scores=np.asarray(self.evaluations_task_scores, dtype=np.float32),
            selection_scores=np.asarray(
                self.evaluations_selection_scores, dtype=np.float32
            ),
            task_eligible=np.asarray(self.evaluations_task_eligible, dtype=np.bool_),
        )
        write_json(
            self.history_path,
            {"evaluations": to_json_compatible(self.evaluation_summaries)},
        )

    def _save_best_model(self) -> None:
        """Save the current model to the configured best-model directory.

        The save is skipped when no best-model directory was configured.
        """
        if self.best_model_save_path is None:
            return

        self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.best_model_save_path / "best_model"))
        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(str(self.best_model_save_path / "vecnormalize.pkl"))

    def _after_evaluation(
        self,
        *,
        bundle: dict[str, Any],
        timesteps: int,
        eval_index: int,
        improved: bool,
    ) -> None:
        """Hook executed after each evaluation.

        Subclasses override this to add logging or side effects.

        Args:
            bundle (dict[str, Any]): Evaluation bundle just recorded.
            timesteps (int): Training timestep associated with the evaluation.
            eval_index (int): Sequential evaluation index.
            improved (bool): Whether the evaluation improved the best bundle.
        """
        article_metrics = bundle["metrics_deterministic"].get("article_metrics", {})
        model_logger = getattr(self.model, "_logger", None)
        if model_logger is not None:
            model_logger.record(
                "eval/episode_length",
                float(article_metrics.get("mean_episode_length", 0.0)),
            )
            model_logger.record(
                "eval/mean_planar_speed",
                float(article_metrics.get("mean_planar_speed", 0.0)),
            )
        del timesteps, eval_index, improved


class TrialProgressTaskEvalCallback(TaskEvalCallback):
    """Task-aware eval callback with structured Optuna progress logging.

    Extends ``TaskEvalCallback`` to emit trial progress messages compatible
    with Optuna's status reporting during hyperparameter tuning.
    """

    def __init__(
        self,
        *,
        logger,
        experiment_name: str,
        algorithm: str,
        trial_number: int,
        target_index: int,
        total_trials: int,
        status_writer,
        **kwargs,
    ) -> None:
        """Create an Optuna-aware task evaluation callback.

        Args:
            logger: Logger used for structured trial progress messages.
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            trial_number (int): Optuna trial number.
            target_index (int): Human-friendly completed-trial index target.
            total_trials (int): Total number of target trials.
            status_writer: Callable used to persist progress payloads.
            **kwargs: Additional arguments forwarded to ``TaskEvalCallback``.
        """
        super().__init__(algorithm=algorithm, **kwargs)
        self._logger = logger
        self._experiment_name = experiment_name
        self._trial_number = trial_number
        self._target_index = target_index
        self._total_trials = total_trials
        self._status_writer = status_writer

    def _after_evaluation(
        self,
        *,
        bundle: dict[str, Any],
        timesteps: int,
        eval_index: int,
        improved: bool,
    ) -> None:
        """Log structured Optuna progress after each evaluation pass.

        Args:
            bundle (dict[str, Any]): Evaluation bundle just recorded.
            timesteps (int): Training timestep associated with the evaluation.
            eval_index (int): Sequential evaluation index.
            improved (bool): Whether the evaluation improved the current best.
        """
        super()._after_evaluation(
            bundle=bundle,
            timesteps=timesteps,
            eval_index=eval_index,
            improved=improved,
        )
        article_metrics = bundle["metrics_deterministic"].get("article_metrics", {})
        self._logger.info(
            (
                "Trial %s/%s (optuna #%s) progress | "
                "experiment=%s algorithm=%s eval=%s "
                "timesteps=%s mean_reward=%.4f task_score=%.4f selection_score=%.4f "
                "best_selection_score=%.4f episode_length=%.2f "
                "mean_planar_speed=%.4f improved=%s"
            ),
            self._target_index,
            self._total_trials,
            self._trial_number,
            self._experiment_name,
            self.algorithm,
            eval_index,
            timesteps,
            float(bundle["mean_reward"]),
            float(bundle["task_score"]),
            float(bundle["selection_score"]),
            float(self.best_selection_score),
            float(article_metrics.get("mean_episode_length", 0.0)),
            float(article_metrics.get("mean_planar_speed", 0.0)),
            improved,
        )
        self._status_writer(
            {
                "status": "running",
                "experiment_name": self._experiment_name,
                "algorithm": self.algorithm,
                "trial_number": self._trial_number,
                "target_index": self._target_index,
                "trial_index": self._trial_number + 1,
                "total_trials": self._total_trials,
                "eval_index": eval_index,
                "timesteps": int(timesteps),
                "mean_reward": float(bundle["mean_reward"]),
                "task_score": float(bundle["task_score"]),
                "selection_score": float(bundle["selection_score"]),
                "best_selection_score": float(self.best_selection_score),
                "episode_length": float(
                    article_metrics.get("mean_episode_length", 0.0)
                ),
                "mean_planar_speed": float(
                    article_metrics.get("mean_planar_speed", 0.0)
                ),
            }
        )
