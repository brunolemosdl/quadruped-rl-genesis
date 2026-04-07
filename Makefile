.DEFAULT_GOAL := help
SHELL := /bin/bash
VENV_PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3
PYTHON ?= PYTHONPATH=src $(VENV_PYTHON)
RUFF ?= $(VENV_PYTHON) -m ruff
EXPERIMENT ?= default
ALGORITHM ?= ppo
HYPERPARAMS_SOURCE ?= default
SEED ?= 42
DEVICE ?= auto
GENESIS_DEVICE ?=
ALGORITHM_DEVICE ?=
MODEL_KIND ?= best
MODEL_PATH ?=
RUN_ID ?=
TRAIN_SEED ?=
EPISODES ?= 1
MAX_STEPS ?= 2000
FAST_VIZ ?= 0
NO_MODEL ?= 0
HEADLESS ?= 0
RESUME ?= 0
TUNE_BASE_SOURCE ?= default
BENCHMARK_PROFILE ?=
BENCHMARK_EXPERIMENTS ?=
BENCHMARK_ALGORITHMS ?=
BENCHMARK_SEEDS ?=
TUNE_SEED ?=
EVAL_SEED ?=
MONITOR_MODE ?= summary
OUTPUT_ROOT ?=
ARTIFACTS_ROOT ?=
HOST ?= 0.0.0.0
PORT ?= 6006
RELOAD_INTERVAL ?= 15
WATCH ?= 0
INTERVAL ?= 15
LIMIT ?= 20
MONITOR_EXPERIMENT ?=
MONITOR_ALGORITHM ?=

.PHONY: help setup install check benchmark format lint quality train tune evaluate visualize record monitor

help: ## Show available commands
	@echo "Available commands:" && \
	awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Configure or refresh the project environment
	PYTHONPATH=src $(SYSTEM_PYTHON) main.py setup

install: setup ## Alias for setup

check: ## Validate the local runtime
	$(PYTHON) main.py check

benchmark: ## Run the final article benchmark (optional: BENCHMARK_PROFILE=default, BENCHMARK_EXPERIMENTS, SEED, TUNE_SEED, EVAL_SEED)
	$(PYTHON) main.py benchmark \
		$(if $(BENCHMARK_PROFILE),--profile $(BENCHMARK_PROFILE),) \
		--device $(DEVICE) \
		$(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) \
		$(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),) \
		--hyperparams-source $(HYPERPARAMS_SOURCE) \
		--tune-base-source $(TUNE_BASE_SOURCE) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(TUNE_SEED),--tune-seed $(TUNE_SEED),) \
		$(if $(EVAL_SEED),--eval-seed $(EVAL_SEED),) \
		$(if $(BENCHMARK_EXPERIMENTS),--experiments $(BENCHMARK_EXPERIMENTS),) \
		$(if $(BENCHMARK_ALGORITHMS),--algorithms $(BENCHMARK_ALGORITHMS),) \
		$(if $(BENCHMARK_SEEDS),--seeds $(BENCHMARK_SEEDS),)

format: ## Format the codebase
	$(RUFF) format .

lint: ## Run static lint checks
	$(RUFF) check .

quality: lint ## Run static quality checks

train: ## Train the selected algorithm (ALGORITHM=ppo|sac|td3)
	$(PYTHON) main.py train --experiment $(EXPERIMENT) --algorithm $(ALGORITHM) --hyperparams-source $(HYPERPARAMS_SOURCE) --seed $(SEED) --device $(DEVICE) $(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) $(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),) $(if $(filter 1,$(RESUME)),--resume,)

tune: ## Run Optuna for the selected algorithm
	$(PYTHON) main.py tune --experiment $(EXPERIMENT) --algorithm $(ALGORITHM) --hyperparams-source $(HYPERPARAMS_SOURCE) --seed $(SEED) --device $(DEVICE) $(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) $(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),)

evaluate: ## Evaluate a model by path, run id, train seed, or latest source-matching run (optional: EVAL_EXPERIMENT=irregular|rough; baseline: ALGORITHM=random)
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --algorithm $(ALGORITHM) --hyperparams-source $(HYPERPARAMS_SOURCE) --seed $(SEED) --device $(DEVICE) $(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) $(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),) --model-kind $(MODEL_KIND) $(if $(MODEL_PATH),--model-path $(MODEL_PATH),) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(TRAIN_SEED),--train-seed $(TRAIN_SEED),) $(if $(EVAL_EXPERIMENT),--eval-experiment $(EVAL_EXPERIMENT),)

visualize: ## Visualize a model by path, run id, train seed, or latest source-matching run (optional: EVAL_EXPERIMENT=irregular|rough; baseline: ALGORITHM=random)
	$(PYTHON) main.py visualize --experiment $(EXPERIMENT) --algorithm $(ALGORITHM) --hyperparams-source $(HYPERPARAMS_SOURCE) --seed $(SEED) --device $(DEVICE) $(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) $(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),) --model-kind $(MODEL_KIND) --episodes $(EPISODES) --max-steps $(MAX_STEPS) $(if $(MODEL_PATH),--model-path $(MODEL_PATH),) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(TRAIN_SEED),--train-seed $(TRAIN_SEED),) $(if $(EVAL_EXPERIMENT),--eval-experiment $(EVAL_EXPERIMENT),) $(if $(filter 1,$(FAST_VIZ)),--fast-viz,) $(if $(filter 1,$(NO_MODEL)),--no-model,) $(if $(filter 1,$(HEADLESS)),--headless,)

record: ## Record a video from an explicit model zip (requires MODEL_PATH; optional: EVAL_EXPERIMENT=irregular|rough)
	@[ -n "$(MODEL_PATH)" ] || (echo "MODEL_PATH is required. Example: make record ALGORITHM=sac EXPERIMENT=flat MODEL_PATH=artifacts/.../best_model.zip" && exit 1)
	$(PYTHON) main.py visualize --experiment $(EXPERIMENT) --algorithm $(ALGORITHM) --hyperparams-source $(HYPERPARAMS_SOURCE) --seed $(SEED) --device $(DEVICE) $(if $(GENESIS_DEVICE),--genesis-device $(GENESIS_DEVICE),) $(if $(ALGORITHM_DEVICE),--algorithm-device $(ALGORITHM_DEVICE),) --model-kind $(MODEL_KIND) --model-path $(MODEL_PATH) --episodes $(EPISODES) --max-steps $(MAX_STEPS) --record-video $(if $(EVAL_EXPERIMENT),--eval-experiment $(EVAL_EXPERIMENT),) $(if $(filter 1,$(FAST_VIZ)),--fast-viz,) $(if $(filter 1,$(NO_MODEL)),--no-model,) $(if $(filter 1,$(HEADLESS)),--headless,)

monitor: ## Monitor artifacts via summary or TensorBoard
	$(PYTHON) main.py monitor --mode $(MONITOR_MODE) $(if $(OUTPUT_ROOT),--output-root $(OUTPUT_ROOT),) $(if $(ARTIFACTS_ROOT),--artifacts-root $(ARTIFACTS_ROOT),) $(if $(MONITOR_EXPERIMENT),--experiment $(MONITOR_EXPERIMENT),) $(if $(MONITOR_ALGORITHM),--algorithm $(MONITOR_ALGORITHM),) --host $(HOST) --port $(PORT) --reload-interval $(RELOAD_INTERVAL) --limit $(LIMIT) $(if $(filter 1,$(WATCH)),--watch --interval $(INTERVAL),)
