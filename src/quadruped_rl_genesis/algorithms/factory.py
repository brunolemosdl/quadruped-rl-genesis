"""Stable-Baselines3 model construction and loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import configure_logger

from quadruped_rl_genesis.algorithms.instrumented import (
    InstrumentedSAC,
    InstrumentedTD3,
)

SB3_ALGORITHMS = {
    "ppo": PPO,
    "sac": InstrumentedSAC,
    "td3": InstrumentedTD3,
}


ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _resolve_policy_kwargs(policy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve config-only policy options into SB3-ready arguments.

    Config files may store activation functions as strings such as ``"relu"``
    or ``"elu"``. This helper deep-copies the input mapping and replaces those
    string names with the matching PyTorch classes expected by SB3.

    Args:
        policy_kwargs (dict[str, Any] | None): Policy kwargs loaded from config.
            If ``None``, an empty dictionary is assumed.

    Returns:
        dict[str, Any]: New dictionary ready to be passed to Stable-Baselines3.

    Raises:
        ValueError: If ``activation_fn`` is a string that is not supported by
            ``ACTIVATION_FUNCTIONS``.
    """
    resolved = copy.deepcopy(policy_kwargs or {})

    activation_name = resolved.get("activation_fn")

    if isinstance(activation_name, str):
        key = activation_name.lower()

        if key not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation function: {activation_name}")

        resolved["activation_fn"] = ACTIVATION_FUNCTIONS[key]

    return resolved


def create_sb3_model(
    algorithm_config_payload: dict[str, Any],
    env,
    tensorboard_log: Path,
    device: str,
):
    """Instantiate an SB3 model from a resolved algorithm configuration.

    Args:
        algorithm_config_payload (dict[str, Any]): Resolved algorithm section
            containing the ``algorithm`` block expected by the project.
        env: Vectorized environment bound to the model.
        tensorboard_log (Path): TensorBoard output directory.
        device (str): Runtime device used by SB3.

    Returns:
        Any: Stable-Baselines3 model instance matching the configured
            algorithm.
    """
    algorithm_config = copy.deepcopy(algorithm_config_payload["algorithm"])
    algorithm_name = algorithm_config["name"].lower()
    model_class = SB3_ALGORITHMS[algorithm_name]

    model_kwargs = copy.deepcopy(algorithm_config.get("model_kwargs", {}))
    policy_kwargs = _resolve_policy_kwargs(algorithm_config.get("policy_kwargs"))

    if algorithm_name == "td3" and "action_noise_sigma" in model_kwargs:
        sigma = float(model_kwargs.pop("action_noise_sigma"))
        action_dim = env.action_space.shape[-1]

        model_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=sigma * np.ones(action_dim),
        )

    model = model_class(
        policy=algorithm_config["policy"],
        env=env,
        tensorboard_log=str(tensorboard_log),
        device=device,
        policy_kwargs=policy_kwargs,
        **model_kwargs,
    )

    model.set_logger(
        configure_logger(
            verbose=int(getattr(model, "verbose", 0)),
            tensorboard_log=str(tensorboard_log),
            tb_log_name="run",
            reset_num_timesteps=True,
        )
    )

    return model


def load_sb3_model(
    algorithm: str,
    model_path: Path,
    env=None,
    device: str = "auto",
):
    """Load a saved SB3 model and optionally attach it to an environment.

    Args:
        algorithm (str): Algorithm name used to select the SB3 model class.
        model_path (Path): Path to the saved model zip file.
        env (Any, optional): Environment optionally bound during loading.
        device (str, optional): Runtime device forwarded to SB3.

    Returns:
        Any: Loaded Stable-Baselines3 model instance.
    """
    model_class = SB3_ALGORITHMS[algorithm.lower()]

    return model_class.load(str(model_path), env=env, device=device)
