"""SB3 ``VecEnv`` wrapper around the Go2 navigation task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from quadruped_rl_genesis.navigation.task import Go2NavigationTask


class GenesisEnv(VecEnv):
    """Stable-Baselines3 vectorized environment backed by the navigation task.

    Wraps ``Go2NavigationTask`` and provides the VecEnv interface expected by
    Stable-Baselines3, including async step, attribute forwarding, and
    optional rendering.
    """

    def __init__(
        self,
        experiment_config: dict[str, Any],
        num_envs: int,
        show_viewer: bool = False,
        add_camera: bool = False,
        fast_viz: bool = False,
        viewer_help_text: bool = True,
        disable_reward_curriculum: bool = False,
        disable_terrain_curriculum: bool = False,
    ) -> None:
        """Create a vectorized wrapper around ``Go2NavigationTask``.

        Args:
            experiment_config (dict[str, Any]): Resolved experiment
                configuration.
            num_envs (int): Number of parallel environments.
            show_viewer (bool, optional): Whether to open the Genesis viewer.
            add_camera (bool, optional): Whether to attach a render camera.
            fast_viz (bool, optional): Whether to use lightweight visualization
                settings.
            viewer_help_text (bool, optional): Whether to show default viewer
                keyboard instructions.
            disable_reward_curriculum (bool, optional): Forwarded to the task;
                use True for visualization or eval so all reward terms apply.
            disable_terrain_curriculum (bool, optional): Forwarded to the task;
                use True for visualization or eval so terrain resets use the
                configured evaluation/final stage instead of training stages.
        """
        self.task = Go2NavigationTask(
            experiment_config=experiment_config,
            num_envs=num_envs,
            show_viewer=show_viewer,
            add_camera=add_camera,
            fast_viz=fast_viz,
            viewer_help_text=viewer_help_text,
            disable_reward_curriculum=disable_reward_curriculum,
            disable_terrain_curriculum=disable_terrain_curriculum,
        )
        self.render_mode = (
            "rgb_array" if add_camera else ("human" if show_viewer else None)
        )
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.task.num_observations,),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.task.num_actions,),
            dtype=np.float32,
        )
        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._pending_actions: np.ndarray | None = None
        self.reset_infos = [{} for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        """Reset all environments and return NumPy observations.

        The underlying task still works in Torch tensors, so this method
        performs the conversion expected by Stable-Baselines3.

        Returns:
            np.ndarray: Batched observations with ``float32`` dtype.
        """
        observations, infos = self.task.reset()
        self.reset_infos = infos

        return observations.detach().cpu().numpy().astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        """Cache actions until ``step_wait`` is called.

        Args:
            actions (np.ndarray): Batched action array produced by SB3.
        """
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Execute one environment step using the actions stored by ``step_async``.

        This is the point where cached NumPy actions become Torch tensors for
        the task and the resulting tensors are converted back to NumPy arrays.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
                Observations, rewards, done flags, and info dictionaries.

        Raises:
            RuntimeError: If ``step_async`` was not called before this method.
        """
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait.")

        action_tensor = torch.tensor(
            self._pending_actions,
            device=self.task.device,
            dtype=torch.float32,
        )
        observations, rewards, dones, infos = self.task.step(action_tensor)
        self._pending_actions = None

        return (
            observations.detach().cpu().numpy().astype(np.float32),
            rewards.detach().cpu().numpy().astype(np.float32),
            dones.detach().cpu().numpy().astype(bool),
            infos,
        )

    def close(self) -> None:
        """Release task-level resources associated with the environment.

        The wrapper simply forwards cleanup to the underlying navigation task.
        """
        self.task.close()

    def get_attr(
        self,
        attr_name: str,
        indices: Iterable[int] | None = None,
    ) -> list[Any]:
        """Return an attribute from the wrapper or underlying task.

        Args:
            attr_name (str): Attribute name to retrieve.
            indices (Iterable[int] | None, optional): Requested environment
                indices.

        Returns:
            list[Any]: One value per requested environment index.
        """
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
        else:
            value = getattr(self.task, attr_name)

        indices = self._get_indices(indices)

        return [value for _ in indices]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Iterable[int] | None = None,
    ) -> None:
        """Set an attribute on the wrapper or underlying task.

        Args:
            attr_name (str): Attribute name to update.
            value (Any): Value assigned to the attribute.
            indices (Iterable[int] | None, optional): Ignored by this wrapper
                because the task is shared across vectorized environments.
        """
        del indices

        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
            return

        setattr(self.task, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Iterable[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Invoke a task method and broadcast the result across indices.

        Args:
            method_name (str): Method name on the underlying task.
            *method_args (Any): Positional arguments forwarded to the method.
            indices (Iterable[int] | None, optional): Requested environment
                indices.
            **method_kwargs (Any): Keyword arguments forwarded to the method.

        Returns:
            list[Any]: One copy of the result per requested environment index.
        """
        method = getattr(self.task, method_name)
        result = method(*method_args, **method_kwargs)
        indices = self._get_indices(indices)

        return [result for _ in indices]

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper],
        indices: Iterable[int] | None = None,
    ) -> list[bool]:
        """Report wrapper status for SB3 compatibility.

        Args:
            wrapper_class (type[gym.Wrapper]): Wrapper type requested by SB3.
            indices (Iterable[int] | None, optional): Requested environment
                indices.

        Returns:
            list[bool]: Always ``False`` for each requested index because this
                environment does not use Gym wrappers internally.
        """
        del wrapper_class
        indices = self._get_indices(indices)

        return [False for _ in indices]

    def get_images(self) -> list[np.ndarray | None]:
        """Return one rendered RGB frame per environment slot.

        The underlying task renders a single frame, which is replicated across
        vector slots to satisfy the ``VecEnv`` API.

        Returns:
            list[np.ndarray | None]: Captured frame replicated for each vector
                environment slot.
        """
        frame = self.task.capture_rgb_frame()

        return [frame for _ in range(self.num_envs)]
