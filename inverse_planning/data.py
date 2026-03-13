from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inverse_planning.inference import trajectory_to_observer_labels
from inverse_planning.memo_backend import MemoPolicyBackend
from inverse_planning.simulate import sample_trajectory
from inverse_planning.task import GridworldTask


@dataclass
class DatasetBundle:
    grids: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    final_posteriors: np.ndarray
    online_posteriors: np.ndarray
    goal_condition_inputs: np.ndarray


def encode_grid_frame(task: GridworldTask, position: tuple[int, int], step: int, horizon: int) -> np.ndarray:
    h, w = task.shape
    frame = np.zeros((2 + task.n_goals + 1, h, w), dtype=np.float32)
    frame[0] = task.grid.astype(np.float32)
    frame[1, position[0], position[1]] = 1.0
    for goal_index, goal in enumerate(task.goal_locs):
        frame[2 + goal_index, goal[0], goal[1]] = 1.0
    frame[-1].fill(step / max(horizon, 1))
    return frame


def collect_dataset(
    task: GridworldTask,
    n_episodes: int,
    horizon: int,
    seed: int = 0,
    policy_backend: MemoPolicyBackend | None = None,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    h, w = task.shape
    n_channels = 2 + task.n_goals + 1

    grids = np.zeros((n_episodes, horizon, n_channels, h, w), dtype=np.float32)
    positions = np.zeros((n_episodes, horizon + 1, 2), dtype=np.int64)
    actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    goals = np.zeros(n_episodes, dtype=np.int64)
    final_posteriors = np.zeros((n_episodes, task.n_goals), dtype=np.float32)
    online_posteriors = np.zeros((n_episodes, horizon, task.n_goals), dtype=np.float32)
    goal_condition_inputs = np.zeros((n_episodes, horizon), dtype=np.int64)

    for episode_idx in range(n_episodes):
        trajectory = sample_trajectory(task, horizon=horizon, rng=rng, policy_backend=policy_backend)
        labels = trajectory_to_observer_labels(task, trajectory, policy_backend=policy_backend)

        positions[episode_idx] = np.array(trajectory.positions, dtype=np.int64)
        actions[episode_idx] = np.array(trajectory.action_indices, dtype=np.int64)
        goals[episode_idx] = int(labels["goal_index"])
        final_posteriors[episode_idx] = labels["final_posterior"].astype(np.float32)
        online_posteriors[episode_idx] = labels["online_posteriors"].astype(np.float32)
        goal_condition_inputs[episode_idx].fill(goals[episode_idx])

        for step in range(horizon):
            grids[episode_idx, step] = encode_grid_frame(
                task=task,
                position=trajectory.positions[step],
                step=step,
                horizon=horizon,
            )

    return DatasetBundle(
        grids=grids,
        positions=positions,
        actions=actions,
        goals=goals,
        final_posteriors=final_posteriors,
        online_posteriors=online_posteriors,
        goal_condition_inputs=goal_condition_inputs,
    )


def save_dataset(path: str | Path, dataset: DatasetBundle) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grids=dataset.grids,
        positions=dataset.positions,
        actions=dataset.actions,
        goals=dataset.goals,
        final_posteriors=dataset.final_posteriors,
        online_posteriors=dataset.online_posteriors,
        goal_condition_inputs=dataset.goal_condition_inputs,
    )
