from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_planning.memo_backend import MemoPolicyBackend
from inverse_planning.planning import boltzmann_action_probs, env_step
from inverse_planning.task import Action, GridworldTask, Location


@dataclass
class Trajectory:
    goal_index: int
    goal: Location
    positions: list[Location]
    actions: list[Action]
    action_indices: list[int]
    action_probs: list[np.ndarray]


def sample_trajectory(
    task: GridworldTask,
    horizon: int,
    rng: np.random.Generator,
    goal_index: int | None = None,
    policy_backend: MemoPolicyBackend | None = None,
) -> Trajectory:
    if goal_index is None:
        goal_index = int(rng.integers(task.n_goals))
    goal = task.goal_locs[goal_index]
    loc = task.init_loc

    positions = [loc]
    actions: list[Action] = []
    action_indices: list[int] = []
    action_probs: list[np.ndarray] = []

    for _ in range(horizon):
        if policy_backend is None:
            probs = boltzmann_action_probs(task, loc, goal)
        else:
            probs = policy_backend.action_probs(loc, goal_index)
        action_index = int(rng.choice(task.n_actions, p=probs))
        action = task.actions[action_index]
        loc = env_step(task, loc, action)
        positions.append(loc)
        actions.append(action)
        action_indices.append(action_index)
        action_probs.append(probs)

    return Trajectory(
        goal_index=goal_index,
        goal=goal,
        positions=positions,
        actions=actions,
        action_indices=action_indices,
        action_probs=action_probs,
    )
