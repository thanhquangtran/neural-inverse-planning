from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from inverse_planning.task import Action, GridworldTask, Location


def in_bounds(grid: np.ndarray, loc: Location) -> bool:
    return 0 <= loc[0] < grid.shape[0] and 0 <= loc[1] < grid.shape[1]


def env_step(task: GridworldTask, loc: Location, action: Action) -> Location:
    next_loc = (loc[0] + action[0], loc[1] + action[1])
    if in_bounds(task.grid, next_loc) and not task.grid[next_loc]:
        return next_loc
    return loc


@lru_cache(maxsize=None)
def distance_map(grid_key: bytes, shape: tuple[int, int], goal: Location) -> np.ndarray:
    grid = np.frombuffer(grid_key, dtype=bool).reshape(shape)
    distances = np.full(shape, np.inf, dtype=np.float64)
    queue: deque[Location] = deque([goal])
    distances[goal] = 0.0

    while queue:
        loc = queue.popleft()
        base = distances[loc]
        for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (loc[0] + delta[0], loc[1] + delta[1])
            if in_bounds(grid, nxt) and not grid[nxt] and distances[nxt] == np.inf:
                distances[nxt] = base + 1.0
                queue.append(nxt)
    return distances


def shortest_path_length(task: GridworldTask, start: Location, goal: Location) -> float:
    distances = distance_map(task.grid.tobytes(), task.shape, goal)
    return float(distances[start])


def compute_q_values(task: GridworldTask, loc: Location, goal: Location) -> np.ndarray:
    distances = distance_map(task.grid.tobytes(), task.shape, goal)
    q_values = np.full(task.n_actions, -np.inf, dtype=np.float64)
    for idx, action in enumerate(task.actions):
        nxt = env_step(task, loc, action)
        if nxt == loc and action != (0, 0):
            continue
        dist = distances[nxt]
        if np.isfinite(dist):
            q_values[idx] = -1.0 - dist
    return q_values


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    total = exps.sum()
    if total == 0.0:
        raise ValueError("softmax received all-zero exponentials")
    return exps / total


def boltzmann_action_probs(task: GridworldTask, loc: Location, goal: Location, beta: float | None = None) -> np.ndarray:
    beta = task.beta if beta is None else beta
    return softmax(beta * compute_q_values(task, loc, goal))
