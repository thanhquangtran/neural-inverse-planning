from __future__ import annotations

import numpy as np

from inverse_planning.memo_backend import MemoPolicyBackend
from inverse_planning.planning import boltzmann_action_probs, env_step
from inverse_planning.simulate import Trajectory
from inverse_planning.task import GridworldTask, action_to_index_map


def logsumexp(values: np.ndarray) -> float:
    vmax = np.max(values)
    return float(vmax + np.log(np.exp(values - vmax).sum()))


def exact_goal_posterior(
    task: GridworldTask,
    action_indices: list[int] | np.ndarray,
    beta: float | None = None,
    policy_backend: MemoPolicyBackend | None = None,
) -> np.ndarray:
    beta = task.beta if beta is None else beta
    action_indices = list(map(int, action_indices))
    goal_logps = np.zeros(task.n_goals, dtype=np.float64)

    for goal_index, goal in enumerate(task.goal_locs):
        loc = task.init_loc
        logp = -np.log(task.n_goals)
        for action_index in action_indices:
            if policy_backend is None:
                probs = boltzmann_action_probs(task, loc, goal, beta=beta)
            else:
                probs = policy_backend.action_probs(loc, goal_index)
            logp += np.log(np.clip(probs[action_index], 1e-12, 1.0))
            loc = env_step(task, loc, task.actions[action_index])
        goal_logps[goal_index] = logp

    return np.exp(goal_logps - logsumexp(goal_logps))


def online_goal_posteriors(
    task: GridworldTask,
    action_indices: list[int] | np.ndarray,
    beta: float | None = None,
    policy_backend: MemoPolicyBackend | None = None,
) -> np.ndarray:
    beta = task.beta if beta is None else beta
    action_indices = list(map(int, action_indices))
    goal_logps = np.full(task.n_goals, -np.log(task.n_goals), dtype=np.float64)
    locs = [task.init_loc for _ in range(task.n_goals)]
    out = np.zeros((len(action_indices), task.n_goals), dtype=np.float64)

    for t, action_index in enumerate(action_indices):
        for goal_index, goal in enumerate(task.goal_locs):
            if policy_backend is None:
                probs = boltzmann_action_probs(task, locs[goal_index], goal, beta=beta)
            else:
                probs = policy_backend.action_probs(locs[goal_index], goal_index)
            goal_logps[goal_index] += np.log(np.clip(probs[action_index], 1e-12, 1.0))
            locs[goal_index] = env_step(task, locs[goal_index], task.actions[action_index])
        out[t] = np.exp(goal_logps - logsumexp(goal_logps))
    return out


def score_goal_conditioned_policy(
    action_probabilities: np.ndarray,
    action_indices: list[int] | np.ndarray,
) -> float:
    action_indices = np.asarray(action_indices, dtype=np.int64)
    step_ids = np.arange(len(action_indices))
    chosen = action_probabilities[step_ids, action_indices]
    return float(np.log(np.clip(chosen, 1e-12, 1.0)).sum())


def posterior_from_goal_conditioned_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.log(len(scores))
    return np.exp(scores - logsumexp(scores))


def online_posteriors_from_goal_conditioned_action_probs(
    action_probabilities_by_goal: np.ndarray,
    action_indices: list[int] | np.ndarray,
) -> np.ndarray:
    action_probabilities_by_goal = np.asarray(action_probabilities_by_goal, dtype=np.float64)
    action_indices = np.asarray(action_indices, dtype=np.int64)
    if action_probabilities_by_goal.ndim != 3:
        raise ValueError("action_probabilities_by_goal must have shape (n_goals, horizon, n_actions)")
    if action_probabilities_by_goal.shape[1] != len(action_indices):
        raise ValueError("horizon dimension must match the length of action_indices")

    n_goals = action_probabilities_by_goal.shape[0]
    goal_logps = np.full(n_goals, -np.log(n_goals), dtype=np.float64)
    out = np.zeros((len(action_indices), n_goals), dtype=np.float64)

    for step, action_index in enumerate(action_indices):
        chosen = action_probabilities_by_goal[:, step, action_index]
        goal_logps += np.log(np.clip(chosen, 1e-12, 1.0))
        out[step] = np.exp(goal_logps - logsumexp(goal_logps))
    return out


def trajectory_to_observer_labels(
    task: GridworldTask,
    trajectory: Trajectory,
    policy_backend: MemoPolicyBackend | None = None,
) -> dict[str, np.ndarray]:
    action_map = action_to_index_map(task.actions)
    action_indices = np.array([action_map[action] for action in trajectory.actions], dtype=np.int64)
    posteriors = online_goal_posteriors(task, action_indices, policy_backend=policy_backend)
    return {
        "goal_index": np.array(trajectory.goal_index, dtype=np.int64),
        "final_posterior": posteriors[-1],
        "online_posteriors": posteriors,
    }
