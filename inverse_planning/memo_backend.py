from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from inverse_planning.planning import softmax
from inverse_planning.task import GridworldTask

try:
    import jax
    import jax.numpy as jnp
    from memo import make_module, memo
except ModuleNotFoundError:  # pragma: no cover
    jax = None
    jnp = None
    make_module = None
    memo = None


def _build_q_function(task: GridworldTask):
    if jax is None or jnp is None or memo is None or make_module is None:
        raise RuntimeError("memo backend dependencies are unavailable")

    h, w = task.shape
    n_states = h * w
    n_actions = task.n_actions
    n_goals = task.n_goals
    actions = jnp.asarray(task.actions, dtype=jnp.int32)
    walls = jnp.asarray(task.grid.reshape(-1), dtype=bool)
    goal_states = jnp.asarray([goal[0] * w + goal[1] for goal in task.goal_locs], dtype=jnp.int32)
    S = jnp.arange(n_states)
    A = jnp.arange(n_actions)
    G = jnp.arange(n_goals)

    @jax.jit
    def next_state(s, a):
        row = s // w
        col = s % w
        dr = actions[a, 0]
        dc = actions[a, 1]
        row_ = row + dr
        col_ = col + dc
        in_bounds = (0 <= row_) & (row_ < h) & (0 <= col_) & (col_ < w)
        candidate = row_ * w + col_
        candidate = jnp.clip(candidate, 0, n_states - 1)
        passable = in_bounds & (~walls[candidate])
        return jnp.where(passable, candidate, s)

    @jax.jit
    def valid_action(s, a):
        return (a == 0) | (next_state(s, a) != s)

    @jax.jit
    def tr(s, a, s_):
        return 1.0 * (s_ == next_state(s, a))

    @jax.jit
    def is_goal(s, g):
        return s == goal_states[g]

    def q_value[s: S, a: A, g: G](t):
        agent: knows(s, a, g)
        agent: given(s_ in S, wpp=tr(s, a, s_))
        agent: chooses(a_ in A, to_maximize=0.0 if t < 1 else q_value[s_, a_, g](t - 1))
        return E[
            -1.0 + (0.0 if t < 1 else 0.0 if is_goal(agent.s_, g) else q_value[agent.s_, agent.a_, g](t - 1))
        ] if valid_action(s, a) else -1000000.0

    mod = make_module(f"inverse_planning.memo_backend_{abs(hash((task.grid.tobytes(), task.goal_locs, task.actions)))}")
    mod.jax = jax
    mod.cache = cache
    mod.S = S
    mod.A = A
    mod.G = G
    mod.tr = tr
    mod.valid_action = valid_action
    mod.is_goal = is_goal
    memo(q_value, install_module=mod.install, cache=True)
    return mod.q_value


@dataclass
class MemoPolicyBackend:
    task: GridworldTask
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.horizon is None:
            self.horizon = self.task.grid.size
        self._q_value = None

    def available(self) -> bool:
        return memo is not None

    def build(self) -> None:
        if memo is None:
            raise RuntimeError(
                "memo is not installed. Install the optional dependency set with "
                "`python3 -m pip install -e .[memo]` before using this backend."
            )
        self._q_value = _build_q_function(self.task)

    def _state_index(self, loc: tuple[int, int]) -> int:
        return int(loc[0] * self.task.shape[1] + loc[1])

    def q_values(self, loc: tuple[int, int], goal_index: int) -> np.ndarray:
        if self._q_value is None:
            self.build()
        assert self._q_value is not None

        state_index = self._state_index(loc)
        q_values = np.asarray(self._q_value(int(self.horizon))[state_index, :, goal_index], dtype=np.float64)

        for action_index, action in enumerate(self.task.actions):
            if action == (0, 0):
                continue
            row = loc[0] + action[0]
            col = loc[1] + action[1]
            if not (0 <= row < self.task.shape[0] and 0 <= col < self.task.shape[1]):
                q_values[action_index] = -np.inf
            elif self.task.grid[row, col]:
                q_values[action_index] = -np.inf
        return q_values

    def action_probs(self, _loc: tuple[int, int], _goal_index: int) -> np.ndarray:
        return softmax(self.task.beta * self.q_values(_loc, _goal_index))
