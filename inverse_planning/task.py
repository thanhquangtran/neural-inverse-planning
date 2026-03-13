from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Action = tuple[int, int]
Location = tuple[int, int]

DEFAULT_ACTIONS: tuple[Action, ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True)
class GridworldTask:
    grid: np.ndarray
    init_loc: Location
    goal_locs: tuple[Location, ...]
    beta: float = 2.0
    actions: tuple[Action, ...] = DEFAULT_ACTIONS

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.grid.shape)

    @property
    def n_goals(self) -> int:
        return len(self.goal_locs)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def validate(self) -> None:
        if self.grid.ndim != 2:
            raise ValueError("grid must be 2D")
        if self.grid[self.init_loc]:
            raise ValueError("init_loc cannot be a wall")
        for goal in self.goal_locs:
            if self.grid[goal]:
                raise ValueError(f"goal {goal} cannot be a wall")


def make_default_task(beta: float = 2.0) -> GridworldTask:
    grid = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
        ],
        dtype=bool,
    )
    task = GridworldTask(
        grid=grid,
        init_loc=(4, 2),
        goal_locs=((0, 1), (0, 4), (4, 0)),
        beta=beta,
    )
    task.validate()
    return task


def action_to_index_map(actions: Iterable[Action]) -> dict[Action, int]:
    return {action: idx for idx, action in enumerate(actions)}
