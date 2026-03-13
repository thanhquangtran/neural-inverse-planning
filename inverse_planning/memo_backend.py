from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_planning.task import GridworldTask

try:
    import memo  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    memo = None


@dataclass
class MemoPolicyBackend:
    task: GridworldTask

    def available(self) -> bool:
        return memo is not None

    def build(self) -> None:
        if memo is None:
            raise RuntimeError(
                "memo is not installed. Install the optional dependency set with "
                "`python3 -m pip install -e .[memo]` before using this backend."
            )

    def action_probs(self, _loc: tuple[int, int], _goal_index: int) -> np.ndarray:
        raise NotImplementedError(
            "Hook your memo MDP/IPOMDP policy here after you settle the final "
            "state encoding and planning horizon."
        )
