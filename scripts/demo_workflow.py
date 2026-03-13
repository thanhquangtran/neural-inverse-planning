from __future__ import annotations

import numpy as np

from inverse_planning.data import collect_dataset
from inverse_planning.inference import exact_goal_posterior
from inverse_planning.task import make_default_task


def main() -> None:
    task = make_default_task(beta=2.0)
    dataset = collect_dataset(task, n_episodes=4, horizon=8, seed=0)

    print("dataset shapes:")
    print("  grids:", dataset.grids.shape)
    print("  actions:", dataset.actions.shape)
    print("  goals:", dataset.goals.shape)
    print("  online_posteriors:", dataset.online_posteriors.shape)

    first_actions = dataset.actions[0].tolist()
    posterior = exact_goal_posterior(task, first_actions)
    print("first trajectory actions:", first_actions)
    print("exact posterior:", np.round(posterior, 3).tolist())


if __name__ == "__main__":
    main()
