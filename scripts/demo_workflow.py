from __future__ import annotations

from pathlib import Path

import numpy as np

from inverse_planning.data import collect_dataset
from inverse_planning.inference import exact_goal_posterior
from inverse_planning.memo_backend import MemoPolicyBackend
from inverse_planning.planning import boltzmann_action_probs
from inverse_planning.simulate import sample_trajectory
from inverse_planning.task import make_default_task
from inverse_planning.visualize import render_gridworld_svg


def main() -> None:
    task = make_default_task(beta=2.0)
    memo_backend = MemoPolicyBackend(task)
    memo_backend.build()

    init_loc = task.init_loc
    print("manual vs memo action probabilities at the initial state:")
    for goal_index, goal in enumerate(task.goal_locs):
        manual = boltzmann_action_probs(task, init_loc, goal)
        memo_probs = memo_backend.action_probs(init_loc, goal_index)
        print(
            f"  goal {goal_index} {goal}:",
            "manual =", np.round(manual, 6).tolist(),
            "memo =", np.round(memo_probs, 6).tolist(),
            "max_abs_diff =", float(np.max(np.abs(manual - memo_probs))),
        )

    dataset = collect_dataset(task, n_episodes=4, horizon=8, seed=0, policy_backend=memo_backend)

    print("dataset shapes:")
    print("  grids:", dataset.grids.shape)
    print("  actions:", dataset.actions.shape)
    print("  goals:", dataset.goals.shape)
    print("  online_posteriors:", dataset.online_posteriors.shape)

    first_actions = dataset.actions[0].tolist()
    posterior = exact_goal_posterior(task, first_actions, policy_backend=memo_backend)
    print("first trajectory actions:", first_actions)
    print("exact posterior:", np.round(posterior, 3).tolist())

    trajectory = sample_trajectory(
        task,
        horizon=8,
        rng=np.random.default_rng(1),
        goal_index=0,
        policy_backend=memo_backend,
    )
    svg = render_gridworld_svg(task, trajectory=trajectory, cell_size=64, title="Memo-planned trajectory")
    out_path = Path("artifacts") / "memo_demo_trajectory.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print("wrote visualization:", out_path)


if __name__ == "__main__":
    main()
