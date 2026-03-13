from __future__ import annotations

import argparse

from inverse_planning.data import collect_dataset, save_dataset
from inverse_planning.task import make_default_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = make_default_task(beta=args.beta)
    dataset = collect_dataset(task, n_episodes=args.episodes, horizon=args.horizon, seed=args.seed)
    save_dataset(args.output, dataset)
    print(f"saved dataset to {args.output}")


if __name__ == "__main__":
    main()
