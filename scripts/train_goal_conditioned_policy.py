from __future__ import annotations

import argparse

import numpy as np
import torch

from inverse_planning.rnn_models import ModelFactory
from inverse_planning.training import TrainingConfig, build_policy_loader, train_policy_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/goal_conditioned_policy.pt")
    args = parser.parse_args()

    dataset = np.load(args.dataset)
    loader = build_policy_loader(dataset, batch_size=args.batch_size)
    factory = ModelFactory(
        in_channels=dataset["grids"].shape[2],
        n_actions=int(dataset["actions"].max()) + 1,
        n_goals=dataset["final_posteriors"].shape[-1],
    )
    model = factory.build_policy_model()
    losses = train_policy_model(
        model,
        loader,
        TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
        ),
    )
    torch.save(model.state_dict(), args.output)
    print(f"saved model to {args.output}")
    print(f"final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
