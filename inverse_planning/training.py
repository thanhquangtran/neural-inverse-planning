from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"


def _prepare_prev_actions(actions: np.ndarray, start_token: int) -> np.ndarray:
    prev = np.full_like(actions, start_token)
    prev[:, 1:] = actions[:, :-1]
    return prev


def build_classifier_loader(dataset: np.lib.npyio.NpzFile, batch_size: int) -> DataLoader:
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1
    tensors = TensorDataset(
        torch.tensor(dataset["grids"], dtype=torch.float32),
        torch.tensor(_prepare_prev_actions(actions, start_token), dtype=torch.long),
        torch.tensor(dataset["goals"], dtype=torch.long),
        torch.tensor(dataset["online_posteriors"], dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=True)


def build_policy_loader(dataset: np.lib.npyio.NpzFile, batch_size: int) -> DataLoader:
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1
    tensors = TensorDataset(
        torch.tensor(dataset["grids"], dtype=torch.float32),
        torch.tensor(dataset["goal_condition_inputs"], dtype=torch.long),
        torch.tensor(_prepare_prev_actions(actions, start_token), dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=True)


def train_classifier(model: nn.Module, loader: DataLoader, config: TrainingConfig) -> list[float]:
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []

    for _ in range(config.epochs):
        for frames, prev_actions, goals, posteriors in loader:
            frames = frames.to(config.device)
            prev_actions = prev_actions.to(config.device)
            goals = goals.to(config.device)
            posteriors = posteriors.to(config.device)

            out = model(frames, prev_actions)
            goal_loss = F.cross_entropy(out["goal_logits"], goals)
            posterior_loss = F.kl_div(
                F.log_softmax(out["posterior_logits"], dim=-1),
                posteriors,
                reduction="batchmean",
            )
            loss = goal_loss + posterior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return losses


def train_policy_model(model: nn.Module, loader: DataLoader, config: TrainingConfig) -> list[float]:
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []

    for _ in range(config.epochs):
        for frames, goal_ids, prev_actions, targets in loader:
            frames = frames.to(config.device)
            goal_ids = goal_ids.to(config.device)
            prev_actions = prev_actions.to(config.device)
            targets = targets.to(config.device)

            logits = model(frames, goal_ids, prev_actions)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return losses
