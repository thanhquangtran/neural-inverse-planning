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


def build_classifier_loader(
    dataset: np.lib.npyio.NpzFile,
    batch_size: int,
    start_token: int | None = None,
) -> DataLoader:
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1 if start_token is None else start_token
    tensors = TensorDataset(
        torch.tensor(dataset["grids"], dtype=torch.float32),
        torch.tensor(_prepare_prev_actions(actions, start_token), dtype=torch.long),
        torch.tensor(dataset["goals"], dtype=torch.long),
        torch.tensor(dataset["online_posteriors"], dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=True)


def build_policy_loader(
    dataset: np.lib.npyio.NpzFile,
    batch_size: int,
    start_token: int | None = None,
) -> DataLoader:
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1 if start_token is None else start_token
    tensors = TensorDataset(
        torch.tensor(dataset["grids"], dtype=torch.float32),
        torch.tensor(dataset["goals"], dtype=torch.long),
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


def evaluate_classifier(
    model: nn.Module,
    dataset: np.lib.npyio.NpzFile,
    device: str = "cpu",
    start_token: int | None = None,
) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1 if start_token is None else start_token
    frames = torch.tensor(dataset["grids"], dtype=torch.float32, device=device)
    prev_actions = torch.tensor(_prepare_prev_actions(actions, start_token), dtype=torch.long, device=device)
    goals = np.asarray(dataset["goals"], dtype=np.int64)
    target_posteriors = np.asarray(dataset["online_posteriors"], dtype=np.float64)

    with torch.no_grad():
        out = model(frames, prev_actions)
        goal_probs = torch.softmax(out["goal_logits"], dim=-1).detach().cpu().numpy()
        posterior_probs = torch.softmax(out["posterior_logits"], dim=-1).detach().cpu().numpy()

    final_goal_accuracy = float((goal_probs.argmax(axis=-1) == goals).mean())
    per_step_kl = (
        target_posteriors
        * (
            np.log(np.clip(target_posteriors, 1e-12, 1.0))
            - np.log(np.clip(posterior_probs, 1e-12, 1.0))
        )
    ).sum(axis=-1)
    posterior_kl = float(per_step_kl.mean())
    return {
        "final_goal_accuracy": final_goal_accuracy,
        "posterior_kl": posterior_kl,
        "posterior_kl_by_step": per_step_kl.mean(axis=0).tolist(),
    }


def evaluate_policy_model(
    model: nn.Module,
    dataset: np.lib.npyio.NpzFile,
    device: str = "cpu",
    batch_size: int = 256,
    start_token: int | None = None,
) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    goals = np.asarray(dataset["goals"], dtype=np.int64)
    target_posteriors = np.asarray(dataset["online_posteriors"], dtype=np.float64)
    n_episodes, horizon = actions.shape
    n_goals = target_posteriors.shape[-1]
    start_token = int(actions.max()) + 1 if start_token is None else start_token
    prev_actions = _prepare_prev_actions(actions, start_token)

    final_goal_predictions = []
    posterior_kl_values = []
    posterior_kl_by_step = np.zeros(horizon, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_episodes, batch_size):
            stop = min(start + batch_size, n_episodes)
            batch_len = stop - start

            frames_t = torch.tensor(dataset["grids"][start:stop], dtype=torch.float32, device=device)
            prev_actions_t = torch.tensor(prev_actions[start:stop], dtype=torch.long, device=device)
            actions_t = torch.tensor(actions[start:stop], dtype=torch.long, device=device)

            # Evaluate every candidate goal in parallel.  Shape convention:
            #   frames_rep:       (batch * n_goals, horizon, channels, height, width)
            #   goal_ids_rep:     (batch * n_goals,)
            #   prev_actions_rep: (batch * n_goals, horizon)
            frames_rep = (
                frames_t[:, None]
                .expand(batch_len, n_goals, *frames_t.shape[1:])
                .reshape(batch_len * n_goals, *frames_t.shape[1:])
            )
            prev_actions_rep = (
                prev_actions_t[:, None]
                .expand(batch_len, n_goals, *prev_actions_t.shape[1:])
                .reshape(batch_len * n_goals, *prev_actions_t.shape[1:])
            )
            goal_ids_rep = (
                torch.arange(n_goals, dtype=torch.long, device=device)
                .repeat(batch_len)
            )

            logits = model(frames_rep, goal_ids_rep, prev_actions_rep)
            action_probs = torch.softmax(logits, dim=-1).reshape(batch_len, n_goals, horizon, -1)

            chosen = action_probs.gather(
                dim=-1,
                index=actions_t[:, None, :, None].expand(batch_len, n_goals, horizon, 1),
            ).squeeze(-1)
            goal_logps = torch.log(chosen.clamp_min(1e-12)).cumsum(dim=-1) - np.log(n_goals)
            online = torch.softmax(goal_logps, dim=1).permute(0, 2, 1)

            target_t = torch.tensor(target_posteriors[start:stop], dtype=torch.float64, device=device)
            online_t = online.to(torch.float64).clamp_min(1e-12)
            per_step_kl_t = (
                target_t
                * (
                    torch.log(target_t.clamp_min(1e-12))
                    - torch.log(online_t)
                )
            ).sum(dim=-1)

            final_goal_predictions.append(online[:, -1].argmax(dim=-1).detach().cpu().numpy())
            per_step_kl = per_step_kl_t.detach().cpu().numpy()
            posterior_kl_values.append(per_step_kl.mean(axis=1))
            posterior_kl_by_step += per_step_kl.sum(axis=0)

    final_goal_predictions = np.concatenate(final_goal_predictions, axis=0)
    posterior_kl_values = np.concatenate(posterior_kl_values, axis=0)

    return {
        "final_goal_accuracy": float((final_goal_predictions == goals).mean()),
        "posterior_kl": float(posterior_kl_values.mean()),
        "posterior_kl_by_step": (posterior_kl_by_step / n_episodes).tolist(),
    }
