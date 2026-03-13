from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from inverse_planning.inference import online_posteriors_from_goal_conditioned_action_probs


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


def evaluate_classifier(model: nn.Module, dataset: np.lib.npyio.NpzFile, device: str = "cpu") -> dict[str, float]:
    model = model.to(device)
    model.eval()
    actions = dataset["actions"]
    start_token = int(actions.max()) + 1
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


def evaluate_policy_model(model: nn.Module, dataset: np.lib.npyio.NpzFile, device: str = "cpu") -> dict[str, float]:
    model = model.to(device)
    model.eval()
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    goals = np.asarray(dataset["goals"], dtype=np.int64)
    target_posteriors = np.asarray(dataset["online_posteriors"], dtype=np.float64)
    goal_ids = np.arange(target_posteriors.shape[-1], dtype=np.int64)
    start_token = int(actions.max()) + 1
    prev_actions = _prepare_prev_actions(actions, start_token)

    final_goal_predictions = np.zeros(len(goals), dtype=np.int64)
    posterior_kl_values = np.zeros(len(goals), dtype=np.float64)
    posterior_kl_by_step = np.zeros(actions.shape[1], dtype=np.float64)

    for episode_index in range(len(goals)):
        frames_t = torch.tensor(dataset["grids"][episode_index : episode_index + 1], dtype=torch.float32, device=device)
        prev_actions_t = torch.tensor(prev_actions[episode_index : episode_index + 1], dtype=torch.long, device=device)
        per_goal_action_probs = []

        with torch.no_grad():
            for goal_index in goal_ids:
                goal_t = torch.tensor([goal_index], dtype=torch.long, device=device)
                logits = model(frames_t, goal_t, prev_actions_t)
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                per_goal_action_probs.append(probs)

        per_goal_action_probs = np.asarray(per_goal_action_probs, dtype=np.float64)
        online = online_posteriors_from_goal_conditioned_action_probs(
            per_goal_action_probs,
            actions[episode_index],
        )
        final_goal_predictions[episode_index] = int(online[-1].argmax())
        per_step_kl = (
            target_posteriors[episode_index]
            * (
                np.log(np.clip(target_posteriors[episode_index], 1e-12, 1.0))
                - np.log(np.clip(online, 1e-12, 1.0))
            )
        ).sum(axis=-1)
        posterior_kl_values[episode_index] = float(per_step_kl.mean())
        posterior_kl_by_step += per_step_kl

    return {
        "final_goal_accuracy": float((final_goal_predictions == goals).mean()),
        "posterior_kl": float(posterior_kl_values.mean()),
        "posterior_kl_by_step": (posterior_kl_by_step / len(goals)).tolist(),
    }
