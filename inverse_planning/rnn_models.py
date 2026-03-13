from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionHistoryGRU(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, n_goals: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
        )
        self.action_embed = nn.Embedding(n_actions + 1, 16)
        self.rnn = nn.GRU(hidden_dim + 16, hidden_dim, batch_first=True)
        self.goal_head = nn.Linear(hidden_dim, n_goals)
        self.posterior_head = nn.Linear(hidden_dim, n_goals)

    def forward(self, frames: torch.Tensor, prev_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps = frames.shape[:2]
        frame_features = self.frame_encoder(frames.reshape(batch * steps, *frames.shape[2:]))
        frame_features = frame_features.reshape(batch, steps, -1)
        action_features = self.action_embed(prev_actions)
        features = torch.cat([frame_features, action_features], dim=-1)
        outputs, _ = self.rnn(features)
        last = outputs[:, -1]
        return {
            "goal_logits": self.goal_head(last),
            "posterior_logits": self.posterior_head(outputs),
        }


class ConvGRUClassifier(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, n_goals: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, hidden_dim)
        self.action_embed = nn.Embedding(n_actions + 1, 16)
        self.rnn = nn.GRU(hidden_dim + 16, hidden_dim, batch_first=True)
        self.goal_head = nn.Linear(hidden_dim, n_goals)
        self.posterior_head = nn.Linear(hidden_dim, n_goals)

    def forward(self, frames: torch.Tensor, prev_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps = frames.shape[:2]
        encoded = self.encoder(frames.reshape(batch * steps, *frames.shape[2:]))
        encoded = encoded.reshape(batch, steps, -1)
        action_features = self.action_embed(prev_actions)
        outputs, _ = self.rnn(torch.cat([encoded, action_features], dim=-1))
        last = outputs[:, -1]
        return {
            "goal_logits": self.goal_head(last),
            "posterior_logits": self.posterior_head(outputs),
        }


class GoalConditionedPolicyRNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, n_goals: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, hidden_dim)
        self.goal_embed = nn.Embedding(n_goals, 16)
        self.prev_action_embed = nn.Embedding(n_actions + 1, 16)
        self.rnn = nn.GRU(hidden_dim + 32, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        frames: torch.Tensor,
        goal_ids: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        encoded = self.encoder(frames.reshape(batch * steps, *frames.shape[2:]))
        encoded = encoded.reshape(batch, steps, -1)
        goal_features = self.goal_embed(goal_ids).unsqueeze(1).expand(-1, steps, -1)
        action_features = self.prev_action_embed(prev_actions)
        outputs, _ = self.rnn(torch.cat([encoded, goal_features, action_features], dim=-1))
        return self.policy_head(outputs)


@dataclass(frozen=True)
class ModelFactory:
    in_channels: int
    n_actions: int
    n_goals: int
    hidden_dim: int = 128

    def build_classifier(self, variant: str) -> nn.Module:
        if variant == "action_gru":
            return ActionHistoryGRU(self.in_channels, self.n_actions, self.n_goals, self.hidden_dim)
        if variant == "conv_gru":
            return ConvGRUClassifier(self.in_channels, self.n_actions, self.n_goals, self.hidden_dim)
        raise ValueError(f"Unknown classifier variant: {variant}")

    def build_policy_model(self) -> nn.Module:
        return GoalConditionedPolicyRNN(self.in_channels, self.n_actions, self.n_goals, self.hidden_dim)
