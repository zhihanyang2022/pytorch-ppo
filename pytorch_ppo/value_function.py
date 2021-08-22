import torch
import torch.nn as nn


class MLPValueFunction(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, states: torch.tensor) -> torch.tensor:
        return self.layers(states)
