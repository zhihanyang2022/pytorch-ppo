import torch
import torch.nn as nn


class MLPValueFunction(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # self.apply(self.init_weights)

    # @staticmethod
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.orthogonal_(m.weight, gain=1)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)

    def forward(self, states: torch.tensor) -> torch.tensor:
        return self.layers(states)
