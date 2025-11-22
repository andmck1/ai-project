import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 2**10)
        self.layer2 = nn.Linear(2**10, 2**10)
        self.layer3 = nn.Linear(2**10, 2**7)
        self.layer4 = nn.Linear(2**7, n_actions)

    def forward(self, x: torch.Tensor) -> nn.Linear:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
