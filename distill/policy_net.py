import torch.nn as nn
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=400, hidden=512, actions=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )
    def forward(self, x): return self.net(x)
