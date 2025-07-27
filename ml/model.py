import torch.nn as nn

class HeartModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)