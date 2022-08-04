import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPModel(nn.Module):

    def __init__(self, layer, dp_rate=0.1):
        super().__init__()
        layers = []
        for idx in range(len(layer) - 1):
            layers += [
                nn.Linear(layer[idx], layer[idx + 1]),
                nn.BatchNorm1d(layer[idx + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)