import torch
from torch import nn


class WideAndDeepRanker(nn.Module):
    def __init__(self, input_dim, deep_hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)

        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, x):
        wide_score = self.wide(x)
        deep_score = self.deep(x)
        return (wide_score + deep_score).squeeze(-1)