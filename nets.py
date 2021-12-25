import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):

    def __init__(self, code_dim=1, dim=3, n_layers=8, size=512):

        super().__init__()

        self.code_dim = code_dim
        self.dim = dim

        self.name = "DeepSDF"
        
        self.input_layer = nn.utils.weight_norm(nn.Linear(dim+code_dim, size))

        layers = []
        for _ in range(n_layers):
            layers.append(nn.utils.weight_norm(nn.Linear(size,size)))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.utils.weight_norm(nn.Linear(size, 1))

    def forward(self, x):
        out = self.input_layer(x)
        out = F.relu(out)
        out = self.output_layer(out)
        return torch.tanh(out)
