import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):

    def __init__(self, code_dim=1, dim=3, n_layers=8):

        super().__init__()

        self.code_dim = code_dim
        self.dim = dim
        
        self.input_layer = nn.Linear(dim+code_dim, 512)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(512,512))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(512, 1)

    def forward(self, x):
        out = self.input_layer(x)
        out = F.relu(out)
        out = self.output_layer(out)

        return F.tanh(out)
