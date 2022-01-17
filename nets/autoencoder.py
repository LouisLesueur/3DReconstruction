import torch
import torch.nn as nn
from nets.encoders import *
from nets.decoders import *
from torch import distributions as dist

class ONet(nn.Module):

    def __init__(self, code_dim=128):
        super().__init__()

        self.name = "ONet"

        self.encoder = PointNet(code_dim=code_dim)
        self.decoder = ONetDecoder(code_dim=code_dim)

    def forward(self, inputs, x):

        code = self.encoder(inputs)
        logits = self.decoder(code, x)

        return logits
