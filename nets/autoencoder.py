import torch
import torch.nn as nn
from nets.encoders import *
from nets.decoders import *
from torch import distributions as dist

class ONet(nn.Module):

    def __init__(self, code_dim=128, pc_size=300):
        super().__init__()

        self.name = "ONet"

#        self.encoder = PointNet(code_dim=code_dim)
        self.encoder = PointNetBasic(code_dim=code_dim, pc_size=pc_size)
        self.decoder = ONetDecoder(code_dim=code_dim)

    def forward(self, inputs, x):

        code = self.encoder(inputs)
        logits = self.decoder(code, x)

        return logits
