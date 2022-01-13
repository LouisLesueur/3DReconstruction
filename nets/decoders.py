import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.blocks import ResBlock

class DeepSDF(nn.Module):

    def __init__(self, code_dim=128, dim=3, n_layers=8, size=128):

        super().__init__()

        self.code_dim = code_dim
        self.dim = dim

        self.name = "DeepSDF"

        med_layer = n_layers//2

        self.latent_layer = nn.Linear(code_dim, size)
        self.point_layer = nn.Linear(dim, size)

        first_layers = [nn.Linear(size, size)]
        for _ in range(med_layer):
            first_layers.append(nn.ReLU(True))
            first_layers.append(nn.Linear(size,size))
        self.first_layers = nn.Sequential(*first_layers)

        last_layers = []
        for _ in range(n_layers-med_layer-1):
            last_layers.append(nn.Linear(size,size))
            last_layers.append(nn.ReLU(True))
        last_layers.append(nn.Linear(size,1))
        last_layers.append(nn.Tanh())
        self.last_layers = nn.Sequential(*last_layers)


    def forward(self, latent_vector, x):
        
        code = self.latent_layer(latent_vector)
        point = self.point_layer(x)
        
        out = self.first_layers(code + point)
        out = self.last_layers(out+code+point)
        
        return out

class ONetDecoder(nn.Module):
    ''' Decoder class for occupancy network. Inspired by:
    https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/onet/models/decoder.py
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
    '''

    def __init__(self, code_dim=128, dim=3, size=128):
        super().__init__()
        self.code_dim = code_dim
        self.dim = dim

        self.name = "OccupancyNet"

        self.latent_layer = nn.Linear(code_dim, size)
        self.point_layer = nn.Linear(dim, size)

        self.hidden_layers = nn.Sequential(
                ResBlock(size),
                ResBlock(size),
                ResBlock(size),
                ResBlock(size),
                ResBlock(size),
                nn.ReLU(),
                nn.Linear(size, 1)
                )

    def forward(self, latent_vector, x):

        code = self.latent_layer(latent_vector)
        point = self.point_layer(x)

        out = self.hidden_layers(point+code)
        return out
