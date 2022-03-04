import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.blocks import *

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
            first_layers.append(nn.ReLU())
            first_layers.append(nn.Linear(size,size))
        self.first_layers = nn.Sequential(*first_layers)

        last_layers = []
        for _ in range(n_layers-med_layer-1):
            last_layers.append(nn.Linear(size,size))
            last_layers.append(nn.ReLU())
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

    def __init__(self, dim=3, code_dim=128,
                 hidden_size=256):
        super().__init__()

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(code_dim, hidden_size)
        self.block1 = CResnetBlockConv1d(code_dim, hidden_size)
        self.block2 = CResnetBlockConv1d(code_dim, hidden_size)
        self.block3 = CResnetBlockConv1d(code_dim, hidden_size)
        self.block4 = CResnetBlockConv1d(code_dim, hidden_size)

        self.bn = CBatchNorm1d(code_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, latent_vector, x):

        x = x.T.unsqueeze(0)
        
        net = self.fc_p(x)

        net = self.block0(net, latent_vector)
        net = self.block1(net, latent_vector)
        net = self.block2(net, latent_vector)
        net = self.block3(net, latent_vector)
        net = self.block4(net, latent_vector)

        out = self.fc_out(self.actvn(self.bn(net, latent_vector)))

        return out.squeeze(0).squeeze(0)
