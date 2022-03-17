import torch
import torch.nn as nn
from nets.blocks import *

class PointNet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, code_dim=128, dim=3, hidden_dim=256):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, code_dim)

        self.actvn = nn.ReLU()
        self.pool = nn.MaxPool1d(hidden_dim)

    def forward(self, x):

        x = x.unsqueeze(0)

        net = self.fc_pos(x)
        net = self.block_0(net)
        pooled = self.pool(net).view(-1, self.hidden_dim)

        net = self.block_1(net)
        pooled = self.pool(net).view(-1, self.hidden_dim)

        net = self.block_2(net)
        pooled = self.pool(net).view(-1, self.hidden_dim)

        pooled = self.pool(net).view(-1, self.hidden_dim)

        net = self.block_4(net)
        pooled = self.pool(net).view(-1, self.hidden_dim)

        c = self.fc_c(self.actvn(net))

        return c

class PointNetBasic(nn.Module):
    def __init__(self, code_dim = 128, dim=3, pc_size=300):
        
        super().__init__()
        self.pc_size = 300

        # For plotting
        self.name = "PointNetBasic"

        self.MLP1 = nn.Sequential(
            nn.Conv1d(dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.MLP2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, pc_size, 1),
            nn.BatchNorm1d(pc_size),
            nn.ReLU())

        self.maxpool = nn.MaxPool1d(pc_size)

        self.MLP3 = nn.Sequential(
            nn.Linear(pc_size, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, code_dim))


    def forward(self, input):
        x = self.MLP1(input)
        x = self.MLP2(x)
        x = self.maxpool(x).view(-1, self.pc_size)
        return self.MLP3(x)
