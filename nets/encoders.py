import torch
import torch.nn as nn

# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class PointNet(nn.Module):
    ''' PointNet-based encoder network, inspired by 
    https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/encoder/pointnet.py.
    Args:
        code_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, code_dim=128, n_pts=300, dim=3, hidden_dim=128):
        super().__init__()
        self.code_dim = code_dim

        self.fc_pos = nn.Linear(n_pts*dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, code_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        
        p = p.view(-1).unsqueeze(0)
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_3(self.actvn(net))

        c = self.fc_c(self.actvn(net))

        return c
