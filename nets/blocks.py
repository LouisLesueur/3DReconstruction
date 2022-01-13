import torch
import torch.nn as nn

class ResBlock(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size (int): dimension
    '''

    def __init__(self, size):
        super().__init__()
        # Attributes
        self.size = size
        
        # Submodules
        self.layer1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(size, size)
                )

        self.layer2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(size, size)
                )

        self.bn = nn.BatchNorm1d(size)

    def forward(self, x):
        x_s = self.layer1(x)
        dx = self.layer2(x_s)

        return self.bn(x_s + dx)
