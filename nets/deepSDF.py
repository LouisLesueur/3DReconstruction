import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):

    def __init__(self, code_dim=256, dim=3, n_layers=8, size=512):

        super().__init__()

        self.code_dim = code_dim
        self.dim = dim

        self.name = "DeepSDF"

        med_layer = n_layers//2

        first_layers = [nn.Linear(dim+code_dim, size)]
        for _ in range(med_layer):
            first_layers.append(nn.ReLU(True))
            first_layers.append(nn.Linear(size,size))
        first_layers.append(nn.ReLU(True))
        first_layers.append(nn.Linear(size, size-(dim+code_dim)))
        self.first_layers = nn.Sequential(*first_layers)

        last_layers = []
        for _ in range(n_layers-med_layer-1):
            last_layers.append(nn.Linear(size,size))
            last_layers.append(nn.ReLU(True))
        last_layers.append(nn.Linear(size,1))
        last_layers.append(nn.Tanh())
        self.last_layers = nn.Sequential(*last_layers)

        

    def forward(self, latent_vector, x):
        code = latent_vector.repeat(x.shape[0], 1)
        
        data = torch.cat((code, x), dim=1)
        out = self.first_layers(data)

        data = torch.cat((out, data), dim=1)
        out = self.last_layers(data)
        
        return out

    def codes(self):
        return self.latent_vectors

if __name__ == "__main__":
    # Testing the network
    model = DeepSDF(100).to("cuda")

    for p in model.parameters():
        print(p.shape)
