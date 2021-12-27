import torch

class SDFRegLoss:

    def __init__(self, delta, sigma):
        self.delta = delta
        self.sigma = sigma

    def __call__(self, x1, x2, latent):
        '''
        Clamped L1 loss
        '''
        l1_loss = torch.nn.L1Loss(reduction="sum")
        X1 = torch.clamp(x1, min=-self.delta, max=self.delta)
        X2 = torch.clamp(x2, min=-self.delta, max=self.delta)

        reg = 1/(self.sigma**2) * torch.sum(latent**2)

        return l1_loss(X1, X2) + reg
