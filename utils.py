import torch

class SDFRegLoss:

    def __init__(self, delta, sigma):
        self.delta = delta
        self.sigma = sigma
        self.shape_loss = 0
        self.reg_loss = 0

    def __call__(self, x1, x2, latent):
        '''
        Clamped L1 loss
        '''
        l1_loss = torch.nn.L1Loss(reduction="sum")
        X1 = torch.clamp(x1, min=-self.delta, max=self.delta)
        X2 = torch.clamp(x2, min=-self.delta, max=self.delta)


        self.shape_loss = l1_loss(X1, X2)
        self.reg_loss = self.sigma * (torch.norm(latent)**2)

        return self.shape_loss + self.reg_loss
