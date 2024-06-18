# create a loss class, using monai mse at first

from monai.losses import MSELoss, L1Loss
import torch

class Loss:
    def __init__(self):
        self.recon_loss = MSELoss()
        self.commitment_loss = L1Loss()

    def __call__(self, x: torch.FloatTensor, y: torch.FloatTensor, diff: torch.FloatTensor) -> torch.FloatTensor:
        recon_loss = self.recon_loss(x, y)
        commitment_loss = self.commitment_loss(diff)

        return recon_loss + commitment_loss
