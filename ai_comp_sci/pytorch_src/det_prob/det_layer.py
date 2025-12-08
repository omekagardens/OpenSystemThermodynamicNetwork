import torch
import torch.nn as nn
import torch.nn.functional as F

class DetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gamma=1.0, eta=1.0, sigma_init=0.1, F_init=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

        # Free level
        self.F = nn.Parameter(torch.tensor(float(F_init)))

        # Parameterize sigma in log-space so sigma >= 0
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))

        self.gamma = gamma
        self.eta = eta

    def forward(self, x, F_res=1.0):
        x_in = x
        y = F.relu(self.linear(x_in))

        B = x.shape[0]

        # Flux estimates
        G_in = x_in.pow(2).sum(dim=1).sqrt().mean() / (B + 1e-8)
        G_out = y.pow(2).sum(dim=1).sqrt().mean() / (B + 1e-8)

        # Enforce sigma >= 0
        sigma = torch.exp(self.log_sigma)

        # Reservoir flux, potential-driven
        J_res = torch.relu(F_res - self.F) * sigma

        # DET mismatch loss
        det_loss = (self.gamma * G_out - (self.eta * G_in + J_res))**2

        # Safe debug values (detach to avoid warning & accidental autograd weirdness)
        debug = {
            "F": self.F.detach().item(),
            "sigma": sigma.detach().item(),
            "G_in": G_in.detach().item(),
            "G_out": G_out.detach().item(),
            "J_res": J_res.detach().item()
        }

        return y, det_loss, debug