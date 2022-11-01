import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=16, hidden_units=1000) -> None:

        nn.Module.__init__(self)


        self.weights_dl_1 = torch.nn.Parameter(torch.randn(hidden_units, latent_dim))
        self.bias_dl_1 = torch.nn.Parameter(torch.randn(hidden_units))
        self.layers = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, 2),
        )
        self.depth = len(self.layers)
        self.init_weights()
       

    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i], nn.Conv2d) or isinstance(self.layers[i], nn.Linear):
                self.layers[i].weight.data.normal_(0, 0.01)
                nn.init.constant_(self.layers[i].bias.data, 0)
        self.weights_dl_1.data.normal_(0, 0.01)
        nn.init.constant_(self.bias_dl_1.data, 0)
    

    def forward(self, z: torch.Tensor):
        z = F.linear(z, self.weights_dl_1, self.bias_dl_1)
        return self.layers(z)
