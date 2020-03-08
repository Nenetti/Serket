import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from vae.vae import VAE


class Encoder(Normal):
    def __init__(self, name, cond_var, var, x_dim, z_dim, device="cuda"):
        super(Encoder, self).__init__(name=name, var=var, cond_var=cond_var)

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

        self.to(device)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


class Decoder(Bernoulli):
    def __init__(self, name, cond_var, var, x_dim, z_dim, device="cuda"):
        super(Decoder, self).__init__(name=name, var=var, cond_var=cond_var)

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

        self.to(device)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}


class TestVAE(VAE):

    def __init__(self, name, x_dim, z_dim, data_loader=None, device="cuda"):
        encoder = Encoder(name="q", var=["z"], cond_var=["x"], x_dim=x_dim, z_dim=z_dim, device=device)
        decoder = Decoder(name="p", var=["x"], cond_var=["z"], x_dim=x_dim, z_dim=z_dim, device=device)
        prior = Normal(name="p_{prior}", var=["z"], loc=torch.tensor(0.0), scale=torch.tensor(1.0), features_shape=[z_dim]).to(device)
        kl = KullbackLeibler(encoder, prior)

        super(TestVAE, self).__init__(name=name, encoder=encoder, decoder=decoder, prior=prior, regularizer=kl,
                                      obs_nodes=["x"], nodes=["z1"])

        self.data_loader = data_loader
