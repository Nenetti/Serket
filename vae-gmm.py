#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import
import sys

import torch
from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from torch import nn, optim
from torch.nn import functional as F

import serket
import numpy as np

from gmm.gmm import GMM
from vae.vae import VAE

encoder_dim = 128
decoder_dim = 128


# class Inference(Normal):
#     def __init__(self, name, cond_var, var):
#         super(Inference, self).__init__(name=name, var=var, cond_var=cond_var)
#
#         self.fc1 = nn.Linear(x_dim, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc31 = nn.Linear(512, z_dim)
#         self.fc32 = nn.Linear(512, z_dim)
#
#     def forward(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}
#
#
# # generative model p(x|z)
# class Generator(Bernoulli):
#     def __init__(self, name, cond_var, var):
#         super(Generator, self).__init__(name=name, var=var, cond_var=cond_var)
#
#         self.fc1 = nn.Linear(z_dim, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, x_dim)
#
#     def forward(self, z):
#         h = F.relu(self.fc1(z))
#         h = F.relu(self.fc2(h))
#         return {"probs": torch.sigmoid(self.fc3(h))}


def main():
    # encoder = Inference(name="q", var=["z"], cond_var=["x"]).to(device)
    # decoder = Generator(name="p", var=["x"], cond_var=["z"]).to(device)
    # prior = Normal(name="p_{prior}", var=["z"], loc=torch.tensor(0.), scale=torch.tensor(1.)).to(device)
    # kl = KullbackLeibler(encoder, prior)
    #
    # vae = VAE(name="VAE1", encoder=encoder, decoder=decoder, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 1e-3},
    #           obs_nodes=["x"], nodes=["z"])

    # obs = srk.Observation(np.loadtxt("data.txt"))
    # data_category = np.loadtxt("category.txt")

    gmm = GMM(name="GMM1", K=100)
    gmm.train(1)

    # vae1 = vae.VAE(18, itr=200, batch_size=500)
    # gmm1 = gmm.GMM(10, category=data_category)
    #
    # vae1.connect(obs)
    # gmm1.connect(vae1)
    #
    # for i in range(5):
    #     print(i)
    #     vae1.update()
    #     gmm1.update()


if __name__ == "__main__":
    main()
