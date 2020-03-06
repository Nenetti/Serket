#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import
import sys

import torch
import torch.utils.data
from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from torch import nn, optim
from torch.nn import functional as F

import serket
import numpy as np

from gmm.gmm import GMM
from vae.vae import VAE
from torchvision import datasets, transforms

encoder_dim = 128
decoder_dim = 128

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

DEVICE = "cuda"

root = "./data"

BATCH_SIZE = 10000


def load_dataset(transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambd=lambda x: x.view(-1))])

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)

    return train_dataset, test_dataset


def to_data_loader(train_dataset, test_dataset, batch_size=128, **kwargs):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


class Encoder(Normal):
    def __init__(self, name, cond_var, var, x_dim, z_dim):
        super(Encoder, self).__init__(name=name, var=var, cond_var=cond_var)

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


class Decoder(Bernoulli):
    def __init__(self, name, cond_var, var, x_dim, z_dim):
        super(Decoder, self).__init__(name=name, var=var, cond_var=cond_var)

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}


def main():
    x_dim = 784
    z_dim = 2

    train_dataset, test_dataset = load_dataset()

    train_data = train_dataset.data.float().view(train_dataset.data.shape[0], -1)

    train_loader, test_loader = to_data_loader(train_dataset, test_dataset, batch_size=BATCH_SIZE)

    # obs = srk.Observation(np.loadtxt("data.txt"))
    # data_category = np.loadtxt("category.txt")

    cluster1 = GMM.sample(
        torch.Tensor([1.5, 2.5]),
        torch.Tensor([1.2, .8]),
        nb_samples=150
    )

    cluster2 = GMM.sample(
        torch.Tensor([7.5, 7.5]),
        torch.Tensor([.75, .5]),
        nb_samples=50
    )

    cluster3 = GMM.sample(
        torch.Tensor([8, 1.5]),
        torch.Tensor([.6, .8]),
        nb_samples=100
    )

    # create the dummy dataset, by combining the clusters.
    clusters = [cluster1, cluster2, cluster3]
    samples = torch.cat(clusters)
    samples = (samples - samples.mean(dim=0)) / samples.std(dim=0)

    ############################################################################################################################################################
    # モジュール宣言
    ############################################################################################################################################################
    # VAE
    encoder = Encoder(name="q", var=["z"], cond_var=["x"], x_dim=x_dim, z_dim=z_dim).to(DEVICE)
    decoder = Decoder(name="p", var=["x"], cond_var=["z"], x_dim=x_dim, z_dim=z_dim).to(DEVICE)
    prior = Normal(name="p_{prior}", var=["z"], loc=torch.tensor(0.0), scale=torch.tensor(1.0)).to(DEVICE)
    kl = KullbackLeibler(encoder, prior)
    vae = VAE(name="VAE1", encoder=encoder, decoder=decoder,
              regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 1e-3},
              obs_nodes=["x"], nodes=["z1"])

    # GMM
    gmm = GMM(name="GMM1", K=3, n_dim=2)

    ############################################################################################################################################################
    # モジュール接続
    ############################################################################################################################################################
    vae.connect(module=gmm, shared_nodes=["z1"])

    ############################################################################################################################################################
    # 学習
    ############################################################################################################################################################
    for i in range(100):
        for x, label in train_loader:
            # loss = gmm.train({"x": samples})
            train_x = {"x": x.to(DEVICE)}
            vae_loss = vae.train(train_x)
            print(vae_loss)
        z1 = vae.sampling(x=train_data.to(DEVICE))["z"]
        vae.update_params(z1=z1)
        vae.update()

        gmm_log_likehoods = gmm.train({"z": z1})
        print(gmm_log_likehoods)
        gmm.update()

        # samples = vae.sampling(train_x)["z"]

        # print(vae.params.keys())
        # print(i)

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
