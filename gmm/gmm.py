#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys

import torch

import numpy as np
import os

from pixyz.distributions import Normal, Categorical
from pixyz.distributions.mixture_distributions import MixtureModel
from pixyz.utils import print_latex

import serket


class GMM(serket.Module):

    @staticmethod
    def sample(mu, var, nb_samples=500):
        """
        Return a tensor of (nb_samples, features), sampled
        from the parameterized gaussian.
        :param mu: torch.Tensor of the means
        :param var: torch.Tensor of variances (NOTE: zero covars.)
        """
        out = []
        for i in range(nb_samples):
            out += [
                torch.normal(mu, var.sqrt())
            ]
        return torch.stack(out, dim=0)

    def __init__(self, K, category=None,
                 name="gmm",
                 obs_nodes=[],
                 nodes=[]):
        super().__init__(name=name, obs_nodes=obs_nodes, nodes=nodes)

        self.__K = K
        self.__category = category

        self.log_likehoods = []
        self.__losses = []

        n_dim = 2

        self.distributions = []
        for i in range(K):
            loc = torch.randn(n_dim)
            scale = torch.empty(n_dim).fill_(0.6)
            normal = Normal(loc=loc, scale=scale, var=["x"], name=f"p_{i:d}")
            self.distributions.append(normal)

        probs = torch.empty(K).fill_(1.0 / K)
        self.prior = Categorical(name="p_{prior}", probs=probs, var=["z"])

        self.model = MixtureModel(name="p", distributions=self.distributions, prior=self.prior)

        self.posterior_model = self.model.posterior()

        cluster1 = self.sample(
            torch.Tensor([1.5, 2.5]),
            torch.Tensor([1.2, .8]),
            nb_samples=150
        )

        cluster2 = self.sample(
            torch.Tensor([7.5, 7.5]),
            torch.Tensor([.75, .5]),
            nb_samples=50
        )

        cluster3 = self.sample(
            torch.Tensor([8, 1.5]),
            torch.Tensor([.6, .8]),
            nb_samples=100
        )
        # create the dummy dataset, by combining the clusters.
        clusters = [cluster1, cluster2, cluster3]
        self.samples = torch.cat(clusters)
        self.samples = (self.samples - self.samples.mean(dim=0)) / self.samples.std(dim=0)
        self.samples_dict = {"x": self.samples}

    def train(self, epoch):
        eps = 1e-6
        min_scale = 1e-6
        for epoch in range(1000):

            # E-step
            posterior = self.posterior_model.prob().eval(self.samples_dict)

            # M-step
            N_k = posterior.sum(dim=1)  # (n_mix,)

            # update probs
            probs = N_k / N_k.sum()  # (n_mix,)
            self.prior.probs[0] = probs

            # update loc & scale
            loc = (posterior[:, None] @ self.samples[None]).squeeze(1)  # (n_mix, n_dim)
            loc /= (N_k[:, None] + eps)

            covariance = (self.samples[None, :, :] - loc[:, None, :]) ** 2  # Covariances are set to 0.
            var = (posterior[:, None, :] @ covariance).squeeze(1)  # (n_mix, n_dim)
            var /= (N_k[:, None] + eps)
            scale = var.sqrt()

            # 各分布のパラメータを更新
            for i, distribution in enumerate(self.distributions):
                distribution.loc[0] = loc[i]
                distribution.scale[0] = scale[i]

            log_likehood = self.model.log_prob().mean().eval({"x": self.samples}).mean()
            if min_scale < log_likehood:
                break
            print("Epoch: {}, log-likelihood: {}".format(epoch + 1, ))

    def update(self):
        pass

# def update(self):
#
#     data = self.get_observations()
#     Pdz = self.get_backward_msg()  # P(z|d)
#
#     N = len(data[0])  # データ数
#
#     # backward messageがまだ計算されていないときは一様分布にする
#     if Pdz is None:
#         Pdz = np.ones((N, self.__K)) / self.__K
#
#     data[0] = np.array(data[0], dtype=np.float32)
#
#     if self.__load_dir is None:
#         save_dir = os.path.join(self.get_name(), "%03d" % self.__n)
#     else:
#         save_dir = os.path.join(self.get_name(), "recog")
#
#     # GMM学習
#     Pdz, mu = gmm.train(data[0], self.__K, self.__itr, save_dir, Pdz, self.__category, self.__load_dir)
#
#     # メッセージの送信
#     self.set_forward_msg(Pdz)
#     self.send_backward_msgs([mu])
