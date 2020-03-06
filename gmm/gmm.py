#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import os

import torch
from pixyz.distributions import Normal, Categorical
from pixyz.distributions.mixture_distributions import MixtureModel

import serket


class GMM(serket.Module):

    def __init__(self, K, n_dim,
                 category=None,
                 name="gmm",
                 obs_nodes=[],
                 nodes=[],
                 eps=1e-6,
                 min_scale=1e-6):

        super().__init__(name=name, obs_nodes=obs_nodes, nodes=nodes)

        self.__K = K
        self.__category = category

        self.__log_likehoods = []

        self.distributions = []
        for i in range(K):
            loc = torch.randn(n_dim)
            scale = torch.empty(n_dim).fill_(0.6)
            normal = Normal(name=f"p_{i:d}", var=["x"], loc=loc, scale=scale)
            self.distributions.append(normal)

        probs = torch.empty(K).fill_(1.0 / K)
        self.prior = Categorical(name="p_{prior}", var=["z"], probs=probs)

        self.model = MixtureModel(name="p", distributions=self.distributions, prior=self.prior)

        self.posterior_model = self.model.posterior()

        self.eps = eps
        self.min_scale = min_scale

    def train(self, train_x_dict={}, **kwargs):
        """
        モジュールの学習を行う

        Args:
            train_x_dict (dict[str, torch.Tensor]): 入力データ

        """
        samples = train_x_dict["z"].detach().cpu()
        train_x_dict = {"x": samples}
        with torch.no_grad():
            # E-step
            # 負担率(期待値)の計算 z_{nk}
            # データnが分布kに属する確率
            # n ∈ N, k ∈ K (クラスタ数K, データ数N)
            # shape = K * N
            posterior = self.posterior_model.prob().eval(train_x_dict)
            N_k = posterior.sum(dim=1)

            # 正規化
            probs = N_k / N_k.sum()
            self.prior.probs[0] = probs

            # M-step
            # 最尤解を求める
            loc = (posterior[:, None] @ samples[None]).squeeze(1)  # (n_mix, n_dim)
            loc /= (N_k[:, None] + self.eps)

            covariance = (samples[None, :, :] - loc[:, None, :]) ** 2  # Covariances are set to 0.
            variance = (posterior[:, None, :] @ covariance).squeeze(1)
            variance /= (N_k[:, None] + self.eps)
            scale = variance.sqrt()

            # 各分布のパラメータを更新
            for i, distribution in enumerate(self.distributions):
                distribution.loc[0] = loc[i]
                distribution.scale[0] = scale[i]

            log_likehood = self.model.log_prob().mean().eval({"x": samples}).mean()
            self.__log_likehoods.append(log_likehood)

            return log_likehood

    def test(self, train_x_dict={}, **kwargs):
        """
        モジュールのテストを行う

        Args:
            train_x_dict (dict[str, torch.Tensor]): 入力データ

        """
        self.train(train_x_dict=train_x_dict, **kwargs)

    def save_result(self, save_dir):
        """
        学習結果を保存

        Args:
            save_dir: 保存先のディレクトリ

        """
        np.savetxt(os.path.join(save_dir, "log_likehood.txt"), self.__log_likehoods)
        for name, value in self.params.items():
            np_value = value.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"{name}.npy"), np_value)

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
