#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import os

import torch
from pixyz.distributions import Normal, Categorical
from pixyz.distributions.mixture_distributions import MixtureModel

import serket


class GMM(serket.Module):

    def __init__(self, model, prior,
                 name="gmm",
                 obs_nodes=[],
                 nodes=[],
                 eps=1e-6):
        super().__init__(name=name, obs_nodes=obs_nodes, nodes=nodes)

        self.prior = prior
        self.model = model
        self.posterior_model = self.model.posterior()
        self.distributions = self.model.distributions
        self._eps = eps

        self.__log_likehoods = []

    def update(self, epoch=1):
        self._train(epoch)
        self._forward_connections()
        self._backward_connections()

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

    def _train(self, epoch):
        """
        モジュールの学習を行う

        """
        params = self.forward_params.get_params(self.obs_nodes[0])
        # TODO モジュール間の変数名問題
        params.replace_key("z", "x")
        params = params.params
        samples = params["x"]
        with torch.no_grad():
            for e in range(epoch):
                # E-step
                # 負担率(期待値)の計算 z_{nk}
                # データnが分布kに属する確率
                # n ∈ N, k ∈ K (クラスタ数K, データ数N)
                # shape = K * N
                posterior = self.posterior_model.prob().eval(params)
                N_k = posterior.sum(dim=1)

                # 正規化
                probs = N_k / N_k.sum()
                self.prior.probs[0] = probs

                # M-step
                # 最尤解を求める
                loc = (posterior[:, None] @ samples[None]).squeeze(1)  # (n_mix, n_dim)
                loc /= (N_k[:, None] + self._eps)

                covariance = (samples[None, :, :] - loc[:, None, :]) ** 2  # Covariances are set to 0.
                variance = (posterior[:, None, :] @ covariance).squeeze(1)
                variance /= (N_k[:, None] + self._eps)
                scale = variance.sqrt()

                # 各分布のパラメータを更新
                for i, distribution in enumerate(self.distributions):
                    distribution.loc[0] = loc[i]
                    distribution.scale[0] = scale[i]

                log_likehood = self.model.log_prob().mean().eval(params).mean()
                self.__log_likehoods.append(log_likehood)

                # VAEに返すμ_{z2}を計算
                # print(posterior.transpose(1, 0) / N_k)
                # print(torch.mm((posterior / posterior.sum(dim=0)).transpose(1, 0), loc)[0])

                # print(posterior.transpose(1, 0).shape, loc.shape)
                # print(N_k.shape)
                # print(posterior.shape, loc.shape)
                # 60000 * 10, 10 * 2
                loc = torch.mm(posterior.transpose(1, 0), loc)
                scale = torch.mm(posterior.transpose(1, 0), scale)
                self.update_params(loc=loc, scale=scale)

                self.__log_likehoods.append(log_likehood)
