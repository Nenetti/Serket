#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys

from torch import optim

sys.path.append("../")

import pixyz.models.vae
from pixyz.distributions.distributions import DistributionBase
import serket
import numpy as np
import os


class VAE(pixyz.models.VAE, serket.Module):
    """
    VAE
    """

    def __init__(self, encoder, decoder,
                 other_distributions=[],
                 regularizer=None,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None,

                 obs_nodes=[],
                 nodes=[],
                 name="vae"):

        """
        Args:

            # pixyzの初期化パラメータ
            encoder (DistributionBase):
            decoder (DistributionBase):
            other_distributions (list[DistributionBase]):
            regularizer (torch.losses.Loss):
            optimizer (torch.optim):
            optimizer_params (dict):
            clip_grad_norm (float or int):
            clip_grad_value (float or int):

            # SERKETの初期化パラメータ
            obs_nodes (list[str]): 観測ノード名
            nodes (list[str]): 全ノード名
            name (str): モジュール名

        """

        # pixyzのVAEを初期化
        pixyz.models.VAE.__init__(self, encoder=encoder, decoder=decoder, other_distributions=other_distributions,
                                  regularizer=regularizer, optimizer=optimizer, optimizer_params=optimizer_params,
                                  clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

        # SERKETのモジュールを初期化
        serket.Module.__init__(self, obs_nodes=obs_nodes, nodes=nodes, name=name)

        self.encoder = encoder
        self.decoder = decoder

        self.node_keys = encoder.var + encoder.cond_var + encoder.params_keys

        self.__losses = []

    def set_loss_func(self, loss):
        self.set_loss(loss)

    def train(self, train_x_dict={}, **kwargs):
        """
        モジュールの学習を行う

        Args:
            train_x_dict (dict[str, torch.Tensor]): 入力データ

        """
        loss = super().train(train_x_dict=train_x_dict, **kwargs)
        self.params = self.encoder.params
        self.__losses.append(loss)
        return loss

    def test(self, train_x_dict={}, **kwargs):
        """
        モジュールのテストを行う

        Args:
            train_x_dict (dict[str, torch.Tensor]): 入力データ

        """
        loss = super().test(train_x_dict=train_x_dict)
        self.params = self.encoder.params
        self.__losses.append(loss)
        return loss

    def update(self):
        """
        接続モジュールにパラメータを送る
        """
        for module, shared_nodes in self.forward_connections.items():
            # 共有されるノードの値のみ取得して対象モジュールのパラメータを更新
            params = self.get_params(shared_nodes)
            module.update_params(params)

    def get_params(self, node_names):
        """
        指定のパラメータ名の値を返す

        Args:
            node_names (list[str]): ノード名

        Returns:
            dict[str, Tensor]

        """
        params = {}
        for name in node_names:
            params[name] = self.params[name]
        return params

    def save_result(self, save_dir):
        """
        学習結果を保存

        Args:
            save_dir: 保存先のディレクトリ

        """
        np.savetxt(os.path.join(save_dir, "loss.txt"), self.__losses)
        for name, value in self.params.items():
            np_value = value.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"{name}.npy"), np_value)
