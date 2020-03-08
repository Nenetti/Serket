#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from serket import Connection
from serket.parameters import Parameters

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

    def __init__(self,
                 encoder, decoder, prior,
                 other_distributions=[],
                 regularizer=None,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None,
                 device="cuda",
                 name="vae",
                 obs_nodes=[],
                 nodes=[]):
        """
        Args:

            # pixyzの初期化パラメータ
            encoder (DistributionBase):
            decoder (DistributionBase):
            prior (Normal):
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
        serket.Module.__init__(self, name=name, obs_nodes=obs_nodes, nodes=nodes)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.node_keys = encoder.var + encoder.cond_var + encoder.params_keys

        self.__losses = []

        self.data_loader = None

        self._device = device

    def set_data(self, data_loader):
        self.data_loader = data_loader

    def update(self, input_vars, epoch=1):
        self._train(input_vars=input_vars, epoch=epoch)
        self._sampling_latent_variable(input_vars=input_vars)
        self._forward_connections()
        self._backward_connections()

    def _update_prior(self, loc=None, scale=None, start=0, stop=-1):
        if loc is not None:
            self.prior.register_buffer("loc", loc[start:stop])
        if scale is not None:
            self.prior.register_buffer("scale", scale[start:stop])

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

    def _train(self, input_vars, epoch, **kwargs):
        """
        モジュールの学習を行う

        Args:

        """
        self._parameters.clear()

        losses = []
        batch_size = self.data_loader.batch_size

        for e in range(epoch):
            batch_losses = []
            for i, x in enumerate(self.data_loader):
                x = Parameters(**x).get_params(input_vars).to(self._device)
                self._update_prior(**self.backward_params.params, start=i * batch_size, stop=i * batch_size + len(x))
                loss = super().train(x.params, **kwargs).item()
                batch_losses.append(loss)

            losses.append(np.array(batch_losses).mean())

        self.__losses += losses

    def _sampling_latent_variable(self, input_vars):
        with torch.no_grad():
            for x in self.data_loader:
                x = Parameters(**x).get_params(input_vars).to(self._device)
                params = self.encoder.get_params(x.params)
                params[self.encoder.var[0]] = self.encoder.distribution_torch_class(**params).sample()
                self._parameters.add_params(**params)
