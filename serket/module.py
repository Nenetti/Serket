#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import abc
import os

import numpy as np

from serket.connection import Connection
from serket.parameters import Parameters


class Module(object, metaclass=abc.ABCMeta):
    """
    このクラスは各モジュールの抽象クラス。以下の機能を提供します

    各モジュールの「通常計算」「誤差逆伝播」
    モジュール間の「連結」「通信」
    """

    def __init__(self, obs_nodes=[], nodes=[], name=""):
        """
        Args:
            obs_nodes(list):    観測ノード名
            nodes(list):        ノード名
            name(str):          モジュール名
        """
        if len(obs_nodes) != len(set(obs_nodes)):
            raise ValueError("観測ノード名が重複しています。 {}".format(obs_nodes))

        if len(nodes) != len(set(nodes)):
            raise ValueError("ノード名が重複しています。 {}".format(nodes))

        all_nodes = set(obs_nodes + nodes)

        self.obs_nodes = obs_nodes
        self.__nodes = all_nodes
        self.__name = name

        self._parameters = Parameters()
        self.forward_params = Parameters()
        self.backward_params = Parameters()

    @property
    def name(self):
        """
        Returns:
            str
        """
        return self.__name

    def connect(self, module, forward, backward=[]):
        """
        このモジュールに別モジュールを接続する
        接続方向はこのモジュールから指定モジュールへの方向

        Args:
            module (Module):             接続するモジュール
            forward (list or str):  モジュール間で共有するノード
        """
        Connection.register_connection(Connection(parent=self, child=module, forward=forward, backward=backward))

    def update_params(self, params=None, **kwargs):
        self._parameters.update_params(params, **kwargs)

    def get_params(self, keys):
        """
        Args:
            keys (list[str]):

        Returns:
            Parameters:

        """
        return self._parameters.get_params(keys)

    @property
    def params(self):
        """

        Returns:
            Parameters:
        """
        return self._parameters

    @abc.abstractmethod
    def update(self, **kwagrs):
        raise NotImplementedError()

    def save(self, result_dir, model_dir):
        result_path = os.path.join(result_dir, self.name)
        model_path = os.path.join(model_dir, self.name)

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self._save_result(result_path)
        self._save_model(model_path)

    def load(self, result_dir, model_dir):
        result_path = os.path.join(result_dir, self.name)
        model_path = os.path.join(model_dir, self.name)

        self._load_result(result_path)
        self._load_model(model_path)

    @abc.abstractmethod
    def _save_result(self, save_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def _load_result(self, load_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_model(self, save_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def _load_model(self, load_path):
        raise NotImplementedError()

    # @abc.abstractmethod
    # def load_parameters(self, **kwargs):
    #     raise NotImplementedError()

    def _forward_connections(self):
        connections = Connection.get_forward_connections(self)
        for connection in connections:
            connection.forward()

    def _backward_connections(self):
        connections = Connection.get_backward_connections(self)
        for connection in connections:
            connection.backward()
