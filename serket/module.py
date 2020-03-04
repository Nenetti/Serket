#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import abc

import numpy as np

from serket.connection import Connection


class Module(object, metaclass=abc.ABCMeta):
    """
    このクラスは各モジュールの抽象クラス。以下の機能を提供します

    各モジュールの「通常計算」「誤差逆伝播」
    モジュール間の「連結」「通信」
    """

    def __init__(self, obs_nodes=[], nodes=[], name="", learnable=True):
        """
        Args:
            obs_nodes(list):    観測ノード名
            nodes(list):        ノード名
            name(str):          モジュール名
            learnable:
        """
        if len(obs_nodes) != len(set(obs_nodes)):
            raise ValueError("観測ノード名が重複しています。 {}".format(obs_nodes))

        if len(nodes) != len(set(nodes)):
            raise ValueError("ノード名が重複しています。 {}".format(nodes))

        all_nodes = set(obs_nodes + nodes)

        self.forward_params = {}
        self.backward_params = {}
        self.obs_nodes = obs_nodes
        self.__obs_nodes = obs_nodes
        self.__nodes = all_nodes
        self.__name = name

        self.__forward_prob = None
        self.__backward_prob = None
        self.__learnable = learnable
        self.__observations = None

        self.forward_connections = {}
        self.params = {}
        self.observations = {}

    @property
    def name(self):
        """
        Returns: str
        """
        return self.__name

    def connect(self, module, shared_nodes):
        """
        このモジュールに別モジュールを接続する
        接続方向はこのモジュールから指定モジュールへの方向

        Args:
            module(Module):             接続するモジュール
            shared_nodes(list or str):  モジュール間で共有するノード
        """
        Connection.register_connection(Connection(parent=self, child=module, shared_nodes=shared_nodes))

    def set_params(self, **params):
        """
        観測データを格納

        Args:
            **params (dict[str, np.ndarray]): 入力データ
        """
        self.observations = params

    def set_forward_msg(self, prob):
        self.__forward_prob = prob

    def get_forward_msg(self):
        return self.__forward_prob

    def get_observations(self):
        return [np.array(o.get_forward_msg()) for o in self.__observations]

    def get_backward_msg(self):
        return self.__backward_prob

    def set_backward_msg(self, prob):
        self.__backward_prob = prob

    def send_backward_msgs(self, probs):
        for i in range(len(self.__observations)):
            self.__observations[i].set_backward_msg(probs[i])

    @abc.abstractmethod
    def update(self):
        raise NotImplementedError()
