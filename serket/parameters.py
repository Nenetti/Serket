import torch
from torch import Tensor


class Parameters():

    def __init__(self, **kwargs):
        """
        Args:
            values (dict[str, Tensor]): key=変数名, value=Tensor型の値の辞書型データ
        """
        self.__params = kwargs

    def add_params(self, **params):
        for key, value in params.items():
            if key in self.__params.keys():
                # if isinstance(value, torch.Tensor):
                self.__params[key] = torch.cat((self.__params[key], value))
            else:
                self.__params[key] = value

    def get_params(self, keys):
        p = Parameters()
        for key in keys:
            p.add_params(**{key: self.__params[key]})

        return p

    def update_params(self, params=None, **kwargs):
        if len(kwargs) != 0:
            if params is None:
                params = Parameters(**kwargs)
            else:
                params.update_params(**kwargs)

        self.__params.update(params.params)

    def get_value(self, key):
        """
        指定の変数名(複数可)のデータを取り出す
        Args:
            keys (list): 変数名のKeyのList
            return_dict (bool): 返り値を辞書型で返すかどうか(FalseならList型)
        Returns:
            list or dict: 指定の変数名のデータを返す
        Examples:
        # >>> get_values({"a":1,"b":2,"c":3}, ["b"])
        # [2]
        # >>> get_values({"a":1,"b":2,"c":3}, ["b", "d"], True)
        # {'b': 2}
        """
        return self.__params[key]

    def replace_keys_split(self, replaces):
        """
        変数名を置き換える
        Args:
            replaces (dict[str, str]): 置き換える変数名の辞書型 (key=置き換え前の変数名, value=置き換え後の変数名)
        Returns:
            dict[str, Tensor] and dict[str, Tensor]: 置き換えた変数名のデータと何の処理もしなかったデータ
        Examples:
        # >>> replace_list_dict = {'a': 'loc'}
        # >>> x_dict = {'a': 0, 'b': 1}
        # >>> print(replace_dict_keys_split(x_dict, replace_list_dict))
        # ({'loc': 0}, {'b': 1})
        """
        replaced_dict = {replaces[key]: value for key, value in self.__params.items() if key in list(replaces.keys())}

        remain_dict = {key: value for key, value in self.__params.items()
                       if key not in list(replaces.keys())}

        return replaced_dict, remain_dict

    def replace_key(self, key, new_key):
        value = self.__params.pop(key)
        self.__params[new_key] = value

    @property
    def params(self):
        return self.__params

    def param_keys(self):
        """
        Keyを返す
        Returns:
            dict_keys: 辞書型のキー
        """
        return self.__params.keys()

    def clear(self):
        self.__params.clear()

    def to(self, device):
        params = {}
        for key in self.__params.keys():
            params[key] = self.__params[key].to(device)
        return Parameters(**params)

    def __str__(self):
        return str(self.__params.keys())

    def __len__(self):
        return max([len(value) for value in self.__params.values()])
