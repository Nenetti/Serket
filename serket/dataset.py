import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super().__init__()
        self.data = {}
        self.length = 0
        for key, value in kwargs.items():
            self.data[key] = value
            self.length = len(value)

    def __getitem__(self, index):
        data = {}
        for key, value in self.data.items():
            data[key] = value[index]
        return data

    def get_item(self, vars, index):
        data = {}
        for var in vars:
            data[var] = self.data[var][index]
        return data

    def __len__(self):
        return self.length