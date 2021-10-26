from torch.utils.data.dataset import Dataset
from torchvision import datasets


class FR_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return (x, y)

    def __len__(self):
        return self.x.shape[0]


# class M1(datasets.MNIST):
#
#     def __init__(self, max_items=None):
#         self._max_items = max_items
#         super(M1, self).__init__()
#
#     def __len__(self):
#         if self._max_items:
#             return self._max_items
#         return super(M1, self).__len__()

class M2(Dataset):

    def __init__(self, dataset=None, max_items=None):
        self.MNIST = dataset
        self._max_items = max_items

    def __getitem__(self, index):
        data, target = self.MNIST[index]
        reshaped_data = data.reshape(784)
        return reshaped_data, target

    def __len__(self):
        if self._max_items:
            return self._max_items
        return len(self.MNIST)


class M3(Dataset):

    def __init__(self, dataset=None, targets=None, max_items=None):
        self.dataset = dataset
        self.targets = targets
        self._max_items = max_items

    def __getitem__(self, index):
        data = self.dataset[index]
        target = self.targets[index]
        reshaped_data = data.reshape(64 * 64 * 3)
        return reshaped_data, target

    def __len__(self):
        if self._max_items:
            return self._max_items
        return len(self.dataset)
