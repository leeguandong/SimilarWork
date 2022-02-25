import numpy as np
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class SingleDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = list(range(len(self.dataset)))

    def shuffle(self):
        np.random.shuffle(self.index)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError("inedx out of range")
        return self.dataset[self.index[idx]]


class BatchDataset(Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = list(range(len(self.dataset)))

    def shuffle(self):
        np.random.shuffle(self.index)

    def __len__(self):
        return self.dataset.__len__() // self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("index out of range")
        return self.dataset[self.index[idx:idx + self.batch_size]]
