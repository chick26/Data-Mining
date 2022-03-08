import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.df = df
        self.size = self.df.index.size

    def __len__(self):
        return self.df.index.size

    def __getitem__(self, idx):
        x = self.df.iloc[idx, 0:-1]
        x = np.outer(x, x)
        x = x.reshape(1, x.shape[0], x.shape[1])
        return {'data': torch.from_numpy(x).type(torch.FloatTensor),
                'label': torch.from_numpy(np.array([self.df.iloc[idx, -1]])).type(torch.LongTensor)}
