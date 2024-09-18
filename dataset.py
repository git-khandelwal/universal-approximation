import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class customDataset(Dataset):
    def __init__(self, funcApprox, limits, input_size) -> None:
        super().__init__()
        assert type(limits) == int
        assert type(input_size) == float
        self.x = np.random.uniform(-1 * input_size, input_size, limits)
        self.x = self.x[:, np.newaxis]
        self.y = np.array([funcApprox(val) for val in self.x]).reshape(-1)    

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        return torch.tensor(x,dtype = torch.float32), torch.tensor(y, dtype = torch.float32)

class csvDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('dataset.csv',header=None)
        print(self.data.iloc[:,:-1].shape)
        print(self.data.iloc[:,-1].shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x = self.data.iloc[idx,:-1]
        y = self.data.iloc[idx,-1]

        return torch.tensor(x.values,dtype = torch.float32), torch.tensor(y, dtype = torch.float32)


# d = mDataset()
# dd = DataLoader(d, batch_size=64, shuffle=True)
# for a,b in dd:
#     print(a.shape)
#     print(b.shape)

# exit()

# def funcApprox(x):
#     return np.sin(x) - x + 2* x**2

# c = customDataset(funcApprox)
# data = DataLoader(c,batch_size=1,shuffle=True)
# for a,b in data:
#     print(a.shape)
#     print(b.shape)

# csv = mDataset()
# cc = DataLoader(csv, batch_size=64)
# for a,b in cc:
