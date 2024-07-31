import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True  # for deterministic behavior
    torch.backends.cudnn.benchmark = True

set_seed(42)  # Set the seed to any number you like


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1,16,device=device)
        # self.fc2 = nn.Linear(16,64,device=device)
        # self.fc3 = nn.Linear(64,8,device=device)
        self.fc4 = nn.Linear(16,1,device=device)
        # self.dropout1 = nn.Dropout(p=0.2)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.dropout3 = nn.Dropout(p=0.2)
        # self.bn1 = nn.BatchNorm1d(16)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(8)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        # # x = self.dropout1(x)
        # # x = self.bn1(x)
        # x = self.fc2(x)
        # x = torch.relu(x)
        # # x = self.dropout2(x)
        # # x = self.bn2(x)
        # x = self.fc3(x)
        # x = torch.relu(x)
        # # x = self.dropout3(x)
        # # x = self.bn3(x)
        x = self.fc4(x)
        return x
    

class Model_Quad_2(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1,16,device=device)
        #
        self.dropout1 = nn.Dropout(p=0.2) 
        self.fc2 = nn.Linear(16,64,device=device)
        #
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64,8,device=device)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(8,1,device=device)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
  


class Model_Quad_1(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1,16,device=device)

        # self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(16,64,device=device)
        # self.dropout = nn.Dropout(p=0.3)
        # self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(64,8,device=device)
        # self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8,1,device=device)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.bn3(x)
        x = self.fc4(x)
        return x
    

