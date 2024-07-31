import torch
from torch import nn
from torch.utils.data import DataLoader
from models import Model
from dataset import customDataset
import numpy as np
import types




class Pipeline():
    def __init__(self, funcApprox, limits, input_size, lr, device):
        assert isinstance(funcApprox, types.FunctionType), "The variable is not a function"

        self.funcApprox = funcApprox
        self.input_size = input_size
        self.lr = lr
        self.limits = limits
        self.device = device
        self.model = Model()
        self.model.to(device=self.device) 
        obj = customDataset(funcApprox, limits, input_size)
        self.data = DataLoader(obj,batch_size=64,shuffle=True)
        

        
    def train(self, epochs):
        self.opt = torch.optim.Adam(params=self.model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()
        for epoch in range(epochs):
            for inputs, target in self.data:
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs).squeeze()
                loss = self.criterion(target,output)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}')
        print(f"Model Trained")

        torch.save(self.model.state_dict(), f'model.pth')
        

    def pred(self):
        model = Model()
        model.to(self.device)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        print("Model loaded.")

        arr = np.random.uniform(-1*self.input_size, self.input_size, self.limits//10)
        x = torch.tensor(arr, dtype = torch.float32).to(self.device)
        y_pred = model(x.view(-1,1)).detach().cpu().numpy()
        np.reshape(y_pred,-1)
        y_actual = [self.funcApprox(val) for val in arr]

        return arr,y_actual,y_pred
    
