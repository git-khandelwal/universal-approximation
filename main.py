from train import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch


# print(torch.cuda.is_available())
# print(torch.__version__)

torch.backends.cudnn.benchmark = True
limits = 10000 # Number of examples of x
input_size = 5. # Limits for values of x (-input_size, input_size)
lr = 0.01 # Learning Rate
epochs = 100 # Training Epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def funcApprox(x):
    return np.log(x) + x - x**2



ob = Pipeline(funcApprox=funcApprox, limits=limits, input_size=input_size, lr=lr, device=device)
ob.train(epochs=epochs)
arr,y_actual,y_pred = ob.pred()

# Plotting True/Predicted
plt.scatter(arr, y_actual, color="red")
plt.scatter(arr, y_pred, color="blue")
plt.show()


