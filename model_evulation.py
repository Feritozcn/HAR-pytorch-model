import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 6)
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        X = self.fc5(X)
        return (X)
model=Net()
model.load_state_dict(torch.load(r"saved_model.pt"))
df=pd.read_csv(r"Heterogeneous_accelerometer_HAR.csv")
data=df[["x", "y", "z","gt"]]
sit=data[data["gt"]=="sit"].iloc[0:100,:]
walk=data[data["gt"]=="walk"].iloc[0:100,:]
stairup=data[data["gt"]=="stairup"].iloc[0:100,:]
stairsdown=data[data["gt"]=="stairdown"].iloc[0:100,:]
bike=data[data["gt"]=="bike"].iloc[0:100,:]
new_df=pd.concat([sit,walk,stairup,stairsdown,bike],ignore_index=True)
X=new_df.iloc[:,:-1]
y=new_df.iloc[:,-1]
y.replace(to_replace = 'stand', value = 0, inplace = True)
y.replace(to_replace = 'sit', value = 1, inplace = True)
y.replace(to_replace = 'walk', value = 2, inplace = True)
y.replace(to_replace = 'stairsup', value = 3, inplace = True)
y.replace(to_replace = 'stairsdown', value = 4, inplace = True)
y.replace(to_replace = 'bike', value = 5, inplace = True)

X=torch.tensor(X.values).float()
y=torch.tensor(y.values).long()

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=50)
model.eval()
corrects=0
total=0
with torch.no_grad():
    for X,y in dataloader:
        output = model(X)
        _, predicted = torch.max(output, 1)

        corrects += (predicted == y).sum().item()
        total += y.size(0)
print((corrects/total)*100)

        