import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
##  ı am not going to apply cuda beacuse my compueter does not have a cuda supported gpu
df=pd.read_csv("Heterogeneous_accelerometer_HAR.csv")
data=df[["x", "y", "z","gt"]]
sit=data[data["gt"]=="sit"].iloc[0:5500,:]
walk=data[data["gt"]=="walk"].iloc[0:5500,:]
stairup=data[data["gt"]=="stairup"].iloc[0:5500,:]
stairsdown=data[data["gt"]=="stairdown"].iloc[0:5500,:]
bike=data[data["gt"]=="bike"].iloc[0:5500,:]
new_df=pd.concat([sit,walk,stairup,stairsdown,bike],ignore_index=True)
X=new_df.iloc[:,:-1]
y=new_df.iloc[:,-1]
y.replace(to_replace = 'stand', value = 0, inplace = True)
y.replace(to_replace = 'sit', value = 1, inplace = True)
y.replace(to_replace = 'walk', value = 2, inplace = True)
y.replace(to_replace = 'stairsup', value = 3, inplace = True)
y.replace(to_replace = 'stairsdown', value = 4, inplace = True)
y.replace(to_replace = 'bike', value = 5, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
train_data = torch.tensor(X_train.values, dtype=torch.float32)
train_labels = torch.tensor(y_train.values, dtype=torch.long)  

test_data = torch.tensor(X_test.values, dtype=torch.float32)
test_labels = torch.tensor(y_test.values, dtype=torch.long)  

train_data=TensorDataset(train_data,train_labels)
test_data=TensorDataset(test_data,test_labels)
batch_size=64
train_loader=DataLoader(train_data,batch_size=64,drop_last=True)
test_loader=DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

# creating the class
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
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),momentum=0.9,lr=0.001)
epochs=100
losses=torch.zeros(epochs)
train_acc=[]
test_acc=[]
for epoch in range(epochs):
    model.train()
    batch_acc=[]
    batch_loss=[]
    for X,y in train_loader:
        y_pred=model(X)
        loss=loss_func(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        acc=100*torch.mean((torch.argmax(y_pred, dim=1) == y).float())
        batch_acc.append(acc)
    train_acc.append(np.mean(batch_acc))
    losses[epoch]=np.mean(batch_loss)

    #now test data
    model.eval()
    with torch.no_grad():
        for X,y in test_loader:
            y_pred=model(X)
            acc=100*torch.mean((torch.argmax(y_pred, dim=1) == y).float())
            test_acc.append(acc)
    if epoch%10==0:
        print(f'Epoch:{epoch}, Loss: {losses[epoch]:.4f}')
fig,ax = plt.subplots(1,2,figsize=(18,6))

ax[0].plot(losses,'g', lw = 3)
ax[0].set_xlabel('Epochs', fontsize = 15)
ax[0].set_ylabel('Loss', fontsize = 15)
ax[0].legend(['Train','Test'], fontsize = 15)
ax[0].set_title('Training loss', fontsize = 25)

ax[1].plot(train_acc,label='Training Acc', lw =3)
ax[1].plot(test_acc,label='Testing Acc', lw = 3)
ax[1].set_xlabel('Epochs', fontsize = 15)
ax[1].set_ylabel('Accuracy (%)', fontsize = 15)
ax[1].set_ylim([10,100])
ax[1].set_title(f'Train Accuracy: {train_acc[-1]:.2f}% \n Test Accuracy: {test_acc[-1]:.2f}%', fontsize = 15)
ax[1].legend(fontsize = 15)
plt.show()
torch.save(model.state_dict(),"saved_model.pt")