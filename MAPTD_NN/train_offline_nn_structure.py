"""
This code trains an offline MLP network for Skyscraper environment.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from timeit import default_timer
from functools import reduce, partial
import operator

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class dnn(nn.Module):
    def __init__(self, in_dim, out_dim, mlp=[512,384,128], act=nn.GELU()):
        super(dnn, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.layers = len(mlp)
        self.mlp = mlp
        
        net = []
        net.append( nn.Linear(in_dim, mlp[0]) )
        net.append( act )
        for i in range( self.layers-1 ):
            net.append( nn.Linear(in_features=mlp[i], out_features=mlp[i+1]) )
            net.append( act )
        net.append( nn.Linear(in_features=mlp[-1], out_features=out_dim) )
        self.net = nn.Sequential(*net)
    
    def forward(self, x, u):
        batch_size, dof = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, -1)
        z = torch.cat((x,u), dim=-1)
        return self.net(z).reshape(batch_size, dof, -1)
    
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
    
# %%
""" Model configurations """
PATH = 'data/structure_mpc_data.mat'
ntrain = 1900
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 1000
step_size = 100      # weight-decay step size
gamma = 0.5         # weight-decay rate

dof = 77
in_dim = 2*dof + 12   # states + action
out_dim = 2*dof       # states
mlp = [512,384,512,512,384,512]

# %%
""" Read data """
data = sio.loadmat(PATH)
xt = torch.tensor(data['xHc'], dtype=torch.float32)
xt = xt[:2, ...]
ut = torch.tensor(data['uH'], dtype=torch.float32)

xdata = xt[..., :-1].permute(2,1,0)
ydata = xt[..., 1:].permute(2,1,0)
udata = ut[:, :].permute(1,0)

# %%
""" Split the data into train and test """
x_train = xdata[:ntrain, ...]
y_train = ydata[:ntrain, ...]
x_test = xdata[-ntest:, ...]
y_test = ydata[-ntest:, ...]

u_train = udata[:ntrain, ...]
u_test = udata[-ntest:, ...]

train_loader = DataLoader(TensorDataset(x_train, u_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, u_test, y_test), batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = dnn(in_dim=in_dim, out_dim=out_dim, mlp=mlp, act=nn.ReLU()).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

ep_train_loss = torch.zeros(epochs)
ep_test_loss = torch.zeros(epochs)
myloss = lambda true, pred: torch.norm(true - pred)/ torch.norm(true)

for ep in range(epochs):
    t1 = default_timer()
    model.train()
    train_loss = 0
    for xx, uin, yy in train_loader:
        xx, uin, yy = xx.to(device), uin.to(device), yy.to(device) 
        
        loss = 0
        optimizer.zero_grad()
        
        out = model(xx, uin)
        loss = myloss(out, yy) 
        
        loss.backward()
        optimizer.step()
        train_loss += loss

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xx, uin, yy in test_loader:
            xx, uin, yy = xx.to(device), uin.to(device), yy.to(device) 
            
            loss = 0            
            out = model(xx, uin)
            loss = myloss(out, yy).item()
            test_loss += loss

    ep_train_loss[ep] = train_loss/ntrain 
    ep_test_loss[ep] = test_loss/ntest 
    
    t2 = default_timer()
    scheduler.step()
    print('Epoch-{}, Time-{:0.4f}, Train loss-{:0.4f}, Test loss-{:0.4f}'.format(
            ep, t2-t1, ep_train_loss[ep], ep_test_loss[ep]))

# %%
""" Prediction """
prediction = []
test_e = []
with torch.no_grad():
    index = 0
    for xx, uin, yy in test_loader:
        xx, uin, yy = xx.to(device), uin.to(device), yy.to(device) 
        
        loss = 0            
        out = model(xx, uin)
        loss = myloss(out, yy).item()
        test_loss = loss
            
        prediction.append( out.cpu() )
        
        test_e.append( test_loss/batch_size )
        index += 1
        
        print("Batch-{}, Test loss-{:0.6f}".format(index, test_loss/batch_size) )
        
prediction = torch.cat((prediction)).reshape(ntest, dof, -1)
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16

floor = [10,30,60,74]

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(y_test[:, floor[i], 0], label='True')
    ax[i,0].plot(prediction[:, floor[i], 0], label='NN')
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('Disp: {}'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    ax[i,0].legend()
    
    ax[i,1].plot(y_test[:, floor[i], 1], label='True')
    ax[i,1].plot(prediction[:, floor[i], 1], label='NN')
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('Vel: {}'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    ax[i,1].legend()

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

# %%
torch.save(model.state_dict(), 'results/offline_model/Offline_NN_76dof')

