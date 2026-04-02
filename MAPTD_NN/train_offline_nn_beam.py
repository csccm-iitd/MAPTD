"""
This code trains an offline MLP network for Beam environment.
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
        batch_size, points = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, -1)
        u = u.reshape(batch_size, -1)
        z = torch.cat((x,u), dim=-1)
        return self.net(z).reshape(batch_size, points, -1)
    
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
    
# %%
""" Model configurations """
PATH = 'data/beam_mpc2.mat'
ntrain = 500
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 1000
step_size = 100      # weight-decay step size
gamma = 0.5         # weight-decay rate

Ne = 200
in_dim = 3*Ne + 64   # states + action
out_dim = 3*Ne       # states
mlp = [512,384,512,512,384,512]

# %%
""" Read data """
data = sio.loadmat(PATH)
xt = torch.tensor(data['xtHistory'], dtype=torch.float32)
ut = torch.tensor(data['uHistory'], dtype=torch.float32)

xdata = xt[..., :-1].permute(2,1,0)
ydata = xt[..., 1:].permute(2,1,0)
udata = ut[32:96, 1:].permute(1,0)[:, :, None]

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
        loss = myloss(out[...,0], yy[...,0]) + myloss(out[...,1], yy[...,1]) + myloss(out[...,2], yy[...,2]) 
        
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
        
prediction = torch.cat((prediction)).reshape(ntest, Ne, -1)
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

colormap = plt.cm.jet  
colors = [colormap(i) for i in np.linspace(0, 1, 3)]
samples = np.linspace(10,ntest-10,3, dtype=int)

""" Plotting """ 
figure7, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,7), dpi=100)
plt.subplots_adjust(hspace=0.7)

for i in range(len(samples)):
    ax[0,i].set_title('Sample: {}'.format(samples[i]))
    ax[0,i].plot(y_test[samples[i], ::2, 0].cpu().numpy(), color=colors[i], label='Actual')
    ax[0,i].plot(prediction[samples[i], ::2, 0].cpu().numpy(), '--', color=colors[i], label='Pred')
    ax[0,i].legend(ncol=2, loc=1, borderaxespad=0.1, columnspacing=0.75, handletextpad=0.25)
    ax[0,i].set_xlabel('Space ($x$)')
    ax[0,i].set_ylabel('$u$($x$)')

    ax[1,i].set_title('Sample: {}'.format(samples[i]))
    ax[1,i].plot(y_test[samples[i], ::2, 1].cpu().numpy(), color=colors[i], label='Actual')
    ax[1,i].plot(prediction[samples[i], ::2, 1].cpu().numpy(), '--', color=colors[i], label='Pred')
    ax[1,i].legend(ncol=2, loc=1, borderaxespad=0.1, columnspacing=0.75, handletextpad=0.25)
    ax[1,i].set_xlabel('Space ($x$)')
    ax[1,i].set_ylabel('$\dot{u}$($x$)')
    
    ax[2,i].set_title('Sample: {}'.format(samples[i]))
    ax[2,i].plot(y_test[samples[i], ::2, 2].cpu().numpy(), color=colors[i], label='Actual')
    ax[2,i].plot(prediction[samples[i], ::2, 2].cpu().numpy(), '--', color=colors[i], label='Pred')
    ax[2,i].legend(ncol=2, loc=1, borderaxespad=0.1, columnspacing=0.75, handletextpad=0.25)
    ax[2,i].set_xlabel('Space ($x$)')
    ax[2,i].set_ylabel('$\ddot{u}$($x$)')
plt.margins(0)
plt.show()

# %%
torch.save(model.state_dict(), 'results/offline_model/Offline_NN_beam')

