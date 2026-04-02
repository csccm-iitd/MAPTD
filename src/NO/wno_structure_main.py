"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burger's equation (time-independent problem).
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import os
directory = os.path.abspath(os.path.join(os.path.dirname('NO'), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class dnn(nn.Module):
    def __init__(self, x_dim, f_dim, u_dim, emd_dim, out_dim, mlps_net=[512,384,256,128]):
        super(dnn, self).__init__()

        """
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        in_channel: scalar, channels in input including grid
        """
        
        self.states = x_dim[0]
        self.x_dim = x_dim[1]
        self.in_dim = np.prod(x_dim)
        self.f_dim = f_dim
        self.u_dim = u_dim
        self.emd_dim = emd_dim
        self.out_dim = np.prod(out_dim)
        
        self.net00 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
        self.net01 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
        
        self.net10 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
        self.net11 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
        
        self.enc_x0 = self.encoder(self.x_dim, self.emd_dim)
        self.enc_x1 = self.encoder(self.x_dim, self.emd_dim)
        
        self.enc_f = self.encoder(self.f_dim, self.emd_dim)
        self.enc_u = self.encoder(self.u_dim, self.emd_dim)
        
        self.dec0 = self.decoder(self.emd_dim, self.x_dim)
        self.dec1 = self.decoder(self.emd_dim, self.x_dim)
        self.dec2 = self.decoder(self.emd_dim, self.x_dim)
        
    def net(self, in_dim, out_dim, mlp, act=nn.ReLU()):
        layers = len(mlp)
        net = []
        net.append( nn.Linear(in_dim, mlp[0]) )
        net.append( act )
        for i in range( layers-1 ):
            net.append( nn.Linear(in_features=mlp[i], out_features=mlp[i+1]) )
            net.append( act )
        net.append( nn.Linear(in_features=mlp[-1], out_features=out_dim) )
        return nn.Sequential(*net)
    
    def encoder(self, in_dim, out_dim):
        return nn.Linear(in_features=in_dim, out_features=out_dim)
    
    def decoder(self, in_dim, out_dim):
        # fc = nn.Sequential(nn.Linear(in_features=in_dim, out_features=in_dim),
        #                    nn.ReLU(),
        #                    nn.Linear(in_features=in_dim, out_features=out_dim))
        fc = nn.Linear(in_features=in_dim, out_features=out_dim)
        return fc
    
    def forward(self, x, f, u):
        # pre-processing
        x0, x1 = x[:,0], x[:,1]
        
        # get the latent space
        x0, x1 = self.enc_x0(x0), self.enc_x1(x1)
        
        f = self.enc_f(f)
        u = self.enc_u(u)
        
        z = torch.cat((x0,x1,f,u), dim=-1)
        x0, x1 = self.net00(z), self.net01(z)
        z = torch.cat((x0,x1,f,u), dim=-1)
        x0, x1 = self.net10(z), self.net11(z)
        
        # decode the latent space
        x0 = self.dec0(x0)
        x1 = self.dec1(x1)
        return torch.stack((x0,x1)).permute(1,0,2)

    
# %%
""" Model configurations """
PATH = directory + '/MPC/results/structure_mpc_data.mat'
ntrain = 1900
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 1000
step_size = 100   # weight-decay step size
gamma = 0.5      # weight-decay rate

# %%
""" Read data """
dataloader = MatReader(PATH)
xt = dataloader.read_field('xHc')[:2]     # iscontrol x isforced x state x grid x time
ut = dataloader.read_field('uH')      # iscontrol x isforced x state x grid x time
ft = dataloader.read_field('fH')

xdata = xt[..., :-1].permute(2,0,1)
ydata = xt[..., 1:].permute(2,0,1)
udata = ut[:, :].permute(1,0) 
fdata = ft[..., 1:].permute(1,0) 

# %%
""" Split the data into train and test """
x_train = xdata[:ntrain, ...]
y_train = ydata[:ntrain, ...]
x_test = xdata[-ntest:, ...]
y_test = ydata[-ntest:, ...]

f_train = fdata[:ntrain, ...]
f_test = fdata[-ntest:, ...]

u_train = udata[:ntrain, ...]
u_test = udata[-ntest:, ...]

train_loader = DataLoader(TensorDataset(x_train, f_train, u_train, y_train),
                          batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, f_test, u_test, y_test),
                         batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = dnn(x_dim=(2,77), f_dim=76, u_dim=12, emd_dim=128, out_dim=(2,77)).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

ep_train_loss = torch.zeros(epochs)
ep_test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    t1 = default_timer()
    model.train()
    
    train_loss = 0
    index = 0
    for xx, fin, uin, yy in train_loader:
        xx, fin, uin, yy = xx.to(device), fin.to(device), uin.to(device), yy.to(device) 
        
        optimizer.zero_grad()
        
        if index == 0:
            input_ = xx
        else: input_ = next_
        
        out = model(input_, fin, uin)
        next_ = out.detach().clone()
        
        loss = myloss(out.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1))
                
        loss.backward()
        optimizer.step()
        train_loss += loss

    model.eval()
    test_loss = 0
    index = 0
    with torch.no_grad():
        for xx, fin, uin, yy in test_loader:
            xx, fin, uin, yy = xx.to(device), fin.to(device), uin.to(device), yy.to(device) 
            
            if index == 0:
                input_ = xx
            else: input_ = out
            
            out = model(input_, fin, uin)
            loss = myloss(out.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1)).item()
            test_loss += loss

    ep_train_loss[ep] = train_loss/len(train_loader) 
    ep_test_loss[ep] = test_loss/len(test_loader)  
    
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
    for xx, fin, uin, yy in test_loader:
        xx, fin, uin, yy = xx.to(device), fin.to(device), uin.to(device), yy.to(device) 
        
        if index == 0:
            input_ = xx
        else: input_ = out
        
        out = model(input_, fin, uin)
        loss = myloss(out.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1)).item()
        test_loss = loss
        
        prediction.append( out.cpu() )
        
        test_e.append( test_loss/batch_size )
        index += 1
        
        print("Batch-{}, Test loss-{:0.6f}".format(index, test_loss/batch_size) )
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16

floor = [10,30,60,74]

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(y_test[:, 0, floor[i]], label='True')
    ax[i,0].plot(prediction[:, 0, floor[i]], label='NN')
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('Disp: {}'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    ax[i,0].legend()
    
    ax[i,1].plot(y_test[:, 1, floor[i]], label='True')
    ax[i,1].plot(prediction[:, 1, floor[i]], label='NN')
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('Vel: {}'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    ax[i,1].legend()

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/WNO_structure_main')
scipy.io.savemat('results/wno_results_structure_main.mat', mdict={'x_test':x_test.cpu().numpy(),
                                                                       'y_test':y_test.cpu().numpy(),
                                                                       'pred':prediction.cpu().numpy(),  
                                                                       'test_e':test_e.cpu().numpy()})

