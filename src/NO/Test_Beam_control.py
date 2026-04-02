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
from wavelet_convolution import WaveConv1d

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
                
        self.conv0, self.w0 = nn.ModuleList(), nn.ModuleList()
        self.conv1, self.w1 = nn.ModuleList(), nn.ModuleList()
        self.conv2, self.w2 = nn.ModuleList(), nn.ModuleList()
        
        self.fc10 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)
        self.fc11 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)
        self.fc12 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)
        
        self.ft = nn.Linear(self.in_channel[1], self.width[0]) # input channel is 2: (f(x), u(x))

        for i in range( self.layers - 1 ):
            self.conv0.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level, 
                                         self.size, self.wavelet, omega=self.omega) )
            self.w0.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )
            
            self.conv1.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level, 
                                         self.size, self.wavelet, omega=self.omega) )
            self.w1.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )
            
            self.conv2.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level, 
                                         self.size, self.wavelet, omega=self.omega) )
            self.w2.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )
            
        self.fc21 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))
        self.fc22 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))
        self.fc23 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x, f, u):
        grid = self.get_grid(x.shape, x.device)
        
        # append the grid:        
        x0 = torch.cat((x[:,:,0:1], grid), dim=-1) 
        x1 = torch.cat((x[:,:,1:2], grid), dim=-1) 
        x2 = torch.cat((x[:,:,2:3], grid), dim=-1) 
        fu = torch.cat((f,u), dim=-1)

        # get the latent space:
        x0 = self.fc10(x0)              # Shape: Batch * x * Channel
        x1 = self.fc11(x1)              # Shape: Batch * x * Channel
        x2 = self.fc12(x2)              # Shape: Batch * x * Channel
        fu = self.ft(fu)
        
        x0 = x0.permute(0, 2, 1)       # Shape: Batch * Channel * x
        x1 = x1.permute(0, 2, 1)       # Shape: Batch * Channel * x
        x2 = x2.permute(0, 2, 1)       # Shape: Batch * Channel * x
        fu = fu.permute(0, 2, 1)
        
        for index, (cl0, wl0, cl1, wl1, cl2, wl2) in enumerate( zip(self.conv0, self.w0, 
                                                                    self.conv1, self.w1,
                                                                    self.conv2, self.w2) ):
            if index == 0:
                z = torch.cat((x0,x1,x2,fu), dim=1)
                v0 = cl0(z) + wl0(z) 
                v1 = cl1(z) + wl1(z) 
                v2 = cl2(z) + wl2(z) 
            else:
                z = torch.cat((v0,v1,v2,fu), dim=1)                
                v0 = cl0(z) + wl0(z) 
                v1 = cl1(z) + wl1(z) 
                v2 = cl2(z) + wl2(z) 

            if index != self.layers - 1:   # Final layer has no activation    
                v0 = F.mish(v0)            # Shape: Batch * Channel * x  
                v1 = F.mish(v1)            # Shape: Batch * Channel * x  
                v2 = F.mish(v2)            # Shape: Batch * Channel * x  
        
        v0 = v0.permute(0, 2, 1)       # Shape: Batch * x * Channel
        v1 = v1.permute(0, 2, 1)       # Shape: Batch * x * Channel
        v2 = v2.permute(0, 2, 1)       # Shape: Batch * x * Channel        
        
        v0 = self.fc21(v0)    # Shape: Batch * x * Channel
        v1 = self.fc22(v1)    # Shape: Batch * x * Channel
        v2 = self.fc23(v2)    # Shape: Batch * x * Channel
        return torch.cat((v0,v1,v2), dim=-1)

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    
# %%
""" Model configurations """
PATH = directory + '/MPC/results/beam_mpc.mat'
ntrain = 490
ntest = 500

batch_size = 20
learning_rate = 0.001

epochs = 1000
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db2'  # wavelet basis function
level = 1        # lavel of wavelet decomposition, size=3 for [ut, ft, at]

in_channel = [1+1, 1+1]
width = [40, 48, 30, 30]       # uplifting dimension

layers = len(width)       # no of wavelet layers

h = 200
grid_range = 1

# %%
""" Read data """
dataloader = MatReader(PATH)
xt = dataloader.read_field('xtHistory')     # iscontrol x isforced x state x grid x time
ut = dataloader.read_field('uHistory')      # iscontrol x isforced x state x grid x time
ft = dataloader.read_field('fHistory')

xdata = xt[..., :-1].permute(2,1,0)
ydata = xt[..., 1:].permute(2,1,0)
udata = ut[:, 1:].permute(1,0)[:, :, None]
fdata = ft[..., :-1].permute(1,0)[:, :, None]

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
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, f_test, u_test, y_test),
                         batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = torch.load('model/WNO_beam_control2', map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
""" Prediction """
prediction = []
test_e = []
t0 = default_timer()
with torch.no_grad():
    index = 0
    for xx, fin, uin, yy in test_loader:
        xx, fin, uin, yy = xx.to(device), fin.to(device), uin.to(device), yy.to(device) 
        
        out = model(xx, fin, uin)
        loss = myloss(out.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1)).item()
        test_loss = loss
            
        prediction.append( out.cpu() )
        
        test_e.append( test_loss/batch_size )
        index += 1
        
        print("Batch-{}, Test loss-{:0.6f}".format(index, test_loss/batch_size) )
t1 = default_timer()
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
print('Mean time: {}'.format((t1-t0)/len(test_loader)))

# %%
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

colormap = plt.cm.jet  
# colors = [colormap(i) for i in np.linspace(0, 1, 3)]
colors = ['brown', 'tab:blue', 'tab:green', 'tab:red']
clrs = ['brown', 'blue', 'green', 'red']

ylabel = ['$f(x)$', '$u$($x$)\n', '$\dot{u}$($x$)', '$\ddot{u}$($x$)']
ylabel = ['$f(x)$', '$u$($x$) / $\phi$($x$)\n', '$\dot{u}$($x$) / $\dot{\phi}$($x$)',
          '$\ddot{u}$($x$) / $\ddot{\phi}$($x$)']

np.random.seed(15)
# samples = np.sort(np.random.choice(ntest,3))
samples = [5,16,450]

sub = 2
dt = 0.01
xx = np.linspace(0,1,100)

# """ Plotting """ 
# figure7, ax = plt.subplots(nrows=4, ncols=3, figsize=(12,6), dpi=100)
# plt.subplots_adjust(hspace=0.35,wspace=0.25)

# for i in range(len(samples)):
#     ax[0,i].set_title('Time: {}'.format(samples[i]*dt))
#     ax[0,i].plot(xx,f_test[samples[i], :100, 0].cpu().numpy(), color=colors[0], linewidth=1)
#     ax[0,i].set_ylim([0,0.022])
    
#     ax[1,i].plot(xx,y_test[samples[i], ::sub, 0].cpu().numpy(), color='k', label='True', linewidth=3, linestyle='--')
#     ax[1,i].plot(xx,prediction[samples[i], ::sub, 0].cpu().numpy(), color=colors[1], label='Pred')

#     ax[2,i].plot(xx,y_test[samples[i], ::sub, 1].cpu().numpy(), color='k', label='True', linewidth=3, linestyle='--')
#     ax[2,i].plot(xx,y_test[samples[i], ::sub, 1].cpu().numpy(), color=colors[2], label='Pred')
    
#     ax[3,i].plot(xx,y_test[samples[i], ::sub, 2].cpu().numpy(), color='k', label='True', linewidth=3, linestyle='--')
#     ax[3,i].plot(xx,prediction[samples[i], ::sub, 2].cpu().numpy(), color=colors[3], label='Pred')

# for i in range(4):
#     for j in range(3):    
#         ax[i,j].grid('True', alpha=0.25)

# for i in range(4):
#     ax[i,0].set_ylabel("{}".format(ylabel[i]), color=clrs[i])
#     if i > 0: ax[i,0].legend(ncol=1, borderaxespad=0.1, columnspacing=0.75, handletextpad=0.25)

# figure7.supxlabel('Spatial dimension', y=0.03)
# plt.show()

""" Plotting """ 
figure7, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,6), dpi=100)
plt.subplots_adjust(hspace=0.35,wspace=0.25)

for i in range(len(samples)):
    ax[0,i].set_title('Time: {}s'.format(samples[i]*dt))

    ax[0,i].plot(xx,y_test[samples[i], ::sub, 0].cpu().numpy(), color='k', label='True $u$', linewidth=3, linestyle='--')
    ax[0,i].plot(xx,prediction[samples[i], ::sub, 0].cpu().numpy(), color=colors[1], label='Pred $u$')
    ax[0,i].plot(xx,y_test[samples[i], 1::sub, 0].cpu().numpy(), color='k', label='True $\phi$', linewidth=3, linestyle=':')
    ax[0,i].plot(xx,prediction[samples[i], 1::sub, 0].cpu().numpy(), color=colors[1], label='Pred $\phi$', linestyle='-.')

    ax[1,i].plot(xx,y_test[samples[i], ::sub, 1].cpu().numpy(), color='k', label='True $\dot{u}$', linewidth=3, linestyle='--')
    ax[1,i].plot(xx,y_test[samples[i], ::sub, 1].cpu().numpy(), color=colors[2], label='Pred $\dot{u}$')
    ax[1,i].plot(xx,y_test[samples[i], 1::sub, 1].cpu().numpy(), color='k', label='True $\dot{\phi}$', linewidth=3, linestyle=':')
    ax[1,i].plot(xx,y_test[samples[i], 1::sub, 1].cpu().numpy(), color=colors[2], label='Pred $\dot{\phi}$', linestyle='-.')
    
    ax[2,i].plot(xx,y_test[samples[i], ::sub, 2].cpu().numpy(), color='k', label='True $\ddot{u}$', linewidth=3, linestyle='--')
    ax[2,i].plot(xx,prediction[samples[i], ::sub, 2].cpu().numpy(), color=colors[3], label='Pred $\ddot{\phi}$')
    ax[2,i].plot(xx,y_test[samples[i], 1::sub, 2].cpu().numpy(), color='k', label='True $\ddot{u}$', linewidth=3, linestyle=':')
    ax[2,i].plot(xx,prediction[samples[i], 1::sub, 2].cpu().numpy(), color=colors[3], label='Pred $\ddot{\phi}$', linestyle='-.')

for i in range(3):
    for j in range(3):    
        ax[i,j].grid('True', alpha=0.25)

for i in range(3):
    ax[i,0].set_ylabel("{}".format(ylabel[i+1]), color=clrs[i+1])
    ax[i,0].legend(ncol=1, borderaxespad=0.1, columnspacing=0.75, handletextpad=0.25, 
                   labelspacing=0.15, borderpad=0.25, handlelength=2)

figure7.supxlabel('Spatial dimension', y=0.03)
plt.show()

figure7.savefig("images/beam_no.pdf", format='pdf', dpi=300, bbox_inches='tight')

# %%
figure5, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,8), dpi=100)
plt.subplots_adjust(hspace=0.7)

for i in range(3):
    im = ax[i,1].imshow(prediction[:,::2,i], aspect='auto', cmap='jet')
    plt.colorbar(im,ax=ax[i,1])
    im = ax[i,0].imshow(y_test[:,::2,i], aspect='auto', cmap='jet')
    plt.colorbar(im,ax=ax[i,0])
    
ax[0,0].set_title('True')
ax[0,1].set_title('WNO')
plt.show()

