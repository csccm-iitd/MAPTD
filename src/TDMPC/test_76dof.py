"""
Testing script for TDMPC agent of the Skyscraper
-- The code is taken from TDMPC repository
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import sys
import torch
import numpy as np
import scipy as sp
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc_test import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, task_name = 'cfgs', 'logs', 'skyscraper_rom'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
        
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


# -------------------------------------------------
# Script
# -------------------------------------------------
assert torch.cuda.is_available()

cfg = parse_cfg(Path().cwd() / __CONFIG__, task_name)
set_seed(cfg.seed)

work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)

# Load the pretrained model
# -------------------------------------------------
agent.load('logs/structure-Skyscraper_rom/state/default/1/models/model.pt')

# %%
# -------------------------------------------------
# Test for one sample
# -------------------------------------------------
frames = []
rewards = []
actions = []
actions_std = []
obs, done, ep_reward, t = env.reset(), False, 0, 0
time = []
while not done:
    print('Case: Controlled --> Time step: {}'.format(t))
    action, action_std, time_ = agent.plan(obs, eval_mode=True, step=0, t0=t==0)
    obs, reward, done, _ = env.step(action.cpu().numpy())
    frames.append(obs)
    rewards.append(reward)
    actions.append(action.cpu().numpy())
    actions_std.append(action_std.cpu().numpy())
    time.append(time_)
    t += 1

Ne = 77
xt_controlled = np.stack(( frames ))
xt_cd = xt_controlled[:, :Ne]
xt_cv = xt_controlled[:, Ne:]

rewards = np.stack((rewards))
actions = np.stack((actions))
actions_std = np.stack((actions_std))
rewards = np.stack((rewards))
time = np.stack((time))

frames_true = []
obs, done, ep_reward, t = env.reset(), False, 0, 0
while not done:
    print('Case: UNcontrolled --> Time step: {}'.format(t))
    obs, reward, done, _ = env.step(np.zeros(action.cpu().numpy().shape))
    frames_true.append(obs)
    t += 1

xt_uncontrolled = np.stack(( frames_true ))
xt_ud = xt_uncontrolled[:, :Ne]
xt_uv = xt_uncontrolled[:, Ne:]

# %%
# Plotting
# -------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

floor = [4,13,17,22]
t = np.arange(0,1,0.01)

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(xt_ud[:, floor[i]], color='r', label='True')
    ax[i,0].plot(xt_cd[:, floor[i]], linestyle='--', label='Controlled')
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    ax[i,0].legend()
    
    ax[i,1].plot(xt_uv[:, floor[i]], color='r', label='True')
    ax[i,1].plot(xt_cv[:, floor[i]], linestyle='--', label='Controlled')
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    ax[i,1].legend()

ax[0,0].set_title('Displacement')
ax[0,1].set_title('Velocity')
plt.suptitle('Response of 76 DOF Skyscraper (RL ENV)', y=0.94)
plt.show()

# figure1.savefig('results/images/Controlled_response_76dof.png', format='png', dpi=100, bbox_inches='tight')

# %%
# Save the results,
# -------------------------------------------------
# sp.io.savemat('results/tdmpc_structure.mat', mdict={'xt_controlled':xt_controlled,
#                                                     'xt_uncontrolled':xt_uncontrolled,
#                                                     'rewards':rewards,
#                                                     'actions':actions})

# sp.io.savemat('results/skyscraper_tdmpc_time.mat', mdict={'time':time}) 

