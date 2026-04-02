"""
Testing script for TDMPC agent of the Beam
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
__CONFIG__, __LOGS__, task_name = 'cfgs', 'logs', 'beam'


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
# Training script
# -------------------------------------------------
assert torch.cuda.is_available()

cfg = parse_cfg(Path().cwd() / __CONFIG__, task_name)
set_seed(cfg.seed)

work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)

# Load the pretrained model
# -------------------------------------------------
agent.load('logs/beam-EulerBeam/state/default/1/models/model.pt')

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
    print('Case: UNcontrolled --> Time step: {}'.format(t))
    action, action_std, time_ = agent.plan(obs, eval_mode=True, step=0, t0=t==0)
    obs, reward, done, _ = env.step(action.cpu().numpy())
    frames.append(obs)
    rewards.append(reward)
    actions.append(action.cpu().numpy())
    actions_std.append(action_std.cpu().numpy())
    time.append(time_)
    t += 1

disp = np.stack(( frames ))[:, 0:200:2]

frames_true = []
obs, done, ep_reward, t = env.reset(), False, 0, 0
while not done:
    print('Case: UNcontrolled --> Time step: {}'.format(t))
    obs, reward, done, _ = env.step(np.zeros(action.cpu().numpy().shape))
    frames_true.append(obs)
    t += 1

true_disp = np.stack(( frames_true ))[:, 0:200:2]
actions = np.stack((actions))
actions_std = np.stack((actions_std))
rewards = np.stack((rewards))
time = np.stack((time))

# %%
# Plotting
# -------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,6), dpi=100,
                        gridspec_kw={'height_ratios': [1,0.5]})
plt.subplots_adjust(hspace=0.5)

ax[0].set_title('Profile of Displacement', fontweight='bold')
im = ax[0].imshow(disp, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(im, ax=ax[0], pad=0.02)
ax[0].set_xlabel('Space (x)')
ax[0].set_ylabel('Time (sec)')

ax[1].set_title('Displacement evolution at the Free end')
ax[1].plot(disp[:,-1], color='b', label='Controlled')
ax[1].plot(true_disp[:,-1], color='r', label='Uncontrolled')
ax[1].set_ylabel('$u(x=1,t)$')
ax[1].set_xlabel('Time (sec)')
ax[1].axhline(y=0.1, color='grey', linestyle='--')
ax[1].axhline(y=-0.1, color='grey', linestyle='--')
ax[1].grid(True, alpha=0.25)
ax[1].legend(ncol=2, loc=4, labelspacing=0.1, columnspacing=0.75, handletextpad=0.2,
             borderaxespad=0.15, borderpad=0.25)
ax[1].margins(0)
plt.show()

# fig1.savefig('results/images/Controlled_response_beam.png', format='png', dpi=100, bbox_inches='tight')

# %%
# Save the results,
# -------------------------------------------------
# sp.io.savemat('results/tdmpc_beam.mat', mdict={'xt_controlled':disp,
#                                                'xt_uncontrolled':true_disp,
#                                                'rewards':rewards,
#                                                'actions':actions})

# sp.io.savemat('results/beam_tdmpc_time.mat', mdict={'time':time}) 

