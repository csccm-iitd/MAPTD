"""
Training script for agent of the Beam
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import torch
import numpy as np
import scipy as sp
import time
import random
from pathlib import Path
from cfg import parse_cfg
from envs import make_env
from algorithm.maptd import MAPTD
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


def evaluate(env, agent, num_episodes, step, env_step):
    """Evaluate a trained agent."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        
        while not done:
            action = agent.plan(obs, force_t=t, eval_mode=True, step=step, t0=t==0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return np.nanmean(episode_rewards)


# -------------------------------------------------
# Training script
# -------------------------------------------------
assert torch.cuda.is_available()

cfg = parse_cfg(Path().cwd() / __CONFIG__, task_name)
set_seed(cfg.seed)

work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.exp_name / str(cfg.seed)
env, agent, buffer = make_env(cfg), MAPTD(cfg), ReplayBuffer(cfg)

# -------------------------------------------------
# Run training
# -------------------------------------------------
L = logger.Logger(work_dir, cfg)
episode_idx, start_time = 0, time.time()
for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
    print('Step', step)

    # Collect trajectory
    # -------------------------------------------------
    obs = env.reset()
    episode = Episode(cfg, obs)
    force_t = 0     # Time step for forcing input
    while not episode.done:
        force_t += 1
        action = agent.plan(obs, step=step, force_t=force_t, t0=episode.first)
        obs, reward, done, _ = env.step(action.cpu().numpy())
        episode += (obs, action, reward, done)
    assert len(episode) == cfg.episode_length
    buffer += episode
    
    # Update model
    # -------------------------------------------------
    train_metrics = {}
    if step >= cfg.seed_steps:
        num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
        for i in range(num_updates):
            train_metrics.update(agent.update(buffer, step+i))
    
    # Log training episode
    # -------------------------------------------------
    episode_idx += 1
    env_step = int(step*cfg.action_repeat)
    common_metrics = {'episode': episode_idx,
                      'ep': step,
                      'env_step': env_step,
                      'total_time': time.time() - start_time,
                      'episode_reward': episode.cumulative_reward}
    train_metrics.update(common_metrics)
    L.log(train_metrics, category='train')
    
    # Evaluate agent periodically
    # -------------------------------------------------
    if env_step % cfg.eval_freq == 0:
    	common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step)
    	L.log(common_metrics, category='eval')

L.finish(agent)
print('Training completed successfully')

# %%
# -------------------------------------------------
# Test for one sample
# -------------------------------------------------
frames = []
rewards = []
actions = []
obs, done, ep_reward, t = env.reset(), False, 0, 0
while not done:
    print('Case: UNcontrolled --> Time step: {}'.format(t))
    action = agent.plan(obs, force_t=t, eval_mode=True, step=step, t0=t==0)
    obs, reward, done, _ = env.step(action.cpu().numpy())
    frames.append(obs)
    rewards.append(reward)
    actions.append(action.cpu().numpy())
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
rewards = np.stack((rewards))

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
ax[1].legend(ncols=2, loc=4, labelspacing=0.1, columnspacing=0.75, handletextpad=0.2,
             borderaxespad=0.15, borderpad=0.25)
ax[1].margins(0)
plt.show()

fig1.savefig('results/images/Controlled_response_beam.png', format='png', dpi=100, bbox_inches='tight')

# %%
# Save the results,
# -------------------------------------------------
sp.io.savemat('results/beam_control.mat', mdict={'xt_controlled':disp,
                                                      'xt_uncontrolled':true_disp,
                                                      'rewards':rewards,
                                                      'actions':actions})
