"""
Training script for agent of the Skyscraper
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
__CONFIG__, __LOGS__, task_name = 'cfgs', 'logs', 'structure_rom'


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
    action = agent.plan(obs, force_t=t, eval_mode=True, step=step, t0=t==0)
    obs, reward, done, _ = env.step(action.cpu().numpy())
    print('Case: Controlled --> Time step: {}, Action: {}'.format(t,action.cpu().numpy()))
    frames.append(obs)
    rewards.append(reward)
    actions.append(action.cpu().numpy())
    t += 1

xt_controlled = np.stack(( frames ))
xt_cd = xt_controlled[:, :env.Ne]
xt_cv = xt_controlled[:, env.Ne:]
rewards = np.stack(( rewards ))
actions = np.stack(( actions ))

frames_true = []
obs, done, ep_reward, t = env.reset(), False, 0, 0
while not done:
    obs, reward, done, _ = env.step(np.zeros(action.cpu().numpy().shape))
    print('Case: UNcontrolled --> Time step: {}'.format(t))
    frames_true.append(obs)
    t += 1

xt_uncontrolled = np.stack(( frames_true ))
xt_ud = xt_uncontrolled[:, :env.Ne]
xt_uv = xt_uncontrolled[:, env.Ne:]

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

figure1.savefig('results/images/Controlled_response_76dof.png', format='png', dpi=100, bbox_inches='tight')


# %%
# Save the results,
# -------------------------------------------------
sp.io.savemat('results/structure_control.mat', mdict={'xt_controlled':xt_controlled,
                                                      'xt_uncontrolled':xt_uncontrolled,
                                                      'rewards':rewards,
                                                      'actions':actions})
