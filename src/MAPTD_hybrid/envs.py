# -*- coding: utf-8 -*-
"""
This script creates wrapping functions for the DRL algorithm to use
"""

import numpy as np
from typing import Any, NamedTuple
from collections import deque, defaultdict
import gym

import dynamics_gym
import dm_env
from dm_env import StepType, specs
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        try:
            self.Ne = env.dof
        except:
            self.Ne = env.Ne
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self.Ne = env.Ne
        self._num_repeats = num_repeats
   
    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)

            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)
   
    def observation_spec(self):
   		return self._env.observation_spec()
   
    def action_spec(self):
   		return self._env.action_spec()
   
    def reset(self):
   		return self._env.reset()
   
    def __getattr__(self, name):
   		return getattr(self._env, name)
    

class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST
    
    
class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self.Ne = env.Ne

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TimeStepToGymWrapper(object):
    def __init__(self, cfg, env, task, action_repeat):
        obs_shp = []
        for v in env.observation_spec().values():
            try:
                shp = np.prod(v.shape)
            except:
                shp = 1
            obs_shp.append(shp)
        obs_shp = (np.sum(obs_shp, dtype=np.int32),)
        
        act_shp = env.action_spec().shape
        obs_dtype = np.float32
        self.observation_space = gym.spaces.Box(
                        			low=np.full(obs_shp,-np.inf,dtype=obs_dtype),
                                    high=np.full(obs_shp,np.inf,dtype=obs_dtype),
                                    shape=obs_shp,
                                    dtype=obs_dtype,
                                    )
        self.action_space = gym.spaces.Box(
                        			low=np.full(act_shp, env.action_spec().minimum),
                                    high=np.full(act_shp, env.action_spec().maximum),
                                    shape=act_shp,
                                    dtype=env.action_spec().dtype
                                    )
        self.env = env
        self.Ne = env.Ne
        self.task = task
        self.ep_len = cfg.episode_length
        self.t = 0
	
    @property
    def unwrapped(self):
        return self.env

    @property
    def reward_range(self):
        return None

    @property
    def metadata(self):
        return None
	
    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values()])

    def reset(self):
        self.t = 0
        return self._obs_to_array(self.env.reset().observation)
	
    def step(self, action):
        self.t += 1
        time_step = self.env.step(action)
        return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)
    

class DefaultDictWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, defaultdict(float, info)
    
    
def make_env(cfg):
    """
    Make DMControl environment for MAPTD experiments.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    task = cfg.task 
    env = dynamics_gym.env[cfg.task_title]()
    
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, cfg.action_repeat)
    env = action_scale.Wrapper(env, minimum=cfg.lb, maximum=cfg.ub)
    
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(cfg, env, task, cfg.action_repeat)
    env = DefaultDictWrapper(env)

	# Convenience
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env

