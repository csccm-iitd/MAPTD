#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:32:31 2026

@author: user
"""

import numpy as np
from numpy import random as rdm


def numpy_topk(arr, k, largest=True):
    """
    Finds the k largest/smallest elements and their indices in a NumPy array.
    """
    if largest:
        # Get the indices of the top k elements (unsorted)
        idx_unsorted = np.argpartition(arr, -k)[-k:]
        # Sort the indices by their values to get the final sorted top k
        idx_sorted = idx_unsorted[np.argsort(arr[idx_unsorted])][::-1]
    else:
        # Get the indices of the bottom k elements (unsorted)
        idx_unsorted = np.argpartition(arr, k)[:k]
        # Sort the indices by their values to get the final sorted bottom k
        idx_sorted = idx_unsorted[np.argsort(arr[idx_unsorted])]
    
    # values = arr[idx_sorted]
    return idx_sorted


# Iterate CEM
class CEM:
    def __init__(self, action_dim, bounds, horizon=5, num_samples=32, num_elites=6, iterations=6,
                 temperature=0.5):
        self.action_dim  = action_dim
        self.horizon     = horizon
        self.num_samples = num_samples
        self.prev_mean   = np.zeros((self.horizon, self.action_dim))
        self.step        = 0
        self.iterations  = iterations
        self.num_elites  = num_elites
        self.temperature = temperature
        self.min_std     = 0.05
        self.momentum    = 0.1
        self.lb          = bounds.lb[0]
        self.ub          = bounds.ub[0]
        
    def plan(self, cost_fun, prev_action, eps=1e-9):
        
        prev_action = prev_action.reshape(self.horizon, -1)
    
        # Initialize state and parameters
        mean = np.zeros((self.horizon, self.action_dim))
        std = 2*np.ones((self.horizon, self.action_dim))
        if self.step > 0:
            mean[:-1] = self._prev_mean[1:]
        
        # CEM iterations
        for i in range(self.iterations):
            actions = mean[None,:,:] + std[None,:,:] * rdm.randn(self.num_samples,
                                                                 self.horizon, 
                                                                 self.action_dim)
            actions = np.clip(actions, self.lb, self.ub)
            
            # Compute elite actions
            value = np.zeros(self.num_samples)
            for idx in range(self.num_samples):
                value[idx] = np.nan_to_num(cost_fun(actions[idx, ...]))
            elite_idxs = numpy_topk(value, self.num_elites)
            elite_value, elite_actions = value[elite_idxs], actions[elite_idxs, ...]
            
            # Update parameters
            max_value = elite_value.max() 
            score = np.exp(self.temperature*(elite_value - max_value))
            score /= score.sum()
            _mean = np.sum(score[:,None,None] * elite_actions, axis=0) / (score.sum() + eps)
            _std = np.sqrt(np.sum(score[:,None,None] * (elite_actions - _mean[None,:,:])**2, axis=0) / (score.sum() + eps))
            _std = _std.clip(self.min_std, self.ub**2)
            mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std
    
        # Outputs
        actions = elite_actions[rdm.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a = mean + std * rdm.randn(self.horizon, self.action_dim)
            
        return a.reshape(self.horizon * self.action_dim)
    