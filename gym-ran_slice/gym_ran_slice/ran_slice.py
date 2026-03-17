#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a slice RAN environment for the OpenAI gym environment

Created on December 13, 2021

@author: juanjosealcaraz
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RanSlice(gym.Env):
    ''' 
    class for a ran slicing environment 
    '''
    def __init__(self, node_b = None, penalty = 10, reward_scale = 50.0):
        self.node_b = node_b
        self.penalty = penalty
        self.reward_scale = reward_scale  # NEW: for normalization
        self.n_prbs = node_b.n_prbs
        self.n_slices = node_b.n_slices_l1
        self.n_variables = node_b.get_n_variables()
        self.action_space = spaces.Box(low=0, high = self.n_prbs,
                                        shape=(self.n_slices,), dtype=int)
        self.observation_space = spaces.Box(low=-float('inf'), high=+float('inf'),
                                            shape=(self.n_variables,), dtype=float)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment
        Returns a tuple: (obs, info)
        """
        if seed is not None:
            np.random.seed(seed)

        state = self.node_b.reset()
        info = {}
        return state, info

    def step(self, action):
        """
        :action: [int, int, ...] Action taken by the agent
        :return: (obs, reward, terminated, truncated, info)
        """
        # apply the action
        state, info = self.node_b.step(action)
        total_violations = info['violations'].sum()
        info['total_violations'] = total_violations
        
        if total_violations > 0:
            # if SLA not fulfilled the reward is negative
            reward = -1 * self.penalty * total_violations
        else:
            # if SLA fulfilled the reward is the amount of free resources
            reward = max(0, self.node_b.n_prbs - action.sum())
        
        # Normalize reward to [-1, 1] range approximately
        normalized_reward = reward / self.reward_scale
        
        terminated = False  # Episode doesn't end naturally
        truncated = False   # Episode doesn't get cut off early
        
        return state, float(normalized_reward), terminated, truncated, info

    def render(self):
        pass