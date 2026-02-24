#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium-compatible wrappers for network-slicing environment.

@author: juanjosealcaraz
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product
import time

PENALTY = 1000
SLICES = 5

class ReportWrapper(gym.Wrapper):
    def __init__(self, env, steps=2000, control_steps=500, env_id=1,
                 extra_samples=10, path='./logs/', verbose=False, 
                 n_slices=SLICES, n_prbs=200, n_variables=None):
        super().__init__(env)
        self.n_slices = n_slices
        self.n_prbs = n_prbs
        # detect n_variables from env if not provided
        if n_variables is None:
            try:
                obs, _ = self.env.reset()
                self.n_variables = len(obs)
            except:
                self.n_variables = 10
        else:
            self.n_variables = n_variables

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_slices + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_variables,), dtype=np.float32)
        
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = verbose
        self.path = path
        self.file_path = f'{path}history_{env_id}.npz'
        self.extra_samples = extra_samples
        
        self.reset_history()
        print(f'n_prbs = {self.n_prbs}, n_slices = {self.n_slices}, n_variables = {self.n_variables}')
    
    def reset_history(self):
        self.violation_history = np.zeros(self.steps, dtype=int)
        self.reward_history = np.zeros(self.steps, dtype=float)
        self.action_history = np.zeros(self.steps, dtype=int)
  
    def reset(self, *, seed=None, options=None):
        self.step_counter = 0
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        return obs, info

    def step(self, action):
        # normalize actions
        if len(action) > self.n_slices:
            action = np.abs(action)
            t_action = action.sum() or 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=int)

        obs, reward, done, info = self.env.step(action)
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs

        violations = info.get('total_violations', 0)
        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = int(np.sum(action))

        self.step_counter += 1
        if self.step_counter % self.control_steps == 0:
            self.save_results()
        
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def save_results(self):
        np.savez(self.file_path, violation=self.violation_history, 
                 reward=self.reward_history,
                 resources=self.action_history)
    
    def set_evaluation(self, eval_steps, new_path=None, change_name=False):
        self.step_counter = self.steps
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, (0, eval_steps))
        self.reward_history = np.pad(self.reward_history, (0, eval_steps))
        self.action_history = np.pad(self.action_history, (0, eval_steps))
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = f'{self.path}evaluation_{self.env_id}.npz'

class DQNWrapper(ReportWrapper):
    def __init__(self, env, steps=2000, control_steps=500, env_id=1, 
                 extra_samples=10, path='./logs/', verbose=False, n_variables=None):
        super().__init__(env, steps=steps, control_steps=control_steps, 
                         env_id=env_id, extra_samples=extra_samples, 
                         path=path, verbose=verbose, n_variables=n_variables)
        g_eMBB = 2
        max_eMBB = 51
        self.actions = []
        a = list(range(0, max_eMBB, g_eMBB))
        for a1, a2 in product(a, a):
            if a1 + a2 <= self.n_prbs:
                self.actions.append(np.array([a1, a2], dtype=int))
        self.action_space = spaces.Discrete(len(self.actions))
    
    def step(self, action):
        a = self.actions[action]
        return super().step(a)

class TimerWrapper(ReportWrapper):
    def __init__(self, env, steps=2000, n_variables=None):
        super().__init__(env, steps=steps, n_variables=n_variables)
        self.simtime = 0
        self.time_samples = np.zeros(self.steps, dtype=float)
  
    def reset(self, *, seed=None, options=None):
        self.step_counter = 0
        self.simtime = 0
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        return obs, info
    
    def step(self, action):
        t1 = time.time()
        obs, reward, done, info = self.env.step(action)
        self.simtime += time.time() - t1
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        self.step_counter += 1
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

class TimerWrapper(gym.Wrapper):
    '''
    Auxiliary wrapper for time measurement
    '''
    def __init__(self, env, steps = 2000):
        # Call the parent constructor, so we can access self.env later
        super(TimerWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high = 1,
                                        shape=(self.n_slices + 1,), dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.n_variables,), dtype=float)
        self.steps = steps
        self.step_counter = 0
        self.simtime = 0
        self.time_samples = np.zeros((self.steps), dtype = float)
        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
  
    def reset(self):
        """
        Reset the environment 
        """
        self.step_counter = 0
        self.simtime = 0
        self.obs = self.env.reset()
        return self.obs
    
    def get_simtime(self):
        return self.simtime

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        # this should operate well with actions like [0.5, 0.2, 0.3]
        if len(action) > self.n_slices: # action = [0.5, 0.2, 0.3]
            action = abs(action) # no negative values allowed
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=int)
        
        # measure simulation time
        t1 = time.time()
        obs, reward, _, _ = self.env.step(action)
        self.simtime += t1 - time.time()
        
        # RL algorithms work better with normalized observations between -0.5 and 0.5
        obs = np.clip(obs,-0.5,1.5) 
        obs = obs - 0.5
        self.obs = obs

        # increment counter
        self.step_counter += 1

        # return obs, reward, done, info
        return obs, reward, False, {0:0} # for keras rl this avoids problems