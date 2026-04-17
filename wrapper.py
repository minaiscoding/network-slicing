#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a wrapper for the slice environment with the OpenAI gym environment

@author: juanjosealcaraz

Classes:

ReportWrapper
DQNWrapper
TimerWrapper
OmniSafeWrapper

"""

import numpy as np
import gym
from gym import spaces
from itertools import product
import time




PENALTY = 1000
SLICES = 5

# SLICES = 2 # scenario 3

class ReportWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    this environment holds the history of the env variables
    - self.violation_history (total violations per step)
    - self.reward_history
    - self.action_history (total PRBs per step)
    - self.violation_per_slice_history (violations per slice per step)
    - self.resource_per_slice_history (PRBs allocated per slice per step)
    - self.throughput_per_slice_history (throughput per slice per step)
    - self.queue_per_slice_history (queue size per slice per step)
    done = True if the number of steps is reached
    """
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False, continuous_mode = False):
        # Call the parent constructor, so we can access self.env later
        
        super(ReportWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high = 1,
                                        shape=(self.env.unwrapped.n_slices + 1,), dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.env.unwrapped.n_variables,), dtype=float)
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = verbose
        self.path = path
        self.file_path = '{}history_{}.npz'.format(path, env_id)
        self.extra_samples = extra_samples # for safety
        self.continuous_mode = continuous_mode  # If True, don't reset step_counter on reset()
        self.n_slices = self.env.unwrapped.n_slices
        self.n_prbs = self.env.unwrapped.n_prbs
        self.n_variables = self.env.unwrapped.n_variables
        self.reset_history()

        print('n_prbs = {}'.format(self.env.unwrapped.n_prbs))
        print('n_slices = {}'.format(self.env.unwrapped.n_slices))
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = int)
        self.reward_history = np.zeros((self.steps), dtype = float)
        self.action_history = np.zeros((self.steps), dtype = int)
        # Per-slice metrics
        self.violation_per_slice_history = np.zeros((self.steps, self.n_slices), dtype = int)
        self.resource_per_slice_history = np.zeros((self.steps, self.n_slices), dtype = int)
        self.throughput_per_slice_history = np.zeros((self.steps, self.n_slices), dtype = float)
        self.queue_per_slice_history = np.zeros((self.steps, self.n_slices), dtype = float)
  
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        print('Resetting environment {}...'.format(self.env_id))
        if not self.continuous_mode:
            self.step_counter = 0
        result = self.env.reset(seed=seed, options=options)
        # Handle both old (obs,) and new (obs, info) reset signatures
        if isinstance(result, tuple):
            self.obs, info = result[0], result[1] if len(result) > 1 else {}
        else:
            self.obs, info = result, {}
        if self.verbose:
            print('Environment {} RESET'.format(self.env_id))
        return self.obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, terminated, truncated, info
        """
        if len(action) > self.n_slices:
            action = action[:self.n_slices]
            action = abs(action)
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=int)
            # action = np.array([np.floor(self.n_prbs * action[i]/t_action) + 1 for i in range(self.n_slices)], dtype=np.int)

        result = self.env.step(action)
    
#   Handle both old 4-value and new 5-value step signatures
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated, truncated = done, False

        # Normalize observations
        obs = np.clip(obs, -0.5, 1.5)
        obs = obs - 0.5
        self.obs = obs

        # Collect historical data
        violations = info.get('total_violations', 0)
        per_slice_violations = info.get('violations', np.zeros(self.n_slices, dtype=int))

        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = action.sum()
            # Per-slice metrics
            if hasattr(per_slice_violations, '__len__') and len(per_slice_violations) == self.n_slices:
                self.violation_per_slice_history[self.step_counter] = per_slice_violations
            self.resource_per_slice_history[self.step_counter] = action[:self.n_slices] if len(action) >= self.n_slices else action

            # Per-slice throughput and queue from l1_info
            l1_info = info.get('l1_info', [])
            for si in range(min(self.n_slices, len(l1_info))):
                slice_info = l1_info[si]
                if isinstance(slice_info, dict):
                    for _, ri in slice_info.items():
                        if isinstance(ri, dict):
                            self.throughput_per_slice_history[self.step_counter, si] = (
                                ri.get('cbr_th', 0.0) + ri.get('vbr_th', 0.0)
                            )
                            self.queue_per_slice_history[self.step_counter, si] = (
                                ri.get('cbr_queue', 0.0) + ri.get('vbr_queue', 0.0)
                                + ri.get('delay', 0.0)  # mMTC uses 'delay' instead
                            )

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()

        if self.verbose:
            print('Environment {}: {}/{} steps, reward: {}, violations: {}'.format(
                self.env_id, self.step_counter, self.steps, reward, violations))

        info["cost"] = float(violations)
        return obs, reward, terminated, truncated, info

    def save_results(self):
        np.savez(self.file_path, 
                 violation = self.violation_history, 
                 reward = self.reward_history,
                 resources = self.action_history,
                 violation_per_slice = self.violation_per_slice_history,
                 resource_per_slice = self.resource_per_slice_history,
                 throughput_per_slice = self.throughput_per_slice_history,
                 queue_per_slice = self.queue_per_slice_history,
                 n_slices = self.n_slices)
    
    def set_evaluation(self, eval_steps, new_path = None, change_name = False):
        self.step_counter = self.steps
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, [(0, eval_steps)])
        self.reward_history = np.pad(self.reward_history, [(0, eval_steps)])
        self.action_history = np.pad(self.action_history, [(0, eval_steps)])
        # Pad per-slice arrays
        self.violation_per_slice_history = np.pad(self.violation_per_slice_history, [(0, eval_steps), (0, 0)])
        self.resource_per_slice_history = np.pad(self.resource_per_slice_history, [(0, eval_steps), (0, 0)])
        self.throughput_per_slice_history = np.pad(self.throughput_per_slice_history, [(0, eval_steps), (0, 0)])
        self.queue_per_slice_history = np.pad(self.queue_per_slice_history, [(0, eval_steps), (0, 0)])
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = '{}evaluation_{}.npz'.format(self.path, self.env_id)

class DQNWrapper(ReportWrapper):
    '''
    Variation for DQN
    '''
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env, steps = steps, control_steps = control_steps, env_id = env_id, extra_samples = extra_samples, path = path, verbose = verbose)
        g_eMBB = 2 # ganularity
        max_eMBB = 51 # max prbs for a single slice
        self.actions = []
        a = list(range(0,max_eMBB,g_eMBB))
        for (a1,a2) in product(a,a):
            if a1 + a2 <= self.n_prbs:
                self.actions.append(np.array([a1, a2], dtype = np.int16))
        self.action_space = spaces.Discrete(len(self.actions))
    
    def step(self, action):
        a = self.actions[action]
        return super(DQNWrapper, self).step(a)

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


