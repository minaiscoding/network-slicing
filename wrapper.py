#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a wrapper for the slice environment with the OpenAI gym environment

@author: juanjosealcaraz

Classes:

ReportWrapper
DQNWrapper
TimerWrapper

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
    - self.violation_history
    - self.reward_history
    - self.action_history 
    done = True if the number of steps is reached
    """
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False, continuous_mode = False):
        # Call the parent constructor, so we can access self.env later
        super(ReportWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high = 1,
                                        shape=(self.n_slices + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                            shape=(self.n_variables,), dtype=np.float32)
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = verbose
        self.path = path
        self.file_path = '{}history_{}.npz'.format(path, env_id)
        self.extra_samples = extra_samples # for safety
        self.continuous_mode = continuous_mode
        self.reset_history()

        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = np.int16)
        self.reward_history = np.zeros((self.steps), dtype = np.float32)
        self.action_history = np.zeros((self.steps), dtype = np.int16)
  
    def reset(self):
        """
        Reset the environment (but only when it is created)
        """
        self.step_counter = 0
        result = self.env.reset()
        # Handle both Gym and Gymnasium return formats
        if isinstance(result, tuple):
            self.obs, _ = result
        else:
            self.obs = result
        if self.verbose:
            print('Environment {} RESET'.format(self.env_id))
        return self.obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # this works with actions like [0.5, 0.2, 0.3]
        if len(action) > self.n_slices: # action = [0.5, 0.2, 0.3]
            action = abs(action) # no negative values allowed
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=np.int64)
            # action = np.array([np.floor(self.n_prbs * action[i]/t_action) + 1 for i in range(self.n_slices)], dtype=np.int)

        # Handle both Gym and Gymnasium return formats
        result = self.env.step(action)
        
        if len(result) == 5:
            # Gymnasium format: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # Old Gym format: obs, reward, done, info
            obs, reward, done, info = result
            terminated, truncated = done, False

        self.obs = obs

        if self.step_counter % 500 == 0:
            print(f'[ReportWrapper step {self.step_counter}] obs (pre-omnisafe): min={obs.min():.4f} max={obs.max():.4f} mean={obs.mean():.4f}')
            print(f'  values: {np.array2string(obs.astype(np.float32), precision=4, suppress_small=True)}')

        # collect historical data
        violations = info.get('total_violations', 0)

        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = action.sum()

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()
        
        if self.verbose:
            print('Environment {}: {}/{} steps, reward: {}, violations: {}'.format(self.env_id, self.step_counter, self.steps, reward, violations))

        # Return consistent format for the caller
        if self.continuous_mode:
            # Return 5 values for continuous training (Gymnasium style)
            return obs, reward, terminated, truncated, info
        else:
            # Return 4 values for compatibility
            return obs, reward, done, {0:0}

    def save_results(self):
        np.savez(self.file_path, violation = self.violation_history, 
                                reward = self.reward_history,
                                resources = self.action_history)
    
    def set_evaluation(self, eval_steps, new_path = None, change_name = False):
        self.step_counter = self.steps
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, [(0, eval_steps)])
        self.reward_history = np.pad(self.reward_history, [(0, eval_steps)])
        self.action_history = np.pad(self.action_history, [(0, eval_steps)])
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = '{}evaluation_{}.npz'.format(self.path, self.env_id)

    @property
    def n_variables(self):
        return self.env.n_variables
    
    @property
    def n_slices(self):
        return self.env.n_slices
    
    @property
    def n_prbs(self):
        return self.env.n_prbs


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
        a = list(range(0, max_eMBB, g_eMBB))
        for (a1, a2) in product(a, a):
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
                                        shape=(self.n_slices + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                            shape=(self.n_variables,), dtype=np.float32)
        self.steps = steps
        self.step_counter = 0
        self.simtime = 0
        self.time_samples = np.zeros((self.steps), dtype = np.float32)
        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
  
    def reset(self):
        """
        Reset the environment 
        """
        self.step_counter = 0
        self.simtime = 0
        result = self.env.reset()
        # Handle both Gym and Gymnasium return formats
        if isinstance(result, tuple):
            self.obs, _ = result
        else:
            self.obs = result
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
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=np.int64)
        
        # measure simulation time
        t1 = time.time()
        
        # Handle both Gym and Gymnasium return formats
        result = self.env.step(action)
        
        if len(result) == 5:
            # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # Old Gym format
            obs, reward, done, info = result
        
        self.simtime += time.time() - t1
        
        self.obs = obs

        # increment counter
        self.step_counter += 1

        # return obs, reward, done, info
        return obs, reward, False, {0:0} # for keras rl this avoids problems

    @property
    def n_variables(self):
        return self.env.n_variables
    
    @property
    def n_slices(self):
        return self.env.n_slices
    
    @property
    def n_prbs(self):
        return self.env.n_prbs