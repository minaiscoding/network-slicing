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
    - self.violation_history
    - self.reward_history
    - self.action_history 
    done = True if the number of steps is reached
    """
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False):
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
        self.reset_history()
        self.n_slices = self.env.unwrapped.n_slices
        self.n_prbs = self.env.unwrapped.n_prbs
        self.n_variables = self.env.unwrapped.n_variables

        print('n_prbs = {}'.format(self.env.unwrapped.n_prbs))
        print('n_slices = {}'.format(self.env.unwrapped.n_slices))
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = int)
        self.reward_history = np.zeros((self.steps), dtype = float)
        self.action_history = np.zeros((self.steps), dtype = int)
  
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        print('Resetting environment {}...'.format(self.env_id))
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

        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = action.sum()

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()

        if self.verbose:
            print('Environment {}: {}/{} steps, reward: {}, violations: {}'.format(
                self.env_id, self.step_counter, self.steps, reward, violations))

        return obs, reward, terminated, truncated, {0: 0}

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


# ============================================================================
# OmniSafe Wrapper for Constrained MDPs
# ============================================================================
 

ENV_ID = "RanSlicePPOLag-v0"

_RNG         = None
_SCEN        = 0
_PENALTY     = 100.0
_TOTAL_STEPS = 200 * 1000

@env_register
@env_unregister
class RanSliceEnv(CMDP):

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = False
    need_time_limit_wrapper  = True
    _num_envs                = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        raw_env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=1,
            path='./results/scenario_0/PPOLag/',
            verbose=False,
        )

        self._max_episode_steps = 500
        self._n_prbs            = 200
        self._action_space      = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=float)
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._env.n_variables,), dtype=float
        )

    def reset(self, seed=None, options=None):
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        act = np.abs(act)
        total = act.sum()

        excess = max(0.0, total - 1.0)
        cost = excess * _PENALTY

        if total > 1.0:
            act = act / total

        act = np.array([int(np.floor(a * self._n_prbs)) for a in act], dtype=int)

        result = self._env.step(act)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost = float(cost)
        if np.isnan(cost):
            cost = 0.0

        if isinstance(info, dict):
            info = {str(k): v for k, v in info.items()}
        else:
            info = {}
        if terminated or truncated:
    # Save final obs before reset
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
    
    # Reset and get new obs
            new_obs, _ = self._env.reset()
            obs = torch.as_tensor(new_obs, dtype=torch.float32)
    
    # OmniSafe requires this
            info["final_observation"] = final_obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            info["final_observation"] = obs

        return (
            obs,
            torch.as_tensor(reward,     dtype=torch.float32),
            torch.as_tensor(cost,       dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated,  dtype=torch.bool),
            info,
        )

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def set_seed(self, seed: int) -> None:
        pass

    def spec_log(self, logger) -> None:
        pass

    def close(self) -> None:
        self._env.close()

    @property
    def max_episode_steps(self) -> None:
        return 1