#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: juanjosealcaraz

Train and evaluate PPO (OmniSafe) in network-slicing scenarios.
Each epoch resets the environment with a proper RNG.
'''

import os
import argparse
import numpy as np
from numpy.random import default_rng
from gymnasium import spaces
from wrapper import ReportWrapper
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from scenario_creator import create_env

SCENARIO = 4
RUNS = 2
PROCESSES = 4
EPOCHS = 10
STEPS_PER_EPOCH = 1000
SLOTS_PER_STEP = 500
PENALTY = 1000
EVALUATION_STEPS = 5000

ENV_ID = "RanSlicePPO-v0"

_RNG         = np.random.default_rng(3)
_SCEN        = 4
_PENALTY     = 100.0
_TOTAL_STEPS = 20000

@env_register
@env_unregister
class RanSliceEnv(CMDP):

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = False
    need_time_limit_wrapper  = False
    _num_envs                = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        raw_env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=1,
            path='./results/scenario_4/PPO/',
            verbose=False,

        )

        self._max_episode_steps = 500
        self._n_prbs            = 100
        self._step_count        = 0  # track steps for forced truncation
        self._action_space      = spaces.Box(low=0.0, high=1.0, shape=(self._env.n_slices + 1,), dtype=float)
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._env.n_variables,), dtype=float
        )

    def reset(self, seed=None, options=None):
        self._step_count = 0
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        act = np.abs(act)
    
        # Normalize to sum to 1
        total = act.sum()
        if total > 0:
            act = act / total
        # Now act sums to exactly 1.0, split into 5 alloc + 1 excess
        alloc = act[:self._env.n_slices]   # sums to < 1.0
        excess = act[self._env.n_slices]   # saved budget, added to reward

        alloc_prbs = np.array([int(np.floor(a * self._n_prbs)) for a in alloc], dtype=int)
        result = self._env.step(alloc_prbs)
        

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        reward = float(reward) + float(excess)
        cost = float(info.get("cost", 0.0))

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0

        if isinstance(info, dict):
            info = {str(k): v for k, v in info.items()}
        else:
            info = {}

        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            new_obs, _ = self._env.reset()
            obs = torch.as_tensor(new_obs, dtype=torch.float32)
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
    def max_episode_steps(self) -> int:
        return self._max_episode_steps 
class Evaluator():
    def __init__(self, scenario=SCENARIO):
        self.scenario = scenario
        self.train_path = f'./results/scenario_{scenario}/PPO/'
        self.test_path = f'./results/scenario_{scenario}/PPO_t/'
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

    def evaluate(self, run_id):
        rng = default_rng(seed=run_id)
        env = RanSliceEnv(env_id=ENV_ID, rng=rng, scenario=self.scenario, penalty=PENALTY)

        # ==================== TRAINING ====================
        print(f'\n=== Training Run {run_id} ===')
        custom_cfgs = {
            "train_cfgs": {
                "total_steps": _TOTAL_STEPS,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "update_iters": 1,
            },
            "logger_cfgs": {
                "use_wandb": False,
                "save_model_freq": 1,
            },
        }

        agent = omnisafe.Agent(
            algo="PPO",
            env_id=ENV_ID,
            custom_cfgs=custom_cfgs,
        )

        print('Training started...')
        agent.learn()
        print('Training done!')
        agent.plot(smooth=1)

      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train PPO on RanSlice with OmniSafe")
    parser.add_argument("--scenario", type=int, default=SCENARIO)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    evaluator = Evaluator(scenario=args.scenario)

    for run in range(args.runs):
        evaluator.evaluate(run)