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
from wrapper import PPOReportWrapper
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from scenario_creator import create_env

SCENARIO = 4
RUNS = 30
PROCESSES = 4
EPOCHS = 100
STEPS_PER_EPOCH = 36000
SLOTS_PER_STEP = 100
PENALTY = 1000
EVALUATION_STEPS = 360000

ENV_ID = "RanSlicePPOLag-v0"

_RNG         = np.random.default_rng(3)
_SCEN        = 4
_PENALTY     = 100.0
_TOTAL_STEPS = 3600000

@env_register
@env_unregister
class RanSliceEnv(CMDP):

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id)

        raw_env = create_env(_RNG, _SCEN, penalty=_PENALTY)

        self._env = PPOReportWrapper(               # was: ReportWrapper
    raw_env,
    steps=_TOTAL_STEPS,
    control_steps=500,
    env_id=_RNG.integers(0, 20),                           # tip: change this from 1 → run_id so files don't overwrite
    path='./results/scenario_test/PPO/',
    verbose=False,
    continuous_mode=True
)

        self._max_episode_steps = 600
        self._step_count = 0

        # slice info
        self._n_slices = self._env.n_slices

        # ACTION SPACE
        self._action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n_slices + 1,),
            dtype=np.float32
        )

        self._observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.n_variables,),
            dtype=np.float32
        )

        self._last_allocation = np.zeros(self._n_slices, dtype=float)

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._last_allocation = np.zeros(self._n_slices, dtype=float)

        obs, info = self._env.reset(seed=seed, options=options)

        obs = np.asarray(obs, dtype=np.float32)

        return torch.as_tensor(obs, dtype=torch.float32), info

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action: torch.Tensor):

        act = action.cpu().numpy()
        act = np.abs(act)

        # normalize allocation
        total = act.sum()
        if total > 0:
            act = act / total

        alloc = act[:self._n_slices]
        excess = act[self._n_slices]

        alloc_prbs = np.array(
            [int(np.floor(a * self._env.n_prbs)) for a in alloc],
            dtype=int
        )

        obs, reward, terminated, truncated, info = self._env.step(alloc_prbs)

        obs = np.asarray(obs, dtype=np.float32)

        cost = float(info.get("cost", 0.0))

        reward = self._compute_custom_reward(alloc_prbs, excess, info)

        # episode truncation
        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0

        # reset handling
        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)

            new_obs, new_info = self._env.reset()
            new_obs = np.asarray(new_obs, dtype=np.float32)

            obs = torch.as_tensor(new_obs, dtype=torch.float32)
            info["final_observation"] = final_obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            info["final_observation"] = obs

        return (
            obs,
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )

    # =========================================================
    # CUSTOM REWARD (unchanged)
    # =========================================================
    def _compute_custom_reward(self, alloc_prbs, excess, info):

        reward = 0.0

        l1_info = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])

        for slice_idx in range(self._n_slices):
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]

                if isinstance(slice_info, dict):
                    for _, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):

                            slice_type = ran_info.get('type', '')

                            # URLLC
                            if 'urllc' in slice_type.lower() or slice_idx == 2:
                                d = ran_info.get('cbr_delay', 0.0)
                                reward += -3.0 * np.clip(d / 100.0, 0, 1)

                            # eMBB
                            elif 'embb' in slice_type.lower() or slice_idx == 0:
                                d = ran_info.get('cbr_delay', 0.0)
                                reward += -2.0 * np.clip(d / 100.0, 0, 1)

                                th = ran_info.get('cbr_th', 0.0)
                                reward += 2.0 * np.clip(th / 1e6, 0, 1)

                            # mMTC
                            elif 'mmtc' in slice_type.lower() or slice_idx == 1:
                                d = ran_info.get('delay', 0.0)
                                reward += -1.0 * np.clip(d / 100.0, 0, 1)

        remaining = self._env.n_prbs - alloc_prbs.sum()
        reward += 0.1 * (remaining / self._env.n_prbs)

        change = np.sum(np.abs(alloc_prbs - self._last_allocation))
        reward -= 0.05 * (change / self._env.n_prbs)

        self._last_allocation = alloc_prbs.astype(float)

        return reward
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
            algo="PPOLag",
            env_id=ENV_ID,
            custom_cfgs=custom_cfgs,
        )

        print('Training started...')
        agent.learn()
        print('Training done!')
        agent.plot(smooth=1)
        # ==================== EVALUATION ====================
      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train PPO  lag on RanSlice with OmniSafe")
    parser.add_argument("--scenario", type=int, default=SCENARIO)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    evaluator = Evaluator(scenario=args.scenario)

    for run in range(args.runs):
        evaluator.evaluate(run)