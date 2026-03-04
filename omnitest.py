#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO-Lag (OmniSafe 0.5.0) on the RanSlice gymnasium environment.

Usage:
    python omnitest.py
    python omnitest.py --scenario 1 --epochs 200 --steps 1000
"""

import argparse
from typing import ClassVar

import numpy as np
import torch
import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from gymnasium import spaces

from scenario_creator import create_env
from wrapper import ReportWrapper 


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
            path='./results/scenario_0/PPOLag/',
            verbose=False,
            n_slices=raw_env.n_slices,
            n_prbs=raw_env.n_prbs,
            n_variables=raw_env.n_variables
        )

        self._max_episode_steps = 500
        self._n_prbs            = self._env.n_prbs  # Read from environment, not hard-coded
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
        print(f'Action taken: {alloc_prbs}, Excess bonus: {excess:.4f}')

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        reward = float(reward) + float(excess)
        cost = 0.0

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
        return self._max_episode_steps  # 500, was wrongly returning 1 before


def build_custom_cfgs(epochs: int, steps_per_epoch: int,
                      cost_limit: float, device: str) -> dict:
    return {
        "train_cfgs": {
            "total_steps": epochs * steps_per_epoch,
            "device": device,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }


def main():
    global _RNG, _SCEN, _PENALTY, _TOTAL_STEPS

    parser = argparse.ArgumentParser(description="Train PPO-Lag on RanSlice env")
    parser.add_argument("--scenario",   type=int,   default=0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--steps",      type=int,   default=1000)  # must be >= 500 (max_episode_steps)
    parser.add_argument("--cost_limit", type=float, default=25.0)
    parser.add_argument("--penalty",    type=float, default=100.0)
    parser.add_argument("--device",     type=str,   default="cpu")
    args = parser.parse_args()

    _RNG         = np.random.default_rng(args.seed)
    _SCEN        = args.scenario
    _PENALTY     = args.penalty
    _TOTAL_STEPS = args.epochs * args.steps

    custom_cfgs = build_custom_cfgs(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        cost_limit=args.cost_limit,
        device=args.device,
    )

    agent = omnisafe.Agent(
        algo="PPOLag",
        env_id=ENV_ID,
        custom_cfgs=custom_cfgs,
    )

    agent.learn()
    print(agent)
    agent.plot(smooth=1)

    agent.evaluate(num_episodes=1)


if __name__ == "__main__":
    main()