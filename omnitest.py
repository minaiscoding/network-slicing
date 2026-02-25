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
    need_auto_reset_wrapper  = True
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

        return (
            torch.as_tensor(obs,        dtype=torch.float32),
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


def build_custom_cfgs(epochs: int, steps_per_epoch: int,
                      cost_limit: float, device: str) -> dict:
    return {
        "train_cfgs": {
            "total_steps": epochs * steps_per_epoch,
            "device": device,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "update_iters": 1,
        },
        "lagrange_cfgs": {
            "cost_limit": cost_limit,
            "lagrangian_multiplier_init": 0.001,
            "lambda_lr": 0.035,
            "lambda_optimizer": "Adam",
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 10,
        },
    }


def main():
    global _RNG, _SCEN, _PENALTY, _TOTAL_STEPS

    parser = argparse.ArgumentParser(description="Train PPO-Lag on RanSlice env")
    parser.add_argument("--scenario",   type=int,   default=0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=200)   # ✅ good default
    parser.add_argument("--steps",      type=int,   default=1000)  # ✅ good default
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
    agent.plot(smooth=1)
    agent.evaluate(num_episodes=10)


if __name__ == "__main__":
    main()