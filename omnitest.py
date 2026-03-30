#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO (OmniSafe 0.5.0) on the RanSlice gymnasium environment.

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
class RanSliceEnv(CMDP):

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = True
    need_time_limit_wrapper  = False
    _num_envs                = 1          # ← required by OmniSafe AutoReset

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        self._env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._n_prbs            = 200     # total PRBs to distribute
        self._action_space      = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self._observation_space = self._env.observation_space

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return torch.as_tensor(obs, dtype=torch.float32), info

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        # Convert continuous [0,1] action from OmniSafe → integer PRBs
        act = action.cpu().numpy()
        act = np.abs(act)
        t_action = act.sum() or 1.0
        act = np.array(
            [int(np.floor(self._n_prbs * act[i] / t_action)) for i in range(len(act))],
            dtype=int,
        )

        result = self._env.step(act)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost = float(reward < 0)

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

def build_custom_cfgs(epochs: int, steps_per_epoch: int, device: str) -> dict:
    return {
        "train_cfgs": {
            "total_steps": epochs * steps_per_epoch,
            "device": device,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "update_iters": 1,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }


def main():
    global _RNG, _SCEN, _PENALTY, _TOTAL_STEPS

    parser = argparse.ArgumentParser(description="Train PPO on RanSlice env")
    parser.add_argument("--scenario",   type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--steps",      type=int,   default=1000)  # must be >= 500 (max_episode_steps)
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
        device=args.device,
    )

    agent = omnisafe.Agent(
        algo="PPO",
        env_id=ENV_ID,
        custom_cfgs=custom_cfgs,
    )

    agent.learn()
    agent.plot(smooth=1)

    agent.evaluate(num_episodes=1)


if __name__ == "__main__":
    main()