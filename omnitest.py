#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO-Lag (OmniSafe 0.5.0) on the RanSlice gymnasium environment.

In OmniSafe 0.5.0 the correct base class is `CMDP` (not `BaseEnv`),
and the registration decorator is `env_register` — both from omnisafe.envs.core.

Usage:
    python train_ppo_lag.py
    python train_ppo_lag.py --scenario 1 --epochs 200 --steps 20000
"""

import argparse
from typing import ClassVar

import numpy as np
import torch
import omnisafe
from omnisafe.envs.core import CMDP, env_register

# ── import your env factory ──────────────────────────────────────────────────
from create_env import create_env   # the file you shared


# ── Register the custom CMDP env ─────────────────────────────────────────────

ENV_ID = "RanSlicePPOLag-v0"

# Module-level globals so the factory closure survives re-imports
_RNG     = None
_SCEN    = 0
_PENALTY = 100.0


@env_register
class RanSliceEnv(CMDP):
    """
    OmniSafe 0.5.0 CMDP wrapper around the RanSlice gymnasium environment.

    Required interface:
      - _support_envs  : list of accepted env-id strings
      - __init__(env_id, **kwargs)
      - step(action: Tensor) -> (obs, reward, cost, terminated, truncated, info)
      - reset(**kwargs)      -> (obs: Tensor, info: dict)
      - set_seed(seed: int)
      - spec_log(logger)     [no-op is fine]
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = True
    need_time_limit_wrapper  = True

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)

        # Use the module-level globals set before agent creation
        self._env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._action_space      = self._env.action_space
        self._observation_space = self._env.observation_space

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        result = self._env.reset()
        # Handle old gym API (obs only) vs new (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return torch.as_tensor(obs, dtype=torch.float32), info

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        result = self._env.step(act)

        # Handle old 4-value step API (obs, reward, done, info)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        # ── cost signal ───────────────────────────────────────────────────────
        # Replace with your actual SLA-violation signal if available, e.g.:
        #   cost = float(info.get("sla_violation", False))
        cost = float(reward < 0)

        return (
            torch.as_tensor(obs,    dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost,   dtype=torch.float32),
            terminated,
            truncated,
            info,
        )

    # ── seed ──────────────────────────────────────────────────────────────────
    def set_seed(self, seed: int) -> None:
        pass  # rng fixed at construction; override if you want per-run seeding

    # ── optional per-epoch logging hook ──────────────────────────────────────
    def spec_log(self, logger) -> None:
        pass

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self) -> None:
        self._env.close()


# ── OmniSafe custom cfgs ──────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global _RNG, _SCEN, _PENALTY

    parser = argparse.ArgumentParser(description="Train PPO-Lag on RanSlice env")
    parser.add_argument("--scenario",   type=int,   default=0,     help="Scenario index (0-3)")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    parser.add_argument("--epochs",     type=int,   default=100,   help="Number of training epochs")
    parser.add_argument("--steps",      type=int,   default=10000, help="Steps per epoch")
    parser.add_argument("--cost_limit", type=float, default=25.0,  help="Cost constraint limit")
    parser.add_argument("--penalty",    type=float, default=100.0, help="SLA penalty in the env")
    parser.add_argument("--device",     type=str,   default="cpu", help="Torch device (cpu / cuda)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PPO-Lag (OmniSafe 0.5.0) on RanSlice — scenario {args.scenario}")
    print(f"  seed={args.seed}  epochs={args.epochs}  steps/epoch={args.steps}")
    print(f"  cost_limit={args.cost_limit}  device={args.device}")
    print(f"{'='*60}\n")

    # Set module-level globals before OmniSafe instantiates the env
    _RNG     = np.random.default_rng(args.seed)
    _SCEN    = args.scenario
    _PENALTY = args.penalty

    # ── quick sanity check ────────────────────────────────────────────────────
    test_env = RanSliceEnv(ENV_ID)
    obs, info = test_env.reset()
    print(f"Observation space : {test_env.observation_space}")
    print(f"Action space      : {test_env.action_space}")
    act = torch.as_tensor(test_env.action_space.sample(), dtype=torch.float32)
    obs, rew, cost, term, trunc, info = test_env.step(act)
    print(f"Sample step  →  reward={rew.item():.4f}  cost={cost.item():.4f}\n")
    test_env.close()

    # ── build & train agent ───────────────────────────────────────────────────
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

    print("Starting training …\n")
    agent.learn()

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating trained policy …")
    agent.plot(smooth=1)
    agent.evaluate(num_episodes=10)

    print("\nDone. Results are in the 'runs/' directory.")


if __name__ == "__main__":
    main()