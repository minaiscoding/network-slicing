#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallelized PPO-Lag (OmniSafe 0.5.0) training on RanSlice environment.
Each run uses a unique run_id for proper result separation and saving.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import ClassVar
import numpy as np
import torch
import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from gymnasium import spaces
from scenario_creator import create_env
from wrapper import ReportWrapper

ENV_ID = "RanSlicePPOLag-v0"

_RNG = None
_SCEN = 0
_PENALTY = 100.0
_TOTAL_STEPS = 20000
RUNS = 1

@env_register
@env_unregister
class RanSliceEnv(CMDP):
    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs = 1

    def __init__(self, env_id: str, run_id: int = 0, **kwargs):
        super().__init__(env_id)
        raw_env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=run_id,
            path=f'./results/scenario_{_SCEN}/PPOLag/',
            verbose=False,
            n_slices=raw_env.n_slices,
            n_prbs=raw_env.n_prbs,
            n_variables=raw_env.n_variables,
        )

        self._max_episode_steps = 500
        self._n_prbs = self._env.n_prbs
        self._step_count = 0
        self._prev_alloc = np.zeros(self._env.n_slices, dtype=int)

        self._action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._env.n_slices + 1,), dtype=float
        )
        self._observation_space = spaces.Box(
            low=-1, high=1, shape=(self._env.n_variables,), dtype=float
        )

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._prev_alloc = np.zeros(self._env.n_slices, dtype=int)
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action: torch.Tensor):
        act = np.abs(action.cpu().numpy())
        if act.sum() > 0:
            act = act / act.sum()

        alloc = act[:self._env.n_slices]
        excess = act[self._env.n_slices]
        alloc_prbs = np.floor(alloc * self._n_prbs).astype(int)
        obs, env_reward, terminated, truncated, info = self._env.step(alloc_prbs)

        og_reward = float(env_reward)
        alloc_diff = np.sum(np.abs(alloc_prbs - self._prev_alloc))
        stability_bonus = 1.0 / (1.0 + alloc_diff)
        reward = float(excess) + stability_bonus
        self._prev_alloc = alloc_prbs.copy()
        cost = -og_reward

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0

        info = {} if not isinstance(info, dict) else {str(k): v for k, v in info.items()}

        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            new_obs, _ = self._env.reset()
            obs = torch.as_tensor(new_obs, dtype=torch.float32)
            info["final_observation"] = final_obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            info["final_observation"] = torch.as_tensor(obs, dtype=torch.float32)

        return (
            obs,
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )

    def close(self):
        self._env.close()

    def render(self, mode=None):
        if hasattr(self._env, "render"):
            return self._env.render(mode=mode)
        return None

    def set_seed(self, seed: int) -> None:
        global _RNG
        _RNG = np.random.default_rng(seed)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps


def build_custom_cfgs(epochs: int, steps_per_epoch: int, cost_limit: float, device: str) -> dict:
    return {
        "train_cfgs": {"total_steps": epochs * steps_per_epoch, "device": device},
        "algo_cfgs": {"steps_per_epoch": steps_per_epoch},
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }


def train_run(run_id, scenario, seed, epochs, steps, penalty, cost_limit, device):
    global _RNG, _SCEN, _PENALTY, _TOTAL_STEPS
    _RNG = np.random.default_rng(seed)
    _SCEN = scenario
    _PENALTY = penalty
    _TOTAL_STEPS = epochs * steps

    # Create env first and keep reference
    env = RanSliceEnv(ENV_ID, run_id=run_id)

    custom_cfgs = build_custom_cfgs(epochs, steps, cost_limit, device)
    custom_cfgs["env_cfgs"] = {"run_id": run_id}

    print(f"Starting run {run_id} (seed={seed})...")
    agent = omnisafe.Agent(algo="PPOLag", env_id=ENV_ID, custom_cfgs=custom_cfgs)

    # train
    agent.learn()
    agent.plot(smooth=1)
    env._env.save_results()
    print(f"Run {run_id} training results saved.")

    # evaluate
    agent.evaluate(num_episodes=1)
    env._env.save_results()
    print(f"Run {run_id} evaluation results saved.")


def main():
    parser = argparse.ArgumentParser(description="Train PPO-Lag on RanSlice env (parallel runs)")
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200000)  # adjust to cover total steps
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cost_limit", type=float, default=5.0)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    seeds = list(range(args.runs))
    with ProcessPoolExecutor(min(args.runs, os.cpu_count())) as executor:
        futures = [
            executor.submit(
                train_run,
                run_id=i,
                scenario=args.scenario,
                seed=seeds[i],
                epochs=args.epochs,
                steps=args.steps,
                penalty=args.penalty,
                cost_limit=args.cost_limit,
                device=args.device
            )
            for i in range(args.runs)
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()