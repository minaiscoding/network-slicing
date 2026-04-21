#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import omnisafe

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _PROJECT_ROOT)

import algorithms.cpo as train_mod

from scenario_creator import print_bps

ENV_ID = "RanSliceCPOSingleScenario-v0"


def load_checkpoint_into_agent(agent, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    agent.agent._actor_critic.actor.load_state_dict(ckpt["pi"], strict=True)


def run_eval(agent, steps):
    env = train_mod.RanSliceCPOEnv(ENV_ID)

    obs, _ = env.reset()
    total_reward = 0.0
    total_cost = 0.0

    print(f"\nRunning evaluation for {steps} steps...")

    for step in range(steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)

        # IMPORTANT: no manual processing
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        print(obs)

        total_reward += reward.item()
        total_cost += cost.item()

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    print("\n=== Evaluation Results ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total cost:   {total_cost:.2f}")
    print(f"Avg reward:   {total_reward / steps:.4f}")
    print(f"Avg cost:     {total_cost / steps:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    # Sync global config with training
    train_mod._scenario = "medium"
    train_mod._steps_per_episode = 600
    train_mod._steps_per_epoch = 36000
    train_mod._total_steps = args.steps
    train_mod._rng = np.random.default_rng(3)
    train_mod._current_run_id = 999  # separate from training

    custom_cfgs = {
        "train_cfgs": {"total_steps": args.steps, "device": args.device},
        "algo_cfgs": {
            "steps_per_epoch": 36000,
            "cost_limit": 100.0,
            "use_cost": True,
            "obs_normalize": True,
            "reward_normalize": True,
            "cost_normalize": True,
        },
        "model_cfgs": {
            "actor": {"hidden_sizes": [256, 256], "activation": "tanh"},
            "critic": {"hidden_sizes": [256, 256], "activation": "tanh", "lr": 3e-4},
        },
        "logger_cfgs": {"use_wandb": False},
    }

    print("Loading agent...")
    agent = omnisafe.Agent(algo="CPO", env_id=ENV_ID, custom_cfgs=custom_cfgs)
    load_checkpoint_into_agent(agent, args.checkpoint)

    run_eval(agent, args.steps)


if __name__ == "__main__":
    main()