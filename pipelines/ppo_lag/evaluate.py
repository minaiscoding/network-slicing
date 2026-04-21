#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained PPOLag checkpoint across low / medium / congested scenarios.

Uses ReportWrapper and saves eval_history_{scenario}.npz inside pipelines/ppo_lag/.

Usage:
    python pipelines/ppo_lag/evaluate.py --checkpoint <path-to-.pt>
"""

import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import omnisafe

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import algos.ppo_lag as ppolag_mod
from config_loader import load_scenarios
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper

PIPELINE_DIR   = os.path.dirname(os.path.abspath(__file__))
SCENARIO_YAML  = "scenarios.yaml"
SCENARIO_NAMES = ["low", "medium", "congested"]
SLICE_LABELS   = ["eMBB", "mMTC", "URLLC"]


def load_checkpoint_into_agent(agent, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in ckpt:
        raise KeyError("Checkpoint missing 'pi' key.")
    agent.agent._actor_critic.actor.load_state_dict(ckpt["pi"], strict=True)


@dataclass
class ScenarioResult:
    name: str
    n_slices: int
    violation_per_slice: np.ndarray
    resource_per_slice:  np.ndarray


def _compute_observation(info, n_slices, n_prbs):
    obs = np.zeros(n_slices * 4, dtype=float)
    l1_info     = info.get('l1_info', [])
    n_prbs_list = info.get('n_prbs', [])
    idx = 0
    for s in range(n_slices):
        bler = snr = traffic = 0.0
        if s < len(l1_info):
            si = l1_info[s]
            if isinstance(si, dict):
                for _, ri in si.items():
                    if isinstance(ri, dict):
                        bler    = max(ri.get('cbr_bler', 0.0), ri.get('vbr_bler', 0.0))
                        snr     = (ri.get('cbr_snr', 0.0) + ri.get('vbr_snr', 0.0)) / 2.0
                        traffic = (ri.get('cbr_queue', 0.0) + ri.get('vbr_queue', 0.0)) / 2.0
        bler    = np.clip(bler, 0, 1)
        snr     = np.clip(snr / 30.0, -1, 1)
        traffic = np.clip(traffic / 100000.0, 0, 1)
        alloc   = np.clip(n_prbs_list[s] / max(n_prbs, 1), 0, 1) if s < len(n_prbs_list) else 0.0
        obs[idx]                  = bler
        obs[idx + n_slices]       = snr
        obs[idx + 2 * n_slices]   = traffic
        obs[idx + 3 * n_slices]   = alloc
        idx += 1
    return obs


def run_scenario(agent, scenario_name, seed, epochs, steps_per_epoch, penalty):
    total_steps = epochs * steps_per_epoch
    configs = load_scenarios(SCENARIO_YAML)
    cfg     = configs[scenario_name]

    rng     = np.random.default_rng(seed)
    raw_env = create_env_from_config(cfg, rng, penalty=penalty)
    path    = os.path.join(PIPELINE_DIR, '')
    wrapped = ReportWrapper(
        raw_env, steps=total_steps, control_steps=total_steps + 1,
        env_id=f"eval_{scenario_name}", path=path, verbose=False,
    )
    n_slices = wrapped.n_slices
    _n_prbs  = cfg.n_prbs

    obs_raw, info = wrapped.reset()
    if isinstance(obs_raw, tuple):
        obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
    obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs), dtype=torch.float32)

    print(f"\nEvaluating {scenario_name} ({total_steps} steps)...")
    for step in range(total_steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)
        act = np.abs(action.detach().cpu().numpy())
        total = float(act.sum())
        if total > 0:
            act = act / total
        alloc_prbs = np.array([int(np.floor(a * _n_prbs)) for a in act[:n_slices]], dtype=int)

        result = wrapped.step(alloc_prbs)
        if len(result) == 4:
            _, _, _, info = result
        else:
            _, _, _, _, info = result
        obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs), dtype=torch.float32)

    wrapped.save_results()
    print(f"Completed {scenario_name} — saved eval_history.")

    return ScenarioResult(
        name=scenario_name, n_slices=n_slices,
        violation_per_slice=wrapped.violation_per_slice_history.copy(),
        resource_per_slice=wrapped.resource_per_slice_history.copy(),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPOLag checkpoint")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--seed",            type=int,   default=3)
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--steps-per-epoch", type=int,   default=1000)
    parser.add_argument("--penalty",         type=float, default=100.0)
    parser.add_argument("--device",          type=str,   default="cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    total_steps = args.epochs * args.steps_per_epoch

    ppolag_mod._penalty     = args.penalty
    ppolag_mod._total_steps = total_steps
    ppolag_mod._rng         = np.random.default_rng(args.seed)

    custom_cfgs = {
        "train_cfgs": {"total_steps": max(total_steps, 1), "device": args.device},
        "algo_cfgs":  {"steps_per_epoch": max(args.steps_per_epoch, 1), "cost_limit": 25.0, "use_cost": True},
        "logger_cfgs": {"use_wandb": False, "save_model_freq": 1},
    }

    print("Initializing PPOLag agent from checkpoint...")
    agent = omnisafe.Agent(algo="PPOLag", env_id=ppolag_mod.ENV_ID, custom_cfgs=custom_cfgs)
    load_checkpoint_into_agent(agent, args.checkpoint)

    results = {}
    for name in SCENARIO_NAMES:
        results[name] = run_scenario(
            agent, name, args.seed, args.epochs, args.steps_per_epoch, args.penalty,
        )

    # Summary
    print(f"\n{'Scenario':12} | {'Slice':6} | {'Violations':>10} | {'Avg PRBs':>9}")
    print("-" * 50)
    for name in SCENARIO_NAMES:
        r = results[name]
        for si, label in enumerate(SLICE_LABELS[:r.n_slices]):
            viols    = r.violation_per_slice[:, si].sum()
            avg_prbs = r.resource_per_slice[:, si].mean()
            prefix   = f"{name.upper():12}" if si == 0 else f"{'':12}"
            print(f"{prefix} | {label:6} | {viols:10.0f} | {avg_prbs:9.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
