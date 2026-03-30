#!/usr/bin/env python3
"""Evaluate a trained PPOLag checkpoint across low / medium / congested scenarios.

Generates a single PNG (scenario_comparison.png) with three panels:
  1. Total allocated PRBs per step
  2. SLA violations per step
  3. Cumulative SLA violations

Example:
  python evaluate_ppolag_checkpoint.py \
    --checkpoint runs/PPOLag-{RanSlicePPOLag-v0}/seed-000-2026-03-17-23-09-24/torch_save/epoch-20.pt \
    --epochs 5 --steps-per-epoch 1000
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import omnisafe

import experiments_ppo_lag as train_exp
from config_loader import load_scenarios
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper
from gymnasium import spaces

SCENARIO_YAML  = "scenarios.yaml"
SCENARIO_NAMES = ["low", "medium", "congested"]
COLORS         = ["#2196F3", "#FF9800", "#F44336"]   # blue, orange, red

# Scenario index used only to satisfy omnisafe agent construction — the actual
# env built for evaluation is created from the YAML config, not this index.
_DUMMY_SCEN = 4


def load_checkpoint_into_agent(agent: omnisafe.Agent, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(checkpoint["pi"], strict=True)


def make_wrapped_env(cfg, seed: int, penalty: float, total_steps: int):
    """Build a ReportWrapper-wrapped env from a ScenarioConfig, mirroring
    the setup inside RanSliceEnv.__init__ but driven by the YAML config."""
    path = f"./results/{cfg.name}/PPOLag_eval/"
    os.makedirs(path, exist_ok=True)
    rng     = np.random.default_rng(seed)
    raw_env = create_env_from_config(cfg, rng, penalty=penalty)
    env = ReportWrapper(
        raw_env,
        steps=total_steps,
        control_steps=500,
        env_id=1,
        path=path,
        verbose=False,
    )
    return env


def run_scenario(
    agent: omnisafe.Agent,
    scenario_name: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out the frozen agent on one scenario.

    Returns
    -------
    prb_history        : (total_steps,) total allocated PRBs per step
    violations_history : (total_steps,) total SLA violations per step
    """
    total_steps = epochs * steps_per_epoch

    configs = load_scenarios(SCENARIO_YAML)
    cfg     = configs[scenario_name]

    wrapped = make_wrapped_env(cfg, seed, penalty, total_steps)
    n_slices = wrapped.n_slices
    n_prbs   = train_exp.RanSliceEnv.__init__.__defaults__  # not used directly

    # Read n_prbs the same way RanSliceEnv does.
    _n_prbs = 100   # matches train_exp.RanSliceEnv._n_prbs

    obs_raw, _ = wrapped.reset() if hasattr(wrapped.reset, '__call__') else (wrapped.reset(), {})
    if isinstance(obs_raw, tuple):
        obs_raw = obs_raw[0]
    obs = torch.as_tensor(obs_raw, dtype=torch.float32)

    prb_history        = np.zeros(total_steps, dtype=float)
    violations_history = np.zeros(total_steps, dtype=float)

    for global_step in range(total_steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)

        # Mirror the exact action mapping from RanSliceEnv.step.
        act_np = action.detach().cpu().numpy()
        act_np = np.abs(act_np)
        total  = float(act_np.sum())
        if total > 0:
            act_np = act_np / total
        alloc      = act_np[:n_slices]
        alloc_prbs = np.array([int(np.floor(a * _n_prbs)) for a in alloc], dtype=int)

        prb_history[global_step] = alloc_prbs.sum()

        result = wrapped.step(alloc_prbs)
        if len(result) == 4:
            obs_raw, _reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs_raw, _reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        # violations is a numpy array of per-slice violation flags from NodeB.
        violations = info.get("violations", np.zeros(n_slices))
        violations_history[global_step] = int(np.array(violations).sum())



        obs = torch.as_tensor(
            obs_raw if not isinstance(obs_raw, torch.Tensor) else obs_raw.numpy(),
            dtype=torch.float32,
        )

    wrapped.close()
    return prb_history, violations_history


def save_comparison_plot(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    steps_per_epoch: int,
    epochs: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total_steps = steps_per_epoch * epochs
    x = np.arange(total_steps)
    epoch_boundaries = [i * steps_per_epoch for i in range(1, epochs)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax in axes:
        for b in epoch_boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25)

    # panel 1 — allocated PRBs per step
    ax = axes[0]
    for name, color in zip(SCENARIO_NAMES, COLORS):
        prbs, _ = results[name]
        ax.plot(x, prbs, label=name, color=color, linewidth=1.2)
    ax.set_title("Total Allocated PRBs per Step")
    ax.set_ylabel("PRBs")
    ax.legend(fontsize=9)

    # panel 2 — violations per step
    ax = axes[1]
    for name, color in zip(SCENARIO_NAMES, COLORS):
        _, viols = results[name]
        ax.plot(x, viols, label=name, color=color, linewidth=1.2)
    ax.set_title("SLA Violations per Step")
    ax.set_ylabel("Violations")
    ax.legend(fontsize=9)

    # panel 3 — cumulative violations
    ax = axes[2]
    for name, color in zip(SCENARIO_NAMES, COLORS):
        _, viols = results[name]
        ax.plot(x, np.cumsum(viols), label=name, color=color, linewidth=1.2)
    ax.set_title("Cumulative SLA Violations")
    ax.set_ylabel("Cumulative violations")
    ax.legend(fontsize=9)

    out_path = os.path.join(out_dir, "scenario_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/PPOLag-{RanSlicePPOLag-v0}/seed-000-2026-03-17-23-09-24/torch_save/epoch-20.pt",
    )
    parser.add_argument("--seed",            type=int,   default=3)
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--steps-per-epoch", type=int,   default=1000)
    parser.add_argument("--penalty",         type=float, default=100.0)
    parser.add_argument("--device",          type=str,   default="cpu")
    parser.add_argument("--out-dir",         type=str,   default="results/scenario_comparison")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    total_steps = args.epochs * args.steps_per_epoch

    # Set globals so RanSliceEnv registration succeeds (omnisafe requires it).
    train_exp._SCEN        = _DUMMY_SCEN
    train_exp._PENALTY     = args.penalty
    train_exp._TOTAL_STEPS = total_steps
    train_exp._RNG         = np.random.default_rng(args.seed)

    custom_cfgs = {
        "train_cfgs": {
            "total_steps": max(total_steps, 1),
            "device": args.device,
        },
        "algo_cfgs": {
            "steps_per_epoch": max(args.steps_per_epoch, 1),
            "update_iters": 1,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }

    agent = omnisafe.Agent(algo="PPOLag", env_id=train_exp.ENV_ID, custom_cfgs=custom_cfgs)
    load_checkpoint_into_agent(agent, checkpoint_path)

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in SCENARIO_NAMES:
        results[name] = run_scenario(
            agent          = agent,
            scenario_name  = name,
            seed           = args.seed,
            epochs         = args.epochs,
            steps_per_epoch= args.steps_per_epoch,
            penalty        = args.penalty,
        )

    save_comparison_plot(
        results         = results,
        steps_per_epoch = args.steps_per_epoch,
        epochs          = args.epochs,
        out_dir         = args.out_dir,
    )


if __name__ == "__main__":
    main()