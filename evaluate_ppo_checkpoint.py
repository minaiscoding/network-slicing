#!/usr/bin/env python3
"""Evaluate a trained PPOLag checkpoint across low / medium / congested scenarios.

Generates per-scenario plots:
  1. Cumulative SLA violations per slice (eMBB, mMTC, URLLC)
  2. PRB allocation per slice over time
  3. SLA violation rate (rolling average)

Plus an overall scenario comparison summary plot.

Example:
  python evaluate_ppolag_checkpoint.py \
    --checkpoint runs/PPOLag-{RanSlicePPOLag-v0}/seed-000-.../torch_save/epoch-10.pt \
    --epochs 5 --steps-per-epoch 1000
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import omnisafe

import experiments_ppo as train_exp
from config_loader import load_scenarios
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper

SCENARIO_YAML  = "scenarios.yaml"
SCENARIO_NAMES = ["low", "medium", "congested"]

SLICE_LABELS = ["eMBB", "mMTC", "URLLC"]
SLICE_COLORS = ["#2196F3", "#4CAF50", "#F44336"]

SCENARIO_COLORS = ["#2196F3", "#FF9800", "#F44336"]

SMOOTH_WINDOW = 50


def load_checkpoint_into_agent(agent: omnisafe.Agent, checkpoint_path: str) -> None:
    """Load frozen policy weights from checkpoint into agent's actor network."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(checkpoint["pi"], strict=True)


def make_wrapped_env(cfg, seed: int, penalty: float, total_steps: int):
    """Build a ReportWrapper-wrapped env from a ScenarioConfig."""
    path = f"./results/{cfg.name}/PPOLag_eval/"
    os.makedirs(path, exist_ok=True)
    rng     = np.random.default_rng(seed)
    raw_env = create_env_from_config(cfg, rng, penalty=penalty)
    env = ReportWrapper(
        raw_env,
        steps=total_steps,
        control_steps=total_steps + 1,
        env_id=f"eval_{cfg.name}",
        path=path,
        verbose=False,
    )
    return env


@dataclass
class ScenarioResult:
    name: str
    n_slices: int
    violation_per_slice: np.ndarray
    resource_per_slice:  np.ndarray


def _compute_observation(info: dict, n_slices: int, n_prbs: int) -> np.ndarray:
    """Replicate the 12-dim observation used during PPOLag training."""
    obs_size = n_slices * 4
    obs = np.zeros(obs_size, dtype=float)
    l1_info     = info.get('l1_info', [])
    n_prbs_list = info.get('n_prbs', [])

    idx = 0
    for slice_idx in range(n_slices):
        bler = snr = traffic = 0.0

        if slice_idx < len(l1_info):
            slice_info = l1_info[slice_idx]
            if isinstance(slice_info, dict):
                for ran_id, ran_info in slice_info.items():
                    if isinstance(ran_info, dict):
                        cbr_bler = ran_info.get('cbr_bler', 0.0)
                        vbr_bler = ran_info.get('vbr_bler', 0.0)
                        bler     = max(cbr_bler, vbr_bler)

                        cbr_snr = ran_info.get('cbr_snr', 0.0)
                        vbr_snr = ran_info.get('vbr_snr', 0.0)
                        snr     = (cbr_snr + vbr_snr) / 2.0

                        cbr_queue = ran_info.get('cbr_queue', 0.0)
                        vbr_queue = ran_info.get('vbr_queue', 0.0)
                        traffic   = (cbr_queue + vbr_queue) / 2.0

        bler    = np.clip(bler, 0, 1)
        snr     = np.clip(snr / 30.0, -1, 1)
        traffic = np.clip(traffic / 100000.0, 0, 1)
        alloc   = (n_prbs_list[slice_idx] / max(n_prbs, 1)
                   if slice_idx < len(n_prbs_list) else 0.0)
        alloc   = np.clip(alloc, 0, 1)

        obs[idx]                    = bler
        obs[idx + n_slices]         = snr
        obs[idx + 2 * n_slices]     = traffic
        obs[idx + 3 * n_slices]     = alloc
        idx += 1

    return obs


def run_scenario(
    agent: omnisafe.Agent,
    scenario_name: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    penalty: float,
) -> ScenarioResult:
    """Roll out the frozen agent on one scenario."""
    total_steps = epochs * steps_per_epoch

    configs = load_scenarios(SCENARIO_YAML)
    cfg     = configs[scenario_name]

    wrapped = make_wrapped_env(cfg, seed, penalty, total_steps)
    n_slices = wrapped.n_slices
    _n_prbs  = cfg.n_prbs

    obs_raw, info = wrapped.reset()
    if isinstance(obs_raw, tuple):
        obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
    obs = torch.as_tensor(
        _compute_observation(info, n_slices, _n_prbs), dtype=torch.float32
    )

    print(f"\nEvaluating {scenario_name} ({total_steps} steps, {n_slices} slices)...")
    for global_step in range(total_steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)

        act_np = action.detach().cpu().numpy()
        act_np = np.abs(act_np)
        total  = float(act_np.sum())
        if total > 0:
            act_np = act_np / total
        alloc      = act_np[:n_slices]
        alloc_prbs = np.array([int(np.floor(a * _n_prbs)) for a in alloc], dtype=int)

        result = wrapped.step(alloc_prbs)
        if len(result) == 4:
            _obs_raw, _reward, done, info = result
        else:
            _obs_raw, _reward, terminated, truncated, info = result

        obs = torch.as_tensor(
            _compute_observation(info, n_slices, _n_prbs), dtype=torch.float32
        )

        if (global_step + 1) % (steps_per_epoch * 5) == 0:
            print(f"  Step {global_step + 1}/{total_steps}")

    viol_per_slice = wrapped.violation_per_slice_history.copy()
    res_per_slice  = wrapped.resource_per_slice_history.copy()

    wrapped.save_results()
    wrapped.close()
    print(f"Completed {scenario_name} evaluation.")

    return ScenarioResult(
        name=scenario_name,
        n_slices=n_slices,
        violation_per_slice=viol_per_slice,
        resource_per_slice=res_per_slice,
    )


def _smooth(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def save_per_scenario_plots(
    result: ScenarioResult,
    steps_per_epoch: int,
    epochs: int,
    out_dir: str,
) -> None:
    """SLA-violations + resource-allocation graphs for one scenario."""
    os.makedirs(out_dir, exist_ok=True)
    total_steps = steps_per_epoch * epochs
    x = np.arange(total_steps)
    epoch_boundaries = [i * steps_per_epoch for i in range(1, epochs)]
    n_slices = result.n_slices
    labels = SLICE_LABELS[:n_slices]
    colors = SLICE_COLORS[:n_slices]

    # ── Cumulative violations + PRB allocation ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(f"PPOLag — Scenario: {result.name.upper()}", fontsize=14, fontweight="bold")

    for ax in axes:
        for b in epoch_boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25)

    ax = axes[0]
    for s_idx in range(n_slices):
        cum_viols = np.cumsum(result.violation_per_slice[:total_steps, s_idx])
        ax.plot(x, cum_viols, label=labels[s_idx], color=colors[s_idx], linewidth=1.4)
    ax.set_title("Cumulative SLA Violations per Slice")
    ax.set_ylabel("Cumulative violations")
    ax.legend(fontsize=10)

    ax = axes[1]
    for s_idx in range(n_slices):
        prbs = _smooth(result.resource_per_slice[:total_steps, s_idx].astype(float), SMOOTH_WINDOW)
        ax.plot(x, prbs, label=labels[s_idx], color=colors[s_idx], linewidth=1.4)
    ax.set_title(f"PRB Allocation per Slice (smoothed, w={SMOOTH_WINDOW})")
    ax.set_ylabel("PRBs")
    ax.legend(fontsize=10)

    fname = os.path.join(out_dir, f"{result.name}_sla_and_resources_ppolag.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(fname)}")

    # ── Violation rate ──
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_title(f"PPOLag SLA Violation Rate — {result.name.upper()}", fontsize=13)
    for s_idx in range(n_slices):
        rate = _smooth(result.violation_per_slice[:total_steps, s_idx].astype(float), SMOOTH_WINDOW)
        ax.plot(x, rate, label=labels[s_idx], color=colors[s_idx], linewidth=1.4)
    for b in epoch_boundaries:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Violation rate (rolling avg, w={SMOOTH_WINDOW})")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    fname = os.path.join(out_dir, f"{result.name}_violation_rate_ppolag.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(fname)}")


def save_summary_plot(
    results: dict[str, ScenarioResult],
    steps_per_epoch: int,
    epochs: int,
    out_dir: str,
) -> None:
    """Overall comparison across all scenarios."""
    os.makedirs(out_dir, exist_ok=True)
    total_steps = steps_per_epoch * epochs
    x = np.arange(total_steps)
    epoch_boundaries = [i * steps_per_epoch for i in range(1, epochs)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("PPOLag Evaluation — Scenario Comparison", fontsize=14, fontweight="bold")

    for ax in axes:
        for b in epoch_boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25)

    ax = axes[0]
    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        r = results[name]
        total_prbs = r.resource_per_slice[:total_steps].sum(axis=1)
        ax.plot(x, _smooth(total_prbs.astype(float), SMOOTH_WINDOW),
                label=name, color=color, linewidth=1.2)
    ax.set_title("Total PRBs Allocated per Step")
    ax.set_ylabel("PRBs")
    ax.legend(fontsize=9)

    ax = axes[1]
    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        r = results[name]
        total_viols = r.violation_per_slice[:total_steps].sum(axis=1)
        ax.plot(x, _smooth(total_viols.astype(float), SMOOTH_WINDOW),
                label=name, color=color, linewidth=1.2)
    ax.set_title("SLA Violations per Step")
    ax.set_ylabel("Violations")
    ax.legend(fontsize=9)

    ax = axes[2]
    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        r = results[name]
        total_viols = r.violation_per_slice[:total_steps].sum(axis=1)
        ax.plot(x, np.cumsum(total_viols), label=name, color=color, linewidth=1.2)
    ax.set_title("Cumulative SLA Violations")
    ax.set_ylabel("Cumulative violations")
    ax.legend(fontsize=9)

    fname = os.path.join(out_dir, "scenario_comparison_ppolag.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {os.path.abspath(fname)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPOLag checkpoint across scenarios")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PPOLag checkpoint (.pt file)"
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
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }

    print("Initializing PPOLag agent from checkpoint...")
    agent = omnisafe.Agent(algo="PPOLag", env_id=train_exp.ENV_ID, custom_cfgs=custom_cfgs)
    load_checkpoint_into_agent(agent, checkpoint_path)
    print("Agent initialized. Starting evaluation...\n")

    results: dict[str, ScenarioResult] = {}
    for name in SCENARIO_NAMES:
        results[name] = run_scenario(
            agent           = agent,
            scenario_name   = name,
            seed            = args.seed,
            epochs          = args.epochs,
            steps_per_epoch = args.steps_per_epoch,
            penalty         = args.penalty,
        )

    print("\n=== Generating per-scenario plots ===")
    for name in SCENARIO_NAMES:
        save_per_scenario_plots(
            result          = results[name],
            steps_per_epoch = args.steps_per_epoch,
            epochs          = args.epochs,
            out_dir         = args.out_dir,
        )

    save_summary_plot(
        results         = results,
        steps_per_epoch = args.steps_per_epoch,
        epochs          = args.epochs,
        out_dir         = args.out_dir,
    )

    # Summary table
    print("\n=== PPOLag Evaluation Summary ===")
    header = f"{'Scenario':12} | {'Slice':6} | {'Violations':>10} | {'Avg PRBs':>9}"
    print(header)
    print("-" * len(header))
    for name in SCENARIO_NAMES:
        r = results[name]
        labels = SLICE_LABELS[:r.n_slices]
        for s_idx, label in enumerate(labels):
            viols = r.violation_per_slice[:, s_idx].sum()
            avg_prbs = r.resource_per_slice[:, s_idx].mean()
            prefix = f"{name.upper():12}" if s_idx == 0 else f"{'':12}"
            print(f"{prefix} | {label:6} | {viols:10.0f} | {avg_prbs:9.2f}")
        total_viols = r.violation_per_slice.sum()
        total_prbs  = r.resource_per_slice.sum(axis=1).mean()
        print(f"{'':12} | {'TOTAL':6} | {total_viols:10.0f} | {total_prbs:9.2f}")
        print("-" * len(header))


if __name__ == "__main__":
    main()