#!/usr/bin/env python3
"""Evaluate CPO No-Forecast vs Forecast checkpoints across low / medium / congested.

Runs each agent for 20k steps per scenario and produces per-scenario plots
comparing both agents side by side, plus an overall summary.

Example:
  python evaluate_cpo_comparison.py \
    --checkpoint-no-forecast runs/CPO-{RanSliceCPONoForecast-v0}/seed-000-.../torch_save/epoch-500.pt \
    --checkpoint-forecast    runs/CPO-{RanSliceCPOForecast-v0}/seed-000-.../torch_save/epoch-124.pt
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import omnisafe

import experiments_cpo_no_forecast as exp_nf
import experiments_cpo_forecast as exp_f
from config_loader import load_scenarios
from scenario_creator import create_env_from_config, create_env_from_config_forecast
from wrapper import ReportWrapper

SCENARIO_YAML  = "scenarios.yaml"
SCENARIO_NAMES = ["low", "medium", "congested"]
STEPS_PER_SCENARIO = 20000

SLICE_LABELS = ["eMBB", "mMTC", "URLLC"]
SLICE_COLORS = ["#2196F3", "#4CAF50", "#F44336"]

AGENT_LABELS = ["No Forecast", "Forecast"]
AGENT_COLORS = ["#F44336", "#2196F3"]
AGENT_STYLES = ["-", "--"]

SMOOTH_WINDOW = 50
FORECAST_HORIZON = 5


def load_checkpoint_into_agent(agent: omnisafe.Agent, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(checkpoint["pi"], strict=True)


@dataclass
class ScenarioResult:
    name: str
    agent_label: str
    n_slices: int
    violation_per_slice: np.ndarray   # (steps, n_slices)
    resource_per_slice:  np.ndarray   # (steps, n_slices)


def _compute_obs_no_forecast(info: dict, n_slices: int, n_prbs: int) -> np.ndarray:
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


def _compute_obs_forecast(info: dict, n_slices: int, n_prbs: int,
                          prb_demand: np.ndarray, trace_idx: int,
                          forecast_horizon: int) -> np.ndarray:
    base_size = n_slices * 4
    forecast_size = n_slices * forecast_horizon
    obs = np.zeros(base_size + forecast_size, dtype=float)
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
    # Append forecast
    for h in range(forecast_horizon):
        future_idx = trace_idx + 1 + h
        if future_idx < len(prb_demand):
            demand = prb_demand[future_idx]
        else:
            demand = prb_demand[-1]
        normalized = np.clip(demand / max(n_prbs, 1), 0.0, 1.0)
        start = base_size + h * n_slices
        obs[start:start + n_slices] = normalized
    return obs


def make_wrapped_env(cfg, seed: int, penalty: float, total_steps: int, tag: str):
    path = f"./results/comparison_eval/{tag}/"
    os.makedirs(path, exist_ok=True)
    rng     = np.random.default_rng(seed)
    raw_env = create_env_from_config(cfg, rng, penalty=penalty)
    return ReportWrapper(
        raw_env,
        steps=total_steps,
        control_steps=total_steps + 1,
        env_id=f"eval_{tag}",
        path=path,
        verbose=False,
    )


def run_no_forecast(agent, scenario_name, cfg, seed, total_steps, penalty) -> ScenarioResult:
    tag = f"no_forecast_{scenario_name}"
    wrapped = make_wrapped_env(cfg, seed, penalty, total_steps, tag)
    n_slices = wrapped.n_slices
    n_prbs = cfg.n_prbs

    obs_raw, info = wrapped.reset()
    if isinstance(obs_raw, tuple):
        obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
    obs = torch.as_tensor(_compute_obs_no_forecast(info, n_slices, n_prbs), dtype=torch.float32)

    print(f"  [No Forecast] {scenario_name} — {total_steps} steps ...")
    for step in range(total_steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)
        act_np = np.abs(action.detach().cpu().numpy())
        total = float(act_np.sum())
        if total > 0:
            act_np /= total
        alloc_prbs = np.array([int(np.floor(a * n_prbs)) for a in act_np[:n_slices]], dtype=int)
        result = wrapped.step(alloc_prbs)
        if len(result) == 4:
            _, _, _, info = result
        else:
            _, _, _, _, info = result
        obs = torch.as_tensor(_compute_obs_no_forecast(info, n_slices, n_prbs), dtype=torch.float32)
        if (step + 1) % 5000 == 0:
            print(f"    Step {step + 1}/{total_steps}")

    res = ScenarioResult(
        name=scenario_name, agent_label="No Forecast", n_slices=n_slices,
        violation_per_slice=wrapped.violation_per_slice_history[:total_steps].copy(),
        resource_per_slice=wrapped.resource_per_slice_history[:total_steps].copy(),
    )
    wrapped.save_results()
    wrapped.close()
    return res


def run_forecast(agent, scenario_name, cfg, seed, total_steps, penalty,
                 forecast_horizon) -> ScenarioResult:
    tag = f"forecast_{scenario_name}"
    wrapped = make_wrapped_env(cfg, seed, penalty, total_steps, tag)
    n_slices = wrapped.n_slices
    n_prbs = cfg.n_prbs

    trace_path = f"./datasets/trace_db_{scenario_name}.npz"
    prb_demand = np.load(trace_path)['prb_demand']
    trace_idx = 0

    obs_raw, info = wrapped.reset()
    if isinstance(obs_raw, tuple):
        obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
    obs = torch.as_tensor(
        _compute_obs_forecast(info, n_slices, n_prbs, prb_demand, trace_idx, forecast_horizon),
        dtype=torch.float32,
    )

    print(f"  [Forecast]    {scenario_name} — {total_steps} steps ...")
    for step in range(total_steps):
        with torch.no_grad():
            action = agent.agent._actor_critic.actor.predict(obs)
        act_np = np.abs(action.detach().cpu().numpy())
        total = float(act_np.sum())
        if total > 0:
            act_np /= total
        alloc_prbs = np.array([int(np.floor(a * n_prbs)) for a in act_np[:n_slices]], dtype=int)
        result = wrapped.step(alloc_prbs)
        if len(result) == 4:
            _, _, _, info = result
        else:
            _, _, _, _, info = result
        trace_idx += 1
        obs = torch.as_tensor(
            _compute_obs_forecast(info, n_slices, n_prbs, prb_demand, trace_idx, forecast_horizon),
            dtype=torch.float32,
        )
        if (step + 1) % 5000 == 0:
            print(f"    Step {step + 1}/{total_steps}")

    res = ScenarioResult(
        name=scenario_name, agent_label="Forecast", n_slices=n_slices,
        violation_per_slice=wrapped.violation_per_slice_history[:total_steps].copy(),
        resource_per_slice=wrapped.resource_per_slice_history[:total_steps].copy(),
    )
    wrapped.save_results()
    wrapped.close()
    return res


# ─── Plotting ─────────────────────────────────────────────────────────────

def _smooth(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def plot_scenario(nf: ScenarioResult, fc: ScenarioResult,
                  out_dir: str) -> None:
    """Generate comparison plots for a single scenario."""
    os.makedirs(out_dir, exist_ok=True)
    name = nf.name
    total_steps = nf.violation_per_slice.shape[0]
    x = np.arange(total_steps)
    n_slices = nf.n_slices
    labels = SLICE_LABELS[:n_slices]
    colors = SLICE_COLORS[:n_slices]

    # ── 1. Cumulative SLA violations per slice (NF vs F) ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f"Cumulative SLA Violations — {name.upper()}", fontsize=14, fontweight="bold")
    for ax, res, title in zip(axes, [nf, fc], AGENT_LABELS):
        for s_idx in range(n_slices):
            cum = np.cumsum(res.violation_per_slice[:, s_idx])
            ax.plot(x, cum, label=labels[s_idx], color=colors[s_idx], linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative violations")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
    fname = os.path.join(out_dir, f"{name}_cumulative_violations.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── 2. PRB allocation per slice (NF vs F) ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f"PRB Allocation per Slice — {name.upper()}", fontsize=14, fontweight="bold")
    for ax, res, title in zip(axes, [nf, fc], AGENT_LABELS):
        for s_idx in range(n_slices):
            prbs = _smooth(res.resource_per_slice[:, s_idx].astype(float), SMOOTH_WINDOW)
            ax.plot(x, prbs, label=labels[s_idx], color=colors[s_idx], linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("PRBs")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
    fname = os.path.join(out_dir, f"{name}_prb_allocation.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── 3. Violation rate overlay (both agents, per slice) ──
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 5), constrained_layout=True)
    if n_slices == 1:
        axes = [axes]
    fig.suptitle(f"SLA Violation Rate — {name.upper()}", fontsize=14, fontweight="bold")
    for s_idx, ax in enumerate(axes):
        for res, ag_label, ag_color, ag_style in zip(
            [nf, fc], AGENT_LABELS, AGENT_COLORS, AGENT_STYLES
        ):
            rate = _smooth(res.violation_per_slice[:, s_idx].astype(float), SMOOTH_WINDOW)
            ax.plot(x, rate, label=ag_label, color=ag_color, linestyle=ag_style, linewidth=1.4)
        ax.set_title(labels[s_idx])
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Violation rate (w={SMOOTH_WINDOW})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fname = os.path.join(out_dir, f"{name}_violation_rate.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── 4. Total violations comparison bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_title(f"Total SLA Violations — {name.upper()}", fontsize=13, fontweight="bold")
    bar_x = np.arange(n_slices)
    width = 0.35
    nf_totals = [nf.violation_per_slice[:, s].sum() for s in range(n_slices)]
    fc_totals = [fc.violation_per_slice[:, s].sum() for s in range(n_slices)]
    ax.bar(bar_x - width / 2, nf_totals, width, label="No Forecast", color=AGENT_COLORS[0], alpha=0.85)
    ax.bar(bar_x + width / 2, fc_totals, width, label="Forecast", color=AGENT_COLORS[1], alpha=0.85)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(labels[:n_slices])
    ax.set_ylabel("Total violations")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fname = os.path.join(out_dir, f"{name}_violation_bar.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_summary(all_nf: dict, all_fc: dict, out_dir: str) -> None:
    """Cross-scenario summary comparing both agents."""
    os.makedirs(out_dir, exist_ok=True)

    n_scenarios = len(SCENARIO_NAMES)
    n_slices = list(all_nf.values())[0].n_slices
    labels = SLICE_LABELS[:n_slices]

    # ── Total violations per scenario (grouped bar) ──
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title("Total SLA Violations — All Scenarios", fontsize=14, fontweight="bold")
    bar_x = np.arange(n_scenarios)
    width = 0.35
    nf_vals = [all_nf[s].violation_per_slice.sum() for s in SCENARIO_NAMES]
    fc_vals = [all_fc[s].violation_per_slice.sum() for s in SCENARIO_NAMES]
    ax.bar(bar_x - width / 2, nf_vals, width, label="No Forecast", color=AGENT_COLORS[0], alpha=0.85)
    ax.bar(bar_x + width / 2, fc_vals, width, label="Forecast", color=AGENT_COLORS[1], alpha=0.85)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([s.upper() for s in SCENARIO_NAMES])
    ax.set_ylabel("Total violations")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fname = os.path.join(out_dir, "summary_violations_bar.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Average PRBs per scenario ──
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title("Average Total PRBs Allocated — All Scenarios", fontsize=14, fontweight="bold")
    nf_prbs = [all_nf[s].resource_per_slice.sum(axis=1).mean() for s in SCENARIO_NAMES]
    fc_prbs = [all_fc[s].resource_per_slice.sum(axis=1).mean() for s in SCENARIO_NAMES]
    ax.bar(bar_x - width / 2, nf_prbs, width, label="No Forecast", color=AGENT_COLORS[0], alpha=0.85)
    ax.bar(bar_x + width / 2, fc_prbs, width, label="Forecast", color=AGENT_COLORS[1], alpha=0.85)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([s.upper() for s in SCENARIO_NAMES])
    ax.set_ylabel("Avg PRBs / step")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fname = os.path.join(out_dir, "summary_prbs_bar.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Per-slice violation breakdown across scenarios ──
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 5), constrained_layout=True)
    if n_slices == 1:
        axes = [axes]
    fig.suptitle("Per-Slice SLA Violations — All Scenarios", fontsize=14, fontweight="bold")
    for s_idx, ax in enumerate(axes):
        nf_v = [all_nf[s].violation_per_slice[:, s_idx].sum() for s in SCENARIO_NAMES]
        fc_v = [all_fc[s].violation_per_slice[:, s_idx].sum() for s in SCENARIO_NAMES]
        ax.bar(bar_x - width / 2, nf_v, width, label="No Forecast", color=AGENT_COLORS[0], alpha=0.85)
        ax.bar(bar_x + width / 2, fc_v, width, label="Forecast", color=AGENT_COLORS[1], alpha=0.85)
        ax.set_title(labels[s_idx])
        ax.set_xticks(bar_x)
        ax.set_xticklabels([s.upper() for s in SCENARIO_NAMES])
        ax.set_ylabel("Total violations")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fname = os.path.join(out_dir, "summary_per_slice_violations.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CPO No-Forecast vs Forecast across scenarios"
    )
    parser.add_argument(
        "--checkpoint-no-forecast", type=str, required=True,
        help="Path to no-forecast CPO checkpoint (.pt)",
    )
    parser.add_argument(
        "--checkpoint-forecast", type=str, required=True,
        help="Path to forecast CPO checkpoint (.pt)",
    )
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--steps-per-scenario", type=int,  default=STEPS_PER_SCENARIO)
    parser.add_argument("--penalty",           type=float, default=100.0)
    parser.add_argument("--forecast-horizon",  type=int,   default=FORECAST_HORIZON)
    parser.add_argument("--device",            type=str,   default="cpu")
    parser.add_argument("--out-dir",           type=str,   default="results/comparison_eval")
    args = parser.parse_args()

    ckpt_nf = os.path.abspath(args.checkpoint_no_forecast)
    ckpt_fc = os.path.abspath(args.checkpoint_forecast)
    for p in [ckpt_nf, ckpt_fc]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    total_steps = args.steps_per_scenario

    # ── Initialise no-forecast agent ──
    print("Initializing NO-FORECAST agent...")
    exp_nf._PENALTY = args.penalty
    exp_nf._TOTAL_STEPS = total_steps
    exp_nf._RNG = np.random.default_rng(args.seed)

    nf_cfgs = {
        "train_cfgs": {"total_steps": max(total_steps, 1), "device": args.device},
        "algo_cfgs": {
            "steps_per_epoch": 1000, "update_iters": 10, "batch_size": 128,
            "target_kl": 0.02, "cost_limit": 25.0, "use_cost": True,
        },
        "model_cfgs": {
            "actor":  {"hidden_sizes": [64, 64], "activation": "tanh"},
            "critic": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 0.001},
        },
        "logger_cfgs": {"use_wandb": False, "save_model_freq": 1},
    }
    agent_nf = omnisafe.Agent(algo="CPO", env_id=exp_nf.ENV_ID, custom_cfgs=nf_cfgs)
    load_checkpoint_into_agent(agent_nf, ckpt_nf)
    print("  Loaded checkpoint:", ckpt_nf)

    # ── Initialise forecast agent ──
    print("Initializing FORECAST agent...")
    exp_f._PENALTY = args.penalty
    exp_f._TOTAL_STEPS = total_steps
    exp_f._RNG = np.random.default_rng(args.seed)
    exp_f._FORECAST_HORIZON = args.forecast_horizon

    fc_cfgs = {
        "train_cfgs": {"total_steps": max(total_steps, 1), "device": args.device},
        "algo_cfgs": {
            "steps_per_epoch": 1000, "update_iters": 10, "batch_size": 128,
            "target_kl": 0.02, "cost_limit": 25.0, "use_cost": True,
        },
        "model_cfgs": {
            "actor":  {"hidden_sizes": [128, 128], "activation": "tanh"},
            "critic": {"hidden_sizes": [128, 128], "activation": "tanh", "lr": 0.001},
        },
        "logger_cfgs": {"use_wandb": False, "save_model_freq": 1},
    }
    agent_fc = omnisafe.Agent(algo="CPO", env_id=exp_f.ENV_ID, custom_cfgs=fc_cfgs)
    load_checkpoint_into_agent(agent_fc, ckpt_fc)
    print("  Loaded checkpoint:", ckpt_fc)

    # ── Evaluate on each scenario (threaded) ──
    configs = load_scenarios(SCENARIO_YAML)
    all_nf: dict[str, ScenarioResult] = {}
    all_fc: dict[str, ScenarioResult] = {}

    futures = {}
    with ThreadPoolExecutor(max_workers=len(SCENARIO_NAMES) * 2) as pool:
        for scenario_name in SCENARIO_NAMES:
            cfg = configs[scenario_name]
            print(f"Submitting {scenario_name.upper()} (NF + FC, {total_steps} steps each)")
            futures[pool.submit(
                run_no_forecast,
                agent_nf, scenario_name, cfg, args.seed, total_steps, args.penalty,
            )] = (scenario_name, "nf")
            futures[pool.submit(
                run_forecast,
                agent_fc, scenario_name, cfg, args.seed, total_steps, args.penalty,
                args.forecast_horizon,
            )] = (scenario_name, "fc")

        for future in as_completed(futures):
            scenario_name, variant = futures[future]
            result = future.result()
            if variant == "nf":
                all_nf[scenario_name] = result
            else:
                all_fc[scenario_name] = result
            print(f"  Done: {scenario_name.upper()} — {result.agent_label}")

    # ── Generate plots ──
    print(f"\n{'='*60}")
    print("  Generating plots")
    print(f"{'='*60}")

    for scenario_name in SCENARIO_NAMES:
        scenario_dir = os.path.join(args.out_dir, scenario_name)
        plot_scenario(all_nf[scenario_name], all_fc[scenario_name], scenario_dir)

    plot_summary(all_nf, all_fc, args.out_dir)

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("  Evaluation Summary")
    print(f"{'='*60}")
    header = f"{'Scenario':12} | {'Agent':14} | {'Slice':6} | {'Violations':>10} | {'Avg PRBs':>9}"
    print(header)
    print("-" * len(header))
    for scenario_name in SCENARIO_NAMES:
        for res, ag_label in [(all_nf[scenario_name], "No Forecast"),
                              (all_fc[scenario_name], "Forecast")]:
            for s_idx, label in enumerate(SLICE_LABELS[:res.n_slices]):
                viols = res.violation_per_slice[:, s_idx].sum()
                avg_prbs = res.resource_per_slice[:, s_idx].mean()
                prefix = f"{scenario_name.upper():12}" if s_idx == 0 and ag_label == "No Forecast" else f"{'':12}"
                print(f"{prefix} | {ag_label:14} | {label:6} | {viols:10.0f} | {avg_prbs:9.2f}")
            total_v = res.violation_per_slice.sum()
            total_p = res.resource_per_slice.sum(axis=1).mean()
            print(f"{'':12} | {ag_label:14} | {'TOTAL':6} | {total_v:10.0f} | {total_p:9.2f}")
        print("-" * len(header))

    print(f"\nPlots saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
