#!/usr/bin/env python3
"""Evaluate a trained PPOLag checkpoint with the exact training environment.

Example:
  ./omnivenv/bin/python evaluate_ppolag_checkpoint.py \
    --checkpoint runs/PPOLag-{RanSlicePPOLag-v0}/seed-000-2026-03-17-18-27-38/torch_save/epoch-20.pt \
    --scenario 4 --epochs 5 --steps-per-epoch 1000
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib

# Use a non-interactive backend so plots are saved on headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import omnisafe

import experiments_ppo_lag as train_exp


SLOT_MS = 0.1


def load_checkpoint_into_agent(agent: omnisafe.Agent, checkpoint_path: str) -> None:
    """Load PPOLag checkpoint weights into the internal actor module."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(checkpoint["pi"], strict=True)


def collect_raw_slice_metrics(env: train_exp.RanSliceEnv) -> dict[str, dict[str, float]]:
    """Collect exact raw metrics from each `slice_ran.info` dictionary."""
    out: dict[str, dict[str, float]] = {}
    node_b = env._env.unwrapped.node_b
    for l1_idx, l1 in enumerate(node_b.slices_l1):
        for ran_idx, ran in enumerate(l1.slices_ran):
            sid = f"l1{l1_idx}_{l1.type}_ran{ran_idx}"
            metrics = {}

            for key, value in ran.info.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

            # UE count is not always stored in ran.info for non-mMTC slices.
            if hasattr(ran, "cbr_ues") and hasattr(ran, "vbr_ues"):
                metrics["n_ues"] = float(len(ran.cbr_ues) + len(ran.vbr_ues))
            elif "devices" in metrics:
                metrics["n_ues"] = float(metrics["devices"])

            # Convenience totals based on exact raw fields.
            if "cbr_th" in metrics or "vbr_th" in metrics:
                metrics["throughput_total"] = float(metrics.get("cbr_th", 0.0) + metrics.get("vbr_th", 0.0))
            if "cbr_delay" in metrics or "vbr_delay" in metrics:
                metrics["delay_total"] = float(metrics.get("cbr_delay", 0.0) + metrics.get("vbr_delay", 0.0))

            # Unit-aware derived metrics for plotting.
            # Throughput fields in info are accumulated bits over one env step.
            if hasattr(ran, "observation_time"):
                obs_t = float(max(ran.observation_time, 1e-12))
                bits_per_sec = (metrics.get("cbr_th", 0.0) + metrics.get("vbr_th", 0.0)) / obs_t
                metrics["plot_throughput_mbps"] = bits_per_sec / 1e6

            # Queue in info is sum of per-slot average queue over the step.
            if hasattr(ran, "slots_per_step"):
                sps = float(max(ran.slots_per_step, 1.0))
                metrics["plot_queue_bits"] = (
                    metrics.get("cbr_queue", 0.0) + metrics.get("vbr_queue", 0.0)
                ) / sps

                # Delay in eMBB is accumulated per-slot mean HOL delay; URLLC uses
                # step-maximum HOL delay and should not be divided by slots_per_step.
                if l1.type == "URLLC":
                    metrics["plot_delay_slots"] = metrics.get("cbr_delay", 0.0) + metrics.get("vbr_delay", 0.0)
                elif "cbr_delay" in metrics or "vbr_delay" in metrics:
                    metrics["plot_delay_slots"] = (
                        metrics.get("cbr_delay", 0.0) + metrics.get("vbr_delay", 0.0)
                    ) / sps

                if "plot_delay_slots" in metrics:
                    metrics["plot_delay_ms"] = metrics["plot_delay_slots"] * SLOT_MS

                if l1.type == "mMTC":
                    # mMTC devices and delay are also accumulated over the step.
                    metrics["plot_mmtc_traffic_devices"] = metrics.get("devices", metrics.get("n_ues", 0.0)) / sps
                    metrics["plot_mmtc_delay_slots"] = metrics.get("delay", 0.0) / sps
                    metrics["plot_mmtc_delay_ms"] = metrics["plot_mmtc_delay_slots"] * SLOT_MS

            out[sid] = metrics
    return out


def save_all_plots(
    metric_store: dict[str, dict[str, list[float]]],
    prb_alloc_history: np.ndarray,
    steps_per_epoch: int,
    epochs: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total_steps = steps_per_epoch * epochs
    x = list(range(total_steps))
    epoch_boundaries = [i * steps_per_epoch for i in range(1, epochs)]

    embb_urllc_sids = [
        sid for sid in sorted(metric_store.keys()) if ("_eMBB_" in sid or "_URLLC_" in sid)
    ]
    mmtc_sids = [sid for sid in sorted(metric_store.keys()) if "_mMTC_" in sid]

    color_map = plt.get_cmap("tab10")
    sid_colors = {sid: color_map(i % 10) for i, sid in enumerate(embb_urllc_sids + mmtc_sids)}

    def series_or_zeros(per_metric: dict[str, list[float]], key: str) -> np.ndarray:
        vals = per_metric.get(key, None)
        if vals is None or len(vals) != total_steps:
            return np.zeros(total_steps, dtype=float)
        return np.asarray(vals, dtype=float)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()

    # eMBB/URLLC delay.
    ax = axes[0]
    for sid in embb_urllc_sids:
        y = series_or_zeros(metric_store[sid], "plot_delay_ms")
        ax.plot(x, y, label=sid, color=sid_colors[sid], linewidth=1.2)
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("eMBB/URLLC Delay per Slice")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("HOL delay [ms]")
    ax.grid(alpha=0.3)
    if embb_urllc_sids:
        ax.legend(loc="best", fontsize=8)

    # eMBB/URLLC throughput.
    ax = axes[1]
    for sid in embb_urllc_sids:
        y = series_or_zeros(metric_store[sid], "plot_throughput_mbps")
        ax.plot(x, y, label=sid, color=sid_colors[sid], linewidth=1.2)
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("eMBB/URLLC Throughput per Slice")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("Throughput [Mb/s]")
    ax.grid(alpha=0.3)
    if embb_urllc_sids:
        ax.legend(loc="best", fontsize=8)

    # eMBB/URLLC queue.
    ax = axes[2]
    for sid in embb_urllc_sids:
        y = series_or_zeros(metric_store[sid], "plot_queue_bits")
        ax.plot(x, y, label=sid, color=sid_colors[sid], linewidth=1.2)
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("eMBB/URLLC Queue per Slice")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("Queue [bits]")
    ax.grid(alpha=0.3)
    if embb_urllc_sids:
        ax.legend(loc="best", fontsize=8)

    # mMTC traffic.
    ax = axes[3]
    for sid in mmtc_sids:
        traffic = series_or_zeros(metric_store[sid], "plot_mmtc_traffic_devices")
        if np.allclose(traffic, 0):
            traffic = series_or_zeros(metric_store[sid], "n_ues")
        ax.plot(x, traffic, label=sid, color=sid_colors[sid], linewidth=1.2)
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("mMTC Traffic per Slice")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("Traffic [devices]")
    ax.grid(alpha=0.3)
    if mmtc_sids:
        ax.legend(loc="best", fontsize=8)

    # mMTC delay.
    ax = axes[4]
    for sid in mmtc_sids:
        delay = series_or_zeros(metric_store[sid], "plot_mmtc_delay_ms")
        ax.plot(x, delay, label=sid, color=sid_colors[sid], linewidth=1.2)
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("mMTC Delay per Slice")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("Delay [ms]")
    ax.grid(alpha=0.3)
    if mmtc_sids:
        ax.legend(loc="best", fontsize=8)

    # Total allocated PRBs (single line, not per-slice).
    ax = axes[5]
    total_prbs = np.asarray(prb_alloc_history, dtype=float).sum(axis=1)
    ax.plot(x, total_prbs, color="black", linewidth=1.4, label="total_allocated_prbs")
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Total Allocated PRBs")
    ax.set_xlabel("Global step [env step]")
    ax.set_ylabel("Allocated PRBs [PRB]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(os.path.join(out_dir, "evaluation_all_plots.png"), dpi=150)
    plt.close(fig)


def main() -> None:

    parser = argparse.ArgumentParser(description="Evaluate a trained PPOLag checkpoint on RanSlice")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/PPOLag-{RanSlicePPOLag-v0}/seed-000-2026-03-17-23-09-24/torch_save/epoch-20.pt",
        help="Path to OmniSafe checkpoint .pt file",
    )
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNG plots (defaults to results/scenario_<n>/PPOLag_checkpoint_eval)",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Use the exact same environment implementation and globals as training.
    train_exp._RNG = train_exp.np.random.default_rng(args.seed)
    train_exp._SCEN = args.scenario
    train_exp._PENALTY = args.penalty
    train_exp._TOTAL_STEPS = max(args.epochs * args.steps_per_epoch, 1)

    out_dir = args.out_dir or f"results/scenario_{args.scenario}/PPOLag_checkpoint_eval"
    os.makedirs(out_dir, exist_ok=True)

    custom_cfgs = {
        "train_cfgs": {
            "total_steps": max(args.epochs * args.steps_per_epoch, 1),
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

    env = train_exp.RanSliceEnv(env_id=train_exp.ENV_ID)
    obs, _ = env.reset()

    n_slices = int(env._env.n_slices)
    n_prbs = int(env._n_prbs)
    prb_alloc = np.zeros((args.epochs * args.steps_per_epoch, n_slices), dtype=int)
    global_step = 0

    first_metrics = collect_raw_slice_metrics(env)
    slice_ids = sorted(first_metrics.keys())
    metric_store: dict[str, dict[str, list[float]]] = {sid: defaultdict(list) for sid in slice_ids}

    total_reward = 0.0
    total_cost = 0.0

    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            with torch.no_grad():
                action = agent.agent._actor_critic.actor.predict(obs)

            # Mirror the exact PRB mapping used in train_exp.RanSliceEnv.step.
            act_np = action.detach().cpu().numpy()
            act_np = np.abs(act_np)
            total = float(act_np.sum())
            if total > 0:
                act_np = act_np / total
            alloc = act_np[:n_slices]
            alloc_prbs = np.array([int(np.floor(a * n_prbs)) for a in alloc], dtype=int)
            prb_alloc[global_step, :] = alloc_prbs

            obs, reward, cost, terminated, truncated, _ = env.step(action)
            total_reward += float(reward.item())
            total_cost += float(cost.item())

            step_metrics = collect_raw_slice_metrics(env)
            for sid in slice_ids:
                for metric, value in step_metrics[sid].items():
                    metric_store[sid][metric].append(float(value))

            if bool(terminated.item()) or bool(truncated.item()):
                obs, _ = env.reset()

            global_step += 1

        print(
            f"Epoch {epoch + 1}/{args.epochs} done | "
            f"cumulative_reward={total_reward:.4f} cumulative_cost={total_cost:.4f}"
        )

    save_all_plots(
        metric_store=metric_store,
        prb_alloc_history=prb_alloc,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        out_dir=out_dir,
    )

    env.close()

    print("\nEvaluation finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Epochs: {args.epochs}, Steps/Epoch: {args.steps_per_epoch}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Total cost: {total_cost:.4f}")
    print(f"Saved PNG plots to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()