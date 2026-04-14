#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified OmniSafe pipeline: train → plot → select best → evaluate.

Supports: CPO, PPO, PPOLag, SACLag, TD3Lag

Usage:
    python pipelines/pipeline.py --algo CPO --runs 30 --device cuda:0
    python pipelines/pipeline.py --algo PPOLag --runs 10 --total-steps 500000
    python pipelines/pipeline.py --algo TD3Lag --runs 5 --skip-training
    python pipelines/pipeline.py --algo SACLag --runs 1 --total-steps 10000 --device cpu
    python pipelines/pipeline.py --algo CPO --runs 30 --workers 20
"""

import os
import sys
import glob
import argparse
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipelines.configs import build_config, SUPPORTED_ALGOS, ON_POLICY, OFF_POLICY

# ── Defaults ──
NUM_RUNS    = 30
WINDOW      = 400
MAX_WORKERS = 20


def _algo_dir_name(algo):
    """Lowercase folder name for algo, e.g. PPOLag -> ppolag."""
    return algo.lower()


def _make_paths(algo):
    """Return (results_path, plots_path, runs_dir, mapping_file)."""
    base = _PROJECT_ROOT
    adir = _algo_dir_name(algo)
    results_path = os.path.join(base, "pipeline", adir, "results")
    plots_path   = os.path.join(base, "pipeline", adir, "plots")
    runs_dir     = os.path.join(base, "runs", f"{algo}-{{RanSlicePipeline-v1}}")
    mapping_file = os.path.join(results_path, "run_checkpoint_map.json")
    return results_path, plots_path, runs_dir, mapping_file


# =====================================================================
# Phase 1: Training
# =====================================================================
def _existing_omnisafe_dirs(runs_dir):
    if not os.path.isdir(runs_dir):
        return set()
    return set(os.listdir(runs_dir))


def _find_latest_checkpoint(run_dir):
    save_dir = os.path.join(run_dir, "torch_save")
    if not os.path.isdir(save_dir):
        return None
    pts = glob.glob(os.path.join(save_dir, "epoch-*.pt"))
    if not pts:
        return None
    pts.sort(key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    return pts[-1]


def _train_single_run(run_id, algo, total_steps, steps_per_epoch, device,
                       results_path, runs_dir):
    """
    Train a single run in its own process.
    Returns (run_id, checkpoint_path or None).
    """
    # Re-setup imports inside subprocess (fresh process, fresh module state)
    import os, sys, glob
    import numpy as np

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pipelines.wrappers as wrap_mod
    import omnisafe
    from pipelines.configs import build_config

    os.makedirs(results_path, exist_ok=True)

    # Patch wrappers module globals for this process
    wrap_mod._RESULTS_PATH     = results_path + "/"
    wrap_mod._TOTAL_STEPS      = total_steps
    wrap_mod.STEPS_PER_EPOCH   = steps_per_epoch
    wrap_mod._CURRENT_RUN_ID   = run_id
    wrap_mod._ENV_INSTANCE_COUNT[run_id] = 0
    wrap_mod._EPOCH_COUNTER    = 0
    wrap_mod._RNG              = np.random.default_rng(3 + run_id)

    custom_cfgs = build_config(algo, total_steps, steps_per_epoch, device)

    print(f'\n{"="*60}', flush=True)
    print(f'=== {algo} CURRICULUM Training Run {run_id} (PID {os.getpid()}) ===', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Total steps: {total_steps}, Steps/epoch: {steps_per_epoch}', flush=True)

    before = set()
    if os.path.isdir(runs_dir):
        before = set(os.listdir(runs_dir))

    agent = omnisafe.Agent(
        algo=algo,
        env_id=wrap_mod.ENV_ID,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()

    try:
        agent.plot(smooth=1)
    except Exception as e:
        print(f'[Run {run_id}] Plotting failed: {e}', flush=True)

    after = set()
    if os.path.isdir(runs_dir):
        after = set(os.listdir(runs_dir))

    new_dirs = after - before
    checkpoint_path = None
    if new_dirs:
        newest = sorted(new_dirs)[-1]
        save_dir = os.path.join(runs_dir, newest, "torch_save")
        if os.path.isdir(save_dir):
            pts = glob.glob(os.path.join(save_dir, "epoch-*.pt"))
            if pts:
                pts.sort(key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
                checkpoint_path = pts[-1]
                print(f"[Pipeline] Run {run_id} -> checkpoint: {checkpoint_path}", flush=True)

    return run_id, checkpoint_path


def run_training(algo, num_runs, total_steps, steps_per_epoch, device,
                 results_path, runs_dir, mapping_file, max_workers=MAX_WORKERS):
    """Train num_runs in parallel using ProcessPoolExecutor."""
    os.makedirs(results_path, exist_ok=True)

    checkpoint_map = {}
    effective_workers = min(max_workers, num_runs)

    print(f"[Pipeline] Launching {num_runs} runs across {effective_workers} parallel workers")

    if effective_workers <= 1:
        # Sequential fallback for single run
        rid, ckpt = _train_single_run(
            0, algo, total_steps, steps_per_epoch, device,
            results_path, runs_dir,
        )
        if ckpt:
            checkpoint_map[rid] = ckpt
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for run_id in range(num_runs):
                future = executor.submit(
                    _train_single_run,
                    run_id, algo, total_steps, steps_per_epoch, device,
                    results_path, runs_dir,
                )
                futures[future] = run_id

            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    rid, ckpt = future.result()
                    if ckpt:
                        checkpoint_map[rid] = ckpt
                    print(f"[Pipeline] Run {rid} finished successfully.", flush=True)
                except Exception as e:
                    print(f"[Pipeline] Run {run_id} FAILED: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    with open(mapping_file, "w") as f:
        json.dump({str(k): v for k, v in checkpoint_map.items()}, f, indent=2)
    print(f"\n[Pipeline] Checkpoint map saved to {mapping_file}")
    print(f"[Pipeline] {len(checkpoint_map)}/{num_runs} runs produced checkpoints.")
    return checkpoint_map


# ...existing code... (load_checkpoint_map stays the same)
def load_checkpoint_map(mapping_file):
    if not os.path.isfile(mapping_file):
        return {}
    with open(mapping_file) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ...existing code... (run_plot stays the same)
def run_plot(algo, num_runs, results_path, plots_path):
    import plot_results as pr

    runs      = range(0, num_runs)
    base_path = results_path + "/"
    save_path = plots_path + "/"
    os.makedirs(save_path, exist_ok=True)

    curriculum_available = pr.has_curriculum_data(base_path, runs)

    if curriculum_available:
        pr.plot_curriculum_comparison(base_path, runs, WINDOW, pr.PRBS, algo, save_path)
        for scenario in pr.SCENARIOS:
            data = pr.load_data(base_path, runs, pr.START, pr.END, WINDOW, scenario=scenario)
            if data is not None:
                metrics = pr.compute_metrics(data, WINDOW)
                pr.plot_results(metrics, data, algo, save_path,
                                prbs=pr.PRBS, suffix=f"_{scenario}")
    else:
        data = pr.load_data(base_path, runs, pr.START, pr.END, WINDOW)
        if data is None:
            print("[Pipeline] No data found for plotting. Skipping.")
            return
        metrics = pr.compute_metrics(data, WINDOW)
        pr.plot_results(metrics, data, algo, save_path, prbs=pr.PRBS)

    print(f"\n[Pipeline] Plots saved to {save_path}")


# ...existing code... (find_best_run stays the same)
def find_best_run(num_runs, results_path):
    best_id    = None
    best_score = float("inf")
    scores     = {}

    for run_id in range(num_runs):
        total_violations = 0
        found = False

        for scenario in ["low", "medium", "congested"]:
            path = os.path.join(results_path, f"history_{run_id}_{scenario}.npz")
            if os.path.isfile(path):
                data = np.load(path)
                total_violations += float(data["violation"].sum())
                found = True

        if not found:
            path = os.path.join(results_path, f"history_{run_id}.npz")
            if os.path.isfile(path):
                data = np.load(path)
                total_violations = float(data["violation"].sum())
                found = True

        if found:
            scores[run_id] = total_violations
            if total_violations < best_score:
                best_score = total_violations
                best_id    = run_id

    print("\n[Pipeline] === Run Convergence Ranking (by total violations) ===")
    for rid, score in sorted(scores.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if rid == best_id else ""
        print(f"  Run {rid:3d}: {score:10.0f} violations{marker}")

    if best_id is not None:
        print(f"\n[Pipeline] Best run: {best_id} with {best_score:.0f} total violations")
    else:
        print("[Pipeline] No result files found!")

    return best_id, best_score


# ...existing code... (run_evaluation stays the same)
def run_evaluation(algo, checkpoint_path, results_path, plots_path,
                   seed=3, epochs=5, steps_per_epoch=2000,
                   penalty=10.0, device="cpu"):
    """Evaluate the best checkpoint across low / medium / congested scenarios."""
    eval_dir       = os.path.join(results_path, "eval")
    eval_plots_dir = os.path.join(plots_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)

    import pipelines.wrappers as wrap_mod
    import omnisafe
    import torch
    from config_loader import load_scenarios
    from scenario_creator import create_env_from_config
    from wrapper import ReportWrapper

    SCENARIO_NAMES  = ["low", "medium", "congested"]
    SLICE_LABELS    = ["eMBB", "mMTC", "URLLC"]
    SLICE_COLORS    = ["#2196F3", "#4CAF50", "#F44336"]
    SCENARIO_COLORS = ["#2196F3", "#FF9800", "#F44336"]
    SMOOTH_WINDOW   = 50

    total_steps = epochs * steps_per_epoch

    # Patch wrappers for eval agent init
    wrap_mod._PENALTY      = penalty
    wrap_mod._TOTAL_STEPS  = total_steps
    wrap_mod._RNG          = np.random.default_rng(seed)
    wrap_mod._RESULTS_PATH = eval_dir + "/"

    # Build a minimal config just so we can instantiate the agent for weight loading
    eval_cfgs = build_config(algo, total_steps, max(steps_per_epoch, 1), device)

    print(f"\n[Pipeline] Initializing {algo} agent from {checkpoint_path} ...")
    agent = omnisafe.Agent(algo=algo, env_id=wrap_mod.ENV_ID, custom_cfgs=eval_cfgs)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in ckpt:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(ckpt["pi"], strict=True)
    print("[Pipeline] Agent initialized. Starting evaluation ...\n")

    # ── Helpers ──
    def _compute_observation(info, n_slices, n_prbs):
        obs         = np.zeros(n_slices * 4, dtype=float)
        l1_info     = info.get("l1_info", [])
        n_prbs_list = info.get("n_prbs", [])
        idx = 0
        for s in range(n_slices):
            bler = snr = traffic = 0.0
            if s < len(l1_info):
                si = l1_info[s]
                if isinstance(si, dict):
                    for _, ri in si.items():
                        if isinstance(ri, dict):
                            bler    = max(ri.get("cbr_bler", 0.0), ri.get("vbr_bler", 0.0))
                            snr     = (ri.get("cbr_snr", 0.0) + ri.get("vbr_snr", 0.0)) / 2.0
                            traffic = (ri.get("cbr_queue", 0.0) + ri.get("vbr_queue", 0.0)) / 2.0
            bler    = np.clip(bler, 0, 1)
            snr     = np.clip(snr / 30.0, -1, 1)
            traffic = np.clip(traffic / 100000.0, 0, 1)
            alloc   = (n_prbs_list[s] / max(n_prbs, 1) if s < len(n_prbs_list) else 0.0)
            alloc   = np.clip(alloc, 0, 1)
            obs[idx]                = bler
            obs[idx + n_slices]     = snr
            obs[idx + 2 * n_slices] = traffic
            obs[idx + 3 * n_slices] = alloc
            idx += 1
        return obs

    def _smooth(data, window):
        if window <= 1 or len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="same")

    def make_wrapped_env(cfg, seed_val, pen, ts):
        path = os.path.join(eval_dir, cfg.name) + "/"
        os.makedirs(path, exist_ok=True)
        rng     = np.random.default_rng(seed_val)
        raw_env = create_env_from_config(cfg, rng, penalty=pen)
        return ReportWrapper(raw_env, steps=ts, control_steps=ts + 1,
                             env_id=f"eval_{cfg.name}", path=path, verbose=False)

    # ── Run evaluation per scenario ──
    import matplotlib.pyplot as plt
    from dataclasses import dataclass

    @dataclass
    class ScenarioResult:
        name: str
        n_slices: int
        violation_per_slice: np.ndarray
        resource_per_slice:  np.ndarray

    configs = load_scenarios("scenarios.yaml")
    results = {}

    for name in SCENARIO_NAMES:
        cfg      = configs[name]
        wrapped  = make_wrapped_env(cfg, seed, penalty, total_steps)
        n_slices = wrapped.n_slices
        _n_prbs  = cfg.n_prbs

        obs_raw, info = wrapped.reset()
        if isinstance(obs_raw, tuple):
            obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
        obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs),
                              dtype=torch.float32)

        print(f"  Evaluating {name} ({total_steps} steps, {n_slices} slices) ...")
        for step in range(total_steps):
            with torch.no_grad():
                action = agent.agent._actor_critic.actor.predict(obs)
            act_np = np.abs(action.detach().cpu().numpy())
            total  = float(act_np.sum())
            if total > 0:
                act_np = act_np / total
            alloc_prbs = np.array(
                [int(np.floor(a * _n_prbs)) for a in act_np[:n_slices]], dtype=int
            )

            result = wrapped.step(alloc_prbs)
            if len(result) == 4:
                _, _, done, info = result
            else:
                _, _, _, _, info = result
            obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs),
                                  dtype=torch.float32)

        results[name] = ScenarioResult(
            name=name, n_slices=n_slices,
            violation_per_slice=wrapped.violation_per_slice_history.copy(),
            resource_per_slice=wrapped.resource_per_slice_history.copy(),
        )
        wrapped.save_results()
        wrapped.close()
        print(f"  Completed {name}.")

    # ── Per-scenario plots ──
    print("\n[Pipeline] Generating evaluation plots ...")
    for name in SCENARIO_NAMES:
        r      = results[name]
        x      = np.arange(total_steps)
        n_s    = r.n_slices
        labels = SLICE_LABELS[:n_s]
        colors = SLICE_COLORS[:n_s]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
        fig.suptitle(f"{algo} Eval — {name.upper()}", fontsize=14, fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Step"); ax.grid(alpha=0.25)

        for s in range(n_s):
            axes[0].plot(x, np.cumsum(r.violation_per_slice[:total_steps, s]),
                         label=labels[s], color=colors[s], linewidth=1.4)
        axes[0].set_title("Cumulative SLA Violations per Slice")
        axes[0].set_ylabel("Cumulative violations"); axes[0].legend()

        for s in range(n_s):
            axes[1].plot(x, _smooth(r.resource_per_slice[:total_steps, s].astype(float),
                                    SMOOTH_WINDOW),
                         label=labels[s], color=colors[s], linewidth=1.4)
        axes[1].set_title(f"PRB Allocation per Slice (smoothed, w={SMOOTH_WINDOW})")
        axes[1].set_ylabel("PRBs"); axes[1].legend()

        fname = os.path.join(eval_plots_dir, f"{name}_sla_and_resources.png")
        fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  Saved: {fname}")

    # ── Summary comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(f"{algo} Eval — Scenario Comparison", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Step"); ax.grid(alpha=0.25)

    x = np.arange(total_steps)
    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        r = results[name]
        total_prbs  = r.resource_per_slice[:total_steps].sum(axis=1)
        total_viols = r.violation_per_slice[:total_steps].sum(axis=1)
        axes[0].plot(x, _smooth(total_prbs.astype(float), SMOOTH_WINDOW),
                     label=name, color=color)
        axes[1].plot(x, _smooth(total_viols.astype(float), SMOOTH_WINDOW),
                     label=name, color=color)
        axes[2].plot(x, np.cumsum(total_viols), label=name, color=color)
    axes[0].set_title("Total PRBs Allocated"); axes[0].set_ylabel("PRBs")
    axes[0].legend()
    axes[1].set_title("SLA Violations per Step"); axes[1].set_ylabel("Violations")
    axes[1].legend()
    axes[2].set_title("Cumulative SLA Violations"); axes[2].set_ylabel("Cumulative")
    axes[2].legend()

    fname = os.path.join(eval_plots_dir, "scenario_comparison.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Summary table ──
    print(f"\n[Pipeline] === {algo} Evaluation Summary ===")
    header = f"{'Scenario':12} | {'Slice':6} | {'Violations':>10} | {'Avg PRBs':>9}"
    print(header); print("-" * len(header))
    for name in SCENARIO_NAMES:
        r      = results[name]
        labels = SLICE_LABELS[:r.n_slices]
        for s, label in enumerate(labels):
            viols    = r.violation_per_slice[:, s].sum()
            avg_prbs = r.resource_per_slice[:, s].mean()
            prefix   = f"{name.upper():12}" if s == 0 else f"{'':12}"
            print(f"{prefix} | {label:6} | {viols:10.0f} | {avg_prbs:9.2f}")
        tv = r.violation_per_slice.sum()
        tp = r.resource_per_slice.sum(axis=1).mean()
        print(f"{'':12} | {'TOTAL':6} | {tv:10.0f} | {tp:9.2f}")
        print("-" * len(header))


# ...existing code... (find_checkpoint_for_run stays the same)
def find_checkpoint_for_run(run_id, checkpoint_map, runs_dir):
    if run_id in checkpoint_map:
        path = checkpoint_map[run_id]
        if os.path.isfile(path):
            return path

    if not os.path.isdir(runs_dir):
        return None
    all_ckpts = []
    for d in os.listdir(runs_dir):
        ckpt = _find_latest_checkpoint(os.path.join(runs_dir, d))
        if ckpt:
            all_ckpts.append(ckpt)
    if not all_ckpts:
        return None
    all_ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f"[Pipeline] WARNING: Could not map run {run_id} to a specific checkpoint.")
    print(f"[Pipeline] Using most recent checkpoint: {all_ckpts[0]}")
    return all_ckpts[0]


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified OmniSafe pipeline: train -> plot -> evaluate best"
    )
    parser.add_argument("--algo", type=str, required=True,
                        choices=SUPPORTED_ALGOS,
                        help=f"Algorithm: {', '.join(SUPPORTED_ALGOS)}")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help="Number of training runs (default: 30)")
    parser.add_argument("--total-steps", type=int, default=2000000,
                        help="Total training steps (default: 2000000)")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Steps per epoch (default: 8000 on-policy, 2000 off-policy)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only plot and evaluate")
    parser.add_argument("--eval-epochs", type=int, default=5,
                        help="Evaluation epochs per scenario")
    parser.add_argument("--eval-steps-per-epoch", type=int, default=2000,
                        help="Steps per evaluation epoch")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training/evaluation")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Max parallel training workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    algo = args.algo
    results_path, plots_path, runs_dir, mapping_file = _make_paths(algo)

    # Default steps_per_epoch by algo family
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    elif algo in ON_POLICY:
        steps_per_epoch = 8000
    else:
        steps_per_epoch = 2000

    # ── Phase 1: Train ──
    if not args.skip_training:
        print("=" * 60)
        print(f"  PHASE 1: Training {args.runs} {algo} runs ({args.workers} workers)")
        print("=" * 60)
        checkpoint_map = run_training(
            algo, args.runs, args.total_steps, steps_per_epoch,
            args.device, results_path, runs_dir, mapping_file,
            max_workers=args.workers,
        )
    else:
        print(f"[Pipeline] Skipping training (--skip-training)")
        checkpoint_map = load_checkpoint_map(mapping_file)

    # ── Phase 2: Plot ──
    print("\n" + "=" * 60)
    print("  PHASE 2: Plotting results")
    print("=" * 60)
    run_plot(algo, args.runs, results_path, plots_path)

    # ── Phase 3: Find best run ──
    print("\n" + "=" * 60)
    print("  PHASE 3: Selecting best run")
    print("=" * 60)
    best_run_id, best_score = find_best_run(args.runs, results_path)
    if best_run_id is None:
        print("[Pipeline] ERROR: No result files found. Cannot evaluate.")
        sys.exit(1)

    # ── Phase 4: Evaluate ──
    checkpoint_path = find_checkpoint_for_run(best_run_id, checkpoint_map, runs_dir)
    if checkpoint_path is None:
        print(f"[Pipeline] ERROR: No checkpoint found for best run {best_run_id}.")
        print("[Pipeline] Make sure OmniSafe saved checkpoints in runs/ directory.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  PHASE 4: Evaluating best run {best_run_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)
    run_evaluation(
        algo=algo,
        checkpoint_path=checkpoint_path,
        results_path=results_path,
        plots_path=plots_path,
        epochs=args.eval_epochs,
        steps_per_epoch=args.eval_steps_per_epoch,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Algorithm        : {algo}")
    print(f"  Training results : {results_path}")
    print(f"  Plots            : {plots_path}")
    print(f"  Eval results     : {os.path.join(results_path, 'eval')}")
    print(f"  Eval plots       : {os.path.join(plots_path, 'eval')}")
    print(f"  Best run         : {best_run_id} ({best_score:.0f} violations)")


if __name__ == "__main__":
    main()