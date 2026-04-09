#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete pipeline for CPO (with curriculum learning):
  1. Train 30 runs using experiments_cpo.py
  2. Plot results to pipeline/cpo/plots/
  3. Select the best run (lowest final cumulative violations)
  4. Evaluate the best run's checkpoint, save to pipeline/cpo/results/

Usage:
    python pipeline_cpo.py
    python pipeline_cpo.py --runs 30 --processes 4
    python pipeline_cpo.py --skip-training        # re-plot & re-evaluate only
"""

import os
import sys
import glob
import argparse
import json
import numpy as np
import concurrent.futures as cf

import matplotlib
matplotlib.use("Agg")

# ── Paths ──
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH   = os.path.join(BASE_DIR, "pipeline", "cpo", "results")
PLOTS_PATH     = os.path.join(BASE_DIR, "pipeline", "cpo", "plots")
RUNS_DIR       = os.path.join(BASE_DIR, "runs", "CPO-{RanSliceCPO-v1}")
MAPPING_FILE   = os.path.join(RESULTS_PATH, "run_checkpoint_map.json")

NUM_RUNS       = 30
WINDOW         = 400   # moving-average window for convergence metric


# =====================================================================
# Phase 1: Training
# =====================================================================
def _existing_omnisafe_dirs():
    """Return the set of OmniSafe run directory names that currently exist."""
    if not os.path.isdir(RUNS_DIR):
        return set()
    return set(os.listdir(RUNS_DIR))


def _find_latest_checkpoint(run_dir):
    """Return the path to the highest-epoch checkpoint in a run dir."""
    save_dir = os.path.join(run_dir, "torch_save")
    if not os.path.isdir(save_dir):
        return None
    pts = glob.glob(os.path.join(save_dir, "epoch-*.pt"))
    if not pts:
        return None
    pts.sort(key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    return pts[-1]


def run_training(num_runs, processes):
    """Train all runs and return {run_id: checkpoint_path} mapping."""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Import training module and redirect results to pipeline/cpo/results/
    import experiments_cpo as train_mod
    train_mod._RESULTS_PATH = RESULTS_PATH + "/"

    trainer = train_mod.TrainerCPO()
    checkpoint_map = {}

    if processes <= 1:
        for run_id in range(num_runs):
            before = _existing_omnisafe_dirs()
            trainer.train(run_id)
            after = _existing_omnisafe_dirs()

            new_dirs = after - before
            if new_dirs:
                newest = sorted(new_dirs)[-1]
                ckpt = _find_latest_checkpoint(os.path.join(RUNS_DIR, newest))
                if ckpt:
                    checkpoint_map[run_id] = ckpt
                    print(f"[Pipeline] Run {run_id} -> checkpoint: {ckpt}")
    else:
        before = _existing_omnisafe_dirs()
        with cf.ProcessPoolExecutor(processes) as pool:
            list(pool.map(trainer.train, range(num_runs)))
        after = _existing_omnisafe_dirs()

        new_dirs = sorted(after - before)
        for i, d in enumerate(new_dirs):
            if i >= num_runs:
                break
            ckpt = _find_latest_checkpoint(os.path.join(RUNS_DIR, d))
            if ckpt:
                checkpoint_map[i] = ckpt

    with open(MAPPING_FILE, "w") as f:
        json.dump({str(k): v for k, v in checkpoint_map.items()}, f, indent=2)
    print(f"\n[Pipeline] Checkpoint map saved to {MAPPING_FILE}")

    return checkpoint_map


def load_checkpoint_map():
    """Load a previously-saved checkpoint map."""
    if not os.path.isfile(MAPPING_FILE):
        return {}
    with open(MAPPING_FILE) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# =====================================================================
# Phase 2: Plot results
# =====================================================================
def run_plot(num_runs):
    """Call plot_results with the pipeline results directory."""
    import plot_results as pr

    runs = range(0, num_runs)
    base_path = RESULTS_PATH + "/"
    algorithm_name = "CPO"
    save_path = PLOTS_PATH + "/"
    os.makedirs(save_path, exist_ok=True)

    curriculum_available = pr.has_curriculum_data(base_path, runs)

    if curriculum_available:
        pr.plot_curriculum_comparison(base_path, runs, WINDOW, pr.PRBS, algorithm_name, save_path)
        for scenario in pr.SCENARIOS:
            data = pr.load_data(base_path, runs, pr.START, pr.END, WINDOW, scenario=scenario)
            if data is not None:
                metrics = pr.compute_metrics(data, WINDOW)
                pr.plot_results(metrics, data, algorithm_name, save_path, prbs=pr.PRBS, suffix=f"_{scenario}")
    else:
        data = pr.load_data(base_path, runs, pr.START, pr.END, WINDOW)
        if data is None:
            print("[Pipeline] No data found for plotting. Skipping.")
            return
        metrics = pr.compute_metrics(data, WINDOW)
        pr.plot_results(metrics, data, algorithm_name, save_path, prbs=pr.PRBS)

    print(f"\n[Pipeline] Plots saved to {save_path}")


# =====================================================================
# Phase 3: Select best run (lowest final cumulative violations)
# =====================================================================
def find_best_run(num_runs):
    """
    Determine the best run by convergence:
      - Load each history_{run_id}*.npz
      - Compute total cumulative violations at the end
      - The run with the LOWEST final cumulative violations wins
    Returns (best_run_id, best_score).
    """
    best_id = None
    best_score = float("inf")
    scores = {}

    for run_id in range(num_runs):
        total_violations = 0
        found = False

        # Try curriculum files first, fall back to single file
        for scenario in ["low", "medium", "congested"]:
            path = os.path.join(RESULTS_PATH, f"history_{run_id}_{scenario}.npz")
            if os.path.isfile(path):
                data = np.load(path)
                total_violations += float(data["violation"].sum())
                found = True

        if not found:
            path = os.path.join(RESULTS_PATH, f"history_{run_id}.npz")
            if os.path.isfile(path):
                data = np.load(path)
                total_violations = float(data["violation"].sum())
                found = True

        if found:
            scores[run_id] = total_violations
            if total_violations < best_score:
                best_score = total_violations
                best_id = run_id

    print("\n[Pipeline] === Run Convergence Ranking (by total violations) ===")
    for rid, score in sorted(scores.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if rid == best_id else ""
        print(f"  Run {rid:3d}: {score:10.0f} violations{marker}")

    if best_id is not None:
        print(f"\n[Pipeline] Best run: {best_id} with {best_score:.0f} total violations")
    else:
        print("[Pipeline] No result files found!")

    return best_id, best_score


# =====================================================================
# Phase 4: Evaluate best checkpoint
# =====================================================================
def run_evaluation(checkpoint_path, seed=3, epochs=5,
                   steps_per_epoch=1000, penalty=100.0, device="cpu"):
    """Evaluate the best checkpoint across low / medium / congested scenarios."""
    eval_dir = os.path.join(RESULTS_PATH, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_plots_dir = os.path.join(PLOTS_PATH, "eval")
    os.makedirs(eval_plots_dir, exist_ok=True)

    import experiments_cpo as train_exp
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

    # Patch module-level globals used by the env
    train_exp._PENALTY     = penalty
    train_exp._TOTAL_STEPS = total_steps
    train_exp._RNG         = np.random.default_rng(seed)
    train_exp._RESULTS_PATH = eval_dir + "/"

    custom_cfgs = {
        "train_cfgs": {
            "total_steps": max(total_steps, 1),
            "device": device,
        },
        "algo_cfgs": {
            "steps_per_epoch": max(steps_per_epoch, 1),
            "update_iters": 10,
            "batch_size": 128,
            "target_kl": 0.02,
            "cost_limit": 25.0,
            "use_cost": True,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }

    print(f"\n[Pipeline] Initializing CPO agent from {checkpoint_path} ...")
    agent = omnisafe.Agent(algo="CPO", env_id=train_exp.ENV_ID, custom_cfgs=custom_cfgs)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in ckpt:
        raise KeyError("Checkpoint does not contain 'pi' policy weights.")
    agent.agent._actor_critic.actor.load_state_dict(ckpt["pi"], strict=True)
    print("[Pipeline] Agent initialized. Starting evaluation ...\n")

    # --- helpers ---
    def _compute_observation(info, n_slices, n_prbs):
        obs = np.zeros(n_slices * 4, dtype=float)
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

    # --- Run evaluation per scenario ---
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
        cfg     = configs[name]
        wrapped = make_wrapped_env(cfg, seed, penalty, total_steps)
        n_slices = wrapped.n_slices
        _n_prbs  = cfg.n_prbs

        obs_raw, info = wrapped.reset()
        if isinstance(obs_raw, tuple):
            obs_raw, info = obs_raw[0], obs_raw[1] if len(obs_raw) > 1 else {}
        obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs), dtype=torch.float32)

        print(f"  Evaluating {name} ({total_steps} steps, {n_slices} slices) ...")
        for step in range(total_steps):
            with torch.no_grad():
                action = agent.agent._actor_critic.actor.predict(obs)
            act_np = np.abs(action.detach().cpu().numpy())
            total  = float(act_np.sum())
            if total > 0:
                act_np = act_np / total
            alloc_prbs = np.array([int(np.floor(a * _n_prbs)) for a in act_np[:n_slices]], dtype=int)

            result = wrapped.step(alloc_prbs)
            if len(result) == 4:
                _, _, done, info = result
            else:
                _, _, _, _, info = result
            obs = torch.as_tensor(_compute_observation(info, n_slices, _n_prbs), dtype=torch.float32)

        results[name] = ScenarioResult(
            name=name, n_slices=n_slices,
            violation_per_slice=wrapped.violation_per_slice_history.copy(),
            resource_per_slice=wrapped.resource_per_slice_history.copy(),
        )
        wrapped.save_results()
        wrapped.close()
        print(f"  Completed {name}.")

    # --- Per-scenario plots ---
    print("\n[Pipeline] Generating evaluation plots ...")
    for name in SCENARIO_NAMES:
        r   = results[name]
        x   = np.arange(total_steps)
        n_s = r.n_slices
        labels = SLICE_LABELS[:n_s]
        colors = SLICE_COLORS[:n_s]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
        fig.suptitle(f"CPO Eval — {name.upper()}", fontsize=14, fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Step"); ax.grid(alpha=0.25)

        for s in range(n_s):
            axes[0].plot(x, np.cumsum(r.violation_per_slice[:total_steps, s]),
                         label=labels[s], color=colors[s], linewidth=1.4)
        axes[0].set_title("Cumulative SLA Violations per Slice")
        axes[0].set_ylabel("Cumulative violations"); axes[0].legend()

        for s in range(n_s):
            axes[1].plot(x, _smooth(r.resource_per_slice[:total_steps, s].astype(float), SMOOTH_WINDOW),
                         label=labels[s], color=colors[s], linewidth=1.4)
        axes[1].set_title(f"PRB Allocation per Slice (smoothed, w={SMOOTH_WINDOW})")
        axes[1].set_ylabel("PRBs"); axes[1].legend()

        fname = os.path.join(eval_plots_dir, f"{name}_sla_and_resources.png")
        fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  Saved: {fname}")

    # --- Summary comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("CPO Eval — Scenario Comparison", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Step"); ax.grid(alpha=0.25)

    x = np.arange(total_steps)
    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        r = results[name]
        total_prbs  = r.resource_per_slice[:total_steps].sum(axis=1)
        total_viols = r.violation_per_slice[:total_steps].sum(axis=1)
        axes[0].plot(x, _smooth(total_prbs.astype(float), SMOOTH_WINDOW), label=name, color=color)
        axes[1].plot(x, _smooth(total_viols.astype(float), SMOOTH_WINDOW), label=name, color=color)
        axes[2].plot(x, np.cumsum(total_viols), label=name, color=color)
    axes[0].set_title("Total PRBs Allocated"); axes[0].set_ylabel("PRBs"); axes[0].legend()
    axes[1].set_title("SLA Violations per Step"); axes[1].set_ylabel("Violations"); axes[1].legend()
    axes[2].set_title("Cumulative SLA Violations"); axes[2].set_ylabel("Cumulative"); axes[2].legend()

    fname = os.path.join(eval_plots_dir, "scenario_comparison.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Saved: {fname}")

    # --- Summary table ---
    print("\n[Pipeline] === Evaluation Summary ===")
    header = f"{'Scenario':12} | {'Slice':6} | {'Violations':>10} | {'Avg PRBs':>9}"
    print(header); print("-" * len(header))
    for name in SCENARIO_NAMES:
        r = results[name]
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


# =====================================================================
# Fallback checkpoint finder
# =====================================================================
def find_checkpoint_for_run(run_id, checkpoint_map):
    """Try the map first; fall back to scanning OmniSafe dirs."""
    if run_id in checkpoint_map:
        path = checkpoint_map[run_id]
        if os.path.isfile(path):
            return path

    if not os.path.isdir(RUNS_DIR):
        return None
    all_ckpts = []
    for d in os.listdir(RUNS_DIR):
        ckpt = _find_latest_checkpoint(os.path.join(RUNS_DIR, d))
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
        description="Full CPO pipeline: train -> plot -> evaluate best"
    )
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help="Number of training runs (default: 30)")
    parser.add_argument("--processes", type=int, default=1,
                        help="Parallel workers for training (1 = sequential, recommended)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only plot and evaluate")
    parser.add_argument("--eval-epochs", type=int, default=5,
                        help="Evaluation epochs per scenario")
    parser.add_argument("--eval-steps-per-epoch", type=int, default=2000,
                        help="Steps per evaluation epoch")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training/evaluation")
    args = parser.parse_args()

    # ── Phase 1: Train ──
    if not args.skip_training:
        print("=" * 60)
        print("  PHASE 1: Training 30 CPO runs")
        print("=" * 60)
        checkpoint_map = run_training(args.runs, args.processes)
    else:
        print("[Pipeline] Skipping training (--skip-training)")
        checkpoint_map = load_checkpoint_map()

    # ── Phase 2: Plot ──
    print("\n" + "=" * 60)
    print("  PHASE 2: Plotting results")
    print("=" * 60)
    run_plot(args.runs)

    # ── Phase 3: Find best run ──
    print("\n" + "=" * 60)
    print("  PHASE 3: Selecting best run")
    print("=" * 60)
    best_run_id, best_score = find_best_run(args.runs)
    if best_run_id is None:
        print("[Pipeline] ERROR: No result files found. Cannot evaluate.")
        sys.exit(1)

    # ── Phase 4: Evaluate ──
    checkpoint_path = find_checkpoint_for_run(best_run_id, checkpoint_map)
    if checkpoint_path is None:
        print(f"[Pipeline] ERROR: No checkpoint found for best run {best_run_id}.")
        print("[Pipeline] Make sure OmniSafe saved checkpoints in runs/ directory.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  PHASE 4: Evaluating best run {best_run_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)
    run_evaluation(
        checkpoint_path=checkpoint_path,
        epochs=args.eval_epochs,
        steps_per_epoch=args.eval_steps_per_epoch,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Training results : {RESULTS_PATH}")
    print(f"  Plots            : {PLOTS_PATH}")
    print(f"  Eval results     : {os.path.join(RESULTS_PATH, 'eval')}")
    print(f"  Eval plots       : {os.path.join(PLOTS_PATH, 'eval')}")
    print(f"  Best run         : {best_run_id} ({best_score:.0f} violations)")


if __name__ == "__main__":
    main()
