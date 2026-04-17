#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot evaluation results for PPOLag.

Reads eval_history_*.npz from pipelines/ppo_lag/ and generates per-scenario
and summary plots in pipelines/ppo_lag/figures/.

Usage:
    python pipelines/ppo_lag/plot_eval.py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PIPELINE_DIR   = os.path.dirname(os.path.abspath(__file__))
ALGO_NAME      = "PPOLag"
SCENARIO_NAMES = ["low", "medium", "congested"]
SLICE_LABELS   = ["eMBB", "mMTC", "URLLC"]
SLICE_COLORS   = ["#2196F3", "#4CAF50", "#F44336"]
SCENARIO_COLORS = ["#2196F3", "#FF9800", "#F44336"]
SMOOTH_WINDOW  = 50


def _smooth(data, window):
    if window <= 1 or len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="same")


def load_eval_data(base_path, scenario_names):
    results = {}
    for name in scenario_names:
        fp = os.path.join(base_path, f"history_eval_{name}.npz")
        if not os.path.exists(fp):
            print(f"Missing: {fp}")
            continue
        h = np.load(fp)
        results[name] = {
            'violation_per_slice': h['violation_per_slice'],
            'resource_per_slice':  h['resource_per_slice'],
            'n_slices': int(h['n_slices']) if 'n_slices' in h else h['violation_per_slice'].shape[1],
        }
    return results


def save_per_scenario_plots(name, data, save_path):
    os.makedirs(save_path, exist_ok=True)
    n_slices = data['n_slices']
    total_steps = data['violation_per_slice'].shape[0]
    x = np.arange(total_steps)
    labels = SLICE_LABELS[:n_slices]
    colors = SLICE_COLORS[:n_slices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(f"{ALGO_NAME} — Scenario: {name.upper()}", fontsize=14, fontweight="bold")

    for s in range(n_slices):
        axes[0].plot(x, np.cumsum(data['violation_per_slice'][:, s]),
                     label=labels[s], color=colors[s], linewidth=1.4)
    axes[0].set_title("Cumulative SLA Violations per Slice")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Cumulative violations")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    for s in range(n_slices):
        axes[1].plot(x, _smooth(data['resource_per_slice'][:, s].astype(float), SMOOTH_WINDOW),
                     label=labels[s], color=colors[s], linewidth=1.4)
    axes[1].set_title(f"PRB Allocation per Slice (smoothed, w={SMOOTH_WINDOW})")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("PRBs")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.savefig(os.path.join(save_path, f"{name}_sla_and_resources_{ALGO_NAME.lower()}.png"), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax2.set_title(f"{ALGO_NAME} SLA Violation Rate — {name.upper()}", fontsize=13)
    for s in range(n_slices):
        ax2.plot(x, _smooth(data['violation_per_slice'][:, s].astype(float), SMOOTH_WINDOW),
                 label=labels[s], color=colors[s], linewidth=1.4)
    ax2.set_xlabel("Step"); ax2.set_ylabel(f"Violation rate (rolling avg, w={SMOOTH_WINDOW})")
    ax2.legend(); ax2.grid(alpha=0.25)
    fig2.savefig(os.path.join(save_path, f"{name}_violation_rate_{ALGO_NAME.lower()}.png"), dpi=150)
    plt.close(fig2)


def save_summary_plot(results, save_path):
    os.makedirs(save_path, exist_ok=True)
    available = [n for n in SCENARIO_NAMES if n in results]
    if not available:
        return
    total_steps = results[available[0]]['violation_per_slice'].shape[0]
    x = np.arange(total_steps)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(f"{ALGO_NAME} Evaluation — Scenario Comparison", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Step"); ax.grid(alpha=0.25)

    for name, color in zip(SCENARIO_NAMES, SCENARIO_COLORS):
        if name not in results:
            continue
        r = results[name]
        total_prbs = r['resource_per_slice'].sum(axis=1)
        total_viols = r['violation_per_slice'].sum(axis=1)
        axes[0].plot(x, _smooth(total_prbs.astype(float), SMOOTH_WINDOW),
                     label=name, color=color, linewidth=1.2)
        axes[1].plot(x, _smooth(total_viols.astype(float), SMOOTH_WINDOW),
                     label=name, color=color, linewidth=1.2)
        axes[2].plot(x, np.cumsum(total_viols), label=name, color=color, linewidth=1.2)

    axes[0].set_title("Total PRBs Allocated"); axes[0].set_ylabel("PRBs"); axes[0].legend()
    axes[1].set_title("SLA Violations per Step"); axes[1].set_ylabel("Violations"); axes[1].legend()
    axes[2].set_title("Cumulative SLA Violations"); axes[2].set_ylabel("Cumulative"); axes[2].legend()

    fig.savefig(os.path.join(save_path, f"scenario_comparison_{ALGO_NAME.lower()}.png"), dpi=150)
    plt.close(fig)
    print(f"Summary plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPOLag evaluation results")
    args = parser.parse_args()

    results = load_eval_data(PIPELINE_DIR, SCENARIO_NAMES)
    if not results:
        print("No evaluation data found. Run evaluate.py first.")
        sys.exit(1)

    fig_path = os.path.join(PIPELINE_DIR, 'figures')
    for name in SCENARIO_NAMES:
        if name in results:
            save_per_scenario_plots(name, results[name], fig_path)
    save_summary_plot(results, fig_path)
