#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_forecast_comparison.py

Compare CPO with perfect forecast vs CPO without forecast (baseline).

Loads results from:
    results/500k/CPO_forecast/history_{run_id}.npz
    results/500k/CPO_no_forecast/history_{run_id}.npz

Generates plots:
    - SLA violations comparison (cumulative + moving average)
    - PRB allocation efficiency comparison
    - Per-slice performance breakdown
    - Learning curve comparison (reward)

Usage:
    python evaluate_forecast_comparison.py
    python evaluate_forecast_comparison.py --runs 30
    python evaluate_forecast_comparison.py --output figures/forecast_comparison/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──
WINDOW = 400                    # Moving average window
SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC']
COLORS_FORECAST    = ['#2196F3', '#1565C0', '#0D47A1']  # Blue family
COLORS_NO_FORECAST = ['#FF9800', '#E65100', '#BF360C']  # Orange family


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def mean_ci(data, confidence=0.90):
    """Compute mean and confidence interval across runs."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n = data.shape[0]
    # z for 90% CI ≈ 1.645; for 95% ≈ 1.96
    z = 1.645 if confidence == 0.90 else 1.96
    ci = z * std / np.sqrt(n)
    return mean, ci


def load_history(base_path, runs):
    """Load .npz history files from all runs."""
    violations_list = []
    rewards_list = []
    resources_list = []
    violations_per_slice_list = []
    resources_per_slice_list = []

    loaded = 0
    for run_id in runs:
        path = os.path.join(base_path, f'history_{run_id}.npz')
        if not os.path.exists(path):
            continue
        data = np.load(path)
        violations_list.append(data['violation'])
        rewards_list.append(data['reward'])
        resources_list.append(data['resources'])
        if 'violation_per_slice' in data:
            violations_per_slice_list.append(data['violation_per_slice'])
        if 'resource_per_slice' in data:
            resources_per_slice_list.append(data['resource_per_slice'])
        loaded += 1

    if loaded == 0:
        return None

    result = {
        'violations': np.array(violations_list),
        'rewards': np.array(rewards_list),
        'resources': np.array(resources_list),
        'n_runs': loaded,
    }
    if violations_per_slice_list:
        result['violations_per_slice'] = np.array(violations_per_slice_list)
    if resources_per_slice_list:
        result['resources_per_slice'] = np.array(resources_per_slice_list)
    return result


def plot_comparison(forecast_data, baseline_data, output_dir, window=WINDOW):
    """Generate all comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    n_steps = min(forecast_data['violations'].shape[1],
                  baseline_data['violations'].shape[1])

    # ── 1. Cumulative SLA Violations ──
    fig, ax = plt.subplots(figsize=(12, 6))

    fc_cumviol = np.cumsum(forecast_data['violations'][:, :n_steps], axis=1)
    bl_cumviol = np.cumsum(baseline_data['violations'][:, :n_steps], axis=1)

    fc_mean, fc_ci = mean_ci(fc_cumviol)
    bl_mean, bl_ci = mean_ci(bl_cumviol)

    x = np.arange(n_steps)
    ax.plot(x, fc_mean, label=f"CPO + Forecast (n={forecast_data['n_runs']})",
            color='#2196F3', linewidth=2)
    ax.fill_between(x, fc_mean - fc_ci, fc_mean + fc_ci, alpha=0.2, color='#2196F3')

    ax.plot(x, bl_mean, label=f"CPO Baseline (n={baseline_data['n_runs']})",
            color='#FF9800', linewidth=2)
    ax.fill_between(x, bl_mean - bl_ci, bl_mean + bl_ci, alpha=0.2, color='#FF9800')

    ax.set_xlabel('Steps')
    ax.set_ylabel('Cumulative SLA Violations')
    ax.set_title('Cumulative SLA Violations: Forecast vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_violations.png'), dpi=150)
    plt.close(fig)

    # ── 2. Moving Average Violation Rate ──
    fig, ax = plt.subplots(figsize=(12, 6))

    fc_viol_mean = np.mean(forecast_data['violations'][:, :n_steps], axis=0)
    bl_viol_mean = np.mean(baseline_data['violations'][:, :n_steps], axis=0)

    fc_ma = movingaverage(fc_viol_mean, window)
    bl_ma = movingaverage(bl_viol_mean, window)

    x_ma = np.arange(len(fc_ma))
    ax.plot(x_ma, fc_ma, label='CPO + Forecast', color='#2196F3', linewidth=1.5)
    ax.plot(x_ma[:len(bl_ma)], bl_ma, label='CPO Baseline', color='#FF9800', linewidth=1.5)

    ax.set_xlabel('Steps')
    ax.set_ylabel(f'Violation Rate (MA, window={window})')
    ax.set_title('SLA Violation Rate: Forecast vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'violation_rate.png'), dpi=150)
    plt.close(fig)

    # ── 3. Moving Average Reward ──
    fig, ax = plt.subplots(figsize=(12, 6))

    fc_rew_mean = np.mean(forecast_data['rewards'][:, :n_steps], axis=0)
    bl_rew_mean = np.mean(baseline_data['rewards'][:, :n_steps], axis=0)

    fc_rew_ma = movingaverage(fc_rew_mean, window)
    bl_rew_ma = movingaverage(bl_rew_mean, window)

    ax.plot(np.arange(len(fc_rew_ma)), fc_rew_ma, label='CPO + Forecast',
            color='#2196F3', linewidth=1.5)
    ax.plot(np.arange(len(bl_rew_ma)), bl_rew_ma, label='CPO Baseline',
            color='#FF9800', linewidth=1.5)

    ax.set_xlabel('Steps')
    ax.set_ylabel(f'Reward (MA, window={window})')
    ax.set_title('Learning Curve: Forecast vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=150)
    plt.close(fig)

    # ── 4. PRB Allocation Efficiency ──
    fig, ax = plt.subplots(figsize=(12, 6))

    fc_res_mean = np.mean(forecast_data['resources'][:, :n_steps], axis=0)
    bl_res_mean = np.mean(baseline_data['resources'][:, :n_steps], axis=0)

    fc_res_ma = movingaverage(fc_res_mean, window)
    bl_res_ma = movingaverage(bl_res_mean, window)

    ax.plot(np.arange(len(fc_res_ma)), fc_res_ma, label='CPO + Forecast',
            color='#2196F3', linewidth=1.5)
    ax.plot(np.arange(len(bl_res_ma)), bl_res_ma, label='CPO Baseline',
            color='#FF9800', linewidth=1.5)

    ax.set_xlabel('Steps')
    ax.set_ylabel(f'Total PRBs Allocated (MA, window={window})')
    ax.set_title('Resource Allocation: Forecast vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'resource_allocation.png'), dpi=150)
    plt.close(fig)

    # ── 5. Per-Slice Cumulative Violations ──
    if 'violations_per_slice' in forecast_data and 'violations_per_slice' in baseline_data:
        n_slices = min(forecast_data['violations_per_slice'].shape[2],
                       baseline_data['violations_per_slice'].shape[2])

        fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 5), sharey=True)
        if n_slices == 1:
            axes = [axes]

        for s in range(n_slices):
            ax = axes[s]
            fc_slice = np.cumsum(forecast_data['violations_per_slice'][:, :n_steps, s], axis=1)
            bl_slice = np.cumsum(baseline_data['violations_per_slice'][:, :n_steps, s], axis=1)

            fc_m, fc_c = mean_ci(fc_slice)
            bl_m, bl_c = mean_ci(bl_slice)

            ax.plot(x, fc_m, label='Forecast', color=COLORS_FORECAST[s % 3], linewidth=1.5)
            ax.fill_between(x, fc_m - fc_c, fc_m + fc_c, alpha=0.15, color=COLORS_FORECAST[s % 3])

            ax.plot(x, bl_m, label='Baseline', color=COLORS_NO_FORECAST[s % 3], linewidth=1.5)
            ax.fill_between(x, bl_m - bl_c, bl_m + bl_c, alpha=0.15, color=COLORS_NO_FORECAST[s % 3])

            slice_name = SLICE_NAMES[s] if s < len(SLICE_NAMES) else f'Slice {s}'
            ax.set_title(f'{slice_name}')
            ax.set_xlabel('Steps')
            if s == 0:
                ax.set_ylabel('Cumulative Violations')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Per-Slice Cumulative Violations: Forecast vs Baseline', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'per_slice_violations.png'), dpi=150)
        plt.close(fig)

    # ── 6. Summary Statistics ──
    print("\n" + "=" * 60)
    print("FORECAST COMPARISON SUMMARY")
    print("=" * 60)

    fc_total_viol = forecast_data['violations'][:, :n_steps].sum(axis=1)
    bl_total_viol = baseline_data['violations'][:, :n_steps].sum(axis=1)

    print(f"\nTotal SLA Violations (over {n_steps} steps):")
    print(f"  Forecast:  {fc_total_viol.mean():.1f} ± {fc_total_viol.std():.1f}")
    print(f"  Baseline:  {bl_total_viol.mean():.1f} ± {bl_total_viol.std():.1f}")
    improvement = (bl_total_viol.mean() - fc_total_viol.mean()) / max(bl_total_viol.mean(), 1) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    fc_total_rew = forecast_data['rewards'][:, :n_steps].sum(axis=1)
    bl_total_rew = baseline_data['rewards'][:, :n_steps].sum(axis=1)

    print(f"\nTotal Reward:")
    print(f"  Forecast:  {fc_total_rew.mean():.1f} ± {fc_total_rew.std():.1f}")
    print(f"  Baseline:  {bl_total_rew.mean():.1f} ± {bl_total_rew.std():.1f}")

    fc_avg_res = forecast_data['resources'][:, :n_steps].mean(axis=1)
    bl_avg_res = baseline_data['resources'][:, :n_steps].mean(axis=1)

    print(f"\nAvg PRBs Allocated per Step:")
    print(f"  Forecast:  {fc_avg_res.mean():.1f} ± {fc_avg_res.std():.1f}")
    print(f"  Baseline:  {bl_avg_res.mean():.1f} ± {bl_avg_res.std():.1f}")

    print(f"\nRuns loaded: Forecast={forecast_data['n_runs']}, Baseline={baseline_data['n_runs']}")
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPO with forecast vs without forecast"
    )
    parser.add_argument("--forecast-path", type=str,
                        default='./results/500k/CPO_forecast/')
    parser.add_argument("--baseline-path", type=str,
                        default='./results/500k/CPO_no_forecast/')
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=str,
                        default='./figures/forecast_comparison/')
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    WINDOW_VAL = args.window

    runs = range(args.runs)

    print(f"Loading forecast results from: {args.forecast_path}")
    forecast_data = load_history(args.forecast_path, runs)

    print(f"Loading baseline results from: {args.baseline_path}")
    baseline_data = load_history(args.baseline_path, runs)

    if forecast_data is None:
        print("ERROR: No forecast results found. Run experiments_cpo_forecast.py first.")
        return
    if baseline_data is None:
        print("ERROR: No baseline results found. Run experiments_cpo_no_forecast.py first.")
        return

    print(f"Loaded {forecast_data['n_runs']} forecast runs, "
          f"{baseline_data['n_runs']} baseline runs")

    plot_comparison(forecast_data, baseline_data, args.output, window=WINDOW_VAL)


if __name__ == '__main__':
    main()
