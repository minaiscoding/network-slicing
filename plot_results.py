#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot comprehensive results from TD3 curriculum learning runs.

Supports:
- Per-scenario files: history_{run_id}_{scenario}.npz (curriculum learning)
- Single files: history_{run_id}.npz (legacy)

Plots:
- SLA violations (total, moving average)
- SLA violations per slice
- Cumulative SLA violations (total and per slice)
- Resource allocation (direct and moving average)
- Resource allocation per slice
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --- Config ---
START = 0
END = 20000
WINDOW = 400  # moving average
RUNS = range(0, 22)  # 0 to 21 inclusive
BASE_PATH = './results/scenario_comparison/TD3/'
PRBS = 150  # adjust if different per run
SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC', 'Slice4', 'Slice5']  # Default slice names
SCENARIOS = ['low', 'medium', 'congested']  # Curriculum learning scenarios


def has_curriculum_data(base_path, runs):
    """Return True if at least one curriculum file exists for configured scenarios."""
    for run_id in runs:
        for scenario in SCENARIOS:
            file_path = os.path.join(base_path, f'history_{run_id}_{scenario}.npz')
            if os.path.exists(file_path):
                return True
    return False

def movingaverage(values, window):
    """Compute moving average with convolution."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def mean_ci(data):
    """Compute mean and 90% confidence interval."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n = data.shape[0]
    ci = 1.697 * std / np.sqrt(n)  # 90% confidence
    return mean, ci


def load_data(base_path, runs, start, end, window, scenario=None):
    """
    Load data from all runs and compute statistics.
    
    Args:
        scenario: If specified, loads history_{run_id}_{scenario}.npz
                  Otherwise loads history_{run_id}.npz
    
    Returns dict with raw and processed data.
    """
    violations_list = []
    resources_list = []
    violations_per_slice_list = []
    resources_per_slice_list = []
    n_slices = None
    
    for run_id in runs:
        if scenario:
            file_path = os.path.join(base_path, f'history_{run_id}_{scenario}.npz')
        else:
            file_path = os.path.join(base_path, f'history_{run_id}.npz')
            
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        print(f"Loading {file_path}")
        histories = np.load(file_path)
        
        # Get number of slices from data if available
        if 'n_slices' in histories:
            n_slices = int(histories['n_slices'])
        
        # Total violations and resources
        _violations = histories['violation'][start:end]
        _resources = histories['resources'][start:end]
        violations_list.append(_violations)
        resources_list.append(_resources)
        
        # Per-slice data (if available)
        if 'violation_per_slice' in histories:
            _viol_per_slice = histories['violation_per_slice'][start:end]
            violations_per_slice_list.append(_viol_per_slice)
        
        if 'resource_per_slice' in histories:
            _res_per_slice = histories['resource_per_slice'][start:end]
            resources_per_slice_list.append(_res_per_slice)
    
    if not violations_list:
        return None
    
    # Determine n_slices from per-slice data if not in file
    if n_slices is None and violations_per_slice_list:
        n_slices = violations_per_slice_list[0].shape[1]
    if n_slices is None:
        n_slices = 5  # Default
    
    return {
        'violations_raw': np.array(violations_list),
        'resources_raw': np.array(resources_list),
        'violations_per_slice': np.array(violations_per_slice_list) if violations_per_slice_list else None,
        'resources_per_slice': np.array(resources_per_slice_list) if resources_per_slice_list else None,
        'n_slices': n_slices,
        'n_runs': len(violations_list),
        'window': window,
    }


def compute_metrics(data, window):
    """Compute all metrics from raw data."""
    metrics = {}
    
    # Total violations - raw and moving average
    viol_raw = data['violations_raw']
    metrics['violations_raw'] = viol_raw
    metrics['violations_ma'] = np.array([movingaverage(v, window) for v in viol_raw])
    
    # Cumulative violations
    metrics['violations_cumulative'] = np.cumsum(viol_raw, axis=1)
    
    # Resources - raw and moving average
    res_raw = data['resources_raw']
    metrics['resources_raw'] = res_raw
    metrics['resources_ma'] = np.array([movingaverage(r, window) for r in res_raw])
    
    # Per-slice violations
    if data['violations_per_slice'] is not None:
        viol_ps = data['violations_per_slice']  # shape: (n_runs, n_steps, n_slices)
        n_runs, n_steps, n_slices = viol_ps.shape
        
        # Moving average per slice
        viol_ps_ma = np.zeros((n_runs, n_steps - window + 1, n_slices))
        for run_idx in range(n_runs):
            for slice_idx in range(n_slices):
                viol_ps_ma[run_idx, :, slice_idx] = movingaverage(viol_ps[run_idx, :, slice_idx], window)
        metrics['violations_per_slice_ma'] = viol_ps_ma
        
        # Cumulative per slice
        metrics['violations_per_slice_cumulative'] = np.cumsum(viol_ps, axis=1)
    
    # Per-slice resources
    if data['resources_per_slice'] is not None:
        res_ps = data['resources_per_slice']  # shape: (n_runs, n_steps, n_slices)
        n_runs, n_steps, n_slices = res_ps.shape
        
        # Moving average per slice
        res_ps_ma = np.zeros((n_runs, n_steps - window + 1, n_slices))
        for run_idx in range(n_runs):
            for slice_idx in range(n_slices):
                res_ps_ma[run_idx, :, slice_idx] = movingaverage(res_ps[run_idx, :, slice_idx], window)
        metrics['resources_per_slice_ma'] = res_ps_ma
        metrics['resources_per_slice_raw'] = res_ps
    
    return metrics


def plot_results(metrics, data, save_path='./figures/', prbs=150, suffix=''):
    """Generate all plots. suffix is appended to filenames (e.g., '_low')."""
    os.makedirs(save_path, exist_ok=True)
    n_slices = data['n_slices']
    slice_names = SLICE_NAMES[:n_slices]
    colors = plt.cm.tab10(np.linspace(0, 1, n_slices))
    
    # ═══════════════════════════════════════════════════════════════════
    # Figure 1: Overview (3 subplots - like original)
    # ═══════════════════════════════════════════════════════════════════
    fig1, axs1 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    
    # SLA violations (moving average)
    viol_mean, viol_ci = mean_ci(metrics['violations_ma'])
    steps_ma = np.arange(len(viol_mean))
    axs1[0].plot(steps_ma, viol_mean, label='TD3')
    axs1[0].fill_between(steps_ma, viol_mean - viol_ci, viol_mean + viol_ci, color='#DDDDDD')
    axs1[0].set_title('SLA Violations (Moving Avg)')
    axs1[0].set_xlabel('Steps')
    axs1[0].set_ylabel('Violations')
    axs1[0].grid()
    axs1[0].legend()
    
    # Cumulative violations
    cum_mean, cum_ci = mean_ci(metrics['violations_cumulative'])
    steps_cum = np.arange(len(cum_mean))
    axs1[1].plot(steps_cum, cum_mean, label='TD3')
    axs1[1].fill_between(steps_cum, cum_mean - cum_ci, cum_mean + cum_ci, color='#DDDDDD')
    axs1[1].set_title('Cumulative SLA Violations')
    axs1[1].set_xlabel('Steps')
    axs1[1].set_ylabel('Total Violations')
    axs1[1].grid()
    axs1[1].legend()
    
    # Resource allocation (moving average)
    res_mean, res_ci = mean_ci(metrics['resources_ma'])
    axs1[2].plot(steps_ma, res_mean, label='TD3')
    axs1[2].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDDDDD')
    axs1[2].set_title('Resource Allocation (Moving Avg)')
    axs1[2].set_xlabel('Steps')
    axs1[2].set_ylabel('PRBs')
    axs1[2].set_ylim(0, prbs)
    axs1[2].grid()
    axs1[2].legend()
    
    fig1.savefig(f'{save_path}TD3_overview{suffix}.png', dpi=150)
    
    # ═══════════════════════════════════════════════════════════════════
    # Figure 2: Resource Allocation - Direct vs Moving Average
    # ═══════════════════════════════════════════════════════════════════
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    
    # Raw/Direct resource allocation (subsample for visibility)
    res_raw_mean, res_raw_ci = mean_ci(metrics['resources_raw'])
    steps_raw = np.arange(len(res_raw_mean))
    subsample = max(1, len(steps_raw) // 500)  # Show ~500 points
    axs2[0].plot(steps_raw[::subsample], res_raw_mean[::subsample], 
                 label='TD3', alpha=0.8, linewidth=0.8)
    axs2[0].fill_between(steps_raw[::subsample], 
                         (res_raw_mean - res_raw_ci)[::subsample], 
                         (res_raw_mean + res_raw_ci)[::subsample], 
                         color='#DDDDDD', alpha=0.5)
    axs2[0].set_title('Resource Allocation (Direct)')
    axs2[0].set_xlabel('Steps')
    axs2[0].set_ylabel('PRBs')
    axs2[0].set_ylim(0, prbs)
    axs2[0].grid()
    axs2[0].legend()
    
    # Moving average
    axs2[1].plot(steps_ma, res_mean, label='TD3')
    axs2[1].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDDDDD')
    axs2[1].set_title('Resource Allocation (Moving Avg)')
    axs2[1].set_xlabel('Steps')
    axs2[1].set_ylabel('PRBs')
    axs2[1].set_ylim(0, prbs)
    axs2[1].grid()
    axs2[1].legend()
    
    fig2.savefig(f'{save_path}TD3_resources_comparison{suffix}.png', dpi=150)
    
    # ═══════════════════════════════════════════════════════════════════
    # Figure 3: SLA Violations Per Slice
    # ═══════════════════════════════════════════════════════════════════
    if 'violations_per_slice_ma' in metrics:
        fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        
        viol_ps_ma = metrics['violations_per_slice_ma']
        steps_ps = np.arange(viol_ps_ma.shape[1])
        
        # Per-slice violations (moving average)
        for slice_idx in range(n_slices):
            slice_data = viol_ps_ma[:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs3[0].plot(steps_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
            axs3[0].fill_between(steps_ps, mean - ci, mean + ci, color=colors[slice_idx], alpha=0.2)
        
        axs3[0].set_title('SLA Violations Per Slice (Moving Avg)')
        axs3[0].set_xlabel('Steps')
        axs3[0].set_ylabel('Violations')
        axs3[0].grid()
        axs3[0].legend()
        
        # Cumulative violations per slice
        viol_ps_cum = metrics['violations_per_slice_cumulative']
        steps_cum_ps = np.arange(viol_ps_cum.shape[1])
        
        for slice_idx in range(n_slices):
            slice_data = viol_ps_cum[:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs3[1].plot(steps_cum_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
            axs3[1].fill_between(steps_cum_ps, mean - ci, mean + ci, color=colors[slice_idx], alpha=0.2)
        
        axs3[1].set_title('Cumulative SLA Violations Per Slice')
        axs3[1].set_xlabel('Steps')
        axs3[1].set_ylabel('Total Violations')
        axs3[1].grid()
        axs3[1].legend()
        
        fig3.savefig(f'{save_path}TD3_violations_per_slice{suffix}.png', dpi=150)
    
    # ═══════════════════════════════════════════════════════════════════
    # Figure 4: Resource Allocation Per Slice
    # ═══════════════════════════════════════════════════════════════════
    if 'resources_per_slice_ma' in metrics:
        fig4, axs4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        
        # Per-slice resources (moving average)
        res_ps_ma = metrics['resources_per_slice_ma']
        steps_res_ps = np.arange(res_ps_ma.shape[1])
        
        for slice_idx in range(n_slices):
            slice_data = res_ps_ma[:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs4[0].plot(steps_res_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
            axs4[0].fill_between(steps_res_ps, mean - ci, mean + ci, color=colors[slice_idx], alpha=0.2)
        
        axs4[0].set_title('Resource Allocation Per Slice (Moving Avg)')
        axs4[0].set_xlabel('Steps')
        axs4[0].set_ylabel('PRBs')
        axs4[0].grid()
        axs4[0].legend()
        
        # Per-slice resources (raw, subsampled)
        res_ps_raw = metrics['resources_per_slice_raw']
        steps_raw_ps = np.arange(res_ps_raw.shape[1])
        subsample = max(1, len(steps_raw_ps) // 500)
        
        for slice_idx in range(n_slices):
            slice_data = res_ps_raw[:, ::subsample, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs4[1].plot(steps_raw_ps[::subsample], mean, 
                        label=slice_names[slice_idx], color=colors[slice_idx], 
                        alpha=0.8, linewidth=0.8)
        
        axs4[1].set_title('Resource Allocation Per Slice (Direct)')
        axs4[1].set_xlabel('Steps')
        axs4[1].set_ylabel('PRBs')
        axs4[1].grid()
        axs4[1].legend()
        
        fig4.savefig(f'{save_path}TD3_resources_per_slice{suffix}.png', dpi=150)
    
    # ═══════════════════════════════════════════════════════════════════
    # Figure 5: Combined Dashboard
    # ═══════════════════════════════════════════════════════════════════
    fig5, axs5 = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    
    # Row 1: Total metrics
    axs5[0, 0].plot(steps_ma, viol_mean, 'b-', label='Moving Avg')
    axs5[0, 0].fill_between(steps_ma, viol_mean - viol_ci, viol_mean + viol_ci, color='#DDDDDD')
    axs5[0, 0].set_title('Total SLA Violations')
    axs5[0, 0].set_xlabel('Steps')
    axs5[0, 0].set_ylabel('Violations')
    axs5[0, 0].grid()
    axs5[0, 0].legend()
    
    axs5[0, 1].plot(steps_cum, cum_mean, 'r-')
    axs5[0, 1].fill_between(steps_cum, cum_mean - cum_ci, cum_mean + cum_ci, color='#FFDDDD')
    axs5[0, 1].set_title('Cumulative Violations')
    axs5[0, 1].set_xlabel('Steps')
    axs5[0, 1].set_ylabel('Total')
    axs5[0, 1].grid()
    
    axs5[0, 2].plot(steps_ma, res_mean, 'g-')
    axs5[0, 2].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDFFDD')
    axs5[0, 2].set_title('Total Resource Allocation')
    axs5[0, 2].set_xlabel('Steps')
    axs5[0, 2].set_ylabel('PRBs')
    axs5[0, 2].set_ylim(0, prbs)
    axs5[0, 2].grid()
    
    # Row 2: Per-slice metrics (if available)
    if 'violations_per_slice_ma' in metrics:
        for slice_idx in range(n_slices):
            slice_data = metrics['violations_per_slice_ma'][:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs5[1, 0].plot(steps_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 0].set_title('Violations Per Slice')
        axs5[1, 0].set_xlabel('Steps')
        axs5[1, 0].set_ylabel('Violations')
        axs5[1, 0].grid()
        axs5[1, 0].legend(fontsize=8)
        
        for slice_idx in range(n_slices):
            slice_data = metrics['violations_per_slice_cumulative'][:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs5[1, 1].plot(steps_cum_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 1].set_title('Cumulative Per Slice')
        axs5[1, 1].set_xlabel('Steps')
        axs5[1, 1].set_ylabel('Total')
        axs5[1, 1].grid()
        axs5[1, 1].legend(fontsize=8)
    
    if 'resources_per_slice_ma' in metrics:
        for slice_idx in range(n_slices):
            slice_data = metrics['resources_per_slice_ma'][:, :, slice_idx]
            mean, ci = mean_ci(slice_data)
            axs5[1, 2].plot(steps_res_ps, mean, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 2].set_title('Resources Per Slice')
        axs5[1, 2].set_xlabel('Steps')
        axs5[1, 2].set_ylabel('PRBs')
        axs5[1, 2].grid()
        axs5[1, 2].legend(fontsize=8)
    
    fig5.savefig(f'{save_path}TD3_dashboard{suffix}.png', dpi=150)
    
    print(f"\nPlots saved to {save_path}")
    return [fig1, fig2, fig3 if 'violations_per_slice_ma' in metrics else None, 
            fig4 if 'resources_per_slice_ma' in metrics else None, fig5]


def plot_curriculum_comparison(base_path, runs, window, prbs, save_path='./figures/'):
    """Plot comparison across all curriculum scenarios."""
    os.makedirs(save_path, exist_ok=True)
    
    scenario_data = {}
    for scenario in SCENARIOS:
        data = load_data(base_path, runs, START, END, window, scenario=scenario)
        if data is not None:
            scenario_data[scenario] = data
    
    if not scenario_data:
        print("No curriculum data found.")
        return
    
    # Colors for scenarios
    scenario_colors = {'low': 'green', 'medium': 'orange', 'congested': 'red'}
    
    # Figure: Compare violations across scenarios
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    
    for scenario, data in scenario_data.items():
        metrics = compute_metrics(data, window)
        color = scenario_colors.get(scenario, 'blue')
        
        # Violations (moving avg)
        viol_mean, viol_ci = mean_ci(metrics['violations_ma'])
        steps = np.arange(len(viol_mean))
        axs[0].plot(steps, viol_mean, label=scenario.capitalize(), color=color)
        axs[0].fill_between(steps, viol_mean - viol_ci, viol_mean + viol_ci, color=color, alpha=0.2)
        
        # Cumulative violations
        cum_mean, cum_ci = mean_ci(metrics['violations_cumulative'])
        axs[1].plot(np.arange(len(cum_mean)), cum_mean, label=scenario.capitalize(), color=color)
        axs[1].fill_between(np.arange(len(cum_mean)), cum_mean - cum_ci, cum_mean + cum_ci, color=color, alpha=0.2)
        
        # Resources
        res_mean, res_ci = mean_ci(metrics['resources_ma'])
        axs[2].plot(steps, res_mean, label=scenario.capitalize(), color=color)
        axs[2].fill_between(steps, res_mean - res_ci, res_mean + res_ci, color=color, alpha=0.2)
    
    axs[0].set_title('SLA Violations by Scenario')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Violations')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].set_title('Cumulative Violations by Scenario')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Total Violations')
    axs[1].legend()
    axs[1].grid()
    
    axs[2].set_title('Resource Allocation by Scenario')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('PRBs')
    axs[2].set_ylim(0, prbs)
    axs[2].legend()
    axs[2].grid()
    
    fig.savefig(f'{save_path}TD3_curriculum_comparison.png', dpi=150)
    print(f"Saved curriculum comparison to {save_path}TD3_curriculum_comparison.png")
    
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot TD3 results")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=['low', 'medium', 'congested', 'all'],
                        help="Scenario to plot (curriculum mode). Use 'all' for comparison.")
    parser.add_argument("--runs", type=int, default=22,
                        help="Number of runs to include (0 to N-1)")
    parser.add_argument("--path", type=str, default=BASE_PATH,
                        help="Path to results directory")
    parser.add_argument("--prbs", type=int, default=PRBS,
                        help="Total PRBs for y-axis limit")
    args = parser.parse_args()
    
    runs = range(0, args.runs)
    
    curriculum_available = has_curriculum_data(args.path, runs)

    if args.scenario == 'all':
        # Plot comparison across all scenarios
        plot_curriculum_comparison(args.path, runs, WINDOW, args.prbs)
        
        # Also plot each scenario individually
        for scenario in SCENARIOS:
            print(f"\n=== Plotting {scenario.upper()} scenario ===")
            data = load_data(args.path, runs, START, END, WINDOW, scenario=scenario)
            if data is not None:
                metrics = compute_metrics(data, WINDOW)
                plot_results(metrics, data, prbs=args.prbs, suffix=f"_{scenario}")
    elif args.scenario:
        # Plot specific scenario
        data = load_data(args.path, runs, START, END, WINDOW, scenario=args.scenario)
        if data is None:
            print("No data loaded. Exiting.")
            exit()
        print(f"\nLoaded {data['n_runs']} runs with {data['n_slices']} slices for {args.scenario}")
        metrics = compute_metrics(data, WINDOW)
        plot_results(metrics, data, prbs=args.prbs, suffix=f"_{args.scenario}")
    else:
        if curriculum_available:
            # Auto mode: curriculum files detected, plot them by default.
            print("Curriculum files detected. Plotting low/medium/congested outputs.")
            plot_curriculum_comparison(args.path, runs, WINDOW, args.prbs)
            for scenario in SCENARIOS:
                print(f"\n=== Plotting {scenario.upper()} scenario ===")
                data = load_data(args.path, runs, START, END, WINDOW, scenario=scenario)
                if data is not None:
                    print(f"Loaded {data['n_runs']} runs with {data['n_slices']} slices for {scenario}")
                    metrics = compute_metrics(data, WINDOW)
                    plot_results(metrics, data, prbs=args.prbs, suffix=f"_{scenario}")
        else:
            # Legacy mode - single file per run
            data = load_data(args.path, runs, START, END, WINDOW)
            if data is None:
                print("No data loaded. Exiting.")
                exit()
            print(f"\nLoaded {data['n_runs']} runs with {data['n_slices']} slices")
            metrics = compute_metrics(data, WINDOW)
            plot_results(metrics, data, prbs=args.prbs)
    
    plt.show()