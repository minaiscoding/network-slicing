#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot comprehensive results from SAC-Lag / TD3 / PPO runs.

Supports three file layouts, detected automatically:

1. Single-scenario runs (this project's current SAC-Lag setup):
      history_{run_id}_medium.npz          (or _low / _congested — one scenario only)

2. Full curriculum runs:
      history_{run_id}_low.npz
      history_{run_id}_medium.npz
      history_{run_id}_congested.npz

3. Legacy (no scenario suffix):
      history_{run_id}.npz

Plots:
- SLA violations (total, moving average)
- SLA violations per slice
- Cumulative SLA violations (total and per slice)
- Resource allocation (direct and moving average)
- Resource allocation per slice
- Reward (moving avg + cumulative), if present in the npz
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --- Config ---
START = 0
END = 360_000                                        # full 360k-step SAC-Lag runs
WINDOW = 2_000                                       # moving-avg window ≈ one episode
RUNS = range(0, 30)                                  # 0..29 (matches RUNS=30 in trainer)
BASE_PATH = 'results/scenario_comparison/SACLag'     # SAC-Lag output folder
PRBS = 150                                           # total PRBs (for y-axis cap)
SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC', 'Slice4', 'Slice5']
SCENARIOS = ['low', 'medium', 'congested']


def _discover_scenarios(base_path, runs):
    """
    Return the list of scenarios for which at least one run file exists.
    Empty list means 'no scenario-suffixed files found'.
    """
    found = []
    for scenario in SCENARIOS:
        for run_id in runs:
            if os.path.exists(os.path.join(base_path, f'history_{run_id}_{scenario}.npz')):
                found.append(scenario)
                break
    return found


def has_curriculum_data(base_path, runs):
    """True if files for MORE THAN ONE scenario exist (= full curriculum run)."""
    return len(_discover_scenarios(base_path, runs)) > 1


def movingaverage(values, window):
    """Compute moving average with convolution."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def mean_ci(data):
    """Compute mean and 90% confidence interval."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n = data.shape[0]
    ci = 1.697 * std / np.sqrt(max(n, 1))  # 90% confidence
    return mean, ci


def load_data(base_path, runs, start, end, window, scenario=None):
    """
    Load data from all runs and compute statistics.

    Args:
        scenario: If specified, loads history_{run_id}_{scenario}.npz
                  Otherwise loads history_{run_id}.npz

    Returns dict with raw and processed data, or None if nothing was loaded.
    """
    violations_list = []
    resources_list = []
    rewards_list = []
    costs_list = []
    ep_returns_list = []
    ep_costs_list   = []
    violations_per_slice_list = []
    resources_per_slice_list = []
    n_slices = None

    min_len = None   # clip every array to the shortest run length

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

        if 'n_slices' in histories:
            n_slices = int(histories['n_slices'])

        # Clip to whatever is available.  If the wrapper saved step_count, use it
        # so trailing pre-allocated zeros are excluded from partial runs.
        avail = len(histories['violation'])
        if 'step_count' in histories:
            avail = min(avail, int(histories['step_count']))
        lo    = max(0, min(start, avail))
        hi    = min(end, avail)
        if hi - lo <= 0:
            print(f"  (empty range for run {run_id})")
            continue

        v_run = histories['violation'][lo:hi]
        r_run = histories['resources'][lo:hi]
        violations_list.append(v_run)
        resources_list.append(r_run)

        if 'reward' in histories:
            rewards_list.append(histories['reward'][lo:hi])

        if 'cost' in histories:
            costs_list.append(histories['cost'][lo:hi])

        if 'episode_return' in histories and len(histories['episode_return']) > 0:
            ep_returns_list.append(histories['episode_return'])

        if 'episode_cost' in histories and len(histories['episode_cost']) > 0:
            ep_costs_list.append(histories['episode_cost'])

        if 'violation_per_slice' in histories:
            violations_per_slice_list.append(histories['violation_per_slice'][lo:hi])

        if 'resource_per_slice' in histories:
            resources_per_slice_list.append(histories['resource_per_slice'][lo:hi])

        cur_len = hi - lo
        min_len = cur_len if min_len is None else min(min_len, cur_len)

    if not violations_list:
        return None

    # Clip all arrays to min_len so np.array(...) gets a rectangular stack.
    violations_list = [v[:min_len] for v in violations_list]
    resources_list  = [r[:min_len] for r in resources_list]
    if rewards_list:
        rewards_list = [r[:min_len] for r in rewards_list]
    if violations_per_slice_list:
        violations_per_slice_list = [v[:min_len] for v in violations_per_slice_list]
    if resources_per_slice_list:
        resources_per_slice_list = [r[:min_len] for r in resources_per_slice_list]

    if costs_list:
        costs_list = [c[:min_len] for c in costs_list]

    # Episode-level arrays have variable length across runs — clip to shortest.
    if ep_returns_list:
        min_ep = min(len(e) for e in ep_returns_list)
        ep_returns_list = [e[:min_ep] for e in ep_returns_list]
    if ep_costs_list:
        min_ep_c = min(len(e) for e in ep_costs_list)
        ep_costs_list = [e[:min_ep_c] for e in ep_costs_list]

    # Guard: window must be smaller than the run length, otherwise moving avg is empty.
    if min_len <= window:
        print(f"WARNING: run length ({min_len}) ≤ window ({window}); "
              f"shrinking window to {max(min_len // 4, 1)}.")
        window = max(min_len // 4, 1)

    if n_slices is None and violations_per_slice_list:
        n_slices = violations_per_slice_list[0].shape[1]
    if n_slices is None:
        n_slices = 5

    return {
        'violations_raw': np.array(violations_list),
        'resources_raw': np.array(resources_list),
        'rewards_raw': np.array(rewards_list) if rewards_list else None,
        'costs_raw': np.array(costs_list) if costs_list else None,
        'ep_returns': np.array(ep_returns_list) if ep_returns_list else None,
        'ep_costs':   np.array(ep_costs_list)   if ep_costs_list   else None,
        'violations_per_slice': np.array(violations_per_slice_list) if violations_per_slice_list else None,
        'resources_per_slice': np.array(resources_per_slice_list) if resources_per_slice_list else None,
        'n_slices': n_slices,
        'n_runs': len(violations_list),
        'window': window,
    }


def compute_metrics(data, window):
    """Compute all metrics from raw data."""
    # If load_data had to shrink the window, honour it here too.
    window = data.get('window', window)

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

    # Rewards - raw and moving average (if available)
    if data.get('rewards_raw') is not None:
        rew_raw = data['rewards_raw']
        metrics['rewards_raw'] = rew_raw
        metrics['rewards_ma'] = np.array([movingaverage(r, window) for r in rew_raw])

    # OmniSafe cost signal - raw and moving average (if available)
    if data.get('costs_raw') is not None:
        cost_raw = data['costs_raw']
        metrics['costs_raw'] = cost_raw
        metrics['costs_ma'] = np.array([movingaverage(c, window) for c in cost_raw])

    # Per-slice violations
    if data['violations_per_slice'] is not None:
        viol_ps = data['violations_per_slice']
        n_runs, n_steps, n_slices = viol_ps.shape
        viol_ps_ma = np.zeros((n_runs, n_steps - window + 1, n_slices))
        for run_idx in range(n_runs):
            for slice_idx in range(n_slices):
                viol_ps_ma[run_idx, :, slice_idx] = movingaverage(viol_ps[run_idx, :, slice_idx], window)
        metrics['violations_per_slice_ma'] = viol_ps_ma
        metrics['violations_per_slice_cumulative'] = np.cumsum(viol_ps, axis=1)

    # Per-slice resources
    if data['resources_per_slice'] is not None:
        res_ps = data['resources_per_slice']
        n_runs, n_steps, n_slices = res_ps.shape
        res_ps_ma = np.zeros((n_runs, n_steps - window + 1, n_slices))
        for run_idx in range(n_runs):
            for slice_idx in range(n_slices):
                res_ps_ma[run_idx, :, slice_idx] = movingaverage(res_ps[run_idx, :, slice_idx], window)
        metrics['resources_per_slice_ma'] = res_ps_ma
        metrics['resources_per_slice_raw'] = res_ps

    return metrics


def plot_results(metrics, data, save_path='./figures/', prbs=150, suffix='', algo_name=''):
    """Generate all plots. suffix is appended to filenames (e.g., '_medium')."""
    os.makedirs(save_path, exist_ok=True)
    if not algo_name:
        algo_name = os.path.basename(os.path.normpath(save_path))
    n_slices = data['n_slices']
    slice_names = SLICE_NAMES[:n_slices]
    colors = plt.cm.tab10(np.linspace(0, 1, n_slices))

    label = algo_name

    # ═══════════════════════════════════════════════════════════════════
    # Figure 1: Overview
    # ═══════════════════════════════════════════════════════════════════
    fig1, axs1 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    viol_mean, viol_ci = mean_ci(metrics['violations_ma'])
    steps_ma = np.arange(len(viol_mean))
    axs1[0].plot(steps_ma, viol_mean, label=label)
    axs1[0].fill_between(steps_ma, viol_mean - viol_ci, viol_mean + viol_ci, color='#DDDDDD')
    axs1[0].set_title('SLA Violations (Moving Avg)')
    axs1[0].set_xlabel('Steps'); axs1[0].set_ylabel('Violations')
    axs1[0].grid(); axs1[0].legend()

    cum_mean, cum_ci = mean_ci(metrics['violations_cumulative'])
    steps_cum = np.arange(len(cum_mean))
    axs1[1].plot(steps_cum, cum_mean, label=label)
    axs1[1].fill_between(steps_cum, cum_mean - cum_ci, cum_mean + cum_ci, color='#DDDDDD')
    axs1[1].set_title('Cumulative SLA Violations')
    axs1[1].set_xlabel('Steps'); axs1[1].set_ylabel('Total Violations')
    axs1[1].grid(); axs1[1].legend()

    res_mean, res_ci = mean_ci(metrics['resources_ma'])
    axs1[2].plot(steps_ma, res_mean, label=label)
    axs1[2].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDDDDD')
    axs1[2].set_title('Resource Allocation (Moving Avg)')
    axs1[2].set_xlabel('Steps'); axs1[2].set_ylabel('PRBs')
    axs1[2].set_ylim(0, prbs); axs1[2].grid(); axs1[2].legend()

    fig1.savefig(f'{save_path}{algo_name}_overview{suffix}.png', dpi=150)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 2: Resource Allocation - Direct vs Moving Average
    # ═══════════════════════════════════════════════════════════════════
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    res_raw_mean, res_raw_ci = mean_ci(metrics['resources_raw'])
    steps_raw = np.arange(len(res_raw_mean))
    subsample = max(1, len(steps_raw) // 500)
    axs2[0].plot(steps_raw[::subsample], res_raw_mean[::subsample],
                 label=label, alpha=0.8, linewidth=0.8)
    axs2[0].fill_between(steps_raw[::subsample],
                         (res_raw_mean - res_raw_ci)[::subsample],
                         (res_raw_mean + res_raw_ci)[::subsample],
                         color='#DDDDDD', alpha=0.5)
    axs2[0].set_title('Resource Allocation (Direct)')
    axs2[0].set_xlabel('Steps'); axs2[0].set_ylabel('PRBs')
    axs2[0].set_ylim(0, prbs); axs2[0].grid(); axs2[0].legend()

    axs2[1].plot(steps_ma, res_mean, label=label)
    axs2[1].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDDDDD')
    axs2[1].set_title('Resource Allocation (Moving Avg)')
    axs2[1].set_xlabel('Steps'); axs2[1].set_ylabel('PRBs')
    axs2[1].set_ylim(0, prbs); axs2[1].grid(); axs2[1].legend()

    fig2.savefig(f'{save_path}{algo_name}_resources_comparison{suffix}.png', dpi=150)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 3: Episode Return + Episode Cost  (matches OmniSafe's plot)
    # ═══════════════════════════════════════════════════════════════════
    ep_ret  = data.get('ep_returns')
    ep_cost = data.get('ep_costs')

    if ep_ret is not None:
        fig_rew, axs_rew = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        ep_ret_mean, ep_ret_ci = mean_ci(ep_ret)
        ep_ret_episodes = np.arange(len(ep_ret_mean))
        axs_rew[0].plot(ep_ret_episodes, ep_ret_mean, label=label)
        axs_rew[0].fill_between(ep_ret_episodes,
                                ep_ret_mean - ep_ret_ci, ep_ret_mean + ep_ret_ci, color='#DDDDDD')
        axs_rew[0].set_title('Episode Return (matches OmniSafe)')
        axs_rew[0].set_xlabel('Episodes'); axs_rew[0].set_ylabel('Return')
        axs_rew[0].grid(); axs_rew[0].legend()

        ep_ret_cum_mean, ep_ret_cum_ci = mean_ci(np.cumsum(ep_ret, axis=1))
        axs_rew[1].plot(np.arange(len(ep_ret_cum_mean)), ep_ret_cum_mean, label=label)
        axs_rew[1].fill_between(np.arange(len(ep_ret_cum_mean)),
                                ep_ret_cum_mean - ep_ret_cum_ci,
                                ep_ret_cum_mean + ep_ret_cum_ci, color='#DDDDDD')
        axs_rew[1].set_title('Cumulative Episode Return')
        axs_rew[1].set_xlabel('Episodes'); axs_rew[1].set_ylabel('Total Return')
        axs_rew[1].grid(); axs_rew[1].legend()

        fig_rew.savefig(f'{save_path}{algo_name}_reward{suffix}.png', dpi=150)

    if ep_cost is not None:
        fig_cost, axs_cost = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        ep_cost_mean, ep_cost_ci = mean_ci(ep_cost)
        ep_cost_episodes = np.arange(len(ep_cost_mean))
        axs_cost[0].plot(ep_cost_episodes, ep_cost_mean, label=label, color='orange')
        axs_cost[0].fill_between(ep_cost_episodes,
                                 ep_cost_mean - ep_cost_ci, ep_cost_mean + ep_cost_ci,
                                 color='#DDDDDD')
        axs_cost[0].set_title('Episode Cost (matches OmniSafe)')
        axs_cost[0].set_xlabel('Episodes'); axs_cost[0].set_ylabel('Cost')
        axs_cost[0].grid(); axs_cost[0].legend()

        ep_cost_cum_mean, ep_cost_cum_ci = mean_ci(np.cumsum(ep_cost, axis=1))
        axs_cost[1].plot(np.arange(len(ep_cost_cum_mean)), ep_cost_cum_mean,
                         label=label, color='orange')
        axs_cost[1].fill_between(np.arange(len(ep_cost_cum_mean)),
                                 ep_cost_cum_mean - ep_cost_cum_ci,
                                 ep_cost_cum_mean + ep_cost_cum_ci, color='#DDDDDD')
        axs_cost[1].set_title('Cumulative Episode Cost')
        axs_cost[1].set_xlabel('Episodes'); axs_cost[1].set_ylabel('Total Cost')
        axs_cost[1].grid(); axs_cost[1].legend()

        fig_cost.savefig(f'{save_path}{algo_name}_cost{suffix}.png', dpi=150)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 4: SLA Violations Per Slice
    # ═══════════════════════════════════════════════════════════════════
    if 'violations_per_slice_ma' in metrics:
        fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        viol_ps_ma = metrics['violations_per_slice_ma']
        steps_ps = np.arange(viol_ps_ma.shape[1])

        for slice_idx in range(n_slices):
            m, ci = mean_ci(viol_ps_ma[:, :, slice_idx])
            axs3[0].plot(steps_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
            axs3[0].fill_between(steps_ps, m - ci, m + ci, color=colors[slice_idx], alpha=0.2)
        axs3[0].set_title('SLA Violations Per Slice (Moving Avg)')
        axs3[0].set_xlabel('Steps'); axs3[0].set_ylabel('Violations')
        axs3[0].grid(); axs3[0].legend()

        viol_ps_cum = metrics['violations_per_slice_cumulative']
        steps_cum_ps = np.arange(viol_ps_cum.shape[1])
        for slice_idx in range(n_slices):
            m, ci = mean_ci(viol_ps_cum[:, :, slice_idx])
            axs3[1].plot(steps_cum_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
            axs3[1].fill_between(steps_cum_ps, m - ci, m + ci, color=colors[slice_idx], alpha=0.2)
        axs3[1].set_title('Cumulative SLA Violations Per Slice')
        axs3[1].set_xlabel('Steps'); axs3[1].set_ylabel('Total Violations')
        axs3[1].grid(); axs3[1].legend()

        fig3.savefig(f'{save_path}{algo_name}_violations_per_slice{suffix}.png', dpi=150)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 5: Resource Allocation Per Slice
    # ═══════════════════════════════════════════════════════════════════
    if 'resources_per_slice_ma' in metrics:
        fig4, axs4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        res_ps_ma = metrics['resources_per_slice_ma']
        steps_res_ps = np.arange(res_ps_ma.shape[1])
        for slice_idx in range(n_slices):
            m, ci = mean_ci(res_ps_ma[:, :, slice_idx])
            axs4[0].plot(steps_res_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
            axs4[0].fill_between(steps_res_ps, m - ci, m + ci, color=colors[slice_idx], alpha=0.2)
        axs4[0].set_title('Resource Allocation Per Slice (Moving Avg)')
        axs4[0].set_xlabel('Steps'); axs4[0].set_ylabel('PRBs')
        axs4[0].grid(); axs4[0].legend()

        res_ps_raw = metrics['resources_per_slice_raw']
        steps_raw_ps = np.arange(res_ps_raw.shape[1])
        subsample = max(1, len(steps_raw_ps) // 500)
        for slice_idx in range(n_slices):
            m, _ = mean_ci(res_ps_raw[:, ::subsample, slice_idx])
            axs4[1].plot(steps_raw_ps[::subsample], m,
                         label=slice_names[slice_idx], color=colors[slice_idx],
                         alpha=0.8, linewidth=0.8)
        axs4[1].set_title('Resource Allocation Per Slice (Direct)')
        axs4[1].set_xlabel('Steps'); axs4[1].set_ylabel('PRBs')
        axs4[1].grid(); axs4[1].legend()

        fig4.savefig(f'{save_path}{algo_name}_resources_per_slice{suffix}.png', dpi=150)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 6: Dashboard
    # ═══════════════════════════════════════════════════════════════════
    fig5, axs5 = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    axs5[0, 0].plot(steps_ma, viol_mean, 'b-', label='Moving Avg')
    axs5[0, 0].fill_between(steps_ma, viol_mean - viol_ci, viol_mean + viol_ci, color='#DDDDDD')
    axs5[0, 0].set_title('Total SLA Violations')
    axs5[0, 0].set_xlabel('Steps'); axs5[0, 0].set_ylabel('Violations')
    axs5[0, 0].grid(); axs5[0, 0].legend()

    axs5[0, 1].plot(steps_cum, cum_mean, 'r-')
    axs5[0, 1].fill_between(steps_cum, cum_mean - cum_ci, cum_mean + cum_ci, color='#FFDDDD')
    axs5[0, 1].set_title('Cumulative Violations')
    axs5[0, 1].set_xlabel('Steps'); axs5[0, 1].set_ylabel('Total')
    axs5[0, 1].grid()

    axs5[0, 2].plot(steps_ma, res_mean, 'g-')
    axs5[0, 2].fill_between(steps_ma, res_mean - res_ci, res_mean + res_ci, color='#DDFFDD')
    axs5[0, 2].set_title('Total Resource Allocation')
    axs5[0, 2].set_xlabel('Steps'); axs5[0, 2].set_ylabel('PRBs')
    axs5[0, 2].set_ylim(0, prbs); axs5[0, 2].grid()

    if 'violations_per_slice_ma' in metrics:
        for slice_idx in range(n_slices):
            m, _ = mean_ci(metrics['violations_per_slice_ma'][:, :, slice_idx])
            axs5[1, 0].plot(steps_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 0].set_title('Violations Per Slice')
        axs5[1, 0].set_xlabel('Steps'); axs5[1, 0].set_ylabel('Violations')
        axs5[1, 0].grid(); axs5[1, 0].legend(fontsize=8)

        for slice_idx in range(n_slices):
            m, _ = mean_ci(metrics['violations_per_slice_cumulative'][:, :, slice_idx])
            axs5[1, 1].plot(steps_cum_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 1].set_title('Cumulative Per Slice')
        axs5[1, 1].set_xlabel('Steps'); axs5[1, 1].set_ylabel('Total')
        axs5[1, 1].grid(); axs5[1, 1].legend(fontsize=8)

    # Bottom-right: per-slice resources if available, else reward if available.
    if 'resources_per_slice_ma' in metrics:
        for slice_idx in range(n_slices):
            m, _ = mean_ci(metrics['resources_per_slice_ma'][:, :, slice_idx])
            axs5[1, 2].plot(steps_res_ps, m, label=slice_names[slice_idx], color=colors[slice_idx])
        axs5[1, 2].set_title('Resources Per Slice')
        axs5[1, 2].set_xlabel('Steps'); axs5[1, 2].set_ylabel('PRBs')
        axs5[1, 2].grid(); axs5[1, 2].legend(fontsize=8)
    elif 'rewards_ma' in metrics:
        rew_mean_d, rew_ci_d = mean_ci(metrics['rewards_ma'])
        axs5[1, 2].plot(np.arange(len(rew_mean_d)), rew_mean_d, color='purple')
        axs5[1, 2].fill_between(np.arange(len(rew_mean_d)),
                                rew_mean_d - rew_ci_d, rew_mean_d + rew_ci_d,
                                color='#DDDDFF')
        axs5[1, 2].set_title('Reward (Moving Avg)')
        axs5[1, 2].set_xlabel('Steps'); axs5[1, 2].set_ylabel('Reward')
        axs5[1, 2].grid()

    fig5.savefig(f'{save_path}{algo_name}_dashboard{suffix}.png', dpi=150)

    print(f"\nPlots saved to {save_path}")


def plot_curriculum_comparison(base_path, runs, window, prbs, save_path='./figures/', algo_name=''):
    """Plot comparison across all curriculum scenarios (only used when >1 scenario exists)."""
    os.makedirs(save_path, exist_ok=True)
    if not algo_name:
        algo_name = os.path.basename(os.path.normpath(base_path))

    scenario_data = {}
    for scenario in SCENARIOS:
        data = load_data(base_path, runs, START, END, window, scenario=scenario)
        if data is not None:
            scenario_data[scenario] = data

    if not scenario_data:
        print("No curriculum data found.")
        return

    scenario_colors = {'low': 'green', 'medium': 'orange', 'congested': 'red'}

    has_reward = any(
        compute_metrics(data, window).get('rewards_ma') is not None
        for data in scenario_data.values()
    )
    has_cost = any(
        compute_metrics(data, window).get('costs_ma') is not None
        for data in scenario_data.values()
    )
    n_cols = 3 + int(has_reward) + int(has_cost)
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols + 1, 5), constrained_layout=True)

    for scenario, data in scenario_data.items():
        metrics = compute_metrics(data, window)
        color = scenario_colors.get(scenario, 'blue')

        viol_mean, viol_ci = mean_ci(metrics['violations_ma'])
        steps = np.arange(len(viol_mean))
        axs[0].plot(steps, viol_mean, label=scenario.capitalize(), color=color)
        axs[0].fill_between(steps, viol_mean - viol_ci, viol_mean + viol_ci, color=color, alpha=0.2)

        cum_mean, cum_ci = mean_ci(metrics['violations_cumulative'])
        axs[1].plot(np.arange(len(cum_mean)), cum_mean, label=scenario.capitalize(), color=color)
        axs[1].fill_between(np.arange(len(cum_mean)), cum_mean - cum_ci, cum_mean + cum_ci,
                            color=color, alpha=0.2)

        res_mean, res_ci = mean_ci(metrics['resources_ma'])
        axs[2].plot(steps, res_mean, label=scenario.capitalize(), color=color)
        axs[2].fill_between(steps, res_mean - res_ci, res_mean + res_ci, color=color, alpha=0.2)

        col = 3
        if has_reward and 'rewards_ma' in metrics:
            rew_mean, rew_ci = mean_ci(metrics['rewards_ma'])
            axs[col].plot(np.arange(len(rew_mean)), rew_mean, label=scenario.capitalize(), color=color)
            axs[col].fill_between(np.arange(len(rew_mean)), rew_mean - rew_ci, rew_mean + rew_ci,
                                  color=color, alpha=0.2)
        if has_reward:
            col += 1
        if has_cost and 'costs_ma' in metrics:
            cost_mean, cost_ci = mean_ci(metrics['costs_ma'])
            axs[col].plot(np.arange(len(cost_mean)), cost_mean, label=scenario.capitalize(), color=color)
            axs[col].fill_between(np.arange(len(cost_mean)), cost_mean - cost_ci, cost_mean + cost_ci,
                                  color=color, alpha=0.2)

    axs[0].set_title('SLA Violations by Scenario');      axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Violations');                     axs[0].legend(); axs[0].grid()
    axs[1].set_title('Cumulative Violations by Scenario'); axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Total Violations');               axs[1].legend(); axs[1].grid()
    axs[2].set_title('Resource Allocation by Scenario'); axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('PRBs');                           axs[2].set_ylim(0, prbs)
    axs[2].legend(); axs[2].grid()
    _col = 3
    if has_reward:
        axs[_col].set_title('Reward by Scenario');       axs[_col].set_xlabel('Steps')
        axs[_col].set_ylabel('Reward (Moving Avg)');     axs[_col].legend(); axs[_col].grid()
        _col += 1
    if has_cost:
        axs[_col].set_title('Cost by Scenario');         axs[_col].set_xlabel('Steps')
        axs[_col].set_ylabel('Cost (Moving Avg)');       axs[_col].legend(); axs[_col].grid()

    fig.savefig(f'{save_path}{algo_name}_curriculum_comparison.png', dpi=150)
    print(f"Saved curriculum comparison to {save_path}{algo_name}_curriculum_comparison.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot RAN-slicing RL results")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=['low', 'medium', 'congested', 'all'],
                        help="Scenario to plot. 'all' plots a curriculum comparison. "
                             "If omitted, auto-detects available files.")
    parser.add_argument("--runs", type=int, default=len(RUNS),
                        help="Number of runs to include (0 to N-1)")
    parser.add_argument("--path", type=str, default=BASE_PATH,
                        help="Path to results directory")
    parser.add_argument("--prbs", type=int, default=PRBS,
                        help="Total PRBs for y-axis limit")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="Moving-average window size")
    parser.add_argument("--start", type=int, default=START,
                        help="First step to include")
    parser.add_argument("--end", type=int, default=END,
                        help="Last step (exclusive) to include")
    parser.add_argument("--figures", type=str, default='./figures/',
                        help="Output directory for figures")
    args = parser.parse_args()

    runs      = range(0, args.runs)
    algo_name = os.path.basename(os.path.normpath(args.path))
    figures   = args.figures if args.figures.endswith('/') else args.figures + '/'

    available_scenarios = _discover_scenarios(args.path, runs)

    if args.scenario == 'all':
        if len(available_scenarios) > 1:
            plot_curriculum_comparison(args.path, runs, args.window, args.prbs,
                                       save_path=figures, algo_name=algo_name)
        for scenario in available_scenarios:
            print(f"\n=== Plotting {scenario.upper()} scenario ===")
            data = load_data(args.path, runs, args.start, args.end, args.window,
                             scenario=scenario)
            if data is not None:
                metrics = compute_metrics(data, args.window)
                plot_results(metrics, data, save_path=figures, prbs=args.prbs,
                             suffix=f"_{scenario}", algo_name=algo_name)

    elif args.scenario:
        data = load_data(args.path, runs, args.start, args.end, args.window,
                         scenario=args.scenario)
        if data is None:
            print("No data loaded. Exiting."); exit()
        print(f"\nLoaded {data['n_runs']} runs with {data['n_slices']} slices "
              f"for {args.scenario}")
        metrics = compute_metrics(data, args.window)
        plot_results(metrics, data, save_path=figures, prbs=args.prbs,
                     suffix=f"_{args.scenario}", algo_name=algo_name)

    else:
        # Auto-detect mode.
        if len(available_scenarios) > 1:
            print(f"Multiple scenarios found ({available_scenarios}). "
                  f"Plotting each + a comparison.")
            plot_curriculum_comparison(args.path, runs, args.window, args.prbs,
                                       save_path=figures, algo_name=algo_name)
            for scenario in available_scenarios:
                print(f"\n=== Plotting {scenario.upper()} scenario ===")
                data = load_data(args.path, runs, args.start, args.end, args.window,
                                 scenario=scenario)
                if data is not None:
                    print(f"Loaded {data['n_runs']} runs with {data['n_slices']} slices "
                          f"for {scenario}")
                    metrics = compute_metrics(data, args.window)
                    plot_results(metrics, data, save_path=figures, prbs=args.prbs,
                                 suffix=f"_{scenario}", algo_name=algo_name)

        elif len(available_scenarios) == 1:
            # Single-scenario run (our SAC-Lag case): plot that one scenario.
            scenario = available_scenarios[0]
            print(f"Single scenario detected: '{scenario}'.")
            data = load_data(args.path, runs, args.start, args.end, args.window,
                             scenario=scenario)
            if data is None:
                print("No data loaded. Exiting."); exit()
            print(f"Loaded {data['n_runs']} runs with {data['n_slices']} slices "
                  f"for {scenario}")
            metrics = compute_metrics(data, args.window)
            plot_results(metrics, data, save_path=figures, prbs=args.prbs,
                         suffix=f"_{scenario}", algo_name=algo_name)

        else:
            # Legacy: files without a scenario suffix.
            data = load_data(args.path, runs, args.start, args.end, args.window)
            if data is None:
                print("No data loaded. Exiting."); exit()
            print(f"\nLoaded {data['n_runs']} runs with {data['n_slices']} slices")
            metrics = compute_metrics(data, args.window)
            plot_results(metrics, data, save_path=figures, prbs=args.prbs,
                         algo_name=algo_name)

    plt.show()