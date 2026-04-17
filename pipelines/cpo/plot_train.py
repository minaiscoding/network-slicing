#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training results for CPO.

Reads history_{run_id}.npz from pipelines/cpo/ and generates plots in
pipelines/cpo/figures/.

Usage:
    python pipelines/cpo/plot_train.py
    python pipelines/cpo/plot_train.py --runs 5 --window 400
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

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_NAME    = 'CPO'
SLICE_NAMES  = ['eMBB', 'mMTC', 'URLLC']


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def mean_ci(data):
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    n    = data.shape[0]
    ci   = 1.697 * std / np.sqrt(n)
    return mean, ci


def load_data(base_path, runs, window):
    violations, resources, rewards = [], [], []
    viol_ps, res_ps = [], []
    th_ps, q_ps = [], []
    n_slices = None

    for rid in runs:
        fp = os.path.join(base_path, f'history_{rid}.npz')
        if not os.path.exists(fp):
            print(f"Missing: {fp}")
            continue
        h = np.load(fp)
        violations.append(h['violation'])
        resources.append(h['resources'])
        if 'reward' in h:
            rewards.append(h['reward'])
        if 'violation_per_slice' in h:
            viol_ps.append(h['violation_per_slice'])
        if 'resource_per_slice' in h:
            res_ps.append(h['resource_per_slice'])
        if 'throughput_per_slice' in h:
            th_ps.append(h['throughput_per_slice'])
        if 'queue_per_slice' in h:
            q_ps.append(h['queue_per_slice'])
        if 'n_slices' in h:
            n_slices = int(h['n_slices'])

    if not violations:
        print("No data found.")
        return None

    minlen = min(len(v) for v in violations)
    violations = [v[:minlen] for v in violations]
    resources  = [r[:minlen] for r in resources]
    rewards    = [r[:minlen] for r in rewards] if rewards else []
    viol_ps    = [v[:minlen] for v in viol_ps] if viol_ps else []
    res_ps     = [r[:minlen] for r in res_ps]  if res_ps  else []
    th_ps      = [t[:minlen] for t in th_ps]   if th_ps   else []
    q_ps       = [q[:minlen] for q in q_ps]    if q_ps    else []

    if n_slices is None and viol_ps:
        n_slices = viol_ps[0].shape[1]
    if n_slices is None:
        n_slices = 3

    return {
        'violations': np.array(violations),
        'resources':  np.array(resources),
        'rewards':    np.array(rewards) if rewards else None,
        'viol_ps':    np.array(viol_ps) if viol_ps else None,
        'res_ps':     np.array(res_ps)  if res_ps  else None,
        'th_ps':      np.array(th_ps)   if th_ps   else None,
        'q_ps':       np.array(q_ps)    if q_ps    else None,
        'n_slices':   n_slices,
        'n_runs':     len(violations),
    }


def plot(data, window, save_path):
    os.makedirs(save_path, exist_ok=True)
    n_slices = data['n_slices']
    slice_names = SLICE_NAMES[:n_slices]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_slices, 3)))

    viol_ma = np.array([movingaverage(v, window) for v in data['violations']])
    res_ma  = np.array([movingaverage(r, window) for r in data['resources']])
    steps_ma = np.arange(viol_ma.shape[1])

    # Overview
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    vm, vc = mean_ci(viol_ma)
    axs[0].plot(steps_ma, vm, label=ALGO_NAME)
    axs[0].fill_between(steps_ma, vm - vc, vm + vc, color='#DDDDDD')
    axs[0].set_title('SLA Violations (Moving Avg)')
    axs[0].set_xlabel('Steps'); axs[0].set_ylabel('Violations'); axs[0].grid(); axs[0].legend()

    cum = np.cumsum(data['violations'], axis=1)
    cm, cc = mean_ci(cum)
    axs[1].plot(np.arange(len(cm)), cm, label=ALGO_NAME)
    axs[1].fill_between(np.arange(len(cm)), cm - cc, cm + cc, color='#DDDDDD')
    axs[1].set_title('Cumulative SLA Violations')
    axs[1].set_xlabel('Steps'); axs[1].set_ylabel('Total Violations'); axs[1].grid(); axs[1].legend()

    rm, rc = mean_ci(res_ma)
    axs[2].plot(steps_ma, rm, label=ALGO_NAME)
    axs[2].fill_between(steps_ma, rm - rc, rm + rc, color='#DDDDDD')
    axs[2].set_title('Resource Allocation (Moving Avg)')
    axs[2].set_xlabel('Steps'); axs[2].set_ylabel('PRBs'); axs[2].grid(); axs[2].legend()
    fig.savefig(os.path.join(save_path, f'{ALGO_NAME}_overview.png'), dpi=150)
    plt.close(fig)

    # Reward & Cost
    fig2, axs2 = plt.subplots(1, 4 if data['rewards'] is not None else 2,
                               figsize=(18 if data['rewards'] is not None else 10, 4),
                               constrained_layout=True)
    col = 0
    if data['rewards'] is not None:
        rew_ma = np.array([movingaverage(r, window) for r in data['rewards']])
        rwm, rwc = mean_ci(rew_ma)
        axs2[col].plot(steps_ma, rwm, color='tab:green', label=ALGO_NAME)
        axs2[col].fill_between(steps_ma, rwm - rwc, rwm + rwc, color='#DDFFDD')
        axs2[col].set_title('Reward (Moving Avg)')
        axs2[col].set_xlabel('Steps'); axs2[col].set_ylabel('Reward'); axs2[col].grid(); axs2[col].legend()
        col += 1
        rew_cum = np.cumsum(data['rewards'], axis=1)
        rcm, rcc = mean_ci(rew_cum)
        axs2[col].plot(np.arange(len(rcm)), rcm, color='tab:green', label=ALGO_NAME)
        axs2[col].fill_between(np.arange(len(rcm)), rcm - rcc, rcm + rcc, color='#DDFFDD')
        axs2[col].set_title('Cumulative Reward')
        axs2[col].set_xlabel('Steps'); axs2[col].set_ylabel('Total Reward'); axs2[col].grid(); axs2[col].legend()
        col += 1

    cost_ma = np.array([movingaverage(v.astype(float), window) for v in data['violations']])
    com, coc = mean_ci(cost_ma)
    axs2[col].plot(steps_ma, com, color='tab:red', label=ALGO_NAME)
    axs2[col].fill_between(steps_ma, com - coc, com + coc, color='#FFDDDD')
    axs2[col].set_title('Cost (Moving Avg)')
    axs2[col].set_xlabel('Steps'); axs2[col].set_ylabel('Cost'); axs2[col].grid(); axs2[col].legend()
    col += 1
    cost_cum = np.cumsum(data['violations'].astype(float), axis=1)
    ccm, ccc = mean_ci(cost_cum)
    axs2[col].plot(np.arange(len(ccm)), ccm, color='tab:red', label=ALGO_NAME)
    axs2[col].fill_between(np.arange(len(ccm)), ccm - ccc, ccm + ccc, color='#FFDDDD')
    axs2[col].set_title('Cumulative Cost')
    axs2[col].set_xlabel('Steps'); axs2[col].set_ylabel('Total Cost'); axs2[col].grid(); axs2[col].legend()
    fig2.savefig(os.path.join(save_path, f'{ALGO_NAME}_reward_cost.png'), dpi=150)
    plt.close(fig2)

    # Per-slice violations
    if data['viol_ps'] is not None:
        fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        vps = data['viol_ps']
        for si in range(n_slices):
            ma = np.array([movingaverage(vps[r, :, si].astype(float), window) for r in range(vps.shape[0])])
            m, c = mean_ci(ma)
            axs3[0].plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            axs3[0].fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        axs3[0].set_title('SLA Violations Per Slice (Moving Avg)')
        axs3[0].set_xlabel('Steps'); axs3[0].set_ylabel('Violations'); axs3[0].grid(); axs3[0].legend()

        cum_ps = np.cumsum(vps, axis=1)
        for si in range(n_slices):
            m, c = mean_ci(cum_ps[:, :, si])
            axs3[1].plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            axs3[1].fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        axs3[1].set_title('Cumulative SLA Violations Per Slice')
        axs3[1].set_xlabel('Steps'); axs3[1].set_ylabel('Total Violations'); axs3[1].grid(); axs3[1].legend()
        fig3.savefig(os.path.join(save_path, f'{ALGO_NAME}_violations_per_slice.png'), dpi=150)
        plt.close(fig3)

    # Per-slice resources
    if data['res_ps'] is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 5), constrained_layout=True)
        rps = data['res_ps']
        for si in range(n_slices):
            ma = np.array([movingaverage(rps[r, :, si].astype(float), window) for r in range(rps.shape[0])])
            m, c = mean_ci(ma)
            ax4.plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            ax4.fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        ax4.set_title('Resource Allocation Per Slice (Moving Avg)')
        ax4.set_xlabel('Steps'); ax4.set_ylabel('PRBs'); ax4.grid(); ax4.legend()
        fig4.savefig(os.path.join(save_path, f'{ALGO_NAME}_resources_per_slice.png'), dpi=150)
        plt.close(fig4)

    # Per-slice throughput
    if data['th_ps'] is not None:
        fig5, axs5 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        tps = data['th_ps']
        for si in range(n_slices):
            ma = np.array([movingaverage(tps[r, :, si], window) for r in range(tps.shape[0])])
            m, c = mean_ci(ma)
            axs5[0].plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            axs5[0].fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        axs5[0].set_title('Throughput Per Slice (Moving Avg)')
        axs5[0].set_xlabel('Steps'); axs5[0].set_ylabel('Throughput (bits)'); axs5[0].grid(); axs5[0].legend()

        cum_th = np.cumsum(tps, axis=1)
        for si in range(n_slices):
            m, c = mean_ci(cum_th[:, :, si])
            axs5[1].plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            axs5[1].fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        axs5[1].set_title('Cumulative Throughput Per Slice')
        axs5[1].set_xlabel('Steps'); axs5[1].set_ylabel('Total Throughput (bits)'); axs5[1].grid(); axs5[1].legend()
        fig5.savefig(os.path.join(save_path, f'{ALGO_NAME}_throughput_per_slice.png'), dpi=150)
        plt.close(fig5)

    # Per-slice queue size
    if data['q_ps'] is not None:
        fig6, ax6 = plt.subplots(figsize=(10, 5), constrained_layout=True)
        qps = data['q_ps']
        for si in range(n_slices):
            ma = np.array([movingaverage(qps[r, :, si], window) for r in range(qps.shape[0])])
            m, c = mean_ci(ma)
            ax6.plot(np.arange(len(m)), m, label=slice_names[si], color=colors[si])
            ax6.fill_between(np.arange(len(m)), m - c, m + c, color=colors[si], alpha=0.2)
        ax6.set_title('Queue Size Per Slice (Moving Avg)')
        ax6.set_xlabel('Steps'); ax6.set_ylabel('Queue (bits)'); ax6.grid(); ax6.legend()
        fig6.savefig(os.path.join(save_path, f'{ALGO_NAME}_queue_per_slice.png'), dpi=150)
        plt.close(fig6)

    print(f"Plots saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot CPO training results")
    parser.add_argument("--runs",   type=int, default=30)
    parser.add_argument("--window", type=int, default=400)
    args = parser.parse_args()

    data = load_data(PIPELINE_DIR, range(args.runs), args.window)
    if data is not None:
        plot(data, args.window, os.path.join(PIPELINE_DIR, 'figures'))
