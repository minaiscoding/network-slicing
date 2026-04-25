#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OmniSafe progress.csv files — mirrors agent.plot() methodology.

Data loading (matches omnisafe/utils/plotter.py get_datasets):
  - X-axis : Train/Epoch × steps_per_epoch  (steps_per_epoch from config.json)
  - Reward : Metrics/TestEpRet  (falls back to Metrics/EpRet)
  - Cost   : Metrics/TestEpCost (falls back to Metrics/EpCost)
  - Values are raw episode totals (not divided by EpLen)

Smoothing (matches plot_data):
  - Symmetric uniform convolution, window = --smooth  (same as OmniSafe)

Visual style:
  - Each individual run: thin grey line (alpha=0.3) in the background
  - Cross-run mean ± std: bold coloured line + shaded band (seaborn)

Usage:
  python3 plot_omnisafe_progress.py                       # all algos in ./runs/
  python3 plot_omnisafe_progress.py --algo SACLag
  python3 plot_omnisafe_progress.py --algo SACLag --smooth 5
  python3 plot_omnisafe_progress.py --runs /path/to/runs --out ./figures/
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

RUNS_DIR    = './runs'
OUT_DIR     = './figures'
SMOOTH      = 1          # match OmniSafe default (1 = no smoothing)
COST_LIMIT  = None       # set to a float to draw a dashed reference line

ALGO_COLORS = {
    'SACLag': '#1f77b4',
    'PPOLag': '#2ca02c',
}
DEFAULT_COLOR = '#d62728'


# ── Data loading (mirrors get_datasets) ──────────────────────────────────────

def load_run(csv_path: str) -> pd.DataFrame | None:
    """Load one progress.csv, returning a tidy DataFrame or None on failure."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'  skipping {csv_path}: CSV error — {e}')
        return None

    required = {'TotalEnvSteps'}
    if not required.issubset(df.columns):
        print(f'  skipping {csv_path}: missing TotalEnvSteps')
        return None

    # Mirror OmniSafe's preference: test metrics first, then train
    reward_col = 'Metrics/TestEpRet'  if 'Metrics/TestEpRet'  in df.columns else 'Metrics/EpRet'
    cost_col   = 'Metrics/TestEpCost' if 'Metrics/TestEpCost' in df.columns else 'Metrics/EpCost'

    if reward_col not in df.columns or cost_col not in df.columns:
        print(f'  skipping {csv_path}: missing reward/cost columns')
        return None

    df = df.dropna(subset=[reward_col, cost_col, 'TotalEnvSteps'])
    if len(df) == 0:
        return None

    return pd.DataFrame({
        'Steps':   df['TotalEnvSteps'].values,
        'Rewards': df[reward_col].values,
        'Costs':   df[cost_col].values,
    })


def find_runs(runs_dir: str, algo_filter: str | None) -> dict[str, list[str]]:
    algos: dict[str, list[str]] = {}
    for csv_path in sorted(glob.glob(os.path.join(runs_dir, '*', '*', 'progress.csv'))):
        algo_dir  = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        algo_name = algo_dir.split('-{')[0]
        if algo_filter and algo_name != algo_filter:
            continue
        algos.setdefault(algo_name, []).append(csv_path)
    return algos


# ── Smoothing (mirrors plot_data convolution) ─────────────────────────────────

def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """Symmetric uniform moving average — same as OmniSafe's convolution."""
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    window = min(window, len(arr))  # 'same' returns max(M,N) elements — must not exceed arr
    y = np.ones(window)
    z = np.ones(len(arr))
    return np.convolve(arr, y, 'same') / np.convolve(z, y, 'same')


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_algo(
    algo_name: str,
    csv_paths: list[str],
    smooth: int,
    out_dir: str,
) -> None:
    dfs = []
    for p in csv_paths:
        df = load_run(p)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print(f'  [{algo_name}] no valid runs — skipping.')
        return

    print(f'  [{algo_name}] {len(dfs)} valid runs  (smooth={smooth})')
    color = ALGO_COLORS.get(algo_name, DEFAULT_COLOR)

    # Apply smoothing to each run (mirrors OmniSafe's in-place convolution)
    smoothed = []
    for i, df in enumerate(dfs):
        s = df.copy()
        s['Rewards'] = smooth_series(s['Rewards'].values, smooth)
        s['Costs']   = smooth_series(s['Costs'].values,   smooth)
        s['Run'] = i
        smoothed.append(s)

    long_df = pd.concat(smoothed, ignore_index=True)

    sns.set_theme(style='darkgrid')
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f'{algo_name} — training progress  ({len(dfs)} runs)', fontsize=13)

    for metric, ax, ylabel, title in [
        ('Rewards', ax_r, 'Episode Return (TestEpRet)', 'Rewards'),
        ('Costs',   ax_c, 'Episode Cost   (TestEpCost)', 'Costs'),
    ]:
        # ── Individual grey runs ──────────────────────────────────────────────
        for df in smoothed:
            ax.plot(df['Steps'], df[metric],
                    color='#999999', linewidth=0.7, alpha=0.30, zorder=1)

        # ── seaborn mean ± sd band (matches OmniSafe's errorbar='sd') ─────────
        sns.lineplot(
            data=long_df,
            x='Steps', y=metric,
            errorbar='sd',
            color=color,
            linewidth=2.0,
            ax=ax,
            zorder=2,
            label=f'{algo_name}  (n={len(dfs)})',
        )

        if metric == 'Costs' and COST_LIMIT is not None:
            ax.axhline(COST_LIMIT, color='black', linewidth=1.5,
                       linestyle='--', label=f'cost limit ({COST_LIMIT})', zorder=3)

        ax.set_title(title)
        ax.set_xlabel('Steps')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

        # scientific notation on x-axis if steps are large (mirrors OmniSafe)
        if long_df['Steps'].max() > 5e3:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{algo_name}_omnisafe_progress.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved → {out_path}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot OmniSafe progress.csv (mirrors agent.plot)')
    parser.add_argument('--runs',   default=RUNS_DIR,  help='Root runs directory')
    parser.add_argument('--algo',   default=None,      help='Filter to one algo (e.g. SACLag)')
    parser.add_argument('--out',    default=OUT_DIR,   help='Output directory for figures')
    parser.add_argument('--smooth', type=int, default=SMOOTH,
                        help='Smoothing window (epochs) — same as OmniSafe --smooth flag')
    args = parser.parse_args()

    algos = find_runs(args.runs, args.algo)
    if not algos:
        print(f'No progress.csv files found under {args.runs}'
              + (f' for algo={args.algo}' if args.algo else ''))
        raise SystemExit(1)

    for algo_name, csv_paths in sorted(algos.items()):
        print(f'\n=== {algo_name} ({len(csv_paths)} run files) ===')
        plot_algo(algo_name, csv_paths, args.smooth, args.out)

    plt.show()
