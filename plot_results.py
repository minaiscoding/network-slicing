#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot PPO results for low / medium / congested scenarios on the same axes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

WINDOW    = 400
START     = 0
MAX_STEPS = 30000
PRBS_MAX  = 150

OUTPUT_FILE = './figures/subplots_scenario_comparison.png'

SCENARIOS = {
    'low':       ('results/low/PPOLag_eval/history_1.npz',       '#2196F3'),
    'medium':    ('results/medium/PPOLag_eval/history_1.npz',    '#FF9800'),
    'congested': ('results/congested/PPOLag_eval/history_1.npz', '#F44336'),
}

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), constrained_layout=True)

    for name, (path, color) in SCENARIOS.items():
        if not os.path.isfile(path):
            print(f'Skipping {name}: file not found ({path})')
            continue

        histories  = np.load(path)
        violations = histories['violation']
        resources  = histories['resources']

        end = min(len(violations), MAX_STEPS)
        violations = violations[START:end]
        resources  = resources[START:end]

        window = min(WINDOW, len(violations))

        # moving average trims (window-1) samples from the front —
        # align x so step 0 is always the first real step
        violations_ma = movingaverage(violations, window)
        resources_ma  = movingaverage(resources,  window)
        ma_offset = window - 1          # how many steps were consumed
        steps_ma  = np.arange(ma_offset, ma_offset + len(violations_ma))

        # cumulative is plotted raw — no smoothing, starts at 0
        steps_raw = np.arange(len(violations))
        cumulative = violations.cumsum()

        axs[0].plot(steps_ma,  violations_ma, color=color, linewidth=1.2, label=name)
        axs[1].plot(steps_raw, cumulative,    color=color, linewidth=1.2, label=name)
        axs[2].plot(steps_ma,  resources_ma,  color=color, linewidth=1.2, label=name)

    axs[0].set_title('SLA violations (moving avg)')
    axs[0].set_xlabel('step')
    axs[0].set_ylabel('mean violations per step')
    axs[0].legend(loc='best', fontsize=9)
    axs[0].grid(alpha=0.3)

    axs[1].set_title('Cumulative SLA violations')
    axs[1].set_xlabel('step')
    axs[1].set_ylabel('cumulative violations')
    axs[1].legend(loc='best', fontsize=9)
    axs[1].grid(alpha=0.3)

    axs[2].set_title('Resource allocation (moving avg)')
    axs[2].set_xlabel('step')
    axs[2].set_ylabel('PRBs')
    axs[2].set_ylim((0, PRBS_MAX))
    axs[2].legend(loc='best', fontsize=9)
    axs[2].grid(alpha=0.3)

    fig.savefig(OUTPUT_FILE, format='png', dpi=150)
    print(f'Plot saved to {OUTPUT_FILE}')