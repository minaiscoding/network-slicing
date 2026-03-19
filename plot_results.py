#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot PPO results with moving average
"""
import numpy as np
import matplotlib.pyplot as plt

# training results
WINDOW = 400
START = 0
MAX_EPOCHS = 30000  # maximum epochs to display (will use all available if less)
INPUT_FILE = './results/scenario_4/PPO/history_1.npz'
OUTPUT_FILE = './figures/subplots_scenario_ppo.png'

PRBS_MAX = 100

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__ == '__main__':
    if not INPUT_FILE.endswith('.npz'):
        raise ValueError('INPUT_FILE must point to a .npz file')

    histories = np.load(INPUT_FILE)
    violations = histories['violation']
    resources = histories['resources']

    end = min(len(violations), MAX_EPOCHS)
    if end <= START:
        raise RuntimeError('No data available in the selected START/MAX_EPOCHS range.')

    violations = violations[START:end]
    resources = resources[START:end]

    effective_window = min(WINDOW, len(violations))
    violations_ma = movingaverage(violations, effective_window)
    regret_ma = movingaverage(violations.cumsum(), effective_window)
    resources_ma = movingaverage(resources, effective_window)

    steps = np.arange(len(violations_ma))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5), constrained_layout=True)

    axs[2].set_title('Resource allocation')
    axs[2].plot(steps, resources_ma)
    axs[2].set_ylim((0, PRBS_MAX))
    axs[2].set_xlabel('stages')
    axs[2].set_ylabel('PRBs')
    axs[2].grid()

    axs[0].set_title('SLA violations')
    axs[0].plot(steps, violations_ma, label='PPO history_1')
    axs[0].set_xlabel('stages')
    axs[0].set_ylabel('SLA violations')
    axs[0].legend(loc='best')
    axs[0].grid()

    axs[1].set_title('Cumulative SLA violations')
    axs[1].plot(steps, regret_ma, label='PPO history_1')
    axs[1].set_xlabel('stages')
    axs[1].set_ylabel('cumulative SLA violations')
    axs[1].grid()

    fig.savefig(OUTPUT_FILE, format='png')
    print(f'Loaded: {INPUT_FILE}')
    print(f'Plot saved to {OUTPUT_FILE}')
