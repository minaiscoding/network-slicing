#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed October 2020

@author: juanjosealcaraz

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# test results
START = 0
END = None

BASE_PATH = './results/scenario_comparison/CPO'
titles = ['Low', 'Medium', 'Congested']
scenarios = ['low', 'medium', 'congested']
prbs_values = [200, 150, 100]

def mean_confidence_radius(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(a[0]), 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def load_run_points(base_path, scenario, prbs):
    violations = []
    resources = []

    for filename in sorted(os.listdir(base_path)):
        if not filename.endswith(f'_{scenario}.npz'):
            continue

        histories = np.load(os.path.join(base_path, filename))
        run_violations = histories['violation']
        run_resources = histories['resources']

        max_end = len(run_violations)
        if END is None:
            end = min(max_end, len(run_resources))
        else:
            end = min(END, max_end, len(run_resources))
        start = min(START, end)
        if start >= end:
            continue

        violations.append(run_violations[start:end].mean())
        resources.append(run_resources[start:end].mean() / prbs)

    return np.array(resources), np.array(violations)

# subplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5), constrained_layout=True)

for i, (scenario, title) in enumerate(zip(scenarios, titles)):
    axs[i].set_title(title)
    prbs = prbs_values[i]
    resources, violations = load_run_points(BASE_PATH, scenario, prbs)

    axs[i].scatter(resources, violations, alpha=0.7, s=28, color='C0', label='Runs')

    v_mean, v_radius = mean_confidence_radius(violations)
    r_mean, r_radius = mean_confidence_radius(resources)
    if not np.isnan(v_mean) and not np.isnan(r_mean):
        axs[i].errorbar(
            r_mean,
            v_mean,
            xerr=r_radius,
            yerr=v_radius,
            fmt='o',
            markersize=7,
            color='C3',
            ecolor='C3',
            elinewidth=1.5,
            capsize=4,
            label='Mean +/- 95% CI',
        )
    
    axs[i].set_xlim((0.4,1.))
    axs[i].set_ylim((0.,1.))
    axs[i].set_xlabel('Resource occupation')  # Add an x-label to the axes.
    axs[i].set_ylabel('SLA violations per stage')
    axs[i].grid()
    axs[i].text(
        0.98,
        0.04,
        f'{len(resources)} runs',
        transform=axs[i].transAxes,
        ha='right',
        va='bottom',
    )
    if i == 0:
        axs[i].legend(loc='upper left')

fig.savefig('./figures/trained_figure.png', format='png')   