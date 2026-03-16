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
END = 20000

titles = ['Scenario 5']
scenarios = [4]

#algo_names = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL_97','KBRL_99']
#labels = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL 0.97', 'KBRL 0.99']
algo_names = ['PPO']    
labels = ['PPO']
SPAN = END - START

prbs_values = [100]

def mean_confidence_radius(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# subplot
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5), constrained_layout=True)
axs = np.array([axs])  # Ensure axs is always an array

for i, (j, title) in enumerate(zip(scenarios,titles)):
    axs[i].set_title(title)
    PRBS = prbs_values[i]
    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        data = False
        path = './results/scenario_{}/{}/'.format(j,algo)
        runs = 0
        violations = []
        resources = []
        # iterate over files
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            continue
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                print(f"Loading {filename} from {path}")
                histories = np.load(path + filename)
                _violations = histories['violation']
                _resources = histories['resources']
                print(f"  Data length: {len(_violations)}, required: {END}")
                if len(_violations) < END:
                    print(f"  Skipping (not enough data)")
                    continue
                violations.append(_violations[START:END].mean())
                resources.append(_resources[START:END].mean()/PRBS)
                print(f"  Added to plot")

        if len(violations) == 0:
            print(f"No data found for {algo} in scenario {j}")
            continue

        v, v_h = mean_confidence_radius(violations)

        r, r_h = mean_confidence_radius(resources)

        axs[i].errorbar(r, v, xerr = r_h, yerr = v_h, fmt='o', label = label)
    
    axs[i].set_xlim((0.4,1.))
    axs[i].set_ylim((0.,1.))
    axs[i].set_xlabel('Resource occupation')  # Add an x-label to the axes.
    axs[i].set_ylabel('SLA violations per stage')
    axs[i].grid()
    if i==0:
        axs[i].legend(loc='upper left')

fig.savefig('./figures/trained_figure.png', format='png')   
