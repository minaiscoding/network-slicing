#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot DQN results for Scenario 3 only
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# training results
WINDOW = 400
START = 0
END = 20000  # up to 20000 steps

# only DQN
algo_names = ['DQN']
labels = ['DQN']

SPAN = END - START

# only Scenario 3
scenario = 3  # 0-based index: 0,1,2
prbs_values = [200, 150, 100,100]
prbs = prbs_values[scenario]

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# --------------------- plot -------------------------------
dir_path = f'./results/scenario_{scenario}/'

# subplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5), constrained_layout=True)

for algo, label in zip(algo_names, labels):
    violations = np.empty([1])
    actions = np.empty([1])
    regret = np.empty([1])
    data = False
    proposal = False
    path = f'./results/scenario_{scenario}/{algo}/'
    runs = 0

    for filename in os.listdir(path):
        if filename.endswith(".npz"):
            histories = np.load(os.path.join(path, filename))
            _violations = histories['violation']
            _resources = histories['resources']
            if len(_violations) < END:
                continue
            _violations = _violations[START:END]
            _resources = _resources[START:END]
            runs += 1
            if not data:
                violations = movingaverage(_violations, WINDOW)
                regret = movingaverage(_violations.cumsum(), WINDOW)
                actions = movingaverage(_resources, WINDOW)
                if proposal:
                    accuracy = movingaverage(np.mean(histories['hits'], axis=0), WINDOW)
                data = True
            else:
                violations = np.vstack((violations, movingaverage(_violations, WINDOW)))
                regret = np.vstack((regret, movingaverage(_violations.cumsum(), WINDOW)))
                actions = np.vstack((actions, movingaverage(_resources, WINDOW)))
                if proposal:
                    accuracy = np.vstack((accuracy, movingaverage(np.mean(histories['hits'], axis=0), WINDOW)))

    print(f'Algorithm {algo}')
    
    # average over runs
    actions_mean = np.mean(actions, axis=0)
    actions_std = np.std(actions, axis=0)

    violations_mean = np.mean(violations, axis=0)
    violations_std = np.std(violations, axis=0)

    regret_mean = np.mean(regret, axis=0)
    regret_std = np.std(regret, axis=0)

    # plot results
    steps = np.arange(len(actions_mean[0:SPAN]))

    axs[2].set_title('Resource allocation')
    axs[2].plot(steps, actions_mean[0:SPAN])
    axs[2].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs),
                        actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color='#DDDDDD')
    axs[2].set_ylim((0, prbs))
    axs[2].set_xlabel('stages')
    axs[2].set_ylabel('PRBs')
    axs[2].grid()

    axs[0].set_title('SLA violations')
    axs[0].plot(steps, violations_mean[0:SPAN], label=label)
    axs[0].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs),
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color='#DDDDDD')
    axs[0].set_xlabel('stages')
    axs[0].set_ylabel('SLA violations')
    axs[0].legend(loc='best')
    axs[0].grid()

    axs[1].set_title('Cumulative SLA violations')
    axs[1].plot(steps, regret_mean[0:SPAN], label=label)
    axs[1].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs),
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color='#DDDDDD')
    axs[1].set_xlabel('stages')
    axs[1].set_ylabel('cumulative SLA violations')
    axs[1].set_ylim((0,15000))
    axs[1].grid()

fig.savefig('./figures/subplots_scenario3_dqn.png', format='png')