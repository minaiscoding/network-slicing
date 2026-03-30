#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: juanjosealcaraz

Evaluates TD3 in network-slicing scenarios.
'''

import os
import concurrent.futures as cf
from numpy.random import default_rng
from scenario_creator import create_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3
from wrapper import ReportWrapper

SCENARIO = 4
RUNS = 2
PROCESSES = 4 # 30 if enough threads 
TRAIN_STEPS = 20000
EVALUATION_STEPS = 5000
CONTROL_STEPS = 20000
PENALTY = 1000
SLOTS_PER_STEP = 500
PRBS = [200, 150, 100, 70]
run_list = list(range(RUNS))

class Evaluator():
    def __init__(self, scenario=SCENARIO):
        self.scenario = scenario
        self.train_path = f'./results/scenario_{scenario}/TD3/'
        self.test_path = f'./results/scenario_{scenario}/TD3_t/'
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
    
    def evaluate(self, run_id):
        rng = default_rng(seed=run_id)

        # ----------------- Training -----------------
        env = create_env(rng, self.scenario, penalty=PENALTY)

        node_env = ReportWrapper(
            env,
            steps=TRAIN_STEPS,
            control_steps=CONTROL_STEPS,
            env_id=run_id,
            path=self.train_path,
            verbose=False
        )
        print('Wrapped environment created for training')
        vec_env = make_vec_env(lambda: node_env, n_envs=1)
        print('Vectorized environment created for training')

        agent = TD3('MlpPolicy', vec_env, verbose=True, device='cpu')
        agent.learn(total_timesteps=TRAIN_STEPS)
        print('Training done!')
        node_env.save_results()
        print('Training results saved.')

        # ----------------- Evaluation -----------------
        print('Test starts...')
        env = create_env(rng, self.scenario, penalty=PENALTY)
        node_env = ReportWrapper(
            env,
            steps=EVALUATION_STEPS,
            control_steps=CONTROL_STEPS,
            env_id=run_id,
            path=self.test_path,
            verbose=False
        )
        print('Wrapped environment created for evaluation')

        # Reset returns (obs, info)
        obs, _ = node_env.reset()
        state = None
        for _ in range(EVALUATION_STEPS):
            # Only pass observation array to predict
            action, state = agent.predict(obs, state=state, deterministic=True)
            obs, reward, terminated, truncated, info = node_env.step(action)
            if terminated or truncated:
                obs, _ = node_env.reset()
                state = None

        print('Evaluation done')
        node_env.save_results()
        print('Evaluation results saved.')


if __name__=='__main__':
    evaluator = Evaluator()
    # ################################################################
    # # use this code for sequential execution
    for run in run_list:
        evaluator.evaluate(run)
    # ################################################################

    # ################################################################
    # use this code for parallel execution
    #with cf.ProcessPoolExecutor(PROCESSES) as E:
     #   list(E.map(evaluator.evaluate, run_list))
    # ################################################################
