#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO on RAN slicing with cycling curriculum.

Results (history_*.npz) are saved inside pipelines/ppo/.

Usage:
    python pipelines/ppo/train.py
    python pipelines/ppo/train.py --runs 1 --sequential --total-steps 10000
"""

import os
import sys
import argparse
import concurrent.futures as cf

# Ensure project root is on the path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from algorithms.ppo import TrainerPPO

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO on RAN slicing")
    parser.add_argument("--runs",               type=int,   default=30)
    parser.add_argument("--processes",           type=int,   default=4)
    parser.add_argument("--sequential",          action="store_true")
    parser.add_argument("--steps-per-scenario",  type=int,   default=20000)
    parser.add_argument("--total-steps",         type=int,   default=500000)
    parser.add_argument("--penalty",             type=float, default=100.0)
    parser.add_argument("--device",              type=str,   default="cuda:0")
    args = parser.parse_args()

    results_path = os.path.join(PIPELINE_DIR, '')  # trailing separator

    trainer = TrainerPPO(
        results_path=results_path,
        total_steps=args.total_steps,
        steps_per_scenario=args.steps_per_scenario,
        penalty=args.penalty,
        device=args.device,
    )

    run_list = list(range(args.runs))
    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            list(E.map(trainer.train, run_list))
