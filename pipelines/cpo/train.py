#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CPO on RAN slicing with sequential curriculum.

Results (history_*.npz) are saved inside pipelines/cpo/.

Usage:
    python pipelines/cpo/train.py
    python pipelines/cpo/train.py --runs 1 --sequential --total-steps 10000
"""

import os
import sys
import argparse
import concurrent.futures as cf

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from algorithms.cpo import TrainerCPO

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CPO on RAN slicing")
    parser.add_argument("--runs",              type=int,   default=1)
    parser.add_argument("--processes",          type=int,   default=1)
    parser.add_argument("--sequential",         action="store_true")
    parser.add_argument("--total-steps",        type=int,   default=360000)
    parser.add_argument("--cost-limit",         type=float, default=100.0)
    parser.add_argument("--steps-per-episode",  type=int,   default=1800)
    parser.add_argument("--steps-per-epoch",    type=int,   default=3600)
    parser.add_argument("--penalty",            type=float, default=10.0)
    parser.add_argument("--device",             type=str,   default="cpu")
    args = parser.parse_args()

    results_path = os.path.join(PIPELINE_DIR, '')

    trainer = TrainerCPO(
        results_path=results_path,
        total_steps=args.total_steps,
        steps_per_episode=args.steps_per_episode,
        penalty=args.penalty,
        cost_limit=args.cost_limit,
        steps_per_epoch=args.steps_per_epoch,
        device=args.device,
    )

    run_list = list(range(args.runs))
    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            list(E.map(trainer.train, run_list))
