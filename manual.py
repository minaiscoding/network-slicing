#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual interactive oracle for network slicing.

Enter PRB allocations each step. If a violation occurs you can revert
to the previous (pre-step) state and try a different allocation.

Usage:
    python manual.py
    python manual.py --scenario medium --steps 200
"""

import copy
import argparse
import numpy as np
from numpy.random import default_rng

from config_loader import load_scenario
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper

SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC']

# Observation indices (normalised by the wrapper to ~[-0.5, 0.5])
OBS_LABELS = [
    'eMBB cbr_traffic', 'eMBB cbr_th', 'eMBB cbr_queue', 'eMBB cbr_snr', 'eMBB cbr_delay',
    'eMBB vbr_traffic', 'eMBB vbr_th', 'eMBB vbr_queue', 'eMBB vbr_snr', 'eMBB vbr_delay',
    'mMTC devices',     'mMTC avg_rep','mMTC delay',
    'URLLC cbr_traffic','URLLC cbr_th','URLLC cbr_queue','URLLC cbr_snr','URLLC cbr_delay',
    'URLLC vbr_traffic','URLLC vbr_th','URLLC vbr_queue','URLLC vbr_snr','URLLC vbr_delay',
]


# ── Snapshot helpers ────────────────────────────────────────────────────

def save_snapshot(env, wrapper):
    """Deep-copy the environment core + wrapper counters."""
    return {
        'node_b': copy.deepcopy(env.unwrapped.node_b),
        'step_counter': wrapper.step_counter,
        'violation_history': wrapper.violation_history.copy(),
        'reward_history': wrapper.reward_history.copy(),
        'action_history': wrapper.action_history.copy(),
        'violation_per_slice_history': wrapper.violation_per_slice_history.copy(),
        'resource_per_slice_history': wrapper.resource_per_slice_history.copy(),
    }


def restore_snapshot(env, wrapper, snap):
    """Restore environment + wrapper state from a snapshot."""
    env.unwrapped.node_b = copy.deepcopy(snap['node_b'])
    wrapper.step_counter = snap['step_counter']
    wrapper.violation_history = snap['violation_history'].copy()
    wrapper.reward_history = snap['reward_history'].copy()
    wrapper.action_history = snap['action_history'].copy()
    wrapper.violation_per_slice_history = snap['violation_per_slice_history'].copy()
    wrapper.resource_per_slice_history = snap['resource_per_slice_history'].copy()


# ── Display helpers ─────────────────────────────────────────────────────

def print_obs(obs):
    """Print observation values with labels."""
    print("  Observations:")
    for i, val in enumerate(obs):
        label = OBS_LABELS[i] if i < len(OBS_LABELS) else f'obs[{i}]'
        bar = '+' * int(max(val + 0.5, 0) * 20) if val > -0.5 else ''
        print(f"    {label:>22s}: {val:+.3f}  {bar}")


def print_violations(info):
    """Print per-slice violation details."""
    per_slice = info.get('violations', [])
    total = info.get('total_violations', 0)
    parts = []
    for i, name in enumerate(SLICE_NAMES):
        v = int(per_slice[i]) if i < len(per_slice) else 0
        parts.append(f"{name}={v}")
    print(f"  Violations: {' | '.join(parts)}  (total={total})")


def read_action(n_slices, n_prbs, last_action):
    """
    Read PRB allocation from the user.

    Special commands:
        r / revert  — revert to previous state (returns None)
        q / quit    — exit (raises SystemExit)
        <enter>     — repeat last action
    """
    while True:
        prompt = f"  PRBs [{'/'.join(SLICE_NAMES)}, sum<={n_prbs}]: "
        raw = input(prompt).strip().lower()

        if raw in ('q', 'quit', 'exit'):
            raise SystemExit

        if raw in ('r', 'revert'):
            return None  # signal revert

        if raw == '' and last_action is not None:
            return last_action.copy()

        try:
            vals = list(map(int, raw.split()))
            if len(vals) != n_slices:
                print(f"    Need {n_slices} values, got {len(vals)}")
                continue
            if any(v < 0 for v in vals):
                print("    Values must be >= 0")
                continue
            if sum(vals) > n_prbs:
                print(f"    Sum {sum(vals)} exceeds {n_prbs}")
                continue
            return np.array(vals, dtype=int)
        except ValueError:
            print("    Enter space-separated integers, 'r' to revert, or 'q' to quit")


# ── Main loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manual interactive oracle with revert")
    parser.add_argument("--scenario", default="medium", choices=['low', 'medium', 'congested'])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto", nargs='+', type=int, metavar='PRB',
                        help="Run all steps with a fixed allocation, e.g. --auto 60 30 60")
    args = parser.parse_args()

    rng = default_rng(seed=args.seed)
    cfg = load_scenario('scenarios.yaml', args.scenario)
    n_prbs = cfg.n_prbs

    raw_env = create_env_from_config(cfg, rng, penalty=150)
    env = ReportWrapper(
        raw_env,
        steps=args.steps,
        control_steps=args.steps + 1,
        env_id="manual",
        path=f'./results/manual/{args.scenario}/',
        verbose=False,
    )

    obs, info = env.reset()
    n_slices = env.n_slices

    # Validate --auto if provided
    if args.auto is not None:
        if len(args.auto) != n_slices:
            parser.error(f"--auto needs {n_slices} values, got {len(args.auto)}")
        if sum(args.auto) > n_prbs:
            parser.error(f"--auto sum {sum(args.auto)} exceeds {n_prbs} PRBs")
        if any(v < 0 for v in args.auto):
            parser.error("--auto values must be >= 0")

    # ── Auto mode ───────────────────────────────────────────────────
    if args.auto is not None:
        fixed_action = np.array(args.auto, dtype=int)
        alloc_str = '  '.join(f"{SLICE_NAMES[i]}={fixed_action[i]}" for i in range(n_slices))
        print("=" * 65)
        print("  AUTO MODE — fixed PRB allocation")
        print(f"  Scenario: {args.scenario}  |  PRBs: {n_prbs}  |  Steps: {args.steps}")
        print(f"  Allocation: {alloc_str}")
        print("=" * 65)

        total_violations = 0
        per_slice_total = np.zeros(n_slices, dtype=int)
        rewards = []

        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(fixed_action)
            step_v = info.get('total_violations', 0)
            per_slice_v = info.get('violations', np.zeros(n_slices, dtype=int))
            total_violations += step_v
            per_slice_total += np.array(per_slice_v, dtype=int)
            rewards.append(reward)

            if step % 100 == 0 or step_v > 0:
                if step_v > 0:
                    culprits = [SLICE_NAMES[i] for i in range(n_slices) if i < len(per_slice_v) and per_slice_v[i] > 0]
                    tag = f" ⚠ VIOLATION [{', '.join(culprits)}]"
                else:
                    tag = ""
                print(f"  step {step:4d}  reward={reward:+.1f}  violations={step_v}{tag}")

            if terminated or truncated:
                print(f"  Episode finished at step {step}.")
                break

        import os
        os.makedirs(f'./results/manual/{args.scenario}/', exist_ok=True)
        env.save_results()

        print(f"\n{'='*65}")
        print(f"  AUTO RESULTS — {step + 1} steps")
        for i, name in enumerate(SLICE_NAMES):
            pct = per_slice_total[i] / (step + 1) * 100
            print(f"    {name:>5s} violations: {per_slice_total[i]:5d}  ({pct:.2f}%)")
        print(f"    Total violations: {total_violations:5d}  ({total_violations / (step + 1) * 100:.2f}%)")
        print(f"    Mean reward: {np.mean(rewards):.1f}")
        print(f"  Results saved to ./results/manual/{args.scenario}/")
        print(f"{'='*65}")
        return

    # ── Interactive mode ────────────────────────────────────────────
    print("=" * 65)
    print("  MANUAL ORACLE — interactive PRB allocation")
    print(f"  Scenario: {args.scenario}  |  PRBs: {n_prbs}  |  Slices: {n_slices}")
    print(f"  Max steps: {args.steps}")
    print("-" * 65)
    print("  Commands:")
    print("    <eMBB> <mMTC> <URLLC>  — allocate PRBs (e.g. 80 10 60)")
    print("    <enter>                 — repeat last allocation")
    print("    r / revert              — undo last step (revert on violation)")
    print("    q / quit                — stop and save")
    print("=" * 65)

    last_action = None
    prev_snapshot = None
    step = 0
    total_violations = 0
    reverted = False

    while step < args.steps:
        print(f"\n{'─'*65}")
        print(f"  Step {step}/{args.steps}   |   Total violations so far: {total_violations}")
        if last_action is not None:
            alloc_str = '  '.join(f"{SLICE_NAMES[i]}={last_action[i]}" for i in range(n_slices))
            print(f"  Last allocation: {alloc_str}")
        print_obs(obs)

        action = read_action(n_slices, n_prbs, last_action)

        # Handle revert request
        if action is None:
            if prev_snapshot is None:
                print("  ⚠  Nothing to revert to (first step).")
                continue
            restore_snapshot(raw_env, env, prev_snapshot)
            obs = env.obs
            step = env.step_counter
            print("  ↩  Reverted to previous state. Try a different allocation.")
            reverted = True
            continue

        # Save snapshot before stepping
        prev_snapshot = save_snapshot(raw_env, env)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_violations = info.get('total_violations', 0)
        total_violations += step_violations
        last_action = action.copy()
        step += 1
        reverted = False

        # Display result
        alloc_str = '  '.join(f"{SLICE_NAMES[i]}={action[i]}" for i in range(n_slices))
        print(f"\n  → Allocation: {alloc_str}  |  Reward: {reward:.1f}")
        print_violations(info)

        if step_violations > 0:
            print("  ⚠  VIOLATION!  Type 'r' to revert and try a different allocation.")

        if terminated or truncated:
            print("\n  Episode finished.")
            break

    # Save results
    import os
    os.makedirs(f'./results/manual/{args.scenario}/', exist_ok=True)
    env.save_results()
    print(f"\n{'='*65}")
    print(f"  Done — {step} steps, {total_violations} total violations")
    print(f"  Results saved to ./results/manual/{args.scenario}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()