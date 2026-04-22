#!/usr/bin/env python3
"""
Quick profiler for experiments_ppo_lag.py.

Runs two passes:
  1. cProfile on 500 env steps directly (no OmniSafe overhead) to find hot spots
     in our own code: _compute_observation, _compute_reward, _compute_cost, env.step
  2. Saves a sorted profile report to profile_report.txt
"""

import cProfile
import pstats
import io
import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

# ── Build the env exactly as the training script would ──────────────────────
from config_loader import load_scenario
from scenario_creator import create_env_from_config
from experiments_ppo_lag import (
    RanSlicePPOLagEnv, PPOLagReportWrapper, _extract_sla,
    ENV_ID, _EMBB_IDX, _MMTC_IDX, _URLLC_IDX,
    STEPS_PER_SCENARIO,
)
import torch

N_STEPS  = 500
SCENARIO = 'low'

print(f"Profiling {N_STEPS} env steps on scenario '{SCENARIO}' ...")

cfg     = load_scenario('scenarios.yaml', SCENARIO)
rng     = np.random.default_rng(42)
raw_env = create_env_from_config(cfg, rng, penalty=10.0)

wrapper = PPOLagReportWrapper(
    raw_env,
    steps=N_STEPS + 100,
    control_steps=N_STEPS + 200,
    env_id='profile',
    path='./logs/',
    verbose=False,
    continuous_mode=True,
)
os.makedirs('./logs/', exist_ok=True)
wrapper.reset()

n_slices = wrapper.n_slices
n_prbs   = cfg.n_prbs
sla      = _extract_sla(cfg)

# Replicate the env logic inline so we profile our own methods
last_allocation  = np.full(n_slices, n_prbs // n_slices, dtype=float)
last_change_flag = 0.0

def _safe(x, ref):
    return float(x) / max(float(ref), 1e-9)

def compute_observation(info):
    obs         = np.zeros(n_slices * 4 + 1, dtype=float)
    l1_info     = info.get('l1_info', [])
    n_prbs_list = info.get('n_prbs', [n_prbs // n_slices] * n_slices)

    for si in range(n_slices):
        base     = si * 4
        prb_frac = float(n_prbs_list[si]) / max(n_prbs, 1) if si < len(n_prbs_list) else 0.0

        if si < len(l1_info) and isinstance(l1_info[si], dict):
            for ran_id, ran_info in l1_info[si].items():
                if not isinstance(ran_info, dict):
                    continue
                if si == _EMBB_IDX:
                    cbr_delay = ran_info.get('cbr_delay', 0.0)
                    cbr_th    = ran_info.get('cbr_th',    0.0)
                    vbr_th    = ran_info.get('vbr_th',    0.0)
                    cbr_q     = ran_info.get('cbr_queue', 0.0)
                    vbr_q     = ran_info.get('vbr_queue', 0.0)
                    obs[base]   = np.clip(_safe(cbr_delay, sla['embb_cbr_delay']), 0, 3) / 3
                    obs[base+1] = np.clip(_safe(cbr_th + vbr_th, sla['embb_cbr_th'] + sla['embb_vbr_th']), 0, 3) / 3
                    obs[base+2] = np.clip(_safe(cbr_q + vbr_q, sla['embb_cbr_queue'] + sla['embb_vbr_queue']), 0, 3) / 3
                elif si == _MMTC_IDX:
                    delay   = ran_info.get('delay',   0.0)
                    devices = ran_info.get('devices', 0.0)
                    obs[base]   = np.clip(_safe(delay, sla['mmtc_delay']), 0, 3) / 3
                    obs[base+1] = 1.0 - np.clip(_safe(devices, sla['mmtc_max_dev']), 0, 1)
                    obs[base+2] = 0.0
                elif si == _URLLC_IDX:
                    cbr_delay = ran_info.get('cbr_delay', 0.0)
                    cbr_th    = ran_info.get('cbr_th',    0.0)
                    vbr_th    = ran_info.get('vbr_th',    0.0)
                    cbr_q     = ran_info.get('cbr_queue', 0.0)
                    vbr_q     = ran_info.get('vbr_queue', 0.0)
                    obs[base]   = np.clip(_safe(cbr_delay, sla['urllc_cbr_delay']), 0, 3) / 3
                    obs[base+1] = np.clip(_safe(cbr_th + vbr_th, sla['urllc_cbr_th'] + sla['urllc_vbr_th']), 0, 3) / 3
                    obs[base+2] = np.clip(_safe(cbr_q + vbr_q, sla['urllc_cbr_queue'] + sla['urllc_vbr_queue']), 0, 3) / 3
        obs[base+3] = np.clip(prb_frac, 0.0, 1.0)

    obs[n_slices * 4] = last_change_flag
    return obs


def run_steps():
    global last_allocation, last_change_flag
    for _ in range(N_STEPS):
        # Random action
        act = np.abs(np.random.randn(n_slices + 1).astype(np.float32))
        change_flag = float(act[n_slices])
        alloc_raw   = act[:n_slices]
        total       = alloc_raw.sum()
        if total > 0:
            alloc_raw /= total
        new_alloc = np.array([int(np.floor(a * n_prbs)) for a in alloc_raw], dtype=int)

        if change_flag < 0.5:
            alloc_prbs = last_allocation.copy().astype(int)
        else:
            alloc_prbs = new_alloc

        result = wrapper.step(alloc_prbs)
        if len(result) == 5:
            _, _, terminated, truncated, info = result
        else:
            _, _, done, info = result
            terminated = done

        compute_observation(info)

        if change_flag >= 0.5:
            last_change_flag = 1.0 if not np.array_equal(alloc_prbs, last_allocation.astype(int)) else 0.0
            last_allocation  = alloc_prbs.copy().astype(float)
        else:
            last_change_flag = 0.0

        if terminated:
            wrapper.reset()


# ── Run cProfile ─────────────────────────────────────────────────────────────
pr = cProfile.Profile()
pr.enable()
run_steps()
pr.disable()

# Print top-30 by cumulative time
buf = io.StringIO()
ps  = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
ps.print_stats(30)
report = buf.getvalue()
print(report)

# Also save full report
with open('profile_report.txt', 'w') as f:
    buf2 = io.StringIO()
    pstats.Stats(pr, stream=buf2).sort_stats('cumulative').print_stats()
    f.write(buf2.getvalue())
print("Full report saved to profile_report.txt")

# ── Also dump top-30 by tottime (time IN the function, no children) ──────────
print("\n--- Top 30 by tottime (self time, no children) ---")
buf3 = io.StringIO()
pstats.Stats(pr, stream=buf3).sort_stats('tottime').print_stats(30)
print(buf3.getvalue())
