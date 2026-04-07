#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_trace_db.py

Generate a 3-hour trace database of traffic arrivals, SNR values, and
per-slice PRB demand for each scenario (low, medium, congested).

The trace is produced by running the simulator with a fixed RNG seed and
equal-split PRB allocation so that realistic queue dynamics are captured.

For each RL step (50 slots = 5 ms) the following per-slice quantities are
recorded:
    traffic_arrivals  – total new bits arriving across all UEs
    avg_snr           – mean estimated SNR across active UEs
    queue_depth       – total queued bits at end of step
    n_active_ues      – number of active UEs
    prb_demand        – ideal PRBs to drain all queued data given channel

Output
------
    datasets/trace_db_{scenario}.npz

Usage
-----
    python generate_trace_db.py                       # all scenarios
    python generate_trace_db.py --scenario low        # single scenario
    python generate_trace_db.py --duration 1800       # 30 minutes
"""

import os
import argparse
import numpy as np
from numpy.random import default_rng
from itertools import count

from config_loader import load_scenario
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC, SliceL1URLLC
from slice_ran import SliceRANmMTC, SliceRANeMBB, SliceRANURLC
from schedulers import ProportionalFair
from channel_models import (SINRSelectiveFading, MCSCodeset,
                            MCS_CODESET_EMBB, MCS_CODESET_URLLC)
from scenario_creator import (
    _resolve_traffic, _resolve_sla,
    state_variables_embb, state_variables_mmtc, state_variables_urllc,
)

# ── Constants ──
SLOT_LENGTH = 1e-4          # 0.1 ms per slot
SLOTS_PER_STEP = 50         # RL decision interval
SYM_PER_PRB = 158           # symbols per PRB per slot
SEED = 42                   # reproducible traces
DEFAULT_DURATION_S = 10800  # 3 hours


# ── PRB demand computation helpers ──

def compute_prb_demand_embb(ues, mcs_codeset, error_bound=0.1):
    """
    Compute the ideal total PRB demand across all UEs in a slice
    to drain every queue in a single slot.

    For each UE:
        bits_per_prb = sym_per_prb * bits_per_symbol(SNR)
        rx_prob      = codeset.estimate_rx_prob(mcs, ue_snr)
        prbs_needed  = ceil(queue / (bits_per_prb * max(rx_prob, 0.01)))
    """
    total_prb_demand = 0
    for ue in ues:
        if ue.queue <= 0:
            continue
        mcs, bits_per_sym = mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
        bits_per_prb = SYM_PER_PRB * bits_per_sym
        rx_prob = mcs_codeset.estimate_rx_prob(mcs, ue.e_snr)
        effective_rate = bits_per_prb * max(rx_prob, 0.01)
        prbs_needed = int(np.ceil(ue.queue / effective_rate))
        total_prb_demand += prbs_needed
    return total_prb_demand


def compute_prb_demand_mmtc(slice_l1):
    """
    For mMTC the demand is simply the number of active devices
    (each device needs 1 PRB = 1 NB-IoT carrier).
    """
    return slice_l1.n_users


# ── Environment builder (no gym wrapper, direct NodeB) ──

def build_node_b(cfg, rng, slots_per_step=SLOTS_PER_STEP):
    """
    Build a NodeB + supporting objects from a ScenarioConfig.
    Returns (node_b, slices_l1, mcs_codesets) where mcs_codesets
    is a list aligned with slices_l1 giving the codeset for each slice.
    """
    time_per_step = slots_per_step * SLOT_LENGTH

    (CBR_desc, VBR_desc, MTC_desc,
     URLLC_CBR_desc, URLLC_VBR_desc) = _resolve_traffic(cfg.traffic_config)
    SLA_embb, SLA_mmtc, SLA_urllc = _resolve_sla(cfg.sla_config)

    n_prbs  = cfg.n_prbs
    n_embb  = cfg.n_embb
    n_mmtc  = cfg.n_mmtc
    n_urllc = cfg.n_urllc

    # normalization constants (needed by SliceRAN constructors)
    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th':      10e6 * time_per_step,
        'cbr_queue':   10e4 * slots_per_step,
        'cbr_snr':     35 * slots_per_step,
        'cbr_delay':   20,
        'vbr_traffic': 5e6 * time_per_step,
        'vbr_th':      10e6 * time_per_step,
        'vbr_queue':   10e4 * slots_per_step,
        'vbr_snr':     35 * slots_per_step,
        'vbr_delay':   20,
    }
    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay':   100 * slots_per_step,
    }
    norm_const_urllc = {
        'cbr_traffic': 2e6 * time_per_step,
        'cbr_th':      5e6 * time_per_step,
        'cbr_queue':   5e3 * slots_per_step,
        'cbr_snr':     35 * slots_per_step,
        'cbr_delay':   10,
        'vbr_traffic': 2e6 * time_per_step,
        'vbr_th':      7e6 * time_per_step,
        'vbr_queue':   5e3 * slots_per_step,
        'vbr_snr':     35 * slots_per_step,
        'vbr_delay':   10,
    }

    user_counter = count()

    def new_slice_embb(id):
        ue_profiles = cfg.traffic_config.get('embb', {}).get('ue_profiles', None)
        return SliceRANeMBB(rng, user_counter, id, SLA_embb,
                            CBR_desc, VBR_desc,
                            state_variables_embb, norm_const_embb, slots_per_step,
                            ue_profiles=ue_profiles)

    def new_slice_mmtc(id):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_desc,
                            state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_urllc(id):
        ue_profiles = cfg.traffic_config.get('urllc', {}).get('ue_profiles', None)
        return SliceRANURLC(rng, user_counter, id, SLA_urllc,
                            URLLC_CBR_desc, URLLC_VBR_desc,
                            state_variables_urllc, norm_const_urllc, slots_per_step,
                            ue_profiles=ue_profiles)

    snr_generator     = SINRSelectiveFading(rng, 'macro_cell_urban_2GHz', n_prbs=n_prbs)
    mcs_codeset       = MCSCodeset(MCS_CODESET_EMBB)
    mcs_codeset_urllc = MCSCodeset(MCS_CODESET_URLLC)
    scheduler         = ProportionalFair(mcs_codeset)
    scheduler_urllc   = ProportionalFair(mcs_codeset_urllc)

    slices_l1 = []
    mcs_codesets = []   # aligned with slices_l1

    for id in range(n_embb):
        sl = SliceL1eMBB(rng, snr_generator, 20, [new_slice_embb(id)], scheduler)
        slices_l1.append(sl)
        mcs_codesets.append(mcs_codeset)

    for id in range(n_mmtc):
        sl = SliceL1mMTC(5, [new_slice_mmtc(id)])
        slices_l1.append(sl)
        mcs_codesets.append(None)  # mMTC uses carrier count, not MCS

    for id in range(n_urllc):
        sl = SliceL1URLLC(rng, snr_generator, 15, [new_slice_urllc(id)], scheduler_urllc)
        slices_l1.append(sl)
        mcs_codesets.append(mcs_codeset_urllc)

    node = NodeB(slices_l1, slots_per_step, n_prbs)
    return node, slices_l1, mcs_codesets


# ── Main trace generation ──

def generate_trace(scenario_name, duration_s=DEFAULT_DURATION_S, seed=SEED, n_steps_override=None):
    """
    Run the simulator for *duration_s* seconds and record per-step metrics.
    If n_steps_override is given, ignore duration_s and use that directly.

    Returns dict with arrays of shape (n_steps, n_slices):
        traffic_arrivals, avg_snr, queue_depth, n_active_ues, prb_demand
    Plus scalar metadata.
    """
    rng = default_rng(seed)
    cfg = load_scenario('scenarios.yaml', scenario_name)
    node, slices_l1, mcs_codesets = build_node_b(cfg, rng)

    n_slices = len(slices_l1)
    n_prbs   = cfg.n_prbs
    if n_steps_override is not None:
        n_steps = n_steps_override
        total_slots = n_steps * SLOTS_PER_STEP
        duration_s = total_slots * SLOT_LENGTH
    else:
        total_slots = int(duration_s / SLOT_LENGTH)
        n_steps = total_slots // SLOTS_PER_STEP

    print(f"[{scenario_name}] Generating trace: {duration_s}s = {total_slots} slots = {n_steps} steps, "
          f"{n_slices} slices, {n_prbs} PRBs")

    # Pre-allocate output arrays
    traffic_arrivals = np.zeros((n_steps, n_slices), dtype=np.float32)
    avg_snr          = np.zeros((n_steps, n_slices), dtype=np.float32)
    queue_depth      = np.zeros((n_steps, n_slices), dtype=np.float32)
    n_active_ues     = np.zeros((n_steps, n_slices), dtype=np.int32)
    prb_demand       = np.zeros((n_steps, n_slices), dtype=np.float32)

    # Equal-split allocation for driving the simulation
    prbs_per_slice = n_prbs // n_slices
    equal_action = np.full(n_slices, prbs_per_slice, dtype=int)
    # Give remainder to first slice
    equal_action[0] += n_prbs - equal_action.sum()

    for step_idx in range(n_steps):
        # Execute one RL step
        state, info = node.step(equal_action)

        # Extract per-slice metrics from l1_info
        l1_info = info.get('l1_info', [])

        for s_idx, sl in enumerate(slices_l1):
            slice_info = l1_info[s_idx] if s_idx < len(l1_info) else {}

            if sl.type in ('eMBB', 'URLLC'):
                # Aggregate across UEs
                total_traffic = 0.0
                total_snr = 0.0
                total_queue = 0.0
                n_ues = len(sl.ues)

                for ue in sl.ues:
                    total_traffic += getattr(ue, 'new_bits', 0)
                    total_snr += getattr(ue, 'e_snr', 0.0)
                    total_queue += ue.queue

                traffic_arrivals[step_idx, s_idx] = total_traffic
                avg_snr[step_idx, s_idx] = (total_snr / n_ues) if n_ues > 0 else 0.0
                queue_depth[step_idx, s_idx] = total_queue
                n_active_ues[step_idx, s_idx] = n_ues

                # Compute ideal PRB demand
                prb_demand[step_idx, s_idx] = compute_prb_demand_embb(
                    sl.ues, mcs_codesets[s_idx])

            elif sl.type == 'mMTC':
                # mMTC metrics from slice_ran info
                n_active = sl.n_users
                n_active_ues[step_idx, s_idx] = n_active
                prb_demand[step_idx, s_idx] = compute_prb_demand_mmtc(sl)

                # Extract traffic/queue from ran_info
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            n_active_ues[step_idx, s_idx] = ran_info.get('devices', n_active)

        # Progress reporting every 10%
        report_interval = max(1, n_steps // 10)
        if (step_idx + 1) % report_interval == 0 or step_idx == 0:
            pct = 100.0 * (step_idx + 1) / n_steps
            print(f"  [{scenario_name}] Step {step_idx + 1}/{n_steps} ({pct:.1f}%)")

    trace = {
        'traffic_arrivals': traffic_arrivals,
        'avg_snr': avg_snr,
        'queue_depth': queue_depth,
        'n_active_ues': n_active_ues,
        'prb_demand': prb_demand,
        # Metadata
        'n_slices': n_slices,
        'n_prbs': n_prbs,
        'n_steps': n_steps,
        'duration_s': duration_s,
        'slots_per_step': SLOTS_PER_STEP,
        'seed': seed,
        'scenario': scenario_name,
    }

    return trace


def save_trace(trace, output_dir='./datasets'):
    """Save trace to a compressed .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    scenario = trace['scenario']
    path = os.path.join(output_dir, f'trace_db_{scenario}.npz')
    np.savez_compressed(path, **trace)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {path} ({size_mb:.1f} MB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate trace databases for perfect forecast"
    )
    parser.add_argument("--scenario", type=str, default=None,
                        help="Single scenario name (default: all)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of RL steps to generate (overrides --duration)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help="Simulation duration in seconds (default: 10800 = 3h)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="RNG seed for reproducibility")
    parser.add_argument("--output", type=str, default='./datasets',
                        help="Output directory")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ['low', 'medium', 'congested']

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"  Generating trace for: {scenario_name}")
        print(f"{'='*60}")
        trace = generate_trace(scenario_name, duration_s=args.duration, seed=args.seed,
                               n_steps_override=args.steps)
        save_trace(trace, output_dir=args.output)

        # Print summary statistics
        print(f"\n  Summary for {scenario_name}:")
        print(f"    Steps: {trace['n_steps']}")
        print(f"    Slices: {trace['n_slices']}")
        for s in range(trace['n_slices']):
            print(f"    Slice {s}:")
            print(f"      Avg traffic arrivals: {trace['traffic_arrivals'][:, s].mean():.0f} bits/step")
            print(f"      Avg SNR: {trace['avg_snr'][:, s].mean():.1f} dB")
            print(f"      Avg queue depth: {trace['queue_depth'][:, s].mean():.0f} bits")
            print(f"      Avg PRB demand: {trace['prb_demand'][:, s].mean():.1f} PRBs")
            print(f"      Max PRB demand: {trace['prb_demand'][:, s].max():.0f} PRBs")


if __name__ == '__main__':
    main()
