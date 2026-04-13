#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test file to simulate overlapping gNodeBs with multiple UEs and track:
- UE throughput
- UE delay / wait time
- queue evolution
- SINR / SNR
- serving gNB distribution
- connected / disconnected users

This script is intentionally practical and standalone around your current codebase.
It uses the existing NodeB, slices, scheduler, channel model, and the
MultiGNBWrapper public API.

Important note:
The uploaded MultiGNBWrapper.step() currently contains an issue where
`current_id` and `target_id` are referenced before being defined.
This test file avoids that broken path by using a safe "no-handover" stepping
routine built on the wrapper internals, so you can still run overlap tests and
collect KPIs right now.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from channel_models import SINRSelectiveFading, MCSCodeset
from multi_gnb_wrapper import MultiGNBWrapper
from node_b import NodeB
from schedulers import ProportionalFair
from senario_multi_gnodeb import scenarios
from slice_l1 import SliceL1eMBB, SliceL1mMTC, SliceL1URLLC
from slice_ran import SliceRANeMBB, SliceRANmMTC, SliceRANURLC


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    scenario_name: str = "senario3gb"
    seed: int = 42
    n_ues: int = 120
    n_steps: int = 300
    slots_per_step: int = 10
    step_dt: float = 1e-3
    n_prbs_default: int = 150
    save_dir: str = "test_outputs_overlap"
    add_plots: bool = True


# -----------------------------------------------------------------------------
# Slice / gNB factory
# -----------------------------------------------------------------------------

def build_embb_slice(rng, user_counter, slice_id: int, slots_per_step: int, n_prbs: int):
    cbr_desc = {
        "lambda": 2.0 / 60.0,
        "t_mean": 30.0,
        "bit_rate": 1_000_000,
    }
    vbr_desc = {
        "lambda": 5.0 / 60.0,
        "t_mean": 20.0,
        "p_size": 1000,
        "b_size": 500,
        "b_rate": 2,
    }
    sla = {
        "cbr_th": 8e6,
        "cbr_prb": 20,
        "cbr_queue": 2e5,
        "vbr_th": 10e6,
        "vbr_prb": 30,
        "vbr_queue": 3e5,
    }
    state_variables = [
        "cbr_traffic", "cbr_th", "cbr_prb", "cbr_queue", "cbr_snr",
        "vbr_traffic", "vbr_th", "vbr_prb", "vbr_queue", "vbr_snr",
    ]
    time_per_step = slots_per_step * 1e-3
    norm_const = {
        "cbr_traffic": 5e6 * time_per_step,
        "cbr_th": 10e6 * time_per_step,
        "cbr_prb": 25 * slots_per_step,
        "cbr_queue": 2e5 * slots_per_step,
        "cbr_snr": 35 * slots_per_step,
        "vbr_traffic": 5e6 * time_per_step,
        "vbr_th": 10e6 * time_per_step,
        "vbr_prb": 35 * slots_per_step,
        "vbr_queue": 3e5 * slots_per_step,
        "vbr_snr": 35 * slots_per_step,
    }

    ran = SliceRANeMBB(
        rng=rng,
        user_counter=user_counter,
        id=slice_id,
        SLA=sla,
        CBR_description=cbr_desc,
        VBR_description=vbr_desc,
        state_variables=state_variables,
        norm_const=norm_const,
        slots_per_step=slots_per_step,
        slot_length=1e-3,
    )

    snr_generator = SINRSelectiveFading(rng, "macro_cell_urban_2GHz", n_prbs=n_prbs)
    scheduler = ProportionalFair(MCSCodeset())
    return SliceL1eMBB(rng, snr_generator, n_prbs, [ran], scheduler, external_ues=True)


def build_urllc_slice(rng, user_counter, slice_id: int, slots_per_step: int, n_prbs: int):
    cbr_desc = {
        "lambda": 4.0 / 60.0,
        "t_mean": 10.0,
        "bit_rate": 2_000_000,
    }
    vbr_desc = {
        "lambda": 6.0 / 60.0,
        "t_mean": 8.0,
        "p_size": 500,
        "b_size": 200,
        "b_rate": 5,
    }
    sla = {
        "cbr_th": 5e6,
        "cbr_prb": 25,
        "cbr_queue": 8e4,
        "vbr_th": 6e6,
        "vbr_prb": 30,
        "vbr_queue": 1e5,
    }
    state_variables = [
        "cbr_traffic", "cbr_th", "cbr_prb", "cbr_queue", "cbr_snr",
        "vbr_traffic", "vbr_th", "vbr_prb", "vbr_queue", "vbr_snr",
    ]
    time_per_step = slots_per_step * 1e-3
    norm_const = {
        "cbr_traffic": 3e6 * time_per_step,
        "cbr_th": 6e6 * time_per_step,
        "cbr_prb": 30 * slots_per_step,
        "cbr_queue": 8e4 * slots_per_step,
        "cbr_snr": 35 * slots_per_step,
        "vbr_traffic": 3e6 * time_per_step,
        "vbr_th": 6e6 * time_per_step,
        "vbr_prb": 30 * slots_per_step,
        "vbr_queue": 1e5 * slots_per_step,
        "vbr_snr": 35 * slots_per_step,
    }

    ran = SliceRANURLC(
        rng=rng,
        user_counter=user_counter,
        id=slice_id,
        SLA=sla,
        CBR_description=cbr_desc,
        VBR_description=vbr_desc,
        state_variables=state_variables,
        norm_const=norm_const,
        slots_per_step=slots_per_step,
        slot_length=1e-3,
    )

    snr_generator = SINRSelectiveFading(rng, "macro_cell_urban_2GHz", n_prbs=n_prbs)
    scheduler = ProportionalFair(MCSCodeset())
    return SliceL1URLLC(rng, snr_generator, n_prbs, [ran], scheduler)


def build_mmtc_slice(rng, slice_id: int, slots_per_step: int, n_prbs: int):
    mtc_desc = {
        "n_devices": 200,
        "repetition_set": [2, 4, 8, 16],
        "period_set": [1000, 5000, 10000, 20000],
    }
    sla = {"delay": 300}
    state_variables = ["devices", "avg_rep", "delay"]
    norm_const = {
        "devices": 100 * slots_per_step,
        "avg_rep": 20 * slots_per_step,
        "delay": 100 * slots_per_step,
    }

    ran = SliceRANmMTC(
        rng=rng,
        id=slice_id,
        SLA=sla,
        MTCdescription=mtc_desc,
        state_variables=state_variables,
        norm_const=norm_const,
        slots_per_step=slots_per_step,
    )
    return SliceL1mMTC(n_prbs=n_prbs, slices_ran=[ran])


def build_gnbs_from_scenario(cfg: SimConfig) -> List[NodeB]:
    rng = np.random.default_rng(cfg.seed)
    user_counter = count()
    sc = scenarios[cfg.scenario_name]

    gnbs: List[NodeB] = []
    for i, ((x, y), radius, carrier_id, prbs) in enumerate(
        zip(sc["gnb_positions"], sc["coverage_radius"], sc["carrier_ids"], sc["max_prbs_per_gnb"])
    ):
        l1_slices = [
            build_embb_slice(rng, user_counter, slice_id=10 * i + 0, slots_per_step=cfg.slots_per_step, n_prbs=max(prbs // 2, 1)),
            build_urllc_slice(rng, user_counter, slice_id=10 * i + 1, slots_per_step=cfg.slots_per_step, n_prbs=max(prbs // 4, 1)),
            build_mmtc_slice(rng, slice_id=10 * i + 2, slots_per_step=cfg.slots_per_step, n_prbs=max(prbs - (prbs // 2) - (prbs // 4), 1)),
        ]

        gnb = NodeB(
            id=i,
            x=float(x),
            y=float(y),
            slices_l1=l1_slices,
            slots_per_step=cfg.slots_per_step,
            n_prbs=int(prbs),
            coverage_radius=float(radius),
            slot_length=cfg.step_dt,
            carrier_id=int(carrier_id),
            center_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
            tx_power_dbm=30.0,
            noise_figure_db=7.0,
        )
        gnbs.append(gnb)

    return gnbs


# -----------------------------------------------------------------------------
# UE placement
# -----------------------------------------------------------------------------

def generate_overlap_ues(env: MultiGNBWrapper, cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed + 123)
    sc = scenarios[cfg.scenario_name]
    positions = sc["gnb_positions"]
    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)

    center_x = float(xs.mean())
    center_y = float(ys.mean())

    slice_choices = ["eMBB", "eMBB", "URLLC", "mMTC"]
    bit_rate_map = {
        "eMBB": 4_000_000.0,
        "URLLC": 2_000_000.0,
        "mMTC": 200_000.0,
    }

    for _ in range(cfg.n_ues):
        # Most users near the overlap center, some around edges.
        if rng.random() < 0.7:
            x = rng.normal(center_x, 90.0)
            y = rng.normal(center_y, 90.0)
        else:
            ref = rng.integers(len(positions))
            x = rng.normal(xs[ref], 120.0)
            y = rng.normal(ys[ref], 120.0)

        vx = rng.uniform(-12.0, 12.0)
        vy = rng.uniform(-12.0, 12.0)
        slice_type = slice_choices[rng.integers(len(slice_choices))]
        bit_rate = bit_rate_map[slice_type]
        buffer_size = 2_000_000 if slice_type != "mMTC" else 200_000

        try:
            env.add_ue(
                x=float(x),
                y=float(y),
                vx=float(vx),
                vy=float(vy),
                slice_type=slice_type,
                bit_rate=bit_rate,
                buffer_size=buffer_size,
            )
        except ValueError:
            # Skip rare incompatible placements / types if any slice mismatch happens.
            continue


# -----------------------------------------------------------------------------
# Safe test stepping (avoids the current wrapper.step issue)
# -----------------------------------------------------------------------------

def safe_no_handover_step(env: MultiGNBWrapper) -> Dict:
    env._step_count += 1

    # Advance gNB internal slice logic.
    per_gnb_rewards = env._advance_gnbs()

    # Mobility + traffic generation.
    for ue in env.get_all_ues():
        ue.update_position(env.step_dt)
        ue.traffic_step()
        if ue.queue > 0:
            ue.wait_time += 1
        else:
            ue.wait_time = max(ue.wait_time - 1, 0)

    # Keep current serving cells and just simulate radio + service.
    env._simulate_radio_and_service()

    # Log a basic reward from current control UE if available.
    reward = 0.0
    if env._current_control_ue_id is not None and env._current_control_ue_id in env._ues:
        control_ue = env._ues[env._current_control_ue_id]
        serving = env._get_gnb_by_id(control_ue.serving_gnb) if control_ue.serving_gnb is not None else None
        reward = env.compute_reassociation_reward(
            ue=control_ue,
            target_gnb=serving,
            handover_done=False,
            pingpong=False,
        )

    env._current_control_ue_id = env._pick_next_control_ue() if env._ues else None
    env._log_step(float(reward))

    return env._build_info(per_gnb_rewards=per_gnb_rewards)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def collect_step_metrics(env: MultiGNBWrapper, step: int) -> Tuple[Dict, List[Dict]]:
    ues = env.get_all_ues()
    radio_rows = [env.get_ue_radio_metrics(ue.id) for ue in ues]

    connected = [ue for ue in ues if ue.connected]
    disconnected = [ue for ue in ues if not ue.connected]

    step_row = {
        "step": step,
        "n_ues": len(ues),
        "connected_ues": len(connected),
        "disconnected_ues": len(disconnected),
        "mean_throughput_bps": float(np.mean([ue.th for ue in ues])) if ues else 0.0,
        "sum_throughput_bps": float(np.sum([ue.th for ue in ues])) if ues else 0.0,
        "mean_queue_bits": float(np.mean([ue.queue for ue in ues])) if ues else 0.0,
        "max_queue_bits": float(np.max([ue.queue for ue in ues])) if ues else 0.0,
        "mean_wait_steps": float(np.mean([ue.wait_time for ue in ues])) if ues else 0.0,
        "p95_wait_steps": float(np.percentile([ue.wait_time for ue in ues], 95)) if ues else 0.0,
        "mean_sinr_db": float(np.mean([r["sinr_db"] for r in radio_rows if np.isfinite(r["sinr_db"])])) if radio_rows else -np.inf,
        "mean_snr_db": float(np.mean([r["snr_db"] for r in radio_rows if np.isfinite(r["snr_db"])])) if radio_rows else -np.inf,
    }

    serving_counts: Dict[int, int] = {}
    for ue in ues:
        sid = -1 if ue.serving_gnb is None else int(ue.serving_gnb)
        serving_counts[sid] = serving_counts.get(sid, 0) + 1
    for sid, count_val in sorted(serving_counts.items()):
        step_row[f"serving_gnb_{sid}_count"] = count_val

    ue_rows = []
    for ue in ues:
        ue_rows.append({
            "step": step,
            "ue_id": ue.id,
            "slice_type": getattr(ue, "slice_type", "unknown"),
            "x": ue.x,
            "y": ue.y,
            "vx": ue.vx,
            "vy": ue.vy,
            "serving_gnb": ue.serving_gnb,
            "connected": ue.connected,
            "queue_bits": ue.queue,
            "throughput_bps": ue.th,
            "wait_steps": ue.wait_time,
            "sinr_db": next((r["sinr_db"] for r in radio_rows if r["ue_id"] == ue.id), -np.inf),
            "snr_db": next((r["snr_db"] for r in radio_rows if r["ue_id"] == ue.id), -np.inf),
            "rx_power_dbm": next((r["rx_power_dbm"] for r in radio_rows if r["ue_id"] == ue.id), -np.inf),
            "interference_dbm": next((r["interference_dbm"] for r in radio_rows if r["ue_id"] == ue.id), -np.inf),
        })

    return step_row, ue_rows


def summarize_results(step_df: pd.DataFrame, ue_df: pd.DataFrame) -> pd.DataFrame:
    final_step = step_df.iloc[-1]

    summary = {
        "n_steps": int(step_df["step"].max()),
        "mean_system_throughput_bps": float(step_df["sum_throughput_bps"].mean()),
        "peak_system_throughput_bps": float(step_df["sum_throughput_bps"].max()),
        "mean_ue_throughput_bps": float(ue_df.groupby("ue_id")["throughput_bps"].mean().mean()),
        "mean_queue_bits": float(step_df["mean_queue_bits"].mean()),
        "legacy_mean_wait_steps": float(step_df["mean_wait_steps"].mean()),
        "legacy_p95_wait_steps_over_time": float(step_df["p95_wait_steps"].mean()),
        "mean_packet_delay_steps": float(step_df["mean_packet_delay_steps"].dropna().mean()) if "mean_packet_delay_steps" in step_df else np.nan,
        "p95_packet_delay_steps_over_time": float(step_df["p95_packet_delay_steps"].dropna().mean()) if "p95_packet_delay_steps" in step_df else np.nan,
        "mean_connected_ues": float(step_df["connected_ues"].mean()),
        "final_connected_ues": int(final_step["connected_ues"]),
        "final_disconnected_ues": int(final_step["disconnected_ues"]),
        "mean_sinr_db": float(step_df.replace([np.inf, -np.inf], np.nan)["mean_sinr_db"].mean()),
    }

    return pd.DataFrame([summary])


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_plots(step_df: pd.DataFrame, save_dir: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(step_df["step"], step_df["sum_throughput_bps"])
    plt.xlabel("Step")
    plt.ylabel("System throughput (bps)")
    plt.title("System Throughput")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "system_throughput.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    if "mean_packet_delay_steps" in step_df:
        plt.plot(step_df["step"], step_df["mean_packet_delay_steps"])
        plt.ylabel("Mean packet delay (steps)")
        plt.title("True Packet Delay")
        out_name = "packet_delay.png"
    else:
        plt.plot(step_df["step"], step_df["mean_wait_steps"])
        plt.ylabel("Mean wait steps")
        plt.title("Delay / Waiting Time")
        out_name = "delay_wait_time.png"
    plt.xlabel("Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / out_name, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(step_df["step"], step_df["mean_queue_bits"])
    plt.xlabel("Step")
    plt.ylabel("Mean queue (bits)")
    plt.title("Average Queue")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "mean_queue.png", dpi=200)
    plt.close()

    serving_cols = [c for c in step_df.columns if c.startswith("serving_gnb_") and c.endswith("_count")]
    if serving_cols:
        plt.figure(figsize=(10, 6))
        for col in serving_cols:
            plt.plot(step_df["step"], step_df[col], label=col.replace("_count", ""))
        plt.xlabel("Step")
        plt.ylabel("Attached UEs")
        plt.title("Serving gNB Load Evolution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "serving_gnb_loads.png", dpi=200)
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    cfg = SimConfig()
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building scenario: {cfg.scenario_name}")
    gnbs = build_gnbs_from_scenario(cfg)

    env = MultiGNBWrapper(
        gnb_list=gnbs,
        handover_hysteresis=0.05,
        handover_ttt=3,
        outage_penalty=1.0,
        handover_penalty=0.1,
        use_mean_gnb_reward=True,
        verbose=False,
        step_dt=cfg.step_dt,
        max_candidates=3,
        max_episode_steps=cfg.n_steps,
    )

    env.reset(seed=cfg.seed)
    generate_overlap_ues(env, cfg)

    print(f"[INFO] Number of gNBs: {len(gnbs)}")
    print(f"[INFO] Number of UEs inserted: {len(env.get_all_ues())}")

    packet_queues: Dict[int, deque] = defaultdict(deque)
    ue_delay_samples: Dict[int, List[float]] = defaultdict(list)
    system_delay_samples: List[float] = []

    step_rows: List[Dict] = []
    ue_rows: List[Dict] = []

    for step in range(1, cfg.n_steps + 1):
        safe_no_handover_step(env)

        for ue in env.get_all_ues():
            if ue.new_bits > 0:
                packet_queues[ue.id].append([float(ue.new_bits), int(step)])

            served_bits = float(max(getattr(ue, "bits", 0), 0))
            while served_bits > 0 and packet_queues[ue.id]:
                head_remaining, arrival_step = packet_queues[ue.id][0]
                take_bits = min(served_bits, head_remaining)
                head_remaining -= take_bits
                served_bits -= take_bits

                if head_remaining <= 1e-9:
                    delay_steps = step - arrival_step + 1
                    ue_delay_samples[ue.id].append(float(delay_steps))
                    system_delay_samples.append(float(delay_steps))
                    packet_queues[ue.id].popleft()
                else:
                    packet_queues[ue.id][0][0] = head_remaining

        step_row, ue_step_rows = collect_step_metrics(env, step)
        step_row["mean_packet_delay_steps"] = float(np.mean(system_delay_samples)) if system_delay_samples else np.nan
        step_row["p95_packet_delay_steps"] = float(np.percentile(system_delay_samples, 95)) if system_delay_samples else np.nan

        for row in ue_step_rows:
            samples = ue_delay_samples.get(row["ue_id"], [])
            row["mean_packet_delay_steps"] = float(np.mean(samples)) if samples else np.nan
            row["p95_packet_delay_steps"] = float(np.percentile(samples, 95)) if samples else np.nan
            row["backlog_packets"] = len(packet_queues[row["ue_id"]])
            row["backlog_bits_est"] = float(sum(pkt[0] for pkt in packet_queues[row["ue_id"]]))

        step_rows.append(step_row)
        ue_rows.extend(ue_step_rows)

        if step % 25 == 0 or step == 1 or step == cfg.n_steps:
            delay_txt = "nan" if np.isnan(step_row["mean_packet_delay_steps"]) else f"{step_row['mean_packet_delay_steps']:.2f}"
            print(
                f"[step {step:4d}] "
                f"thr_sum={step_row['sum_throughput_bps']:.2f} bps | "
                f"pkt_delay={delay_txt} steps | "
                f"connected={step_row['connected_ues']}/{step_row['n_ues']} | "
                f"mean_sinr={step_row['mean_sinr_db']:.2f} dB"
            )

    step_df = pd.DataFrame(step_rows)
    ue_df = pd.DataFrame(ue_rows)
    summary_df = summarize_results(step_df, ue_df)

    step_csv = save_dir / "step_metrics.csv"
    ue_csv = save_dir / "ue_metrics.csv"
    summary_csv = save_dir / "summary_metrics.csv"
    history_csv = save_dir / "wrapper_history.csv"

    step_df.to_csv(step_csv, index=False)
    ue_df.to_csv(ue_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(env.history).to_csv(history_csv, index=False)

    if cfg.add_plots:
        save_plots(step_df, save_dir)

    print("\n[INFO] Simulation complete.")
    print(f"[INFO] Step metrics saved to   : {step_csv}")
    print(f"[INFO] UE metrics saved to     : {ue_csv}")
    print(f"[INFO] Summary saved to        : {summary_csv}")
    print(f"[INFO] Wrapper history saved to: {history_csv}")
    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
