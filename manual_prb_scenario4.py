#!/usr/bin/env python3
"""Interactive PRB allocation tool for scenario_5.

Scenario_5 has three slice types: eMBB, mMTC, and URLLC.
This script lets you enter PRB allocations manually and prints per-slice SLA
status and key metrics after every environment step.
"""

import argparse
import numpy as np

from scenario_creator import create_env


SCENARIO_INDEX = 4  # scenario_5 in scenario_creator.py


def sla_text(label: int) -> str:
    return "FULFILLED" if int(label) > 0 else "VIOLATED"


def print_step_report(env, action, reward, info):
    slots_per_step = env.node_b.slots_per_step
    slot_length = env.node_b.slot_length
    observation_time = slots_per_step * slot_length

    sla_labels = info["SLA_labels"]
    violations = info["violations"]
    total_violations = int(info["total_violations"])

    embb_info = info["l1_info"][0][0]
    mmtc_info = info["l1_info"][1][0]
    urllc_info = info["l1_info"][2][0]

    embb_cbr_mbps = embb_info["cbr_th"] / observation_time / 1e6
    embb_vbr_mbps = embb_info["vbr_th"] / observation_time / 1e6
    embb_cbr_lat_ms = embb_info["cbr_latency"] / slots_per_step
    embb_vbr_lat_ms = embb_info["vbr_latency"] / slots_per_step

    mmtc_delay = mmtc_info["delay"] / slots_per_step

    urllc_cbr_mbps = urllc_info["cbr_th"] / observation_time / 1e6
    urllc_vbr_mbps = urllc_info["vbr_th"] / observation_time / 1e6
    urllc_cbr_lat_ms = urllc_info["cbr_latency"]
    urllc_vbr_lat_ms = urllc_info["vbr_latency"]
    urllc_cbr_loss = urllc_info.get("cbr_pkt_loss", 0.0)
    urllc_vbr_loss = urllc_info.get("vbr_pkt_loss", 0.0)

    print("\n--- Step Report ---")
    print(
        f"Action PRBs: eMBB={action[0]}  mMTC={action[1]}  URLLC={action[2]}  "
        f"total={int(np.sum(action))}/{env.n_prbs}"
    )
    print(f"Reward: {reward:.2f}  Total SLA violations: {total_violations}")
    print(
        f"eMBB SLA: {sla_text(sla_labels[0])} "
        f"(label={int(sla_labels[0])}, violations={int(violations[0])})"
    )
    print(
        f"mMTC SLA: {sla_text(sla_labels[1])} "
        f"(label={int(sla_labels[1])}, violations={int(violations[1])})"
    )
    print(
        f"URLLC SLA: {sla_text(sla_labels[2])} "
        f"(label={int(sla_labels[2])}, violations={int(violations[2])})"
    )
    print(
        f"eMBB metrics: CBR_th={embb_cbr_mbps:.3f} Mbps, VBR_th={embb_vbr_mbps:.3f} Mbps, "
        f"CBR_lat={embb_cbr_lat_ms:.3f} ms, VBR_lat={embb_vbr_lat_ms:.3f} ms"
    )
    print(f"mMTC metrics: avg_delay={mmtc_delay:.3f} slots")
    print(
        f"URLLC metrics: CBR_th={urllc_cbr_mbps:.3f} Mbps, VBR_th={urllc_vbr_mbps:.3f} Mbps, "
        f"CBR_max_lat={urllc_cbr_lat_ms:.3f} ms, VBR_max_lat={urllc_vbr_lat_ms:.3f} ms, "
        f"CBR_loss={urllc_cbr_loss:.5f}, VBR_loss={urllc_vbr_loss:.5f}"
    )


def parse_action(raw: str, n_prbs: int):
    parts = raw.strip().split()
    if len(parts) != 3:
        raise ValueError(
            "Please enter exactly three integers: '<eMBB_prbs> <mMTC_prbs> <URLLC_prbs>'."
        )

    embb_prbs = int(parts[0])
    mmtc_prbs = int(parts[1])
    urllc_prbs = int(parts[2])

    if embb_prbs < 0 or mmtc_prbs < 0 or urllc_prbs < 0:
        raise ValueError("PRBs must be non-negative integers.")

    total = embb_prbs + mmtc_prbs + urllc_prbs
    if total > n_prbs:
        raise ValueError(f"Total PRBs cannot exceed {n_prbs}. You entered {total}.")

    return np.array([embb_prbs, mmtc_prbs, urllc_prbs], dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Manual PRB control for scenario_5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--penalty", type=float, default=100.0, help="Penalty per SLA violation")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = create_env(rng, SCENARIO_INDEX, penalty=args.penalty)

    obs, _ = env.reset(seed=args.seed)
    _ = obs  # observation is not used in this manual tool

    print("Manual PRB controller for scenario_5 (index 4)")
    print(f"Total PRBs available: {env.n_prbs}")
    print("Slice order: [0]=eMBB, [1]=mMTC, [2]=URLLC")
    print("Type: <eMBB_prbs> <mMTC_prbs> <URLLC_prbs>")
    print("Type 'q' to quit.\n")

    while True:
        try:
            raw = input("Enter PRBs (e.g., '80 40 80'): ").strip()
        except EOFError:
            print("\nExiting.")
            break

        if raw.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        try:
            action = parse_action(raw, env.n_prbs)
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            continue

        _, reward, _, info = env.step(action)
        print_step_report(env, action, reward, info)


if __name__ == "__main__":
    main()
