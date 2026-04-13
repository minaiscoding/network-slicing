#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scenario_creator import create_env


SCENARIO_3GNB_OVERLAP = {
    "n_gnbs": 3,
    "max_prbs_per_gnb": [150, 150, 150],
    "gnb_positions": [
        (0.0, 0.0),
        (400.0, 0.0),
        (200.0, 350.0),
    ],
    "coverage_radius": [350.0, 350.0, 350.0],
    "carrier_ids": [0, 0, 0],
    "n_ues": 1,
    "ue_distribution": "manual",
    "slices": [
        {"type": "eMBB", "count": 2},
        {"type": "mMTC", "count": 1},
        {"type": "URLLC", "count": 1},
    ],
}


def scenario_to_gnb_configs(scenario_dict):
    gnb_configs = []

    positions = scenario_dict["gnb_positions"]
    radii = scenario_dict["coverage_radius"]
    carriers = scenario_dict["carrier_ids"]
    max_prbs = scenario_dict["max_prbs_per_gnb"]

    for i, ((x, y), r, c, p) in enumerate(zip(positions, radii, carriers, max_prbs)):
        gnb_configs.append({
            "id": i,
            "x": x,
            "y": y,
            "coverage_radius": r,
            "carrier_id": c,
            "n_prbs": p,
        })

    return gnb_configs


def make_env(seed=42, max_episode_steps=80):
    rng = np.random.default_rng(seed)
    gnb_configs = scenario_to_gnb_configs(SCENARIO_3GNB_OVERLAP)

    env = create_env(
        rng=rng,
        n=1,
        multi_gnb=True,
        gnb_configs=gnb_configs,
        slots_per_step=10,
        coverage_radius=350,
        handover_hysteresis=0.0,
        handover_ttt=1,
        outage_penalty=1.0,
        handover_penalty=0.1,
        use_mean_gnb_reward=True,
        verbose=False,
        max_episode_steps=max_episode_steps,
    )
    return env


def print_ue_state(env, ue_id, prefix=""):
    ue = env.get_ue(ue_id)
    metrics = env.get_ue_radio_metrics(ue_id)
    candidates = env.get_candidate_gnbs(ue, top_k=3)
    candidate_ids = [g.id for g in candidates]

    print(
        f"{prefix}"
        f"UE={ue_id} | "
        f"pos=({ue.x:.2f}, {ue.y:.2f}) | "
        f"vel=({ue.vx:.2f}, {ue.vy:.2f}) | "
        f"serving={ue.serving_gnb} | "
        f"connected={ue.connected} | "
        f"sinr={metrics['sinr_db']:.2f} dB | "
        f"rx={metrics['rx_power_dbm']:.2f} dBm | "
        f"intf={metrics['interference_dbm']:.2f} dBm | "
        f"cand={candidate_ids}"
    )


def plot_network_state(env, ue_id, filename, title):
    fig, ax = plt.subplots(figsize=(8, 8))

    # gNBs + coverage
    for gnb in env.gnbs:
        gx = float(gnb.x)
        gy = float(gnb.y)
        gr = float(getattr(gnb, "coverage_radius", 0.0))

        circle = Circle((gx, gy), gr, fill=False, alpha=0.35)
        ax.add_patch(circle)

        ax.scatter(gx, gy, marker="^", s=160)
        ax.text(gx, gy, f"gNB {gnb.id}", fontsize=10, ha="left", va="bottom")

    # UE
    ue = env.get_ue(ue_id)
    ax.scatter(float(ue.x), float(ue.y), s=120, marker="o")
    ax.text(float(ue.x), float(ue.y), f"UE {ue.id}", fontsize=10, ha="left", va="bottom")

    # velocity arrow
    ax.arrow(
        float(ue.x),
        float(ue.y),
        float(ue.vx) * 0.5,
        float(ue.vy) * 0.5,
        length_includes_head=True,
        head_width=12,
        head_length=18,
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_trajectory(env, trajectory_x, trajectory_y, ue_id, filename):
    fig, ax = plt.subplots(figsize=(8, 8))

    for gnb in env.gnbs:
        gx = float(gnb.x)
        gy = float(gnb.y)
        gr = float(getattr(gnb, "coverage_radius", 0.0))

        circle = Circle((gx, gy), gr, fill=False, alpha=0.25)
        ax.add_patch(circle)

        ax.scatter(gx, gy, marker="^", s=160)
        ax.text(gx, gy, f"gNB {gnb.id}", fontsize=10, ha="left", va="bottom")

    ax.plot(trajectory_x, trajectory_y, linewidth=2, label=f"UE {ue_id} trajectory")
    ax.scatter(trajectory_x[0], trajectory_y[0], s=100, marker="o", label="start")
    ax.scatter(trajectory_x[-1], trajectory_y[-1], s=100, marker="x", label="end")

    ax.set_title("UE trajectory over the network")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def main():
    env = make_env(seed=42, max_episode_steps=100)

    ue_id = env.add_ue(
        x=180.0,
        y=120.0,
        vx=8000.0,
        vy=1000.0,
        slice_type="eMBB",
    )

    obs, info = env.reset(seed=123)

    print("=" * 90)
    print("INITIAL UE STATE")
    print("=" * 90)
    print_ue_state(env, ue_id)

    # plot at start
    plot_network_state(
        env,
        ue_id,
        filename="network_start.png",
        title="Network state at start",
    )

    trajectory_x = []
    trajectory_y = []

    print("\n" + "=" * 90)
    print("HANDOVER TEST: always choose action=1")
    print("=" * 90)

    last_serving = env.get_ue(ue_id).serving_gnb

    for step in range(40):
        ue = env.get_ue(ue_id)
        trajectory_x.append(float(ue.x))
        trajectory_y.append(float(ue.y))

        action = 1
        obs, reward, terminated, truncated, info = env.step(action)

        ue = env.get_ue(ue_id)
        metrics = env.get_ue_radio_metrics(ue_id)

        changed = ue.serving_gnb != last_serving

        print(
            f"step={step:02d} | "
            f"x={ue.x:.2f} | y={ue.y:.2f} | "
            f"serving={ue.serving_gnb} | "
            f"target={ue.target_gnb} | "
            f"pending={ue.ho_pending} | "
            f"counter={ue.ho_counter} | "
            f"sinr={metrics['sinr_db']:.2f} dB | "
            f"reward={reward:.4f}"
        )

        if changed:
            print(
                f"*** HANDOVER DETECTED at step {step}: "
                f"{last_serving} -> {ue.serving_gnb} ***"
            )
            last_serving = ue.serving_gnb

        if terminated or truncated:
            print("Episode ended.")
            break

    # append final position
    ue = env.get_ue(ue_id)
    trajectory_x.append(float(ue.x))
    trajectory_y.append(float(ue.y))

    print("\n" + "=" * 90)
    print("FINAL UE STATE")
    print("=" * 90)
    print_ue_state(env, ue_id)

    # plot at end
    plot_network_state(
        env,
        ue_id,
        filename="network_end.png",
        title="Network state at end",
    )

    # plot trajectory
    plot_trajectory(
        env,
        trajectory_x,
        trajectory_y,
        ue_id,
        filename="ue_trajectory.png",
    )


if __name__ == "__main__":
    main()