#!/usr/bin/env python3
import numpy as np
import gymnasium as gym

from scenario_creator import create_env  # adjust import

def main():
    # -------- CONFIG --------
    scenario_id = 4   # change scenario (0..4)
    steps = 50
    seed = 42

    rng = np.random.default_rng(seed)

    # -------- CREATE ENV --------
    env = create_env(rng, scenario_id)
    obs, info = env.reset()

    n_slices = env.unwrapped.n_slices
    n_prbs = env.unwrapped.n_prbs

    print(f"\nEnvironment ready:")
    print(f"  slices = {n_slices}")
    print(f"  total PRBs = {n_prbs}\n")

    # -------- MAIN LOOP --------
    for step in range(steps):
        print(f"\n--- Step {step} ---")

        # ---- USER INPUT ----
        print(f"Enter allocation for {n_slices} slices (space separated, sum <= {n_prbs})")

        while True:
            try:
                action = list(map(int, input("PRBs: ").split()))
                if len(action) != n_slices:
                    raise ValueError("Wrong number of slices")
                if sum(action) > n_prbs:
                    raise ValueError("Exceeds total PRBs")
                break
            except Exception as e:
                print(f"Invalid input: {e}")

        action = np.array(action, dtype=int)

        # ---- STEP ----
        obs, reward, terminated, truncated, info = env.step(action)

        # ---- KPIs ----
        violations = info.get("total_violations", None)

        print(f"Reward: {reward}")
        print(f"Violations: {violations}")
        print(f"Observation: {obs}")
        print(f"Info: {info}")

        # OPTIONAL: print detailed KPIs if available
        for key, value in info.items():
            if key != "total_violations":
                print(f"{key}: {value}")

        if terminated :
            print("Episode finished")
            break

    env.close()


if __name__ == "__main__":
    main()