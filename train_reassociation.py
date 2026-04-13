#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train PPO on the MultiGNBWrapper reassociation environment.

Requirements:
    pip install stable-baselines3 gymnasium

Run:
    python train_reassociation.py
"""

from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from scenario_creator import create_env
from senario_multi_gnodeb import scenarios


# ---------------------------------------------------------------------
# Scenario conversion
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_env(
    scenario_name="scenario_1",
    seed=42,
    slots_per_step=10,
    max_episode_steps=100,
):
    rng = np.random.default_rng(seed)
    sc = scenarios[scenario_name]
    gnb_configs = scenario_to_gnb_configs(sc)

    env = create_env(
        rng=rng,
        n=1,  # legacy slice config index still needed by create_env()
        multi_gnb=True,
        gnb_configs=gnb_configs,
        slots_per_step=slots_per_step,
        coverage_radius=250,
        handover_hysteresis=0.0,   # easier first training
        handover_ttt=1,            # immediate HO for first version
        outage_penalty=1.0,
        handover_penalty=0.1,
        use_mean_gnb_reward=True,
        verbose=False,
        max_episode_steps=max_episode_steps,
    )

    # Add a small deterministic set of UEs near overlap/border zones
    # so the agent can actually learn reassociation decisions.
    env.reset(seed=seed)

    env.add_ue(x=150.0, y=20.0,  vx=0.0, vy=0.0, slice_type="eMBB")
    env.add_ue(x=140.0, y=140.0, vx=0.0, vy=0.0, slice_type="URLLC")
    env.add_ue(x=280.0, y=20.0,  vx=0.0, vy=0.0, slice_type="mMTC")
    env.add_ue(x=150.0, y=120.0, vx=0.0, vy=0.0, slice_type="eMBB")

    return Monitor(env)


# ---------------------------------------------------------------------
# Callback for simple logging
# ---------------------------------------------------------------------



class TrainingLogger(BaseCallback):
    def __init__(self, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_info = self.model.ep_info_buffer
            if len(ep_info) > 0:
                mean_reward = np.mean([ep["r"] for ep in ep_info])
                self.rewards.append(float(mean_reward))
                self.steps.append(int(self.num_timesteps))

                print(
                    f"[step {self.num_timesteps}] "
                    f"mean_ep_reward={mean_reward:.4f}"
                )
        return True

    def plot_rewards(self, save_path=None):
        if len(self.steps) == 0:
            print("No reward data to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.steps, self.rewards, marker="o")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.title("PPO Training Reward Curve")
        plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Reward plot saved to: {save_path}")

        plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    save_dir = Path("models_reassociation")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        scenario_name="scenario_1",
        seed=42,
        slots_per_step=10,
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=42,
    )

    callback = TrainingLogger(check_freq=2000, verbose=1)

    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = save_dir / "ppo_reassociation_v1"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Quick rollout test
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(20):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Eval step={step} "
            f"reward={reward:.4f} "
            f"connected={info.get('n_connected_ues')} "
            f"ue_per_gnb={info.get('ue_per_gnb')}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    print(f"Total eval reward over 20 steps: {total_reward:.4f}")


if __name__ == "__main__":
    main()