#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-algorithm OmniSafe config builders.

Usage:
    from pipelines.configs import build_config, SUPPORTED_ALGOS
    cfgs = build_config("CPO", total_steps=2000000, steps_per_epoch=8000, device="cuda:0")
"""

SUPPORTED_ALGOS = ["CPO", "PPO", "PPOLag", "SACLag", "TD3Lag"]
ON_POLICY       = {"CPO", "PPO", "PPOLag"}
OFF_POLICY      = {"SACLag", "TD3Lag"}


def build_config(algo: str, total_steps: int, steps_per_epoch: int, device: str) -> dict:
    """Return the complete custom_cfgs dict for *algo*."""
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algorithm '{algo}'. Choose from {SUPPORTED_ALGOS}")

    if algo in ON_POLICY:
        return _on_policy_config(algo, total_steps, steps_per_epoch, device)
    else:
        return _off_policy_config(algo, total_steps, steps_per_epoch, device)


# ── On-policy (CPO, PPO, PPOLag) ──────────────────────────────────────

def _on_policy_config(algo, total_steps, steps_per_epoch, device):
    cfgs = {
        "train_cfgs": {
            "total_steps": total_steps,
            "device": device,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "update_iters": 3,
            "batch_size": 128,
            "target_kl": 0.02,
            "entropy_coef": 0.01,
            "use_max_grad_norm": True,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "cost_gamma": 0.99,
            "lam": 0.95,
            "lam_c": 0.95,
            "obs_normalize": True,
            "reward_normalize": True,
            "standardized_rew_adv": True,
            "standardized_cost_adv": True,
        },
        "model_cfgs": {
            "actor":  {"hidden_sizes": [256, 256], "activation": "tanh"},
            "critic": {"hidden_sizes": [256, 256], "activation": "tanh", "lr": 3e-4},
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }

    if algo == "CPO":
        cfgs["algo_cfgs"].update({
            "use_cost": True,
            "cost_normalize": True,
            "cg_damping": 0.1,
            "cg_iters": 15,
            "cost_limit": 100.0,
        })

    elif algo == "PPO":
        cfgs["algo_cfgs"].update({
            "use_cost": False,
            "clip": 0.2,
        })

    elif algo == "PPOLag":
        cfgs["algo_cfgs"].update({
            "use_cost": True,
            "cost_normalize": True,
            "clip": 0.2,
        })
        cfgs["lagrange_cfgs"] = {
            "cost_limit": 100.0,
            "lagrangian_multiplier_init": 0.001,
            "lambda_lr": 0.035,
            "lambda_optimizer": "Adam",
        }

    return cfgs


# ── Off-policy (TD3Lag, SACLag) ───────────────────────────────────────

def _off_policy_config(algo, total_steps, steps_per_epoch, device):
    cfgs = {
        "train_cfgs": {
            "total_steps": total_steps,
            "device": device,
            "eval_episodes": 0,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "update_cycle": 1,
            "update_iters": 1,
            "size": 1000000,
            "batch_size": 256,
            "reward_normalize": False,
            "cost_normalize": True,
            "obs_normalize": False,
            "max_grad_norm": 40,
            "use_critic_norm": False,
            "polyak": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "use_cost": True,
            "warmup_epochs": 100,
        },
        "model_cfgs": {
            "actor":  {"hidden_sizes": [256, 256], "activation": "relu", "lr": 3e-4},
            "critic": {"num_critics": 2, "hidden_sizes": [256, 256],
                       "activation": "relu", "lr": 3e-4},
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
        "lagrange_cfgs": {
            "cost_limit": 100.0,
            "lagrangian_multiplier_init": 0.001,
            "lambda_lr": 0.00001,
            "lambda_optimizer": "Adam",
        },
    }

    if algo == "TD3Lag":
        cfgs["model_cfgs"]["actor_type"] = "mlp"
        cfgs["algo_cfgs"].update({
            "start_learning_steps": 25000,
            "use_exploration_noise": True,
            "exploration_noise": 0.1,
            "policy_noise": 0.2,
            "policy_noise_clip": 0.5,
        })

    elif algo == "SACLag":
        cfgs["model_cfgs"]["actor_type"] = "gaussian_sac"
        cfgs["algo_cfgs"].update({
            "start_learning_steps": 10000,
            "use_exploration_noise": False,
            "alpha": 0.2,
            "auto_alpha": False,
        })

    return cfgs
