#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train TD3 (OmniSafe) in multi-scenario network-slicing experiments.
Rotates between low, medium, and congested scenarios per epoch.

Enhanced reward function:
1. URLLC: minimize HoL delay (weight 3, highest priority)
2. eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
3. mMTC: minimize delay (weight 1, lowest priority)
Plus: PRB efficiency bonus and allocation stability penalty
'''

import os
import argparse
import numpy as np
from numpy.random import default_rng
from gymnasium import spaces
from wrapper import ReportWrapper
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from config_loader import load_scenario
from scenario_creator import create_env_from_config

RUNS = 1
EPOCHS = 20
STEPS_PER_EPOCH = 1000
PENALTY = 1000
SCENARIOS = ['low', 'medium', 'congested']

ENV_ID = "RanSliceTD3Lag-v0"

_RNG           = np.random.default_rng(3)
_PENALTY       = 100.0
_TOTAL_STEPS   = 20000  # 20 epochs * 1000 steps/epoch

_scenario_configs_cache = {}

def get_scenario_config(scenario_name):
    """Load scenario config from yaml cache."""
    if scenario_name not in _scenario_configs_cache:
        _scenario_configs_cache[scenario_name] = load_scenario('scenarios.yaml', scenario_name)
    return _scenario_configs_cache[scenario_name]


@env_register
@env_unregister
class RanSliceTD3LagEnv(CMDP):
    """
    Multi-scenario TD3Lag environment with scenario rotation per epoch.
    Inherits enhanced reward function with eMBB HoL delay + throughput optimization.
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = False
    need_time_limit_wrapper  = False
    _num_envs                = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        
        # Load scenario configs from yaml
        self.scenario_names = SCENARIOS
        self.scenario_configs = [get_scenario_config(name) for name in self.scenario_names]
        self.current_scenario_idx = 0
        self.epoch_count = 0
        
        # Create initial environment with first scenario
        cfg = self.scenario_configs[self.current_scenario_idx]
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)
        
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=1,
            path=f'./results/scenario_comparison/TD3Lag/',
            verbose=False,
        )

        self._max_episode_steps = 500
        self._n_prbs            = cfg.n_prbs  # Get PRB count from config
        self._step_count        = 0
        self._action_space      = spaces.Box(low=0.0, high=1.0, 
                                             shape=(self._env.n_slices + 1,), dtype=float)
        
        # Custom observation space: avg_bler, avg_snr, traffic_load, current_allocation per slice
        self._n_slices = self._env.n_slices
        self._obs_size = self._n_slices * 4
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._obs_size,), dtype=float
        )
        
        # Track allocation history for reward computation
        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        self._last_reward = 0.0
        self._steps_in_epoch = 0

    def _rotate_scenario(self):
        """Rotate to next scenario and recreate environment."""
        self.epoch_count += 1
        self.current_scenario_idx = self.epoch_count % len(self.scenario_names)
        
        cfg = self.scenario_configs[self.current_scenario_idx]
        scenario_name = self.scenario_names[self.current_scenario_idx]
        
        print(f"\n[Epoch {self.epoch_count}] Rotating to scenario: {scenario_name}")
        
        # Close current environment
        if hasattr(self, '_env'):
            self._env.close()
        
        # Create new environment with rotated scenario
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=1,
            path=f'./results/scenario_comparison/TD3Lag_{scenario_name}/',
            verbose=False,
        )
        
        self._n_prbs = cfg.n_prbs
        self._step_count = 0
        self._steps_in_epoch = 0

    def reset(self, seed=None, options=None):
        # Check if we need to rotate scenario (every STEPS_PER_EPOCH steps approx)
        if self._steps_in_epoch >= STEPS_PER_EPOCH:
            self._rotate_scenario()
        
        self._step_count = 0
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        obs = self._compute_observation(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def _compute_observation(self, info):
        """
        Compute observation: avg_bler, avg_snr, traffic_load, current_allocation per slice.
        """
        obs = np.zeros(self._obs_size, dtype=float)
        idx = 0
        
        l1_info = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])
        
        for slice_idx in range(self._n_slices):
            bler = 0.0
            snr = 0.0
            traffic = 0.0
            
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            cbr_bler = ran_info.get('cbr_bler', 0.0)
                            vbr_bler = ran_info.get('vbr_bler', 0.0)
                            bler = max(cbr_bler, vbr_bler)
                            
                            cbr_snr = ran_info.get('cbr_snr', 0.0)
                            vbr_snr = ran_info.get('vbr_snr', 0.0)
                            snr = (cbr_snr + vbr_snr) / 2.0
                            
                            cbr_queue = ran_info.get('cbr_queue', 0.0)
                            vbr_queue = ran_info.get('vbr_queue', 0.0)
                            traffic = (cbr_queue + vbr_queue) / 2.0
            
            bler = np.clip(bler, 0, 1)
            snr = np.clip(snr / 30.0, -1, 1)
            traffic = np.clip(traffic / 100000.0, 0, 1)
            
            alloc = n_prbs_list[slice_idx] / max(self._n_prbs, 1) if slice_idx < len(n_prbs_list) else 0.0
            alloc = np.clip(alloc, 0, 1)
            
            obs[idx] = bler
            obs[idx + self._n_slices] = snr
            obs[idx + 2*self._n_slices] = traffic
            obs[idx + 3*self._n_slices] = alloc
            idx += 1
        
        return obs

    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        act = np.abs(act)
    
        total = act.sum()
        if total > 0:
            act = act / total
        
        alloc = act[:self._env.n_slices]
        excess = act[self._env.n_slices]

        alloc_prbs = np.array([int(np.floor(a * self._n_prbs)) for a in alloc], dtype=int)
        result = self._env.step(alloc_prbs)
        
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost = float(info.get("cost", 0.0))
        
        # Compute custom reward with enhanced objectives
        reward = self._compute_custom_reward(alloc_prbs, excess, info)
     

        self._step_count += 1
        self._steps_in_epoch += 1
        
        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0
            self._steps_in_epoch = 0

        if isinstance(info, dict):
            info = {str(k): v for k, v in info.items()}
        else:
            info = {}

        obs = self._compute_observation(info)
        
        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            new_obs_raw, new_info = self._env.reset()
            new_obs = self._compute_observation(new_info)
            obs = torch.as_tensor(new_obs, dtype=torch.float32)
            info["final_observation"] = final_obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            info["final_observation"] = obs

        return (
            obs,
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )
    
    def _compute_custom_reward(self, alloc_prbs, excess, info):
        """
        Enhanced reward function with priorities:
        1. URLLC: minimize HoL delay (weight 3, highest priority)
        2. eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
        3. mMTC: minimize delay (weight 1, lowest priority)
        Plus:
        - Remaining PRBs bonus (PRB efficiency)
        - Allocation stability (penalize large changes)
        """
        reward = 0.0
        
        l1_info = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])
        
        for slice_idx in range(self._n_slices):
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            slice_type = ran_info.get('type', '')
                            
                            # URLLC: minimize HoL delay (weight 3, highest priority)
                            if 'urllc' in slice_type.lower() or slice_idx == 2:
                                cbr_delay = ran_info.get('cbr_delay', 0.0)
                                reward += -3.0 * np.clip(cbr_delay / 100.0, 0, 1)
                            
                            # eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
                            elif 'embb' in slice_type.lower() or slice_idx == 0:
                                # eMBB HoL delay minimization
                                cbr_delay = ran_info.get('cbr_delay', 0.0)
                                reward += -2.0 * np.clip(cbr_delay / 100.0, 0, 1)
                                
                                # eMBB throughput maximization
                                cbr_th = ran_info.get('cbr_th', 0.0)
                                vbr_th = ran_info.get('vbr_th', 0.0)
                                throughput = (cbr_th + vbr_th) / 2.0
                                reward += 2.0 * np.clip(throughput / 1e6, 0, 1)
                            
                            # mMTC: minimize delay (weight 1, lowest priority)
                            elif 'mmtc' in slice_type.lower() or slice_idx == 1:
                                delay = ran_info.get('delay', 0.0)
                                reward += -1.0 * np.clip(delay / 100.0, 0, 1)
        
        # Bonus for remaining PRBs (PRB efficiency)
        remaining_prbs = self._n_prbs - alloc_prbs.sum()
        reward += 0.1 * (remaining_prbs / self._n_prbs)
        
        # Penalize large allocation changes (stability)
        allocation_change = np.sum(np.abs(alloc_prbs - self._last_allocation))
        reward -= 0.05 * (allocation_change / self._n_prbs)
        
        self._last_allocation = alloc_prbs.copy().astype(float)
        
        return reward

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def set_seed(self, seed: int) -> None:
        pass

    def spec_log(self, logger) -> None:
        pass

    def close(self) -> None:
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps


class TrainerTD3Lag:
    """Train TD3Lag agent across multi-scenario rotating environment."""
    
    def __init__(self):
        os.makedirs('./results/scenario_comparison/TD3Lag/', exist_ok=True)
        os.makedirs('./results/scenario_comparison/TD3Lag_low/', exist_ok=True)
        os.makedirs('./results/scenario_comparison/TD3Lag_medium/', exist_ok=True)
        os.makedirs('./results/scenario_comparison/TD3Lag_congested/', exist_ok=True)

    def train(self, run_id):
        """Train TD3Lag agent for 20000 steps across rotating scenarios."""
        rng = default_rng(seed=run_id)
        
        print(f'\n=== TD3Lag Training Run {run_id} ===')
        print(f'Scenarios: {", ".join(SCENARIOS)}')
        print(f'Total steps: {_TOTAL_STEPS} ({EPOCHS} epochs × {STEPS_PER_EPOCH} steps/epoch)')
        print('Scenario rotation: epoch % 3 → {0: low, 1: medium, 2: congested}')
        
        custom_cfgs = {
            "train_cfgs": {
                "total_steps": _TOTAL_STEPS,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "update_iters": 1,
            },
            "logger_cfgs": {
                "use_wandb": False,
                "save_model_freq": 1,
            },
        }

        agent = omnisafe.Agent(
            algo="TD3",
            env_id=ENV_ID,
            custom_cfgs=custom_cfgs,
        )

        print('Training started...')
        agent.learn()
        print('Training done!')
        print(f'Checkpoints saved to: runs/')
        agent.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train TD3Lag on multi-scenario RAN slicing with OmniSafe"
    )
    parser.add_argument("--runs", type=int, default=RUNS, help="Number of training runs")
    args = parser.parse_args()

    trainer = TrainerTD3Lag()
    
    for run in range(args.runs):
        trainer.train(run)
