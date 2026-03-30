#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: juanjosealcaraz

Train and evaluate PPO (OmniSafe) in network-slicing scenarios.
Each epoch resets the environment with a proper RNG.
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
from scenario_creator import create_env

SCENARIO = 4
RUNS = 1
PROCESSES = 4
EPOCHS = 100
STEPS_PER_EPOCH = 1000
SLOTS_PER_STEP = 500
PENALTY = 1000
EVALUATION_STEPS = 5000

ENV_ID = "RanSlicePPOLag-v0"

_RNG         = np.random.default_rng(3)
_SCEN        = 4
_PENALTY     = 100.0
_TOTAL_STEPS = 200000

@env_register
@env_unregister
class RanSliceEnv(CMDP):

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper  = False
    need_time_limit_wrapper  = False
    _num_envs                = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        raw_env = create_env(_RNG, _SCEN, penalty=_PENALTY)
        self._env = ReportWrapper(
            raw_env,
            steps=_TOTAL_STEPS,
            control_steps=500,
            env_id=1,
            path='./results/scenario_4/PPO/',
            verbose=False,

        )

        self._max_episode_steps = 500
        self._n_prbs            = 100
        self._step_count        = 0  # track steps for forced truncation
        self._action_space      = spaces.Box(low=0.0, high=1.0, shape=(self._env.n_slices + 1,), dtype=float)
        
        # Custom observation space: avg_bler, avg_snr, traffic_load, current_allocation per slice
        # For 5 slices: 5 (bler) + 5 (snr) + 5 (traffic) + 5 (allocation) = 20 variables
        self._n_slices = self._env.n_slices
        self._obs_size = self._n_slices * 4  # bler, snr, traffic, allocation per slice
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._obs_size,), dtype=float
        )
        
        # Track allocation history for reward computation
        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        self._last_reward = 0.0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        # Compute custom observation
        obs = self._compute_observation(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def _compute_observation(self, info):
        """
        Compute observation space with: avg_bler, avg_snr, traffic_load, current_allocation per slice
        """
        obs = np.zeros(self._obs_size, dtype=float)
        idx = 0
        
        # Extract info from nested structure: info['l1_info'] is a dict with slice indices
        l1_info = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])
        
        for slice_idx in range(self._n_slices):
            bler = 0.0
            snr = 0.0
            traffic = 0.0
            
            # Get BLER and SNR from l1_info
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    # eMBB or URLLC slices (have CBR/VBR traffic)
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            # Average BLER from CBR and VBR
                            cbr_bler = ran_info.get('cbr_bler', 0.0)
                            vbr_bler = ran_info.get('vbr_bler', 0.0)
                            bler = max(cbr_bler, vbr_bler)  # use max as worst case
                            
                            # Average SNR from CBR and VBR
                            cbr_snr = ran_info.get('cbr_snr', 0.0)
                            vbr_snr = ran_info.get('vbr_snr', 0.0)
                            snr = (cbr_snr + vbr_snr) / 2.0
                            
                            # Traffic load (queue depth)
                            cbr_queue = ran_info.get('cbr_queue', 0.0)
                            vbr_queue = ran_info.get('vbr_queue', 0.0)
                            traffic = (cbr_queue + vbr_queue) / 2.0
            
            # Normalize and clip values
            bler = np.clip(bler, 0, 1)
            snr = np.clip(snr / 30.0, -1, 1)  # normalize SNR (assume max ~30 dB)
            traffic = np.clip(traffic / 100000.0, 0, 1)  # normalize queue length
            
            # Current allocation as fraction of total PRBs
            alloc = n_prbs_list[slice_idx] / max(self._n_prbs, 1) if slice_idx < len(n_prbs_list) else 0.0
            alloc = np.clip(alloc, 0, 1)
            
            # Fill observation
            obs[idx] = bler
            obs[idx + self._n_slices] = snr
            obs[idx + 2*self._n_slices] = traffic
            obs[idx + 3*self._n_slices] = alloc
            idx += 1
        
        return obs

    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        act = np.abs(act)
    
        # Normalize to sum to 1
        total = act.sum()
        if total > 0:
            act = act / total
        # Now act sums to exactly 1.0, split into 5 alloc + 1 excess
        alloc = act[:self._env.n_slices]   # sums to < 1.0
        excess = act[self._env.n_slices]   # saved budget, added to reward

        alloc_prbs = np.array([int(np.floor(a * self._n_prbs)) for a in alloc], dtype=int)
        result = self._env.step(alloc_prbs)
        

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        # Keep original cost function (violations)
        cost = float(info.get("cost", 0.0))
        
        # Compute custom reward with priorities
        reward = self._compute_custom_reward(alloc_prbs, excess, info)


        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0

        if isinstance(info, dict):
            info = {str(k): v for k, v in info.items()}
        else:
            info = {}

        # Compute custom observation
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
            torch.as_tensor(reward,     dtype=torch.float32),
            torch.as_tensor(cost,       dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated,  dtype=torch.bool),
            info,
        )
    
    def _compute_custom_reward(self, alloc_prbs, excess, info):
        """
        Custom reward with priorities:
        1. URLLC: minimize hold delay (weight 3, highest priority)
        2. eMBB: minimize hold delay (weight 2) AND maximize throughput (weight 2)
        3. mMTC: minimize delay (weight 1, lowest priority)
        Plus:
        - Remaining PRBs bonus (PRB efficiency)
        - Allocation stability (penalize large changes)
        """
        reward = 0.0
        
        l1_info = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])
        
        # Extract metrics from each slice
        for slice_idx in range(self._n_slices):
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            slice_type = ran_info.get('type', '')
                            
                            # URLLC: minimize HoL delay (weight 3, highest priority)
                            if 'urllc' in slice_type.lower() or slice_idx == 2:  # assuming URLLC is slice 2
                                cbr_delay = ran_info.get('cbr_delay', 0.0)
                                # Negative delay (lower is better)
                                reward += -3.0 * np.clip(cbr_delay / 100.0, 0, 1)
                            
                            # eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
                            elif 'embb' in slice_type.lower() or slice_idx == 0:  # assuming eMBB is slice 0
                                # eMBB HoL delay minimization (weight 2)
                                cbr_delay = ran_info.get('cbr_delay', 0.0)
                                reward += -2.0 * np.clip(cbr_delay / 100.0, 0, 1)
                                
                                # eMBB throughput maximization (weight 2)
                                cbr_th = ran_info.get('cbr_th', 0.0)
                                vbr_th = ran_info.get('vbr_th', 0.0)
                                throughput = (cbr_th + vbr_th) / 2.0
                                # Positive throughput (higher is better)
                                reward += 2.0 * np.clip(throughput / 1e6, 0, 1)
                            
                            # mMTC: minimize delay (weight 1, lowest priority)
                            elif 'mmtc' in slice_type.lower() or slice_idx == 1:  # assuming mMTC is slice 1
                                delay = ran_info.get('delay', 0.0)
                                # Negative delay (lower is better)
                                reward += -1.0 * np.clip(delay / 100.0, 0, 1)
        
        # Bonus for remaining PRBs (PRB efficiency)
        remaining_prbs = self._n_prbs - alloc_prbs.sum()
        reward += 0.1 * (remaining_prbs / self._n_prbs)
        
        # Penalize large allocation changes (stability)
        allocation_change = np.sum(np.abs(alloc_prbs - self._last_allocation))
        reward -= 0.05 * (allocation_change / self._n_prbs)
        
        # Update tracking variables
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
class Evaluator():
    def __init__(self, scenario=SCENARIO):
        self.scenario = scenario
        self.train_path = f'./results/scenario_{scenario}/PPO/'
        self.test_path = f'./results/scenario_{scenario}/PPO_t/'
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

    def evaluate(self, run_id):
        rng = default_rng(seed=run_id)
        env = RanSliceEnv(env_id=ENV_ID, rng=rng, scenario=self.scenario, penalty=PENALTY)

        # ==================== TRAINING ====================
        print(f'\n=== Training Run {run_id} ===')
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
            algo="PPOLag",
            env_id=ENV_ID,
            custom_cfgs=custom_cfgs,
        )

        print('Training started...')
        agent.learn()
        print('Training done!')
        agent.plot(smooth=1)
        # ==================== EVALUATION ====================
      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train PPO  lag on RanSlice with OmniSafe")
    parser.add_argument("--scenario", type=int, default=SCENARIO)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    evaluator = Evaluator(scenario=args.scenario)

    for run in range(args.runs):
        evaluator.evaluate(run)