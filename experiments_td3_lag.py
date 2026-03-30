#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train TD3 (OmniSafe) with CURRICULUM LEARNING across network-slicing scenarios.
Trains 20,000 steps per scenario: low → medium → congested (60,000 total).
Same agent throughout, separate npz files per scenario for plotting.

Enhanced reward function:
1. URLLC: minimize HoL delay (weight 3, highest priority)
2. eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
3. mMTC: minimize delay (weight 1, lowest priority)
Plus: PRB efficiency bonus and allocation stability penalty
'''

import os
import argparse
import numpy as np
import concurrent.futures as cf
from numpy.random import default_rng
from gymnasium import spaces
from wrapper import ReportWrapper
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from config_loader import load_scenario
from scenario_creator import create_env_from_config

RUNS                = 30
PROCESSES           = 4   # Adjust based on available CPU cores
STEPS_PER_SCENARIO  = 20000
PENALTY             = 1000
SCENARIOS           = ['low', 'medium', 'congested']

ENV_ID = "RanSliceTD3-v0"

_RNG         = np.random.default_rng(3)
_PENALTY     = 100.0
_TOTAL_STEPS = STEPS_PER_SCENARIO * len(SCENARIOS)  # 60,000

_scenario_configs_cache = {}

# ── module-level state for curriculum learning ──
_CURRENT_RUN_ID = 0
_ENV_INSTANCE_COUNT = {}
_CURRENT_SCENARIO_IDX = 0  # Which scenario we're currently training on
_STEPS_IN_CURRENT_SCENARIO = 0  # Steps taken in current scenario


def get_scenario_config(scenario_name):
    """Load scenario config from yaml cache."""
    if scenario_name not in _scenario_configs_cache:
        _scenario_configs_cache[scenario_name] = load_scenario('scenarios.yaml', scenario_name)
    return _scenario_configs_cache[scenario_name]


@env_register
@env_unregister
class RanSliceTD3Env(CMDP):
    """
    Curriculum Learning TD3 environment.
    
    Training progression:
    - 20,000 steps on 'low' scenario     → saves history_{run_id}_low.npz
    - 20,000 steps on 'medium' scenario  → saves history_{run_id}_medium.npz
    - 20,000 steps on 'congested' scenario → saves history_{run_id}_congested.npz
    
    Same agent trained continuously across all scenarios.
    Environment reset only happens once when switching scenarios.
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        global _ENV_INSTANCE_COUNT
        super().__init__(env_id)

        self._run_id = _CURRENT_RUN_ID
        
        # Track instance count - only first instance (training) records data
        if self._run_id not in _ENV_INSTANCE_COUNT:
            _ENV_INSTANCE_COUNT[self._run_id] = 0
        _ENV_INSTANCE_COUNT[self._run_id] += 1
        self._instance_id = _ENV_INSTANCE_COUNT[self._run_id]
        self._is_training_env = (self._instance_id == 1)
        
        print(f"[Run {self._run_id}] Creating env instance {self._instance_id} "
              f"({'TRAINING - will record' if self._is_training_env else 'EVAL - skipping recording'})")

        # Load scenario configs
        self.scenario_names   = SCENARIOS
        self.scenario_configs = [get_scenario_config(n) for n in self.scenario_names]
        self.current_scenario_idx = 0
        self._global_step_count = 0  # Total steps across all scenarios

        # Create first scenario environment
        cfg     = self.scenario_configs[0]
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)

        # Results path
        self._results_path = './results/scenario_comparison/TD3/'
        os.makedirs(self._results_path, exist_ok=True)

        # Create ReportWrapper for first scenario
        if self._is_training_env:
            self._env = ReportWrapper(
                raw_env,
                steps         = STEPS_PER_SCENARIO,
                control_steps = 500,
                env_id        = f"{self._run_id}_{self.scenario_names[0]}",
                path          = self._results_path,
                verbose       = False,
                continuous_mode = True,
            )
        else:
            self._env = ReportWrapper(
                raw_env,
                steps         = STEPS_PER_SCENARIO,
                control_steps = STEPS_PER_SCENARIO + 1,
                env_id        = f"{self._run_id}_eval",
                path          = self._results_path,
                verbose       = False,
                continuous_mode = True,
            )

        self._max_episode_steps = 500
        self._n_prbs            = cfg.n_prbs
        self._step_count        = 0  # Steps in current episode

        self._action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._env.n_slices + 1,), dtype=float
        )

        self._n_slices  = self._env.n_slices
        self._obs_size  = self._n_slices * 4
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._obs_size,), dtype=float
        )

        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        self._last_reward     = 0.0

    def _switch_scenario(self):
        """Switch to next scenario in curriculum. Called after STEPS_PER_SCENARIO steps."""
        if not self._is_training_env:
            return
            
        # Save current scenario results
        self._env.save_results()
        print(f"\n[Run {self._run_id}] Saved {self.scenario_names[self.current_scenario_idx]} "
              f"results to history_{self._run_id}_{self.scenario_names[self.current_scenario_idx]}.npz")
        
        # Move to next scenario
        self.current_scenario_idx += 1
        if self.current_scenario_idx >= len(self.scenario_names):
            print(f"[Run {self._run_id}] All scenarios completed!")
            return
        
        scenario_name = self.scenario_names[self.current_scenario_idx]
        cfg = self.scenario_configs[self.current_scenario_idx]
        
        print(f"\n[Run {self._run_id}] === CURRICULUM: Switching to {scenario_name.upper()} scenario ===")
        
        # Close old environment
        if hasattr(self._env, 'env') and hasattr(self._env.env, 'close'):
            self._env.env.close()
        
        # Create new raw environment
        new_raw = create_env_from_config(cfg, _RNG, penalty=_PENALTY)
        
        # Create NEW ReportWrapper for new scenario (fresh history arrays)
        self._env = ReportWrapper(
            new_raw,
            steps         = STEPS_PER_SCENARIO,
            control_steps = 500,
            env_id        = f"{self._run_id}_{scenario_name}",
            path          = self._results_path,
            verbose       = False,
            continuous_mode = True,
        )
        
        self._n_prbs = cfg.n_prbs
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        obs = self._compute_observation(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def _compute_observation(self, info):
        obs     = np.zeros(self._obs_size, dtype=float)
        idx     = 0
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])

        for slice_idx in range(self._n_slices):
            bler = snr = traffic = 0.0

            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            cbr_bler = ran_info.get('cbr_bler', 0.0)
                            vbr_bler = ran_info.get('vbr_bler', 0.0)
                            bler     = max(cbr_bler, vbr_bler)

                            cbr_snr = ran_info.get('cbr_snr', 0.0)
                            vbr_snr = ran_info.get('vbr_snr', 0.0)
                            snr     = (cbr_snr + vbr_snr) / 2.0

                            cbr_queue = ran_info.get('cbr_queue', 0.0)
                            vbr_queue = ran_info.get('vbr_queue', 0.0)
                            traffic   = (cbr_queue + vbr_queue) / 2.0

            bler    = np.clip(bler, 0, 1)
            snr     = np.clip(snr / 30.0, -1, 1)
            traffic = np.clip(traffic / 100000.0, 0, 1)
            alloc   = (n_prbs_list[slice_idx] / max(self._n_prbs, 1)
                       if slice_idx < len(n_prbs_list) else 0.0)
            alloc   = np.clip(alloc, 0, 1)

            obs[idx]                     = bler
            obs[idx + self._n_slices]    = snr
            obs[idx + 2*self._n_slices]  = traffic
            obs[idx + 3*self._n_slices]  = alloc
            idx += 1

        return obs

    def step(self, action: torch.Tensor):
        act   = action.cpu().numpy()
        act   = np.abs(act)
        total = act.sum()
        if total > 0:
            act = act / total

        alloc  = act[:self._env.n_slices]
        excess = act[self._env.n_slices]

        alloc_prbs = np.array(
            [int(np.floor(a * self._n_prbs)) for a in alloc], dtype=int
        )
        result = self._env.step(alloc_prbs)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated   = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost   = float(info.get("cost", 0.0))
        reward = self._compute_custom_reward(alloc_prbs, excess, info)

        self._step_count += 1
        self._global_step_count += 1

        # Check if it's time to switch scenarios
        steps_in_scenario = self._env.step_counter
        if self._is_training_env and steps_in_scenario >= STEPS_PER_SCENARIO:
            if self.current_scenario_idx < len(self.scenario_names) - 1:
                self._switch_scenario()
                # Reset after scenario switch
                new_obs_raw, new_info = self._env.reset()
                obs = self._compute_observation(new_info)
                return (
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(reward, dtype=torch.float32),
                    torch.as_tensor(cost, dtype=torch.float32),
                    torch.as_tensor(False, dtype=torch.bool),
                    torch.as_tensor(True, dtype=torch.bool),  # Truncated to trigger reset
                    {"final_observation": torch.as_tensor(obs, dtype=torch.float32)},
                )

        if self._step_count >= self._max_episode_steps:
            truncated = True
            self._step_count = 0

        info = {str(k): v for k, v in info.items()} if isinstance(info, dict) else {}
        obs = self._compute_observation(info)

        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            new_obs_raw, new_info = self._env.reset()
            new_obs = self._compute_observation(new_info)
            obs     = torch.as_tensor(new_obs, dtype=torch.float32)
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
        reward   = 0.0
        l1_info  = info.get('l1_info', [])

        for slice_idx in range(self._n_slices):
            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            slice_type = ran_info.get('type', '')

                            if 'urllc' in slice_type.lower() or slice_idx == 2:
                                cbr_delay = ran_info.get('cbr_delay', 0.0)
                                reward   += -3.0 * np.clip(cbr_delay / 100.0, 0, 1)

                            elif 'embb' in slice_type.lower() or slice_idx == 0:
                                cbr_delay  = ran_info.get('cbr_delay', 0.0)
                                reward    += -2.0 * np.clip(cbr_delay / 100.0, 0, 1)
                                cbr_th     = ran_info.get('cbr_th', 0.0)
                                vbr_th     = ran_info.get('vbr_th', 0.0)
                                throughput = (cbr_th + vbr_th) / 2.0
                                reward    += 2.0 * np.clip(throughput / 1e6, 0, 1)

                            elif 'mmtc' in slice_type.lower() or slice_idx == 1:
                                delay   = ran_info.get('delay', 0.0)
                                reward += -1.0 * np.clip(delay / 100.0, 0, 1)

        remaining_prbs    = self._n_prbs - alloc_prbs.sum()
        reward           += 0.1 * (remaining_prbs / self._n_prbs)

        allocation_change = np.sum(np.abs(alloc_prbs - self._last_allocation))
        reward           -= 0.05 * (allocation_change / self._n_prbs)

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
        # Save final scenario results
        if self._is_training_env:
            self._env.save_results()
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps


# ======================================================================== #

class TrainerTD3:
    """Train TD3 agent with curriculum learning across scenarios."""

    def __init__(self):
        os.makedirs('./results/scenario_comparison/TD3/', exist_ok=True)

    def train(self, run_id: int):
        global _CURRENT_RUN_ID, _ENV_INSTANCE_COUNT
        _CURRENT_RUN_ID = run_id
        _ENV_INSTANCE_COUNT[run_id] = 0

        rng = default_rng(seed=run_id)

        print(f'\n{"="*60}')
        print(f'=== TD3 CURRICULUM Training Run {run_id} ===')
        print(f'{"="*60}')
        print(f'Scenarios : {" → ".join(SCENARIOS)}')
        print(f'Steps per scenario: {STEPS_PER_SCENARIO}')
        print(f'Total steps: {_TOTAL_STEPS}')
        print(f'Output files:')
        for s in SCENARIOS:
            print(f'  - results/scenario_comparison/TD3/history_{run_id}_{s}.npz')

        custom_cfgs = {
            "train_cfgs": {
                "total_steps": _TOTAL_STEPS,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": 1000,
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

        print('\nTraining started...')
        agent.learn()
        print('Training done!')
        
        try:
            agent.plot(smooth=1)
        except Exception as e:
            print(f'OmniSafe plotting failed (expected in parallel mode): {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train TD3 with curriculum learning on RAN slicing"
    )
    parser.add_argument("--runs", type=int, default=RUNS,
                        help="Number of training runs")
    parser.add_argument("--processes", type=int, default=PROCESSES,
                        help="Number of parallel processes")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of parallel")
    parser.add_argument("--steps-per-scenario", type=int, default=STEPS_PER_SCENARIO,
                        help="Steps per scenario (default: 20000)")
    args = parser.parse_args()

    # Update global if provided
    if args.steps_per_scenario != STEPS_PER_SCENARIO:
        STEPS_PER_SCENARIO = args.steps_per_scenario
        _TOTAL_STEPS = STEPS_PER_SCENARIO * len(SCENARIOS)

    trainer = TrainerTD3()
    run_list = list(range(args.runs))

    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            results = list(E.map(trainer.train, run_list))