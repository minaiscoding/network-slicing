#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train CPO (OmniSafe) with SEQUENTIAL CURRICULUM across network-slicing scenarios.
Trains low (250k) then medium (500k) then congested (250k), one pass.
Total 1M steps. Single NPZ file per run recording all metrics.

CPO (Constrained Policy Optimization) is an on-policy projection-based
algorithm that enforces constraint satisfaction via trust-region updates.

Enhanced reward function (same as TD3):
1. URLLC: minimize HoL delay (weight 3, highest priority)
2. eMBB: minimize HoL delay (weight 2) AND maximize throughput (weight 2)
3. mMTC: minimize delay (weight 1, lowest priority)
Plus: PRB efficiency bonus and allocation stability penalty

Usage:
    python experiments_cpo.py
    python experiments_cpo.py --runs 5 --sequential
    python experiments_cpo.py --steps-per-scenario 10000
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
PROCESSES           = 4
TOTAL_STEPS         = 2000000
PENALTY             = 1000
SCENARIOS           = ['low', 'medium', 'congested']
STEPS_PER_SCENARIO  = [500000, 1000000, 500000]   # low 500k, medium 1M, congested 500k
STEPS_PER_EPOCH     = 8000

ENV_ID = "RanSliceCPO-v1"

_RNG          = np.random.default_rng(3)
_PENALTY      = 10.0
_TOTAL_STEPS  = TOTAL_STEPS
_RESULTS_PATH = './results/500k/CPO/'
_EPOCH_COUNTER = 0   # incremented every STEPS_PER_EPOCH steps for reseeding

_scenario_configs_cache = {}

# ── module-level state for curriculum learning ──
_CURRENT_RUN_ID = 0
_ENV_INSTANCE_COUNT = {}


def get_scenario_config(scenario_name):
    """Load scenario config from yaml cache."""
    if scenario_name not in _scenario_configs_cache:
        _scenario_configs_cache[scenario_name] = load_scenario('scenarios.yaml', scenario_name)
    return _scenario_configs_cache[scenario_name]


@env_register
@env_unregister
class RanSliceCPOEnv(CMDP):
    """
    Cycling Curriculum CPO environment.

    Cycles through scenarios (low → medium → congested → low → ...)
    with STEPS_PER_SCENARIO steps per block. Single NPZ file per run.
    Total: 500k steps, recording reward/cost/violations/resources.
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        global _ENV_INSTANCE_COUNT
        super().__init__(env_id)

        self._run_id = _CURRENT_RUN_ID

        if self._run_id not in _ENV_INSTANCE_COUNT:
            _ENV_INSTANCE_COUNT[self._run_id] = 0
        _ENV_INSTANCE_COUNT[self._run_id] += 1
        self._instance_id = _ENV_INSTANCE_COUNT[self._run_id]
        self._is_training_env = (self._instance_id == 1)

        print(f"[Run {self._run_id}] Creating env instance {self._instance_id} "
              f"({'TRAINING' if self._is_training_env else 'EVAL'})")

        # Load scenario configs
        self.scenario_names   = SCENARIOS
        self.scenario_configs = [get_scenario_config(n) for n in self.scenario_names]
        self.current_scenario_idx = 0
        self._global_step_count = 0

        # Create first scenario environment
        cfg     = self.scenario_configs[0]
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)

        self._results_path = _RESULTS_PATH
        os.makedirs(self._results_path, exist_ok=True)

        if self._is_training_env:
            self._env = ReportWrapper(
                raw_env,
                steps         = _TOTAL_STEPS,
                control_steps = 5000,
                env_id        = str(self._run_id),
                path          = self._results_path,
                verbose       = False,
                continuous_mode = True,
            )
        else:
            self._env = ReportWrapper(
                raw_env,
                steps         = _TOTAL_STEPS,
                control_steps = _TOTAL_STEPS + 1,
                env_id        = f"{self._run_id}_eval",
                path          = self._results_path,
                verbose       = False,
                continuous_mode = True,
            )

        self._max_episode_steps = 2000
        self._n_prbs            = cfg.n_prbs
        self._step_count        = 0

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
        self._steps_in_current_scenario = 0

        # Emergency PRB redistribution: track consecutive violations per slice
        # Priority order: URLLC (slice 2) > eMBB (slice 0) > mMTC (slice 1)
        self._consecutive_violations = np.zeros(self._n_slices, dtype=int)
        # Sorted indices by priority: URLLC first, eMBB second, mMTC last
        self._priority_order = [2, 0, 1]   # URLLC > eMBB > mMTC
        self._EXCESS_THRESHOLD = 2    # 2 consecutive violations → give all excess PRBs
        self._STEAL_THRESHOLD  = 5    # 5 consecutive violations → steal from mMTC
        self._MMTC_IDX         = 1    # mMTC slice index (donor for stealing)
        self._MMTC_MIN_PRBS    = 5    # minimum PRBs mMTC keeps when stolen from

    def _switch_scenario(self):
        """Advance to next scenario (low → medium → congested, one pass)."""
        if not self._is_training_env:
            return

        next_idx = self.current_scenario_idx + 1
        if next_idx >= len(self.scenario_names):
            return  # all scenarios done, stay on last
        self.current_scenario_idx = next_idx
        scenario_name = self.scenario_names[self.current_scenario_idx]
        cfg = self.scenario_configs[self.current_scenario_idx]

        print(f"\n[Run {self._run_id}] === CYCLING to {scenario_name.upper()} "
              f"(global step {self._global_step_count}) ===")

        if hasattr(self._env, 'env') and hasattr(self._env.env, 'close'):
            self._env.env.close()

        new_raw = create_env_from_config(cfg, _RNG, penalty=_PENALTY)
        self._env.env = new_raw

        self._n_prbs = cfg.n_prbs
        self._steps_in_current_scenario = 0
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

        # ── Emergency PRB redistribution ──
        # Phase 1: Give excess PRBs to the highest-priority violating slice (>=2 steps)
        # Phase 2: If still violating after 5 steps, steal PRBs from mMTC
        # Priority: URLLC > eMBB > mMTC — serve in order until each is safe
        excess_prbs = self._n_prbs - alloc_prbs.sum()

        # Find which slices are in emergency (2+ consecutive violations)
        emergency_slices = [s for s in self._priority_order
                            if self._consecutive_violations[s] >= self._EXCESS_THRESHOLD]

        if emergency_slices and excess_prbs > 0:
            # Give ALL excess to highest-priority violating slice
            winner = emergency_slices[0]
            alloc_prbs[winner] += excess_prbs

        # Phase 2: Steal from mMTC if any slice has been violating 5+ steps
        severe_slices = [s for s in self._priority_order
                         if (self._consecutive_violations[s] >= self._STEAL_THRESHOLD
                             and s != self._MMTC_IDX)]

        if severe_slices:
            # Steal from mMTC (keep minimum), give to highest-priority severe slice
            mmtc_prbs = alloc_prbs[self._MMTC_IDX]
            stealable = max(0, mmtc_prbs - self._MMTC_MIN_PRBS)
            if stealable > 0:
                winner = severe_slices[0]
                alloc_prbs[self._MMTC_IDX] -= stealable
                alloc_prbs[winner] += stealable

        result = self._env.step(alloc_prbs)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated   = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost   = float(info.get("cost", 0.0))
        reward = self._compute_custom_reward(alloc_prbs, excess, info)

        # ── Update consecutive violation counters per slice ──
        per_slice_viol = info.get('violations', np.zeros(self._n_slices, dtype=int))
        if hasattr(per_slice_viol, '__len__') and len(per_slice_viol) == self._n_slices:
            for s in range(self._n_slices):
                if per_slice_viol[s] > 0:
                    self._consecutive_violations[s] += 1
                else:
                    self._consecutive_violations[s] = 0
        else:
            self._consecutive_violations[:] = 0

        self._step_count += 1
        self._global_step_count += 1

        # ── Reseed RNG at each epoch boundary for diversity ──
        global _EPOCH_COUNTER, _RNG
        if self._is_training_env and self._global_step_count % STEPS_PER_EPOCH == 0:
            _EPOCH_COUNTER += 1
            new_seed = 3 + _EPOCH_COUNTER * 1000 + self._run_id
            _RNG = np.random.default_rng(new_seed)

        # Check if it's time to advance to next scenario
        self._steps_in_current_scenario += 1
        scenario_budget = STEPS_PER_SCENARIO[self.current_scenario_idx]
        if self._is_training_env and self._steps_in_current_scenario >= scenario_budget:
            self._switch_scenario()
            new_obs_raw, new_info = self._env.reset()
            obs = self._compute_observation(new_info)
            return (
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(reward, dtype=torch.float32),
                torch.as_tensor(cost, dtype=torch.float32),
                torch.as_tensor(False, dtype=torch.bool),
                torch.as_tensor(True, dtype=torch.bool),
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
                                reward   += -0.3 * (cbr_delay / 1000.0)

                            elif 'embb' in slice_type.lower() or slice_idx == 0:
                                cbr_delay  = ran_info.get('cbr_delay', 0.0)
                                reward    += -0.2 * (cbr_delay / 1000.0)
                                cbr_th     = ran_info.get('cbr_th', 0.0)
                                vbr_th     = ran_info.get('vbr_th', 0.0)
                                throughput = (cbr_th + vbr_th) / 2.0
                                reward    += 0.2 * (throughput / 1e7)

                            elif 'mmtc' in slice_type.lower() or slice_idx == 1:
                                delay   = ran_info.get('delay', 0.0)
                                reward += -0.1 * (delay / 1000.0)

        remaining_prbs    = self._n_prbs - alloc_prbs.sum()
        reward           += 0.005 * (remaining_prbs / self._n_prbs)

        allocation_change = np.sum(np.abs(alloc_prbs - self._last_allocation))
        reward           -= 0.002 * (allocation_change / self._n_prbs)

        self._last_allocation = alloc_prbs.copy().astype(float)

        # Clip reward to prevent exploding gradients
        reward = np.clip(reward, -1.0, 1.0)
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
        if self._is_training_env:
            self._env.save_results()
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps


# ======================================================================== #

class TrainerCPO:
    """Train CPO agent with cycling curriculum learning."""

    def __init__(self):
        os.makedirs('./results/500k/CPO/', exist_ok=True)

    def train(self, run_id: int):
        global _CURRENT_RUN_ID, _ENV_INSTANCE_COUNT
        _CURRENT_RUN_ID = run_id
        _ENV_INSTANCE_COUNT[run_id] = 0

        print(f'\n{"="*60}')
        print(f'=== CPO SEQUENTIAL CURRICULUM Training Run {run_id} ===')
        print(f'{"="*60}')
        schedule = ' → '.join(f'{s}({n//1000}k)' for s, n in zip(SCENARIOS, STEPS_PER_SCENARIO))
        print(f'Schedule: {schedule}')
        print(f'Total steps: {_TOTAL_STEPS}')
        print(f'Output: results/500k/CPO/history_{run_id}.npz')

        custom_cfgs = {
            "train_cfgs": {
                "total_steps": _TOTAL_STEPS,
                "device": "cuda:0",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "update_iters": 3,
                "batch_size": 128,
                "target_kl": 0.02,
                "entropy_coef": 0.01,
                "cost_limit": 100.0,
                "use_max_grad_norm": True,
                "max_grad_norm": 0.5,
                "gamma": 0.99,
                "cost_gamma": 0.99,
                "lam": 0.95,
                "lam_c": 0.95,
                "cg_damping": 0.1,
                "cg_iters": 15,
                "use_cost": True,
                "obs_normalize": True,
                "reward_normalize": True,
                "cost_normalize": True,
            },
            "model_cfgs": {
                "actor": {"hidden_sizes": [256, 256], "activation": "tanh"},
                "critic": {"hidden_sizes": [256, 256], "activation": "tanh", "lr": 3e-4},
            },
            "logger_cfgs": {
                "use_wandb": False,
                "save_model_freq": 1,
            },
        }

        agent = omnisafe.Agent(
            algo="CPO",
            env_id=ENV_ID,
            custom_cfgs=custom_cfgs,
        )

        print('\nTraining started...')
        agent.learn()
        print('Training done!')

        try:
            agent.plot(smooth=1)
        except Exception as e:
            print(f'Plotting failed: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CPO with curriculum learning on RAN slicing"
    )
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--processes", type=int, default=PROCESSES)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS,
                        help="Total training steps (default: 1000000)")
    args = parser.parse_args()

    _TOTAL_STEPS = args.total_steps

    trainer = TrainerCPO()
    run_list = list(range(args.runs))

    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            results = list(E.map(trainer.train, run_list))
