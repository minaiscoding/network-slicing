#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CMDP environment wrapper for all OmniSafe algorithms.
Registers a single env (RanSlicePipeline-v1) usable by CPO, PPO, PPOLag, SACLag, TD3Lag.

All module-level globals (_RESULTS_PATH, _PENALTY, etc.) can be patched by the pipeline
before training to redirect outputs and configure behaviour.
"""

import os
import sys
import numpy as np
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wrapper import ReportWrapper
from config_loader import load_scenario
from scenario_creator import create_env_from_config

# ── Constants ──
TOTAL_STEPS         = 2000000
SCENARIOS           = ['low', 'medium', 'congested']
STEPS_PER_SCENARIO  = [500000, 1000000, 500000]
STEPS_PER_EPOCH     = 8000

ENV_ID = "RanSlicePipeline-v1"

# ── Module-level mutable state (patched by pipeline before each run) ──
_RNG           = np.random.default_rng(3)
_PENALTY       = 10.0
_TOTAL_STEPS   = TOTAL_STEPS
_RESULTS_PATH  = './results/pipeline/'
_EPOCH_COUNTER = 0

_scenario_configs_cache = {}

_CURRENT_RUN_ID     = 0
_ENV_INSTANCE_COUNT = {}


def get_scenario_config(scenario_name):
    if scenario_name not in _scenario_configs_cache:
        _scenario_configs_cache[scenario_name] = load_scenario('scenarios.yaml', scenario_name)
    return _scenario_configs_cache[scenario_name]


@env_register
@env_unregister
class RanSliceEnv(CMDP):
    """
    Curriculum-learning CMDP environment for RAN network slicing.
    Sequential: low(500k) → medium(1M) → congested(500k).

    Shared across all OmniSafe algorithms (CPO, PPO, PPOLag, SACLag, TD3Lag).
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    # ------------------------------------------------------------------ #
    #  Init
    # ------------------------------------------------------------------ #
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

        # Scenario configs
        self.scenario_names   = SCENARIOS
        self.scenario_configs = [get_scenario_config(n) for n in self.scenario_names]
        self.current_scenario_idx = 0
        self._global_step_count   = 0

        cfg     = self.scenario_configs[0]
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)

        self._results_path = _RESULTS_PATH
        os.makedirs(self._results_path, exist_ok=True)

        if self._is_training_env:
            self._env = ReportWrapper(
                raw_env,
                steps           = _TOTAL_STEPS,
                control_steps   = 5000,
                env_id          = str(self._run_id),
                path            = self._results_path,
                verbose         = False,
                continuous_mode = True,
            )
        else:
            self._env = ReportWrapper(
                raw_env,
                steps           = _TOTAL_STEPS,
                control_steps   = _TOTAL_STEPS + 1,
                env_id          = f"{self._run_id}_eval",
                path            = self._results_path,
                verbose         = False,
                continuous_mode = True,
            )

        self._max_episode_steps = 2000
        self._n_prbs            = cfg.n_prbs
        self._step_count        = 0

        self._action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._env.n_slices + 1,), dtype=float,
        )

        self._n_slices = self._env.n_slices
        self._obs_size = self._n_slices * 4
        self._observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self._obs_size,), dtype=float,
        )

        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        self._steps_in_current_scenario = 0

        # Emergency PRB redistribution
        self._consecutive_violations = np.zeros(self._n_slices, dtype=int)
        self._priority_order   = [2, 0, 1]   # URLLC > eMBB > mMTC
        self._EXCESS_THRESHOLD = 2
        self._STEAL_THRESHOLD  = 5
        self._MMTC_IDX         = 1
        self._MMTC_MIN_PRBS    = 5

    # ------------------------------------------------------------------ #
    #  Curriculum switching
    # ------------------------------------------------------------------ #
    def _switch_scenario(self):
        if not self._is_training_env:
            return
        next_idx = self.current_scenario_idx + 1
        if next_idx >= len(self.scenario_names):
            return
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

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        self._step_count = 0
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        obs = self._compute_observation(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #
    def _compute_observation(self, info):
        obs         = np.zeros(self._obs_size, dtype=float)
        idx         = 0
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])

        for slice_idx in range(self._n_slices):
            bler = snr = traffic = 0.0

            if slice_idx < len(l1_info):
                slice_info = l1_info[slice_idx]
                if isinstance(slice_info, dict):
                    for ran_id, ran_info in slice_info.items():
                        if isinstance(ran_info, dict):
                            bler    = max(ran_info.get('cbr_bler', 0.0),
                                          ran_info.get('vbr_bler', 0.0))
                            snr     = (ran_info.get('cbr_snr', 0.0) +
                                       ran_info.get('vbr_snr', 0.0)) / 2.0
                            traffic = (ran_info.get('cbr_queue', 0.0) +
                                       ran_info.get('vbr_queue', 0.0)) / 2.0

            bler    = np.clip(bler, 0, 1)
            snr     = np.clip(snr / 30.0, -1, 1)
            traffic = np.clip(traffic / 100000.0, 0, 1)
            alloc   = (n_prbs_list[slice_idx] / max(self._n_prbs, 1)
                       if slice_idx < len(n_prbs_list) else 0.0)
            alloc   = np.clip(alloc, 0, 1)

            obs[idx]                        = bler
            obs[idx + self._n_slices]       = snr
            obs[idx + 2 * self._n_slices]   = traffic
            obs[idx + 3 * self._n_slices]   = alloc
            idx += 1

        return obs

    # ------------------------------------------------------------------ #
    #  Step
    # ------------------------------------------------------------------ #
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
        excess_prbs = self._n_prbs - alloc_prbs.sum()

        emergency_slices = [s for s in self._priority_order
                            if self._consecutive_violations[s] >= self._EXCESS_THRESHOLD]
        if emergency_slices and excess_prbs > 0:
            alloc_prbs[emergency_slices[0]] += excess_prbs

        severe_slices = [s for s in self._priority_order
                         if (self._consecutive_violations[s] >= self._STEAL_THRESHOLD
                             and s != self._MMTC_IDX)]
        if severe_slices:
            mmtc_prbs = alloc_prbs[self._MMTC_IDX]
            stealable = max(0, mmtc_prbs - self._MMTC_MIN_PRBS)
            if stealable > 0:
                alloc_prbs[self._MMTC_IDX] -= stealable
                alloc_prbs[severe_slices[0]] += stealable

        # ── Env step ──
        result = self._env.step(alloc_prbs)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost   = float(info.get("cost", 0.0))
        reward = self._compute_custom_reward(alloc_prbs, excess, info)

        # ── Violation tracking ──
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

        # ── Epoch-boundary reseeding ──
        global _EPOCH_COUNTER, _RNG
        if self._is_training_env and self._global_step_count % STEPS_PER_EPOCH == 0:
            _EPOCH_COUNTER += 1
            new_seed = 3 + _EPOCH_COUNTER * 1000 + self._run_id
            _RNG = np.random.default_rng(new_seed)

        # ── Curriculum check ──
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
        obs  = self._compute_observation(info)

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

    # ------------------------------------------------------------------ #
    #  Reward
    # ------------------------------------------------------------------ #
    def _compute_custom_reward(self, alloc_prbs, excess, info):
        reward  = 0.0
        l1_info = info.get('l1_info', [])

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
        reward = np.clip(reward, -1.0, 1.0)
        return reward

    # ------------------------------------------------------------------ #
    #  Boilerplate
    # ------------------------------------------------------------------ #
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
