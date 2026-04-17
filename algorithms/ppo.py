#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO algorithm for RAN slicing using OmniSafe.

Cycling curriculum: low -> medium -> congested -> low -> ...
with configurable steps per block.

Classes:
    RanSlicePPOEnv  — OmniSafe CMDP environment
    TrainerPPO      — Training orchestrator
"""

import os
import sys
import numpy as np
from gymnasium import spaces
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wrapper import ReportWrapper
from config_loader import load_scenario
from scenario_creator import create_env_from_config

# ── Default configuration ──
DEFAULT_SCENARIOS        = ['low', 'medium', 'congested']
DEFAULT_STEPS_PER_SCENARIO = 20000
DEFAULT_TOTAL_STEPS      = 500000
DEFAULT_PENALTY          = 100.0
DEFAULT_RESULTS_PATH     = './pipelines/ppo/'

ENV_ID = "RanSlicePPOPipeline-v0"

# ── Module-level mutable state (set by TrainerPPO before each run) ──
_rng                  = np.random.default_rng(3)
_penalty              = DEFAULT_PENALTY
_total_steps          = DEFAULT_TOTAL_STEPS
_steps_per_scenario   = DEFAULT_STEPS_PER_SCENARIO
_results_path         = DEFAULT_RESULTS_PATH
_scenarios            = list(DEFAULT_SCENARIOS)
_current_run_id       = 0
_env_instance_count   = {}
_scenario_configs_cache = {}


def _get_scenario_config(scenario_name):
    if scenario_name not in _scenario_configs_cache:
        _scenario_configs_cache[scenario_name] = load_scenario('scenarios.yaml', scenario_name)
    return _scenario_configs_cache[scenario_name]


@env_register
@env_unregister
class RanSlicePPOEnv(CMDP):
    """Cycling-curriculum PPO environment for RAN slicing."""

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        global _env_instance_count
        super().__init__(env_id)

        self._run_id = _current_run_id
        if self._run_id not in _env_instance_count:
            _env_instance_count[self._run_id] = 0
        _env_instance_count[self._run_id] += 1
        self._instance_id = _env_instance_count[self._run_id]
        self._is_training_env = (self._instance_id == 1)

        print(f"[Run {self._run_id}] Creating PPO env instance {self._instance_id} "
              f"({'TRAINING' if self._is_training_env else 'EVAL'})")

        self.scenario_names   = list(_scenarios)
        self.scenario_configs = [_get_scenario_config(n) for n in self.scenario_names]
        self.current_scenario_idx = 0
        self._global_step_count = 0

        cfg     = self.scenario_configs[0]
        raw_env = create_env_from_config(cfg, _rng, penalty=_penalty)

        self._results_path = _results_path
        os.makedirs(self._results_path, exist_ok=True)

        if self._is_training_env:
            self._env = ReportWrapper(
                raw_env, steps=_total_steps, control_steps=5000,
                env_id=str(self._run_id), path=self._results_path,
                verbose=False, continuous_mode=True,
            )
        else:
            self._env = ReportWrapper(
                raw_env, steps=_total_steps, control_steps=_total_steps + 1,
                env_id=f"{self._run_id}_eval", path=self._results_path,
                verbose=False, continuous_mode=True,
            )

        self._max_episode_steps = 500
        self._n_prbs   = cfg.n_prbs
        self._step_count = 0
        self._n_slices = self._env.n_slices
        self._obs_size = self._n_slices * 4

        self._action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._n_slices + 1,), dtype=float,
        )
        self._observation_space = spaces.Box(
            low=-1, high=1, shape=(self._obs_size,), dtype=float,
        )
        self._last_allocation = np.zeros(self._n_slices, dtype=float)
        self._steps_in_current_scenario = 0

    # ── scenario cycling ──
    def _switch_scenario(self):
        if not self._is_training_env:
            return
        self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenario_names)
        cfg = self.scenario_configs[self.current_scenario_idx]
        print(f"\n[Run {self._run_id}] === CYCLING to "
              f"{self.scenario_names[self.current_scenario_idx].upper()} "
              f"(global step {self._global_step_count}) ===")
        if hasattr(self._env, 'env') and hasattr(self._env.env, 'close'):
            self._env.env.close()
        self._env.env = create_env_from_config(cfg, _rng, penalty=_penalty)
        self._n_prbs = cfg.n_prbs
        self._steps_in_current_scenario = 0
        self._step_count = 0

    # ── reset / obs ──
    def reset(self, seed=None, options=None):
        self._step_count = 0
        result = self._env.reset()
        obs, info = (result if isinstance(result, tuple) else (result, {}))
        return torch.as_tensor(self._compute_observation(info), dtype=torch.float32), info

    def _compute_observation(self, info):
        obs = np.zeros(self._obs_size, dtype=float)
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [])
        idx = 0
        for s in range(self._n_slices):
            bler = snr = traffic = 0.0
            if s < len(l1_info):
                si = l1_info[s]
                if isinstance(si, dict):
                    for _, ri in si.items():
                        if isinstance(ri, dict):
                            bler    = max(ri.get('cbr_bler', 0.0), ri.get('vbr_bler', 0.0))
                            snr     = (ri.get('cbr_snr', 0.0) + ri.get('vbr_snr', 0.0)) / 2.0
                            traffic = (ri.get('cbr_queue', 0.0) + ri.get('vbr_queue', 0.0)) / 2.0
            bler    = np.clip(bler, 0, 1)
            snr     = np.clip(snr / 30.0, -1, 1)
            traffic = np.clip(traffic / 100000.0, 0, 1)
            alloc   = np.clip(n_prbs_list[s] / max(self._n_prbs, 1), 0, 1) if s < len(n_prbs_list) else 0.0
            obs[idx]                        = bler
            obs[idx + self._n_slices]       = snr
            obs[idx + 2 * self._n_slices]   = traffic
            obs[idx + 3 * self._n_slices]   = alloc
            idx += 1
        return obs

    # ── step ──
    def step(self, action: torch.Tensor):
        act = np.abs(action.cpu().numpy())
        total = act.sum()
        if total > 0:
            act = act / total
        alloc  = act[:self._n_slices]
        excess = act[self._n_slices] if len(act) > self._n_slices else 0.0
        alloc_prbs = np.array([int(np.floor(a * self._n_prbs)) for a in alloc], dtype=int)

        result = self._env.step(alloc_prbs)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost   = float(info.get("cost", 0.0))
        reward = self._compute_custom_reward(alloc_prbs, excess, info)

        self._step_count += 1
        self._global_step_count += 1
        self._steps_in_current_scenario += 1

        if self._is_training_env and self._steps_in_current_scenario >= _steps_per_scenario:
            self._switch_scenario()
            _, new_info = self._env.reset()
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
            _, new_info = self._env.reset()
            obs = torch.as_tensor(self._compute_observation(new_info), dtype=torch.float32)
            info["final_observation"] = final_obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            info["final_observation"] = obs

        return (obs,
                torch.as_tensor(reward, dtype=torch.float32),
                torch.as_tensor(cost, dtype=torch.float32),
                torch.as_tensor(terminated, dtype=torch.bool),
                torch.as_tensor(truncated, dtype=torch.bool),
                info)

    def _compute_custom_reward(self, alloc_prbs, excess, info):
        reward  = 0.0
        l1_info = info.get('l1_info', [])
        for s in range(self._n_slices):
            if s < len(l1_info):
                si = l1_info[s]
                if isinstance(si, dict):
                    for _, ri in si.items():
                        if isinstance(ri, dict):
                            st = ri.get('type', '')
                            if 'urllc' in st.lower() or s == 2:
                                reward += -3.0 * (ri.get('cbr_delay', 0.0) / 100.0)
                            elif 'embb' in st.lower() or s == 0:
                                reward += -2.0 * (ri.get('cbr_delay', 0.0) / 100.0)
                                reward += 2.0 * ((ri.get('cbr_th', 0.0) + ri.get('vbr_th', 0.0)) / 2.0 / 1e6)
                            elif 'mmtc' in st.lower() or s == 1:
                                reward += -1.0 * (ri.get('delay', 0.0) / 100.0)
        reward += 0.1 * ((self._n_prbs - alloc_prbs.sum()) / self._n_prbs)
        reward -= 0.05 * (np.sum(np.abs(alloc_prbs - self._last_allocation)) / self._n_prbs)
        self._last_allocation = alloc_prbs.copy().astype(float)
        return reward

    def render(self, *args, **kwargs):
        return self._env.render() if hasattr(self._env, "render") else None

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


class TrainerPPO:
    """Train PPO agent with cycling curriculum learning."""

    def __init__(self, results_path=DEFAULT_RESULTS_PATH,
                 total_steps=DEFAULT_TOTAL_STEPS,
                 steps_per_scenario=DEFAULT_STEPS_PER_SCENARIO,
                 scenarios=None, penalty=DEFAULT_PENALTY, device="cuda:0"):
        self.results_path = results_path
        self.total_steps = total_steps
        self.steps_per_scenario = steps_per_scenario
        self.scenarios = scenarios or list(DEFAULT_SCENARIOS)
        self.penalty = penalty
        self.device = device
        os.makedirs(self.results_path, exist_ok=True)

    def train(self, run_id: int):
        global _current_run_id, _env_instance_count, _total_steps
        global _steps_per_scenario, _results_path, _scenarios, _penalty
        _current_run_id      = run_id
        _env_instance_count[run_id] = 0
        _total_steps         = self.total_steps
        _steps_per_scenario  = self.steps_per_scenario
        _results_path        = self.results_path
        _scenarios           = self.scenarios
        _penalty             = self.penalty

        print(f'\n{"="*60}')
        print(f'=== PPO Training Run {run_id} ===')
        print(f'{"="*60}')
        print(f'Scenarios cycle: {" -> ".join(self.scenarios)} (repeating)')
        print(f'Steps per scenario block: {self.steps_per_scenario}')
        print(f'Total steps: {self.total_steps}')
        print(f'Output: {self.results_path}history_{run_id}.npz')

        custom_cfgs = {
            "train_cfgs": {"total_steps": self.total_steps, "device": self.device},
            "algo_cfgs": {
                "steps_per_epoch": 2000, "update_iters": 10, "batch_size": 128,
                "target_kl": 0.02, "entropy_coef": 0.01,
                "gamma": 0.99, "cost_gamma": 0.99, "lam": 0.95, "lam_c": 0.95,
                "use_cost": True,
            },
            "model_cfgs": {
                "actor":  {"hidden_sizes": [64, 64], "activation": "tanh"},
                "critic": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 0.001},
            },
            "logger_cfgs": {"use_wandb": False, "save_model_freq": 1},
        }

        agent = omnisafe.Agent(algo="PPO", env_id=ENV_ID, custom_cfgs=custom_cfgs)
        print('\nTraining started...')
        agent.learn()
        print('Training done!')

        try:
            agent.plot(smooth=1)
        except Exception as e:
            print(f'Plotting failed: {e}')
