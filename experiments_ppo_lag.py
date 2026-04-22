#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train PPO-Lag (OmniSafe) with SEQUENTIAL CURRICULUM across network-slicing scenarios.
Trains low (125k) → medium (250k) → congested (125k), one pass. Total 500k steps.
Single NPZ file per run recording reward, cost, violations, resources, throughput,
served_ratio.

Algorithm: PPO-Lag (Lagrangian relaxation of the CMDP constraint).

Action: [alloc_embb, alloc_mmtc, alloc_urllc, change_flag]  (shape n_slices+1)
  - alloc_* are proportional weights normalised to sum to 1, then scaled to n_prbs
  - change_flag ∈ [0,1]: < 0.5 → keep previous allocation (no reallocation cost);
                          ≥ 0.5 → apply new allocation (penalty if it changed)

Reward: maximise throughput for URLLC (weight 0.50) and eMBB (weight 0.35),
        maximise mMTC served-device ratio (weight 0.15),
        minus reallocation penalty (0.10 × |Δalloc| / n_prbs, only when change_flag≥0.5)

Cost: delay weighted by slice SLA threshold, normalised so cost≈1 when each slice
      is exactly at its SLA delay limit.
      cost = (3·urllc_delay/urllc_sla + 2·embb_delay/embb_sla + 1·mmtc_delay/mmtc_sla) / 6
      cost_limit = 1.0 → all delays at or below their SLA thresholds

Observation (13 features, all ∈ [0,1]):
  Per slice (4 features × 3 slices):
    [0] delay_ratio     = clip(delay / sla_delay, 0, 3) / 3
    [1] throughput_ratio = clip(throughput / sla_min_th, 0, 3) / 3  (served_ratio for mMTC)
    [2] queue_ratio     = clip(queue / sla_max_queue, 0, 3) / 3  (0 for mMTC)
    [3] prb_fraction    = allocated_prbs / n_prbs
  Global:
    [12] last_change_flag  (1 if allocation was changed on the previous step, else 0)

Usage:
    python experiments_ppo_lag.py
    python experiments_ppo_lag.py --runs 5 --sequential
    python experiments_ppo_lag.py --total-steps 200000
'''

import os
import argparse
import numpy as np
import concurrent.futures as cf
from gymnasium import spaces
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from config_loader import load_scenario
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper

# ── Training constants ────────────────────────────────────────────────────────
RUNS               = 30
PROCESSES          = 4
TOTAL_STEPS        = 500_000
SCENARIOS          = ['low', 'medium', 'congested']
STEPS_PER_SCENARIO = [125_000, 250_000, 125_000]   # must sum to TOTAL_STEPS
STEPS_PER_EPOCH    = 4_000
PENALTY            = 10.0

ENV_ID = "RanSlicePPOLag-v0"

_RNG          = np.random.default_rng(3)
_PENALTY      = PENALTY
_TOTAL_STEPS  = TOTAL_STEPS
_RESULTS_PATH = './results/500k/PPOLag/'

_scenario_configs_cache: dict = {}
_CURRENT_RUN_ID = 0
_ENV_INSTANCE_COUNT: dict = {}

# ── Slice indices ─────────────────────────────────────────────────────────────
_EMBB_IDX  = 0
_MMTC_IDX  = 1
_URLLC_IDX = 2


# ── Extended ReportWrapper ────────────────────────────────────────────────────

class PPOLagReportWrapper(ReportWrapper):
    """ReportWrapper extended with cost, throughput, and served-ratio tracking."""

    def reset_history(self):
        super().reset_history()
        self.cost_history         = np.zeros(self.steps, dtype=np.float32)
        self.throughput_history   = np.zeros(self.steps, dtype=np.float32)
        self.served_ratio_history = np.zeros(self.steps, dtype=np.float32)

    def record_extras(self, cost: float, throughput: float, served_ratio: float):
        """Record per-step cost/throughput/served-ratio. Call right after step()."""
        idx = self.step_counter - 1
        if 0 <= idx < self.steps:
            self.cost_history[idx]         = cost
            self.throughput_history[idx]   = throughput
            self.served_ratio_history[idx] = served_ratio

    def set_evaluation(self, eval_steps, new_path=None, change_name=False):
        super().set_evaluation(eval_steps, new_path=new_path, change_name=change_name)
        self.cost_history         = np.pad(self.cost_history,         [(0, eval_steps)])
        self.throughput_history   = np.pad(self.throughput_history,   [(0, eval_steps)])
        self.served_ratio_history = np.pad(self.served_ratio_history, [(0, eval_steps)])

    def save_results(self):
        np.savez(
            self.file_path,
            violation    = self.violation_history,
            reward       = self.reward_history,
            resources    = self.action_history,
            reallocations= self.reallocation_history,
            cost         = self.cost_history,
            throughput   = self.throughput_history,
            served_ratio = self.served_ratio_history,
        )


# ── SLA helpers ───────────────────────────────────────────────────────────────

def _get_scenario_config(name: str):
    if name not in _scenario_configs_cache:
        _scenario_configs_cache[name] = load_scenario('scenarios.yaml', name)
    return _scenario_configs_cache[name]


def _extract_sla(cfg) -> dict:
    """Pull SLA thresholds and mMTC device count out of a ScenarioConfig."""
    s = cfg.sla_config
    t = cfg.traffic_config
    e = s.get('embb',  {})
    m = s.get('mmtc',  {})
    u = s.get('urllc', {})
    return {
        'embb_cbr_delay' : e.get('cbr_delay',  500.0),
        'embb_vbr_delay' : e.get('vbr_delay',  500.0),
        'embb_cbr_th'    : e.get('cbr_th',      5e6),
        'embb_vbr_th'    : e.get('vbr_th',      5e6),
        'embb_cbr_queue' : e.get('cbr_queue',  150e3),
        'embb_vbr_queue' : e.get('vbr_queue',  150e3),
        'mmtc_delay'     : m.get('delay',     3000.0),
        'mmtc_max_dev'   : t.get('mmtc', {}).get('n_devices', 2500),
        'urllc_cbr_delay': u.get('cbr_delay',  100.0),
        'urllc_vbr_delay': u.get('vbr_delay',  100.0),
        'urllc_cbr_th'   : u.get('cbr_th',    200e3),
        'urllc_vbr_th'   : u.get('vbr_th',    100e3),
        'urllc_cbr_queue': u.get('cbr_queue',   50e3),
        'urllc_vbr_queue': u.get('vbr_queue',  100e3),
    }


# ── CMDP environment ──────────────────────────────────────────────────────────

@env_register
@env_unregister
class RanSlicePPOLagEnv(CMDP):
    """
    PPO-Lag CMDP environment with sequential curriculum, change-or-keep action,
    delay-based cost function, and throughput/served-ratio reward.
    """

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        global _ENV_INSTANCE_COUNT
        super().__init__(env_id)

        self._run_id = _CURRENT_RUN_ID

        _ENV_INSTANCE_COUNT.setdefault(self._run_id, 0)
        _ENV_INSTANCE_COUNT[self._run_id] += 1
        self._instance_id      = _ENV_INSTANCE_COUNT[self._run_id]
        self._is_training_env  = (self._instance_id == 1)

        print(f"[Run {self._run_id}] Creating PPOLag env instance {self._instance_id} "
              f"({'TRAINING' if self._is_training_env else 'EVAL'})")

        self.scenario_names   = SCENARIOS
        self.scenario_configs = [_get_scenario_config(n) for n in self.scenario_names]
        self.current_scenario_idx      = 0
        self._global_step_count        = 0
        self._steps_in_current_scenario = 0

        cfg     = self.scenario_configs[0]
        raw_env = create_env_from_config(cfg, _RNG, penalty=_PENALTY)

        os.makedirs(_RESULTS_PATH, exist_ok=True)

        total = _TOTAL_STEPS
        if self._is_training_env:
            self._env = PPOLagReportWrapper(
                raw_env,
                steps         = total,
                control_steps = 5000,
                env_id        = str(self._run_id),
                path          = _RESULTS_PATH,
                verbose       = False,
                continuous_mode = True,
            )
        else:
            self._env = PPOLagReportWrapper(
                raw_env,
                steps         = total,
                control_steps = total + 1,
                env_id        = f"{self._run_id}_eval",
                path          = _RESULTS_PATH,
                verbose       = False,
                continuous_mode = True,
            )

        self._max_episode_steps = 2000
        self._step_count        = 0
        self._n_prbs            = cfg.n_prbs
        self._n_slices          = self._env.n_slices

        # Action: [alloc_0..alloc_{n-1}, change_flag], all in [0,1]
        self._action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._n_slices + 1,), dtype=float
        )

        # Obs: 4 features per slice + 1 global change-flag memory = n_slices*4 + 1
        self._obs_size = self._n_slices * 4 + 1
        self._observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._obs_size,), dtype=float
        )

        # Equal initial allocation so "keep" is sensible on step 0
        self._last_allocation  = np.full(self._n_slices,
                                         self._n_prbs // self._n_slices, dtype=float)
        self._last_change_flag = 0.0

        self._sla = _extract_sla(cfg)

    # ── Curriculum ────────────────────────────────────────────────────────────

    def _switch_scenario(self):
        if not self._is_training_env:
            return
        next_idx = self.current_scenario_idx + 1
        if next_idx >= len(self.scenario_names):
            return  # already on last scenario
        self.current_scenario_idx = next_idx
        name = self.scenario_names[self.current_scenario_idx]
        cfg  = self.scenario_configs[self.current_scenario_idx]

        print(f"\n[Run {self._run_id}] === ADVANCING to {name.upper()} "
              f"(global step {self._global_step_count}) ===")

        if hasattr(self._env, 'env') and hasattr(self._env.env, 'close'):
            self._env.env.close()

        self._env.env  = create_env_from_config(cfg, _RNG, penalty=_PENALTY)
        self._n_prbs   = cfg.n_prbs
        self._sla      = _extract_sla(cfg)
        self._steps_in_current_scenario = 0
        self._step_count = 0

    # ── Observation ───────────────────────────────────────────────────────────

    def _compute_observation(self, info: dict) -> np.ndarray:
        """
        13-dim obs, all features in [0, 1], normalised by the current SLA thresholds.

        Slice layout (4 features each):
          [base+0] delay_ratio  = clip(delay / sla_delay,  0, 3) / 3
          [base+1] th_ratio     = clip(th / sla_min_th,    0, 3) / 3   (served_ratio for mMTC)
          [base+2] queue_ratio  = clip(queue / sla_queue,  0, 3) / 3   (0 for mMTC)
          [base+3] prb_fraction = alloc_prbs / n_prbs
        Global:
          [12] last_change_flag
        """
        obs         = np.zeros(self._obs_size, dtype=float)
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [self._n_prbs // self._n_slices] * self._n_slices)
        sla         = self._sla

        def _safe(x, ref):
            return float(x) / max(float(ref), 1e-9)

        for si in range(self._n_slices):
            base     = si * 4
            prb_frac = float(n_prbs_list[si]) / max(self._n_prbs, 1) \
                       if si < len(n_prbs_list) else 0.0

            if si < len(l1_info) and isinstance(l1_info[si], dict):
                for ran_id, ran_info in l1_info[si].items():
                    if not isinstance(ran_info, dict):
                        continue

                    if si == _EMBB_IDX:
                        cbr_delay = ran_info.get('cbr_delay', 0.0)
                        cbr_th    = ran_info.get('cbr_th',    0.0)
                        vbr_th    = ran_info.get('vbr_th',    0.0)
                        cbr_q     = ran_info.get('cbr_queue', 0.0)
                        vbr_q     = ran_info.get('vbr_queue', 0.0)

                        obs[base]   = np.clip(_safe(cbr_delay, sla['embb_cbr_delay']), 0, 3) / 3
                        obs[base+1] = np.clip(_safe(cbr_th + vbr_th,
                                                    sla['embb_cbr_th'] + sla['embb_vbr_th']),
                                              0, 3) / 3
                        obs[base+2] = np.clip(_safe(cbr_q + vbr_q,
                                                    sla['embb_cbr_queue'] + sla['embb_vbr_queue']),
                                              0, 3) / 3

                    elif si == _MMTC_IDX:
                        delay   = ran_info.get('delay',   0.0)
                        devices = ran_info.get('devices', 0.0)

                        obs[base]   = np.clip(_safe(delay, sla['mmtc_delay']), 0, 3) / 3
                        obs[base+1] = 1.0 - np.clip(_safe(devices, sla['mmtc_max_dev']), 0, 1)
                        obs[base+2] = 0.0  # no queue concept for mMTC

                    elif si == _URLLC_IDX:
                        cbr_delay = ran_info.get('cbr_delay', 0.0)
                        cbr_th    = ran_info.get('cbr_th',    0.0)
                        vbr_th    = ran_info.get('vbr_th',    0.0)
                        cbr_q     = ran_info.get('cbr_queue', 0.0)
                        vbr_q     = ran_info.get('vbr_queue', 0.0)

                        obs[base]   = np.clip(_safe(cbr_delay, sla['urllc_cbr_delay']), 0, 3) / 3
                        obs[base+1] = np.clip(_safe(cbr_th + vbr_th,
                                                    sla['urllc_cbr_th'] + sla['urllc_vbr_th']),
                                              0, 3) / 3
                        obs[base+2] = np.clip(_safe(cbr_q + vbr_q,
                                                    sla['urllc_cbr_queue'] + sla['urllc_vbr_queue']),
                                              0, 3) / 3

            obs[base+3] = np.clip(prb_frac, 0.0, 1.0)

        # Global: was allocation changed last step?
        obs[self._n_slices * 4] = self._last_change_flag
        return obs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, alloc_prbs: np.ndarray, change_flag: float,
                        info: dict) -> float:
        """
        reward = 0.50 × URLLC_th_ratio
               + 0.35 × eMBB_th_ratio
               + 0.15 × mMTC_served_ratio
               − 0.10 × (|Δalloc| / n_prbs)   [only when change_flag ≥ 0.5]

        All throughput ratios are normalised by the corresponding SLA minimum
        throughput requirements and clipped to [0, 1].
        """
        reward   = 0.0
        l1_info  = info.get('l1_info', [])
        sla      = self._sla

        for si in range(self._n_slices):
            if si >= len(l1_info) or not isinstance(l1_info[si], dict):
                continue
            for ran_id, ran_info in l1_info[si].items():
                if not isinstance(ran_info, dict):
                    continue

                if si == _EMBB_IDX:
                    cbr_th = ran_info.get('cbr_th', 0.0)
                    vbr_th = ran_info.get('vbr_th', 0.0)
                    ref    = sla['embb_cbr_th'] + sla['embb_vbr_th']
                    ratio  = np.clip((cbr_th + vbr_th) / max(ref, 1e-9), 0.0, 3.0) / 3.0
                    reward += 0.35 * ratio

                elif si == _MMTC_IDX:
                    devices = ran_info.get('devices', 0.0)
                    served  = 1.0 - np.clip(devices / max(sla['mmtc_max_dev'], 1), 0.0, 1.0)
                    reward += 0.15 * served

                elif si == _URLLC_IDX:
                    cbr_th = ran_info.get('cbr_th', 0.0)
                    vbr_th = ran_info.get('vbr_th', 0.0)
                    ref    = sla['urllc_cbr_th'] + sla['urllc_vbr_th']
                    ratio  = np.clip((cbr_th + vbr_th) / max(ref, 1e-9), 0.0, 3.0) / 3.0
                    reward += 0.50 * ratio

        # Reallocation penalty – only charged when the agent elected to change
        if change_flag >= 0.5:
            delta   = np.sum(np.abs(alloc_prbs.astype(float) - self._last_allocation))
            reward -= 0.10 * (delta / max(self._n_prbs, 1))

        return float(np.clip(reward, -0.2, 1.0))

    # ── Cost ──────────────────────────────────────────────────────────────────

    def _compute_cost(self, info: dict) -> float:
        """
        Delay-based cost normalised so that cost ≈ 1.0 when every slice sits
        exactly at its SLA delay threshold:

          cost = (3·urllc_delay/urllc_sla + 2·embb_delay/embb_sla + 1·mmtc_delay/mmtc_sla) / 6

        Weights reflect slice priority (URLLC > eMBB > mMTC).
        cost_limit = 1.0 constrains mean delays to their SLA limits.
        """
        cost    = 0.0
        l1_info = info.get('l1_info', [])
        sla     = self._sla

        for si in range(self._n_slices):
            if si >= len(l1_info) or not isinstance(l1_info[si], dict):
                continue
            for ran_id, ran_info in l1_info[si].items():
                if not isinstance(ran_info, dict):
                    continue

                if si == _EMBB_IDX:
                    delay = ran_info.get('cbr_delay', 0.0)
                    cost += (2.0 / 6.0) * (delay / max(sla['embb_cbr_delay'], 1e-9))

                elif si == _MMTC_IDX:
                    delay = ran_info.get('delay', 0.0)
                    cost += (1.0 / 6.0) * (delay / max(sla['mmtc_delay'], 1e-9))

                elif si == _URLLC_IDX:
                    delay = ran_info.get('cbr_delay', 0.0)
                    cost += (3.0 / 6.0) * (delay / max(sla['urllc_cbr_delay'], 1e-9))

        return float(cost)

    # ── Tracking helpers ──────────────────────────────────────────────────────

    def _total_throughput(self, info: dict) -> float:
        """Total bits transmitted this step across eMBB and URLLC."""
        l1_info = info.get('l1_info', [])
        total   = 0.0
        for si in [_EMBB_IDX, _URLLC_IDX]:
            if si < len(l1_info) and isinstance(l1_info[si], dict):
                for ran_id, ran_info in l1_info[si].items():
                    if isinstance(ran_info, dict):
                        total += ran_info.get('cbr_th', 0.0) + ran_info.get('vbr_th', 0.0)
        return total

    def _served_ratio(self, info: dict) -> float:
        """mMTC served-device ratio proxy: 1 − queued / max_devices."""
        l1_info = info.get('l1_info', [])
        if _MMTC_IDX < len(l1_info) and isinstance(l1_info[_MMTC_IDX], dict):
            for ran_id, ran_info in l1_info[_MMTC_IDX].items():
                if isinstance(ran_info, dict) and 'devices' in ran_info:
                    dev = ran_info['devices']
                    return 1.0 - float(np.clip(dev / max(self._sla['mmtc_max_dev'], 1), 0, 1))
        return 0.0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._step_count = 0
        result = self._env.reset()
        obs_raw, info = result if isinstance(result, tuple) else (result, {})
        obs = self._compute_observation(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action: torch.Tensor):
        act = action.cpu().numpy()
        act = np.abs(act)

        change_flag = float(act[self._n_slices])   # last element
        alloc_raw   = act[:self._n_slices]

        total = alloc_raw.sum()
        if total > 0:
            alloc_raw = alloc_raw / total

        new_alloc = np.array(
            [int(np.floor(a * self._n_prbs)) for a in alloc_raw], dtype=int
        )

        if change_flag < 0.5:
            # Keep previous allocation; no reallocation penalty this step
            alloc_prbs = self._last_allocation.copy().astype(int)
        else:
            alloc_prbs = new_alloc

        result = self._env.step(alloc_prbs)

        if len(result) == 4:
            obs_raw, _env_reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs_raw, _env_reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        reward = self._compute_reward(alloc_prbs, change_flag, info)
        cost   = self._compute_cost(info)

        # Record extras in the extended wrapper
        self._env.record_extras(cost, self._total_throughput(info), self._served_ratio(info))

        # Update allocation state
        if change_flag >= 0.5:
            self._last_change_flag = 1.0 if not np.array_equal(alloc_prbs,
                                                                self._last_allocation.astype(int)) \
                                         else 0.0
            self._last_allocation = alloc_prbs.copy().astype(float)
        else:
            self._last_change_flag = 0.0

        self._step_count        += 1
        self._global_step_count += 1
        self._steps_in_current_scenario += 1

        # Advance curriculum if this scenario's budget is exhausted
        scenario_budget = STEPS_PER_SCENARIO[self.current_scenario_idx]
        if self._is_training_env and self._steps_in_current_scenario >= scenario_budget:
            self._switch_scenario()
            _r = self._env.reset()
            _info = _r[1] if isinstance(_r, tuple) else {}
            obs = self._compute_observation(_info)
            return (
                torch.as_tensor(obs,                   dtype=torch.float32),
                torch.as_tensor(reward,                dtype=torch.float32),
                torch.as_tensor(cost,                  dtype=torch.float32),
                torch.as_tensor(False,                 dtype=torch.bool),
                torch.as_tensor(True,                  dtype=torch.bool),
                {"final_observation": torch.as_tensor(obs, dtype=torch.float32)},
            )

        if self._step_count >= self._max_episode_steps:
            truncated        = True
            self._step_count = 0

        info = {str(k): v for k, v in info.items()} if isinstance(info, dict) else {}
        obs  = self._compute_observation(info)

        if terminated or truncated:
            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            _r = self._env.reset()
            _info = _r[1] if isinstance(_r, tuple) else {}
            new_obs = self._compute_observation(_info)
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

    def render(self, *args, **kwargs):
        if hasattr(self._env, 'render'):
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


# ── Trainer ───────────────────────────────────────────────────────────────────

class TrainerPPOLag:

    def __init__(self):
        os.makedirs(_RESULTS_PATH, exist_ok=True)

    def train(self, run_id: int):
        global _CURRENT_RUN_ID, _ENV_INSTANCE_COUNT
        _CURRENT_RUN_ID = run_id
        _ENV_INSTANCE_COUNT[run_id] = 0

        print(f'\n{"="*60}')
        print(f'=== PPO-Lag SEQUENTIAL CURRICULUM  Run {run_id} ===')
        print(f'{"="*60}')
        schedule = ' → '.join(f'{s}({n//1000}k)' for s, n in zip(SCENARIOS, STEPS_PER_SCENARIO))
        print(f'Schedule : {schedule}')
        print(f'Total    : {_TOTAL_STEPS} steps')
        print(f'Output   : {_RESULTS_PATH}history_{run_id}.npz')

        custom_cfgs = {
            "train_cfgs": {
                "total_steps": _TOTAL_STEPS,
                "device": "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "update_iters": 40,
                "batch_size": 64,
                "target_kl": 0.02,
                "entropy_coef": 0.0,
                "gamma": 0.99,
                "cost_gamma": 0.99,
                "lam": 0.95,
                "lam_c": 0.95,
                "use_cost": True,
                "obs_normalize": True,
                "reward_normalize": True,
                "cost_normalize": False,  # keep cost in natural units for Lagrangian
            },
            "lagrange_cfgs": {
                "cost_limit": 1.0,   # delays should be at/below their SLA thresholds
                "lagrangian_multiplier_init": 0.1,
                "lambda_lr": 0.035,
                "lambda_optimizer": "Adam",
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

        agent = omnisafe.Agent(
            algo="PPOLag",
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train PPO-Lag with sequential curriculum on RAN slicing'
    )
    parser.add_argument('--runs',        type=int,  default=RUNS)
    parser.add_argument('--processes',   type=int,  default=PROCESSES)
    parser.add_argument('--sequential',  action='store_true')
    parser.add_argument('--total-steps', type=int,  default=TOTAL_STEPS,
                        help='Total training steps (default: 500000)')
    args = parser.parse_args()

    _TOTAL_STEPS = args.total_steps

    trainer  = TrainerPPOLag()
    run_list = list(range(args.runs))

    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            list(E.map(trainer.train, run_list))
