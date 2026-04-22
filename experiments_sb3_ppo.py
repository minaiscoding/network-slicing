#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Train PPO (stable-baselines3) with SEQUENTIAL CURRICULUM across network-slicing
scenarios.  Trains low (125k) → medium (250k) → congested (125k), one pass.
Total 500k steps per run.  Single NPZ file per run.

Algorithm : PPO (stable-baselines3, no cost / Lagrangian)

Action     : [alloc_embb, alloc_mmtc, alloc_urllc, change_flag]  ∈ [0,1]^4
  - alloc_* proportional weights normalised to sum 1, then scaled to n_prbs
  - change_flag < 0.5 → keep previous allocation (no reallocation penalty)
  - change_flag ≥ 0.5 → apply new allocation (penalty proportional to |Δalloc|)

Reward     : identical to PPO-Lag (no cost term):
  0.50 × URLLC_throughput_ratio
+ 0.35 × eMBB_throughput_ratio
+ 0.15 × mMTC_served_device_ratio
− 0.10 × (|Δalloc| / n_prbs)  [only when change_flag ≥ 0.5]
  Clipped to [−0.2, 1.0]

Observation: 13 features, all ∈ [0,1], normalised by current SLA thresholds
  Per slice (4 features × 3 slices):
    delay_ratio     = clip(delay / sla_delay, 0, 3) / 3
    throughput_ratio= clip(th / sla_min_th,   0, 3) / 3  (served_ratio for mMTC)
    queue_ratio     = clip(queue / sla_queue,  0, 3) / 3  (0 for mMTC)
    prb_fraction    = allocated_prbs / n_prbs
  Global:
    last_change_flag  (1 if allocation was changed last step, else 0)

Usage:
    python experiments_sb3_ppo.py
    python experiments_sb3_ppo.py --runs 5 --sequential
    python experiments_sb3_ppo.py --total-steps 200000
'''

import os
import argparse
import numpy as np
import concurrent.futures as cf
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from config_loader import load_scenario
from scenario_creator import create_env_from_config
from wrapper import ReportWrapper

# ── Training constants ────────────────────────────────────────────────────────
RUNS               = 30
PROCESSES          = 4
TOTAL_STEPS        = 500_000
SCENARIOS          = ['low', 'medium', 'congested']
STEPS_PER_SCENARIO = [125_000, 250_000, 125_000]   # must sum to TOTAL_STEPS
PENALTY            = 10.0
RESULTS_PATH       = './results/500k/SB3_PPO/'

# ── Slice indices ─────────────────────────────────────────────────────────────
_EMBB_IDX  = 0
_MMTC_IDX  = 1
_URLLC_IDX = 2

_scenario_configs_cache: dict = {}


# ── Extended ReportWrapper ────────────────────────────────────────────────────

class SB3ReportWrapper(ReportWrapper):
    """ReportWrapper extended with throughput and served-ratio tracking."""

    def reset_history(self):
        super().reset_history()
        self.throughput_history   = np.zeros(self.steps, dtype=np.float32)
        self.served_ratio_history = np.zeros(self.steps, dtype=np.float32)

    def record_extras(self, throughput: float, served_ratio: float):
        """Call right after wrapper.step() to record per-step metrics."""
        idx = self.step_counter - 1
        if 0 <= idx < self.steps:
            self.throughput_history[idx]   = throughput
            self.served_ratio_history[idx] = served_ratio

    def set_evaluation(self, eval_steps, new_path=None, change_name=False):
        super().set_evaluation(eval_steps, new_path=new_path, change_name=change_name)
        self.throughput_history   = np.pad(self.throughput_history,   [(0, eval_steps)])
        self.served_ratio_history = np.pad(self.served_ratio_history, [(0, eval_steps)])

    def save_results(self):
        np.savez(
            self.file_path,
            violation    = self.violation_history,
            reward       = self.reward_history,
            resources    = self.action_history,
            reallocations= self.reallocation_history,
            throughput   = self.throughput_history,
            served_ratio = self.served_ratio_history,
        )


# ── SLA helpers ───────────────────────────────────────────────────────────────

def _get_scenario_config(name: str):
    if name not in _scenario_configs_cache:
        _scenario_configs_cache[name] = load_scenario('scenarios.yaml', name)
    return _scenario_configs_cache[name]


def _extract_sla(cfg) -> dict:
    s = cfg.sla_config
    t = cfg.traffic_config
    e = s.get('embb',  {})
    m = s.get('mmtc',  {})
    u = s.get('urllc', {})
    return {
        'embb_cbr_delay' : e.get('cbr_delay',   50.0),
        'embb_cbr_th'    : e.get('cbr_th',      5e6),
        'embb_vbr_th'    : e.get('vbr_th',      5e6),
        'embb_cbr_queue' : e.get('cbr_queue',  150e3),
        'embb_vbr_queue' : e.get('vbr_queue',  150e3),
        'mmtc_delay'     : m.get('delay',      300.0),
        'mmtc_max_dev'   : t.get('mmtc', {}).get('n_devices', 2500),
        'urllc_cbr_delay': u.get('cbr_delay',   10.0),
        'urllc_cbr_th'   : u.get('cbr_th',    200e3),
        'urllc_vbr_th'   : u.get('vbr_th',    100e3),
        'urllc_cbr_queue': u.get('cbr_queue',   50e3),
        'urllc_vbr_queue': u.get('vbr_queue',  100e3),
    }


# ── Gymnasium environment ─────────────────────────────────────────────────────

class RanSliceSB3Env(gymnasium.Env):
    """
    Gymnasium wrapper around the RAN-slice simulation for SB3 PPO.

    Observation, action, and reward are identical to RanSlicePPOLagEnv so that
    results are directly comparable.  There is no cost signal.
    """

    metadata = {'render_modes': []}

    def __init__(self, cfg, rng, run_id: int, total_steps: int,
                 results_path: str, control_steps: int = 5000):
        super().__init__()

        self._run_id    = run_id
        self._rng       = rng
        self._cfg       = cfg
        self._n_prbs    = cfg.n_prbs
        self._sla       = _extract_sla(cfg)

        raw_env = create_env_from_config(cfg, rng, penalty=PENALTY)
        # gymnasium auto-wraps with OrderEnforcing which hides custom attrs;
        # ReportWrapper (gym.Wrapper) needs the base env directly.
        base_env = raw_env.unwrapped if hasattr(raw_env, 'unwrapped') else raw_env
        self._env = SB3ReportWrapper(
            base_env,
            steps         = total_steps,
            control_steps = control_steps,
            env_id        = str(run_id),
            path          = results_path,
            verbose       = False,
            continuous_mode = True,
        )

        self._n_slices = self._env.n_slices
        # eMBB:4 + mMTC:3 (no queue) + URLLC:4 + global:1 = 12
        self._obs_size = self._n_slices * 4 

        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._n_slices + 1,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._obs_size,), dtype=np.float32,
        )

        self._last_allocation  = np.full(self._n_slices,
                                         self._n_prbs // self._n_slices, dtype=float)
        self._last_change_flag = 0.0
        self._step_count       = 0
        self._max_episode_steps = 2000

    # ── Scenario switching (called between learn() phases) ────────────────────

    def switch_scenario(self, cfg, rng):
        """Swap the underlying simulation to a new scenario. Call before learn()."""
        if hasattr(self._env, 'env') and hasattr(self._env.env, 'close'):
            self._env.env.close()
        raw = create_env_from_config(cfg, rng, penalty=PENALTY)
        self._env.env = raw.unwrapped if hasattr(raw, 'unwrapped') else raw
        self._n_prbs  = cfg.n_prbs
        self._sla     = _extract_sla(cfg)
        self._cfg     = cfg
        print(f"[Run {self._run_id}] === ADVANCING to {cfg.name.upper()} ===")

    def save_results(self):
        self._env.save_results()

    # ── Observation ───────────────────────────────────────────────────────────

    def _compute_observation(self, info: dict) -> np.ndarray:
        """
        12-dim obs, all ∈ [0,1].  Feature layout (running base):
          eMBB  [0-3] : delay_ratio, th_ratio,     queue_ratio, prb_frac
          mMTC  [4-6] : delay_ratio, served_ratio,              prb_frac
          URLLC [7-10]: delay_ratio, th_ratio,     queue_ratio, prb_frac
          global [11] : last_change_flag
        """
        obs         = np.zeros(self._obs_size, dtype=np.float32)
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs', [self._n_prbs // self._n_slices] * self._n_slices)
        sla         = self._sla

        def _r(x, ref):
            return float(x) / max(float(ref), 1e-9)

        # Feature widths per slice: mMTC has 3 (no queue), others have 4
        _widths = [4, 3, 4]
        base = 0

        for si in range(self._n_slices):
            width    = _widths[si]
            prb_frac = (float(n_prbs_list[si]) / max(self._n_prbs, 1)
                        if si < len(n_prbs_list) else 0.0)

            if si < len(l1_info) and isinstance(l1_info[si], dict):
                for _, ran_info in l1_info[si].items():
                    if not isinstance(ran_info, dict):
                        continue

                    if si == _EMBB_IDX:
                        obs[base]   = np.clip(_r(ran_info.get('cbr_delay', 0),
                                                  sla['embb_cbr_delay']), 0, 3) / 3
                        obs[base+1] = np.clip(_r(ran_info.get('cbr_th', 0) + ran_info.get('vbr_th', 0),
                                                  sla['embb_cbr_th'] + sla['embb_vbr_th']), 0, 3) / 3
                        obs[base+2] = np.clip(_r(ran_info.get('cbr_queue', 0) + ran_info.get('vbr_queue', 0),
                                                  sla['embb_cbr_queue'] + sla['embb_vbr_queue']), 0, 3) / 3

                    elif si == _MMTC_IDX:
                        obs[base]   = np.clip(_r(ran_info.get('delay', 0),
                                                  sla['mmtc_delay']), 0, 3) / 3
                        obs[base+1] = 1.0 - np.clip(_r(ran_info.get('devices', 0),
                                                        sla['mmtc_max_dev']), 0, 1)
                        # base+2 is prb_frac — no queue feature for mMTC

                    elif si == _URLLC_IDX:
                        obs[base]   = np.clip(_r(ran_info.get('cbr_delay', 0),
                                                  sla['urllc_cbr_delay']), 0, 3) / 3
                        obs[base+1] = np.clip(_r(ran_info.get('cbr_th', 0) + ran_info.get('vbr_th', 0),
                                                  sla['urllc_cbr_th'] + sla['urllc_vbr_th']), 0, 3) / 3
                        obs[base+2] = np.clip(_r(ran_info.get('cbr_queue', 0) + ran_info.get('vbr_queue', 0),
                                                  sla['urllc_cbr_queue'] + sla['urllc_vbr_queue']), 0, 3) / 3

            obs[base + width - 1] = np.clip(prb_frac, 0.0, 1.0)  # prb_frac always last
            base += width

        obs[base] = self._last_change_flag  # index 11
        return obs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, alloc_prbs: np.ndarray, change_flag: float,
                        info: dict) -> float:
        reward  = 0.0
        l1_info = info.get('l1_info', [])
        sla     = self._sla

        for si in range(self._n_slices):
            if si >= len(l1_info) or not isinstance(l1_info[si], dict):
                continue
            for _, ran_info in l1_info[si].items():
                if not isinstance(ran_info, dict):
                    continue

                if si == _EMBB_IDX:
                    # throughput reward
                    ref   = sla['embb_cbr_th'] + sla['embb_vbr_th']
                    ratio = np.clip((ran_info.get('cbr_th', 0) + ran_info.get('vbr_th', 0))
                                    / max(ref, 1e-9), 0, 3) / 3
                    reward += 0.35 * ratio
                    # delay penalty (weight 0.10, heavier than reallocation constant 0.05)
                    delay_ratio = np.clip(ran_info.get('cbr_delay', 0)
                                          / max(sla['embb_cbr_delay'], 1e-9), 0, 3)
                    reward -= 0.10 * delay_ratio

                elif si == _MMTC_IDX:
                    # served-device reward
                    served = 1.0 - np.clip(ran_info.get('devices', 0)
                                           / max(sla['mmtc_max_dev'], 1), 0, 1)
                    reward += 0.15 * served
                    # delay penalty (weight 0.05)
                    delay_ratio = np.clip(ran_info.get('delay', 0)
                                          / max(sla['mmtc_delay'], 1e-9), 0, 3)
                    reward -= 0.05 * delay_ratio

                elif si == _URLLC_IDX:
                    # throughput reward
                    ref   = sla['urllc_cbr_th'] + sla['urllc_vbr_th']
                    ratio = np.clip((ran_info.get('cbr_th', 0) + ran_info.get('vbr_th', 0))
                                    / max(ref, 1e-9), 0, 3) / 3
                    reward += 0.50 * ratio
                    # delay penalty (weight 0.15, highest priority)
                    delay_ratio = np.clip(ran_info.get('cbr_delay', 0)
                                          / max(sla['urllc_cbr_delay'], 1e-9), 0, 3)
                    reward -= 0.15 * delay_ratio

        # Reallocation: flat constant penalty (smaller than any slice's delay weight)
        if change_flag >= 0.5 and not np.array_equal(alloc_prbs,
                                                      self._last_allocation.astype(int)):
            reward -= 0.05

        return float(np.clip(reward, -1.5, 1.0))

    # ── Tracking helpers ──────────────────────────────────────────────────────

    def _total_throughput(self, info: dict) -> float:
        l1_info = info.get('l1_info', [])
        total   = 0.0
        for si in [_EMBB_IDX, _URLLC_IDX]:
            if si < len(l1_info) and isinstance(l1_info[si], dict):
                for _, ran_info in l1_info[si].items():
                    if isinstance(ran_info, dict):
                        total += ran_info.get('cbr_th', 0) + ran_info.get('vbr_th', 0)
        return total

    def _served_ratio(self, info: dict) -> float:
        l1_info = info.get('l1_info', [])
        if _MMTC_IDX < len(l1_info) and isinstance(l1_info[_MMTC_IDX], dict):
            for _, ran_info in l1_info[_MMTC_IDX].items():
                if isinstance(ran_info, dict) and 'devices' in ran_info:
                    return 1.0 - float(np.clip(
                        ran_info['devices'] / max(self._sla['mmtc_max_dev'], 1), 0, 1))
        return 0.0

    # ── gymnasium.Env interface ───────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        result = self._env.reset()
        info   = result[1] if isinstance(result, tuple) else {}
        obs    = self._compute_observation(info)
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        act         = np.abs(action.astype(float))
        change_flag = float(act[self._n_slices])
        alloc_raw   = act[:self._n_slices]

        total = alloc_raw.sum()
        if total > 0:
            alloc_raw = alloc_raw / total

        new_alloc = np.array([int(np.floor(a * self._n_prbs)) for a in alloc_raw], dtype=int)

        alloc_prbs = self._last_allocation.copy().astype(int) \
                     if change_flag < 0.5 else new_alloc

        result = self._env.step(alloc_prbs)
        if len(result) == 5:
            _, _env_r, terminated, truncated, info = result
        else:
            _, _env_r, done, info = result
            terminated, truncated = bool(done), False

        reward = self._compute_reward(alloc_prbs, change_flag, info)
        self._env.record_extras(self._total_throughput(info), self._served_ratio(info))

        if change_flag >= 0.5:
            self._last_change_flag = 1.0 if not np.array_equal(
                alloc_prbs, self._last_allocation.astype(int)) else 0.0
            self._last_allocation = alloc_prbs.copy().astype(float)
        else:
            self._last_change_flag = 0.0

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated        = True
            self._step_count = 0

        obs = self._compute_observation(info)
        return obs.astype(np.float32), reward, bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        self._env.close()


# ── Trainer ───────────────────────────────────────────────────────────────────

class TrainerSB3PPO:

    def __init__(self):
        os.makedirs(RESULTS_PATH, exist_ok=True)

    def train(self, run_id: int):
        rng = np.random.default_rng(run_id)

        print(f'\n{"="*60}')
        print(f'=== SB3-PPO SEQUENTIAL CURRICULUM  Run {run_id} ===')
        print(f'{"="*60}')
        schedule = ' → '.join(f'{s}({n//1000}k)' for s, n in zip(SCENARIOS, STEPS_PER_SCENARIO))
        print(f'Schedule : {schedule}')
        print(f'Total    : {TOTAL_STEPS} steps')
        print(f'Output   : {RESULTS_PATH}history_{run_id}.npz')

        cfg0 = _get_scenario_config(SCENARIOS[0])
        env  = RanSliceSB3Env(
            cfg0, rng,
            run_id       = run_id,
            total_steps  = TOTAL_STEPS,
            results_path = RESULTS_PATH,
            control_steps= 5000,
        )

        model = PPO(
            'MlpPolicy',
            env,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.0,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            verbose         = 0,
            policy_kwargs   = dict(net_arch=[256, 256]),
        )

        # Sequential curriculum: train on each scenario in turn
        for i, (scenario_name, n_steps) in enumerate(zip(SCENARIOS, STEPS_PER_SCENARIO)):
            if i > 0:
                cfg = _get_scenario_config(scenario_name)
                env.switch_scenario(cfg, rng)
                # Force SB3 to reset its internal state for the new scenario
                model.set_env(env)

            print(f'[Run {run_id}] Training on {scenario_name} for {n_steps} steps...')
            model.learn(
                total_timesteps    = n_steps,
                reset_num_timesteps= (i == 0),  # only reset counters on first phase
            )

        env.save_results()
        print(f'[Run {run_id}] Done. Results saved.')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SB3-PPO with sequential curriculum on RAN slicing'
    )
    parser.add_argument('--runs',        type=int,  default=RUNS)
    parser.add_argument('--processes',   type=int,  default=PROCESSES)
    parser.add_argument('--sequential',  action='store_true')
    parser.add_argument('--total-steps', type=int,  default=TOTAL_STEPS)
    args = parser.parse_args()

    if args.total_steps != TOTAL_STEPS:
        # Rescale each phase proportionally
        ratio = args.total_steps / TOTAL_STEPS
        STEPS_PER_SCENARIO[:] = [int(s * ratio) for s in STEPS_PER_SCENARIO]
        TOTAL_STEPS = args.total_steps

    trainer  = TrainerSB3PPO()
    run_list = list(range(args.runs))

    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as E:
            list(E.map(trainer.train, run_list))
