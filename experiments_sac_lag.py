#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC-Lag (Safe RL with Lagrangian) for RAN network slicing.

New gym env + CMDP wrapper — does NOT modify any existing environment.

Architecture:
    NodeB → RanSliceSACLag (new gym.Env) → ReportWrapper → RanSliceSACLagCMDP (OmniSafe CMDP)

Training regime (this revision):
  - Single scenario: 'medium' (no curriculum).
  - 360 000 total steps, 2 000 steps per episode, 10 epochs
    (steps_per_epoch = 36 000).

Hard feasibility constraints (enforced at action-decoding time — the agent
CANNOT violate them regardless of what it outputs):
  1. If any slice is currently violating its SLA, reconfiguration is FORCED
     (the agent's reconfig signal is overridden).
  2. A violating slice's PRB count can only be reduced if:
       - a higher-priority slice is also violating, AND
       - there are no excess PRBs available to take from.
     Otherwise its allocation is pinned to (at least) its previous count.
  3. If any slice is violating, no PRBs may be left idle — excess is
     redistributed to the highest-priority violating slices.

Priority: URLLC (3) > eMBB (2) > mMTC (1).

RanSliceSACLag defines:
  - Reward  : weighted (throughput + served-devices) ratio per slice, normalised to [0,1],
              minus a constant if the agent reconfigured,
              minus a larger constant if the agent did NOT reconfigure but violations occurred.
  - Cost    : per-slice ratio of (th, delay) vs SLA thresholds, weighted by slice importance,
              stored in info['sla_cost']. Used by the Lagrangian multiplier.

Cost limit: 0.05.
"""

import os
import argparse
import numpy as np
import concurrent.futures as cf
from numpy.random import default_rng

import gymnasium
from gymnasium import spaces
import torch

from wrapper import ReportWrapper
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe
from config_loader import load_scenario
from scenario_creator import create_env_from_config


# ── Training constants ────────────────────────────────────────────────────────

RUNS               = 30
PROCESSES          = 4

SCENARIO           = 'medium'          # train directly on medium, no curriculum
TOTAL_STEPS        = 360000              # 360k steps = 10 epochs of 36k steps each
STEPS_PER_EPISODE  = 1000                # fixed episode length (no curriculum)
EPOCHS             = 50
STEPS_PER_EPOCH    = TOTAL_STEPS // EPOCHS   # 36 000
assert TOTAL_STEPS % EPOCHS == 0, "TOTAL_STEPS must be divisible by EPOCHS"

ENV_ID       = "RanSliceSACLag-v0"
_PENALTY     = 100.0

# ── Safe-RL cost budget (Lagrangian multiplier enforces cost ≤ COST_LIMIT) ───

COST_LIMIT = 0.25

# ── Slice importance / priority weights ──────────────────────────────────────
# Higher weight ⇒ higher priority in the feasibility projection.
_SLICE_WEIGHTS = {'URLLC': 3.0, 'eMBB': 2.0, 'mMTC': 1.0}

# ── Reconfiguration / inaction penalty constants ──────────────────────────────

_RECONFIG_THRESHOLD = 0.05   # fractional PRB shift to count as a reconfiguration
_RECONFIG_PENALTY   = 0.10   # penalty for changing the allocation
_INACTION_PENALTY   = 0.40   # penalty for NOT changing when violations occur
                             # (must be > _RECONFIG_PENALTY)

# ── Module-level state for multi-run parallelism (mirrors TD3 pattern) ────────

_scenario_configs_cache: dict = {}
_CURRENT_RUN_ID    : int  = 0
_ENV_INSTANCE_COUNT: dict = {}


def _load_scenario(name: str):
    if name not in _scenario_configs_cache:
        _scenario_configs_cache[name] = load_scenario('scenarios.yaml', name)
    return _scenario_configs_cache[name]


# ── SLA metadata helpers ──────────────────────────────────────────────────────

def _build_sla_info(node_b) -> list:
    """
    Extract per-slice SLA thresholds and type from NodeB without touching it.
    Returns a list of dicts, one per l1 slice, in the same order as node_b.slices_l1.
    """
    slots = node_b.slots_per_step
    obs_t = slots * node_b.slot_length
    result = []
    for l1 in node_b.slices_l1:
        ran   = l1.slices_ran[0]
        stype = l1.type   # 'eMBB' | 'URLLC' | 'mMTC'
        entry = {
            'type':             stype,
            'sla':              ran.SLA,
            'slots_per_step':   slots,
            'observation_time': obs_t,
            'weight':           _SLICE_WEIGHTS.get(stype, 1.0),
        }
        result.append(entry)
    return result


def _get_ran_info(l1_info, slice_idx: int) -> dict:
    """Extract the first RAN slice info dict for a given l1 slice index."""
    if slice_idx >= len(l1_info):
        return {}
    entry = l1_info[slice_idx]
    return entry.get(0, {}) if isinstance(entry, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────
# New gymnasium env — reward and cost live here, ReportWrapper tracks them
# ─────────────────────────────────────────────────────────────────────────────

class RanSliceSACLag(gymnasium.Env):
    """
    Second gym env for SAC-Lag training.  NodeB is unchanged.

    Action : integer PRBs per slice — shape (n_slices,).
             The CMDP converts the continuous agent action before calling this env.
    Obs    : raw simulator state (same as RanSlice), used by ReportWrapper for logging.
             The CMDP builds its own feature vector for the agent from info.

    Reward : weighted (throughput_ratio + served_ratio)/2 across slices,
             normalised to [0,1], then:
               - RECONFIG_PENALTY  if allocation changed significantly vs previous step
               - INACTION_PENALTY  if allocation unchanged AND violations occurred

    Cost   : stored in info['sla_cost'] ∈ [0,1].
    """

    metadata = {'render_modes': []}

    def __init__(self, node_b):
        super().__init__()
        self.node_b    = node_b
        self.n_prbs    = node_b.n_prbs
        self.n_slices  = node_b.n_slices_l1
        self.n_variables = node_b.get_n_variables()

        self._sla_info   = _build_sla_info(node_b)
        self._last_alloc = np.zeros(self.n_slices, dtype=float)

        self.action_space = spaces.Box(
            low=0, high=self.n_prbs, shape=(self.n_slices,), dtype=int
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_variables,), dtype=float
        )

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._last_alloc = np.zeros(self.n_slices, dtype=float)
        state = self.node_b.reset()
        return state, {}

    def step(self, action):
        action = np.asarray(action, dtype=int)
        state, info = self.node_b.step(action)

        total_violations = info['violations'].sum()
        info['total_violations'] = int(total_violations)

        reward          = self._compute_reward(action, info)
        info['sla_cost'] = self._compute_cost(info)

        self._last_alloc = action.astype(float)

        return state, float(reward), False, False, info

    def render(self):
        pass

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray, info: dict) -> float:
        l1_info    = info.get('l1_info', [])
        violations = info.get('violations', np.zeros(self.n_slices, dtype=int))

        total_w = sum(m['weight'] for m in self._sla_info)
        perf    = 0.0

        for i, meta in enumerate(self._sla_info):
            stype = meta['type']
            sla   = meta['sla']
            slots = meta['slots_per_step']
            w     = meta['weight']
            ran   = _get_ran_info(l1_info, i)

            th_ratio = served_ratio = 0.0

            if stype in ('eMBB', 'URLLC'):
                demand   = max(ran.get('cbr_traffic', 0.0) + ran.get('vbr_traffic', 0.0), 1.0)
                th_ratio = float(np.clip(
                    (ran.get('cbr_th', 0.0) + ran.get('vbr_th', 0.0)) / demand, 0.0, 1.0
                ))
                served_ratio = th_ratio

            elif stype == 'mMTC':
                sla_d  = max(sla.get('delay', 1.0), 1e-9)
                d_rat  = (ran.get('delay', 0.0) / max(slots, 1)) / sla_d
                served_ratio = th_ratio = float(np.clip(2.0 - d_rat, 0.0, 1.0))

            perf += w * (th_ratio + served_ratio) / 2.0

        reward = perf / total_w

        delta        = np.sum(np.abs(action.astype(float) - self._last_alloc))
        reconfigured = (delta / max(self.n_prbs, 1)) > _RECONFIG_THRESHOLD
        any_viol     = bool(np.any(np.asarray(violations) > 0))

        if reconfigured:
            reward -= _RECONFIG_PENALTY
        elif any_viol:
            reward -= _INACTION_PENALTY

        return float(reward)

    # ── cost ─────────────────────────────────────────────────────────────────

    def _compute_cost(self, info: dict) -> float:
        l1_info = info.get('l1_info', [])
        total_w = sum(m['weight'] for m in self._sla_info)
        cost    = 0.0

        for i, meta in enumerate(self._sla_info):
            stype = meta['type']
            sla   = meta['sla']
            slots = meta['slots_per_step']
            w     = meta['weight']
            ran   = _get_ran_info(l1_info, i)

            th_def = delay_ex = 0.0

            if stype in ('eMBB', 'URLLC'):
                demand  = max(ran.get('cbr_traffic', 0.0) + ran.get('vbr_traffic', 0.0), 1.0)
                th_rat  = float(np.clip(
                    (ran.get('cbr_th', 0.0) + ran.get('vbr_th', 0.0)) / demand, 0.0, 1.0
                ))
                th_def  = max(0.0, 1.0 - th_rat)

                sla_d   = max(sla.get('cbr_delay', 1.0), 1e-9)
                d_val   = (ran.get('cbr_delay', 0.0) / max(slots, 1)
                           if stype == 'eMBB'
                           else ran.get('cbr_delay', 0.0))
                delay_ex = float(np.clip(max(0.0, d_val / sla_d - 1.0), 0.0, 1.0))

            elif stype == 'mMTC':
                sla_d    = max(sla.get('delay', 1.0), 1e-9)
                d_rat    = (ran.get('delay', 0.0) / max(slots, 1)) / sla_d
                delay_ex = float(np.clip(max(0.0, d_rat - 1.0), 0.0, 1.0))
                th_def   = delay_ex

            cost += w * (th_def + delay_ex) / 2.0

        return float(cost / total_w)


# ─────────────────────────────────────────────────────────────────────────────
# Factory — builds NodeB + RanSliceSACLag (does not use gym.make)
# ─────────────────────────────────────────────────────────────────────────────

def _build_raw_env(cfg, rng) -> RanSliceSACLag:
    inner  = create_env_from_config(cfg, rng, penalty=_PENALTY)
    node_b = inner.unwrapped.node_b
    return RanSliceSACLag(node_b=node_b)


# ─────────────────────────────────────────────────────────────────────────────
# OmniSafe CMDP — thin wrapper that converts actions and forwards reward/cost
# ─────────────────────────────────────────────────────────────────────────────

@env_register
@env_unregister
class RanSliceSACLagCMDP(CMDP):
    """
    Single-scenario SAC-Lag CMDP (medium scenario, 360k steps).

    - Action  : Box [0,1]^{n_slices+2}.
                [0 : n_slices]   softmax with [n_slices] (excess) → PRB fractions + idle
                [n_slices + 1]   reconfiguration signal ∈ [0,1]; > 0.5 = reconfigure
                The raw action is then projected onto the FEASIBLE set
                (hard rules 1-3 above).
    - Reward  : directly from RanSliceSACLag (tracked by ReportWrapper).
    - Cost    : info['sla_cost'] from RanSliceSACLag.
    - Obs     : [th_ratio, delay_obs, alloc_ratio] per slice (built from info).
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
        self._instance_id     = _ENV_INSTANCE_COUNT[self._run_id]
        self._is_training_env = (self._instance_id == 1)

        print(f"[SACLag run={self._run_id}] instance {self._instance_id} "
              f"({'train-records' if self._is_training_env else 'eval-silent'})")

        # Single scenario — no curriculum.
        self._scenario_name = SCENARIO
        self._scenario_cfg  = _load_scenario(SCENARIO)
        self._global_step   = 0

        self._results_path = './results/SACLag/'
        os.makedirs(self._results_path, exist_ok=True)

        cfg     = self._scenario_cfg
        raw_env = _build_raw_env(cfg, np.random.default_rng(self._run_id))

        # The inner ReportWrapper now records across the full training run.
        self._env = ReportWrapper(
            raw_env,
            steps           = TOTAL_STEPS,
            control_steps   = STEPS_PER_EPISODE if self._is_training_env else TOTAL_STEPS + 1,
            env_id          = self._run_id if self._is_training_env else f'{self._run_id}-eval',
            path            = self._results_path,
            verbose         = False,
            continuous_mode = True,
        )

        self._max_episode_steps = STEPS_PER_EPISODE   # 2 000
        self._step_count        = 0
        self._n_prbs            = cfg.n_prbs
        self._n_slices          = self._env.n_slices

        # Last allocation actually applied (used when not reconfiguring + as
        # the baseline for feasibility checks).
        self._last_applied_prbs = np.zeros(self._n_slices, dtype=int)

        # Cached violation / priority info from the most recent env step. We
        # need this at *decision* time (before calling step) so that the action
        # projection knows who is currently violating.
        self._last_violations = np.zeros(self._n_slices, dtype=int)

        # Episode-level accumulators — mirrors what OmniSafe logs as EpRet / EpCost.
        self._episode_reward = 0.0
        self._episode_cost   = 0.0

        # Priority vector (higher = more important) in slice order.
        self._priorities = np.array(
            [_SLICE_WEIGHTS.get(m['type'], 1.0) for m in self._env.env._sla_info],
            dtype=float,
        )

        obs_dim = self._n_slices * 3
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._n_slices + 2,), dtype=np.float32
        )

    # ── raw env SLA info (dynamic — auto-refreshes when scenario changes) ─────

    @property
    def _sla_info(self) -> list:
        return self._env.env._sla_info

    # ── feasibility projection ────────────────────────────────────────────────

    def _project_feasible(
        self,
        proposed_prbs: np.ndarray,
        proposed_excess_prbs: int,
        reconfig_requested: bool,
    ) -> tuple[np.ndarray, bool]:
        """
        Enforce the three hard rules against the *last observed* violations.

        Inputs
        ------
        proposed_prbs        : int[n_slices] — PRBs the agent wants per slice.
        proposed_excess_prbs : int           — PRBs the agent wants to leave idle.
        reconfig_requested   : bool          — the agent's reconfig signal.

        Returns
        -------
        final_prbs     : int[n_slices], sums to ≤ n_prbs, with no excess if any
                         slice was violating on the previous step.
        reconfigured   : bool — True if the action produced actually differs
                         from the last applied allocation.

        Rules
        -----
        R1. If any slice was violating on the previous step → reconfiguration
            is FORCED (reconfigured = True regardless of reconfig_requested).
        R2. A violating slice cannot receive fewer PRBs than it had last step,
            UNLESS a strictly-higher-priority slice is also violating AND no
            excess PRBs are available (R3 already makes excess=0 whenever any
            slice violates, so this second clause only fires when no donors
            exist among non-violators).
        R3. If any slice is violating, excess PRBs must be 0 — all idle PRBs
            are redistributed to violating slices in priority order.
        """
        n         = self._n_slices
        total     = self._n_prbs
        violating = self._last_violations > 0
        any_viol  = bool(violating.any())

        last_prbs = self._last_applied_prbs.astype(int).copy()
        prbs      = proposed_prbs.astype(int).copy()
        excess    = int(proposed_excess_prbs)

        # If nothing is violating, fall back to the agent's decision entirely.
        if not any_viol:
            reconfigured = reconfig_requested
            if not reconfigured:
                return last_prbs, False
            # Make sure everything stays in-budget and non-negative.
            prbs = np.clip(prbs, 0, total)
            over = prbs.sum() + max(excess, 0) - total
            if over > 0:
                # Trim the over-allocation from the largest non-violating slice.
                order = np.argsort(-prbs)
                i = 0
                while over > 0 and i < n:
                    take = min(over, prbs[order[i]])
                    prbs[order[i]] -= take
                    over -= take
                    i += 1
            return prbs.astype(int), not np.array_equal(prbs, last_prbs)

        # --- From here on: at least one slice is violating. ---
        # R1: force reconfiguration.
        # R3: excess is forbidden — we'll distribute it before returning.
        # R2: violating slice i cannot drop below last_prbs[i] unless a strictly
        #     higher-priority slice is also violating AND after R3 redistribution
        #     no excess PRBs remain to satisfy it.

        # Step 1 — pin violating slices to at least their last allocation,
        # relaxing only when a higher-priority slice is also violating (those
        # slices can donate from peers/lowers).
        prbs = np.clip(prbs, 0, total)

        # Identify the max priority among violating slices.
        viol_prios  = self._priorities[violating]
        max_viol_pr = viol_prios.max()

        for i in range(n):
            if not violating[i]:
                continue
            higher_violators_exist = np.any(
                violating & (self._priorities > self._priorities[i])
            )
            # Rule: pin to last_prbs[i] UNLESS a strictly-higher-priority
            # slice is also violating.  (If one is, the pin is relaxed so the
            # scheduler can route more PRBs to the top-priority victim.)
            if not higher_violators_exist:
                prbs[i] = max(prbs[i], last_prbs[i])

        # Step 2 — budget enforcement (must fit in total PRBs).
        # If the pinned allocation overshoots the budget, trim from the lowest
        # priority non-violating slices first, then from lowest-priority
        # violating slices (never trim from the highest priority violators).
        over = prbs.sum() - total
        if over > 0:
            # Donors, in the order we'll trim: non-violators ascending priority,
            # then violators ascending priority (excluding the top priority
            # violating tier).
            donor_order = sorted(
                range(n),
                key=lambda i: (
                    0 if not violating[i] else 1,                        # non-violators first
                    self._priorities[i],                                  # lowest priority first
                ),
            )
            for i in donor_order:
                if over <= 0:
                    break
                # Never trim a top-priority violator.
                if violating[i] and self._priorities[i] >= max_viol_pr:
                    continue
                # Non-violators can go to zero; violators respect their pin
                # (last_prbs[i]) unless a higher violator exists.
                higher_violators_exist = np.any(
                    violating & (self._priorities > self._priorities[i])
                )
                floor = 0 if (not violating[i] or higher_violators_exist) else last_prbs[i]
                take = min(over, max(prbs[i] - floor, 0))
                prbs[i] -= take
                over    -= take

            if over > 0:
                # Last-resort: trim top-priority violators proportionally.
                # This only triggers in pathological cases (total budget
                # genuinely too small).
                top_idx = np.where(violating & (self._priorities == max_viol_pr))[0]
                while over > 0 and top_idx.size > 0:
                    for i in top_idx:
                        if over <= 0:
                            break
                        if prbs[i] > 0:
                            prbs[i] -= 1
                            over    -= 1
                    if prbs[top_idx].sum() == 0:
                        break

        # Step 3 — R3: redistribute any leftover budget (= excess) to violating
        # slices in priority order until nothing is left idle.
        leftover = total - prbs.sum()
        if leftover > 0:
            # Order violating slices by descending priority; ties broken
            # by larger unmet demand (= larger gap to last_prbs).
            viol_idx = np.where(violating)[0]
            viol_idx = sorted(
                viol_idx,
                key=lambda i: (-self._priorities[i],
                               -(last_prbs[i] - prbs[i])),
            )
            # Round-robin within the highest-priority tier first.
            k = 0
            while leftover > 0 and viol_idx:
                i = viol_idx[k % len(viol_idx)]
                prbs[i] += 1
                leftover -= 1
                k += 1

        # Final safety clip.
        prbs = np.clip(prbs, 0, total).astype(int)
        if prbs.sum() > total:
            # Should not happen, but guard anyway.
            diff = prbs.sum() - total
            prbs[np.argmax(prbs)] -= diff

        # R1: any-violation → reconfigured = True (regardless of signal).
        reconfigured = True
        return prbs, reconfigured

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._last_applied_prbs = np.zeros(self._n_slices, dtype=int)
        self._last_violations   = np.zeros(self._n_slices, dtype=int)
        result = self._env.reset()
        _, info = result if isinstance(result, tuple) else (result, {})
        obs = self._build_obs(info)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action: torch.Tensor):
        # ── action decoding ────────────────────────────────────────────────
        act = (action.cpu().numpy() + 1.0) / 2.0

        # Reconfiguration signal: last element, threshold at 0.5
        reconfig_signal    = act[self._n_slices + 1]
        reconfig_requested = bool(reconfig_signal > 0.5)

        # Softmax over [slice_0..slice_n, excess] → fractions summing to 1
        alloc_excess = act[:self._n_slices + 1]
        total        = alloc_excess.sum()
        alloc_excess = (alloc_excess / total if total > 0
                        else np.ones(self._n_slices + 1) / (self._n_slices + 1))

        alloc_frac = alloc_excess[:self._n_slices]
        excess_frac = alloc_excess[self._n_slices]

        proposed_prbs = np.array(
            [int(np.floor(f * self._n_prbs)) for f in alloc_frac], dtype=int
        )
        proposed_excess = int(np.floor(excess_frac * self._n_prbs))

        # ── HARD feasibility projection ───────────────────────────────────
        applied_prbs, reconfigured = self._project_feasible(
            proposed_prbs          = proposed_prbs,
            proposed_excess_prbs   = proposed_excess,
            reconfig_requested     = reconfig_requested,
        )

        if not reconfigured:
            # Truly replay the previous allocation (bit-exact).
            applied_prbs = self._last_applied_prbs.copy()
        else:
            self._last_applied_prbs = applied_prbs.copy()

        # ── env step ───────────────────────────────────────────────────────
        result = self._env.step(applied_prbs)
        if len(result) == 4:
            _, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            _, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)

        cost = float(info.get('sla_cost', 0.0))

        # Accumulate episode return and cost (mirrors OmniSafe EpRet / EpCost).
        self._episode_reward += float(reward)
        self._episode_cost   += cost

        # Cache violations for the NEXT step's feasibility projection.
        self._last_violations = np.asarray(
            info.get('violations', np.zeros(self._n_slices, dtype=int)),
            dtype=int,
        )

        self._step_count  += 1
        self._global_step += 1

        # Episode truncation (fixed 2 000-step episodes, no curriculum).
        if self._step_count >= self._max_episode_steps:
            truncated        = True
            self._step_count = 0

        info = {str(k): v for k, v in info.items()} if isinstance(info, dict) else {}
        obs  = self._build_obs(info)

        if terminated or truncated:
            # Record episode-level return and cost before resetting.
            if self._is_training_env:
                self._env.record_episode_end(self._episode_reward, self._episode_cost)
            self._episode_reward = 0.0
            self._episode_cost   = 0.0

            final_obs = torch.as_tensor(obs, dtype=torch.float32)
            _, new_info = self._env.reset()
            self._last_applied_prbs = np.zeros(self._n_slices, dtype=int)
            self._last_violations   = np.zeros(self._n_slices, dtype=int)
            obs  = torch.as_tensor(self._build_obs(new_info), dtype=torch.float32)
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

    # ── observation ──────────────────────────────────────────────────────────

    def _build_obs(self, info) -> np.ndarray:
        n           = self._n_slices
        obs         = np.zeros(n * 3, dtype=np.float32)
        l1_info     = info.get('l1_info', [])
        n_prbs_list = info.get('n_prbs',  [])

        for i, meta in enumerate(self._sla_info):
            stype = meta['type']
            sla   = meta['sla']
            slots = meta['slots_per_step']
            ran   = _get_ran_info(l1_info, i)

            th_ratio    = 0.0
            delay_ratio = 0.0

            if stype in ('eMBB', 'URLLC'):
                demand   = max(ran.get('cbr_traffic', 0.0) + ran.get('vbr_traffic', 0.0), 1.0)
                th_ratio = float(np.clip(
                    (ran.get('cbr_th', 0.0) + ran.get('vbr_th', 0.0)) / demand, 0.0, 1.0
                ))
                sla_d       = max(sla.get('cbr_delay', 1.0), 1e-9)
                cbr_delay   = ran.get('cbr_delay', 0.0)
                d_val       = (cbr_delay / max(slots, 1) if stype == 'eMBB' else cbr_delay)
                delay_ratio = float(np.clip(d_val / sla_d, 0.0, 2.0))

            elif stype == 'mMTC':
                sla_d       = max(sla.get('delay', 1.0), 1e-9)
                delay_ratio = float(np.clip(
                    (ran.get('delay', 0.0) / max(slots, 1)) / sla_d, 0.0, 2.0
                ))
                th_ratio    = float(np.clip(2.0 - delay_ratio, 0.0, 1.0))

            alloc_ratio = float(
                n_prbs_list[i] / max(self._n_prbs, 1) if i < len(n_prbs_list) else 0.0
            )

            obs[i]       = float(np.clip(th_ratio,          0.0, 1.0))
            obs[i + n]   = float(np.clip(delay_ratio - 1.0, -1.0, 1.0))
            obs[i + 2*n] = float(np.clip(alloc_ratio,       0.0, 1.0))

        return obs

    # ── boilerplate ───────────────────────────────────────────────────────────

    def render(self, *args, **kwargs):
        if hasattr(self._env, 'render'):
            return self._env.render()

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

class TrainerSACLag:
    """Train SAC-Lag directly on the medium scenario."""

    def __init__(self):
        os.makedirs('./results/scenario_comparison/SACLag/', exist_ok=True)

    def train(self, run_id: int):
        global _CURRENT_RUN_ID, _ENV_INSTANCE_COUNT
        _CURRENT_RUN_ID             = run_id
        _ENV_INSTANCE_COUNT[run_id] = 0

        print(f'\n{"="*60}')
        print(f'=== SAC-Lag  run={run_id} ===')
        print(f'{"="*60}')
        print(f'Scenario         : {SCENARIO}  (no curriculum)')
        print(f'Total steps      : {TOTAL_STEPS}')
        print(f'Steps / episode  : {STEPS_PER_EPISODE}')
        print(f'Epochs           : {EPOCHS}   (steps/epoch = {STEPS_PER_EPOCH})')
        print(f'Cost limit       : {COST_LIMIT}')

        custom_cfgs = {
            "train_cfgs": {
                "total_steps": TOTAL_STEPS,
                "device":      "cpu",
            },
            "algo_cfgs": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "update_iters":    1,
            },
            "lagrange_cfgs": {
                "cost_limit": COST_LIMIT,
            },
            "logger_cfgs": {
                "use_wandb":       False,
                "save_model_freq": 1,
            },
        }

        agent = omnisafe.Agent(
            algo        = "SACLag",
            env_id      = ENV_ID,
            custom_cfgs = custom_cfgs,
        )

        print('Training started …')
        agent.learn()
        print('Training done.')

        try:
            agent.plot(smooth=1)
        except Exception as exc:
            print(f'OmniSafe plotting failed (expected in parallel mode): {exc}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train SAC-Lag directly on the 'medium' scenario for 360k steps."
    )
    parser.add_argument("--runs",      type=int, default=RUNS)
    parser.add_argument("--processes", type=int, default=PROCESSES)
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of in parallel")
    args = parser.parse_args()

    trainer  = TrainerSACLag()
    run_list = list(range(args.runs))

    if args.sequential:
        for run in run_list:
            trainer.train(run)
    else:
        with cf.ProcessPoolExecutor(args.processes) as pool:
            list(pool.map(trainer.train, run_list))