#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGNBWrapper
---------------
A Gymnasium-compatible wrapper that manages multiple NodeB instances,
UE mobility, and A3-event-based handover (hysteresis + Time-to-Trigger).

Designed to work directly with NodeB (node_b.py) — no gym registry needed.

Usage
-----
    from multi_gnb_wrapper import MultiGNBWrapper

    gnb_list = [
        NodeB(id=0, x=200, y=200, slices_l1=[...], slots_per_step=50, n_prbs=100, coverage_radius=500),
        NodeB(id=1, x=600, y=200, slices_l1=[...], slots_per_step=50, n_prbs=100, coverage_radius=500),
    ]

    env = MultiGNBWrapper(gnb_list, slots_per_step=50)
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Helper: simple signal-strength model (matches NodeB.get_ue_signal_strength)
# ---------------------------------------------------------------------------

def _signal_strength(distance: float, coverage_radius: float) -> float:
    """Normalised signal strength in [0, 1]; mirrors NodeB path-loss model."""
    if distance > coverage_radius:
        return 0.0
    nd = distance / coverage_radius
    return 1.0 / (1.0 + nd ** 2)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class MultiGNBWrapper(gym.Env):
    """
    Wraps a list of NodeB objects into a single Gymnasium environment.

    Observation space
    -----------------
    Flat concatenation of each NodeB's get_state() output.

    Action space
    ------------
    Flat float array [0, 1]^(n_gnbs * n_slices_per_gnb).
    Each element is the *fraction* of total PRBs assigned to that slice.
    The wrapper renormalises per-gNB so allocations always sum to n_prbs.

    Reward
    ------
    Mean of per-gNB rewards (each NodeB.compute_reward returns +1/-1).

    Handover model
    --------------
    3GPP A3 event: handover fires when
        RSRP(candidate) - RSRP(serving) > hysteresis
    for `handover_ttt` consecutive steps.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        gnb_list: List,                   # list of NodeB instances
        slots_per_step: int = 50,
        handover_hysteresis: float = 0.05,  # signal-strength units (linear)
        handover_ttt: int = 3,              # steps before handover fires
        verbose: bool = False,
    ):
        super().__init__()

        self.gnbs: List = gnb_list
        self.n_gnbs: int = len(gnb_list)
        self.slots_per_step: int = slots_per_step
        self.handover_hysteresis: float = handover_hysteresis
        self.handover_ttt: int = handover_ttt
        self.verbose: bool = verbose

        # dt per step in seconds (slot_length = 1 ms)
        self.dt: float = slots_per_step * 1e-3

        # Derive per-gNB slice counts from the actual NodeB objects
        self._n_slices: List[int] = [gnb.n_slices_l1 for gnb in self.gnbs]
        self._total_action_dim: int = sum(self._n_slices)

        # Observation dimension: sum of each NodeB.get_state() lengths
        obs_dims = [len(gnb.get_state()) for gnb in self.gnbs]
        self._obs_dim: int = sum(obs_dims)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._total_action_dim,),
            dtype=np.float32,
        )

        # UE registry  ue_id -> dict
        self._ues: Dict[int, Dict] = {}
        self._next_ue_id: int = 0

        # Handover event log
        self.handover_log: List[Dict] = []
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.handover_log.clear()
        self._ues.clear()
        self._next_ue_id = 0

        for gnb in self.gnbs:
            gnb.reset()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : np.ndarray, shape (total_action_dim,)
            Flat array of per-slice PRB fractions for every gNB.
            Order: [gnb0_slice0, gnb0_slice1, ..., gnb1_slice0, ...]
        """
        # 1. Move UEs and handle handovers
        self._update_ue_positions()
        self._check_handovers()

        # 2. Split and scale actions per gNB
        gnb_actions = self._split_action(action)

        # 3. Step each gNB
        rewards = []
        infos = {}
        terminated = False
        truncated = False

        for i, (gnb, gnb_action) in enumerate(zip(self.gnbs, gnb_actions)):
            _, info = gnb.step(gnb_action)
            reward_val, _ = gnb.compute_reward()
            # compute_reward returns arrays; take mean across slices
            rewards.append(float(np.mean(reward_val)))
            infos[f"gnb_{i}"] = info

        global_reward = float(np.mean(rewards))

        # 4. Build global observation
        obs = self._get_obs()

        # 5. Meta-info
        infos["per_gnb_rewards"] = rewards
        infos["handover_count"] = len(self.handover_log)
        infos["ue_per_gnb"] = self._ue_count_per_gnb()

        self.step_count += 1
        return obs, global_reward, terminated, truncated, infos

    def render(self, mode="human"):
        print(f"\n{'='*50}")
        print(f"Step {self.step_count}  |  gNBs: {self.n_gnbs}  |  UEs: {len(self._ues)}")
        for i, gnb in enumerate(self.gnbs):
            n_ues = sum(1 for u in self._ues.values() if u["serving_gnb"] == i)
            print(f"  gNB {gnb.id} ({gnb.x:.0f},{gnb.y:.0f})  UEs={n_ues}  PRBs={gnb.n_prbs}")
        print(f"  Handovers total: {len(self.handover_log)}")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # UE management (public API)
    # ------------------------------------------------------------------

    def add_ue(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> int:
        """
        Register a mobile UE.  Returns the assigned ue_id.

        The wrapper tracks the UE's position and triggers handovers
        automatically during step().  The UE is NOT injected into the
        NodeB's internal slice lists here — it is expected that each
        NodeB's SliceRAN generates its own UE arrivals/departures.
        This tracker exists purely for handover decision-making.
        """
        ue_id = self._next_ue_id
        self._next_ue_id += 1

        serving = self._best_gnb(x, y)
        self._ues[ue_id] = {
            "x": x, "y": y,
            "vx": vx, "vy": vy,
            "serving_gnb": serving,
            "ho_pending": False,
            "ho_candidate": None,
            "ho_counter": 0,
        }
        if self.verbose:
            print(f"[UE added] id={ue_id} pos=({x:.1f},{y:.1f}) serving_gnb={serving}")
        return ue_id

    def remove_ue(self, ue_id: int):
        """Deregister a UE."""
        self._ues.pop(ue_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        parts = [gnb.get_state().astype(np.float32) for gnb in self.gnbs]
        return np.concatenate(parts)

    def _split_action(self, action: np.ndarray) -> List[np.ndarray]:
        """
        Slice the flat action vector and convert fractions → integer PRBs.
        Each gNB's slice allocations are renormalised to sum to gnb.n_prbs.
        """
        gnb_actions = []
        idx = 0
        for gnb, n_slices in zip(self.gnbs, self._n_slices):
            raw = action[idx: idx + n_slices]
            idx += n_slices

            # Avoid zero-sum edge case
            total = raw.sum()
            if total < 1e-8:
                fractions = np.ones(n_slices) / n_slices
            else:
                fractions = raw / total

            prbs = np.floor(fractions * gnb.n_prbs).astype(int)

            # Distribute any leftover PRBs to the slice with the highest fraction
            leftover = gnb.n_prbs - prbs.sum()
            if leftover > 0:
                order = np.argsort(fractions)[::-1]
                for k in range(leftover):
                    prbs[order[k % n_slices]] += 1

            gnb_actions.append(prbs)
        return gnb_actions

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------

    def _update_ue_positions(self):
        for ue in self._ues.values():
            ue["x"] += ue["vx"] * self.dt
            ue["y"] += ue["vy"] * self.dt

    def _best_gnb(self, x: float, y: float) -> Optional[int]:
        """Return index of gNB with strongest signal, or None if out of coverage."""
        best_idx = None
        best_sig = 0.0
        for i, gnb in enumerate(self.gnbs):
            dist = np.hypot(x - gnb.x, y - gnb.y)
            sig = _signal_strength(dist, gnb.coverage_radius)
            if sig > best_sig:
                best_sig = sig
                best_idx = i
        return best_idx

    def _check_handovers(self):
        """A3-event handover check for every registered UE."""
        for ue_id, ue in self._ues.items():
            x, y = ue["x"], ue["y"]
            current = ue["serving_gnb"]
            best = self._best_gnb(x, y)

            # Completely out of coverage
            if best is None:
                if current is not None and self.verbose:
                    print(f"[UE {ue_id}] left all coverage areas")
                ue["serving_gnb"] = None
                ue["ho_pending"] = False
                ue["ho_counter"] = 0
                continue

            # Re-entering coverage
            if current is None:
                ue["serving_gnb"] = best
                ue["ho_pending"] = False
                ue["ho_counter"] = 0
                if self.verbose:
                    print(f"[UE {ue_id}] re-entered coverage, attached to gNB {best}")
                continue

            # Same cell — reset any pending HO
            if best == current:
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0
                continue

            # Compute signal delta (A3 condition)
            gnb_curr = self.gnbs[current]
            gnb_best = self.gnbs[best]
            sig_curr = _signal_strength(np.hypot(x - gnb_curr.x, y - gnb_curr.y), gnb_curr.coverage_radius)
            sig_best = _signal_strength(np.hypot(x - gnb_best.x, y - gnb_best.y), gnb_best.coverage_radius)

            a3_triggered = (sig_best - sig_curr) > self.handover_hysteresis

            if a3_triggered:
                if not ue["ho_pending"] or ue["ho_candidate"] != best:
                    # New candidate — start TTT
                    ue["ho_pending"] = True
                    ue["ho_candidate"] = best
                    ue["ho_counter"] = 1
                else:
                    ue["ho_counter"] += 1

                if ue["ho_counter"] >= self.handover_ttt:
                    self._perform_handover(ue_id, ue, current, best)
            else:
                # Condition no longer met — cancel pending HO
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0

    def _perform_handover(self, ue_id: int, ue: Dict, from_gnb: int, to_gnb: int):
        ue["serving_gnb"] = to_gnb
        ue["ho_pending"] = False
        ue["ho_candidate"] = None
        ue["ho_counter"] = 0

        event = {
            "step": self.step_count,
            "ue_id": ue_id,
            "from_gnb": from_gnb,
            "to_gnb": to_gnb,
        }
        self.handover_log.append(event)

        if self.verbose:
            print(f"[HO] step={self.step_count} UE {ue_id}: gNB {from_gnb} → gNB {to_gnb}")

    def _ue_count_per_gnb(self) -> List[int]:
        counts = [0] * self.n_gnbs
        for ue in self._ues.values():
            s = ue["serving_gnb"]
            if s is not None:
                counts[s] += 1
        return counts