#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_gnb_wrapper.py

Multi-gNodeB Gymnasium environment wrapper for multiple NodeB objects.

Current assumptions
-------------------
- Each gNodeB may use a different or the same carrier/band
- Interference is considered only when two gNodeBs use the same carrier_id
- Wrapper handles:
    * global UE list
    * UE mobility
    * serving-cell association
    * A3-like handover with hysteresis + TTT
    * global reward
    * global observation
    * serving power / interference / SINR computation

Notes
-----
- This wrapper does NOT yet inject tracked UEs into NodeB internals.
  It is a clean network-level coordinator for testing multi-gNodeB logic.
- NodeB objects are still stepped normally with their own internal slice logic.
- This version is suitable to test:
    * multi-cell action splitting
    * multi-cell state aggregation
    * coverage-based handover logic
    * network-level metrics
    * interference and SINR in overlapping areas
"""

from typing import List, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiGNBWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        gnb_list: List,
        handover_hysteresis: float = 0.05,
        handover_ttt: int = 3,
        outage_penalty: float = 1.0,
        handover_penalty: float = 0.1,
        use_mean_gnb_reward: bool = True,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        gnb_list : list
            List of NodeB instances.
        handover_hysteresis : float
            Minimum signal advantage required before starting handover.
        handover_ttt : int
            Time-to-trigger in wrapper steps.
        outage_penalty : float
            Penalty weight for UEs that are outside coverage.
        handover_penalty : float
            Penalty weight per handover event.
        use_mean_gnb_reward : bool
            If True, global reward includes mean of NodeB rewards.
        verbose : bool
            Print debug info.
        """
        super().__init__()

        if gnb_list is None or len(gnb_list) == 0:
            raise ValueError("gnb_list must contain at least one NodeB.")

        self.gnbs: List = gnb_list
        self.n_gnbs: int = len(self.gnbs)
        self.verbose: bool = verbose

        self.handover_hysteresis: float = handover_hysteresis
        self.handover_ttt: int = handover_ttt
        self.outage_penalty: float = outage_penalty
        self.handover_penalty: float = handover_penalty
        self.use_mean_gnb_reward: bool = use_mean_gnb_reward

        # Validate timing consistency across gNBs
        first_slots = self.gnbs[0].slots_per_step
        first_slot_length = self.gnbs[0].slot_length
        for gnb in self.gnbs:
            if gnb.slots_per_step != first_slots:
                raise ValueError("All gNodeBs must have the same slots_per_step.")
            if gnb.slot_length != first_slot_length:
                raise ValueError("All gNodeBs must have the same slot_length.")

        self.slots_per_step: int = first_slots
        self.slot_length: float = first_slot_length
        self.dt: float = self.slots_per_step * self.slot_length

        # Per-gNB slice counts
        self._n_slices_per_gnb: List[int] = [gnb.n_slices_l1 for gnb in self.gnbs]
        self._total_action_dim: int = sum(self._n_slices_per_gnb)

        # Observation dimension:
        # concatenate all gNB states + global wrapper features
        self._gnb_state_dims = [len(gnb.get_state()) for gnb in self.gnbs]
        self._gnb_obs_dim = sum(self._gnb_state_dims)

        # Extra global features:
        # [total_tracked_ues, connected_ues, disconnected_ues, handovers_so_far]
        # + UE count per gNB
        self._global_extra_dim = 4 + self.n_gnbs
        self._obs_dim = self._gnb_obs_dim + self._global_extra_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Flat action: one positive value per slice of each gNB
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._total_action_dim,),
            dtype=np.float32,
        )

        # Global UE registry
        # ue_id -> dict with mobility and serving information
        self._ues: Dict[int, Dict] = {}
        self._next_ue_id: int = 0

        # Logs and counters
        self.handover_log: List[Dict] = []
        self.step_count: int = 0
        self.last_step_handover_count: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.last_step_handover_count = 0
        self.handover_log.clear()
        self._ues.clear()
        self._next_ue_id = 0

        for gnb in self.gnbs:
            gnb.reset()

        obs = self._get_obs()
        info = self._build_info(per_gnb_rewards=[0.0] * self.n_gnbs)

        return obs, info

    def step(self, action):
        action = self._validate_action(action)

        # 1) Update global UE mobility
        self._update_ue_positions()

        # 2) Handle association / handovers
        self.last_step_handover_count = 0
        self._check_handovers()

        # 3) Split global action into per-gNB PRB allocations
        gnb_actions = self._split_action(action)

        # 4) Step each gNB
        per_gnb_rewards = []
        per_gnb_infos = {}

        for i, (gnb, gnb_action) in enumerate(zip(self.gnbs, gnb_actions)):
            _, info = gnb.step(gnb_action)
            per_gnb_infos[f"gnb_{i}"] = info

            reward_labels, _ = gnb.compute_reward()
            local_reward = float(np.mean(reward_labels))
            per_gnb_rewards.append(local_reward)

        # 5) Build global reward
        reward = self._compute_global_reward(per_gnb_rewards)

        # 6) Build observation + info
        obs = self._get_obs()
        info = self._build_info(per_gnb_rewards=per_gnb_rewards)
        info["per_gnb_info"] = per_gnb_infos

        self.step_count += 1

        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"\n{'=' * 60}")
        print(f"Step: {self.step_count}")
        print(f"Tracked UEs: {len(self._ues)}")
        print(f"Connected UEs: {self._count_connected_ues()}")
        print(f"Disconnected UEs: {self._count_disconnected_ues()}")
        print(f"Total handovers: {len(self.handover_log)}")
        print(f"Handovers this step: {self.last_step_handover_count}")
        print("-" * 60)

        ue_counts = self._ue_count_per_gnb()
        for i, gnb in enumerate(self.gnbs):
            print(
                f"gNB idx={i} id={gnb.id} "
                f"pos=({gnb.x:.1f}, {gnb.y:.1f}) "
                f"radius={gnb.coverage_radius:.1f} "
                f"tracked_ues={ue_counts[i]} "
                f"slices={gnb.n_slices_l1} "
                f"n_prbs={gnb.n_prbs}"
            )
        print(f"{'=' * 60}")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Public UE management
    # ------------------------------------------------------------------

    def add_ue(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        slice_type: str = "eMBB",
    ) -> int:
        """
        Add a tracked UE to the global wrapper.

        This UE is currently used for:
        - mobility
        - serving-cell association
        - handover logic
        - network-level statistics
        - radio metric testing

        Returns
        -------
        ue_id : int
        """
        ue_id = self._next_ue_id
        self._next_ue_id += 1

        serving_gnb = self._best_gnb(x, y)

        self._ues[ue_id] = {
            "id": ue_id,
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "slice_type": slice_type,
            "serving_gnb": serving_gnb,
            "ho_pending": False,
            "ho_candidate": None,
            "ho_counter": 0,
        }

        if self.verbose:
            print(
                f"[UE added] ue_id={ue_id}, "
                f"pos=({x:.2f},{y:.2f}), "
                f"vel=({vx:.2f},{vy:.2f}), "
                f"serving_gnb={serving_gnb}"
            )

        return ue_id

    def remove_ue(self, ue_id: int):
        self._ues.pop(ue_id, None)

    def get_ue(self, ue_id: int) -> Optional[Dict]:
        return self._ues.get(ue_id, None)

    def get_all_ues(self) -> Dict[int, Dict]:
        return self._ues

    # ------------------------------------------------------------------
    # Core internal methods
    # ------------------------------------------------------------------

    def _validate_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).flatten()

        if action.shape[0] != self._total_action_dim:
            raise ValueError(
                f"Action has wrong size. "
                f"Expected {self._total_action_dim}, got {action.shape[0]}."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values.")

        action = np.clip(action, 0.0, 1.0)
        return action

    def _get_obs(self) -> np.ndarray:
        gnb_parts = [gnb.get_state().astype(np.float32) for gnb in self.gnbs]

        total_tracked = float(len(self._ues))
        connected = float(self._count_connected_ues())
        disconnected = float(self._count_disconnected_ues())
        total_handover = float(len(self.handover_log))
        ue_counts = np.array(self._ue_count_per_gnb(), dtype=np.float32)

        global_features = np.array(
            [total_tracked, connected, disconnected, total_handover],
            dtype=np.float32
        )

        if len(gnb_parts) > 0:
            obs = np.concatenate(gnb_parts + [global_features, ue_counts], axis=0)
        else:
            obs = np.concatenate([global_features, ue_counts], axis=0)

        return obs.astype(np.float32)

    def _split_action(self, action: np.ndarray) -> List[np.ndarray]:
        """
        Split flat action into one PRB allocation array per gNB.

        For each gNB:
        - take its local slice action segment
        - renormalize
        - convert into integer PRBs summing exactly to gnb.n_prbs
        """
        result = []
        idx = 0

        for gnb, n_slices in zip(self.gnbs, self._n_slices_per_gnb):
            raw = action[idx: idx + n_slices]
            idx += n_slices

            raw_sum = raw.sum()
            if raw_sum <= 1e-12:
                fractions = np.ones(n_slices, dtype=np.float32) / n_slices
            else:
                fractions = raw / raw_sum

            prbs = np.floor(fractions * gnb.n_prbs).astype(int)

            leftover = int(gnb.n_prbs - prbs.sum())
            if leftover > 0:
                order = np.argsort(fractions)[::-1]
                for k in range(leftover):
                    prbs[order[k % n_slices]] += 1

            result.append(prbs)

        return result

    def _compute_global_reward(self, per_gnb_rewards: List[float]) -> float:
        """
        Current reward:
        - mean local gNB reward
        - minus outage penalty
        - minus handover penalty
        """
        reward = 0.0

        if self.use_mean_gnb_reward and len(per_gnb_rewards) > 0:
            reward += float(np.mean(per_gnb_rewards))

        n_tracked = max(len(self._ues), 1)
        n_disconnected = self._count_disconnected_ues()

        outage_term = self.outage_penalty * (n_disconnected / n_tracked)
        ho_term = self.handover_penalty * self.last_step_handover_count

        reward -= outage_term
        reward -= ho_term

        return float(reward)

    def _build_info(self, per_gnb_rewards: List[float]) -> Dict:
        info = {
            "step": self.step_count,
            "n_gnbs": self.n_gnbs,
            "n_tracked_ues": len(self._ues),
            "n_connected_ues": self._count_connected_ues(),
            "n_disconnected_ues": self._count_disconnected_ues(),
            "ue_per_gnb": self._ue_count_per_gnb(),
            "handover_count_total": len(self.handover_log),
            "handover_count_step": self.last_step_handover_count,
            "per_gnb_rewards": per_gnb_rewards,
            "interference_model": "same_carrier => interference, different_carrier => no_interference",
        }
        return info

    # ------------------------------------------------------------------
    # Radio / interference / SINR
    # ------------------------------------------------------------------

    def _dbm_to_w(self, dbm: float) -> float:
        return 10 ** ((dbm - 30.0) / 10.0)

    def _w_to_dbm(self, w: float) -> float:
        if w <= 0:
            return -np.inf
        return 10 * np.log10(w) + 30.0

    def compute_serving_power_dbm(self, ue: Dict) -> float:
        """
        Received serving power at UE position in dBm.
        """
        serving_idx = ue["serving_gnb"]
        if serving_idx is None:
            return -np.inf

        gnb = self.gnbs[serving_idx]
        return gnb.get_received_power_dbm(ue["x"], ue["y"])

    def compute_interference_dbm(self, ue: Dict) -> float:
        """
        Sum interference from all non-serving gNodeBs using the same carrier.
        Returns total interference in dBm.
        """
        serving_idx = ue["serving_gnb"]
        if serving_idx is None:
            return -np.inf

        serving_gnb = self.gnbs[serving_idx]
        interf_w = 0.0

        for i, gnb in enumerate(self.gnbs):
            if i == serving_idx:
                continue

            # only same-carrier gNBs interfere
            if not serving_gnb.uses_same_carrier(gnb):
                continue

            # ignore if UE not covered by interferer
            if not gnb.is_point_in_coverage(ue["x"], ue["y"]):
                continue

            interf_w += gnb.get_received_power_watts(ue["x"], ue["y"])

        return self._w_to_dbm(interf_w)

    def compute_sinr_db(self, ue: Dict, rb_bandwidth_hz: Optional[float] = None) -> float:
        """
        Compute SINR at UE position in dB:
            SINR = S / (N + I)
        """
        serving_idx = ue["serving_gnb"]
        if serving_idx is None:
            return -np.inf

        serving_gnb = self.gnbs[serving_idx]

        signal_dbm = self.compute_serving_power_dbm(ue)
        if not np.isfinite(signal_dbm):
            return -np.inf

        if rb_bandwidth_hz is None:
            rb_bandwidth_hz = serving_gnb.get_rb_bandwidth_hz()

        noise_dbm = serving_gnb.get_noise_power_dbm(rb_bandwidth_hz=rb_bandwidth_hz)
        interf_dbm = self.compute_interference_dbm(ue)

        signal_w = self._dbm_to_w(signal_dbm)
        noise_w = self._dbm_to_w(noise_dbm)
        interf_w = 0.0 if not np.isfinite(interf_dbm) else self._dbm_to_w(interf_dbm)

        sinr_w = signal_w / (noise_w + interf_w)
        return 10 * np.log10(sinr_w)

    def get_ue_radio_metrics(self, ue_id: int) -> Dict:
        """
        Convenience method for testing radio metrics of one UE.
        """
        ue = self._ues.get(ue_id)
        if ue is None:
            raise KeyError(f"UE {ue_id} not found.")

        return {
            "ue_id": ue_id,
            "x": ue["x"],
            "y": ue["y"],
            "serving_gnb": ue["serving_gnb"],
            "serving_power_dbm": self.compute_serving_power_dbm(ue),
            "interference_dbm": self.compute_interference_dbm(ue),
            "sinr_db": self.compute_sinr_db(ue),
        }

    # ------------------------------------------------------------------
    # Mobility and handover
    # ------------------------------------------------------------------

    def _update_ue_positions(self):
        for ue in self._ues.values():
            ue["x"] += ue["vx"] * self.dt
            ue["y"] += ue["vy"] * self.dt

    def _best_gnb(self, x: float, y: float) -> Optional[int]:
        """
        Return the index of the best gNB based on strongest received power.
        If no gNB covers the UE, return None.
        """
        best_idx = None
        best_rx_dbm = -np.inf

        for i, gnb in enumerate(self.gnbs):
            if gnb.is_point_in_coverage(x, y):
                rx_dbm = gnb.get_received_power_dbm(x, y)
                if rx_dbm > best_rx_dbm:
                    best_rx_dbm = rx_dbm
                    best_idx = i

        return best_idx

    def _check_handovers(self):
        for ue_id, ue in self._ues.items():
            x = ue["x"]
            y = ue["y"]

            current = ue["serving_gnb"]
            best = self._best_gnb(x, y)

            # UE is outside all coverage
            if best is None:
                if current is not None and self.verbose:
                    print(f"[UE {ue_id}] left all coverage areas")

                ue["serving_gnb"] = None
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0
                continue

            # UE re-enters coverage
            if current is None:
                ue["serving_gnb"] = best
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0

                if self.verbose:
                    print(f"[UE {ue_id}] attached to gNB {best} after re-entering coverage")
                continue

            # Same serving gNB remains best
            if best == current:
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0
                continue

            # Evaluate A3-like condition using received power
            gnb_curr = self.gnbs[current]
            gnb_best = self.gnbs[best]

            sig_curr = gnb_curr.get_received_power_dbm(x, y) if gnb_curr.is_point_in_coverage(x, y) else -np.inf
            sig_best = gnb_best.get_received_power_dbm(x, y) if gnb_best.is_point_in_coverage(x, y) else -np.inf

            # difference in dB
            a3_triggered = (sig_best - sig_curr) > self.handover_hysteresis

            if a3_triggered:
                if (not ue["ho_pending"]) or (ue["ho_candidate"] != best):
                    ue["ho_pending"] = True
                    ue["ho_candidate"] = best
                    ue["ho_counter"] = 1
                else:
                    ue["ho_counter"] += 1

                if ue["ho_counter"] >= self.handover_ttt:
                    self._perform_handover(ue_id, current, best)
            else:
                ue["ho_pending"] = False
                ue["ho_candidate"] = None
                ue["ho_counter"] = 0

    def _perform_handover(self, ue_id: int, from_gnb: int, to_gnb: int):
        ue = self._ues[ue_id]
        ue["serving_gnb"] = to_gnb
        ue["ho_pending"] = False
        ue["ho_candidate"] = None
        ue["ho_counter"] = 0

        event = {
            "step": self.step_count,
            "ue_id": ue_id,
            "from_gnb": from_gnb,
            "to_gnb": to_gnb,
            "x": ue["x"],
            "y": ue["y"],
        }
        self.handover_log.append(event)
        self.last_step_handover_count += 1

        if self.verbose:
            print(
                f"[HO] step={self.step_count}, ue_id={ue_id}, "
                f"{from_gnb} -> {to_gnb}, "
                f"pos=({ue['x']:.2f}, {ue['y']:.2f})"
            )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _ue_count_per_gnb(self) -> List[int]:
        counts = [0] * self.n_gnbs
        for ue in self._ues.values():
            serving = ue["serving_gnb"]
            if serving is not None:
                counts[serving] += 1
        return counts

    def _count_connected_ues(self) -> int:
        return sum(1 for ue in self._ues.values() if ue["serving_gnb"] is not None)

    def _count_disconnected_ues(self) -> int:
        return sum(1 for ue in self._ues.values() if ue["serving_gnb"] is None)