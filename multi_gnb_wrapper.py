#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import gymnasium as gym
import numpy as np

from slice_ran import UE, CBR
from traffic_generators import CbrSource


class MultiGNBWrapper(gym.Env):
    """
    Prototype multi-gNB RL wrapper for UE reassociation / traffic steering.

    Observation (23 dims):
    [
        ue_x, ue_y, ue_vx, ue_vy, ue_speed, ue_queue, ue_th,
        serving_sinr, serving_load, serving_dist, serving_approach,
        cand1_sinr, cand1_load, cand1_dist, cand1_approach,
        cand2_sinr, cand2_load, cand2_dist, cand2_approach,
        cand3_sinr, cand3_load, cand3_dist, cand3_approach
    ]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gnb_list: List,
        handover_hysteresis: float = 0.05,
        handover_ttt: int = 3,
        outage_penalty: float = 1.0,
        handover_penalty: float = 0.1,
        use_mean_gnb_reward: bool = True,
        verbose: bool = False,
        step_dt: float = 1e-3,
        max_candidates: int = 3,
        max_episode_steps: int = 100,
    ):
        super().__init__()

        if not gnb_list:
            raise ValueError("gnb_list must contain at least one gNB")

        self.gnbs = list(gnb_list)
        self.n_gnbs = len(self.gnbs)
        self.history = []
        self.handover_hysteresis = float(handover_hysteresis)
        self.handover_ttt = int(handover_ttt)
        self.outage_penalty = float(outage_penalty)
        self.handover_penalty = float(handover_penalty)
        self.use_mean_gnb_reward = bool(use_mean_gnb_reward)
        self.verbose = bool(verbose)
        self.step_dt = float(step_dt)
        self.max_candidates = int(max_candidates)
        self.max_episode_steps = int(max_episode_steps)

        self.action_space = gym.spaces.Discrete(1 + self.max_candidates)

        self.observation_space = gym.spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(23,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._next_ue_id = 0
        self._ues: Dict[int, UE] = {}
        self._current_control_ue_id: Optional[int] = None
        self._round_robin_order: List[int] = []
        self._rr_index = 0
        self._step_count = 0

        self._handover_target_counter: Dict[int, Dict[int, int]] = {}
        self._last_serving_gnb: Dict[int, Optional[int]] = {}
        self._prev_serving_gnb: Dict[int, Optional[int]] = {}

        self._last_candidates: List[int] = []
        self._last_info: Dict = {}
        self._last_reward_breakdown: Dict = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.history = []
        self._step_count = 0
        self._current_control_ue_id = None
        self._rr_index = 0
        self._last_candidates = []
        self._last_info = {}
        self._last_reward_breakdown = {}
        self.handover_events = []
        for gnb in self.gnbs:
            if hasattr(gnb, "reset"):
                gnb.reset()

        for ue in self._ues.values():
            ue.queue = 0
            ue.th = 0.0
            ue.bits = 0
            ue.new_bits = 0
            ue.dropped_bits = 0
            ue.total_bits_arrived = 0
            ue.wait_time = 0
            ue.snr = 0
            ue.e_snr = 0
            ue.sinr = 0
            ue.e_sinr = 0
            ue.prbs = 0
            ue.p = 0
            ue.connected = True
            ue.target_gnb = None
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            ue.serving_power_dbm = -np.inf
            ue.interference_dbm = -np.inf
            ue.noise_dbm = -np.inf

            best = self._find_best_gnb_for_ue(ue)
            ue.serving_gnb = best.id if best is not None else None
            ue.connected = ue.serving_gnb is not None
            self._ues[ue.id] = ue

            if best is not None:
                best.attach_ue(ue)
            self._handover_target_counter[ue.id] = {}
            self._last_serving_gnb[ue.id] = ue.serving_gnb
            self._prev_serving_gnb[ue.id] = None

        self._refresh_round_robin_order()
        self._current_control_ue_id = self._pick_next_control_ue()

        obs = self._get_observation_for_current_ue()
        info = self._build_info(per_gnb_rewards=[0.0] * self.n_gnbs)
        self._last_info = info
        return obs, info

    def step(self, action):
        self._step_count += 1

        per_gnb_rewards = self._advance_gnbs()

        if not self._ues:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = False
            truncated = self._step_count >= self.max_episode_steps
            info = self._build_info(per_gnb_rewards=per_gnb_rewards)
            self._last_info = info
            return obs, reward, terminated, truncated, info

        if self._current_control_ue_id is None or self._current_control_ue_id not in self._ues:
            self._current_control_ue_id = self._pick_next_control_ue()

        ue = self._ues[self._current_control_ue_id]

        for tracked_ue in self._ues.values():
            tracked_ue.update_position(self.step_dt)
            tracked_ue.traffic_step()
            if tracked_ue.queue > 0:
                tracked_ue.wait_time += 1
            else:
                tracked_ue.wait_time = max(tracked_ue.wait_time - 1, 0)

        candidates = self.get_candidate_gnbs(ue, top_k=self.max_candidates)
        self._last_candidates = [g.id for g in candidates]

        target_gnb = self._interpret_action_for_ue(action, ue, candidates)
        handover_done = self._apply_handover_logic(ue, target_gnb)
        old_gnb = self._get_gnb_by_id(current_id) if current_id is not None else None
        new_gnb = self._get_gnb_by_id(target_id) if target_id is not None else None

        if old_gnb is not None:
            old_gnb.detach_ue(ue.id)

        if new_gnb is not None:
            new_gnb.attach_ue(ue)

        ue.serving_gnb = target_id
        ue.connected = new_gnb is not None
        self._simulate_radio_and_service()

        pingpong = self._is_ping_pong(ue)
        reward = self.compute_reassociation_reward(
            ue=ue,
            target_gnb=self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None,
            handover_done=handover_done,
            pingpong=pingpong,
        )

        self._current_control_ue_id = self._pick_next_control_ue()
        obs = self._get_observation_for_current_ue()

        terminated = False
        truncated = self._step_count >= int(self.max_episode_steps)

        info = self._build_info(per_gnb_rewards=per_gnb_rewards)
        info["current_control_ue_id"] = self._current_control_ue_id
        info["last_action_candidates"] = self._last_candidates
        self._last_info = info
        self._log_step(float(reward))
        return obs, float(reward), terminated, truncated, info

    def _log_step(self, reward: float):
        row = {
            "step": int(self._step_count),
            "control_ue_id": self._current_control_ue_id,
            "reward": float(reward),
            "n_connected_ues": int(sum(1 for ue in self._ues.values() if ue.connected)),
            "n_disconnected_ues": int(sum(1 for ue in self._ues.values() if not ue.connected)),
        }

        for ue in self._ues.values():
            row[f"ue{ue.id}_x"] = float(ue.x)
            row[f"ue{ue.id}_y"] = float(ue.y)
            row[f"ue{ue.id}_serving_gnb"] = -1 if ue.serving_gnb is None else int(ue.serving_gnb)
            row[f"ue{ue.id}_sinr"] = float(ue.e_sinr if np.isfinite(ue.e_sinr) else -np.inf)
            row[f"ue{ue.id}_throughput"] = float(ue.th)
            row[f"ue{ue.id}_queue"] = float(ue.queue)
            row[f"ue{ue.id}_connected"] = int(bool(ue.connected))
            row[f"ue{ue.id}_handover_pending"] = int(bool(ue.ho_pending))

        self.history.append(row)
    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_ue(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        slice_type: str = "eMBB",
        bit_rate: float = 1_000_000.0,
        buffer_size: float = np.inf,
        **ue_kwargs,
    ) -> int:
        ue_id = self._next_ue_id
        self._next_ue_id += 1

        ue = UE(
            id=ue_id,
            slice_ran_id=0,
            traffic_source=CbrSource(bit_rate=bit_rate, step_length=self.step_dt),
            type=CBR,
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            slot_length=self.step_dt,
            slice_type=slice_type,
            buffer_size=buffer_size,
            **ue_kwargs,
        )

        best = self._find_best_gnb_for_ue(ue)
        ue.serving_gnb = best.id if best is not None else None
        ue.connected = ue.serving_gnb is not None

        self._ues[ue_id] = ue
        self._handover_target_counter[ue_id] = {}
        self._last_serving_gnb[ue_id] = ue.serving_gnb
        self._prev_serving_gnb[ue_id] = None

        self._refresh_round_robin_order()
        if self._current_control_ue_id is None:
            self._current_control_ue_id = ue_id

        return ue_id

    def get_ue(self, ue_id: int) -> UE:
        return self._ues[ue_id]

    def get_all_ues(self) -> List[UE]:
        return list(self._ues.values())

    def get_ue_radio_metrics(self, ue_id: int) -> Dict[str, float]:
        ue = self._ues[ue_id]
        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None

        metrics = self._compute_link_metrics(serving, ue) if serving is not None else {
            "rx_power_dbm": -np.inf,
            "noise_dbm": -np.inf,
            "interference_dbm": -np.inf,
            "snr_db": -np.inf,
            "sinr_db": -np.inf,
        }

        return {
            "ue_id": ue.id,
            "serving_gnb": ue.serving_gnb,
            "connected": bool(ue.connected),
            "x": float(ue.x),
            "y": float(ue.y),
            "vx": float(ue.vx),
            "vy": float(ue.vy),
            "queue": float(ue.queue),
            "throughput": float(ue.th),
            "snr_db": float(metrics["snr_db"]),
            "sinr_db": float(metrics["sinr_db"]),
            "rx_power_dbm": float(metrics["rx_power_dbm"]),
            "noise_dbm": float(metrics["noise_dbm"]),
            "interference_dbm": float(metrics["interference_dbm"]),
            "target_gnb": ue.target_gnb,
            "ho_pending": bool(ue.ho_pending),
            "ho_candidate": ue.ho_candidate,
            "ho_counter": int(ue.ho_counter),
        }

    # ------------------------------------------------------------------
    # Candidate selection and observation
    # ------------------------------------------------------------------

    def get_candidate_gnbs(self, ue: UE, top_k: int = 3) -> List:
        scored: List[Tuple[float, object]] = []
        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None

        for gnb in self.gnbs:
            if not self._is_in_coverage(gnb, ue):
                continue

            sinr_db = self._get_sinr_db(gnb, ue)
            if not np.isfinite(sinr_db):
                continue

            scored.append((sinr_db, gnb))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        seen = set()

        if serving is not None and self._is_in_coverage(serving, ue):
            selected.append(serving)
            seen.add(serving.id)

        for _, gnb in scored:
            if gnb.id in seen:
                continue
            selected.append(gnb)
            seen.add(gnb.id)
            if len(selected) >= top_k + (1 if serving is not None else 0):
                break

        return selected[: top_k + 1]

    def build_ue_observation(self, ue: UE, candidates: List) -> np.ndarray:
        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None
        alt_candidates = [g for g in candidates if serving is None or g.id != serving.id]
        alt_candidates = alt_candidates[: self.max_candidates]

        speed = self._get_speed(ue)

        obs = [
            float(ue.x),
            float(ue.y),
            float(ue.vx),
            float(ue.vy),
            float(speed),
            float(ue.queue),
            float(ue.th),
        ]

        if serving is not None and self._is_in_coverage(serving, ue):
            serving_metrics = self._compute_link_metrics(serving, ue)
            serving_sinr = serving_metrics["sinr_db"]
            serving_load = self._estimate_gnb_load(serving.id)
            serving_dist = self._get_distance(serving, ue)
            serving_approach = self._get_approach_score(serving, ue)
        else:
            serving_sinr = -np.inf
            serving_load = 1.0
            serving_dist = 1e9
            serving_approach = 0.0

        obs.extend([
            float(serving_sinr),
            float(serving_load),
            float(serving_dist),
            float(serving_approach),
        ])

        for gnb in alt_candidates:
            metrics = self._compute_link_metrics(gnb, ue)
            cand_sinr = metrics["sinr_db"]
            cand_load = self._estimate_gnb_load(gnb.id)
            cand_dist = self._get_distance(gnb, ue)
            cand_approach = self._get_approach_score(gnb, ue)

            obs.extend([
                float(cand_sinr),
                float(cand_load),
                float(cand_dist),
                float(cand_approach),
            ])

        while len(obs) < 23:
            obs.extend([float(-np.inf), 1.0, 1e9, 0.0])

        return np.asarray(obs[:23], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reassociation_reward(
        self,
        ue: UE,
        target_gnb,
        handover_done: bool,
        pingpong: bool,
    ) -> float:
        th_norm = self._clip01(ue.th / 1e7)
        q_norm = self._clip01(ue.queue / 1e6)
        w_norm = self._clip01(ue.wait_time / 100.0)

        if target_gnb is None:
            sinr_norm = 0.0
            load_norm = 1.0
        else:
            metrics = self._compute_link_metrics(target_gnb, ue)
            sinr_norm = self._normalize_sinr_db(metrics["sinr_db"])
            load_norm = self._clip01(self._estimate_gnb_load(target_gnb.id))

        ho_pen = 1.0 if handover_done else 0.0
        pp_pen = 1.0 if pingpong else 0.0
        outage = 0.0 if (target_gnb is not None and ue.connected) else 1.0

        sla_term = self._slice_sla_bonus(ue, th_norm, w_norm, sinr_norm, load_norm)

        reward_throughput = 3.0 * th_norm
        reward_queue = -1.0 * q_norm
        reward_delay = -0.5 * w_norm
        reward_handover = -self.handover_penalty * ho_pen
        reward_pingpong = -1.0 * pp_pen
        reward_sla = 1.0 * sla_term
        reward_outage = -2.0 * outage

        reward = (
            reward_throughput
            + reward_queue
            + reward_delay
            + reward_handover
            + reward_pingpong
            + reward_sla
            + reward_outage
        )

        self._last_reward_breakdown = {
            "throughput": float(reward_throughput),
            "queue": float(reward_queue),
            "delay": float(reward_delay),
            "handover": float(reward_handover),
            "pingpong": float(reward_pingpong),
            "sla": float(reward_sla),
            "outage": float(reward_outage),
            "total": float(reward),
        }

        return float(reward)

    # ------------------------------------------------------------------
    # Internal step helpers
    # ------------------------------------------------------------------

    def _advance_gnbs(self) -> List[float]:
        rewards = []
        for gnb in self.gnbs:
            reward = 0.0
            if hasattr(gnb, "n_slices_l1") and hasattr(gnb, "step"):
                n_slices = max(int(gnb.n_slices_l1), 1)
                base = int(gnb.n_prbs // n_slices)
                action = np.full((n_slices,), base, dtype=int)
                rem = int(gnb.n_prbs - action.sum())
                if rem > 0:
                    action[:rem] += 1

                try:
                    _state, info = gnb.step(action.tolist())
                    if isinstance(info, dict):
                        reward = float(info.get("reward", 0.0))
                except Exception:
                    reward = 0.0
            rewards.append(reward)
        return rewards

    def _simulate_radio_and_service(self):
        attached = self._group_ues_by_serving_gnb()

        for gnb_id, ue_list in attached.items():
            if gnb_id is None:
                for ue in ue_list:
                    ue.connected = False
                    ue.bits = 0
                    ue.prbs = 0
                    ue.serving_power_dbm = -np.inf
                    ue.interference_dbm = -np.inf
                    ue.noise_dbm = -np.inf
                    ue.estimate_sinr(-np.inf)
                    ue.transmission_step(received=False)
                continue

            gnb = self._get_gnb_by_id(gnb_id)
            if gnb is None:
                for ue in ue_list:
                    ue.connected = False
                    ue.bits = 0
                    ue.prbs = 0
                    ue.serving_power_dbm = -np.inf
                    ue.interference_dbm = -np.inf
                    ue.noise_dbm = -np.inf
                    ue.estimate_sinr(-np.inf)
                    ue.transmission_step(received=False)
                continue

            n_users = max(len(ue_list), 1)
            prbs_per_ue = max(int(gnb.n_prbs // n_users), 1)

            for ue in ue_list:
                metrics = self._compute_link_metrics(gnb, ue)

                rx_dbm = metrics["rx_power_dbm"]
                noise_dbm = metrics["noise_dbm"]
                interf_dbm = metrics["interference_dbm"]
                sinr_db = metrics["sinr_db"]
                snr_db = metrics["snr_db"]

                ue.connected = np.isfinite(sinr_db)
                ue.serving_power_dbm = rx_dbm
                ue.noise_dbm = noise_dbm
                ue.interference_dbm = interf_dbm
                ue.estimate_sinr(float(sinr_db))
                ue.estimate_snr([float(snr_db)] if np.isfinite(snr_db) else [-np.inf])
                ue.prbs = prbs_per_ue

                if not ue.connected:
                    ue.bits = 0
                    ue.transmission_step(received=False)
                    continue

                ue.bits = self._estimate_bits_for_ue(
                    ue=ue,
                    sinr_db=sinr_db,
                    prbs=prbs_per_ue,
                    gnb=gnb,
                )
                ue.transmission_step(received=True)

    def _interpret_action_for_ue(self, action, ue: UE, candidates: List):
        if np.isscalar(action) or isinstance(action, (int, np.integer)):
            try:
                chosen_idx = int(action)
            except Exception:
                chosen_idx = int(np.asarray(action).item())
        else:
            arr = np.asarray(action, dtype=float).reshape(-1)
            if arr.size == 1:
                chosen_idx = int(arr[0])
            else:
                chosen_idx = int(np.argmax(arr))

        serving = self._get_gnb_by_id(ue.serving_gnb) if ue.serving_gnb is not None else None
        alt_candidates = [g for g in candidates if serving is None or g.id != serving.id]
        alt_candidates = alt_candidates[: self.max_candidates]

        if chosen_idx == 0:
            return serving

        alt_idx = chosen_idx - 1
        if 0 <= alt_idx < len(alt_candidates):
            return alt_candidates[alt_idx]
        return serving

    def _apply_handover_logic(self, ue: UE, target_gnb) -> bool:
        current_id = ue.serving_gnb
        target_id = None if target_gnb is None else target_gnb.id
        self.handover_events = []
        if target_id is None:
            ue.target_gnb = None
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            ue.connected = False
            ue.serving_gnb = None
            return False

        if current_id == target_id:
            ue.target_gnb = current_id
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            ue.connected = True
            return False

        current_gnb = self._get_gnb_by_id(current_id) if current_id is not None else None

        target_sinr = self._get_sinr_db(target_gnb, ue)
        current_sinr = self._get_sinr_db(current_gnb, ue) if current_gnb is not None else -np.inf
        improvement = target_sinr - current_sinr

        if improvement < self.handover_hysteresis:
            ue.target_gnb = current_id
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            return False

        counter_map = self._handover_target_counter.setdefault(ue.id, {})
        counter_map[target_id] = counter_map.get(target_id, 0) + 1

        ue.target_gnb = target_id
        ue.ho_pending = True
        ue.ho_candidate = target_id
        ue.ho_counter = counter_map[target_id]

        if counter_map[target_id] >= self.handover_ttt:
            self._prev_serving_gnb[ue.id] = self._last_serving_gnb.get(ue.id)
            self._last_serving_gnb[ue.id] = current_id
            old_serving = current_id
            new_serving = target_id

            self.handover_events.append({
                "step": int(self._step_count),
                "ue_id": int(ue.id),
                "from_gnb": -1 if old_serving is None else int(old_serving),
                "to_gnb": int(new_serving),
            })
            ue.serving_gnb = target_id
            ue.connected = True
            ue.ho_pending = False
            ue.ho_candidate = None
            ue.ho_counter = 0
            counter_map.clear()
            return True

        return False

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_history(self):
        return list(self.history)

    def get_handover_events(self):
        return list(self.handover_events)
    def _build_info(self, per_gnb_rewards: Optional[List[float]] = None) -> Dict:
        ue_per_gnb = [0] * self.n_gnbs
        connected = 0
        disconnected = 0

        for ue in self._ues.values():
            if ue.connected and ue.serving_gnb is not None:
                connected += 1
                gidx = self._gnb_index_from_id(ue.serving_gnb)
                if gidx is not None:
                    ue_per_gnb[gidx] += 1
            else:
                disconnected += 1

        if per_gnb_rewards is None:
            per_gnb_rewards = [0.0] * self.n_gnbs

        info = {
            "step_count": self._step_count,
            "n_gnbs": self.n_gnbs,
            "n_tracked_ues": len(self._ues),
            "n_connected_ues": connected,
            "n_disconnected_ues": disconnected,
            "ue_per_gnb": ue_per_gnb,
            "per_gnb_rewards": list(per_gnb_rewards),
            "mean_gnb_reward": float(np.mean(per_gnb_rewards)) if per_gnb_rewards else 0.0,
            "current_control_ue_id": self._current_control_ue_id,
        }

        if self._last_reward_breakdown:
            info["reward_breakdown"] = dict(self._last_reward_breakdown)

        return info

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _get_observation_for_current_ue(self) -> np.ndarray:
        if self._current_control_ue_id is None or self._current_control_ue_id not in self._ues:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        ue = self._ues[self._current_control_ue_id]
        candidates = self.get_candidate_gnbs(ue, top_k=self.max_candidates)
        self._last_candidates = [g.id for g in candidates]
        return self.build_ue_observation(ue, candidates)

    def _find_best_gnb_for_ue(self, ue: UE):
        best_gnb = None
        best_sinr = -np.inf
        for gnb in self.gnbs:
            sinr_db = self._get_sinr_db(gnb, ue)
            if sinr_db > best_sinr:
                best_sinr = sinr_db
                best_gnb = gnb
        return best_gnb

    def _pick_next_control_ue(self) -> Optional[int]:
        self._refresh_round_robin_order()
        if not self._round_robin_order:
            return None
        ue_id = self._round_robin_order[self._rr_index % len(self._round_robin_order)]
        self._rr_index = (self._rr_index + 1) % len(self._round_robin_order)
        return ue_id

    def _refresh_round_robin_order(self):
        self._round_robin_order = sorted(self._ues.keys())

    def _group_ues_by_serving_gnb(self) -> Dict[Optional[int], List[UE]]:
        groups: Dict[Optional[int], List[UE]] = {}
        for ue in self._ues.values():
            groups.setdefault(ue.serving_gnb, []).append(ue)
        return groups

    def _gnb_index_from_id(self, gnb_id: Optional[int]) -> Optional[int]:
        if gnb_id is None:
            return None
        for idx, gnb in enumerate(self.gnbs):
            if gnb.id == gnb_id:
                return idx
        return None

    def _get_gnb_by_id(self, gnb_id: Optional[int]):
        if gnb_id is None:
            return None
        for gnb in self.gnbs:
            if gnb.id == gnb_id:
                return gnb
        return None

    def _is_in_coverage(self, gnb, ue: UE) -> bool:
        if hasattr(gnb, "is_point_in_coverage"):
            return bool(gnb.is_point_in_coverage(ue.x, ue.y))
        return True

    def _get_rx_power_dbm(self, gnb, ue: UE) -> float:
        if gnb is None:
            return -np.inf
        if hasattr(gnb, "get_received_power_dbm"):
            try:
                return float(gnb.get_received_power_dbm(ue.x, ue.y))
            except Exception:
                return -np.inf
        return -np.inf

    def _get_noise_power_dbm(self, gnb) -> float:
        if gnb is None:
            return -100.0
        if hasattr(gnb, "get_noise_power_dbm"):
            try:
                return float(gnb.get_noise_power_dbm())
            except Exception:
                pass
        return -100.0

    def _dbm_to_watts(self, p_dbm: float) -> float:
        if not np.isfinite(p_dbm):
            return 0.0
        return 10.0 ** ((p_dbm - 30.0) / 10.0)

    def _watts_to_dbm(self, p_watts: float) -> float:
        if p_watts <= 0.0:
            return -np.inf
        return 10.0 * np.log10(p_watts) + 30.0

    def _compute_interference_watts(self, serving_gnb, ue: UE) -> float:
        if serving_gnb is None:
            return 0.0

        total_watts = 0.0
        for other in self.gnbs:
            if other.id == serving_gnb.id:
                continue

            try:
                same_carrier = serving_gnb.uses_same_carrier(other)
            except Exception:
                same_carrier = False

            if not same_carrier:
                continue

            if not self._is_in_coverage(other, ue):
                continue

            if hasattr(other, "get_received_power_watts"):
                p_w = float(other.get_received_power_watts(ue.x, ue.y))
            else:
                p_dbm = self._get_rx_power_dbm(other, ue)
                p_w = self._dbm_to_watts(p_dbm)

            total_watts += max(p_w, 0.0)

        return total_watts

    def _compute_link_metrics(self, gnb, ue: UE) -> Dict[str, float]:
        if gnb is None or not self._is_in_coverage(gnb, ue):
            noise_dbm = self._get_noise_power_dbm(gnb) if gnb is not None else -np.inf
            return {
                "rx_power_dbm": -np.inf,
                "noise_dbm": float(noise_dbm),
                "interference_dbm": -np.inf,
                "snr_db": -np.inf,
                "sinr_db": -np.inf,
            }

        rx_dbm = self._get_rx_power_dbm(gnb, ue)
        noise_dbm = self._get_noise_power_dbm(gnb)

        if not np.isfinite(rx_dbm) or not np.isfinite(noise_dbm):
            return {
                "rx_power_dbm": float(rx_dbm),
                "noise_dbm": float(noise_dbm),
                "interference_dbm": -np.inf,
                "snr_db": -np.inf,
                "sinr_db": -np.inf,
            }

        sig_w = self._dbm_to_watts(rx_dbm)
        noise_w = self._dbm_to_watts(noise_dbm)
        interf_w = self._compute_interference_watts(gnb, ue)

        snr_db = rx_dbm - noise_dbm
        sinr_lin = sig_w / max(noise_w + interf_w, 1e-15)
        sinr_db = 10.0 * np.log10(max(sinr_lin, 1e-15))

        return {
            "rx_power_dbm": float(rx_dbm),
            "noise_dbm": float(noise_dbm),
            "interference_dbm": float(self._watts_to_dbm(interf_w)) if interf_w > 0 else -np.inf,
            "snr_db": float(snr_db),
            "sinr_db": float(sinr_db),
        }

    def _get_sinr_db(self, gnb, ue: UE) -> float:
        return float(self._compute_link_metrics(gnb, ue)["sinr_db"])

    def _get_snr_db(self, gnb, ue: UE) -> float:
        if gnb is None:
            return -np.inf

        for method_name in ("get_ue_snr", "get_snr_db", "get_snr"):
            if hasattr(gnb, method_name):
                method = getattr(gnb, method_name)
                try:
                    return float(method(ue.x, ue.y))
                except TypeError:
                    try:
                        return float(method(ue))
                    except Exception:
                        pass
                except Exception:
                    pass

        rx = self._get_rx_power_dbm(gnb, ue)
        if not np.isfinite(rx):
            return -np.inf
        noise = self._get_noise_power_dbm(gnb)
        return float(rx - noise)

    def _estimate_gnb_load(self, gnb_id: int) -> float:
        gnb = self._get_gnb_by_id(gnb_id)
        if gnb is None:
            return 1.0

        for method_name in ("get_load", "load"):
            if hasattr(gnb, method_name):
                attr = getattr(gnb, method_name)
                try:
                    return float(attr() if callable(attr) else attr)
                except Exception:
                    pass

        total_prbs = max(int(getattr(gnb, "n_prbs", 1)), 1)
        used_prbs = 0
        for ue in self._ues.values():
            if ue.serving_gnb == gnb_id and ue.connected:
                used_prbs += int(max(ue.prbs, 0))

        return self._clip01(used_prbs / total_prbs)

    def _estimate_bits_for_ue(self, ue: UE, sinr_db: float, prbs: int, gnb) -> int:
        sinr_linear = max(10.0 ** (sinr_db / 10.0), 1e-6)
        rb_bw = 180e3
        spectral_eff = math.log2(1.0 + sinr_linear)
        spectral_eff = min(max(spectral_eff, 0.0), 8.0)

        ue.spectral_efficiency = float(spectral_eff)

        bits = prbs * rb_bw * self.step_dt * spectral_eff
        return max(int(bits), 0)

    def _normalize_sinr_db(self, sinr_db: float) -> float:
        if not np.isfinite(sinr_db):
            return 0.0
        return self._clip01((sinr_db + 10.0) / 40.0)

    def _slice_sla_bonus(
        self,
        ue: UE,
        th_norm: float,
        w_norm: float,
        snr_norm: float,
        load_norm: float,
    ) -> float:
        slice_type = (ue.slice_type or "eMBB").upper()

        if slice_type == "EMBB":
            return self._clip01(0.7 * th_norm + 0.3 * (1.0 - load_norm))
        if slice_type == "URLLC":
            return self._clip01(0.6 * (1.0 - w_norm) + 0.4 * snr_norm)
        if slice_type == "MMTC":
            return self._clip01(1.0 - load_norm)

        return self._clip01(0.5 * th_norm + 0.5 * (1.0 - load_norm))

    def _is_ping_pong(self, ue: UE) -> bool:
        last_serv = self._last_serving_gnb.get(ue.id)
        prev_serv = self._prev_serving_gnb.get(ue.id)
        if last_serv is None or prev_serv is None:
            return False
        return ue.serving_gnb == prev_serv and ue.serving_gnb != last_serv

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _snr_fill_value() -> float:
        return -20.0

    def _get_speed(self, ue):
        return float(np.sqrt(ue.vx ** 2 + ue.vy ** 2))

    def _get_distance(self, gnb, ue):
        if gnb is None:
            return np.inf
        if hasattr(gnb, "distance_to_ue"):
            return float(gnb.distance_to_ue(ue.x, ue.y))
        return float(np.sqrt((gnb.x - ue.x) ** 2 + (gnb.y - ue.y) ** 2))

    def _get_approach_score(self, gnb, ue, eps=1e-9):
        if gnb is None:
            return 0.0

        dx = float(gnb.x - ue.x)
        dy = float(gnb.y - ue.y)

        vx = float(ue.vx)
        vy = float(ue.vy)

        v_norm = np.sqrt(vx * vx + vy * vy)
        d_norm = np.sqrt(dx * dx + dy * dy)

        if v_norm < eps or d_norm < eps:
            return 0.0

        score = (vx * dx + vy * dy) / (v_norm * d_norm + eps)
        return float(np.clip(score, -1.0, 1.0))