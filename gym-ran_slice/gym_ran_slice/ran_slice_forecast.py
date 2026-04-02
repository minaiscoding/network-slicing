#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ran_slice_forecast.py

Forecast-augmented RAN Slice environment.

Wraps the standard RanSlice environment and augments observations with a
"perfect forecast" of future per-slice PRB demand loaded from a pre-generated
trace database (produced by generate_trace_db.py).

The forecast is "perfect" because the same RNG seed is used during both
trace generation and training, so the simulated traffic + channel conditions
exactly match the recorded trace.

Observation layout (for H=5 forecast horizon, 3 slices):
  [ base_obs (n_variables) | prb_demand_t+1_s0..s2 | ... | prb_demand_t+H_s0..s2 ]

The forecast portion is normalized to [0, 1] by dividing by n_prbs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class RanSliceForecast(gym.Env):
    """
    RAN Slice environment with perfect PRB demand forecast.

    Parameters
    ----------
    node_b : NodeB
        The NodeB simulator instance.
    trace_path : str
        Path to the .npz trace database file.
    forecast_horizon : int
        Number of future steps to include in the observation (default: 5).
    penalty : int
        SLA violation penalty (passed through to reward).
    reward_scale : float
        Reward normalization factor.
    """

    def __init__(self, node_b=None, trace_path=None, forecast_horizon=5,
                 penalty=10, reward_scale=50.0):
        super().__init__()

        self.node_b = node_b
        self.penalty = penalty
        self.reward_scale = reward_scale
        self.forecast_horizon = forecast_horizon

        self.n_prbs = node_b.n_prbs
        self.n_slices = node_b.n_slices_l1
        self.n_variables = node_b.get_n_variables()

        # Load the trace database
        self._load_trace(trace_path)

        # Forecast adds n_slices * forecast_horizon floats to the observation
        self._forecast_size = self.n_slices * self.forecast_horizon
        self._total_obs_size = self.n_variables + self._forecast_size

        self.action_space = spaces.Box(
            low=0, high=self.n_prbs,
            shape=(self.n_slices,), dtype=int
        )
        self.observation_space = spaces.Box(
            low=-float('inf'), high=+float('inf'),
            shape=(self._total_obs_size,), dtype=float
        )

        self._step_idx = 0

    def _load_trace(self, trace_path):
        """Load trace DB and extract PRB demand array."""
        if trace_path is None or not os.path.exists(trace_path):
            raise FileNotFoundError(
                f"Trace database not found at {trace_path}. "
                f"Run generate_trace_db.py first."
            )
        data = np.load(trace_path, allow_pickle=True)
        self._prb_demand = data['prb_demand']              # (n_steps, n_slices)
        self._trace_n_steps = int(data['n_steps'])
        self._trace_n_slices = int(data['n_slices'])
        self._trace_traffic = data['traffic_arrivals']      # (n_steps, n_slices)
        self._trace_snr = data['avg_snr']                   # (n_steps, n_slices)
        self._trace_queue = data['queue_depth']             # (n_steps, n_slices)
        print(f"[Forecast] Loaded trace: {trace_path} "
              f"({self._trace_n_steps} steps, {self._trace_n_slices} slices)")

    def _get_forecast(self):
        """
        Return the forecast vector for the current step.

        Shape: (n_slices * forecast_horizon,)
        Contains PRB demand for steps [t+1, t+2, ..., t+H], normalized by n_prbs.
        If near end of trace, pad with the last available value.
        """
        forecast = np.zeros(self._forecast_size, dtype=np.float32)
        for h in range(self.forecast_horizon):
            future_idx = self._step_idx + 1 + h
            if future_idx < self._trace_n_steps:
                demand = self._prb_demand[future_idx]
            else:
                # Pad with last available step
                demand = self._prb_demand[-1]
            # Normalize by n_prbs and clip to [0, 1]
            normalized = np.clip(demand / max(self.n_prbs, 1), 0.0, 1.0)
            start = h * self.n_slices
            forecast[start:start + self.n_slices] = normalized
        return forecast

    def reset(self, *, seed=None, options=None):
        """Reset environment and trace index."""
        if seed is not None:
            np.random.seed(seed)

        state = self.node_b.reset()
        self._step_idx = 0

        # Augment with forecast
        forecast = self._get_forecast()
        obs = np.concatenate([state, forecast])

        return obs, {}

    def step(self, action):
        """Execute one step with forecast-augmented observation."""
        state, info = self.node_b.step(action)
        total_violations = info['violations'].sum()
        info['total_violations'] = total_violations

        if total_violations > 0:
            reward = 0
        else:
            reward = max(0, self.node_b.n_prbs - action.sum())

        normalized_reward = reward / self.reward_scale

        # Advance trace index
        self._step_idx += 1

        # Augment observation with forecast
        forecast = self._get_forecast()
        obs = np.concatenate([state, forecast])

        # Also store current step's trace data in info for debugging
        if self._step_idx < self._trace_n_steps:
            info['trace_prb_demand'] = self._prb_demand[self._step_idx].copy()
            info['trace_traffic'] = self._trace_traffic[self._step_idx].copy()
            info['trace_snr'] = self._trace_snr[self._step_idx].copy()

        terminated = False
        truncated = False

        return obs, float(normalized_reward), terminated, truncated, info

    def set_trace_offset(self, offset):
        """
        Set the trace index to a specific offset.
        Useful when switching scenarios in curriculum learning.
        """
        self._step_idx = offset % self._trace_n_steps

    def render(self):
        pass
