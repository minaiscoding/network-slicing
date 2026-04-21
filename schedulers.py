#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2021

@author: juanjosealcaraz
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(self, mcs_codeset, granularity = 2, slot_length = 1e-3, window = 50, sym_per_prb = 158, reception_prob_boost = 1.0):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1/window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length
        # Pre-allocated scratch buffers — grown only when UE count exceeds capacity
        self._buf_size = 0
        self._ue_rbs   = None
        self._ue_mcs   = None
        self._ue_queue = None
        self._ue_rate  = None
        self._ue_bits  = None
        self._ue_th    = None

    def _ensure_buffers(self, n):
        if n > self._buf_size:
            self._buf_size = n + 16  # pad to avoid frequent reallocs
            self._ue_rbs   = np.zeros(self._buf_size, dtype=int)
            self._ue_mcs   = np.zeros(self._buf_size, dtype=int)
            self._ue_queue = np.zeros(self._buf_size, dtype=int)
            self._ue_rate  = np.zeros(self._buf_size, dtype=int)
            self._ue_bits  = np.zeros(self._buf_size, dtype=int)
            self._ue_th    = np.zeros(self._buf_size)
        else:
            self._ue_rbs[:n]   = 0
            self._ue_mcs[:n]   = 0
            self._ue_queue[:n] = 0
            self._ue_rate[:n]  = 0
            self._ue_bits[:n]  = 0
            self._ue_th[:n]    = 0

    def allocate(self, ues, n_prb, error_bound = 0.1):
        '''
        Updates the following variables of the ues:
        - ue.bits : assigned bits in this subframe
        - ue.prbs : assigned prbs in this subframe
        - ue.p : reception probability
        '''
        n_ues = len(ues)
        self._ensure_buffers(n_ues)
        ue_rbs   = self._ue_rbs
        ue_mcs   = self._ue_mcs
        ue_queue = self._ue_queue
        ue_rate  = self._ue_rate
        ue_bits  = self._ue_bits
        ue_th    = self._ue_th

        # extract ue information
        for i, ue in enumerate(ues):
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
            # achievable rate for the ue
            ue_rate[i] = self.sym_per_prb * bits_per_sym
        
        # views over the active portion only — avoids slicing inside the hot loop
        rbs_v   = ue_rbs[:n_ues]
        queue_v = ue_queue[:n_ues]
        rate_v  = ue_rate[:n_ues]
        bits_v  = ue_bits[:n_ues]
        th_v    = ue_th[:n_ues]

        # loop over the resources
        for r in range(0, n_prb, self.granularity):
            prbs = min(n_prb - r, self.granularity)

            index = np.argmax(rate_v * (queue_v > 0) / th_v)

            rbs_v[index] += prbs
            tx_bits = min(prbs * rate_v[index], queue_v[index])
            queue_v[index] -= tx_bits
            bits_v[index]  += tx_bits
            th_v[index] = self.a * th_v[index] + self.b * bits_v[index] / self.slot_length

        # update ues
        prb_i = 0
        for i, ue in enumerate(ues):
            prbs = rbs_v[i]
            ue.prbs = prbs
            ue.bits = int(bits_v[i])
            if prbs:
                snr_values = ue.snr[prb_i: prb_i + prbs]
                ue.p = self.mcs_codeset.response(ue_mcs[i], snr_values)
            else:
                ue.p = 0
            prb_i += prbs
