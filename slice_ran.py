#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

UE
SliceRANmMTC
SliceRANeMBB
SliceRANURLC

Delay semantics:
- eMBB  : info['cbr_delay'] = sum of per-slot average HOL delay across UEs.
           Divide by slots_per_step in compute_reward() to get time-averaged
           mean HOL delay over the step.
- URLLC : info['cbr_delay'] = maximum HOL delay reached by any UE at any
           slot during the step. Compared directly (no division) against
           SLA['cbr_delay'] in compute_reward().
"""

DEBUG = True
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource


class UE:
    '''
    eMBB/URLLC UE with a CBR or VBR traffic source.
    Tracks HOL delay, HARQ retransmissions, and packet-level BLER.
    '''
    def __init__(self, id, slice_ran_id, traffic_source, type,
                 window=50, slot_length=1e-3, harq_max_retransmissions=2):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.traffic_source = traffic_source
        self.type = type
        self.th = 0
        self.b = 1 / window
        self.a = 1 - self.b
        self.queue = 0
        self.slot_length = slot_length

        # HARQ
        self.harq_max_retransmissions = harq_max_retransmissions

        # packet queue: list of (bits, arrival_slot, harq_retrans_count)
        self.packet_queue = []
        self.current_time = 0    # never reset — needed for correct HOL delay

        # HARQ counters — reset each step
        self.harq_dropped_packets = 0
        self.harq_total_retransmissions = 0

        # packet-level BLER counters — reset each step
        self.packets_delivered = 0
        self.packets_dropped = 0

        # per subframe variables
        self.snr = 0
        self.e_snr = 0
        self.new_bits = 0
        self.bits = 0
        self.prbs = 0
        self.p = 0

    def estimate_snr(self, snr):
        self.snr = snr
        self.e_snr = round(np.mean(snr))

    def traffic_step(self):
        self.new_bits = self.traffic_source.step()
        self.queue += self.new_bits
        if self.new_bits > 0:
            self.packet_queue.append((self.new_bits, self.current_time, 0))

    def transmission_step(self, received):
        if not received:
            if self.packet_queue:
                pkt_bits, arrival_slot, harq_retrans = self.packet_queue[0]
                harq_retrans += 1
                self.harq_total_retransmissions += 1

                if harq_retrans > self.harq_max_retransmissions:
                    self.harq_dropped_packets += 1
                    self.packets_dropped += 1
                    self.packet_queue.pop(0)
                    self.queue = max(self.queue - pkt_bits, 0)
                else:
                    self.packet_queue[0] = (pkt_bits, arrival_slot, harq_retrans)

            self.bits = 0

        bits_to_remove = self.bits
        while bits_to_remove > 0 and self.packet_queue:
            pkt_bits, arrival_slot, harq_retrans = self.packet_queue[0]
            if pkt_bits <= bits_to_remove:
                bits_to_remove -= pkt_bits
                self.packets_delivered += 1
                self.packet_queue.pop(0)
            else:
                self.packet_queue[0] = (pkt_bits - bits_to_remove,
                                        arrival_slot, harq_retrans)
                bits_to_remove = 0

        self.queue = max(self.queue - self.bits, 0)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length

    def get_hol_delay(self):
        """
        Age of the oldest packet currently in the queue in slots.
        Returns 0 if queue is empty.
        Resets naturally when the HOL packet is delivered or dropped.
        """
        if not self.packet_queue:
            return 0
        _, arrival_slot, _ = self.packet_queue[0]
        return self.current_time - arrival_slot

    def get_bler(self):
        """
        Packet-level BLER = dropped / (dropped + delivered).
        Always in [0, 1]. Returns 0 if no packets completed this step.
        """
        total = self.packets_dropped + self.packets_delivered
        if total == 0:
            return 0.0
        return self.packets_dropped / total

    def get_harq_metrics(self):
        queue_packets_with_retrans = sum(
            1 for _, _, retrans in self.packet_queue if retrans > 0)
        return {
            'dropped_packets': self.harq_dropped_packets,
            'total_retransmissions': self.harq_total_retransmissions,
            'queue_packets_with_retrans': queue_packets_with_retrans
        }

    def reset_harq_metrics(self):
        self.harq_dropped_packets = 0
        self.harq_total_retransmissions = 0

    def reset_bler_metrics(self):
        self.packets_delivered = 0
        self.packets_dropped = 0

    def update_time(self):
        self.current_time += 1

    def __repr__(self):
        return 'UE {}'.format(self.id)


class MTCdevice:
    def __init__(self, id, repetitions, slice_ran_id):
        self.id = id
        self.repetitions = repetitions
        self.slice_ran_id = slice_ran_id

    def __repr__(self):
        return 'MTC {}'.format(self.id)


class SliceRANmMTC:
    '''
    Generates message arrivals at the mMTC devices.
    '''
    def __init__(self, rng, id, SLA, MTCdescription, state_variables,
                 norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA
        self.state_variables = state_variables
        self.norm_const = norm_const
        self.slots_per_step = slots_per_step

        self.n_devices = MTCdescription['n_devices']
        self.repetition_set = MTCdescription['repetition_set']
        self.period_set = MTCdescription['period_set']

        self.reset()

    def reset(self):
        self.reset_state()
        self.reset_info()
        self.period = np.ones((self.n_devices), dtype=int)
        self.t_to_arrival = np.zeros((self.n_devices), dtype=int)
        self.devices = []
        for i in range(self.n_devices):
            repetitions = self.rng.choice(self.repetition_set)
            self.period[i] = self.rng.choice(self.period_set)
            self.t_to_arrival[i] = 1 + self.rng.choice(
                np.arange(self.period[i]))
            self.devices.append(MTCdevice(i, repetitions, self.id))

    def slot(self):
        self.slot_counter += 1
        self.t_to_arrival -= 1
        arrival_list = []
        arrivals = self.t_to_arrival == 0
        indices = np.where(arrivals)
        for i in indices[0]:
            arrival_list.append(self.devices[i])
        self.t_to_arrival[arrivals] = self.period[arrivals]
        return arrival_list, []

    def reset_info(self):
        self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype=float)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]
        return self.state

    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices

    def compute_reward(self):
        SLA_fulfilled = (self.info['delay'] / self.slots_per_step
                         < self.SLA['delay'])
        return not SLA_fulfilled


class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB UEs.
    Two traffic types: CBR (GBR) and VBR (non-GBR).

    Delay metric: time-averaged mean HOL delay over the step.
        info['cbr_delay'] = sum of per-slot average HOL across UEs
        compute_reward() divides by slots_per_step to get the mean.
    '''
    def __init__(self, rng, user_counter, id, SLA,
                 CBR_description, VBR_description,
                 state_variables, norm_const, slots_per_step,
                 slot_length=1e-3, harq_max_retransmissions=9):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length
        self.SLA = SLA
        self.state_variables = state_variables
        self.norm_const = norm_const
        self.harq_max_retransmissions = harq_max_retransmissions

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate']

        self.vbr_arrival_rate = VBR_description['lambda']
        self.vbr_mean_time = VBR_description['t_mean']
        self.vbr_source_data = {
            'packet_size': VBR_description['p_size'],
            'burst_size':  VBR_description['b_size'],
            'burst_rate':  VBR_description['b_rate']
        }
        self.reset()

    def reset(self):
        self.slot_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.vbr_steps_next_arrival = 0
        self.vbr_ues = {}
        self.cbr_ues = {}
        self.reset_state()
        self.reset_info()

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        slots = max(self.slot_counter, 1)
        time = slots * self.slot_length
        cbr_th = self.info['cbr_th'] / time
        if cbr_th >= self.SLA['cbr_th']:
            return False
        return True

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0:
            inter_arrival_time = self.rng.exponential(
                1.0 / self.cbr_arrival_rate)
            inter_arrival_time = np.rint(
                inter_arrival_time / self.slot_length)
            self.cbr_steps_next_arrival = inter_arrival_time
            if self.cbr_cac():
                ue_id = next(self.user_counter)
                cbr_source = CbrSource(bit_rate=self.cbr_bit_rate)
                ue = UE(ue_id, self.id, cbr_source, CBR,
                        harq_max_retransmissions=self.harq_max_retransmissions)
                self.cbr_ues[ue_id] = ue
                holding_time = self.rng.exponential(self.cbr_mean_time)
                holding_time = np.rint(holding_time / self.slot_length)
                self.remaining_time[ue_id] = holding_time
                return [ue]
        else:
            self.cbr_steps_next_arrival -= 1
        return []

    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            ue_id = next(self.user_counter)
            vbr_source = VbrSource(**self.vbr_source_data)
            ue = UE(ue_id, self.id, vbr_source, VBR,
                    harq_max_retransmissions=self.harq_max_retransmissions)
            self.vbr_ues[ue_id] = ue
            holding_time = self.rng.exponential(self.vbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[ue_id] = holding_time
            inter_arrival_time = self.rng.exponential(
                1.0 / self.vbr_arrival_rate)
            inter_arrival_time = np.rint(
                inter_arrival_time / self.slot_length)
            self.vbr_steps_next_arrival = inter_arrival_time
            return [ue]
        else:
            self.vbr_steps_next_arrival -= 1
            return []

    def departures(self):
        departures = []
        current_ids = list(self.remaining_time.keys())
        for id in current_ids:
            self.remaining_time[id] -= 1
            if self.remaining_time[id] == 0:
                departures.append(id)
                del self.remaining_time[id]
                self.vbr_ues.pop(id, None)
                self.cbr_ues.pop(id, None)
        return departures

    def slot(self):
        self.slot_counter += 1
        for ue in self.cbr_ues.values():
            ue.update_time()
        for ue in self.vbr_ues.values():
            ue.update_time()
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        self.info = {
            'cbr_traffic': 0, 'cbr_th': 0,
            'cbr_queue':   0, 'cbr_snr': 0,
            'cbr_delay':   0,
            'cbr_harq_drops': 0, 'cbr_harq_retrans': 0,
            'cbr_bler':    0,
            'vbr_traffic': 0, 'vbr_th': 0,
            'vbr_queue':   0, 'vbr_snr': 0,
            'vbr_delay':   0,
            'vbr_harq_drops': 0, 'vbr_harq_retrans': 0,
            'vbr_bler':    0,
        }
        self.slot_counter = 0
        for ue in list(self.cbr_ues.values()) + list(self.vbr_ues.values()):
            ue.reset_bler_metrics()
            ue.reset_harq_metrics()

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype=float)

    def update_info(self):
        
        """
        eMBB delay: accumulates per-slot average HOL delay across UEs.
        info['cbr_delay'] / slots_per_step = time-averaged mean HOL delay.
        """
        # --- CBR ---
        queue        = 0
        snr          = 0
        delay        = 0
        harq_drops   = 0
        harq_retrans = 0
        n = 0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th']      += ue.bits
            queue        += ue.queue
            snr          += ue.e_snr
            delay        += ue.get_hol_delay()
            harq_drops   += ue.harq_dropped_packets
            harq_retrans += ue.harq_total_retransmissions
            n += 1
        n = max(n, 1)
        self.info['cbr_queue']        += queue / n
        self.info['cbr_snr']          += snr   / n
        self.info['cbr_delay']        += delay / n   # sum of per-slot avg
        self.info['cbr_harq_drops']   += harq_drops
        self.info['cbr_harq_retrans'] += harq_retrans

        # --- VBR ---
        queue        = 0
        snr          = 0
        delay        = 0
        harq_drops   = 0
        harq_retrans = 0
        n = 0
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th']      += ue.bits
            queue        += ue.queue
            snr          += ue.e_snr
            delay        += ue.get_hol_delay()
            harq_drops   += ue.harq_dropped_packets
            harq_retrans += ue.harq_total_retransmissions
            n += 1
        n = max(n, 1)
        self.info['vbr_queue']        += queue / n
        self.info['vbr_snr']          += snr   / n
        self.info['vbr_delay']        += delay / n   # sum of per-slot avg
        self.info['vbr_harq_drops']   += harq_drops
        self.info['vbr_harq_retrans'] += harq_retrans

    def update_bler(self):
        """
        BLER computed once per step after slot loop.
        Called from NodeB.step(). Always in [0, 1].
        """
        cbr_bler = 0.0
        n = 0
        for ue in self.cbr_ues.values():
            cbr_bler += ue.get_bler()
            n += 1
        self.info['cbr_bler'] = cbr_bler / max(n, 1)

        vbr_bler = 0.0
        n = 0
        for ue in self.vbr_ues.values():
            vbr_bler += ue.get_bler()
            n += 1
        self.info['vbr_bler'] = vbr_bler / max(n, 1)

    def compute_reward(self):
        """
        eMBB SLA check.
        cbr_delay: time-averaged mean HOL — divide by slots_per_step.
        """
        cbr_th    = self.info['cbr_th']    / self.observation_time > self.SLA['cbr_th']
        cbr_queue = self.info['cbr_queue'] / self.slots_per_step   < self.SLA['cbr_queue']
        cbr_delay = self.info['cbr_delay'] / self.slots_per_step   < self.SLA['cbr_delay']
        print(f"cbr_delay: {self.info['cbr_delay'] / self.slots_per_step}")
        cbr_bler  = self.info['cbr_bler']                          < self.SLA.get('cbr_bler', 1.0)

        vbr_th    = self.info['vbr_th']    / self.observation_time > self.SLA['vbr_th']
        vbr_queue = self.info['vbr_queue'] / self.slots_per_step   < self.SLA['vbr_queue']
        vbr_delay = self.info['vbr_delay'] / self.slots_per_step   < self.SLA['vbr_delay']
        vbr_bler  = self.info['vbr_bler']                          < self.SLA.get('vbr_bler', 1.0)

        cbr_fulfilled = (cbr_th or cbr_queue) and cbr_delay and cbr_bler
        vbr_fulfilled = (vbr_th or vbr_queue) and vbr_delay and vbr_bler
        return not (cbr_fulfilled and vbr_fulfilled)

    def get_state(self):
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]
        return self.state


class SliceRANURLC(SliceRANeMBB):
    '''
    URLLC slice.

    Delay metric: maximum HOL delay reached by any UE at any slot
    during the step. Compared directly against SLA['cbr_delay'] in
    compute_reward() — no division by slots_per_step.

    This reflects worst-case latency which is what URLLC SLAs care about.
    '''
    def __init__(self, rng, user_counter, id, SLA,
                 CBR_description, VBR_description,
                 state_variables, norm_const, slots_per_step,
                 slot_length=1e-3, harq_max_retransmissions=2):
        super().__init__(rng, user_counter, id, SLA,
                         CBR_description, VBR_description,
                         state_variables, norm_const,
                         slots_per_step, slot_length,
                         harq_max_retransmissions)
        self.type = 'URLLC'

    def update_info(self):
        
        """
        URLLC delay: maximum HOL delay across all UEs at this slot.
        info['cbr_delay'] = max HOL reached at any slot during the step.
        Uses = max(...) not += so it tracks the peak, not the sum.
        """
        # --- CBR ---
        max_hol      = 0
        queue        = 0
        snr          = 0
        harq_drops   = 0
        harq_retrans = 0
        n = 0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th']      += ue.bits
            queue        += ue.queue
            snr          += ue.e_snr
            max_hol       = max(max_hol, ue.get_hol_delay())
            harq_drops   += ue.harq_dropped_packets
            harq_retrans += ue.harq_total_retransmissions
            n += 1
        n = max(n, 1)
        self.info['cbr_queue']        += queue / n
        self.info['cbr_snr']          += snr   / n
        self.info['cbr_delay']         = max(self.info['cbr_delay'], max_hol)  # peak, not sum
        self.info['cbr_harq_drops']   += harq_drops
        self.info['cbr_harq_retrans'] += harq_retrans

        # --- VBR ---
        max_hol      = 0
        queue        = 0
        snr          = 0
        harq_drops   = 0
        harq_retrans = 0
        n = 0
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th']      += ue.bits
            queue        += ue.queue
            snr          += ue.e_snr
            max_hol       = max(max_hol, ue.get_hol_delay())
            harq_drops   += ue.harq_dropped_packets
            harq_retrans += ue.harq_total_retransmissions
            n += 1
        n = max(n, 1)
        self.info['vbr_queue']        += queue / n
        self.info['vbr_snr']          += snr   / n
        self.info['vbr_delay']         = max(self.info['vbr_delay'], max_hol)  # peak, not sum
        self.info['vbr_harq_drops']   += harq_drops
        self.info['vbr_harq_retrans'] += harq_retrans

    def compute_reward(self):
        """
        URLLC SLA check.
        cbr_delay: max HOL reached during the step — compare directly,
        no division by slots_per_step.
        SLA['cbr_delay'] should be set in raw slots in system_config.py
        e.g. 10 slots = 10 ms worst-case HOL budget.
        """
        cbr_th    = self.info['cbr_th']    / self.observation_time > self.SLA['cbr_th']
        cbr_queue = self.info['cbr_queue'] / self.slots_per_step   < self.SLA['cbr_queue']
        cbr_delay = self.info['cbr_delay']                         < self.SLA['cbr_delay']  # no division
        cbr_bler  = self.info['cbr_bler']                          < self.SLA.get('cbr_bler', 1.0)

        vbr_th    = self.info['vbr_th']    / self.observation_time > self.SLA['vbr_th']
        vbr_queue = self.info['vbr_queue'] / self.slots_per_step   < self.SLA['vbr_queue']
        vbr_delay = self.info['vbr_delay']                         < self.SLA['vbr_delay']  # no division
        vbr_bler  = self.info['vbr_bler']                          < self.SLA.get('vbr_bler', 1.0)

        cbr_fulfilled = (cbr_th or cbr_queue) and cbr_delay and cbr_bler
        vbr_fulfilled = (vbr_th or vbr_queue) and vbr_delay and vbr_bler
        return not (cbr_fulfilled and vbr_fulfilled)