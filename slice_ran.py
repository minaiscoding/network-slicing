#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

UE
SliceRANmMTC
SliceRANeMBB

"""
DEBUG = False  # Set to True to see HARQ and latency debug prints
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource

class UE:
    '''
    eMBB UE contains a traffic source that can be CRB (GBR) or VBR (non-GBR)
    '''
    def __init__(self, id, slice_ran_id, traffic_source, type, window = 50, slot_length = 1e-3):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.traffic_source = traffic_source
        self.type = type
        self.th = 0
        self.b = 1/window
        self.a = 1 - self.b
        self.queue = 0
        self.slot_length = slot_length

        # per subframe variables
        self.snr = 0 # real error values per prb
        self.e_snr = 0 # estimated error
        self.new_bits = 0 # incoming bits
        self.bits = 0 # assigned bits
        self.prbs = 0 # assigned prbs
        self.p = 0 # reception probability
        
        # Packet timestamp tracking
        self.current_slot = 0  # current slot number for timestamping
        self.packet_queue = []  # list of (bits, arrival_slot, retry_count) tuples
        self.step_packet_latencies = []  # packet latencies during current observation step

        # HARQ and packet loss tracking
        self.max_harq_retries = 3  # Maximum HARQ retransmissions before dropping
        self.tx_attempts = 0       # Physical layer transmission attempts
        self.tx_failures = 0        # Physical layer transmission failures (HARQ retransmissions)
        self.packets_arrived = 0    # Total packets that arrived
        self.packets_transmitted = 0 # Packets successfully transmitted
        self.pkt_drops = 0          # Packets dropped after max HARQ retries
    
    def estimate_snr(self, snr):
        self.snr = snr
        self.e_snr = round(np.mean(snr))

    def traffic_step(self):
        self.new_bits = self.traffic_source.step()
        if self.new_bits > 0:
            # Create a packet with timestamp and zero retries
            self.packet_queue.append((self.new_bits, self.current_slot, 0))
            self.packets_arrived += 1
        self.queue = sum(bits for bits, _, _ in self.packet_queue)
        self.current_slot += 1
    
    def transmission_step(self, received):
        # Track HARQ transmission attempts
        if self.prbs > 0:
            self.tx_attempts += 1
            if not received:
                self.tx_failures += 1
                
                # HARQ: Retry or drop the first packet in queue
                if len(self.packet_queue) > 0:
                    packet_bits, arrival_slot, retry_count = self.packet_queue[0]
                    
                    if retry_count < self.max_harq_retries:
                        # Retry: increment retry count for first packet
                        self.packet_queue[0] = (packet_bits, arrival_slot, retry_count + 1)
                        if DEBUG and retry_count == 0:
                            print(f"[HARQ] UE {self.id}: Retrying packet (retry {retry_count + 1}/{self.max_harq_retries})")
                    else:
                        # Max retries exceeded: drop the packet to avoid latency buildup
                        self.packet_queue.pop(0)
                        self.pkt_drops += 1
                        if DEBUG:
                            print(f"[HARQ] UE {self.id}: Packet DROPPED after {self.max_harq_retries} retries (latency would be {(self.current_slot - arrival_slot) * self.slot_length * 1000:.2f} ms)")
                
                self.bits = 0  # No bits successfully transmitted this slot
                self.queue = sum(bits for bits, _, _ in self.packet_queue)
                self.th = self.a * self.th + self.b * self.bits / self.slot_length
                return
        
        # Successful transmission: process packets in FIFO order
        bits_to_send = self.bits
        while bits_to_send > 0 and len(self.packet_queue) > 0:
            packet_bits, arrival_slot, retry_count = self.packet_queue[0]
            
            if packet_bits <= bits_to_send:
                # Entire packet transmitted successfully
                latency_slots = self.current_slot - arrival_slot
                latency_ms = latency_slots * self.slot_length * 1000
                self.step_packet_latencies.append(latency_ms)
                self.packets_transmitted += 1
                bits_to_send -= packet_bits
                self.packet_queue.pop(0)
            else:
                # Partial packet transmitted (keep remainder, reset retry count)
                latency_slots = self.current_slot - arrival_slot
                latency_ms = latency_slots * self.slot_length * 1000
                self.step_packet_latencies.append(latency_ms)
                # Remainder continues with same arrival time but reset retries
                self.packet_queue[0] = (packet_bits - bits_to_send, arrival_slot, 0)
                bits_to_send = 0
        
        self.queue = sum(bits for bits, _, _ in self.packet_queue)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length

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
    Generates message arrivals at the mMTC devices
    according to the characteristics defined in MTC_description:
    - n_devices: total number of devices
    - repetition_set: possible repetitions
    - period_set: possible times between message arrivals
    '''
    def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA
        self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
        self.norm_const = norm_const # 100 all
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
            self.t_to_arrival[i] = 1 + self.rng.choice(np.arange(self.period[i]))
            self.devices.append(MTCdevice(i, repetitions, self.id))

    def slot(self):
        self.slot_counter += 1

        # advance time
        self.t_to_arrival -= 1

        # arrivals
        arrival_list = []
        arrivals = self.t_to_arrival == 0
        indices = np.where(arrivals)

        # print('indices = {}'.format(indices))
        for i in indices[0]:
            arrival_list.append(self.devices[i])

        # prepare for next arrival (deterministic inter arrival time)
        self.t_to_arrival[arrivals] = self.period[arrivals]

        return arrival_list, []

    def reset_info(self):
        self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = float)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        '''convert the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state

    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices
        

    def compute_reward(self):
        '''assesses SLA violations'''
        SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay']
        return not(SLA_fulfilled)

class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB ues.
    There are two traffic types: CRB (GBR) and VBR (non-GBR)
    CBR traffic parameters are given in CBR_description
    VBR traffic parameters are given in VBR_description
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length
        self.SLA = SLA # service level agreement description
        self.state_variables = state_variables
        self.norm_const = norm_const

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate']

        self.vbr_arrival_rate = VBR_description['lambda']
        self.vbr_mean_time = VBR_description['t_mean']
        self.vbr_source_data = {
            'packet_size': VBR_description['p_size'],
            'burst_size': VBR_description['b_size'],
            'burst_rate':VBR_description['b_rate']
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
        '''Admission control for CBR users'''
        slots = max(self.slot_counter,1)
        time = slots * self.slot_length
        cbr_th = self.info['cbr_th'] / time
        if cbr_th >= self.SLA['cbr_th']:
            return False
        return True

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0:
            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.cbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.cbr_steps_next_arrival = inter_arrival_time

            if self.cbr_cac(): # check admission control
                # generate new user
                ue_id = next(self.user_counter)
                cbr_source = CbrSource(bit_rate = self.cbr_bit_rate)
                ue = UE(ue_id, self.id, cbr_source, CBR)
                self.cbr_ues[ue_id] = ue

                # generate holding time
                holding_time = self.rng.exponential(self.cbr_mean_time)
                holding_time = np.rint(holding_time / self.slot_length)
                self.remaining_time[ue_id] = holding_time

                return [ue] # return user
        else:
            self.cbr_steps_next_arrival -= 1    
        return []

    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            # create new vbr user
            ue_id = next(self.user_counter)
            vbr_source = VbrSource(**self.vbr_source_data)
            ue = UE(ue_id, self.id, vbr_source, VBR)
            self.vbr_ues[ue_id] = ue

            # generate holding time
            holding_time = self.rng.exponential(self.vbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[ue_id] = holding_time

            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.vbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
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
                del self.remaining_time[id] # delete timer
                self.vbr_ues.pop(id, None) # delete ue if here
                self.cbr_ues.pop(id, None) # or here    
        return departures   

    def slot(self):
        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        self.info = {'cbr_traffic': 0, 'cbr_th': 0,  'cbr_latency': 0, 'cbr_snr': 0,\
                    'vbr_traffic': 0, 'vbr_th': 0,  'vbr_latency': 0, 'vbr_snr': 0}
        self.slot_counter = 0
        # Reset UE packet latency tracking for new observation period
        for ue in self.cbr_ues.values():
            ue.step_packet_latencies = []
        for ue in self.vbr_ues.values():
            ue.step_packet_latencies = []

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = float)
    
    def update_info(self):
        queue = 0
        snr = 0
        n = 0
        # Collect packet latencies from all CBR UEs this slot
        cbr_latencies = []
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th'] += ue.bits
            cbr_latencies.extend(ue.step_packet_latencies)
            ue.step_packet_latencies = []  # clear for next slot
            snr += ue.e_snr
            n += 1
        n = max(n, 1)
        # Average packet latency for this slot
        if len(cbr_latencies) > 0:
            slot_avg = sum(cbr_latencies) / len(cbr_latencies)
            self.info['cbr_latency'] += slot_avg
            # if DEBUG:
            #     print(f"[eMBB CBR] Slot latencies: {[f'{lat:.2f}' for lat in cbr_latencies]} ms, avg: {slot_avg:.2f} ms")
        self.info['cbr_snr'] += snr/n

        snr = 0
        n = 0
        # Collect packet latencies from all VBR UEs this slot
        vbr_latencies = []
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th'] += ue.bits
            vbr_latencies.extend(ue.step_packet_latencies)
            ue.step_packet_latencies = []  # clear for next slot
            snr += ue.e_snr
            n += 1
        n = max(n, 1)
        # Average packet latency for this slot
        if len(vbr_latencies) > 0:
            slot_avg = sum(vbr_latencies) / len(vbr_latencies)
            self.info['vbr_latency'] += slot_avg
            # if DEBUG:
            #     print(f"[eMBB VBR] Slot latencies: {[f'{lat:.2f}' for lat in vbr_latencies]} ms, avg: {slot_avg:.2f} ms")
        self.info['vbr_snr'] += snr/n

    def compute_reward(self):
        '''assesses SLA violations'''
        cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']
        cbr_latency = self.info['cbr_latency']/self.slots_per_step < self.SLA['cbr_latency']
        vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
        vbr_latency = self.info['vbr_latency']/self.slots_per_step < self.SLA['vbr_latency']
        print('Latency = {}'.format(vbr_latency))
        # the slice has to guarantee the objective latency for cbr and vbr if their traffics do not surpass the maximum
        cbr_fulfilled = cbr_th or cbr_latency 
        vbr_fulfilled = vbr_th or vbr_latency
        SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
        return not(SLA_fulfilled)

    def get_state(self):
        '''converts the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state


class SliceRANURLC(SliceRANeMBB):
    '''
    Ultra-Reliable Low-Latency Communications (URLLC) slice.
    Extends eMBB with tighter SLAs and different traffic parameters.
    Uses MAX latency per UE instead of average for stricter QoS.
    Tracks packet loss rate per UE for reliability SLA.
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
        super().__init__(rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length)
        self.type = 'URLLC'

    def reset_info(self):
        '''Override to include packet loss fields and reset packet latencies'''
        super().reset_info()
        self.info['cbr_pkt_loss'] = 0.0
        self.info['vbr_pkt_loss'] = 0.0
        # Reset UE packet loss and latency counters for new observation period
        for ue in self.cbr_ues.values():
            ue.tx_attempts = 0
            ue.tx_failures = 0
            ue.packets_arrived = 0
            ue.packets_transmitted = 0
            ue.pkt_drops = 0
            ue.step_packet_latencies = []
        for ue in self.vbr_ues.values():
            ue.tx_attempts = 0
            ue.tx_failures = 0
            ue.packets_arrived = 0
            ue.packets_transmitted = 0
            ue.pkt_drops = 0
            ue.step_packet_latencies = []

    def update_info(self):
        '''Override to use MAX packet latency and actual packet loss (drops) for URLLC'''
        snr = 0
        n = 0
        # Collect packet latencies from all CBR UEs this slot
        cbr_latencies = []
        max_pkt_loss = 0.0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th'] += ue.bits
            cbr_latencies.extend(ue.step_packet_latencies)
            ue.step_packet_latencies = []  # clear for next slot
            # Calculate actual packet loss: dropped packets / total arrived packets
            if ue.packets_arrived > 0:
                ue_loss = ue.pkt_drops / ue.packets_arrived
                max_pkt_loss = max(max_pkt_loss, ue_loss)
            snr += ue.e_snr
            n += 1
        n = max(n, 1)
        # Track max packet latency seen so far
        if len(cbr_latencies) > 0:
            slot_max_latency = max(cbr_latencies)
            self.info['cbr_latency'] = max(self.info['cbr_latency'], slot_max_latency)
            if DEBUG:
                print(f"[URLLC CBR] Slot latencies: {[f'{lat:.2f}' for lat in cbr_latencies]} ms, max: {slot_max_latency:.2f} ms, overall max: {self.info['cbr_latency']:.2f} ms")
        self.info['cbr_pkt_loss'] = max(self.info.get('cbr_pkt_loss', 0.0), max_pkt_loss)
        self.info['cbr_snr'] += snr/n

        max_pkt_loss = 0.0
        snr = 0
        n = 0
        # Collect packet latencies from all VBR UEs this slot
        vbr_latencies = []
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th'] += ue.bits
            vbr_latencies.extend(ue.step_packet_latencies)
            ue.step_packet_latencies = []  # clear for next slot
            # Calculate actual packet loss: dropped packets / total arrived packets
            if ue.packets_arrived > 0:
                ue_loss = ue.pkt_drops / ue.packets_arrived
                max_pkt_loss = max(max_pkt_loss, ue_loss)
            snr += ue.e_snr
            n += 1
        n = max(n, 1)
        # Track max packet latency seen so far
        if len(vbr_latencies) > 0:
            slot_max_latency = max(vbr_latencies)
            self.info['vbr_latency'] = max(self.info['vbr_latency'], slot_max_latency)
            if DEBUG:
                print(f"[URLLC VBR] Slot latencies: {[f'{lat:.2f}' for lat in vbr_latencies]} ms, max: {slot_max_latency:.2f} ms, overall max: {self.info['vbr_latency']:.2f} ms")
        self.info['vbr_pkt_loss'] = max(self.info.get('vbr_pkt_loss', 0.0), max_pkt_loss)
        self.info['vbr_snr'] += snr/n

    def compute_reward(self):
        '''Override to include packet loss in SLA assessment'''
        cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']
        # For URLLC max latency, don't divide by slots_per_step (it's already the max)
        cbr_latency = self.info['cbr_latency'] < self.SLA['cbr_latency']
        cbr_pkt_loss = self.info.get('cbr_pkt_loss', 0.0) < self.SLA.get('cbr_pkt_loss', 1.0)
        vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
        vbr_latency = self.info['vbr_latency'] < self.SLA['vbr_latency']
        vbr_pkt_loss = self.info.get('vbr_pkt_loss', 0.0) < self.SLA.get('vbr_pkt_loss', 1.0)
        # URLLC must satisfy latency AND packet loss constraints
        cbr_fulfilled = cbr_th  or (cbr_latency and cbr_pkt_loss)
        vbr_fulfilled = vbr_th or (vbr_latency and vbr_pkt_loss)
        SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
        return not(SLA_fulfilled)

