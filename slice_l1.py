#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

SliceL1mMTC
SliceL1eMBB

"""
import numpy as np

CBR = 0
VBR = 1
DEBUG = False

class SliceL1mMTC:
    ''' 
    Layer 1 functionality for mMTC slices. It can multiplex several mMTC RAN slices .
    '''
    def __init__(self, n_prbs, slices_ran):
        self.type = 'mMTC'
        self.n_prbs = n_prbs
        self.slices_ran = slices_ran
        self.n_slices_ran = len(self.slices_ran)
        self.reset()

    def reset(self):
        self.ue_ids = np.array([], dtype = np.int64)
        self.repetitions = np.array([], dtype = np.int16)
        self.t_start = np.array([], dtype = np.int64)
        self.slice_ran_ids = np.array([], dtype = np.int16)
        self.time = 0
        self.n_users = 0
        self._prbs_used_accum = 0
        self._prbs_alloc_accum = 0
        self._arrival_count = 0
        self._n_slots = 0
        for slice_ran in self.slices_ran:
            slice_ran.reset()
        self.reset_info()

    def get_n_variables(self):
        n_variables = 0
        for slice_ran in self.slices_ran:
            n_variables += slice_ran.get_n_variables()
        return n_variables

    def reset_info(self):
        self._prbs_used_accum = 0
        self._prbs_alloc_accum = 0
        self._arrival_count = 0
        self._n_slots = 0
        for slice_ran in self.slices_ran:
            slice_ran.reset_info()

    def set_prbs(self, i_prb, n_prbs):
        self.n_prbs = n_prbs

    def get_state(self):
        state = np.array([], dtype = np.float32)
        for slice_ran in self.slices_ran:
            state = np.concatenate((state, slice_ran.get_state()), axis=0) 
        return state

    def get_info(self):
        info = {i: slice_ran.info for i, slice_ran in enumerate(self.slices_ran)}
        # mMTC L1 metrics: device count is the "utilization" proxy
        info['_l1_metrics'] = {
            'prbs_used':       self._prbs_used_accum,
            'prbs_allocated':  self._prbs_alloc_accum,
            'capacity_bits':   0,
            'snr_mean':        0.0,
            'snr_var':         0.0,
            'arrival_count':   self._arrival_count,
            'n_slots':         max(self._n_slots, 1),
        }
        return info

    def compute_reward(self):
        '''
        reward = 1 if SLA fullfiled , -1 if SLA *NOT* filfilled
        reward --> signal (label) for online learning algorithms 
        '''
        violations = 0
        for slice_ran in self.slices_ran:
            violations += slice_ran.compute_reward()
        reward = -1*min(violations, 1)
        if violations == 0:
            reward = 1
        return reward, violations

    def add_users(self, ue_list):
        for ue in ue_list:
            self.n_users += 1
            self.ue_ids = np.append(self.ue_ids, ue.id)
            self.repetitions = np.append(self.repetitions, ue.repetitions)
            self.t_start = np.append(self.t_start, self.time)
            self.slice_ran_ids = np.append(self.slice_ran_ids, ue.slice_ran_id)
    
    def extract_users(self, ue_id_list):
        pass

    def slot(self):
        self.time += 1
        self._n_slots += 1

        # generate arrivals and departures for each slice ran
        for slice_ran in self.slices_ran:
            arrivals, departures = slice_ran.slot()
            self.extract_users(departures)
            self._arrival_count += len(arrivals)
            self.add_users(arrivals)

        n_carriers = self.n_prbs # in NB-IoT one carrier is 1 PRB
        n_tx = min(n_carriers, self.n_users)

        # Track PRB utilization: carriers actually used vs allocated
        self._prbs_used_accum += n_tx
        self._prbs_alloc_accum += self.n_prbs

        # each device transmits one transport block in the whole carrier
        self.repetitions[:n_tx] -= 1

        # remove devices who have finished their transmission
        remain = self.repetitions > 0
        self.ue_ids = self.ue_ids[remain]
        self.repetitions = self.repetitions[remain]
        self.t_start = self.t_start[remain]
        self.slice_ran_ids = self.slice_ran_ids[remain]

        # check delays
        delays = np.maximum(0, self.time - self.t_start)

        # update number of active devices
        self.n_users = len(self.ue_ids)

        # create an info summary for SLA assessment
        for slice_ran in self.slices_ran:
            slice_indexes = self.slice_ran_ids == slice_ran.id
            users_in_slice = len(self.ue_ids[slice_indexes])
            delay = 0
            avg_rep = 0
            devices = users_in_slice
            if users_in_slice > 0:
                delay = delays[slice_indexes].mean() # max() can be used
                avg_rep = np.rint(self.repetitions[slice_indexes].mean())
            slice_ran.update_info(delay, avg_rep, devices)

class SliceL1eMBB:
    ''' 
    Layer 1 functionality for eMBB slices. It can multiplex several eMBB slices.
    '''
    def __init__(self, rng, snr_generator, n_prbs, slices_ran, scheduler):
        self.type = 'eMBB'
        self.rng = rng
        self.snr_generator = snr_generator
        self.n_prbs = n_prbs
        self.prb_slice = slice(0,n_prbs)
        self.slices_ran = slices_ran
        self.scheduler = scheduler
        self.reset()

    def set_prbs(self, i_prb, n_prbs):
        self.n_prbs = n_prbs
        self.prb_slice = slice(i_prb, i_prb + n_prbs)

    def reset(self):
        self.ues = []
        self._slot_metrics = self._empty_slot_metrics()
        for slice_ran in self.slices_ran:
            slice_ran.reset()

    def _empty_slot_metrics(self):
        return {
            'prbs_used': 0,        # PRBs actually assigned by scheduler
            'prbs_allocated': 0,   # PRBs allocated to this slice (budget)
            'capacity_bits': 0,    # total bits transmitted
            'snr_sum': 0.0,        # for mean CQI
            'snr_sq_sum': 0.0,     # for CQI variance
            'snr_count': 0,        # number of SNR samples
            'arrival_count': 0,    # new UE arrivals this step
            'n_slots': 0,          # slot counter
        }

    def reset_info(self):
        self._slot_metrics = self._empty_slot_metrics()
        for slice_ran in self.slices_ran:
            slice_ran.reset_info()

    def get_n_variables(self):
        n_variables = 0
        for slice_ran in self.slices_ran:
            n_variables += slice_ran.get_n_variables()
        return n_variables

    def compute_reward(self):
        '''
        reward = 1 if SLA fullfiled , -1 if SLA *NOT* filfilled
        reward --> signal for online learning algorithms 
        '''
        violations = 0
        for slice_ran in self.slices_ran:
            violations += slice_ran.compute_reward()
        reward = -1*min(violations, 1)
        if violations == 0:
            reward = 1
        return reward, violations

    def get_state(self):
        state = np.array([], dtype = np.float32)
        for slice_ran in self.slices_ran:
            state = np.concatenate((state, slice_ran.get_state()), axis=0) 
        return state

    def get_info(self):
        info = {i: slice_ran.info for i, slice_ran in enumerate(self.slices_ran)}
        return info

    def get_info(self):
        info = {i: slice_ran.info for i, slice_ran in enumerate(self.slices_ran)}
        # Attach accumulated L1 slot-level metrics
        sm = self._slot_metrics
        n = max(sm['n_slots'], 1)
        snr_mean = sm['snr_sum'] / max(sm['snr_count'], 1)
        snr_var  = max(0.0, sm['snr_sq_sum'] / max(sm['snr_count'], 1) - snr_mean ** 2)
        info['_l1_metrics'] = {
            'prbs_used':       sm['prbs_used'],
            'prbs_allocated':  sm['prbs_allocated'],
            'capacity_bits':   sm['capacity_bits'],
            'snr_mean':        snr_mean,
            'snr_var':         snr_var,
            'arrival_count':   sm['arrival_count'],
            'n_slots':         n,
        }
        return info

    def add_users(self, ue_list):
        for ue in ue_list:
            self.ues.append(ue)
            fading_type = getattr(ue, 'fading_type', None)
            self.snr_generator.insert_user(ue.id, fading_type=fading_type)

    def extract_users(self, ue_id_list):
        for ue_id in ue_id_list:
            self.snr_generator.extract_user(ue_id)
        self.ues = [ue for ue in self.ues if ue.id not in ue_id_list]

    def slot(self):
        sm = self._slot_metrics
        sm['n_slots'] += 1
        sm['prbs_allocated'] += self.n_prbs

        # generate arrivals and departures for each slice ran
        for slice_ran in self.slices_ran:
            arrivals, departures = slice_ran.slot()
            sm['arrival_count'] += len(arrivals)
            self.extract_users(departures)
            self.add_users(arrivals)

        queued_data = 0
        for ue in self.ues:
            # data arrival
            ue.traffic_step()
            # update queued_data
            queued_data += ue.queue
            if self.n_prbs > 0:
                snr = self.snr_generator.get_snr(ue.id)
                try:
                    ue.estimate_snr(snr[self.prb_slice])
                except:
                    print('problem with snr estimation!')
                    print('prb_slice = {}'.format(self.prb_slice))
                    print('snr vector = {}'.format(snr[self.prb_slice]))
                # Accumulate CQI statistics
                mean_snr = float(ue.e_snr)
                sm['snr_sum'] += mean_snr
                sm['snr_sq_sum'] += mean_snr ** 2
                sm['snr_count'] += 1

        # Disconnect UEs with SNR below slice-type threshold
        snr_threshold = -2 if self.type == 'URLLC' else -4
        snr_disconnect_ids = [ue.id for ue in self.ues if ue.e_snr <= snr_threshold]
        if snr_disconnect_ids:
            for slice_ran in self.slices_ran:
                for ue_id in snr_disconnect_ids:
                    slice_ran.cbr_ues.pop(ue_id, None)
                    slice_ran.vbr_ues.pop(ue_id, None)
                    slice_ran.remaining_time.pop(ue_id, None)
            self.extract_users(snr_disconnect_ids)

        if queued_data > 0 and self.n_prbs > 0 and len(self.ues) > 0:
            # scheduling
            self.scheduler.allocate(self.ues, self.n_prbs)

            # Track PRBs actually used and capacity bits
            for ue in self.ues:
                sm['prbs_used'] += ue.prbs
                sm['capacity_bits'] += ue.bits

                # transmission and ue update
                received = False
                if ue.prbs:
                    received = self.rng.random() < ue.p
                ue.transmission_step(received)

        # update slice_ran info
        for slice_ran in self.slices_ran:
            slice_ran.update_info()

class SliceL1URLLC(SliceL1eMBB):
    '''
    Layer 1 functionality for URLLC slices.
    Inherits from SliceL1eMBB with same scheduling behavior but stricter resource allocation.
    '''
    def __init__(self, rng, snr_generator, n_prbs, slices_ran, scheduler):
        super().__init__(rng, snr_generator, n_prbs, slices_ran, scheduler)
        self.type = 'URLLC'