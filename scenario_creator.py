#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Defines three public functions:

create_env              -- build an env from explicit parameters
create_env_from_config  -- build an env from a ScenarioConfig (YAML-driven)
create_kbrl_agent       -- build a KBRL agent for a given scenario

Traffic and SLA parameters can be customised by passing a traffic_config
and/or sla_config dict to create_env().  The dicts are merged with the
built-in defaults so callers only need to supply the keys they want to
override.

For YAML-driven workflows use config_loader.load_scenario() to obtain a
ScenarioConfig, then pass it to create_env_from_config():

    from config_loader import load_scenario
    from system_config import create_env_from_config

    cfg = load_scenario('scenarios.yaml', 'urllc_mix')
    env = create_env_from_config(cfg, rng)
"""

import gymnasium as gym
from itertools import count
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC, SliceL1URLLC
from slice_ran import SliceRANmMTC, SliceRANeMBB, SliceRANURLC
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset, MCS_CODESET_EMBB, MCS_CODESET_URLLC
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron
import copy

# ----------------- scenario parameters ------------------------

scenario_1 = {
    'n_prbs': 200,
    'n_embb': 5,
    'n_mmtc': 0
}

scenario_2 = {
    'n_prbs': 150,
    'n_embb': 3,
    'n_mmtc': 2
}

scenario_3 = {
    'n_prbs': 100,
    'n_embb': 1,
    'n_mmtc': 4
}

scenario_4 = {
    'n_prbs': 70,
    'n_embb': 1,
    'n_mmtc': 1
}

# URLLC scenario - can be used programmatically but not in default scenarios list
scenario_5 = {
    'n_prbs': 150,
    'n_embb': 1,
    'n_mmtc': 1,
    'n_urllc': 1
}

scenarios = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]


# -------------------- default eMBB traffic parameters -------------------------

DEFAULT_CBR_description = {
    'lambda': 2.0/60.0,
    't_mean': 30.0,
    'bit_rate': 500000,
    'packet_size': 1500  # bytes
}

DEFAULT_VBR_description = {
    'lambda': 5.0/60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1,
    'packet_size': 500  # bytes
}

DEFAULT_SLA_embb = {
    'cbr_th': 10e6,
    'cbr_queue': 10e4,
    'cbr_delay': 10,    # milliseconds

    'vbr_th': 15e6,
    'vbr_queue': 15e4,
    'vbr_delay': 100,   # 100 slots = 10 ms
}

state_variables_embb = ['cbr_traffic','cbr_th', \
                        'cbr_queue', 'cbr_snr', 'cbr_delay', \
                        'vbr_traffic', 'vbr_th', \
                        'vbr_queue', 'vbr_snr', 'vbr_delay']

# -------------------- default mMTC traffic parameters -------------------------

DEFAULT_MTC_description = {
    'n_devices': 1000,
    'repetition_set': [2,4,8,16,32,64,128],
    'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]
}

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

DEFAULT_SLA_mmtc = {
    'delay': 300
}

# -------------------- default URLLC traffic parameters -------------------------

DEFAULT_URLLC_CBR_description = {
    'lambda': 2.0/60.0,
    't_mean': 30.0,
    'bit_rate': 500000,
    'packet_size': 1500  # bytes
}

DEFAULT_URLLC_VBR_description = {
    'lambda': 5.0/60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1,
    'packet_size': 500  # bytes
}

DEFAULT_SLA_urllc = {
    'cbr_th': 10e6,
    'cbr_queue': 5e4,
    'cbr_delay': 50,    # 50 slots = 5 ms

    'vbr_th': 15e6,
    'vbr_queue': 10e4,
    'vbr_delay': 5,     # milliseconds — URLLC strict
}

state_variables_urllc = ['cbr_traffic','cbr_th', \
                        'cbr_queue', 'cbr_snr', 'cbr_delay', \
                        'vbr_traffic', 'vbr_th', \
                        'vbr_queue', 'vbr_snr', 'vbr_delay']


# -------------------- helpers ------------------------------------------

def _merge(base: dict, overrides: dict) -> dict:
    """
    Shallow-merge *overrides* into a deep copy of *base*.
    Only the keys present in *overrides* are replaced; all others
    keep their default values.
    """
    result = copy.deepcopy(base)
    result.update(overrides)
    return result


def _resolve_traffic(traffic_config: dict | None):
    """
    Return (CBR_desc, VBR_desc, MTC_desc, URLLC_CBR_desc, URLLC_VBR_desc)
    after applying any caller-supplied overrides.

    traffic_config layout (all keys optional):
    {
        'embb': {
            'cbr': { ...keys from DEFAULT_CBR_description... },
            'vbr': { ...keys from DEFAULT_VBR_description... },
        },
        'mmtc': { ...keys from DEFAULT_MTC_description... },
        'urllc': {
            'cbr': { ...keys from DEFAULT_URLLC_CBR_description... },
            'vbr': { ...keys from DEFAULT_URLLC_VBR_description... },
        },
    }
    """
    tc = traffic_config or {}
    embb  = tc.get('embb',  {})
    mmtc  = tc.get('mmtc',  {})
    urllc = tc.get('urllc', {})

    cbr_desc       = _merge(DEFAULT_CBR_description,       embb.get('cbr',  {}))
    vbr_desc       = _merge(DEFAULT_VBR_description,       embb.get('vbr',  {}))
    mtc_desc       = _merge(DEFAULT_MTC_description,       mmtc)
    urllc_cbr_desc = _merge(DEFAULT_URLLC_CBR_description, urllc.get('cbr', {}))
    urllc_vbr_desc = _merge(DEFAULT_URLLC_VBR_description, urllc.get('vbr', {}))

    return cbr_desc, vbr_desc, mtc_desc, urllc_cbr_desc, urllc_vbr_desc


def _resolve_sla(sla_config: dict | None):
    """
    Return (SLA_embb, SLA_mmtc, SLA_urllc) after applying any overrides.

    sla_config layout (all keys optional):
    {
        'embb':  { ...keys from DEFAULT_SLA_embb...  },
        'mmtc':  { ...keys from DEFAULT_SLA_mmtc...  },
        'urllc': { ...keys from DEFAULT_SLA_urllc... },
    }
    """
    sc = sla_config or {}
    sla_embb  = _merge(DEFAULT_SLA_embb,  sc.get('embb',  {}))
    sla_mmtc  = _merge(DEFAULT_SLA_mmtc,  sc.get('mmtc',  {}))
    sla_urllc = _merge(DEFAULT_SLA_urllc, sc.get('urllc', {}))
    return sla_embb, sla_mmtc, sla_urllc


# ========================= create_env =====================================

def create_env(rng, n,
               slots_per_step=50,
               propagation_type='macro_cell_urban_2GHz',
               L1_level=True,
               penalty=100,
               traffic_config: dict | None = None,
               sla_config: dict | None = None):
    '''
    Returns a slice RAN environment.

    Parameters
    ----------
    rng              : numpy RNG for random number generation
    n                : scenario index (0–4)
    slots_per_step   : number of time-slots per RL step
    propagation_type : channel model string
    L1_level         : True = each slice gets its own L1 resources
    penalty          : SLA violation penalty passed to the gym env
    traffic_config   : optional dict to override default traffic parameters
                       (see module docstring for layout)
    sla_config       : optional dict to override default SLA thresholds
                       (see module docstring for layout)
    '''
    time_per_step = slots_per_step * 1e-4

    sc = scenarios[n]
    n_prbs  = sc['n_prbs']
    n_embb  = sc['n_embb']
    n_mmtc  = sc['n_mmtc']
    n_urllc = sc.get('n_urllc', 0)

    # ---------- resolve traffic and SLA (defaults + caller overrides) ----------

    (CBR_description, VBR_description,
     MTC_description,
     URLLC_CBR_description, URLLC_VBR_description) = _resolve_traffic(traffic_config)

    SLA_embb, SLA_mmtc, SLA_urllc = _resolve_sla(sla_config)

    # -------------------- eMBB normalization constants ----------------------

    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th': 10e6 * time_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'cbr_delay': 20,  # milliseconds

        'vbr_traffic': 5e6 * time_per_step,
        'vbr_th': 10e6 * time_per_step,
        'vbr_queue': 10e4 * slots_per_step,
        'vbr_snr': 35 * slots_per_step,
        'vbr_delay': 20,  # milliseconds
    }

    # -------------------- mMTC normalization constants -----------------------

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    # -------------------- URLLC normalization constants -----------------------

    norm_const_urllc = {
        'cbr_traffic': 2e6 * time_per_step,
        'cbr_th': 5e6 * time_per_step,
        'cbr_queue': 5e3 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'cbr_delay': 10,  # milliseconds

        'vbr_traffic': 2e6 * time_per_step,
        'vbr_th': 7e6 * time_per_step,
        'vbr_queue': 5e3 * slots_per_step,
        'vbr_snr': 35 * slots_per_step,
        'vbr_delay': 10,  # milliseconds
    }

    # ------------------- auxiliary functions -----------------------

    def new_slice_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description,
                            state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb,
                            CBR_description, VBR_description,
                            state_variables_embb, norm_const_embb, slots_per_step)

    def new_slice_urllc(id, rng, user_counter):
        return SliceRANURLC(rng, user_counter, id, SLA_urllc,
                            URLLC_CBR_description, URLLC_VBR_description,
                            state_variables_urllc, norm_const_urllc, slots_per_step)

    # ------------------- environment creation ------------------------

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs=n_prbs)

    mcs_codeset       = MCSCodeset(MCS_CODESET_EMBB)
    mcs_codeset_urllc = MCSCodeset(MCS_CODESET_URLLC)

    scheduler       = ProportionalFair(mcs_codeset)
    scheduler_urllc = ProportionalFair(mcs_codeset_urllc)

    user_counter = count()

    slices_l1 = []

    if L1_level:
        for id in range(n_embb):
            slices_ran_embb = [new_slice_embb(id, rng, user_counter)]
            slice_l1_embb   = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)
            slices_l1.append(slice_l1_embb)

        for id in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id, rng)]
            slice_l1_mmtc   = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

        for id in range(n_urllc):
            slices_ran_urllc = [new_slice_urllc(id, rng, user_counter)]
            slice_l1_urllc   = SliceL1URLLC(rng, snr_generator, 15,
                                             slices_ran_urllc, scheduler_urllc)
            slices_l1.append(slice_l1_urllc)

    else:
        slices_ran_embb = [new_slice_embb(id, rng, user_counter) for id in range(n_embb)]
        slice_l1_embb   = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)
        slices_l1       = [slice_l1_embb]

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id, rng) for id in range(n_mmtc)]
            slice_l1_mmtc   = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

        if n_urllc > 0:
            slices_ran_urllc = [new_slice_urllc(id, rng, user_counter) for id in range(n_urllc)]
            slice_l1_urllc   = SliceL1URLLC(rng, snr_generator, 15,
                                             slices_ran_urllc, scheduler_urllc)
            slices_l1.append(slice_l1_urllc)

    node = NodeB(slices_l1, slots_per_step, n_prbs)

    node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b=node, penalty=penalty)

    return node_env


# ------------ KBRL Learner initialization values ------------------

alfa = 0.05  # learning parameter

embb_sec  = (2, 8)
embb_a    = (4, 20)
mmtc_sec  = (1, 4)
mmtc_a    = (2, 10)
urllc_sec = (1, 4)
urllc_a   = (3, 15)


# ========================= create_kbrl_agent ==============================

def create_kbrl_agent(rng, n, accuracy_range=[0.99, 0.999]):
    '''
    Returns a KBRL agent.

    Parameters
    ----------
    rng            : numpy RNG
    n              : scenario index (0–4)
    accuracy_range : accuracy bounds for the learner
    '''
    sc      = scenarios[n]
    n_prbs  = sc['n_prbs']
    n_embb  = sc['n_embb']
    n_mmtc  = sc['n_mmtc']
    n_urllc = sc.get('n_urllc', 0)
    embb_dim  = len(state_variables_embb)
    mmtc_dim  = len(state_variables_mmtc)
    urllc_dim = len(state_variables_urllc)

    learners = []
    i = 0

    for _ in range(n_embb):
        sv             = SVvariable()
        kernel         = GaussianKernel(sv, 1)
        algorithm      = Projectron(kernel)
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec            = rng.integers(embb_sec[0], embb_sec[1])
        learner        = Learner(algorithm, slice(i, i + embb_dim), initial_action, sec)
        learners.append(learner)
        i += embb_dim

    for _ in range(n_mmtc):
        sv             = SVvariable()
        kernel         = GaussianKernel(sv, 1)
        algorithm      = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec            = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learner        = Learner(algorithm, slice(i, i + mmtc_dim), initial_action, sec)
        learners.append(learner)
        i += mmtc_dim

    for _ in range(n_urllc):
        sv             = SVvariable()
        kernel         = GaussianKernel(sv, 1)
        algorithm      = Projectron(kernel)
        initial_action = rng.integers(urllc_a[0], urllc_a[1])
        sec            = rng.integers(urllc_sec[0], urllc_sec[1])
        learner        = Learner(algorithm, slice(i, i + urllc_dim), initial_action, sec)
        learners.append(learner)
        i += urllc_dim

    kbrl_agent = KBRL_Control(learners, n_prbs, alfa=alfa, accuracy_range=accuracy_range)

    return kbrl_agent


# ========================= create_env_from_config =========================

def create_env_from_config(cfg,
                           rng,
                           slots_per_step=50,
                           propagation_type='macro_cell_urban_2GHz',
                           L1_level=True,
                           penalty=100):
    """
    Build an environment directly from a ScenarioConfig produced by
    config_loader.load_scenario() or config_loader.load_scenarios().

    This is the preferred entry point for YAML-driven experiments.

    Parameters
    ----------
    cfg              : ScenarioConfig  (from config_loader)
    rng              : numpy RNG
    slots_per_step   : number of time-slots per RL step
    propagation_type : channel model string
    L1_level         : True = each slice gets its own L1 resources
    penalty          : SLA violation penalty passed to the gym env

    Example
    -------
        from config_loader import load_scenario
        from system_config import create_env_from_config
        import numpy as np

        rng = np.random.default_rng(42)
        cfg = load_scenario('scenarios.yaml', 'urllc_mix')
        env = create_env_from_config(cfg, rng)
    """
    time_per_step = slots_per_step * 1e-4

    # --- resolve traffic and SLA from the ScenarioConfig ---
    (CBR_description, VBR_description,
     MTC_description,
     URLLC_CBR_description, URLLC_VBR_description) = _resolve_traffic(cfg.traffic_config)

    SLA_embb, SLA_mmtc, SLA_urllc = _resolve_sla(cfg.sla_config)

    n_prbs  = cfg.n_prbs
    n_embb  = cfg.n_embb
    n_mmtc  = cfg.n_mmtc
    n_urllc = cfg.n_urllc

    # -------------------- normalization constants (unchanged) ---------------

    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th':      10e6 * time_per_step,
        'cbr_queue':   10e4 * slots_per_step,
        'cbr_snr':     35 * slots_per_step,
        'cbr_delay':   20,

        'vbr_traffic': 5e6 * time_per_step,
        'vbr_th':      10e6 * time_per_step,
        'vbr_queue':   10e4 * slots_per_step,
        'vbr_snr':     35 * slots_per_step,
        'vbr_delay':   20,
    }

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay':   100 * slots_per_step,
    }

    norm_const_urllc = {
        'cbr_traffic': 2e6 * time_per_step,
        'cbr_th':      5e6 * time_per_step,
        'cbr_queue':   5e3 * slots_per_step,
        'cbr_snr':     35 * slots_per_step,
        'cbr_delay':   10,

        'vbr_traffic': 2e6 * time_per_step,
        'vbr_th':      7e6 * time_per_step,
        'vbr_queue':   5e3 * slots_per_step,
        'vbr_snr':     35 * slots_per_step,
        'vbr_delay':   10,
    }

    # ------------------- auxiliary constructors ----------------------------

    user_counter = count()

    def new_slice_mmtc(id):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description,
                            state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb,
                            CBR_description, VBR_description,
                            state_variables_embb, norm_const_embb, slots_per_step)

    def new_slice_urllc(id):
        return SliceRANURLC(rng, user_counter, id, SLA_urllc,
                            URLLC_CBR_description, URLLC_VBR_description,
                            state_variables_urllc, norm_const_urllc, slots_per_step)

    # ------------------- environment creation ------------------------------

    snr_generator     = SINRSelectiveFading(rng, propagation_type, n_prbs=n_prbs)
    mcs_codeset       = MCSCodeset(MCS_CODESET_EMBB)
    mcs_codeset_urllc = MCSCodeset(MCS_CODESET_URLLC)
    scheduler         = ProportionalFair(mcs_codeset)
    scheduler_urllc   = ProportionalFair(mcs_codeset_urllc)

    slices_l1 = []

    if L1_level:
        for id in range(n_embb):
            slices_l1.append(
                SliceL1eMBB(rng, snr_generator, 20, [new_slice_embb(id)], scheduler)
            )
        for id in range(n_mmtc):
            slices_l1.append(
                SliceL1mMTC(5, [new_slice_mmtc(id)])
            )
        for id in range(n_urllc):
            slices_l1.append(
                SliceL1URLLC(rng, snr_generator, 15, [new_slice_urllc(id)], scheduler_urllc)
            )
    else:
        slices_l1.append(
            SliceL1eMBB(rng, snr_generator, 20,
                        [new_slice_embb(id) for id in range(n_embb)], scheduler)
        )
        if n_mmtc > 0:
            slices_l1.append(
                SliceL1mMTC(5, [new_slice_mmtc(id) for id in range(n_mmtc)])
            )
        if n_urllc > 0:
            slices_l1.append(
                SliceL1URLLC(rng, snr_generator, 15,
                             [new_slice_urllc(id) for id in range(n_urllc)],
                             scheduler_urllc)
            )

    node = NodeB(slices_l1, slots_per_step, n_prbs)
    return gym.make('gym_ran_slice:RanSlice-v1', node_b=node, penalty=penalty)