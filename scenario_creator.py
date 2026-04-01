#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Defines functions to create either:
- a single-gNodeB environment (legacy path)
- a multi-gNodeB environment (new path)
- the legacy KBRL agent
"""

import gymnasium as gym
from itertools import count
from typing import Dict, List, Optional

from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC, SliceL1URLLC
from slice_ran import SliceRANmMTC, SliceRANeMBB, SliceRANURLC
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron

try:
    from multi_gnb_wrapper import MultiGNBWrapper
except ImportError:
    MultiGNBWrapper = None

"""
For the senario we need to define the real parameter that we need
"""

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
    'n_prbs': 100,
    'n_embb': 1,
    'n_mmtc': 1,
    'n_urllc': 1
}

scenarios = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]


# -------------------- eMBB parameters -------------------------

CBR_description = {
    #'lambda': 1.0/60.0, # low traffic
    'lambda': 2.0/60.0,
    't_mean': 30.0,
    'bit_rate': 500000
}

VBR_description = {
    #'lambda': 1.0/60.0, # low traffic
    'lambda': 5.0/60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1
}

SLA_embb = {
    'cbr_th': 10e6,
    'cbr_prb': 20, # 30
    'cbr_queue': 10e4, # 5e4

    'vbr_th': 15e6, # 10e6
    'vbr_prb': 30, # 40
    'vbr_queue': 15e4,
}

state_variables_embb = ['cbr_traffic','cbr_th', 'cbr_prb',
                        'cbr_queue', 'cbr_snr',
                        'vbr_traffic', 'vbr_th', 'vbr_prb',
                        'vbr_queue', 'vbr_snr']

# -------------------- mMTC parameters -------------------------

MTC_description = {
    'n_devices': 1000,
    'repetition_set': [2,4,8,16,32,64,128],
    'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]
}

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

SLA_mmtc = {
    'delay': 300
}

# -------------------- URLLC parameters -------------------------

URLLC_CBR_description = {
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 2.0/60.0,
    't_mean': 30.0,
    'bit_rate': 500000
}

URLLC_VBR_description = {
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 5.0/60.0,
    't_mean': 30.0,
    'p_size': 1000,
    'b_size': 500,
    'b_rate': 1
}

SLA_urllc = {
    'cbr_th': 10e6,
    'cbr_prb': 20, # 30
    'cbr_queue': 5e4, # 5e4

    'vbr_th': 15e6, # 10e6
    'vbr_prb': 30, # 40
    'vbr_queue': 10e4,
}

state_variables_urllc = ['cbr_traffic','cbr_th', 'cbr_prb',
                         'cbr_queue', 'cbr_snr',
                         'vbr_traffic', 'vbr_th', 'vbr_prb',
                         'vbr_queue', 'vbr_snr']


# -------------------- helper builders -------------------------

def _get_scenario_config(n: int) -> Dict:
    return scenarios[n]


def _build_norm_constants(slots_per_step: int, slot_length: float):
    time_per_step = slots_per_step * slot_length

    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th': 10e6 * time_per_step,
        'cbr_prb': 25 * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,

        'vbr_traffic': 5e6 * time_per_step,
        'vbr_th': 10e6 * time_per_step,
        'vbr_prb': 35 * slots_per_step,
        'vbr_queue': 10e4 * slots_per_step,
        'vbr_snr': 35 * slots_per_step,
    }

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step,
    }

    norm_const_urllc = {
        'cbr_traffic': 2e6 * time_per_step,
        'cbr_th': 5e6 * time_per_step,
        'cbr_prb': 15 * slots_per_step,
        'cbr_queue': 5e3 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,

        'vbr_traffic': 2e6 * time_per_step,
        'vbr_th': 7e6 * time_per_step,
        'vbr_prb': 20 * slots_per_step,
        'vbr_queue': 5e3 * slots_per_step,
        'vbr_snr': 35 * slots_per_step,
    }

    return norm_const_embb, norm_const_mmtc, norm_const_urllc


def _build_slices_l1(
    rng,
    scenario_idx: int,
    slots_per_step: int,
    propagation_type: str,
    L1_level: bool,
    slot_length: float,
):
    sc = _get_scenario_config(scenario_idx)
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    n_urllc = sc.get('n_urllc', 0)

    norm_const_embb, norm_const_mmtc, norm_const_urllc = _build_norm_constants(
        slots_per_step=slots_per_step,
        slot_length=slot_length,
    )

    def new_slice_mmtc(id_, rng_):
        return SliceRANmMTC(
            rng_, id_, SLA_mmtc, MTC_description,
            state_variables_mmtc, norm_const_mmtc, slots_per_step
        )

    def new_slice_embb(id_, rng_, user_counter_):
        return SliceRANeMBB(
            rng_, user_counter_, id_, SLA_embb,
            CBR_description, VBR_description,
            state_variables_embb, norm_const_embb, slots_per_step,
            slot_length=slot_length
        )

    def new_slice_urllc(id_, rng_, user_counter_):
        return SliceRANURLC(
            rng_, user_counter_, id_, SLA_urllc,
            URLLC_CBR_description, URLLC_VBR_description,
            state_variables_urllc, norm_const_urllc, slots_per_step,
            slot_length=slot_length
        )

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs=n_prbs)
    mcs_codeset = MCSCodeset()
    scheduler = ProportionalFair(mcs_codeset)
    user_counter = count()

    slices_l1 = []

    if L1_level:
        for id_ in range(n_embb):
            slices_ran_embb = [new_slice_embb(id_, rng, user_counter)]
            slices_l1.append(SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler))

        for id_ in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id_, rng)]
            slices_l1.append(SliceL1mMTC(5, slices_ran_mmtc))

        for id_ in range(n_urllc):
            slices_ran_urllc = [new_slice_urllc(id_, rng, user_counter)]
            slices_l1.append(SliceL1URLLC(rng, snr_generator, 15, slices_ran_urllc, scheduler))
    else:
        if n_embb > 0:
            slices_ran_embb = [new_slice_embb(id_, rng, user_counter) for id_ in range(n_embb)]
            slices_l1.append(SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler))

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id_, rng) for id_ in range(n_mmtc)]
            slices_l1.append(SliceL1mMTC(5, slices_ran_mmtc))

        if n_urllc > 0:
            slices_ran_urllc = [new_slice_urllc(id_, rng, user_counter) for id_ in range(n_urllc)]
            slices_l1.append(SliceL1URLLC(rng, snr_generator, 15, slices_ran_urllc, scheduler))

    return slices_l1, n_prbs


def create_nodeb(
    rng,
    n,
    slots_per_step=50,
    propagation_type='macro_cell_urban_2GHz',
    L1_level=True,
    node_id=0,
    node_x=0.0,
    node_y=0.0,
    coverage_radius=500,
    slot_length=1e-3,
    carrier_id=0,
    center_frequency_hz=3.5e9,
    bandwidth_hz=20e6,
    tx_power_dbm=30.0,
    noise_figure_db=7.0,
):
    slices_l1, n_prbs = _build_slices_l1(
        rng=rng,
        scenario_idx=n,
        slots_per_step=slots_per_step,
        propagation_type=propagation_type,
        L1_level=L1_level,
        slot_length=slot_length,
    )

    return NodeB(
        id=node_id,
        x=node_x,
        y=node_y,
        slices_l1=slices_l1,
        slots_per_step=slots_per_step,
        n_prbs=n_prbs,
        coverage_radius=coverage_radius,
        slot_length=slot_length,
        carrier_id=carrier_id,
        center_frequency_hz=center_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        tx_power_dbm=tx_power_dbm,
        noise_figure_db=noise_figure_db,
    )


def default_gnb_configs(n_gnbs: int, coverage_radius: float = 500.0, spacing: Optional[float] = None):
    """
    Build a simple default topology for quick experiments.
    - 1 gNB: origin
    - 2 gNBs: line
    - 3 gNBs: triangle
    - 4+ gNBs: line
    """
    if spacing is None:
        spacing = 1.5 * coverage_radius

    if n_gnbs <= 0:
        raise ValueError("n_gnbs must be >= 1")

    if n_gnbs == 1:
        positions = [(0.0, 0.0)]
    elif n_gnbs == 2:
        positions = [(0.0, 0.0), (spacing, 0.0)]
    elif n_gnbs == 3:
        h = 0.8660254037844386 * spacing
        positions = [(0.0, 0.0), (spacing, 0.0), (0.5 * spacing, h)]
    else:
        positions = [(i * spacing, 0.0) for i in range(n_gnbs)]

    return [
        {
            "id": i,
            "x": float(x),
            "y": float(y),
            "coverage_radius": coverage_radius,
            "carrier_id": 0,
            "center_frequency_hz": 3.5e9,
            "bandwidth_hz": 20e6,
            "tx_power_dbm": 30.0,
            "noise_figure_db": 7.0,
        }
        for i, (x, y) in enumerate(positions)
    ]


def create_multignb_env(
    rng,
    n,
    slots_per_step=50,
    propagation_type='macro_cell_urban_2GHz',
    L1_level=True,
    slot_length=1e-3,
    gnb_configs: Optional[List[Dict]] = None,
    n_gnbs: Optional[int] = None,
    coverage_radius=500,
    handover_hysteresis: float = 0.05,
    handover_ttt: int = 3,
    outage_penalty: float = 1.0,
    handover_penalty: float = 0.1,
    use_mean_gnb_reward: bool = True,
    verbose: bool = False,
):
    if MultiGNBWrapper is None:
        raise ImportError(
            "MultiGNBWrapper could not be imported. "
            "Make sure multi_gnb_wrapper.py is available in the project path."
        )

    if gnb_configs is None:
        n_gnbs = 2 if n_gnbs is None else n_gnbs
        gnb_configs = default_gnb_configs(n_gnbs=n_gnbs, coverage_radius=coverage_radius)

    gnb_list = []
    for idx, cfg in enumerate(gnb_configs):
        node = create_nodeb(
            rng=rng,
            n=n,
            slots_per_step=slots_per_step,
            propagation_type=propagation_type,
            L1_level=L1_level,
            node_id=cfg.get('id', idx),
            node_x=cfg.get('x', 0.0),
            node_y=cfg.get('y', 0.0),
            coverage_radius=cfg.get('coverage_radius', coverage_radius),
            slot_length=slot_length,
            carrier_id=cfg.get('carrier_id', 0),
            center_frequency_hz=cfg.get('center_frequency_hz', 3.5e9),
            bandwidth_hz=cfg.get('bandwidth_hz', 20e6),
            tx_power_dbm=cfg.get('tx_power_dbm', 30.0),
            noise_figure_db=cfg.get('noise_figure_db', 7.0),
        )
        gnb_list.append(node)

    return MultiGNBWrapper(
        gnb_list=gnb_list,
        handover_hysteresis=handover_hysteresis,
        handover_ttt=handover_ttt,
        outage_penalty=outage_penalty,
        handover_penalty=handover_penalty,
        use_mean_gnb_reward=use_mean_gnb_reward,
        verbose=verbose,
    )


def create_env(
    rng,
    n,
    slots_per_step=50,
    propagation_type='macro_cell_urban_2GHz',
    L1_level=True,
    penalty=100,
    node_id=0,
    node_x=0.0,
    node_y=0.0,
    coverage_radius=500,
    slot_length=1e-3,
    multi_gnb: bool = False,
    gnb_configs: Optional[List[Dict]] = None,
    n_gnbs: Optional[int] = None,
    handover_hysteresis: float = 0.05,
    handover_ttt: int = 3,
    outage_penalty: float = 1.0,
    handover_penalty: float = 0.1,
    use_mean_gnb_reward: bool = True,
    verbose: bool = False,
):
    """
    Create either the legacy single-gNB env or the new multi-gNB env.

    Legacy path:
        env = create_env(rng, n)

    Multi-gNB path:
        env = create_env(rng, n, multi_gnb=True, gnb_configs=[...])
    """
    if multi_gnb:
        return create_multignb_env(
            rng=rng,
            n=n,
            slots_per_step=slots_per_step,
            propagation_type=propagation_type,
            L1_level=L1_level,
            slot_length=slot_length,
            gnb_configs=gnb_configs,
            n_gnbs=n_gnbs,
            coverage_radius=coverage_radius,
            handover_hysteresis=handover_hysteresis,
            handover_ttt=handover_ttt,
            outage_penalty=outage_penalty,
            handover_penalty=handover_penalty,
            use_mean_gnb_reward=use_mean_gnb_reward,
            verbose=verbose,
        )

    node = create_nodeb(
        rng=rng,
        n=n,
        slots_per_step=slots_per_step,
        propagation_type=propagation_type,
        L1_level=L1_level,
        node_id=node_id,
        node_x=node_x,
        node_y=node_y,
        coverage_radius=coverage_radius,
        slot_length=slot_length,
    )

    node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b=node, penalty=penalty)
    return node_env


# ------------ KBRL Learner initialization values ------------------

alfa = 0.05 # learning parameter

# initial offset and initial action are initialized at random
embb_sec = (2, 8)
embb_a = (4, 20)
mmtc_sec = (1, 4)
mmtc_a = (2, 10)
urllc_sec = (1, 4)
urllc_a = (3, 15)


# -------------------- create KBRL agent -------------------------

def create_kbrl_agent(rng, n, accuracy_range = [0.99, 0.999]):
    '''
    Returns kbrl agent:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    - accuracy_range: for the learner
    - budget: number of support vectors in memory
    '''
    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    n_urllc = sc.get('n_urllc', 0)
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)
    urllc_dim = len(state_variables_urllc)

    learners = []
    i = 0

    # create one learner instance per slice
    for _ in range(n_embb):
        sv = SVvariable() # create support vector memory
        kernel = GaussianKernel(sv,1) # kernel
        algorithm = Projectron(kernel) # online classifier
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec = rng.integers(embb_sec[0], embb_sec[1])
        learner = Learner(algorithm, slice(i,i+embb_dim), initial_action, sec)
        learners.append(learner)
        i += embb_dim

    for _ in range(n_mmtc):
        sv = SVvariable()
        kernel = GaussianKernel(sv,1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learner = Learner(algorithm, slice(i,i+mmtc_dim), initial_action, sec)
        learners.append(learner)
        i += mmtc_dim

    for _ in range(n_urllc):
        sv = SVvariable()
        kernel = GaussianKernel(sv,1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(urllc_a[0], urllc_a[1])
        sec = rng.integers(urllc_sec[0], urllc_sec[1])
        learner = Learner(algorithm, slice(i,i+urllc_dim), initial_action, sec)
        learners.append(learner)
        i += urllc_dim

    kbrl_agent = KBRL_Control(learners, n_prbs, alfa=alfa, accuracy_range=accuracy_range)
    return kbrl_agent



def _print_header(title: str):
    print("" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_gnb_layout(env):
    print("gNB layout:")
    for i, gnb in enumerate(env.gnbs):
        print(
            f"  gNB {i}: id={gnb.id}, pos=({gnb.x:.2f}, {gnb.y:.2f}), "
            f"radius={gnb.coverage_radius:.2f}, carrier={getattr(gnb, 'carrier_id', 'NA')}, "
            f"n_prbs={gnb.n_prbs}, n_slices={gnb.n_slices_l1}"
        )


def _assert(condition, message: str):
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def _run_multi_gnb_core_test(seed: int = 123):
    rng = np.random.default_rng(seed)
    _print_header("TEST 2 - Multi-gNB environment creation and action flow")
    env = create_env(
        rng=rng,
        n=1,
        multi_gnb=True,
        n_gnbs=3,
        coverage_radius=500,
        handover_hysteresis=0.05,
        handover_ttt=3,
        verbose=False,
    )
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    _print_gnb_layout(env)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"Reward after one step: {reward}")
    print(f"Info keys: {sorted(info.keys())}")
    print(f"UE per gNB: {info['ue_per_gnb']}")
    _assert(env.n_gnbs == 3, "multi-gNB env creates the requested number of gNBs")
    _assert(obs.shape[0] == env.observation_space.shape[0], "observation size matches observation space")
    _assert(action.shape[0] == env.action_space.shape[0], "sampled action size matches action space")
    _assert(len(info['ue_per_gnb']) == env.n_gnbs, "info contains one load entry per gNB")
    _assert(not terminated and not truncated, "multi-gNB smoke step does not terminate immediately")
    return env


def _run_attachment_and_radio_test(seed: int = 456):
    rng = np.random.default_rng(seed)
    _print_header("TEST 3 - UE insertion, attachment, and radio metrics")
    env = create_env(rng=rng, n=1, multi_gnb=True, n_gnbs=3, coverage_radius=500, verbose=False)
    env.reset()

    ue_positions = [(-50.0, 0.0), (30.0, 120.0), (80.0, -140.0), (150.0, 60.0), (-120.0, -90.0)]
    ue_ids = []
    for x, y in ue_positions:
        ue_id = env.add_ue(x=x, y=y, vx=0.0, vy=0.0)
        ue_ids.append(ue_id)
        metrics = env.get_ue_radio_metrics(ue_id)
        print(f"UE {ue_id} -> serving_gnb={metrics['serving_gnb']}, sinr={metrics['sinr_db']:.2f} dB, connected={metrics['connected']}")

    info = env._build_info(per_gnb_rewards=[0.0] * env.n_gnbs)
    print(f"Tracked UEs: {info['n_tracked_ues']}")
    print(f"Connected UEs: {info['n_connected_ues']}")
    print(f"Disconnected UEs: {info['n_disconnected_ues']}")
    print(f"UE per gNB: {info['ue_per_gnb']}")
    _assert(len(env.get_all_ues()) == len(ue_positions), "all inserted UEs are tracked")
    _assert(info['n_connected_ues'] >= 1, "at least one UE attaches to a serving gNB")
    _assert(info['n_tracked_ues'] == len(ue_positions), "tracked UE count matches inserted UE count")
    _assert(all(np.isfinite(env.get_ue_radio_metrics(uid)['sinr_db']) for uid in ue_ids), "attached test UEs have finite SINR values")
    return env, ue_ids


def _run_handover_test(seed: int = 789):
    rng = np.random.default_rng(seed)
    _print_header("TEST 4 - Handover and mobility")
    gnb_configs = [
        {"id": 0, "x": 0.0, "y": 0.0, "coverage_radius": 600.0, "carrier_id": 0},
        {"id": 1, "x": 900.0, "y": 0.0, "coverage_radius": 600.0, "carrier_id": 0},
        {"id": 2, "x": 1800.0, "y": 0.0, "coverage_radius": 600.0, "carrier_id": 1},
    ]
    env = create_env(
        rng=rng,
        n=1,
        multi_gnb=True,
        gnb_configs=gnb_configs,
        coverage_radius=600,
        handover_hysteresis=0.0,
        handover_ttt=1,
        verbose=False,
    )
    env.reset()
    _print_gnb_layout(env)

    ue_id = env.add_ue(x=400.0, y=0.0, vx=120.0, vy=0.0)
    ue = env.get_ue(ue_id)
    print(f"Initial UE state: x={ue.x:.2f}, y={ue.y:.2f}, serving_gnb={ue.serving_gnb}")

    ho_detected = False
    last_serving = ue.serving_gnb
    for step in range(80):
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        ue = env.get_ue(ue_id)
        print(
            f"Step {step:02d}: x={ue.x:.2f}, serving={ue.serving_gnb}, "
            f"connected={ue.connected}, handovers_step={info['handover_count_step']}, reward={reward:.3f}"
        )
        if info['handover_count_step'] > 0 or ue.serving_gnb != last_serving:
            ho_detected = True
            break
        last_serving = ue.serving_gnb

    _assert(ho_detected, "a moving UE triggers at least one handover in the overlap region")
    _assert(len(env.handover_log) >= 1, "handover events are recorded in the log")
    print(f"Handover log: {env.handover_log}")
    return env


def _run_disconnect_test(seed: int = 321):
    rng = np.random.default_rng(seed)
    _print_header("TEST 5 - Out-of-coverage / disconnection handling")
    env = create_env(rng=rng, n=1, multi_gnb=True, n_gnbs=3, coverage_radius=400, verbose=False)
    env.reset()
    ue_id = env.add_ue(x=5000.0, y=5000.0, vx=0.0, vy=0.0)
    metrics = env.get_ue_radio_metrics(ue_id)
    print(f"Far UE metrics: {metrics}")
    info = env._build_info(per_gnb_rewards=[0.0] * env.n_gnbs)
    print(f"Connected={info['n_connected_ues']}, Disconnected={info['n_disconnected_ues']}")
    _assert(metrics['connected'] is False, "far UE starts disconnected when outside all coverages")
    _assert(info['n_disconnected_ues'] >= 1, "disconnected UE count increases for out-of-coverage UE")
    return env


if __name__ == "__main__":
    import numpy as np

    print("=== Comprehensive scenario_creator test suite ===")
    try:

        _run_multi_gnb_core_test()
        _run_attachment_and_radio_test()
        _run_handover_test()
        _run_disconnect_test()
        print("All tests completed successfully.")
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        raise