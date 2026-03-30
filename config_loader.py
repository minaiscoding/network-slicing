#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_loader.py

Loads scenario definitions from a YAML file and converts them into the
traffic_config / sla_config dicts expected by system_config.create_env(),
plus the structural parameters (n_prbs, slice counts) that create_env()
takes directly.

Public API
----------
load_scenarios(path)        -> dict[name -> ScenarioConfig]
load_scenario(path, name)   -> ScenarioConfig

ScenarioConfig fields
---------------------
.name         : str
.n_prbs       : int
.n_embb       : int
.n_mmtc       : int
.n_urllc      : int
.traffic_config : dict   (ready to pass to create_env)
.sla_config     : dict   (ready to pass to create_env)

Usage
-----
    from config_loader import load_scenario
    from system_config import create_env_from_config

    cfg = load_scenario('scenarios.yaml', 'urllc_mix')
    env = create_env_from_config(cfg, rng)
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ZERO_CBR = {
    'lambda': 0.0,   # arrival rate 0 → no CBR UEs ever generated
    't_mean': 1.0,
    'bit_rate': 1,
    'packet_size': 1,
}

_ZERO_VBR = {
    'lambda': 0.0,   # arrival rate 0 → no VBR UEs ever generated
    't_mean': 1.0,
    'p_size': 1,
    'b_size': 1,
    'b_rate': 0,
    'packet_size': 1,
}


def _parse_embb_traffic(slice_cfg: dict) -> dict:
    """
    Build the 'embb' sub-dict for traffic_config from one YAML slice block.
    If a flow key (cbr / vbr) is absent the flow is disabled via a zero
    arrival rate so no UEs of that type are ever generated.
    """
    out = {}
    if 'cbr' in slice_cfg:
        out['cbr'] = dict(slice_cfg['cbr'])
    else:
        out['cbr'] = dict(_ZERO_CBR)   # disabled

    if 'vbr' in slice_cfg:
        out['vbr'] = dict(slice_cfg['vbr'])
    else:
        out['vbr'] = dict(_ZERO_VBR)   # disabled

    return out


def _parse_urllc_traffic(slice_cfg: dict) -> dict:
    """Same logic as eMBB but stored under the 'urllc' key."""
    out = {}
    if 'cbr' in slice_cfg:
        out['cbr'] = dict(slice_cfg['cbr'])
    else:
        out['cbr'] = dict(_ZERO_CBR)

    if 'vbr' in slice_cfg:
        out['vbr'] = dict(slice_cfg['vbr'])
    else:
        out['vbr'] = dict(_ZERO_VBR)

    return out


def _parse_mmtc_traffic(slice_cfg: dict) -> dict:
    """
    mMTC traffic lives flat in the slice block (n_devices, repetition_set,
    period_set).  Only present keys are forwarded; missing ones fall back
    to system_config defaults.
    """
    keys = ('n_devices', 'repetition_set', 'period_set')
    return {k: slice_cfg[k] for k in keys if k in slice_cfg}


def _parse_sla(slice_cfg: dict) -> dict:
    """Return the sla sub-dict if present, else empty dict."""
    return dict(slice_cfg.get('sla', {}))


def _validate_scenario(raw: dict, name: str):
    """Raise ValueError with a clear message on structural problems."""
    for required_type in ('embb', 'mmtc', 'urllc'):
        if required_type not in raw.get('slices', {}):
            raise ValueError(
                f"Scenario '{name}': missing required slice type '{required_type}'. "
                f"Set count: 0 to disable it."
            )
    if 'n_prbs' not in raw:
        raise ValueError(f"Scenario '{name}': missing required field 'n_prbs'.")
    count = raw['slices'].get('urllc', {}).get('count', 0)
    if count > 1:
        raise ValueError(
            f"Scenario '{name}': URLLC count {count} > 1 is not supported yet."
        )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Fully resolved scenario configuration ready to hand to create_env().

    All traffic and SLA keys have already been split into the nested dicts
    that create_env() expects.  Missing YAML keys fall back to the defaults
    defined in system_config.py — this class never fills in those values,
    it only forwards what the YAML explicitly set.
    """
    name:           str
    n_prbs:         int
    n_embb:         int
    n_mmtc:         int
    n_urllc:        int
    traffic_config: dict = field(default_factory=dict)
    sla_config:     dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ScenarioConfig(name={self.name!r}, n_prbs={self.n_prbs}, "
            f"n_embb={self.n_embb}, n_mmtc={self.n_mmtc}, n_urllc={self.n_urllc})"
        )


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------

def _parse_scenario(raw: dict) -> ScenarioConfig:
    """Convert one raw YAML scenario dict into a ScenarioConfig."""
    name = raw.get('name', '<unnamed>')
    _validate_scenario(raw, name)

    slices = raw['slices']
    embb_cfg  = slices.get('embb',  {})
    mmtc_cfg  = slices.get('mmtc',  {})
    urllc_cfg = slices.get('urllc', {})

    n_embb  = int(embb_cfg.get('count',  0))
    n_mmtc  = int(mmtc_cfg.get('count',  0))
    n_urllc = int(urllc_cfg.get('count', 0))

    # --- traffic_config ---
    traffic_config = {}

    if n_embb > 0:
        traffic_config['embb'] = _parse_embb_traffic(embb_cfg)
    elif any(k in embb_cfg for k in ('cbr', 'vbr')):
        # user defined traffic for a disabled slice — warn but continue
        import warnings
        warnings.warn(
            f"Scenario '{name}': embb traffic params defined but count=0, ignoring."
        )

    if n_mmtc > 0:
        mmtc_traffic = _parse_mmtc_traffic(mmtc_cfg)
        if mmtc_traffic:
            traffic_config['mmtc'] = mmtc_traffic

    if n_urllc > 0:
        traffic_config['urllc'] = _parse_urllc_traffic(urllc_cfg)

    # --- sla_config ---
    sla_config = {}

    if n_embb > 0:
        embb_sla = _parse_sla(embb_cfg)
        if embb_sla:
            sla_config['embb'] = embb_sla

    if n_mmtc > 0:
        mmtc_sla = _parse_sla(mmtc_cfg)
        if mmtc_sla:
            sla_config['mmtc'] = mmtc_sla

    if n_urllc > 0:
        urllc_sla = _parse_sla(urllc_cfg)
        if urllc_sla:
            sla_config['urllc'] = urllc_sla

    return ScenarioConfig(
        name=name,
        n_prbs=int(raw['n_prbs']),
        n_embb=n_embb,
        n_mmtc=n_mmtc,
        n_urllc=n_urllc,
        traffic_config=traffic_config,
        sla_config=sla_config,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_scenarios(path: str | Path) -> dict[str, ScenarioConfig]:
    """
    Parse all scenarios from a YAML file.

    Returns a dict keyed by scenario name so you can do:
        configs = load_scenarios('scenarios.yaml')
        cfg = configs['urllc_mix']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    raw_list = data.get('scenarios', [])
    if not raw_list:
        raise ValueError(f"No scenarios found in {path}")

    configs = {}
    for raw in raw_list:
        cfg = _parse_scenario(raw)
        if cfg.name in configs:
            raise ValueError(f"Duplicate scenario name: '{cfg.name}'")
        configs[cfg.name] = cfg

    return configs


def load_scenario(path: str | Path, name: str) -> ScenarioConfig:
    """
    Load a single named scenario from a YAML file.

    Raises KeyError if the name is not found (lists available names).
    """
    configs = load_scenarios(path)
    if name not in configs:
        available = ', '.join(f"'{n}'" for n in configs)
        raise KeyError(
            f"Scenario '{name}' not found in {path}. "
            f"Available: {available}"
        )
    return configs[name]