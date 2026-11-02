"""Isolated surgical repair; the hash-pinned historical constructor stays untouched.

Only the state-link and forbidden-edge blocks are replaced. This is NOT a
validation of the inherited fitting, ramping, objective, or scientific claims.
T=1 is deliberately unsupported (the inherited endpoint objective overlaps).
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np
from scipy import sparse

from .construct_tmilp_project import construct_tmilp_1bin_project as _legacy
from .milp_model import RowAccumulator

SEMANTIC_VERSION = 'tmilp-project-v3-state-edge-repair-20261015'
LEGACY_SHA256 = '40200e9955e50e927c26f5a6e9c493cf12726f9564e3b153248220d493b8c894'
LEGACY_SOURCE = Path(__file__).with_name('construct_tmilp_project.py')


def check_legacy_source():
    if hashlib.sha256(LEGACY_SOURCE.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise RuntimeError('Historical constructor changed: refuse positional block replacement')


def row_layout(S, T, has_q, transitions):
    n = int(sum(S))
    m, nx = n * (T - 1), n * T
    balance = 2 * T * (1 + int(has_q))
    start = balance + m
    trans_start = balance + (5 + 2 * int(has_q)) * m + (3 + 2 * int(has_q)) * nx
    old_edges = (T - 1) * sum(bool(np.any(np.asarray(tm) == 0)) for tm in transitions)
    new_edges = (T - 1) * sum(int(np.count_nonzero(np.asarray(tm) == 0)) for tm in transitions)
    return dict(state_start=start, state_end=start + 2*m,
                transition_start=trans_start, old_transition_rows=old_edges,
                new_transition_rows=new_edges)


def construct_tmilp_1bin_project(*args, **kwargs):
    """Same call signature as historical constructor, new model/cache identity."""
    check_legacy_source()
    p = inspect.signature(_legacy).bind(*args, **kwargs).arguments
    T = p['T']
    S = np.asarray(p['S'], dtype=float).ravel()
    if not np.isscalar(T) or not np.isfinite(T) or T != int(T) or T < 2:
        raise ValueError('v3 requires an integer T >= 2')
    T = int(T)
    if not len(S) or not np.all(np.isfinite(S)) or np.any(S < 1) or np.any(S != S.astype(int)):
        raise ValueError('S must contain positive integer state counts')
    S = S.astype(int)
    n = int(sum(S))
    transitions = p['Transition_model']
    if len(transitions) != len(S):
        raise ValueError('One transition matrix is required per appliance')
    for count, tm in zip(S, transitions):
        tm = np.asarray(tm)
        if tm.shape != (count, count) or not np.all(np.isin(tm, [0, 1])):
            raise ValueError('Transition matrices must be binary, indexed [from, to]')
    has_q = p['Q_Sum'] is not None and np.asarray(p['Q_Sum']).size > 0
    for prefix in ('P', 'Q') if has_q else ('P',):
        for suffix, size in (('top', n), ('bottom', n), ('up', n), ('down', n), ('Sum', T)):
            a = np.asarray(p[prefix+'_'+suffix], dtype=float).ravel()
            if a.size != size or not np.all(np.isfinite(a)):
                raise ValueError(f'Invalid {prefix}_{suffix} shape or nonfinite value')
        if np.any(np.asarray(p[prefix+'_top']).ravel() < np.asarray(p[prefix+'_bottom']).ravel()):
            raise ValueError(f'{prefix}_top < {prefix}_bottom: no silent interval repair')
    limits = np.asarray(p['State_Limit'], dtype=float).ravel()
    if len(limits) != n or not np.all(np.isfinite(limits)) or np.any(limits < 1):
        raise ValueError('State_Limit must contain finite values >= 1')
    for name in ('C_start', 'C_run'):
        if not np.isscalar(p[name]) or not np.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')

    model = _legacy(*args, **kwargs)
    layout = row_layout(S, T, has_q, transitions)
    if model.Aineq.shape != (layout['transition_start'] + layout['old_transition_rows'], model.num_vars()):
        raise RuntimeError('Unexpected historical row layout')
    m = n * (T - 1)
    r = np.arange(m)
    current = n + r
    starts = n*T + current
    acc = RowAccumulator()
    acc.add(r, starts, np.ones(m))
    acc.add(r, current, -np.ones(m))
    acc.add(r+m, r, np.ones(m))
    acc.add(r+m, starts, np.ones(m))
    state = acc.csr(2*m, model.num_vars())
    acc = RowAccumulator()
    row, offset = 0, 0
    for count, tm in zip(S, transitions):
        frm, to = np.where(np.asarray(tm) == 0)
        for t in range(T-1):
            rows = row + np.arange(len(frm))
            acc.add(rows, offset + frm + t*n, np.ones(len(frm)))
            acc.add(rows, offset + to + (t+1)*n, np.ones(len(frm)))
            row += len(frm)
        offset += count
    if row != layout['new_transition_rows']:
        raise RuntimeError('Forbidden-edge row count mismatch')
    edge = acc.csr(row, model.num_vars())
    a, b, c = layout['state_start'], layout['state_end'], layout['transition_start']
    model.Aineq = sparse.vstack([model.Aineq[:a], state, model.Aineq[b:c], edge], format='csr')
    model.bineq = np.concatenate([model.bineq[:a], np.zeros(m), np.ones(m), model.bineq[b:c], np.ones(row)])
    return model
