"""Isolated state/edge/ramp repair, with explicit left-censored boundaries.

The frozen historical constructor and the v3 surgical draft are unchanged.
Transition matrices are [from, to]. Runs already active at the first sample
have unknown age; starts in the last samples may continue beyond the window.
"""
import inspect

import numpy as np
from scipy import sparse

from .construct_tmilp_project import construct_tmilp_1bin_project as _signature
from .construct_tmilp_project_v3 import construct_tmilp_1bin_project as _state_edge

SEMANTIC_VERSION = 'tmilp-project-v4-state-edge-downramp-20261015'


def construct_tmilp_1bin_project(*args, **kwargs):
    p = inspect.signature(_signature).bind(*args, **kwargs).arguments
    n, T = int(np.sum(p['S'])), p['T']
    # Additional checks before the inherited objective can broadcast/truncate.
    for key in ('C', 'penlty_l'):
        a = np.asarray(p[key], dtype=float).ravel()
        if a.size != n or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f'{key} must have one finite nonnegative value per state')
    probability = np.asarray(p['penlty_s'], dtype=float)
    days = p['test_day_num']
    if not np.isscalar(days) or not np.isfinite(days) or days < 1 or days != int(days):
        raise ValueError('test_day_num must be a positive integer')
    if (probability.ndim != 2 or probability.shape[0] != n
            or probability.shape[1] * days != T or not np.all(np.isfinite(probability))
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('penlty_s must be finite state x time probabilities matching T')
    limits = np.asarray(p['State_Limit'], dtype=float)
    if np.any(limits != np.floor(limits)):
        raise ValueError('State_Limit must contain integer sample counts')
    has_q = p['Q_Sum'] is not None and np.asarray(p['Q_Sum']).size > 0
    for prefix in ('P', 'Q') if has_q else ('P',):
        for suffix in ('up', 'down'):
            a = np.asarray(p[prefix + '_' + suffix], dtype=float)
            if np.any(a < 0):
                raise ValueError('Ramp limits must be nonnegative')
    model = _state_edge(*args, **kwargs)
    T = int(T)
    nx, m = n*T, n*(T-1)
    # Historical c13/c15 use +u_t-u_(t-1), whereas MATLAB uses
    # +u_(t-1)-u_t. For an unchanged state this must enforce
    # u_(t-1)-u_t <= down/(top-bottom), independently of ramp-up.
    ramp_start = 2*T*(1+int(has_q)) + 3*m + (3+2*int(has_q))*nx
    correction = sparse.lil_matrix(model.Aineq.shape)
    for block, variable_start in ((1, 2*nx), (3, 3*nx)) if has_q else ((1, 2*nx),):
        rows = ramp_start + block*m + np.arange(m)
        correction[rows, variable_start + np.arange(m)] = 2.
        correction[rows, variable_start + n + np.arange(m)] = -2.
    model.Aineq = (model.Aineq + correction.tocsr()).tocsr()
    # Initial occupancy is not an observed startup.
    model.ub[nx:nx+n] = 0.
    return model
