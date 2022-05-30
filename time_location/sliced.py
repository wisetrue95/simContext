"""
Sliced Wasserstein Distance.

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License


import numpy as np
from ot.lp import emd2_1d

def get_random_projections(n_projections, d, seed=None):


    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed

    projections = random_state.normal(0., 1., [n_projections, d])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections



def sliced_wasserstein_distance(X_s, X_t, a=None, b=None, n_projections=50, seed=None, log=False):

    X_s = np.asanyarray(X_s)
    X_t = np.asanyarray(X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = np.full(n, 1 / n)
    if b is None:
        b = np.full(m, 1 / m)

    d = X_s.shape[1]

    projections = get_random_projections(n_projections, d, seed)

    X_s_projections = np.dot(projections, X_s.T)
    X_t_projections = np.dot(projections, X_t.T)

    if log:
        projected_emd = np.empty(n_projections)
    else:
        projected_emd = None

    res = 0.

    for i, (X_s_proj, X_t_proj) in enumerate(zip(X_s_projections, X_t_projections)):
        emd = emd2_1d(X_s_proj, X_t_proj, a, b, log=False, dense=False)
        if projected_emd is not None:
            projected_emd[i] = emd
        res += emd

    res = (res / n_projections) ** 0.5
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res
