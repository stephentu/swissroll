"""laplacian_eigenmaps.py

"""

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigsh

import numpy as np


def reduce(X, d, k):
    """


    """

    n, D = X.shape

    if k <= 0 or k > n:
        raise ValueError("can't find that many neighbors")

    if d >= D:
        raise ValueError("can't increase dimensionality")

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)

    # adjacency matrix of the graph
    W = nbrs.kneighbors_graph(X)

    # form diagonal
    ds = np.array([[W[i, :].sum() for i in xrange(n)]])
    dinvs = 1. / ds
    D = spdiags(ds, np.array([0]), n, n)
    Dinv = spdiags(dinvs, np.array([0]), n, n)

    # form graph laplacian
    L = D - W

    _, v = eigsh(L, k=d + 1, M=D, sigma=None, which='SA', Minv=Dinv)

    return v[:, 1:]
