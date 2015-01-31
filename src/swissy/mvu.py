"""mvu.py

"""

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh

import numpy as np
import picos as pic
import cvxopt as cvx

import mosek
import time


def reduce(X, d, k):

    n, D = X.shape

    if k <= 0 or k >= n:
        raise ValueError("can't find that many neighbors")

    if d >= D:
        raise ValueError("can't increase dimensionality")

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Make mosek environment
    env = mosek.Env()

    # Create a task object and attach log stream printer
    task = env.Task(0,0)
    task.set_Stream(mosek.streamtype.log, streamprinter)

    barci, barcj, barcval = [], [], []
    for i in xrange(n):
        barci.append(i)
        barcj.append(i)
        barcval.append(1.)

    task.appendcons(1 + n * k)
    task.appendbarvars([n])

    for idx, dists in enumerate(distances):
        for jid, dist in enumerate(dists[1:]):
            dist2 = dist * dist
            task.putconbound(idx * k + jid, mosek.boundkey.fx, dist2, dist2)

    task.putconbound(n * k, mosek.boundkey.fx, 0., 0.)

    for idx, inds in enumerate(indices):
        assert inds[0] == idx
        for jid, jdx in enumerate(inds[1:]):
            idxmax = max(idx, jdx)
            jdxmin = min(idx, jdx)
            lowerx = [idx, idxmax, jdx]
            lowery = [idx, jdxmin, jdx]
            lowerv = [1., -1., 1.]
            syma = task.appendsparsesymmat(n, lowerx, lowery, lowerv)
            task.putbaraij(idx * k + jid, 0, [syma], [1.0])

    xs, ys = np.tril_indices(n)
    syma = task.appendsparsesymmat(n, xs, ys, [1.] * len(xs))
    task.putbaraij(n * k, 0, [syma], [1.0])

    task.putobjsense(mosek.objsense.maximize)

    print "calling optimize()"
    start = time.time()
    task.optimize()
    end = time.time()
    task.solutionsummary(mosek.streamtype.msg)

    prosta = task.getprosta(mosek.soltype.itr)
    solsta = task.getsolsta(mosek.soltype.itr)

    if solsta not in (mosek.solsta.optimal, mosek.solsta.near_optimal):
        raise ValueError("did not converge")

    barvardim = n
    lenbarvar = barvardim * (barvardim + 1) / 2
    barx = np.zeros(lenbarvar, float)
    task.getbarxj(mosek.soltype.itr, 0, barx)

    iu = np.triu_indices(n=barvardim)
    K = np.zeros((barvardim, barvardim))
    K[iu] = barx

    w, v = eigsh(K, k=d, which='LA')
    w = np.sqrt(w)

    return w * v


def _reduce_picos(X, d, k):
    # TODO: if d is None, then infer the dimension by looking at the
    # eigenvalues of the resulting gram matrix

    n, D = X.shape

    if k <= 0 or k >= n:
        raise ValueError("can't find that many neighbors")

    if d >= D:
        raise ValueError("can't increase dimensionality")

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    p = pic.Problem()
    K = p.add_variable('K', (n, n), vtype='symmetric')

    ones = cvx.matrix(-1. * np.ones((n, n)))
    ones = pic.new_param('ones', ones)
    p.add_constraint( (ones | K) == 0. )

    for idx, (inds, dists) in enumerate(zip(indices, distances)):
        assert inds[0] == idx
        for ind, dist in zip(inds[1:], dists[1:]):
            p.add_constraint( K[idx, idx] - 2. * K[idx, ind] + K[ind, ind] == dist * dist )

    p.add_constraint(K >> 0)
    p.set_objective('max', 'I' | K)
    p.solve()

    K = np.array(K.value)

    w, v = eigsh(K, k=d, which='LA')
    w = np.sqrt(w)

    return w * v
