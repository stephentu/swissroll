"""mvu.py

"""


import picos as pic
import cvxopt as cvx


def reduce(X, d, k):


    n, D = X.shape

    if k <= 0 or k > n:
        raise ValueError("can't find that many neighbors")

    if d >= D:
        raise ValueError("can't increase dimensionality")

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
