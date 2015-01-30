
import numpy as np

from swissy import laplacian_eigenmaps


def main():
    X = np.loadtxt("../data/2_class_swiss_roll.csv", delimiter=',')
    X = X[:10, :]
    Xr = laplacian_eigenmaps.reduce(X, 2, 6)
    print Xr.shape


if __name__ == '__main__':
    main()
