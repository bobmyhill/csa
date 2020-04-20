from __future__ import print_function
import sys
import numpy as np
from models.csasolutionmodel import CSAModel, R


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def binary_cluster_energies(wAB, alpha=0., beta=0.):
    """
    Indices:
    0 = A
    1 = B

    Cluster pairs:
    AAAA = 0
    AAAB = 3 A-B pairs
    AABB = 4 A-B pairs
    ABBB = 3 A-B pairs
    BBBB = 0

    See Zhang et al., 2003
    """
    u = np.zeros((2, 2, 2, 2))
    u[1, 1, 1, 1] = 0.

    u[0, 1, 1, 1] = 3.*wAB*(1. + beta)
    u[1, 0, 1, 1] = 3.*wAB*(1. + beta)
    u[1, 1, 0, 1] = 3.*wAB*(1. + beta)
    u[1, 1, 1, 0] = 3.*wAB*(1. + beta)

    u[0, 0, 1, 1] = 4.*wAB
    u[0, 1, 0, 1] = 4.*wAB
    u[0, 1, 1, 0] = 4.*wAB
    u[1, 0, 0, 1] = 4.*wAB
    u[1, 0, 1, 0] = 4.*wAB
    u[1, 1, 0, 0] = 4.*wAB

    u[0, 0, 0, 1] = 3.*wAB*(1. + alpha)
    u[0, 0, 1, 0] = 3.*wAB*(1. + alpha)
    u[0, 1, 0, 0] = 3.*wAB*(1. + alpha)
    u[1, 0, 0, 0] = 3.*wAB*(1. + alpha)

    u[0, 0, 0, 0] = 0.
    return u


prms = [[0.25325, 0.0, 0.0, 1.00, 0.4],
        [0.15, 0.0, 0.0, 1.00, 0.6],
        [0.248, 0.0, 0.0, 1.22, 0.4],
        [0.304, 0.0, 0.0, 1.22, 0.4],
        [0.696, 0.0, 0.0, 1.22, 0.4],
        [0.22, 1.0, 0.92, 1.42, 0.4],
        [0.444, 1.0, 0.92, 1.42, 0.4],
        [0.556, 1.0, 0.92, 1.42, 0.4],
        [0.738, 1.0, 0.92, 1.42, 0.4],
        [0.388, 1.0, 0.92, 1.42, 0.6],
        [0.402, 1.0, 0.92, 1.42, 0.6],
        [0.542, 1.0, 0.92, 1.42, 0.6],
        [0.598, 1.0, 0.92, 1.42, 0.6],
        [0.612, 1.0, 0.92, 1.42, 0.6],
        [0.654, 1.0, 0.92, 1.42, 0.6]]

for x, alpha, beta, gamma, T in prms:

    ss = CSAModel(cluster_energies=binary_cluster_energies(wAB=-R,
                                                           alpha=alpha,
                                                           beta=beta),
                  gamma=gamma,
                  site_species=[['A', 'B'], ['A', 'B'],
                                ['A', 'B'], ['A', 'B']])

    ss.equilibrate(composition={'A': 4. * (1.-x), 'B': 4.*x},
                   temperature=T)
    try:
        ss.equilibrate(composition={'A': 4. * (1.-x), 'B': 4.*x},
                       temperature=T)
        print('Good', x, alpha, beta, gamma, T)
    except Exception as exc:
        print(exc)
        eprint('Bad', x, alpha, beta, gamma, T)
