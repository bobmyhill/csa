import numpy as np
from models.csasolutionmodel import CSAModel, R
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def two_site_cluster_energies(wAB):
    return np.array([[0., 0.96*wAB], [1.04*wAB, 0.]])

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


fig1 = plt.figure(figsize=(16, 12))
ax1 = [fig1.add_subplot(2, 2, i) for i in range(1, 5)]


reduced_temperatures = np.linspace(0.2, 2., 51)
Ss_disordered = np.empty_like(reduced_temperatures)
Ss_equilibrium = np.empty_like(reduced_temperatures)

for lmda in [0.]:  # lmda has no effect in this simple binary system
    for gamma in [1., 1.2]:
        print(gamma)
        interactions = np.array([[0., lmda/2.],
                                 [lmda/2., 0.]])

        ss = CSAModel(cluster_energies=two_site_cluster_energies(wAB=-2.*R),
                      gamma=gamma,
                      site_species=[['A', 'B'], ['A', 'B']],
                      compositional_interactions=interactions)

        for i, T in enumerate(reduced_temperatures):
            ss.equilibrate(composition={'A': 1., 'B': 1.}, temperature=T)
            Ss_equilibrium[i] = ss.molar_entropy

        Cp_equilibrium = reduced_temperatures*np.gradient(Ss_equilibrium, reduced_temperatures)

        ax1[0].plot(reduced_temperatures, Ss_equilibrium/R, label='$\\lambda$:{0} $\\gamma$: {1}'.format(lmda, gamma))
        ax1[1].plot(reduced_temperatures, Cp_equilibrium/R, label='$\\lambda$:{0} $\\gamma$: {1}'.format(lmda, gamma))
ax1[1].legend()
"""
for lmda in [0., R]:  # lmda has an effect at 50:50 composition
    for gamma in [1., 1.2]:
        print(gamma)
        interactions = np.array([[0., lmda/2.],
                                 [lmda/2., 0.]])

        ss = CSAModel(cluster_energies=binary_cluster_energies(wAB=-1.*R),
                      gamma=gamma,
                      site_species=[['A', 'B'], ['A', 'B'],
                                    ['A', 'B'], ['A', 'B']],
                      compositional_interactions=interactions)

        for i, T in enumerate(reduced_temperatures):
            ss.equilibrate(composition={'A': 2., 'B': 2.}, temperature=T)
            Ss_equilibrium[i] = ss.molar_entropy

        Cp_equilibrium = reduced_temperatures*np.gradient(Ss_equilibrium, reduced_temperatures)

        ax1[2].plot(reduced_temperatures, Ss_equilibrium/R, label='$\\lambda$:{0} $\\gamma$: {1}'.format(lmda, gamma))
        ax1[3].plot(reduced_temperatures, Cp_equilibrium/R, label='$\\lambda$:{0} $\\gamma$: {1}'.format(lmda, gamma))
ax1[3].legend()
"""
plt.show()
