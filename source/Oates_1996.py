import numpy as np
from models.csasolutionmodel import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def binary_cluster_energies(wAB):
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
    u[1,1,1,1] = 0.

    u[0,1,1,1] = 3.*wAB
    u[1,0,1,1] = 3.*wAB
    u[1,1,0,1] = 3.*wAB
    u[1,1,1,0] = 3.*wAB

    u[0,0,1,1] = 4.*wAB
    u[0,1,0,1] = 4.*wAB
    u[0,1,1,0] = 4.*wAB
    u[1,0,0,1] = 4.*wAB
    u[1,0,1,0] = 4.*wAB
    u[1,1,0,0] = 4.*wAB

    u[0,0,0,1] = 3.*wAB
    u[0,0,1,0] = 3.*wAB
    u[0,1,0,0] = 3.*wAB
    u[1,0,0,0] = 3.*wAB

    u[0,0,0,0] = 0.
    return u

def ternary_cluster_energies(wAB):
    """
    Indices:
    0 = A
    1 = B

    Cluster pairs:
    AAAA = 0 -> all the same

    AAAB = 3 A-B pairs -> 2 types, one different

    AABB = 4 A-B pairs -> 2 types, 2:2

    AABC = 1 B-C pair, 2 A-B pairs, 2 B-C pairs -> 3 types

    See Zhang et al., 2003
    """
    u = np.zeros((3,3,3,3))

    for indices, v in np.ndenumerate(u):

        if len(set(indices)) == 1:
            u[indices] = 0.
        elif len(set(indices)) == 3:
            u[indices] = 5.*wAB
        elif any(indices.count(x) == 1 for x in list(set(indices))):
            u[indices] = 3.*wAB
        else:
            u[indices] = 4.*wAB

    return u


def minimize_energy(T, p_A, ss):
    def energy(args, T, p_A, ss):
        A1, A2, A3 = args
        A4 = 4.*p_A - (A1 + A2 + A3)
        # independent clusters are AAAA, AAAB, AABA, ABAA, BAAA

        ss.set_state(T)
        ss.set_composition_from_p_s(np.array([1.-A1, 0.+A1,
                                              1.-A2, 0.+A2,
                                              1.-A3, 0.+A3,
                                              1.-A4, 0.+A4]))

        ss.equilibrate_clusters()
        return ss.molar_gibbs

    return minimize(energy, [1.*p_A, 1.*p_A, 1.*p_A], args=(T, p_A, ss))


"""
Figure 1 from Oates et al. (1996)
wAB/RT = -1
"""

fig1 = mpimg.imread('figures/Oates_1996_Fig_1.png')
fig2 = mpimg.imread('figures/Oates_1996_Fig_2.png')
fig3 = mpimg.imread('figures/Oates_1996_Fig_3.png')

fig = plt.figure(figsize=(8, 12))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

ax[0].imshow(fig1, extent=[0.0, 0.5, -5, 0.], aspect='auto')
ax[1].imshow(fig2, extent=[0.0, 0.5, -5.5, -3.5], aspect='auto')
ax[2].imshow(fig3, extent=[0.0, 0.5, 0, 0.15], aspect='auto')


wAB = -1.*R
T = 1.
p_As = np.linspace(0.001, 0.5, 101)
Es = np.empty_like(p_As)

ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=1.)

for i, p_A in enumerate(p_As):
    print(p_A)
    minimize_energy(T, p_A, ss)

    Es[i] = ss.molar_gibbs

ax[0].plot(p_As, Es/R)

# AB - AC
xs = np.linspace(0.001, 0.999, 101)
p_ABCC = np.empty_like(xs)
p_AABC = np.empty_like(xs)
p_B = np.empty_like(xs)

ss = CSAModel(cluster_energies=ternary_cluster_energies(wAB), gamma=1.)
for i, x in enumerate(xs):
    print(x)

    ss.set_state(T)
    ss.set_composition_from_p_s(np.array([0.5, x/2., (1.-x)/2.,
                                          0.5, x/2., (1.-x)/2.,
                                          0.5, x/2., (1.-x)/2.,
                                          0.5, x/2., (1.-x)/2.]))

    ss.equilibrate_clusters()

    Es[i] = ss.molar_gibbs

    # Find which cluster is AABC
    p_B[i] = (x/2.)
    p_ABCC[i] = ss.cluster_proportions[0,1,2,2]*12.
    p_AABC[i] = ss.cluster_proportions[0,0,1,2]*12.

ax[1].plot(p_B, Es/R)
ax[2].plot(p_B, p_ABCC)
ax[3].plot(p_B, p_AABC)

ax[0].set_xlabel('Mole fraction')
ax[0].set_ylabel('Free energy / kT')

ax[1].set_xlabel('Mole fraction B')
ax[1].set_ylabel('Free energy / kT')

ax[2].set_xlabel('Mole fraction B')
ax[2].set_ylabel('[ABCC] probability')

ax[3].set_xlabel('Mole fraction B')
ax[3].set_ylabel('[AABC] probability')
plt.show()
