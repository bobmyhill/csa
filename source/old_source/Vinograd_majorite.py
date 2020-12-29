import numpy as np
from models.csasolutionmodel import CSAModel, R
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


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

    u[0, 0, 1, 1] = 3.8*wAB
    u[0, 1, 0, 1] = 4.2*wAB
    u[0, 1, 1, 0] = 3.8*wAB
    u[1, 0, 0, 1] = 3.8*wAB
    u[1, 0, 1, 0] = 4.2*wAB
    u[1, 1, 0, 0] = 3.8*wAB

    u[0, 0, 0, 1] = 3.*wAB*(1. + alpha)
    u[0, 0, 1, 0] = 3.*wAB*(1. + alpha)
    u[0, 1, 0, 0] = 3.*wAB*(1. + alpha)
    u[1, 0, 0, 0] = 3.*wAB*(1. + alpha)

    u[0, 0, 0, 0] = 0.
    return u

def binary_cluster_energies(wAB, alpha=0., beta=0.):
    """
    Indices:
    0 = Mg
    1 = Si
    2 = Al

    Cluster pairs:
    AAAA = 0
    AAAB = 3 A-B pairs
    AABB = 4 A-B pairs
    ABBB = 3 A-B pairs
    BBBB = 0

    See Zhang et al., 2003
    """
    u = np.zeros((3, 3, 3, 3))
    u[1, 1, 1, 1] = 0.

    u[0, 1, 1, 1] = 3.*wAB*(1. + beta)
    u[1, 0, 1, 1] = 3.*wAB*(1. + beta)
    u[1, 1, 0, 1] = 3.*wAB*(1. + beta)
    u[1, 1, 1, 0] = 3.*wAB*(1. + beta)

    u[0, 0, 1, 1] = 3.8*wAB
    u[0, 1, 0, 1] = 4.2*wAB
    u[0, 1, 1, 0] = 3.8*wAB
    u[1, 0, 0, 1] = 3.8*wAB
    u[1, 0, 1, 0] = 4.2*wAB
    u[1, 1, 0, 0] = 3.8*wAB

    u[0, 0, 0, 1] = 3.*wAB*(1. + alpha)
    u[0, 0, 1, 0] = 3.*wAB*(1. + alpha)
    u[0, 1, 0, 0] = 3.*wAB*(1. + alpha)
    u[1, 0, 0, 0] = 3.*wAB*(1. + alpha)

    u[0, 0, 0, 0] = 0.
    return u




fig = plt.figure(figsize=(16, 12))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


reduced_temperatures = np.linspace(0.2, 2., 51)
Ss_disordered = np.empty_like(reduced_temperatures)
Ss_equilibrium = np.empty_like(reduced_temperatures)


"""
for lmda in [0.]:  # lmda is configuration independent, so has no effect at 50:50 composition
    for alpha in [0.]:
        for gamma in [1.]:
            print(gamma)
            interactions = np.array([[0., lmda/2.],
                                     [lmda/2., 0.]])

            ss = CSAModel(cluster_energies=binary_cluster_energies(wAB=-1.*R,
                                                                   alpha=alpha, beta=alpha),
                          gamma=gamma,
                          site_species=[['A', 'B'], ['A', 'B'],
                                        ['A', 'B'], ['A', 'B']],
                          compositional_interactions=interactions)

            for i, T in enumerate(reduced_temperatures):
                ss.equilibrate(composition={'A': 2., 'B': 2.}, temperature=T)
                Ss_equilibrium[i] = ss.molar_entropy

            Cp_equilibrium = reduced_temperatures*np.gradient(Ss_equilibrium, reduced_temperatures)

            ax[0].plot(reduced_temperatures, Ss_equilibrium,
                        label='$\\alpha$:{0} $\\gamma$: {1}'.format(alpha, gamma))
ax[0].legend()
"""

Vinograd_S = mpimg.imread('figures/Vinograd_pymaj_edited.png')

ax[1].imshow(Vinograd_S, extent=[0.0, 1.0, 0., 11.], aspect='auto')

gamma = 1.
ss = CSAModel(cluster_energies=binary_cluster_energies(wAB=-1.*R),
              gamma=gamma,
              site_species=[['Mg', 'Si', 'Al'], ['Mg', 'Si', 'Al'],
                            ['Mg', 'Si', 'Al'], ['Mg', 'Si', 'Al']])

x_majs = np.linspace(0.01, 0.99, 25)
Ss_equilibrium = np.empty_like(x_majs)

ideal_entropy = (x_majs/2. * np.log(x_majs / 2.)
                 + x_majs/2. * np.log(x_majs / 2.)
                 + (1. - x_majs) * np.log(1. - x_majs))*R*-1.
ax[1].plot(x_majs, ideal_entropy)


for T in [0.5, 1., 1.5, 2., 2.5]:
    for i, x in enumerate(x_majs):
        print(x)
        composition = {'Mg': 2.*x, 'Si': 2.*x, 'Al': 4.*(1. - x)}
        ss.equilibrate(composition=composition, temperature=T)
        Ss_equilibrium[i] = ss.molar_entropy

    ax[1].plot(x_majs, Ss_equilibrium,
               label='{0} K'.format(T))

ax[1].legend()

plt.show()
