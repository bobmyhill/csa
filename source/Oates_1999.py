import numpy as np
from models.csasolutionmodel import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    u[1,1,1,1] = 0.

    u[0,1,1,1] = 3.*wAB*(1. + beta)
    u[1,0,1,1] = 3.*wAB*(1. + beta)
    u[1,1,0,1] = 3.*wAB*(1. + beta)
    u[1,1,1,0] = 3.*wAB*(1. + beta)

    u[0,0,1,1] = 4.*wAB
    u[0,1,0,1] = 4.*wAB
    u[0,1,1,0] = 4.*wAB
    u[1,0,0,1] = 4.*wAB
    u[1,0,1,0] = 4.*wAB
    u[1,1,0,0] = 4.*wAB

    u[0,0,0,1] = 3.*wAB*(1. + alpha)
    u[0,0,1,0] = 3.*wAB*(1. + alpha)
    u[0,1,0,0] = 3.*wAB*(1. + alpha)
    u[1,0,0,0] = 3.*wAB*(1. + alpha)

    u[0,0,0,0] = 0.
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

        #print(args)
        ss.equilibrate_clusters()
        return ss.molar_gibbs

    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: 3.-x[0]-x[1]-x[2]})

    sol = minimize(energy, [p_A, p_A, p_A], args=(T, p_A, ss),
                   method='SLSQP', constraints=cons)
    return sol


"""
Figure 1 from Oates et al. (1999)
wAB/RT = -1
"""

pfig1 = mpimg.imread('figures/Oates_1999_Fig_1.png')
pfig2 = mpimg.imread('figures/Oates_1999_Fig_2.png')
pfig3 = mpimg.imread('figures/Oates_1999_Fig_3.png')
pfig4 = mpimg.imread('figures/Oates_1999_Fig_4.png')
pfig5 = mpimg.imread('figures/Oates_1999_Fig_5.png')

fig1 = plt.figure(figsize=(16, 12))
ax1 = [fig1.add_subplot(2, 3, i) for i in range(1, 7)]

ax1[0].imshow(pfig1, extent=[0.0, 1.0, 0.3, 0.9], aspect='auto')
ax1[1].imshow(pfig2, extent=[0.6, 1.2, 0., 0.6], aspect='auto')
ax1[2].imshow(pfig3, extent=[0.0, 1.0, 0.3, 1.], aspect='auto')

ax1[4].imshow(pfig4, extent=[0.0, 1.0, 0.3, 1.0], aspect='auto')
ax1[5].imshow(pfig5, extent=[0.0, 1.0, 0.3, 1.0], aspect='auto')


reduced_temperatures = np.linspace(0.6, 1.2, 61)
Ss_disordered = np.empty_like(reduced_temperatures)
Ss_equilibrium = np.empty_like(reduced_temperatures)

for gamma in [1., 1.22]:
    wAB = -1.*R/gamma
    ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
                  site_species = [['A', 'B'], ['A', 'B'],
                                  ['A', 'B'], ['A', 'B']])

    for i, T in enumerate(reduced_temperatures):
        ss.equilibrate(composition={'A':2., 'B':2.}, temperature=T)
        Ss_equilibrium[i] = ss.molar_entropy

    ax1[1].plot(reduced_temperatures, Ss_equilibrium/R, label='equilibrium')


xs = np.linspace(0.15, 0.85, 401)
Gs_equilibrium = np.empty_like(xs)

ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
              site_species = [['A', 'B'], ['A', 'B'],
                              ['A', 'B'], ['A', 'B']])

for plti, alpha, beta, gamma in [(0, 0., 0., 1.),
                                 (2, 0., 0., 1.22),
                                 (4, 1., 0.92, 1.42)]:
    wAB = -1.*R/gamma
    ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
                  site_species = [['A', 'B'], ['A', 'B'],
                                  ['A', 'B'], ['A', 'B']])

    for T in [0.4, 0.6]:
        print(gamma, T)
        for i, x in enumerate(xs):
            try:
                ss.equilibrate(composition={'A': 4. * (1.-x), 'B': 4.*x},
                               temperature=T)
                Gs_equilibrium[i] = ss.molar_gibbs
            except:
                print('oh no, couldnae equilibrate. gonna have to fix this some time')
                Gs_equilibrium[i] = Gs_equilibrium[i-1] + 5.

        points = np.array([xs, Gs_equilibrium]).T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax1[3].plot(points[simplex, 0], points[simplex, 1], 'r--')

        starts = []
        ends = []
        n_vertices = len(hull.vertices)
        for i in range(1, n_vertices-1):
            if hull.vertices[i-1] < hull.vertices[i] - 1:
                starts.append(np.mean(points[hull.vertices[i-1]:hull.vertices[i-1]+2,0]))
                ends.append(np.mean(points[hull.vertices[i]-1:hull.vertices[i]+1,0]))

        ax1[plti].scatter(starts, np.ones(len(starts))*T)
        ax1[plti].scatter(ends, np.ones(len(ends))*T)



        ax1[3].plot(xs, Gs_equilibrium)

plt.show()
