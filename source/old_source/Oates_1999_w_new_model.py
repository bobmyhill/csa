import numpy as np
from models.newmodel import NewModel, R
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

ax1[4].imshow(pfig4, extent=[0.0, 1.0, 1.08*3./7., 1.08], aspect='auto')
ax1[5].imshow(pfig5, extent=[0.0, 1.0, 1.08*3./7., 1.08], aspect='auto')


reduced_temperatures = np.linspace(0.6, 1.2, 61)
Ss_disordered = np.empty_like(reduced_temperatures)
Ss_equilibrium = np.empty_like(reduced_temperatures)


for gamma in [1.*4., 1.22*4.]:
    print(gamma)
    ss = NewModel(cluster_energies=binary_cluster_energies(wAB=-R*4.),
                  gamma=gamma,
                  site_species=[['A', 'B'], ['A', 'B'],
                                ['A', 'B'], ['A', 'B']])

    for i, T in enumerate(reduced_temperatures):
        ss.equilibrate(composition={'A': 2., 'B': 2.}, temperature=T)
        Ss_equilibrium[i] = ss.molar_entropy

        print(i, ss.molar_entropy)

    ax1[1].plot(reduced_temperatures, Ss_equilibrium/R, label='equilibrium')

plt.show()
exit()
xs = np.linspace(0.01, 0.99, 201)

for plti, alpha, beta, gamma, lmda in [(0, 0., 0., 1., 0.),
                                       (2, 0., 0., 1.22, 0.),
                                       (4, 0., -0.08, 1.42, 0.),
                                       (5, 0., -0.08, 1.42, 10.2)]:

    interactions = np.array([[0., lmda/4.],
                             [lmda/4., 0.]])

    ss = NewModel(cluster_energies=binary_cluster_energies(wAB=-R,
                                                           alpha=alpha,
                                                           beta=beta),
                  gamma=gamma,
                  site_species=[['A', 'B'], ['A', 'B'],
                                ['A', 'B'], ['A', 'B']],
                  compositional_interactions=interactions)

    for T in [0.5, 0.7, 0.9]:
        print('\n{0} {1}'.format(gamma, T))
        xs_equilibrium = []
        Gs_equilibrium = []
        for i, x in enumerate(xs):
            print('{0} / {1}'.format(i+1, len(xs)), end="\r")

            try:
                ss.equilibrate(composition={'A': 4. * (1.-x), 'B': 4.*x},
                               temperature=T)
                xs_equilibrium.append(x)
                Gs_equilibrium.append(ss.molar_gibbs)
            except Exception as err:
                print(err)
                print('oh no, couldnae equilibrate at x={0}. '
                      'Gonna have to fix this some time'.format(x))

        points = np.array([xs_equilibrium, Gs_equilibrium]).T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax1[3].plot(points[simplex, 0], points[simplex, 1], 'r--')

        v_starts = []
        v_ends = []
        n_vertices = len(hull.vertices)
        for i in range(1, n_vertices-1):
            if hull.vertices[i-1] < hull.vertices[i] - 1:
                starts = points[hull.vertices[i-1]:hull.vertices[i-1]+2, 0]
                v_starts.append(np.mean(starts))
                ends = points[hull.vertices[i]-1:hull.vertices[i]+1, 0]
                v_ends.append(np.mean(ends))

        ax1[plti].scatter(v_starts, np.ones(len(v_starts))*T)
        ax1[plti].scatter(v_ends, np.ones(len(v_ends))*T)

        if T == 0.4:
            ax1[3].plot(xs_equilibrium, Gs_equilibrium, label=plti)

ax1[3].legend()
fig1.savefig('Oates_1999_benchmarks.pdf')
plt.show()
