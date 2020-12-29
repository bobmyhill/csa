import numpy as np
from models.newmodel import NewModel, R
from models.csasolutionmodel import CSAModel
import matplotlib.pyplot as plt

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


gamma = 1.22*4.
wAB = -1.*R*4.
ss = NewModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
              site_species=[['A', 'B'], ['A', 'B'],
                            ['A', 'B'], ['A', 'B']])


gamma = 1.22
wAB = -1.*R
ss2 = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
               site_species=[['A', 'B'], ['A', 'B'],
                             ['A', 'B'], ['A', 'B']])


fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

fs = np.linspace(0.01, 0.99, 101)
Ss1 = np.empty_like(fs)
Ss2 = np.empty_like(fs)
Ss3 = np.empty_like(fs)
Ss4 = np.empty_like(fs)
Gs1 = np.empty_like(fs)
Gs2 = np.empty_like(fs)
for T in [0.1, 4.0]:
    ss.set_state(T)
    for i, f in enumerate(fs):
        ss.set_composition_from_p_s(np.array([f, 1.-f,
                                              1.-f, f,
                                              f, 1.-f,
                                              1.-f, f]))  # ABAB-BABA

        ss.equilibrate_clusters()
        Ss1[i] = ss.molar_entropy
        Gs1[i] = ss.molar_gibbs

        ss.set_composition_from_p_s(np.array([f, 1.-f,
                                              1.-f, f,
                                              1.-f, f,
                                              1.-f, f]))  # ABBB - BAAA

        ss.equilibrate_clusters()
        Ss2[i] = ss.molar_entropy


        ss.set_composition_from_p_s(np.array([f, 1.-f,
                                              1.-f, f,
                                              0.001, 0.999,
                                              0.999, 0.001]))  # ABAB-BAAB

        ss.equilibrate_clusters()
        Ss3[i] = ss.molar_entropy

        ss.set_composition_from_p_s(np.array([f, 1.-f,
                                              1.-f, f,
                                              0.999, 0.001,
                                              0.999, 0.001]))  # ABBB - BABB

        ss.equilibrate_clusters()
        Ss4[i] = ss.molar_entropy

    ax[0].plot(fs, Ss1, label='ABAB-BABA, T={0}'.format(T))
    ax[0].plot(fs, Ss2, label='ABBB-BAAA, T={0}'.format(T))
    ax[1].plot(fs, Ss3, label='ABAB-BAAB, T={0}'.format(T))
    ax[1].plot(fs, Ss4, label='ABBB-BABB, T={0}'.format(T))
ax[0].legend()
ax[1].legend()

temperatures = np.linspace(0.05, 2.0, 41)
fs = np.linspace(0.94, 1., 101)
for T in temperatures:
    ss.set_state(T)
    ss2.set_state(T)
    for i, f in enumerate(fs):
        ss.set_composition_from_p_s(np.array([f, 1.-f,
                                              1.-f, f,
                                              f, 1.-f,
                                              1.-f, f]))  # ABAB-BABA

        ss2.set_composition_from_p_s(np.array([f, 1.-f,
                                               1.-f, f,
                                               f, 1.-f,
                                               1.-f, f]))  # ABAB-BABA

        ss.equilibrate_clusters()
        ss2.equilibrate_clusters()
        Gs1[i] = ss.molar_gibbs
        Gs2[i] = ss2.molar_gibbs
    ax[2].plot(fs, Gs1, label='ABAB-BABA, T={0}'.format(T))
    ax[3].plot(fs, Gs2, label='ABAB-BABA, T={0}'.format(T))

ax[2].set_ylim(-36, -32)
ax[3].set_ylim(-36, -32)
plt.show()

exit()
p0 = ss.p_ind


G0 = ss.molar_gibbs
S0 = ss.molar_entropy
JG = ss.molar_chemical_potentials
JS = ss.partial_molar_entropies
HG = ss.hessian_gibbs
HS = ss.hessian_entropy

c0 = ss.c

p0 = ss.p_ind
pc0 = ss.cluster_proportions_flat

dp_ind = np.identity(5)*1.e-8

mu = np.empty(5)
partial_S = np.empty(5)
hess_G = np.empty((5, 5))
hess_S = np.empty((5, 5))

for i, dp in enumerate(dp_ind):

    ss.set_composition_from_p_ind((p0 + dp)/np.sum(p0 + dp))
    ss.equilibrate_clusters()

    mu[i] = (ss.molar_gibbs*np.sum(p0 + dp)-G0)/1.e-8
    partial_S[i] = (ss.molar_entropy*np.sum(p0 + dp)-S0)/1.e-8

    hess_G[i] = (ss.molar_chemical_potentials - JG)/1.e-8
    hess_S[i] = (ss.partial_molar_entropies - JS)/1.e-8


print('Gibbs, entropy (direct and from partials):')
print(G0, ss.molar_chemical_potentials.dot(ss.p_ind))
print(S0, ss.partial_molar_entropies.dot(ss.p_ind))

print('\nChemical potentials, partial entropy (analytical)')
print(JG)
print(JS)

print('\nChemical potentials, partial entropy (numerical)')
print(mu)
print(partial_S)

print('\n Hessians (analytical)')
print(HG)
print(HS)

print('\n Hessians (numerical)')
print(hess_G)
print(hess_S)
