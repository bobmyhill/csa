import numpy as np
from models.newmodel import NewModel, R


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

T = 0.8
ss.set_state(T)
ss.set_composition_from_p_ind(np.array([-0.6,  0.3,  0.15,  0.5,  0.65]))
ss.equilibrate_clusters()
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
