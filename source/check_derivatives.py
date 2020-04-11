import numpy as np
from models.csasolutionmodel import CSAModel, R
# import matplotlib.pyplot as plt
from numpy.linalg import solve, pinv

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


gamma = 1.22
wAB = -1.*R/gamma
ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=gamma,
              site_species=[['A', 'B'], ['A', 'B'],
                            ['A', 'B'], ['A', 'B']])

T = 0.8
ss.set_state(T)
ss.set_composition_from_p_ind(np.array([-0.6,  0.3,  0.15,  0.5,  0.65]))
ss.equilibrate_clusters()

G0 = ss.molar_gibbs
S0 = ss.molar_entropy
JG = ss.molar_chemical_potentials
JS = ss.partial_molar_entropies
Jp = np.einsum('ij, i -> ij', ss.A_ind, ss.cluster_proportions_flat)
c0 = ss.c

p0 = ss.p_ind
pc0 = ss.cluster_proportions_flat
D0 = ss._dpcl_dp_ind()

dp_ind_dc0 = np.einsum('ij, ik, i -> jk',
                       ss.A_ind, ss.A_ind, pc0)

Minv0 = pinv(dp_ind_dc0)

def _dpcl_dp_ind(p_cl):
    dp_ind_dc = np.einsum('ijk, i -> kj', ss._AA, p_cl)
    dpcl_dc = np.einsum('ij, i -> ij', ss.A_ind, p_cl)

    # the returned solve is equivalent to
    # np.einsum('lj, jk -> lk', dpcl_dc, pinv(dp_ind_dc)))
    return solve(dp_ind_dc.T, dpcl_dc.T).T


print('Gibbs, entropy:')
print(G0, ss.molar_chemical_potentials.dot(ss.p_ind))
print(S0, ss.partial_molar_entropies.dot(ss.p_ind))

print('\nChemical potentials, partial entropy (analytical)')
print(JG)
print(JS)

print(ss.hessian_entropy)
print(ss.hessian_gibbs)

dp_ind = np.identity(5)*1.e-8

mu = np.empty(5)
partial_S = np.empty(5)
hess_G = np.empty((5, 5))
hess_S = np.empty((5, 5))
dcdpind = np.empty((5, 5))

for i, dp in enumerate(dp_ind):

    ss.set_composition_from_p_ind((p0 + dp)/np.sum(p0 + dp))
    ss.equilibrate_clusters()

    mu[i] = (ss.molar_gibbs*np.sum(p0 + dp)-G0)/1.e-8
    partial_S[i] = (ss.molar_entropy*np.sum(p0 + dp)-S0)/1.e-8

    hess_G[i] = (ss.molar_chemical_potentials - JG)/1.e-8
    hess_S[i] = (ss.partial_molar_entropies - JS)/1.e-8


print(hess_G)
exit()

c0 = ss.c
dcs = np.identity(5)*1.e-8

dpc_dcdc = np.empty((16, 5, 5))
dpind_dcdc = np.empty((5, 5, 5))
dDdc = np.empty((16, 5, 5))

pc0 = ss._cluster_proportions(c0, T)
dpc_dc0 = np.einsum('ij, i -> ij', ss.A_ind, pc0)
dpind_dc0 = np.einsum('ijk, i -> kj', ss._AA, pc0)

dMinvdc = np.empty((5, 5, 5))

for i, dc in enumerate(dcs):
    pc1 = ss._cluster_proportions(c0+dc, T)

    D1 = _dpcl_dp_ind(pc1)


    dp_ind_dc1 = np.einsum('ij, ik, i -> jk',
                           ss.A_ind, ss.A_ind, pc1)
    Minv1 = pinv(dp_ind_dc1)

    dDdc[:, :, i] = (D1 - D0)/1.e-8

    dMinvdc[:, :, i] = (Minv1 - Minv0)/1.e-8
    """
    dpc_dc1 = np.einsum('ij, i -> ij', ss.A_ind, pc1)
    dpc_dcdc[:, :, i] = (dpc_dc1 - dpc_dc0)/1.e-8

    dpind_dc1 = np.einsum('ijk, i -> kj', ss._AA, pc1)
    dpind_dcdc[:, :, i] = (dpind_dc1 - dpind_dc0)/1.e-8
    """

dp_ind_dc = np.einsum('ij, ik, i -> jk',
                      ss.A_ind, ss.A_ind, pc0)
dpcl_dc = np.einsum('ij, i -> ij',
                    ss.A_ind, pc0)
dpcl_dcdc = np.einsum('in, ij, i -> ijn',
                      ss.A_ind, ss.A_ind, pc0)

Minv = pinv(dp_ind_dc)
dMdc = np.einsum('il, ij, ik, i -> jkl',
                 ss.A_ind, ss.A_ind, ss.A_ind, pc0)



print(dDdc.shape)
print(dDdc[1,0])

dDdc = (np.einsum('imn, mk -> ikn', dpcl_dcdc, Minv)
        - np.einsum('ij, jl, mln, mk-> ikn',
                    dpcl_dc, Minv, dMdc, Minv))
print(dDdc[1,0])

dpcl_dc = np.einsum('ij, i -> ij', ss.A_ind, pc0)
dp_ind_dc = np.einsum('ijk, i -> jk', ss._AA, pc0)
dpcl_dcdc = np.einsum('ijn, i -> ijn', ss._AA, pc0)
dMdc = np.einsum('il, ij, ik, i -> jkl', ss.A_ind, ss.A_ind, ss.A_ind, pc0)

Minv = pinv(dp_ind_dc)

D = np.einsum('lj, jk -> lk', dpcl_dc, Minv)
E = dpcl_dcdc - np.einsum('il, mln-> imn', D, dMdc)
F = np.einsum('imn, mk, np -> ikp', E, Minv, Minv)

print(F[0,0])
print(F[0,:,0])
"""
Dlk = dpcldc_lj pinv(dp_ind_dc)_jk

D_lk = (A_lj pcl_j) (A_ij A_ik pcl_i)^+
"""


"""
print(dpc_dcdc[:,0,:])
print(np.einsum('ikl, i -> ikl', ss._AA, pc0)[0])


print(dpind_dcdc[:,0,:])
print(np.einsum('ij, ik, il, i -> jkl', ss.A_ind, ss.A_ind, ss.A_ind, pc0)[0])
"""


# print('\nChemical potentials, partial entropy (numerical)')
# print(mu)
# print(partial_S)

print('\n Hessians (numerical)')
print(hess_G)
#print(hess_G)
