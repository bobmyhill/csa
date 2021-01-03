import numpy as np
from models.doublecellmodel import R, DoubleCellModel


# The following matrices describe the FCC cluster model with
# 4 sites per cluster

# The A-B bond energies are equal to -R
site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]
site_species_energies = np.array([-1.,  1., -1.,  1., 1., -1., 1., -1.])*R
site_species_energies = np.array([0., 0., 0., 0., 0., 0., 0., 0.])*R

bond_energies = np.array([[-0.,  0.,  0., -1.,  0., -1.,  0., -1.],
                          [-0.,  0., -1.,  0., -1.,  0., -1.,  0.],
                          [-0., -1.,  0.,  0.,  0., -1.,  0., -1.],
                          [-1.,  0.,  0.,  0., -1.,  0., -1.,  0.],
                          [-0., -1.,  0., -1.,  0.,  0.,  0., -1.],
                          [-1.,  0., -1.,  0.,  0.,  0., -1.,  0.],
                          [-0., -1.,  0., -1.,  0., -1.,  0.,  0.],
                          [-1.,  0., -1.,  0., -1.,  0.,  0.,  0.]])*R

cluster_bonds_per_cluster = np.array([[0., 0., 4., 4., 4., 4., 4., 4.],
                                      [0., 0., 4., 4., 4., 4., 4., 4.],
                                      [4., 4., 0., 0., 4., 4., 4., 4.],
                                      [4., 4., 0., 0., 4., 4., 4., 4.],
                                      [4., 4., 4., 4., 0., 0., 4., 4.],
                                      [4., 4., 4., 4., 0., 0., 4., 4.],
                                      [4., 4., 4., 4., 4., 4., 0., 0.],
                                      [4., 4., 4., 4., 4., 4., 0., 0.]])/2.

# equivalent to external 1*A + 2.*((1-f)*B + f*MF)
mf = 0.36
f_A = (1. + 1.)/4.
f_B = 2.*(1. - mf) / 4.
f_MF = 2.*mf / 4.


# 1*A + 1*B + 1*MF is reasonable.
# 1*A + 0.5*B + 1.5*MF is not

# how about A = B, f = MF/(MF+B), A + B + MF = 1.
# then
# 1 + A/MF = 1/f
# A = (1 - MF)/2.
# 1 + 1/(2*MF) - 1/2 = 1/f
# 1 + 1/MF = 2/f
# 1/MF = 2/f - (f/f)
# MF = f/(2 - f)


# mf = 0. is a completely ordered set of compounds. (2*A, 2*B)
# mf = 1. is the same as the point approximation
mf = 0.36
f_MF = mf / (2. - mf)
f_A = (1. - f_MF)/2.
f_B = (1. - f_MF)/2.
print(f_A*4., f_B*4., f_MF*4.)

"""
# how about B = 0, f = 1, A + B + MF = 1.
# then
f_MF = 0.5
f_A = (1. - f_MF)
f_B = 0.
print(f_A*4.-1., f_B*4., f_MF*4.)
# I don't think you can just add an arbitrary fraction of B to this.
# You end up getting two heat capacity peaks.
"""

internal_bonds_per_cluster = np.copy(cluster_bonds_per_cluster)*f_A
bridging_bonds_per_cluster = np.copy(cluster_bonds_per_cluster)*(f_B + f_MF)


ss = DoubleCellModel(site_species=site_species,
                     site_species_energies=site_species_energies,
                     bond_energies=bond_energies,
                     internal_bonds_per_cluster=internal_bonds_per_cluster,
                     bridging_bonds_per_cluster=bridging_bonds_per_cluster,
                     meanfield_fraction=mf)

T = 0.8
ss.set_state(T)
ss.set_composition_from_p_ind(np.array([-0.6,  0.3,  0.15,  0.5,  0.65]))
ss.equilibrate_clusters()
p0 = ss.p_ind


E0 = ss.molar_internal_energy
S0 = ss.molar_entropy
JE = ss.partial_molar_energies
JS = ss.partial_molar_entropies
HE = ss.hessian_energy
HS = ss.hessian_entropy

c0 = ss.c

p0 = ss.p_ind
pc0 = ss.cluster_proportions

eps = 1.e-8

dp_ind = np.identity(5)*eps

mu = np.empty(5)
partial_S = np.empty(5)
hess_E = np.empty((5, 5))
hess_S = np.empty((5, 5))

for i, dp in enumerate(dp_ind):

    ss.set_composition_from_p_ind((p0 + dp)/np.sum(p0 + dp))
    ss.equilibrate_clusters()

    mu[i] = (ss.molar_internal_energy*np.sum(p0 + dp)-E0)/eps
    partial_S[i] = (ss.molar_entropy*np.sum(p0 + dp)-S0)/eps

    hess_E[i] = (ss.partial_molar_energies - JE)/eps
    hess_S[i] = (ss.partial_molar_entropies - JS)/eps


print('Energy (direct and from partials):')
print(E0)
print(ss.partial_molar_energies.dot(ss.p_ind))

print('Entropy (direct and from partials):')
print(S0)
print(ss.partial_molar_entropies.dot(ss.p_ind))

print('\nPartial energy (analytical and numerical)')
print(JE)
print(mu)

print('\nPartial entropy (analytical and numerical)')
print(JS)
print(partial_S)

print('\n Energy hessians (analytical and numerical)')
print(HE)
print(hess_E)

print('\n Entropy hessians (analytical and numerical)')
print(HS)
print(hess_S)
