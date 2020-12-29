import numpy as np
from models.cmfsolutionmodel import R, CMFModel
import matplotlib.pyplot as plt


# The following matrices describe the FCC cluster model with
# 4 sites per cluster

# The A-B bond energies are equal to -R
site_species_energies = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

bond_energies = np.array([[0., 0., 0., -1., 0., -1., 0., -1.],
                          [0., 0., -1., 0., -1., 0., -1., 0.],
                          [0., -1., 0., 0., 0., -1., 0., -1.],
                          [-1., 0., 0., 0., -1., 0., -1., 0.],
                          [0., -1., 0., -1., 0., 0., 0., -1.],
                          [-1., 0., -1., 0., 0., 0., -1., 0.],
                          [0., -1., 0., -1., 0., -1., 0., 0.],
                          [-1., 0., -1., 0., -1., 0., 0., 0.]])*R

intercluster_connections = np.array([[0., 0., 3., 3., 3., 3., 3., 3.],
                                     [0., 0., 3., 3., 3., 3., 3., 3.],
                                     [3., 3., 0., 0., 3., 3., 3., 3.],
                                     [3., 3., 0., 0., 3., 3., 3., 3.],
                                     [3., 3., 3., 3., 0., 0., 3., 3.],
                                     [3., 3., 3., 3., 0., 0., 3., 3.],
                                     [3., 3., 3., 3., 3., 3., 0., 0.],
                                     [3., 3., 3., 3., 3., 3., 0., 0.]])/2.

intracluster_connections = np.array([[0., 0., 1., 1., 1., 1., 1., 1.],
                                     [0., 0., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 0., 0., 1., 1., 1., 1.],
                                     [1., 1., 0., 0., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 0., 0., 1., 1.],
                                     [1., 1., 1., 1., 0., 0., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 1., 1., 0., 0.]])/2.

site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]

ss = CMFModel(site_species=site_species,
              site_species_energies=site_species_energies,
              bond_energies=bond_energies,
              intracluster_connections=intracluster_connections,
              intercluster_connections=intercluster_connections)

T = 0.8
ss.set_state(T)
ss.set_composition_from_p_ind(np.array([-0.6,  0.3,  0.15,  0.5,  0.65]))
ss.equilibrate_clusters()
p0 = ss.p_ind


F0 = ss.molar_helmholtz
S0 = ss.molar_entropy
JF = ss.molar_chemical_potentials
JS = ss.partial_molar_entropies
#HF = ss.hessian_helmholtz
#HS = ss.hessian_entropy

c0 = ss.c

p0 = ss.p_ind
pc0 = ss.cluster_proportions

dp_ind = np.identity(5)*1.e-8

mu = np.empty(5)
partial_S = np.empty(5)
hess_F = np.empty((5, 5))
hess_S = np.empty((5, 5))

for i, dp in enumerate(dp_ind):

    ss.set_composition_from_p_ind((p0 + dp)/np.sum(p0 + dp))
    ss.equilibrate_clusters()

    mu[i] = (ss.molar_helmholtz*np.sum(p0 + dp)-F0)/1.e-8
    partial_S[i] = (ss.molar_entropy*np.sum(p0 + dp)-S0)/1.e-8

    hess_F[i] = (ss.molar_chemical_potentials - JF)/1.e-8
    hess_S[i] = (ss.partial_molar_entropies - JS)/1.e-8


print('Helmholtz, entropy (direct and from partials):')
print(F0, ss.molar_chemical_potentials.dot(ss.p_ind))
print(S0, ss.partial_molar_entropies.dot(ss.p_ind))

print('\nChemical potentials, partial entropy (analytical)')
print(JF)
print(JS)

print('\nChemical potentials, partial entropy (numerical)')
print(mu)
print(partial_S)

"""
print('\n Hessians (analytical)')
print(HF)
print(HS)

print('\n Hessians (numerical)')
print(hess_F)
print(hess_S)
"""
