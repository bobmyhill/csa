import numpy as np
import matplotlib.pyplot as plt

R = 1.
site_species_energies = np.array([-1., 1.,
                                  -1., 1.,
                                  1., -1.,
                                  1., -1.])*R

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

cluster_occupancies = np.array([[1., 0., 1., 0., 1., 0., 1., 0],
                                [1., 0., 1., 0., 1., 0., 0., 1],
                                [1., 0., 1., 0., 0., 1., 1., 0],
                                [1., 0., 1., 0., 0., 1., 0., 1],
                                [1., 0., 0., 1., 1., 0., 1., 0],
                                [1., 0., 0., 1., 1., 0., 0., 1],
                                [1., 0., 0., 1., 0., 1., 1., 0],
                                [1., 0., 0., 1., 0., 1., 0., 1],
                                [0., 1., 1., 0., 1., 0., 1., 0],
                                [0., 1., 1., 0., 1., 0., 0., 1],
                                [0., 1., 1., 0., 0., 1., 1., 0],
                                [0., 1., 1., 0., 0., 1., 0., 1],
                                [0., 1., 0., 1., 1., 0., 1., 0],
                                [0., 1., 0., 1., 1., 0., 0., 1],
                                [0., 1., 0., 1., 0., 1., 1., 0],
                                [0., 1., 0., 1., 0., 1., 0., 1]])

cluster_energies = (np.einsum('i, ki',
                              site_species_energies,
                              cluster_occupancies)
                    + np.einsum('ij, ij, ki, kj->k',
                                bond_energies,
                                (intercluster_connections
                                 + intracluster_connections),
                                cluster_occupancies,
                                cluster_occupancies)).reshape(2, 2, 2, 2)

W = np.einsum('ij, ij->ij',
              bond_energies,
              (intercluster_connections
               + intracluster_connections))

print(cluster_energies)

ps = np.array([0.6, 0.4, 0.6, 0.4, 0.3, 0.7, 0.5, 0.5])

E1 = np.einsum('ijkl, i, j, k, l', cluster_energies, *ps.reshape(4, 2))
E2 = (np.einsum('i, i', site_species_energies, ps) +
      np.einsum('ij, i, j', W, ps, ps))

print(E1, E2)
