import numpy as np
from collections import Counter
from itertools import combinations, permutations, product
from sympy.utilities.iterables import multiset_permutations
import matplotlib.pyplot as plt

import sys
outfile = sys.argv[0].split('.')[0] + '.dat'
# 0-1 is in x-y plane
# 0-2 is in y-z
# 0-3 is in x-z

# 1-2 is in x-z plane
# 1-3 is in y-z plane

# 2-3 is in x-y plane

# if we rotate by 180 degrees around z axis, 1 becomes 0, 2 becomes 3
# so now we can copy 0-1, 0-2 and 0-3 to 1-0, 1-3 and 1-2
# (taking the negative of the cluster shifts in the x and y axes)

# if we rotate by 180 degrees around x axis, 2 becomes 0, 1 becomes 3
# so now we can copy 0-1, 0-2 and 0-3 to 2-3, 2-0 and 2-1
# (taking the negative of the cluster shifts in the y and z axes)

# if we rotate by 180 degrees around y axis, 3 becomes 0, 1 becomes 2
# so now we can copy 0-1, 0-2 and 0-3 to 3-2, 3-1 and 3-0
# (taking the negative of the cluster shifts in the x and z axes)

bond_connectivities = np.array([[0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 2, 0, 0, 0],
                       [0, 2, 0, 1, 0],
                       [0, 2, 0, 0, 1],
                       [0, 2, 0, 1, 1],
                       [0, 3, 0, 0, 0],
                       [0, 3, 1, 0, 0],
                       [0, 3, 0, 0, 1],
                       [0, 3, 1, 0, 1],

                       [1, 0, 0, 0, 0],
                       [1, 0, -1, 0, 0],
                       [1, 0, 0, -1, 0],
                       [1, 0, -1, -1, 0],
                       [1, 3, 0, 0, 0],
                       [1, 3, 0, -1, 0],
                       [1, 3, 0, 0, 1],
                       [1, 3, 0, -1, 1],
                       [1, 2, 0, 0, 0],
                       [1, 2, -1, 0, 0],
                       [1, 2, 0, 0, 1],
                       [1, 2, -1, 0, 1],

                       [2, 3, 0, 0, 0],
                       [2, 3, 1, 0, 0],
                       [2, 3, 0, -1, 0],
                       [2, 3, 1, -1, 0],
                       [2, 0, 0, 0, 0],
                       [2, 0, 0, -1, 0],
                       [2, 0, 0, 0, -1],
                       [2, 0, 0, -1, -1],
                       [2, 1, 0, 0, 0],
                       [2, 1, 1, 0, 0],
                       [2, 1, 0, 0, -1],
                       [2, 1, 1, 0, -1],

                       [3, 2, 0, 0, 0],
                       [3, 2, -1, 0, 0],
                       [3, 2, 0, 1, 0],
                       [3, 2, -1, 1, 0],
                       [3, 1, 0, 0, 0],
                       [3, 1, 0, 1, 0],
                       [3, 1, 0, 0, -1],
                       [3, 1, 0, 1, -1],
                       [3, 0, 0, 0, 0],
                       [3, 0, -1, 0, 0],
                       [3, 0, 0, 0, -1],
                       [3, 0, -1, 0, -1]])

# occupancies (there will be 70^4 = 24.01 million of these!)

c0 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
pc0 = np.array(list(multiset_permutations(c0))).reshape(70, 2, 2, 2)


c1 = np.array([2, 2, 2, 2, 3, 3, 3, 3])
pc1 = np.array(list(multiset_permutations(c1))).reshape(70, 2, 2, 2)

c2 = np.array([4, 4, 4, 4, 5, 5, 5, 5])
pc2 = np.array(list(multiset_permutations(c2))).reshape(70, 2, 2, 2)

c3 = np.array([6, 6, 6, 6, 7, 7, 7, 7])
pc3 = np.array(list(multiset_permutations(c3))).reshape(70, 2, 2, 2)

bond_energies = np.array([[0., 1., 2., 3., 4., 6., 7., 5.],
                          [0., 0., 7., 7., 4., 4., 3., 4.],
                          [0., 0., 0., 1., 1., 6., 3., 6.],
                          [0., 0., 0., 0., 2., 7., 7., 8.],
                          [0., 0., 0., 0., 0., 2., 5., 3.],
                          [0., 0., 0., 0., 0., 0., 3., 4.],
                          [0., 0., 0., 0., 0., 0., 0., 2.],
                          [0., 0., 0., 0., 0., 0., 0., 0.]])



xc = list(range(2))
clusters = np.array(list(product(xc, xc, xc)))

cls = product(clusters, bond_connectivities)
bonds = []
for cl in cls:
    bonds.append([[cl[1][0], cl[0][0], cl[0][1], cl[0][2]],
                  [cl[1][1], (cl[0][0]+cl[1][2])%2, (cl[0][1]+cl[1][3])%2, (cl[0][2]+cl[1][4])%2]])

bonds = np.array(bonds, dtype='int')

energies = Counter()
for i, d in enumerate(product(pc0, pc1, pc2, pc3)):
    if i%1000 == 0:
        print(i)
    d = np.array(d)
    e = 0.
    for bond in bonds:
        ss = sorted([d[bond[0][0], bond[0][1], bond[0][2], bond[0][3]],
                     d[bond[1][0], bond[1][1], bond[1][2], bond[1][3]]])
        e+=bond_energies[ss[0], ss[1]]

    e_str = '{0:.6f}'.format(e/2.)
    energies[e_str] += 1

    if i%25000 == 0:
        keys = sorted(list(energies.keys()))
        values = [energies[key] for key in keys]
        energy_levels_and_degeneracies = np.array([keys, values]).T
        np.savetxt(outfile, energy_levels_and_degeneracies, delimiter=" ", fmt="%s")

keys = sorted(list(energies.keys()))
values = [energies[key] for key in keys]
energy_levels_and_degeneracies = np.array([keys, values]).T
np.savetxt(outfile, energy_levels_and_degeneracies, delimiter=" ", fmt="%s")

plt.scatter(energy_levels_and_degeneracies[:,0], energy_levels_and_degeneracies[:,1])
plt.show()
