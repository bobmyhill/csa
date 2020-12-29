import numpy as np
import matplotlib.pyplot as plt

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

# Atoms 0-1-2-3
atom_positions = np.array([[0., 0., 0.],
                           [0.5, 0.5, 0.],
                           [0., 0.5, 0.5],
                           [0.5, 0., 0.5]])

bond_starts = np.array([atom_positions[b[0]]
                        for b in bond_connectivities])
bond_ends = np.array([atom_positions[b[1]] + b[2:]
                      for b in bond_connectivities])

"""

bond_starts = np.array([[0., 0., 0.] for i in range(12)])

bond_ends = np.array([[0.5, 0.5, 0.],
                     [-0.5, 0.5, 0.],
                     [0.5, -0.5, 0.],
                     [-0.5, -0.5, 0.],
                     [0.5, 0., 0.5],
                     [-0.5, 0., 0.5],
                     [0.5, 0., -0.5],
                     [-0.5, 0., -0.5],
                     [0., 0.5, 0.5],
                     [0., -0.5, 0.5],
                     [0., 0.5, -0.5],
                     [0., -0.5, -0.5]])
"""

bonds0 = np.hstack((bond_starts, bond_ends))
bonds = np.hstack((bond_starts, bond_ends))

xshift = np.array([1., 0., 0., 1., 0., 0.])
yshift = np.array([0., 1., 0., 0., 1., 0.])
zshift = np.array([0., 0., 1., 0., 0., 1.])


for fx in np.linspace(-4., 10., 15):
    for fy in np.linspace(-4., 10., 15):
        for fz in np.linspace(-4., 10., 15):
            if ((np.abs(fx) > 0.005 or np.abs(fy) > 0.005
                 or np.abs(fz) > 0.005)):
                bonds = np.vstack((bonds,
                                   bonds0
                                   + fx*xshift
                                   + fy*yshift
                                   + fz*zshift))


start = -3.25
data = []
for size in range(1, 11):
    cell_bounds = np.array([start, start + size])

    in_box_start = [((bonds[i, 0] > cell_bounds[0])
                     and (bonds[i, 0] < cell_bounds[1])
                    and (bonds[i, 1] > cell_bounds[0])
                    and (bonds[i, 1] < cell_bounds[1])
                    and (bonds[i, 2] > cell_bounds[0])
                    and (bonds[i, 2] < cell_bounds[1]))
                    for i in range(len(bonds))]
    in_box_end = [((bonds[i, 3] > cell_bounds[0])
                   and (bonds[i, 3] < cell_bounds[1])
                   and (bonds[i, 4] > cell_bounds[0])
                   and (bonds[i, 4] < cell_bounds[1])
                   and (bonds[i, 5] > cell_bounds[0])
                   and (bonds[i, 5] < cell_bounds[1]))
                  for i in range(len(bonds))]

    interior = [(in_box_start[i] and in_box_end[i])
                for i in range(len(bonds))]
    bridge = [(in_box_start[i] ^ in_box_end[i])
              for i in range(len(bonds))]

    # The connectivity above is set up to double bond counting
    # We must therefore halve the interior bonds, and quarter
    # the bridge bonds (which are shared by an exterior atom)
    n_interior_bonds = np.sum(interior)/2.
    n_bridge_bonds = np.sum(bridge)/4.

    n_atoms = 4.*size**3
    data.append([size, n_atoms, n_interior_bonds, n_bridge_bonds])

sizes, n_atoms, n_interior_bonds, n_bridge_bonds = np.array(data).T
n_total_bonds = n_interior_bonds + n_bridge_bonds
# plt.plot(sizes, n_interior_bonds/n_atoms)
# plt.plot(sizes, n_bridge_bonds/n_atoms)
# plt.plot(n_interior_bonds/n_atoms, n_bridge_bonds/n_atoms)

s = np.linspace(1., 10., 901)

plt.plot(sizes, n_total_bonds/n_total_bonds, color='black', label='total')

f_bridge_0 = 3./4.
plt.scatter(sizes, n_interior_bonds/n_total_bonds,
            color='blue', label='interior')
plt.plot(s, (1. - f_bridge_0*(1./s)), linestyle=':',
         color='blue', label='interior (model)')

plt.scatter(sizes, n_bridge_bonds/n_total_bonds,
            color='red', label='bridging')
plt.plot(s, f_bridge_0*(1./s), linestyle=':',
         color='red', label='bridging (model)')

plt.legend()
plt.xlabel('cube size length (n clusters)')
plt.ylabel('fraction bonds')

logplot = False
if logplot:
    plt.xscale("log")
    plt.yscale("log")

plt.savefig('bond_proportions.pdf')
plt.show()
