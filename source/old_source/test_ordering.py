import numpy as np
import matplotlib.pyplot as plt


# Endmembers
# AAAA, AAAB, AABB, ABBB, BBBB


E = np.array([[0., -3.],
              [-5., -4.]])

fs = np.linspace(0., 1., 101)
ps1 = np.array([fs, 1.-fs])
ps2 = np.array([1.-fs, fs])

Es1 = np.einsum('ij, ik, jk->k', E, ps1, ps2)

#plt.plot(fs, Es)


E = np.array([[0., -3.],
              [-3., -4.]])

fs = np.linspace(0., 1., 101)
ps1 = np.array([fs, 1.-fs])
ps2 = np.array([1.-fs, fs])

Es2 = np.einsum('ij, ik, jk->k', E, ps1, ps2)

plt.plot(fs, Es1-Es2)
plt.show()
exit()

# Interactions
Ei = -np.array([0., 3., 4., 3., 0.])

W = -np.array([[0., 0., 2., 4., 8.],
               [0., 0., 0., 2., 4.],
               [2., 0., 0., 0., 2.],
               [4., 2., 0., 0., 0.],
               [8., 4., 2., 0., 0.]])

"""
fs = np.linspace(0., 0.5, 101)
z = fs*0.

ps = np.array([fs, z, 1. - 2.*fs, z, fs])
ps2 = np.array([2.*fs, z, z, z, 1. - 2.*fs])

Es = np.einsum('ij, ik, jk ->k', W, ps, ps) + np.einsum('i, ik', Ei, ps)
Es2 = np.einsum('ij, ik, jk ->k', W, ps2, ps2) + np.einsum('i, ik', Ei, ps2)
"""

fs = np.linspace(0., 0.5, 101)
z = fs*0.
ps = np.array([fs, 1. - 2.*fs, fs, z, z])
Es = np.einsum('ij, ik, jk ->k', W, ps, ps) + np.einsum('i, ik', Ei, ps)
plt.plot(fs, Es)
plt.plot(1.-fs, Es)

fs = np.linspace(0.5, 1., 101)
z = fs*0.
ps = np.array([fs, 1. - 2.*fs, fs, z, z])
Es = np.einsum('ij, ik, jk ->k', W, ps, ps) + np.einsum('i, ik', Ei, ps)
plt.plot(fs, Es, linestyle=':')
plt.plot(1.-fs, Es, linestyle=':')

#plt.plot(2.*fs, Es2)
plt.show()
