import numpy as np
from models.iodasolutionmodel import R, IODAModel
import matplotlib.pyplot as plt


def binary_cluster_energies(wAB):
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

    u[0, 1, 1, 1] = 3.*wAB
    u[1, 0, 1, 1] = 3.*wAB
    u[1, 1, 0, 1] = 3.*wAB
    u[1, 1, 1, 0] = 3.*wAB

    u[0, 0, 1, 1] = 4.*wAB #*1.05
    u[0, 1, 0, 1] = 4.*wAB #*0.95
    u[0, 1, 1, 0] = 4.*wAB #*0.95
    u[1, 0, 0, 1] = 4.*wAB #*0.95
    u[1, 0, 1, 0] = 4.*wAB #*0.95
    u[1, 1, 0, 0] = 4.*wAB #*1.05

    u[0, 0, 0, 1] = 3.*wAB
    u[0, 0, 1, 0] = 3.*wAB
    u[0, 1, 0, 0] = 3.*wAB
    u[1, 0, 0, 0] = 3.*wAB

    u[0, 0, 0, 0] = 0.
    return u

E = np.array([0., 0., -1])
W = np.array([[0., -4., 0.],
              [0., 0., 0.],
              [0., 0., 0.]])

fs = np.linspace(0., 1., 101)

ps = np.array([0.5*fs, 0.5*fs, 1.-fs]) # AA, BB, AB
Es = np.einsum('i, ij -> j', E, ps) + np.einsum('ij, il, jl -> l', W, ps, ps)
plt.plot(fs, Es)

ps = np.array([0.*fs, 1.*fs, 1.-fs]) # AA, BB, AB
Es = np.einsum('i, ij -> j', E, ps) + np.einsum('ij, il, jl -> l', W, ps, ps)
plt.plot(fs, Es)
plt.show()
exit()


wAB = -1.*R
T = 1.
cluster_energies = binary_cluster_energies(wAB*4.)
site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]

ss = IODAModel(cluster_energies, site_species, [[0, 1, 2, 3]])

# Try independent clusters 0000, 1000, 0100, 0010, 0001
A = np.array([[0., 0., 0., 0.],
              [1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
A = np.stack((A, 1. - A))
M = np.einsum('ijkl, im, jn, ko, lp', cluster_energies,
              A[:,:,0], A[:,:,1], A[:,:,2], A[:,:,3])
print(M.shape)

"""
M[1,[1,2],[1,2],[1,2]] = M[1,1,1,1]
M[1,[1,3],[1,3],[1,3]] = M[1,1,1,1]
M[1,[1,4],[1,4],[1,4]] = M[1,1,1,1]


M[2,[2,1],[2,1],[2,1]] = M[1,1,1,1]
M[2,[2,3],[2,3],[2,3]] = M[1,1,1,1]
M[2,[2,4],[2,4],[2,4]] = M[1,1,1,1]


M[3,[3,1],[3,1],[3,1]] = M[1,1,1,1]
M[3,[3,2],[3,2],[3,2]] = M[1,1,1,1]
M[3,[3,4],[3,4],[3,4]] = M[1,1,1,1]

M[4,[4,1],[4,1],[4,1]] = M[1,1,1,1]
M[4,[4,2],[4,2],[4,2]] = M[1,1,1,1]
M[4,[4,3],[4,3],[4,3]] = M[1,1,1,1]
"""
f = 0.5
p = np.array([(1. - f), 0.25*f, 0.25*f, 0.25*f, 0.25*f])

print((M[0,0,0,0] + M[1,1,1,1])/R/2.)
print(np.einsum('ijkl, i, j, k, l', M, p, p, p, p)/R)

print(M[2:4,2:4,2:4,2:4]/R)

print(M[2, 2, 2, 2]/R)
print(M[3, 3, 3, 3]/R)
print(M[2, 3, 2, 3]/R)
print(M[2, 2, 2, 3]/R)
exit()


temperatures = np.linspace(0.01, 1., 1001)
Fdis = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Fs = np.empty_like(temperatures)

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

f = 0.5
ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])
ss.set_composition_from_p_s(ps)

for i, T in enumerate(temperatures):
    ss.set_state(T)
    Fdis[i] = ss.molar_helmholtz

for f in np.linspace(0., 0.5, 11):
    ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])
    ss.set_composition_from_p_s(ps)

    for i, T in enumerate(temperatures):
        ss.set_state(T)

        Ss[i] = ss.molar_entropy
        Es[i] = ss.molar_energy
        Fs[i] = ss.molar_helmholtz

    ax[0].plot(temperatures, Ss/R)
    ax[1].plot(temperatures, Es/R)
    ax[2].plot(temperatures, (Fs-Fdis)/R/temperatures, label=f)

ax[2].set_ylim(-0.025, 0.005)
ax[2].legend()
plt.show()
