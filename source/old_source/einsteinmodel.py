import numpy as np
from models.einsteinsolutionmodel import R, EinsteinModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


wAB = -1.*R
T = 1.
cluster_energies = binary_cluster_energies(wAB*4.)
site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]

ss = EinsteinModel(cluster_energies, site_species, [[0, 1, 2, 3]])

temperatures = np.linspace(0.01, 2.4*2., 101)
Fdis = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Fs = np.empty_like(temperatures)

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

pfig2 = mpimg.imread('figures/Oates_1999_Fig_2.png')
ax[0].imshow(pfig2, extent=[0.6, 1.2, 0., 0.6], aspect='auto')

pfig3 = mpimg.imread('figures/Ngo_2013_Fig_2.png')
ax[1].imshow(pfig3, extent=[1., 2.6, -4., -3.], aspect='auto')

pfig1 = mpimg.imread('figures/Ferreira_Fig1.png')
ax[2].imshow(pfig1, extent=[0.6, 1.4, -5., 1.], aspect='auto')

f = 0.5
ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])
#ps = np.array([3.*f/2., 1.-3.*f/2., 1.-f/2., f/2., 1.-f/2., f/2., 1.-f/2., f/2.])
print(ps)
ss.set_composition_from_p_s(ps)

for i, T in enumerate(temperatures):
    ss.set_state(T)
    Fdis[i] = ss.molar_helmholtz

for f in np.linspace(0.0, 0.5, 11):
    ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])
    #ps = np.array([3.*f/2., 1.-3.*f/2., 1.-f/2., f/2., 1.-f/2., f/2., 1.-f/2., f/2.])
    ss.set_composition_from_p_s(ps)

    for i, T in enumerate(temperatures):
        ss.set_state(T)

        Ss[i] = ss.molar_entropy
        Es[i] = ss.molar_energy
        Fs[i] = ss.molar_helmholtz

    ax[0].plot(temperatures, Ss/R/4.)
    ax[1].plot(temperatures, Es/R/4.)
    #ax[2].plot(temperatures, (Fs)/R, label=f)
    #ax[2].plot(temperatures, (Fs-Fdis)/R/temperatures, label=f)
    ax[2].plot(temperatures, (Fs)/R/4., label=f)

"""
Es = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    ss.equilibrate(composition={'A': 0.5, 'B': 0.5}, temperature=T)
    print(i, ss.molar_helmholtz)
    Es[i] = ss.molar_helmholtz
"""
#ax[2].plot(temperatures, Es)
#ax[2].set_ylim(-0.025, 0.005)
ax[2].legend()
plt.show()
