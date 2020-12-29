import numpy as np
from models.csasolutionmodel import CSAModel
from models.iucasolutionmodel import R, IUCAModel
from models.newmodel import NewModel
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

wAB = -1.*R
T = 1.
cluster_energies = binary_cluster_energies(wAB*4.)
site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]

ss = IUCAModel(cluster_energies, site_species, 3./4., [[0, 1, 2, 3]])

ss2 = NewModel(cluster_energies=binary_cluster_energies(wAB*4.),
               gamma=1.*4.,
               site_species=[['A', 'B'], ['A', 'B'],
                             ['A', 'B'], ['A', 'B']])
ss3 = CSAModel(cluster_energies=binary_cluster_energies(wAB),
               gamma=1.,
               site_species=[['A', 'B'], ['A', 'B'],
                             ['A', 'B'], ['A', 'B']])


temperatures = np.linspace(0.1, 5., 101)
Ss = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Ss3 = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Es2 = np.empty_like(temperatures)
Es3 = np.empty_like(temperatures)
Gs = np.empty_like(temperatures)
Gs2 = np.empty_like(temperatures)
Gs3 = np.empty_like(temperatures)

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
for i, T in enumerate(temperatures):
    ss.set_state(T)
    ss2.set_state(T)
    ss3.set_state(T)
    ps = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    ss.set_composition_from_p_s(ps)
    ss.equilibrate_clusters()
    ss2.set_composition_from_p_s(ps)
    ss2.equilibrate_clusters()
    ss3.set_composition_from_p_s(ps)
    ss3.equilibrate_clusters()
    Ss[i] = ss.molar_entropy/4.
    Ss2[i] = ss2.molar_entropy
    Ss3[i] = ss3.molar_entropy
    Es[i] = (ss.molar_gibbs + T*ss.molar_entropy)/4.
    Es2[i] = ss2.molar_gibbs + T*ss2.molar_entropy
    Es3[i] = ss3.molar_gibbs + T*ss3.molar_entropy

    Gs[i] = (ss.molar_gibbs)/4.
    Gs2[i] = ss2.molar_gibbs
    Gs3[i] = ss3.molar_gibbs
    # if i == 0:
    #    print(ss.cluster_proportions)
    #    print(ss3.cluster_proportions)

ax[0].plot(temperatures, Ss/R)
ax[0].plot(temperatures, Ss2/R, linestyle='--')
ax[0].plot(temperatures, Ss3/R, linestyle=':', c='red')

ax[1].plot(temperatures, Es-Es[0]+Es3[0], label='shift')
ax[1].plot(temperatures, Es)
ax[1].plot(temperatures, Es2, linestyle='--')
ax[1].plot(temperatures, Es3, linestyle=':', c='red')
ax[1].legend()
from scipy.integrate import cumtrapz
S_IUCA = cumtrapz(np.gradient(Es, temperatures)/temperatures, temperatures, initial=0.)
ax[2].plot(temperatures, (S_IUCA - S_IUCA[-1] + Ss2[-1])/R, label='IUCA')
ax[2].plot(temperatures, np.gradient(-Gs2, temperatures)/R, linestyle='--', label='FSA')
ax[2].plot(temperatures, np.gradient(-Gs3, temperatures)/R, linestyle=':', c='red', label='CSA')
ax[2].legend()
plt.show()

print(3*R)
exit()

fig = plt.figure()
ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

temperatures = np.linspace(0.05, 2., 101)
fs = np.linspace(0.01, 0.5, 51)
Ss = np.empty_like(fs)
Ss2 = np.empty_like(fs)
Ss3 = np.empty_like(fs)
Es = np.empty_like(fs)
Es2 = np.empty_like(fs)
Es3 = np.empty_like(fs)
Gs = np.empty_like(fs)
Gs2 = np.empty_like(fs)
Gs3 = np.empty_like(fs)

for i, T in enumerate(temperatures):
    ss.set_state(T)
    ss2.set_state(T)
    ss3.set_state(T)
    for j, f in enumerate(fs):
        ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])

        ss.set_composition_from_p_s(ps)
        ss.equilibrate_clusters()
        ss2.set_composition_from_p_s(ps)
        ss2.equilibrate_clusters()
        ss3.set_composition_from_p_s(ps)
        ss3.equilibrate_clusters()
        Ss[j] = ss.molar_entropy/R/4.
        Ss2[j] = ss2.molar_entropy/R
        Ss3[j] = ss3.molar_entropy/R
        Es[j] = (ss.molar_gibbs + T*ss.molar_entropy)/4.
        Es2[j] = ss2.molar_gibbs + T*ss2.molar_entropy
        Es3[j] = ss3.molar_gibbs + T*ss3.molar_entropy

        Gs[j] = (ss.molar_gibbs)/4.
        Gs2[j] = ss2.molar_gibbs
        Gs3[j] = ss3.molar_gibbs

    ax[0].plot(fs, Gs)
    ax[1].plot(fs, Gs2)
    ax[2].plot(fs, Gs3)
    ax[3].plot(fs, Ss)
    ax[4].plot(fs, Ss2)
    ax[5].plot(fs, Ss3)
plt.show()
