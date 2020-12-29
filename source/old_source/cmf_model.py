import numpy as np
from models.cmfsolutionmodel import R, CMFModel
from models.idealsolutionmodel import CEFModel
import matplotlib.pyplot as plt


# The following matrices describe the FCC cluster model with
# 4 sites per cluster

# The A-B bond energies are equal to -R
site_species_energies = np.array([-1., 1.,
                                  -1., 1.,
                                  1., -1.,
                                  1., -1.])*R

site_species_energies = np.array([0., 0.,
                                  0., 0.,
                                  0., 0.,
                                  0., 0.])*R

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


def binary_cluster_energies(wAB, alpha=0., beta=0., hA=0.): # 1.*R/4.):
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

    u[0, :, :, :] += hA
    u[1, :, :, :] -= hA
    u[:, 0, :, :] += hA
    u[:, 1, :, :] -= hA

    u[:, :, 1, :] += hA
    u[:, :, 0, :] -= hA
    u[:, :, :, 1] += hA
    u[:, :, :, 0] -= hA

    u[1, 1, 1, 1] += 0.

    u[0, 1, 1, 1] += 3.*wAB*(1. + beta)
    u[1, 0, 1, 1] += 3.*wAB*(1. + beta)
    u[1, 1, 0, 1] += 3.*wAB*(1. + beta)
    u[1, 1, 1, 0] += 3.*wAB*(1. + beta)

    u[0, 0, 1, 1] += 4.*wAB
    u[0, 1, 0, 1] += 4.*wAB
    u[0, 1, 1, 0] += 4.*wAB
    u[1, 0, 0, 1] += 4.*wAB
    u[1, 0, 1, 0] += 4.*wAB
    u[1, 1, 0, 0] += 4.*wAB

    u[0, 0, 0, 1] += 3.*wAB*(1. + alpha)
    u[0, 0, 1, 0] += 3.*wAB*(1. + alpha)
    u[0, 1, 0, 0] += 3.*wAB*(1. + alpha)
    u[1, 0, 0, 0] += 3.*wAB*(1. + alpha)

    u[0, 0, 0, 0] += 0.

    return u


gamma = 1.42
ss = CEFModel(cluster_energies=binary_cluster_energies(wAB=-R),
              gamma=gamma,
              site_species=site_species)

ss2 = CMFModel(site_species=site_species,
               site_species_energies=site_species_energies,
               bond_energies=bond_energies,
               intracluster_connections=intracluster_connections,
               intercluster_connections=intercluster_connections)

temperatures = np.linspace(0.1, 4., 51)
Ss = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Fs = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Es2 = np.empty_like(temperatures)
Fs2 = np.empty_like(temperatures)

f, ax = plt.subplots(1, 3)

#for f in [0.5, 1, 1.5, 2]:
for f in [1, 2]:
    for i, T in enumerate(temperatures):
        ss.equilibrate({'A': f, 'B': 4.-f}, T)
        Fs[i] = ss.molar_helmholtz
        Es[i] = ss.molar_internal_energy
        Ss[i] = ss.molar_entropy
        ss2.equilibrate({'A': f, 'B': 4.-f}, T)
        Fs2[i] = ss2.molar_helmholtz
        Es2[i] = ss2.molar_internal_energy
        Ss2[i] = ss2.molar_entropy

    ax[0].plot(temperatures, Es/R, label='A{0}B{1}'.format(f, 4 - f))
    ax[1].plot(temperatures, Ss/R, label='A{0}B{1}'.format(f, 4 - f))
    ax[2].plot(temperatures, Fs/R, label='A{0}B{1}'.format(f, 4 - f))

    ax[0].plot(temperatures, Es2/R/4., label='A{0}B{1}'.format(f, 4 - f))
    ax[1].plot(temperatures, Ss2/R/4., label='A{0}B{1}'.format(f, 4 - f))
    ax[2].plot(temperatures, Fs2/R/4., label='A{0}B{1}'.format(f, 4 - f))

for ps in [np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
           np.array([0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75])]:
    ss.set_composition_from_p_s(ps)
    ss2.set_composition_from_p_s(ps)
    for i, T in enumerate(temperatures):
        ss.set_state(T)
        ss.equilibrate_clusters()
        ss2.set_state(T)
        ss2.equilibrate_clusters()
        Fs[i] = ss.molar_helmholtz
        Ss[i] = ss.molar_entropy
        Fs2[i] = ss2.molar_helmholtz
        Ss2[i] = ss2.molar_entropy

    ax[0].plot(temperatures, Es/R, label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[1].plot(temperatures, Ss/R, label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[2].plot(temperatures, Fs/R, label='A{0}B{1}'.format(f, 4 - f), linestyle=':')

    ax[0].plot(temperatures, Es2/R/4., label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[1].plot(temperatures, Ss2/R/4., label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[2].plot(temperatures, Fs2/R/4., label='A{0}B{1}'.format(f, 4 - f), linestyle=':')

for i in range(2):
    ax[i].legend()

plt.show()
exit()



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
