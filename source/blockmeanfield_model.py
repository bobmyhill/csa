import numpy as np
from models.blockmeanfieldsolutionmodel import R, BlockMeanFieldModel
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from fcc_ising import fcc_ising_data

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

bridging_bonds_per_cluster = np.copy(cluster_bonds_per_cluster)


ss = BlockMeanFieldModel(site_species=site_species,
                         site_species_energies=site_species_energies,
                         bond_energies=bond_energies,
                         cluster_bonds_per_cluster=cluster_bonds_per_cluster,
                         bridging_bonds_per_cluster=bridging_bonds_per_cluster,
                         fraction_bridging_bonds_in_basic_cluster=0.75)

f = 0.5
ps = np.array([1.-f, f,    1.-f, f,
               f,    1.-f, f,    1.-f])
temperatures = np.linspace(0.1, 3., 101)
Ss = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Fs = np.empty_like(temperatures)

fig = plt.figure(figsize=(20, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

for block_size in [1., 2., 4., 8., 16., 32.]:
    for i, T in enumerate(temperatures):
        if T > block_size*0.05:
            print(block_size, i)
            ss.set_composition_from_p_s(ps, block_size)
            ss.set_state(T)
            ss.equilibrate_clusters()
            print(ss.block_energies/R/4.)
            print(-3 + (-3 - ss.block_energies/R/4.))
            exit()

            Es[i] = ss.molar_internal_energy
            Ss[i] = ss.molar_entropy
            Fs[i] = ss.molar_helmholtz
        else:
            Es[i] = np.nan
            Ss[i] = np.nan
            Fs[i] = np.nan

    ax[0].plot(temperatures, Es/R/4., label=f'{block_size}')

    """
    if block_size == 1.:
        T = temperatures/1.2
        E = (4.*Es - 3.*-12.*R)
        S = cumtrapz((np.gradient(E, T) / T), T, initial=0.)
        ax[0].plot(T, E/R/4., label=f'{block_size}')
        ax[1].plot(T, S/R/4., label=f'{block_size}')
    """

    if block_size == 1.:
        ax[0].plot(temperatures/1.22, Es/R - 3.*-3., label=f'{block_size}')
        ax[1].plot(temperatures/1.22, Ss/R - 3.*np.log(2.), label=f'{block_size}')

    ax[1].plot(temperatures, Ss/R/4., label=f'{block_size}')
    ax[2].plot(temperatures, Fs/R/4., label=f'{block_size}')

    ax[1].plot(temperatures, (Ss[0]+cumtrapz((np.gradient(Es, temperatures)
                                              / temperatures),
                                             temperatures, initial=0.))/R/4.,
               label=f'{block_size}', linestyle=':')


# import and plot data for comparison
E = (fcc_ising_data['E']-6.)/2.
T = fcc_ising_data['T']/2.
S = fcc_ising_data['S']
ax[0].plot(T, E)
ax[1].plot(T, S)
ax[2].plot(T, E - T*S)


ax[2].set_ylim(-5., -3.)

for i in range(3):
    ax[i].legend()
plt.show()

exit()

temperatures = np.linspace(0.1, 4., 51)
Ss = np.empty_like(temperatures)
Es = np.empty_like(temperatures)
Fs = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Es2 = np.empty_like(temperatures)
Fs2 = np.empty_like(temperatures)

fig, ax = plt.subplots(1, 2)

# for f in [0.5, 1, 1.5, 2]:
for f in [1, 2]:
    for i, T in enumerate(temperatures):
        ss.equilibrate({'A': f, 'B': 4.-f}, T)
        Fs[i] = ss.molar_helmholtz
        Es[i] = ss.molar_internal_energy
        Ss[i] = ss.molar_entropy

    ax[0].plot(temperatures, Ss/R, label='A{0}B{1}'.format(f, 4 - f))
    ax[1].plot(temperatures, Fs/R, label='A{0}B{1}'.format(f, 4 - f))

    ax[0].plot(temperatures, Ss2/R/4., label='A{0}B{1}'.format(f, 4 - f))
    ax[1].plot(temperatures, Fs2/R/4., label='A{0}B{1}'.format(f, 4 - f))

for ps in [np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
           np.array([0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75])]:
    ss.set_composition_from_p_s(ps, 1.)
    for i, T in enumerate(temperatures):
        ss.set_state(T)
        ss.equilibrate_clusters()
        Fs[i] = ss.molar_helmholtz
        Ss[i] = ss.molar_entropy

    ax[0].plot(temperatures, Ss/R,
               label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[1].plot(temperatures, Fs/R,
               label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[0].plot(temperatures, Ss2/R/4.,
               label='A{0}B{1}'.format(f, 4 - f), linestyle=':')
    ax[1].plot(temperatures, Fs2/R/4.,
               label='A{0}B{1}'.format(f, 4 - f), linestyle=':')

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
    ps = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    ss.set_composition_from_p_s(ps)
    ss.equilibrate_clusters()
    Ss[i] = ss.molar_entropy/4.
    Es[i] = (ss.molar_gibbs + T*ss.molar_entropy)/4.

    Gs[i] = (ss.molar_gibbs)/4.
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

S_IUCA = cumtrapz(np.gradient(Es, temperatures)/temperatures, temperatures,
                  initial=0.)
ax[2].plot(temperatures, (S_IUCA - S_IUCA[-1] + Ss2[-1])/R,
           label='IUCA')
ax[2].plot(temperatures, np.gradient(-Gs2, temperatures)/R,
           linestyle='--', label='FSA')
ax[2].plot(temperatures, np.gradient(-Gs3, temperatures)/R,
           linestyle=':', c='red', label='CSA')
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
    for j, f in enumerate(fs):
        ps = np.array([f, 1.-f, f, 1.-f, 1.-f, f, 1.-f, f])

        ss.set_composition_from_p_s(ps)
        ss.equilibrate_clusters()
        Ss[j] = ss.molar_entropy/R/4.
        Es[j] = (ss.molar_gibbs + T*ss.molar_entropy)/4.

        Gs[j] = (ss.molar_gibbs)/4.

    ax[0].plot(fs, Gs)
    ax[1].plot(fs, Gs2)
    ax[2].plot(fs, Gs3)
    ax[3].plot(fs, Ss)
    ax[4].plot(fs, Ss2)
    ax[5].plot(fs, Ss3)
plt.show()
