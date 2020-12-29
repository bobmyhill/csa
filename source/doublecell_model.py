import numpy as np
from models.doublecellmodel import R, DoubleCellModel
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
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

# equivalent to external 1*A + 2.*((1-f)*B + f*MF)
mf = 0.9
f_A = (1. + 1.)/4.
f_B = 2.*(1. - mf) / 4.
f_MF = 2.*mf / 4.


# 1*A + 1*B + 1*MF is reasonable.
# 1*A + 0.5*B + 1.5*MF is not

# how about A = B, f = MF/(MF+B), A + B + MF = 1.
# then
# 1 + A/MF = 1/f
# A = (1 - MF)/2.
# 1 + 1/(2*MF) - 1/2 = 1/f
# 1 + 1/MF = 2/f
# 1/MF = 2/f - (f/f)
# MF = f/(2 - f)


# mf = 0. is a completely ordered set of compounds. (2*A, 2*B)
# mf = 1. is the same as the point approximation
mf = 0.36
f_MF = mf / (2. - mf)
f_A = (1. - f_MF)/2.
f_B = (1. - f_MF)/2.
print(f_A*4., f_B*4., f_MF*4.)

"""
# how about B = 0, f = 1, A + B + MF = 1.
# then
f_MF = 0.5
f_A = (1. - f_MF)
f_B = 0.
print(f_A*4.-1., f_B*4., f_MF*4.)
# I don't think you can just add an arbitrary fraction of B to this.
# You end up getting two heat capacity peaks.
"""

internal_bonds_per_cluster = np.copy(cluster_bonds_per_cluster)*f_A
bridging_bonds_per_cluster = np.copy(cluster_bonds_per_cluster)*(f_B + f_MF)


ss = DoubleCellModel(site_species=site_species,
                     site_species_energies=site_species_energies,
                     bond_energies=bond_energies,
                     internal_bonds_per_cluster=internal_bonds_per_cluster,
                     bridging_bonds_per_cluster=bridging_bonds_per_cluster,
                     meanfield_fraction=mf)

d, i = np.unique(ss.cluster_occupancies, return_index=True, axis=0)
print(d, len(i))
exit()

"""
plt.hist(ss.cluster_energies/R/8., bins=51, density=True)
plt.yscale('log')
plt.show()
exit()
"""

fig = plt.figure(figsize=(20, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

fig1 = plt.figure(figsize=(20, 5))
ax1 = [fig1.add_subplot(1, 3, i) for i in range(1, 4)]

def helmholtz(f, x, T):
    if x <= 0.25:
        ps_ord = np.array([4.*x, 1. - 4.*x, 0., 1., 0., 1., 0., 1.])
    elif x <= 0.5:
        ps_ord = np.array([1., 0., 4.*(x-0.25), 1. - 4.*(x-0.25), 0., 1., 0., 1.])
    elif x <= 0.75:
        ps_ord = np.array([1., 0., 1., 0., 4.*(x-0.5), 1. - 4.*(x-0.5), 0., 1.])
    elif x <= 1.:
        ps_ord = np.array([1., 0., 1., 0., 1., 0., 4.*(x-0.75), 1. - 4.*(x-0.75)])

    ps_disord = np.array([x, 1. - x, x, 1. - x, x, 1. - x, x, 1. - x])

    ps = f[0]*ps_disord + (1. - f[0])*ps_ord

    ss.set_composition_from_p_s(ps)
    ss.set_state(T)
    ss.equilibrate_clusters()
    return ss.molar_helmholtz/R/8.


def equil(x, T):
    sol = minimize(helmholtz, [0.01],
                   args=(x, T), method=None,
                   bounds=((0, 1),))
    assert sol.success, "nope"


nT = 11
nx = 101
temperatures = np.linspace(2.6, 0.15, nT)
xs = np.linspace(0.001, 0.5, nx)
Ss = np.empty((nx, nT))
Es = np.empty((nx, nT))
Fs = np.empty((nx, nT))
Ss[:, :] = np.NaN
Es[:, :] = np.NaN
Fs[:, :] = np.NaN

for i, x in enumerate(xs):
    print(x)
    for j, T in enumerate(temperatures):
        try:
            equil(x, T)
            Es[i, j] = ss.molar_internal_energy
            Ss[i, j] = ss.molar_entropy
            Fs[i, j] = ss.molar_helmholtz
        except:
            pass

for j, T in enumerate(temperatures):
    ax[0].plot(xs, Es[:, j]/R/8., label=f'{T} K')
    ax[1].plot(xs, Ss[:, j]/R/8., label=f'{T} K')
    ax[2].plot(xs, Fs[:, j]/R/8., label=f'{T} K')

ax[1].plot(xs, -xs*np.log(xs) - (1. - xs)*np.log(1. - xs), linestyle=':')


# import and plot data for comparison
E = (fcc_ising_data['E']-6.)/2.
T = fcc_ising_data['T']/2.
S = fcc_ising_data['S']
ax1[0].plot(T, E, linewidth=3)
ax1[1].plot(T, S, linewidth=3)
ax1[2].plot(T, E - T*S, linewidth=3)


x = 0.5
if x <= 0.25:
    ps_ord = np.array([4.*x, 1. - 4.*x, 0., 1., 0., 1., 0., 1.])
elif x <= 0.5:
    ps_ord = np.array([1., 0., 4.*(x-0.25), 1. - 4.*(x-0.25), 0., 1., 0., 1.])
elif x <= 0.75:
    ps_ord = np.array([1., 0., 1., 0., 4.*(x-0.5), 1. - 4.*(x-0.5), 0., 1.])
elif x <= 1.:
    ps_ord = np.array([1., 0., 1., 0., 1., 0., 4.*(x-0.75), 1. - 4.*(x-0.75)])

ps_disord = np.array([x, 1. - x, x, 1. - x, x, 1. - x, x, 1. - x])

nT = 101
nf = 6
temperatures = np.linspace(0.2, 2.6, nT)
fs = np.linspace(0., 1., nf)

Ss = np.empty((nf, nT))
Es = np.empty((nf, nT))
Fs = np.empty((nf, nT))

for i, f in enumerate(fs):
    print(f)
    ps = f*ps_disord + (1. - f)*ps_ord

    for j, T in enumerate(temperatures):
        ss.set_composition_from_p_s(ps)
        ss.set_state(T)
        ss.equilibrate_clusters()
        Es[i, j] = ss.molar_internal_energy
        Ss[i, j] = ss.molar_entropy
        Fs[i, j] = ss.molar_helmholtz

for i, f in enumerate(fs):
    ax1[0].plot(temperatures, Es[i]/R/8., label=f'{f}')
    ax1[1].plot(temperatures, Ss[i]/R/8., label=f'{f}')
    ax1[2].plot(temperatures, Fs[i]/R/8., label=f'{f}')


#ax1[1].plot(temperatures, (Ss[0]+cumtrapz((np.gradient(Es, temperatures)
#                                          / temperatures),
#                                          temperatures, initial=0.))/R/8.,
#            label=f'.', linestyle=':')

# ONLY RUNS IF TRUE
if False:
    temperatures = np.linspace(0.5, 2.6, 101)
    Ss = np.empty_like(temperatures)
    Es = np.empty_like(temperatures)
    Fs = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        print(T)
        ss.equilibrate({'A': 1.5, 'B': 2.5}, T)
        Es[i] = ss.molar_internal_energy
        Ss[i] = ss.molar_entropy
        Fs[i] = ss.molar_helmholtz
    ax1[0].plot(temperatures, Es/R/8., label=f'eqm', linewidth=3)
    ax1[1].plot(temperatures, Ss/R/8., label=f'eqm', linewidth=3)
    ax1[2].plot(temperatures, Fs/R/8., label=f'eqm', linewidth=3)

ax1[1].plot(temperatures, temperatures*0. + np.log(2.), linestyle=':')


#ax[2].set_ylim(-5., -3.)

for i in range(3):
    ax[i].legend()
    ax1[i].legend()

fig.savefig('equilibrium_potentials.pdf')
fig1.savefig('AABB_potentials.pdf')
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
