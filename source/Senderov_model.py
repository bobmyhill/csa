import numpy as np
from csasolutionmodel import *
import matplotlib.pyplot as plt
# Cluster energies
# Ordering as in Table 1
fac=1.2
u0 = 0.
e1 = 4000.*fac
e2 = 8000.*fac

u = np.zeros((2, 2, 2, 2)) + u0
u[0,1,1,1] = 0.
u[1,1,1,1] = 0.25*e1
u[1,0,1,1] = 0.5*e1
u[1,1,0,1] = 0.5*e1
u[1,1,1,0] = 0.5*e1
u[0,0,1,1] = 0.25*e1 + e2
u[0,1,0,1] = 0.25*e1
u[0,1,1,0] = 0.25*e1 + e2
u[1,0,0,1] = 0.75*e1 + e2
u[1,0,1,0] = 0.75*e1
u[1,1,0,0] = 0.75*e1 + e2
u[0,0,0,1] = 0.5*e1 + 2.*e2
u[0,0,1,0] = 0.5*e1 + 2.*e2
u[0,1,0,0] = 0.5*e1 + 2.*e2
u[1,0,0,0] = e1 + 2.*e2
u[0,0,0,0] = 0.75*e1 + 4.*e2

# gamma
# S_n = -2.*0.5*np.log(0.5)
# S_i = -8.*0.5*np.log(0.5)
gamma = 4./3. # S_i/(S_i-S_n)

# Senderov (1980) albite
ab = CSAModel(cluster_energies=u, gamma=gamma)


ys = np.linspace(0., 0.99, 101)
gibbs = np.empty_like(ys)
entropies = np.empty_like(ys)

fig = plt.figure(figsize=(16,6))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

for T in np.linspace(500., 6000., 11)/R:
    ab.set_state(T)
    for i, y in enumerate(ys):
        print(y)
        ab.set_composition_from_p_s(np.array([y, 1.-y,
                                              0.3, 0.7,
                                              1.-y, y,
                                              y, 1.-y]))
        ab.equilibrate_clusters()

        gibbs[i] = ab.molar_gibbs
        entropies[i] = ab.molar_entropy
        # print(ab.molar_chemical_potentials)

        # print(ab.molar_gibbs)
        # print(np.einsum('i, i', ab.p_ind, ab.molar_chemical_potentials))
    ax[0].plot(ys, gibbs, label='{0:.0f} K'.format(T))
    ax[1].plot(ys, entropies, label='{0:.0f} K'.format(T))
ax[0].set_xlabel('x')
ax[1].set_xlabel('x')

ax[0].set_ylabel('Helmholtz free energy (J/mol)')
ax[1].set_ylabel('Entropy (J/K/mol)')

ax[0].legend()
ax[1].legend()
plt.show()
exit()
