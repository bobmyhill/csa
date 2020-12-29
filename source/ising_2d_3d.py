import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk
from scipy.integrate import cumtrapz

# Taken from Plischke and Bergesen, 1989
# Except there seems to be an error in the expression for q...
# The correct expression can be found on wiki
# (https://en.wikipedia.org/wiki/Square_lattice_Ising_model)
R = 8.31446

def coth(x):
     return 1./np.tanh(x)

def q(K, L):
    k = 1./(np.sinh(2.*K)*np.sinh(2.*L))
    m = 4.*k*np.power(1.+k, -2.)
    return m

def energy(T, J):
    beta = 1./(R*T)
    K = beta*J
    L = beta*J
    K1q = ellipk(q(K, L))
    u = -J*coth(2.*K)*(1. + 2./np.pi*(2.*(np.tanh(2.*K))**2 - 1.)*K1q)
    return u

def entropies_heat_capacities(energies, temperatures):
    c = np.gradient(energies, temperatures)
    s = cumtrapz(np.gradient(energies, temperatures)/temperatures,
                 temperatures, initial=0)
    return s, c

J = R
Tc = 2.*J/(R*np.log(1. + np.sqrt(2.))) # for K=L=betaJ

temperatures = np.linspace(Tc - 0.1, Tc + 0.1, 100001) # *J/R
temperatures = np.linspace(0.01, 10., 1000001)*J/R
energies = energy(temperatures, J)
entropies, heat_capacities = entropies_heat_capacities(energies, temperatures)
# Data from Thanh Ngo et al. (2014) and Beath and Ryan (2006)
T, E = np.loadtxt('FCC_AF_3d_Ising.dat', unpack=True)
S, C = entropies_heat_capacities(E, T)


fig = plt.figure(figsize=(15, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

ax[0].scatter(T, E)
ax[0].plot(T, E, label='3D Ising (AF FCC lattice)')
ax[0].plot(temperatures, energies/R, label='2D Ising (square lattice)')

ax[1].scatter(T, S)
ax[1].plot(T, S, label='3D Ising (AF FCC lattice)')
ax[1].plot(temperatures, entropies/R, label='2D Ising (square lattice)')

ax[2].scatter(T, C)
ax[2].plot(T, C, label='3D Ising (AF FCC lattice)')
ax[2].plot(temperatures, heat_capacities/R, label='2D Ising (square lattice)')

# Here's the Vinograd data for majorite.
Ts = np.array([3673.,  3473,   3273,   3073,   2873,   2673.,  2473.])
Ss = np.array([4.0476, 3.7886, 1.7238, 1.0762, 0.7029, 0.4819, 0.2152])

ax[1].scatter(Ts/1800., Ss/R, label="majorite (T = 1800 T')")
ax[1].scatter(Ts/1400., Ss/R, label="majorite (T = 1400 T')")
#plt.plot(R*temperatures/J, np.gradient(energy(temperatures, J), temperatures))
#plt.plot(R*temperatures/J, cumtrapz(np.gradient(energy(temperatures, J), temperatures)/temperatures,
#                                    temperatures, initial=0))
ax[1].plot(temperatures, temperatures*0. + np.log(2.), label='ideal')

for i in range(3):
    ax[i].set_xlabel('T')
    ax[i].legend()
ax[0].set_ylabel('Energies/R')
ax[1].set_ylabel('Entropies/R')
ax[2].set_ylabel('Heat capacities/R')
ax[2].set_ylim(0, 4)
fig.savefig('ising_2D_3D_models.pdf')
plt.show()
