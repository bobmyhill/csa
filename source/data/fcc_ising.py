import numpy as np
from scipy.integrate import cumtrapz


def entropies_heat_capacities(energies, temperatures):
    c = np.gradient(energies, temperatures)
    s = cumtrapz(np.gradient(energies, temperatures)/temperatures,
                 temperatures, initial=0)
    return s, c


# Data from Thanh Ngo et al. (2014) and Beath and Ryan (2006)
T, E = np.loadtxt('FCC_AF_3d_Ising.dat', unpack=True)
S, C = entropies_heat_capacities(E, T)
fcc_ising_data = {'T': T,
                  'E': E,
                  'S': S,
                  'C': C,
                  'F': E - T*S}
