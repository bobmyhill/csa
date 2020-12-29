import numpy as np
from models.csasolutionmodel import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    u[1,1,1,1] = 0.

    u[0,1,1,1] = 3.*wAB
    u[1,0,1,1] = 3.*wAB
    u[1,1,0,1] = 3.*wAB
    u[1,1,1,0] = 3.*wAB

    u[0,0,1,1] = 4.*wAB
    u[0,1,0,1] = 4.*wAB
    u[0,1,1,0] = 4.*wAB
    u[1,0,0,1] = 4.*wAB
    u[1,0,1,0] = 4.*wAB
    u[1,1,0,0] = 4.*wAB

    u[0,0,0,1] = 3.*wAB
    u[0,0,1,0] = 3.*wAB
    u[0,1,0,0] = 3.*wAB
    u[1,0,0,0] = 3.*wAB

    u[0,0,0,0] = 0.
    return u




wAB = -1.*R

temperatures = np.linspace(0.6, 1.2, 101)
Es = np.empty_like(temperatures)
Ss = np.empty_like(temperatures)

ss = CSAModel(cluster_energies=binary_cluster_energies(wAB), gamma=1.)

for i, T in enumerate(temperatures):

    ss.set_state(T)
    ss.set_composition_from_p_s(np.array([0.4, 0.6,
                                          0.3, 0.7,
                                          0.5, 0.5,
                                          0.4, 0.6]))
    ss.equilibrate_clusters()

    Es[i] = ss.molar_gibbs
    Ss[i] = ss.molar_entropy


plt.plot(temperatures, -np.gradient(Es, temperatures, edge_order=2))
plt.plot(temperatures, Ss)
plt.show()
