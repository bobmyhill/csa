import numpy as np
from models.csasolutionmodel import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def cluster_energies(e1, e2):
    """
    Ordering as in Table 1
    0 is Al
    1 is Si
    """
    u0 = 0.
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
    return u

def minimize_energy(T, ab):
    def energy(r, T, ab):
        ab.set_state(T)
        ab.set_composition_from_p_s(np.array([r, 1.-r,
                                              (1.-r)/3., (2.+r)/3.,
                                              (1.-r)/3., (2.+r)/3.,
                                              (1.-r)/3., (2.+r)/3.]))
        ab.equilibrate_clusters()
        return ab.molar_gibbs

    return minimize_scalar(energy, args=(T, ab), method='bounded', bounds=[0.25, 1.])


"""
Figure 3 from Senderov (1980)
e2/k = 0, e2/k = 40e3
"""
e0 = mpimg.imread('figures/e0.png')
e40000 = mpimg.imread('figures/e40000.png')

fig = plt.figure(figsize=(8, 12))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

ax[0].imshow(e0, extent=[0.0, 3000.0, 0.2, 1.], aspect='auto')
ax[2].imshow(e40000, extent=[0.0, 3000.0, 0.2, 1.], aspect='auto')

f_error = 1. # factor two error?? And e2 seemingly makes bugger all difference
e1s = np.linspace(0., 3000, 21)*R*4.*f_error
Ts = np.linspace(600., 1200., 7)
rs = np.empty((len(Ts), len(e1s)))
Ss = np.empty((len(Ts), len(e1s)))

for k, e2 in enumerate([0.*R, 40.e3*R]):

    for j, e1 in enumerate(e1s):
        ab = CSAModel(cluster_energies=cluster_energies(e1, e2), gamma=0.)

        for i, T in enumerate(Ts):
            rs[i,j] = minimize_energy(T, ab).x
            Ss[i,j] = ab.molar_entropy

    for i, T in enumerate(Ts):
        ax[2*k].plot(e1s/(4.*R*f_error), rs[i], label='{0} K'.format(T))
        ax[2*k+1].plot(e1s/(4.*R*f_error), Ss[i], label='{0} K'.format(T))
    ax[2*k].set_xlabel('e1/4R')
    ax[2*k].set_ylabel('r')
    ax[2*k+1].set_xlabel('e1/4R')
    ax[2*k+1].set_ylabel('S')
    ax[2*k].legend()

plt.show()
