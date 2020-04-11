import numpy as np
from models.csasolutionmodel import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root

R = 8.31446

class Storage(object):
    def __init__(self):
        self.p_cl = None
        self.mu = None
        self.E = None

def cluster_energies(wAB):
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


def cluster_proportions(mus, p_As, cluster_energies, gamma, n, T):
    sum_alnmu = np.sum(p_As*np.log(mus))

    mu_brack = np.ones((2, 2, 2, 2))
    for indices, v in np.ndenumerate(mu_brack):
        for j, i in enumerate(indices):
            if i == 0:
                mu_brack[indices] *= mus[j]
    #print(mu_brack)
    beta = 1./(R*T)

    ppnsphi = mu_brack*np.exp(-beta*cluster_energies)
    phi = np.sum(ppnsphi) # Eq.3b

    return ppnsphi/phi

def free_energy(mus, p_As, cluster_energies, gamma, n, T):
    sum_alnmu = np.sum(p_As*np.log(mus))

    mu_brack = np.ones((2, 2, 2, 2))
    for indices, v in np.ndenumerate(mu_brack):
        for j, i in enumerate(indices):
            if i == 0:
                mu_brack[indices] *= mus[j]
    #print(mu_brack)
    beta = 1./(R*T)
    phi = np.sum(mu_brack*np.exp(-beta*cluster_energies)) # Eq.3b

    S_s = np.sum(p_As*np.log(p_As)) + np.sum((1.-p_As)*np.log(1.-p_As))

    Fm = R*T*(gamma*(sum_alnmu - np.log(phi)) - (n*gamma - 1.)*0.25*S_s) # Eq.6
    return Fm

def cluster_equilibrate(p_As, cluster_energies, gamma, n, T):
    def delta_p_As(mus):
        ppns = cluster_proportions(mus, p_As, cluster_energies, gamma, n, T)
        dev_p_As = np.zeros(4)
        for indices, ppn in np.ndenumerate(ppns):
            for j, i in enumerate(indices):
                if i == 0:
                    dev_p_As[j] += ppn
        return dev_p_As - p_As

    sol = root(delta_p_As, [0.1, 0.1, 0.1, 0.1])
    mus = sol.x

    return mus


def equilibrate(p_A, cluster_energies, gamma, n, T):
    def F(args, sobj):
        p_As = np.zeros(4)
        p_As[0:3] = args
        p_As[3] = p_A*4. - np.sum(p_As[0:3])

        mus = cluster_equilibrate(p_As, cluster_energies, gamma, n, T)

        p_cl = cluster_proportions(mus, p_As, cluster_energies, gamma, n, T)
        sobj.p_cl = p_cl
        sobj.mu = mus
        non_conf = np.sum(p_cl * cluster_energies)
        non_conf=0.
        return non_conf + free_energy(mus, p_As, cluster_energies, gamma, n, T)

    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: 1.-x[0]-x[1]-x[2]})


    #return minimize(F, [p_A, p_A, p_A], method='SLSQP', constraints=cons).fun
    sobj=Storage()
    sol = F([p_A, p_A, p_A], sobj)

    return F([p_A, p_A, p_A], sobj), sobj.p_cl, sobj.mu
gamma = 1.
T = 1.
n = 4.
wAB = -1.*R
p_As = np.linspace(0.001, 0.999, 101)
Fs = np.empty_like(p_As)

F, p_cls, mus = equilibrate(0.4, cluster_energies(wAB), gamma, n, T)
print(p_cls)
print(F/R)
exit()
for i, p_A in enumerate(p_As):
    #print(p_A)
    F, p_cls, mus = equilibrate(p_A, cluster_energies(wAB), gamma, n, T)
    print(mus)
    #print(p_cls)
    Fs[i] = F

e0 = mpimg.imread('figures/Oates_1996_Fig_1.png')
plt.imshow(e0, extent=[0.0, 0.5, -5, 0.], aspect='auto')

plt.plot(p_As, Fs/R)
plt.xlim(0., 0.6)
plt.show()
