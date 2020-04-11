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
            {'type': 'ineq', 'fun': lambda x: 3.-x[0]-x[1]-x[2]})


    sobj=Storage()
    p_As = minimize(F, [0.1*p_A, 0.1*p_A, 0.9*p_A], args=(sobj),
                    method='SLSQP', constraints=cons).x

    return F(p_As, sobj), sobj.p_cl, sobj.mu
gamma = 1.
T = 1.
n = 4.
#p_As = np.linspace(0.001, 0.999, 101)

p_A = 0.5



p_As0 = np.array([0.001, 0.999, 0.001, 0.999])
p_As1 = np.array([0.5, 0.5, 0.5, 0.5])

temperatures = np.linspace(0.6, 1.2, 61)
Ss = np.empty_like(temperatures)

fig = plt.figure(figsize=(12, 5))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

pfig2 = mpimg.imread('figures/Oates_1999_Fig_2.png')
ax[0].imshow(pfig2, extent=[0.6, 1.2, 0., 0.6], aspect='auto')

xs = np.linspace(0.0, 1., 1001)
Fs = np.empty_like(xs)
Ss_temp = np.empty_like(xs)

for gamma, c in [(1., 'red'), (1.22, 'blue')]:
    wAB = -1.*R/gamma
    print('gamma = {0}'.format(gamma))
    for j, T in enumerate(temperatures):
        for i, x in enumerate(xs):
            p_As = x*p_As0 + (1. - x)*p_As1
            E_cl = cluster_energies(wAB)
            mus = cluster_equilibrate(p_As, E_cl, gamma, n, T)

            Fs[i] = free_energy(mus, p_As, E_cl, gamma, n, T)
            p_cl = cluster_proportions(mus, p_As, E_cl, gamma, n, T)

            S_s = -R*(np.sum(p_As*np.log(p_As)) +
                      np.sum((1.-p_As)*np.log(1.-p_As)))
            S_c = -R*np.sum(p_cl*np.log(p_cl))
            S_t = gamma*S_c + (1./n - gamma)*S_s
            Ss_temp[i] = S_t

        i_mins = np.r_[True, Fs[1:] < Fs[:-1]] & np.r_[Fs[:-1] < Fs[1:], True]
        i_local_mins = [i for i, bool in enumerate(i_mins)
                        if bool == True and i != np.argmin(Fs)]
        ax[0].scatter([T for i in i_local_mins], Ss_temp[i_local_mins]/R, c=c, s=10)
        Ss[j] = Ss_temp[np.argmin(Fs)]
        if gamma == 1. and j%10 == 0:
            ax[1].plot(xs, Fs/R, label='T\' = {0:.1f}'.format(T))

        print(T, Fs[i_mins])

    ax[0].plot(temperatures, Ss/R, label='stable states ($\\gamma$ = {0})'.format(gamma), c=c)

    ax[0].scatter([0.], [0.], c=c, s=10, label='metastable states ($\\gamma$ = {0})'.format(gamma))

ax[0].set_xlim(0.6, 1.2)
ax[0].set_ylim(0.,0.6)
ax[0].set_xlabel('Reduced temperature')
ax[0].set_ylabel('Entropy/R')

ax[1].set_xlim(0., 1.)
ax[1].set_xlabel('x (LRO)')
ax[1].set_ylabel('Free energy/R')

ax[0].legend()
ax[1].legend()

fig.tight_layout()
fig.savefig('Oates_1999_Fig_2_benchmark.pdf')
plt.show()
