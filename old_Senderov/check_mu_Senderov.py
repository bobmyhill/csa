import numpy as np
from sympy import Matrix
from sympy.solvers import solve
from sympy import Symbol, symbols, diff
from scipy.optimize import newton_krylov, minimize, curve_fit
import matplotlib.pyplot as plt

import time

def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps: log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    mask = x>eps
    ln = np.where(x<=eps, np.log(eps) - f_eps - f_eps*f_eps/2., 0.)
    ln[mask] = np.log(x[mask])
    return ln

run_inversion=True
start = time.time()

# Ordering as in Table 1
names = ['ASSS',
         'SSSS',
         'SASS',
         'SSAS',
         'SSSA',
         'AASS',
         'ASAS',
         'ASSA',
         'SAAS',
         'SASA',
         'SSAA',
         'AAAS',
         'AASA',
         'ASAA',
         'SAAA',
         'AAAA']

"""
A0_indices = [0, 5, 6, 7, 11, 12, 13, 15]
A1_indices = [2, 5, 8, 9, 11, 12, 14, 15]
A2_indices = [3, 6, 8, 10, 11, 13, 14, 15]
A3_indices = [4, 7, 9, 10, 12, 13, 14, 15]
"""

E_mbr = np.array([[1., 0., 0., 1., 0., 1., 0., 1.],
                  [0., 1., 0., 1., 0., 1., 0., 1.],
                  [0., 1., 1., 0., 0., 1., 0., 1.],
                  [0., 1., 0., 1., 1., 0., 0., 1.],
                  [0., 1., 0., 1., 0., 1., 1., 0.],
                  [1., 0., 1., 0., 0., 1., 0., 1.],
                  [1., 0., 0., 1., 1., 0., 0., 1.],
                  [1., 0., 0., 1., 0., 1., 1., 0.],
                  [0., 1., 1., 0., 1., 0., 0., 1.],
                  [0., 1., 1., 0., 0., 1., 1., 0.],
                  [0., 1., 0., 1., 1., 0., 1., 0.],
                  [1., 0., 1., 0., 1., 0., 0., 1.],
                  [1., 0., 1., 0., 0., 1., 1., 0.],
                  [1., 0., 0., 1., 1., 0., 1., 0.],
                  [0., 1., 1., 0., 1., 0., 1., 0.],
                  [1., 0., 1., 0., 1., 0., 1., 0.]])


E_ind = []
for mbr in E_mbr:
    E_ind.append(np.linalg.lstsq(E_mbr[:5].T, mbr, rcond=None)[0].round(decimals=10))

E_ind = np.array(E_ind)


R = 8.31446

# The following function is equivalent to equations 2a to 2e
# i.e. the total number of groups,
# number of groups with A on Site 1, 2, 3 and 4.
# Solves the system of equations in (9)
def fns(u, n, Appn, T):
    a = np.exp(-u/(R*T))
    def solfn(x):
        return np.array([-n + x[0]*(a[0]*x[1] + a[1] + a[10]*x[3]*x[4] + a[11]*x[3]*x[1]*x[2]
                                    + a[12]*x[4]*x[1]*x[2] + a[13]*x[3]*x[4]*x[1] + a[14]*x[3]*x[4]*x[2] + a[15]*x[3]*x[4]*x[1]*x[2]
                                    + a[2]*x[2] + a[3]*x[3] + a[4]*x[4] + a[5]*x[1]*x[2] + a[6]*x[3]*x[1] + a[7]*x[4]*x[1]
                                    + a[8]*x[3]*x[2] + a[9]*x[4]*x[2]),
                         -Appn[0] + x[0]*x[1]*(a[0] + a[5]*x[2] + a[6]*x[3] + a[7]*x[4] + a[11]*x[2]*x[3] + a[12]*x[2]*x[4] + a[13]*x[3]*x[4] + a[15]*x[2]*x[3]*x[4]),
                         -Appn[1] + x[0]*x[2]*(a[2] + a[5]*x[1] + a[8]*x[3] + a[9]*x[4] + a[11]*x[1]*x[3] + a[12]*x[1]*x[4] + a[14]*x[3]*x[4] + a[15]*x[1]*x[3]*x[4]),
                         -Appn[2] + x[0]*x[3]*(a[3] + a[6]*x[1] + a[8]*x[2] + a[10]*x[4] + a[11]*x[1]*x[2] + a[13]*x[1]*x[4] + a[14]*x[2]*x[4] + a[15]*x[1]*x[2]*x[4]),
                         -Appn[3] + x[0]*x[4]*(a[4] + a[7]*x[1] + a[9]*x[2] + a[10]*x[3] + a[12]*x[1]*x[2] + a[13]*x[1]*x[3] + a[14]*x[2]*x[3] + a[15]*x[1]*x[2]*x[3])])
    return solfn

# User inputs
#fac = 0.0000001; ys = np.linspace(0.93, 0.23, 41)
#fac = 0.4; ys = np.linspace(0.93, 0.23, 41)
fac = 0.8; ys = np.linspace(0.93, 0.23, 41)
fac=1.5; ys = np.linspace(0.73, 0.23, 41)

T = 6000./R
u0 = 0.
e1 = 4000.*fac
e2 = 8000.*fac
n = 1.

# Table 1.
u = u0 + np.array([0., 0.25*e1, 0.5*e1, 0.5*e1, 0.5*e1,
                   0.25*e1 + e2, 0.25*e1, 0.25*e1 + e2,
                   0.75*e1 + e2, 0.75*e1, 0.75*e1 + e2,
                   0.5*e1 + 2.*e2, 0.5*e1 + 2.*e2,
                   0.5*e1 + 2.*e2, e1 + 2.*e2, 0.75*e1 + 4.*e2])

np.random.seed(1000)
for i in range(len(u)):
    u[i] *= np.random.rand()



def calculate_squared_reaction_energies(E_mbr, u):
#we're looking for all the clusters contained in a cluster pair
    ones = np.ones(len(E_mbr))
    diffs = (np.einsum('ij, k, l -> iklj', E_mbr, ones, ones)
             + np.einsum('ij, k, l -> kilj', E_mbr, ones, ones)
             - np.einsum('ij, k, l -> klij', E_mbr, ones, ones)).min(axis=3) + 1
    n_rxn_clusters = diffs.sum(axis=2)

    pair_energies = np.einsum('i, j->ij', u, ones) + np.einsum('i, j->ji', u, ones)
    rxn_cluster_energies = np.einsum('ijk, k', diffs, u)
    return (pair_energies - 2.*rxn_cluster_energies/n_rxn_clusters)**2

half_u_rxn_sqr = calculate_squared_reaction_energies(E_mbr, u)/2.

a = np.exp(-u/(R*T))

"""
pcl_i*u_i
pcl_i*I_ij*u_j

pcl_i.dot(E_ind) = p_ind

pcl_i*u_i = p_ind.dot(E_ind).dot(u_i)
u_ind = E_ind.dot(u_i)
"""

def calculate_energies(ps):
    pcl = np.array([np.prod([p if E_mbr[i][j] > 1.e-10 else 1. for j, p in enumerate(cl)])
                    for i, cl in enumerate(np.einsum('ij, j -> ij', E_mbr, ps))])


    ideal = np.sum(ps*np.log(ps)) # same as position in solution space when energies are equal to zero
    non_ideal = np.sum(pcl*u)/(R*T)
    non_ideal_2 = -np.einsum('i, ij, j ->', pcl, half_u_rxn_sqr/((R*T)**2), pcl)
    cluster_energies = ideal + non_ideal + non_ideal_2

    log_ideal_activities = (E_mbr * logish(ps)).sum(-1)

    #print(log_ideal_activities)
    #log_cluster_activities = log_ideal_activities + log_nonideal_activities + log_nonideal_activities_2

    return cluster_energies , 0. # log_cluster_activities

xs = np.linspace(0.07, 0.93, 1001)
Es = []
for x in xs:
    ps = np.array([x, 1.-x, 1.-x, x,
                   1.-x, x, 1.-x, x])
    E0, a0 = calculate_energies(ps)
    Es.append(E0)

plt.plot(xs, Es)

# Between SAAA and ASSS

for x in [0.4, 0.7]:
    ps = np.array([x, 1.-x, 1.-x, x,
                   1.-x, x, 1.-x, x])
    E0, a0 = calculate_energies(ps)
    plt.scatter([x], [E0])
    ab = a0[names.index('SAAA')]
    ae = a0[names.index('ASSS')]
    am = x*ae + (1. - x)*ab
    plt.scatter([0, x, 1], [ab, am, ae])
    plt.plot([0, x, 1], [ab, am, ae])



plt.show()
