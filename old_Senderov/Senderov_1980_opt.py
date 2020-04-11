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

"""
E_ind = []
for mbr in E_mbr:
    E_ind.append(np.linalg.lstsq(E_mbr[:5].T, mbr, rcond=None)[0].round(decimals=10))

E_ind = np.array(E_ind)
"""

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
#fac = 0.01; ys = np.linspace(0.93, 0.23, 41)
#fac = 0.4; ys = np.linspace(0.93, 0.23, 41)
fac = 0.8; ys = np.linspace(0.93, 0.23, 41)
#fac=1.5; ys = np.linspace(0.73, 0.23, 41)

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

fig = plt.figure(figsize=(12,12))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


us = np.array([u[names.index('SASS')],
               u[names.index('ASSS')],
               u[names.index('SSSS')],
               u[names.index('AASS')]])/(R*T)
ax[1].scatter([0., 1., 0., 1.], us)


for (xb, c) in [(0, 'red'),
                (1, 'orange'),
                (2, 'blue'),
                (3, 'purple')]:
    numerical_solution = []
    ideal_model = []
    nonideal_1st_order = []
    nonideal_2nd_order = []
    for i, y in enumerate(ys):
        #print(y)
        if xb == 0:
            Appn = np.array([1.-y, 1.-y, 1.-y, 1.-y]) # proportion of A on each site
        if xb == 1:
            Appn = np.array([1.-y, 1.-y, 1.-y, 0.0001]) # proportion of A on each site
        if xb == 2:
            Appn = np.array([y, y, 0.0001, 0.0001]) # proportion of A on each site
        if xb == 3:
            Appn = np.array([y, 1.-y, 0.0001, 0.0001]) # proportion of A on each site

        if run_inversion:
            # x = [x, y, z, p, q], the independent Lagrangian multipliers (Equation 7)
            if i == 0:
                x = newton_krylov(fns(u, n, Appn, T), [1., 1., 1., 1., 1.],
                                  method='cgs')
            else:
                x = newton_krylov(fns(u, n, Appn, T), x,
                                  method='cgs', f_tol=1.e-7)


            #print(np.sum(M[A0_indices]), np.sum(M[A1_indices]),
            #          np.sum(M[A2_indices]), np.sum(M[A3_indices]))


            # Table 6
            xfn = x[0]*np.array([x[1], 1., x[2], x[3], x[4],

                                 x[1]*x[2], x[1]*x[3], x[1]*x[4],
                                 x[2]*x[3], x[2]*x[4],
                                 x[3]*x[4],

                                 x[1]*x[2]*x[3], x[1]*x[2]*x[4], x[1]*x[3]*x[4],
                                 x[2]*x[3]*x[4],

                                 x[1]*x[2]*x[3]*x[4]])

            # components of the partition function, phi (Equation 8b)
            pfn = xfn*a
            M = pfn/np.sum(pfn) # fractions of different clusters

            # position in solution space
            # phi - psi, Equation 10.
            arr = np.array([1., Appn[0], Appn[1], Appn[2], Appn[3]])
            p = np.sum(arr*np.log(x))
        else:
            p=0

        #p = np.sum(M*np.log(xfn)) # also position in solution space
        #print(np.sum(M*np.log(M))) # not the same as position in solution space!

        # Proportions of A, S
        ps = np.array([Appn[0], 1.-Appn[0],
                       Appn[1], 1.-Appn[1],
                       Appn[2], 1.-Appn[2],
                       Appn[3], 1.-Appn[3]])

        ideal = np.sum(ps*np.log(ps)) # same as position in solution space when energies are equal to zero
        pcl = np.array([np.prod([p if E_mbr[i][j] > 1.e-10 else 1. for j, p in enumerate(cl)])
                        for i, cl in enumerate(np.einsum('ij, j -> ij', E_mbr, ps))])
        non_ideal = np.sum(pcl*u)/(R*T)
        non_ideal_2 = -np.einsum('i, ij, j ->', pcl, half_u_rxn_sqr/((R*T)**2), pcl)

        numerical_solution.append(p)
        ideal_model.append(ideal)
        nonideal_1st_order.append(non_ideal)
        nonideal_2nd_order.append(non_ideal_2)
        cluster_energies = ideal + non_ideal + non_ideal_2

    end = time.time()
    print(end - start)

    numerical_solution = np.array(numerical_solution)
    ideal_model = np.array(ideal_model)
    nonideal_1st_order = np.array(nonideal_1st_order)
    nonideal_2nd_order = np.array(nonideal_2nd_order)

    ax[0].plot(ys, numerical_solution, label='numerical solution', c=c)
    ax[0].plot(ys, ideal_model, label='ideal', c=c, linestyle=':')
    ax[0].plot(ys, ideal_model+nonideal_1st_order, label='ideal + 1st order non-ideal', c=c, linestyle='--')
    ax[0].plot(ys, ideal_model+nonideal_1st_order+nonideal_2nd_order, label='ideal + 2nd order non-ideal', linestyle=':', c=c)

    ax[1].plot(ys, numerical_solution-ideal_model, label='numerical solution (non-ideal)', c=c)
    ax[1].plot(ys, nonideal_1st_order, label='ideal + 1st order non-ideal', c=c, linestyle='--')
    ax[1].plot(ys, nonideal_1st_order+nonideal_2nd_order, label='ideal + 2nd order non-ideal', linestyle=':', c=c)


    ax[2].plot(ys, numerical_solution-(ideal_model+nonideal_1st_order), label='numerical solution (2nd order non-ideal)', c=c)
    ax[2].plot(ys, nonideal_2nd_order, label='2nd order non-ideal', c=c, linestyle=':')

    ax[3].plot(ys, numerical_solution-(ideal_model+nonideal_1st_order+nonideal_2nd_order), label='3rd order error', c=c)

for i in range(4):
    ax[i].legend()
    ax[i].set_xlim(0., 1.)


plt.show()
