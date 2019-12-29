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

R = 8.31446

# The following function is equivalent to equations 2a to 2e
# i.e. the total number of groups,
# number of groups with A on Site 1, 2, 3 and 4.
# Solves the system of equations in (9)
def fns(u, n, Appn, T):
    a = np.exp(-u/(R*T))
    def solfn(x):
        f = x[0]*np.einsum('i, j, k, l, ijkl->ijkl',
                           [x[1], 1.], [x[2], 1.], [x[3], 1.], [x[4], 1.], a)
        return np.array([-n + np.sum(f),
                         -Appn[0] + np.sum(f[0,:,:,:]),
                         -Appn[1] + np.sum(f[:,0,:,:]),
                         -Appn[2] + np.sum(f[:,:,0,:]),
                         -Appn[3] + np.sum(f[:,:,:,0])])
    return solfn

# User inputs
#fac = 0.01; ys = np.linspace(0.93, 0.23, 41)
#fac = 0.4; ys = np.linspace(0.93, 0.23, 41)
fac = 0.8; ys = np.linspace(0.93, 0.23, 41)
#fac=1.5; ys = np.linspace(0.73, 0.23, 41)

T = 6000./R
u0 = 0.
e1 = 0000.*fac
e2 = 0000.*fac
n = 1.

# Ordering as in Table 1
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

a = np.exp(-u/(R*T))


def calculate_squared_reaction_energies(u):
    #we're looking for all the clusters contained in a cluster pair

    ones = np.ones_like(u)
    n_elements = u.shape[0]

    id = np.identity(n_elements)
    pair_energies = (np.einsum('ijkl, mnop-> ijklmnop', u, ones) +
                     np.einsum('ijkl, mnop-> ijklmnop', ones, u))

    # the next line is not yet correct
    # possibly use the stuff above to select elements (with deltas)
    o = np.ones((2, 2))
    rxn_cluster_energies = (np.einsum('im, jn, ko, lp, ijkl -> ijklmnop',   o,    o,    o,    o,    u)

                            + np.einsum('im, jn, ko, lp, mjkl -> ijklmnop', 1-id, o,    o,    o,    u)
                            + np.einsum('im, jn, ko, lp, inkl -> ijklmnop', o,    1-id, o,    o,    u)
                            + np.einsum('im, jn, ko, lp, ijol -> ijklmnop', o,    o,    1-id, o,    u)
                            + np.einsum('im, jn, ko, lp, ijkp -> ijklmnop', o,    o,    o,    1-id, u)

                            + np.einsum('im, jn, ko, lp, mnkl -> ijklmnop', 1-id, 1-id, o,    o,    u)
                            + np.einsum('im, jn, ko, lp, mjol -> ijklmnop', 1-id, o,    1-id, o,    u)
                            + np.einsum('im, jn, ko, lp, mjkp -> ijklmnop', 1-id, o,    o,    1-id, u)
                            + np.einsum('im, jn, ko, lp, inol -> ijklmnop', o,    1-id, 1-id, o,    u)
                            + np.einsum('im, jn, ko, lp, inkp -> ijklmnop', o,    1-id, o,    1-id, u)
                            + np.einsum('im, jn, ko, lp, ijop -> ijklmnop', o,    o,    1-id, 1-id, u)


                            + np.einsum('im, jn, ko, lp, mnol -> ijklmnop', 1-id, 1-id, 1-id, o,    u)
                            + np.einsum('im, jn, ko, lp, mnkp -> ijklmnop', 1-id, 1-id, o, 1-id,    u)
                            + np.einsum('im, jn, ko, lp, mjop -> ijklmnop', 1-id, o, 1-id, 1-id,    u)
                            + np.einsum('im, jn, ko, lp, inop -> ijklmnop', o, 1-id, 1-id, 1-id,    u)

                            + np.einsum('im, jn, ko, lp, mnop -> ijklmnop', 1-id, 1-id, 1-id, 1-id,    u))


    big_ones = np.ones_like(rxn_cluster_energies)
    n_rxn_clusters = (np.einsum('im, ijklmnop -> ijklmnop', 1-id, big_ones)
                      + np.einsum('jn, ijklmnop -> ijklmnop', 1-id, big_ones)
                      + np.einsum('ko, ijklmnop -> ijklmnop', 1-id, big_ones)
                      + np.einsum('lp, ijklmnop -> ijklmnop', 1-id, big_ones))

    n_rxn_clusters = np.power(2, n_rxn_clusters)

    return (pair_energies - 2.*rxn_cluster_energies/n_rxn_clusters)**2

half_u_rxn_sqr = calculate_squared_reaction_energies(u)/2.



fig = plt.figure(figsize=(12,12))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]



for (xb, c) in [(0, 'red'),
                (1, 'orange'),
                (2, 'blue'),
                (3, 'purple')]:
    numerical_solution = []
    ideal_model = []
    nonideal_1st_order = []
    nonideal_2nd_order = []
    bless = []
    for i, y in enumerate(ys):
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
                                  method='cgs', f_tol=1.e-8)
            else:
                x = newton_krylov(fns(u, n, Appn, T), x,
                                  method='cgs', f_tol=1.e-8)

            # Table 6
            xfn = x[0]*np.einsum('i, j, k, l->ijkl',
                                 [x[1], 1.], [x[2], 1.], [x[3], 1.], [x[4], 1.])

            # components of the partition function, phi (Equation 8b)
            pfn = xfn*a
            M = pfn/np.sum(pfn) # fractions of different clusters
            #print(M)
            #print(x[1], x[2])
            # position in solution space
            # phi - psi, Equation 10.
            arr = np.array([1., Appn[0], Appn[1], Appn[2], Appn[3]])
            p = np.sum(arr*np.log(x))
            #print(arr[0], arr[1], np.log(x[1]))
            #p = np.sum(M*np.log(xfn)) # also position in solution space
        else:
            p=0

        # Proportions of A, S
        ps = np.array([[Appn[0], 1.-Appn[0]],
                       [Appn[1], 1.-Appn[1]],
                       [Appn[2], 1.-Appn[2]],
                       [Appn[3], 1.-Appn[3]]])

        ideal = np.sum(ps*np.log(ps)) # same as position in solution space when energies are equal to zero
        non_ideal = np.einsum('i, j, k, l, ijkl->', ps[0], ps[1], ps[2], ps[3], u/(R*T))
        non_ideal_2 = -np.einsum('i, j, k, l, m, n, o, p, ijklmnop ->',
                                 ps[0], ps[1], ps[2], ps[3],
                                 ps[0], ps[1], ps[2], ps[3],
                                 half_u_rxn_sqr/((R*T)**2))
        cluster_energies = ideal + non_ideal + non_ideal_2

        # log activities for each cluster
        id = np.identity(2)
        id2 = np.einsum('im, jn, ko, lp->ijklmnop', id, id, id, id)
        log_ideal_activities = (np.einsum('ijklmnop, m->ijkl', id2, logish(ps[0]))
                                + np.einsum('ijklmnop, n->ijkl', id2, logish(ps[1]))
                                + np.einsum('ijklmnop, o->ijkl', id2, logish(ps[2]))
                                + np.einsum('ijklmnop, p->ijkl', id2, logish(ps[3])))


        # CORRECT? NEED TO CHECK
        o = np.ones(2)
        idooo = np.einsum('im, j, k, l->imjkl', id, o, o, o)
        log_nonideal_activities = (np.einsum('imjkl, n, o, p, mnop->ijkl', idooo, ps[1], ps[2], ps[3], u/(R*T))
                                   + np.einsum('jnikl, m, o, p, mnop->ijkl', idooo, ps[1], ps[2], ps[3], u/(R*T))
                                   + np.einsum('koijl, m, n, p, mnop->ijkl', idooo, ps[1], ps[2], ps[3], u/(R*T))
                                   + np.einsum('lpijk, m, n, o, mnop->ijkl', idooo, ps[1], ps[2], ps[3], u/(R*T))
                                   )

        # not correct yet!!
        ursq = half_u_rxn_sqr/((R*T)**2)
        log_nonideal_activities_2 = -(np.einsum('imjkl, n, o, p, q, r, s, t, mnopqrst->ijkl', idooo, ps[1], ps[2], ps[3], ps[0], ps[1], ps[2], ps[3], ursq)
                                      + np.einsum('jnikl, m, o, p, q, r, s, t, mnopqrst->ijkl', idooo, ps[0], ps[2], ps[3], ps[0], ps[1], ps[2], ps[3], ursq)
                                      + np.einsum('koijl, m, n, p, q, r, s, t, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[3], ps[0], ps[1], ps[2], ps[3], ursq)
                                      + np.einsum('lpijk, m, n, o, q, r, s, t, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[2], ps[0], ps[1], ps[2], ps[3], ursq)
                                      + np.einsum('iqjkl, m, n, o, p, r, s, t, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[2], ps[3], ps[1], ps[2], ps[3], ursq)
                                      + np.einsum('jrikl, m, n, o, p, q, s, t, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[2], ps[3], ps[0], ps[2], ps[3], ursq)
                                      + np.einsum('ksijl, m, n, o, p, q, r, t, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[2], ps[3], ps[0], ps[1], ps[3], ursq)
                                      + np.einsum('ltijk, m, n, o, p, q, r, s, mnopqrst->ijkl', idooo, ps[0], ps[1], ps[2], ps[3], ps[0], ps[1], ps[2], ursq)
                                      )
        #log_nonideal_activities_2 = 0.
        log_activities = log_ideal_activities + log_nonideal_activities + log_nonideal_activities_2
        mod = cluster_energies - np.einsum('i, j, k, l, ijkl->', ps[0], ps[1], ps[2], ps[3], log_activities)
        log_activities += mod

        activities = np.exp(log_activities)
        print('this definition is not correct! need to find how to determine x from these derivatives. Start with ideal model first!!')
        M2 = activities/np.sum(activities)

        # Arr*log(x) = p

        #print((M2))
        #exit()
        numerical_solution.append(p)
        ideal_model.append(ideal)
        nonideal_1st_order.append(non_ideal)
        nonideal_2nd_order.append(non_ideal_2)

        if i == 20:
            ax[0].scatter([y], [cluster_energies], c=c)
            if xb==0:
                ax[0].plot([0, 1], [log_activities[0,0,0,0], log_activities[1,1,1,1]], c=c)
            if xb==1:
                ax[0].plot([0, 1], [log_activities[0,0,0,1], log_activities[1,1,1,1]], c=c)
            if xb==2:
                ax[0].plot([0, 1], [log_activities[1,1,1,1], log_activities[0,0,1,1]], c=c)
            if xb==3:
                ax[0].plot([0, 1], [log_activities[1,0,1,1], log_activities[0,1,1,1]], c=c)

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
