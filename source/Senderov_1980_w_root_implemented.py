import numpy as np
from scipy.optimize import minimize, root
from sympy.matrices import Matrix
import matplotlib.pyplot as plt

fac=1.2; ys = np.linspace(0.99, 0.5, 21)

R = 8.31446
T = 6000./R
u0 = 0.
e1 = 4000.*fac
e2 = 8000.*fac
n = 1.

# Ordering as in Table 1
n_species = 2
u = np.zeros((n_species, n_species,
              n_species, n_species)) + u0
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


id = np.identity(n_species, dtype='int')
ones = np.ones((n_species,n_species,n_species), dtype='int')
element_occupancies = np.array([np.einsum('im, jkl -> ijklm', id, ones),
                                np.einsum('jm, ikl -> ijklm', id, ones),
                                np.einsum('km, ijl -> ijklm', id, ones),
                                np.einsum('lm, ijk -> ijklm', id, ones)], dtype='int')
element_occupancies = np.moveaxis(element_occupancies, 0, -2)
element_occupancies = element_occupancies.reshape((16,8))
u = u.flatten()

occ = Matrix(element_occupancies)
element_occupancies_ind = np.array(occ.T.columnspace(), dtype='int')
"""
element_occupancies_ind = np.array([[1, 0, 0, 1, 0, 1, 0, 1],
                                    [0, 1, 1, 0, 0, 1, 0, 1],
                                    [0, 1, 0, 1, 1, 0, 0, 1],
                                    [0, 1, 0, 1, 0, 1, 1, 0],
                                    [0, 1, 0, 1, 0, 1, 0, 1]])
"""
E_ind = np.linalg.lstsq(element_occupancies_ind.T,
                        element_occupancies.T,
                        rcond=None)[0].round(decimals=10).T

def cluster_proportions(mu, T):
    lnval = E_ind.dot(mu) - u/(R*T)
    return np.exp(np.where(lnval>100, 100., lnval))

def delta_proportions_minimize(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    deltas = p_ind - E_ind.T.dot(p_cl)
    return np.sum(deltas**2)

def delta_proportions_root(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    deltas = p_ind - E_ind.T.dot(p_cl)
    return deltas

def inv_J(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    return np.einsum('ik, ij, i -> kj', E_ind, E_ind, p_cl)

def dpcl_dp_ind(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    dp_ind_dmu = np.einsum('ik, ij, i -> kj', E_ind, E_ind, p_cl)
    dpcl_dmu = np.einsum('ij, i -> ij', E_ind, p_cl)

    # the returned solve is equivalent to
    # np.einsum('lj, jk -> lk', dpcl_dmu, np.linalg.pinv(dp_ind_dmu)))
    return np.linalg.solve(dp_ind_dmu.T, dpcl_dmu.T).T

def chemical_potentials(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    ps = element_occupancies_ind.T.dot(p_ind)
    D = dpcl_dp_ind(mu, p_ind, T)
    g_E = np.einsum('lk, l', D, u)
    g_S_n = -R*np.einsum('lk, l', D, logish(p_cl) + np.sum(p_cl))
    g_S_i = -R*(np.einsum('ik, i', element_occupancies_ind.T, logish(ps)) + np.sum(ps))
    g_S_t = gamma*g_S_n + (1./n - gamma)*g_S_i
    print(g_S_t)
    exit()
    mu = g_E - T*g_S_t
    print(element_occupancies_ind.T)
    print(n, gamma, g_S_i)
    #return np.array([act_G, act_E, act_S_t, act_S_i, act_S_n])
    return mu


def gibbs_entropies(mu, p_ind, T):
    p_cl = cluster_proportions(mu, T)
    ps = element_occupancies_ind.T.dot(p_ind)

    S_n = -R*np.sum(p_cl*logish(p_cl))
    S_i = -R*np.sum(ps*logish(ps))
    #print(S_n, S_i)
    #print(guess(p_ind, T))
    n = 1. # 1 cluster total
    E_t = np.sum(p_cl*u)
    S_t = gamma*S_n + (1./n - gamma)*S_i

    G_t = E_t - T*S_t

    #return np.array([G_t, E_t, S_t, S_i, S_n])
    return G_t

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

def guess(p_ind, T):
    ps = element_occupancies_ind.T.dot(p_ind)
    ideal_mu = element_occupancies_ind.dot(logish(ps))

    uRT = u/(R*T)
    p = np.array([np.einsum('ij, j->i', element_occupancies[:,0:2], ps[0:2]),
                  np.einsum('ij, j->i', element_occupancies[:,2:4], ps[2:4]),
                  np.einsum('ij, j->i', element_occupancies[:,4:6], ps[4:6]),
                  np.einsum('ij, j->i', element_occupancies[:,6:8], ps[6:8])])

    pcl = np.array([np.einsum('ij, jk->ki', element_occupancies[:,0:2], element_occupancies_ind.T[0:2]),
                    np.einsum('ij, jk->ki', element_occupancies[:,2:4], element_occupancies_ind.T[2:4]),
                    np.einsum('ij, jk->ki', element_occupancies[:,4:6], element_occupancies_ind.T[4:6]),
                    np.einsum('ij, jk->ki', element_occupancies[:,6:8], element_occupancies_ind.T[6:8])])
    nonideal_mu = (np.einsum('ij, j, j, j, j->i', pcl[0], p[1], p[2], p[3], uRT)
                   + np.einsum('ij, j, j, j, j->i', pcl[1], p[0], p[2], p[3], uRT)
                   + np.einsum('ij, j, j, j, j->i', pcl[2], p[0], p[1], p[3], uRT)
                   + np.einsum('ij, j, j, j, j->i', pcl[3], p[0], p[1], p[2], uRT)
                   - 3.*np.einsum('j, j, j, j, j->', p[0], p[1], p[2], p[3], uRT))

    return (ideal_mu + nonideal_mu) + np.random.rand(len(ideal_mu))*1.e-2



ys = np.linspace(0.4, 1., 101)

# S_n = -2.*0.5*np.log(0.5)
# S_i = -8.*0.5*np.log(0.5)
gamma = 4./3. # S_i/(S_i-S_n)

ps_to_p_ind = np.linalg.pinv(element_occupancies_ind.T)

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

for T in np.linspace(1600., 2000., 39):
    print('\r{0} K'.format(T), end='')
    sols = []
    for y in ys:
        ps = np.array([y, 1.-y,
                       y, 1.-y,
                       y, 1.-y,
                       y, 1.-y])
        p_ind = ps_to_p_ind.dot(ps)
        sol = root(delta_proportions_root, guess(p_ind, T), args=(p_ind, T))

        for l in np.identity(5)*1.e-8:
            sol2 = root(delta_proportions_root,
                        guess(p_ind + l, T),
                        args=(p_ind + l, T))
            print((sol2.x - sol.x)*1.e8)
        print(np.linalg.pinv(inv_J(sol.x, p_ind, T)))

        pcl = cluster_proportions(sol.x, T)

        for l in np.identity(5)*1.e-8:
            sol2 = root(delta_proportions_root,
                        guess(p_ind + l, T),
                        args=(p_ind + l, T))
            pcl2 = cluster_proportions(sol2.x, T)
            print((pcl2 - pcl)*1.e8)

        print(dpcl_dp_ind(sol.x, p_ind, T).T)

        for l in np.identity(5)*1.e-8:
            sol2 = root(delta_proportions_root,
                        guess(p_ind + l, T),
                        args=(p_ind + l, T))
            print((gibbs_entropies(sol2.x, p_ind + l, T)
                   - gibbs_entropies(sol.x, p_ind, T))*1.e8)


        print(chemical_potentials(sol.x, p_ind, T))

        print(np.einsum('i, i', p_ind, chemical_potentials(sol.x, p_ind, T)))
        print(gibbs_entropies(sol.x, p_ind, T))



        exit()
        if sol.success or np.max(np.abs(sol.fun)) < 1.e-10: #or sol.fun < 1.e-6:
            mu = sol.x
            #mu = guess(p_ind, T)
            p_cl = cluster_proportions(mu, T)
            S_n = -R*np.sum(p_cl*logish(p_cl))
            S_i = -R*np.sum(ps*logish(ps))
            #print(S_n, S_i)
            #print(guess(p_ind, T))
            n = 1. # 1 cluster total
            S_t = gamma*S_n + (1./n - gamma)*S_i

            G_t = np.sum(p_cl*u) - T*S_t
            sols.append([y, S_n, S_i, S_t, G_t])
        else:
            print(sol.message)

    sols = np.array(sols).T
    if T%100 < 10.:
        c = 'black'
    else:
        c = 'red'

    linewidth = 0.5
    if T%200 < 10.:
        linestyle='-'
    else:
        linestyle=':'

    ax[0].plot(sols[0], sols[1], c=c, linewidth=linewidth, linestyle=linestyle)
    ax[1].plot(sols[0], sols[2], c=c, linewidth=linewidth, linestyle=linestyle)
    ax[2].plot(sols[0], sols[3], c=c, linewidth=linewidth, linestyle=linestyle)
    ax[3].plot(sols[0], sols[4], c=c, linewidth=linewidth, linestyle=linestyle)
print()

for i in range(4):
    ax[i].set_xlim(0,1)
for i in range(3):
    ax[i].set_ylim(0,)

fig.tight_layout()
fig.savefig('model.pdf')
plt.show()
