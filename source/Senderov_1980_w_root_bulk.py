import numpy as np
from scipy.optimize import root
from sympy.matrices import Matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fac=1.2; ys = np.linspace(0.99, 0.5, 21)

R = 8.31446
T = 6000./R
u0 = 0.
e1 = 3000.*fac
e2 = 6000.*fac
n = 1.

n_species = 2

# e1 is the energy preference for different positions in the lattice
# e2 is the difference between interaction of similar and dissimilar
def cluster_energies(e1, e2):
    # Ordering as in Table 1
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
    u = u.flatten()
    return u

id = np.identity(n_species, dtype='int')
ones = np.ones((n_species,n_species,n_species), dtype='int')
element_occupancies = np.array([np.einsum('im, jkl -> ijklm', id, ones),
                                np.einsum('jm, ikl -> ijklm', id, ones),
                                np.einsum('km, ijl -> ijklm', id, ones),
                                np.einsum('lm, ijk -> ijklm', id, ones)], dtype='int')
element_occupancies = np.moveaxis(element_occupancies, 0, -2)
element_occupancies = element_occupancies.reshape((16,8))


occ = Matrix(element_occupancies)
E_ind, independent_indices = occ.T.rref()
E_ind = np.array(E_ind, dtype='float').T[:,:len(independent_indices)]
element_occupancies_ind = np.array([element_occupancies[i]
                                    for i in independent_indices])

els = np.array([np.sum(element_occupancies_ind[:,0::2], axis=1)/4.,
                np.sum(element_occupancies_ind[:,1::2], axis=1)/4.])
null = np.array(Matrix(els).nullspace(), dtype='float')
inv_rxn_matrix = np.vstack((els, null))
rxn_matrix = np.linalg.pinv(inv_rxn_matrix)

"""
# This is the original formulation
element_occupancies_ind = np.array([[1, -1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, -1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, -1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, -1],
                                    [0, 1, 0, 1, 0, 1, 0, 1]])


inv_rxn_matrix = np.array([[0.25, 0.25, 0.25, 0.25, 0.],
                           [-0.25, -0.25, -0.25, -0.25, 1.],
                           [1., -1, 0., 0., 0.],
                           [1., 0., -1., 0., 0.],
                           [1., 0., 0., -1., 0.]])

rxn_matrix = np.linalg.pinv(inv_rxn_matrix)
E_ind = np.linalg.lstsq(element_occupancies_ind.T,
                        element_occupancies.T,
                        rcond=None)[0].round(decimals=10).T
"""

def cluster_proportions(mu, T):
    """
    Calculate the cluster proportions from a set of mus and temperature
    """
    lnval = E_ind.dot(mu) - u/(R*T)
    return np.exp(np.where(lnval>100, 100., lnval))


def delta_mu_rxns_root(mu_rxn, p_elements, T):
    mu = mu_rxn[0:5]
    rxn = mu_rxn[5:8]
    p_ind = rxn_matrix.dot(np.concatenate((p_elements, rxn)))

    p_cl = cluster_proportions(mu, T) # calculate the cluster proportions from the mus
    deltas = p_ind - E_ind.T.dot(p_cl) # compare the user-defined independent cluster proportions with the calculated values

    # TODO: these are certainly not the correct mus.
    affinities = inv_rxn_matrix[2:5].dot(mu)
    return np.concatenate((deltas, affinities))

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

def guess(p_elements, T):
    p_ind = rxn_matrix.dot([p_elements[0], p_elements[1], 0., 0., 0.])
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

    mu = (ideal_mu + nonideal_mu) + np.random.rand(len(ideal_mu))*1.e-2
    mu_rxn = np.concatenate((mu, [0., 0., 0.]))
    mu_rxn = np.zeros(8) # overwrite!!
    return mu_rxn


ys = np.linspace(0., 1., 101)

# S_n = -2.*0.5*np.log(0.5)
# S_i = -8.*0.5*np.log(0.5)
gamma = 4./3. # S_i/(S_i-S_n)

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

for T in np.linspace(100., 2000., 39):
    print('\r{0} K'.format(T), end='')
    sols = []
    for y in ys:
        p_elements = np.array([y, 1.-y])
        u = cluster_energies(e1, e2)

        # need to change things from here, spec. the guess function
        sol = root(delta_mu_rxns_root, guess(p_elements, T), args=(p_elements, T))

        if sol.success or np.max(np.abs(sol.fun)) < 1.e-10: #or sol.fun < 1.e-6:
            mu = sol.x[0:5]
            rxn = sol.x[5:8]
            p_ind = rxn_matrix.dot(np.concatenate((p_elements, rxn)))
            ps = element_occupancies_ind.T.dot(p_ind)

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

    ax[0].plot(sols[0], sols[1]/(4.*R), c=c, linewidth=linewidth, linestyle=linestyle)
    ax[1].plot(sols[0], sols[2]/(4.*R), c=c, linewidth=linewidth, linestyle=linestyle)
    ax[2].plot(sols[0], sols[3]/(4.*R), c=c, linewidth=linewidth, linestyle=linestyle)
    ax[3].plot(sols[0], sols[4], c=c, linewidth=linewidth, linestyle=linestyle)
print()

for i in range(4):
    ax[i].set_xlim(0,1)
for i in range(3):
    ax[i].set_ylim(0,)

fig.tight_layout()
fig.savefig('model.pdf')
plt.show()


gamma = 1.
e1s = 4.*R*np.linspace(1., 3000., 101)
y = 0.25

fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1,3)]

e0 = mpimg.imread('figures/e0.png')
ax[0].imshow(e0, extent=[0.0, 3000.0, 0.2, 1.], aspect='auto')
e40000 = mpimg.imread('figures/e40000.png')
ax[1].imshow(e40000, extent=[0.0, 3000.0, 0.2, 1.], aspect='auto')

for i, (e2, c) in enumerate([(0., 'red'), (R*40000., 'red')]):
    for T in [600., 800., 1000., 1200.]:
        sols = []
        for e1 in e1s:
            #print(e2)
            u = cluster_energies(e1, e2)*2. # factor two????...
            p_elements = np.array([y, 1.-y])
            sol = root(delta_mu_rxns_root, guess(p_elements, T), args=(p_elements, T))

            if sol.success: # or np.max(np.abs(sol.fun)) < 1.e-10: #or sol.fun < 1.e-6:
                mu = sol.x[0:5]
                rxn = sol.x[5:8]
                p_ind = rxn_matrix.dot(np.concatenate((p_elements, rxn)))
                ps = element_occupancies_ind.T.dot(p_ind)

                p_cl = cluster_proportions(mu, T)
                #print(p_ind)
                S_n = -R*np.sum(p_cl*logish(p_cl))
                S_i = -R*np.sum(ps*logish(ps))
                #print(S_n, S_i)
                #print(guess(p_ind, T))
                n = 1. # 1 cluster total
                S_t = gamma*S_n + (1./n - gamma)*S_i

                G_t = np.sum(p_cl*u) - T*S_t
                sols.append([e1/R/4., ps[0]])
            else:
                print(sol.message)

        sols = np.array(sols).T

        ax[i].plot(sols[0], sols[1], c=c, linestyle='--')
    ax[i].set_ylim(0.2, 1.)
plt.show()
