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

connectivity = np.array([[0., 1., 1., 0.],
                         [1., 0., 0., 1.],
                         [0., 1., 1., 0.],
                         [1., 0., 0., 1.]])

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
    lnval = E_ind.dot(mu) - u/(R*T)
    return np.exp(np.where(lnval>100, 100., lnval))

def delta_proportions_root(mu, p_ind, T):
    Mbar = cluster_proportions(mu, T)
    deltas = p_ind - E_ind.T.dot(Mbar)
    return deltas

#p_ind = E_ind.T.dot(np.exp(E_ind.dot(mu) - u/(R*T)))
#p_ind2 - p_ind = E_ind.T.dot(np.exp(E_ind.dot(mu + dmu) - u/(R*T)) -
#                             np.exp(E_ind.dot(mu) - u/(R*T)))
#dp_ind = E_ind.T.dot(E_ind.dot(dmu)*np.exp(E_ind.dot(mu) - u/(R*T))) ????

#pinv(E_ind).dot((pinv(E_ind.T).dot(dp_ind))*np.exp(-E_ind.dot(mu) + u/(R*T))) = dmu

def delta_mu_rxns_root(mu_rxn, p_elements, T):
    mu = mu_rxn[0:5]
    rxn = mu_rxn[5:8]
    p_ind = rxn_matrix.dot(np.concatenate((p_elements, rxn)))

    Mbar = cluster_proportions(mu, T)
    deltas = p_ind - E_ind.T.dot(Mbar)

    # TODO: not totally sure that these are the correct mus.
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

def guess(ps, T):
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

e1 = 4000.
e2 = 8000.
u = cluster_energies(e1, e2)

T = 1000.
p_elements = np.array([0.75, 0.25])
sol = root(delta_mu_rxns_root, [0., 0., 0., 0., 0., 0., 0., 0.],
           args=(p_elements, T))
p_ind = E_ind.T.dot(cluster_proportions(sol.x[0:5], T))
#print(p_ind)
#p_ind = np.array([0.10, 0.20, 0.40, 0.20, 0.1])
zeros = np.zeros(5)
sol = root(delta_proportions_root, zeros, args=(p_ind, T))
p_cl = cluster_proportions(sol.x[0:5], T)
G_cl = np.sum(p_cl*u)

delta = 1.e-6
p_ind2 = (p_ind + delta*np.identity(5))/(1.+delta)
sol2 = [root(delta_proportions_root, zeros, args=(p2, T)) for p2 in p_ind2]

p_cls = [cluster_proportions(s2.x[0:5], T) for s2 in sol2]
G_cl2 = [np.sum(cluster_proportions(s2.x[0:5], T)*u) for s2 in sol2]

"""
# have to find dcluster_props given dinds
C_ind = np.linalg.lstsq(p_ind2 - p_ind,
                        p_cls - p_cl,
                        rcond=None)[0].round(decimals=10).T

print(C_ind.dot(p_ind2[0] - p_ind))
print(p_cls[0]-p_cl)
"""
#print((sol2[0].x[0:5] - sol.x[0:5]))
#print(np.linalg.pinv(E_ind).dot((np.linalg.pinv(E_ind.T).dot(p_ind2[0] - p_ind))
#                                * np.exp(-E_ind.dot(sol.x[0:5]) + u/(R*T))))
#exit()

mus = G_cl + (G_cl2 - G_cl)/delta
print(G_cl)
print(mus)
print(G_cl - np.sum(mus*p_ind))

exit()
ps = element_occupancies_ind.T.dot(p_ind)
print(E_ind[independent_indices[0]],
      (u[independent_indices[0]]))

print(E_ind.T.dot(u))
print(R*T*element_occupancies_ind.dot(logish(ps)))
exit()

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
