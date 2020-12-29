import numpy as np

W = np.array([[0., 2., 4.],
              [0., 0., 10.],
              [0., 0., 0.]])

p0 = np.array([0.3, 0.6, 0.2])

"""
dps = np.identity(3)*1.e-8
E0 = np.einsum('i, j, ij', p0, p0, W)
dEdp = np.zeros(3)

for i, dp in enumerate(dps):
    p = p0 + dp
    E = np.einsum('i, j, ij', p, p, W)

    dEdp[i] = (E - E0)/1.e-8 - E
print(dEdp)
"""


id_minus_p = np.identity(3) - np.einsum('i, j->ij', np.ones(3), p0)
print(-np.einsum('li, lj, ij->l', id_minus_p, id_minus_p, W))

mus = []
for l in range(3):
    mu = 0
    for i in range(3):
        for j in range(3):
            dil = 1. if i == l else 0.
            djl = 1. if j == l else 0.
            mu += -(dil - p0[i])*(djl - p0[j])*W[i, j]
    mus.append(mu)
print(mus)



def non_ideal_hessian(p, n_endmembers, W):
    q = np.eye(n_endmembers) - p*np.ones((n_endmembers, n_endmembers))
    hess = np.einsum('ij, jk, mk->im', q, W, q)
    hess += hess.T
    return hess


def non_ideal_interactions(p, n_endmembers, W):
    # -sum(sum(qi.qj.Wij*)
    # equation (2) of Holland and Powell 2003
    q = np.eye(n_endmembers) - p*np.ones((n_endmembers, n_endmembers))
    # The following are equivalent to
    # np.einsum('ij, jk, ik->i', -q, self.Wx, q)
    Wint = -(q.dot(W)*q).sum(-1)
    return Wint


W_star = np.array([[0., 1., 0., 1., 0., 1., 0., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 1., 0., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]])
