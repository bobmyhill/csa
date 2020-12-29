import numpy as np
from models.csasolutionmodel import CSAModel
from models.iucasolutionmodel import R, IUCAModel
from models.newmodel import NewModel
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def binary_cluster_energies(wAB):
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
    u[1, 1, 1, 1] = 0.

    u[0, 1, 1, 1] = 3.*wAB
    u[1, 0, 1, 1] = 3.*wAB
    u[1, 1, 0, 1] = 3.*wAB
    u[1, 1, 1, 0] = 3.*wAB

    u[0, 0, 1, 1] = 4.*wAB #*1.05
    u[0, 1, 0, 1] = 4.*wAB #*0.95
    u[0, 1, 1, 0] = 4.*wAB #*0.95
    u[1, 0, 0, 1] = 4.*wAB #*0.95
    u[1, 0, 1, 0] = 4.*wAB #*0.95
    u[1, 1, 0, 0] = 4.*wAB #*1.05

    u[0, 0, 0, 1] = 3.*wAB
    u[0, 0, 1, 0] = 3.*wAB
    u[0, 1, 0, 0] = 3.*wAB
    u[1, 0, 0, 0] = 3.*wAB

    u[0, 0, 0, 0] = 0.
    return u

wAB = -1.*R
T = 1.
cluster_energies = binary_cluster_energies(wAB*4.)
site_species = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B']]

ss = IUCAModel(cluster_energies, site_species, 3./4., [[0, 1, 2, 3]])

from numpy.linalg import pinv, lstsq
from itertools import combinations, product, permutations


halfE = np.einsum('ijkl, mnop', ss.cluster_ones, ss.cluster_energies)
halfE = (np.einsum('ijklmnop', halfE) + np.einsum('mnopijkl', halfE))/2.

perms = [permutations(s) for s in ['im', 'jn', 'ko', 'lp']]
perms = [[''.join(p) for p in t] for t in perms]
prod = list(product(*perms))
halfEmin = halfE.copy()
for p in prod:
    s = ''.join([s[0] for s in p])+''.join([s[1] for s in p])
    halfEmin = np.minimum(halfEmin, np.einsum('ijklmnop->'+s, halfE))

n_cl_sqr = ss.n_clusters*ss.n_clusters
halfEmin = halfEmin.reshape(n_cl_sqr,-1)

fik = np.einsum('ij, k', ss.cluster_occupancies, ss.cluster_ones.flatten())
fik = (np.einsum('ijk->ikj', fik) + np.einsum('kji->ikj', fik))/2.


fikE = np.hstack((fik.reshape(n_cl_sqr, ss.n_site_species),
                  halfEmin.reshape(n_cl_sqr,-1)))


fikE = np.unique(fikE, axis=0)
midclEs = fikE[:,-1]
midclps = fikE[:,:-1]


# The following lines find the combinations of site pairs on which ordering
# of each pair of components can take place
# The component pairs are stored as an iterable
# The site pairs are stored as a list of iterables, with each element of
# the list corresponding to a different component pair.

# Finally, a list of list of lists is provided, containing the indices of the
# site pairs as [[[site1, component1], [site1, component2],
# [site2, component1], [site2, component2]], ...].


component_pairs = combinations(ss.components, 2)
ordering_site_pairs = []
ordering_ssis = []
ordering_vectors = []

for c in component_pairs:
    sidx = [i for i, ss in enumerate(site_species) if c[0] in ss and c[1] in ss]
    ordering_site_pairs.append(combinations(sidx, 2))
    for pr in combinations(sidx, 2):
        ordering_vectors.append(np.zeros(ss.n_site_species, dtype='int'))
        ordering_vectors[-1][ss.site_start_indices[pr[0]] + ss.site_species[pr[0]].index(c[0])] = 1.
        ordering_vectors[-1][ss.site_start_indices[pr[0]] + ss.site_species[pr[0]].index(c[1])] = -1.
        ordering_vectors[-1][ss.site_start_indices[pr[1]] + ss.site_species[pr[1]].index(c[0])] = -1.
        ordering_vectors[-1][ss.site_start_indices[pr[1]] + ss.site_species[pr[1]].index(c[1])] = 1.

        ordering_ssis.append([[pr[0], ss.site_species[pr[0]].index(c[0])],
               [pr[0], ss.site_species[pr[0]].index(c[1])],
               [pr[1], ss.site_species[pr[1]].index(c[0])],
               [pr[1], ss.site_species[pr[1]].index(c[1])]])

ordering_vectors = np.array(ordering_vectors)

vord = np.einsum('ij, il, kl->kij', ordering_vectors, ordering_vectors,
                 ss.cluster_occupancies)

halfE = np.einsum('ijkl, mnop', ss.cluster_ones, ss.cluster_energies)
halfE = (np.einsum('ijklmnop', halfE) + np.einsum('mnopijkl', halfE))/2.

perms = [permutations(s) for s in ['im', 'jn', 'ko', 'lp']]
perms = [[''.join(p) for p in t] for t in perms]
prod = list(product(*perms))
halfEmin = halfE.copy()
for p in prod:
    s = ''.join([s[0] for s in p])+''.join([s[1] for s in p])
    halfEmin = np.minimum(halfEmin, np.einsum('ijklmnop->'+s, halfE))

Wfull = (halfEmin - halfE).reshape(16, 16)*2. + np.diag(ss.cluster_energies_flat)


def stable_clusters(ps):
    np.random.seed(seed=20)
    delta_ps = np.random.rand(len(ps))*1.e-8

    pf = np.einsum('kij, j->ki', vord, ps+delta_ps)
    stable = [i for i, ppf in enumerate(pf) if all(ppf > -1.e-12)]
    return stable


def ground_state_energy(ps):
    p_ind = ss._ps_to_p_ind.dot(ps)
    res = linprog(c=(ss.cluster_energies_flat
                     + ss._delta_cluster_energies_flat),
                  A_eq=ss.A_ind_flat.T,
                  b_eq=p_ind,
                  bounds=[(0., 1.) for e in ss.cluster_energies_flat],
                  method='revised simplex')
    # res.fun for approx value
    return ss.cluster_energies_flat.dot(res.x)


ss.groups = {}

def ee(ps):
    ist = stable_clusters(ps)
    cos = ss.cluster_occupancies[ist]
    strist = ','.join([str(i) for i in ist])
    try:
        interactions = ss.groups[strist]
    except KeyError:
        print(strist)
        inds = range(len(ist))

        interactions = []
        for n_simplex in range(1, len(ist)+1):
            idx_group = list(combinations(inds, n_simplex))
            pss = np.array([np.mean(cos[list(i)], axis=0) for i in idx_group])
            energies = np.array([ground_state_energy(psi) for psi in pss])

            #print(ps)
            #print((energies/R).round(8), n_simplex)
            for j, ci in enumerate(idx_group):
                pinds = np.array([1./len(ci) if i in ci else 0.
                                  for i in range(len(ist))])

                for (ig, simp) in interactions:
                    for i, idx in enumerate(ig):
                        energies[j] -= np.prod([pinds[id] for id in idx])*simp[i]
            #print((energies/R).round(8), n_simplex)
            energies *= n_simplex**n_simplex
            interactions.append([idx_group, energies])
        ss.groups[strist] = interactions

    pinds = np.einsum('ij, j', np.linalg.pinv(cos).T, ps)
    #print(pinds)
    energy = 0.
    for (ig, simp) in interactions:
        for i, idx in enumerate(ig):
            energy += np.prod([pinds[id] for id in idx])*simp[i]
    return energy

#print('hi', ee([0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.25, 0.75])/R)
#exit()
"""
es = ss.cluster_energies_flat[ist]

# compute pair energies
fik = np.einsum('ij, k', ss.cluster_occupancies[ist], np.ones_like(ist))
fik = (np.einsum('ijk->ikj', fik) + np.einsum('kji->ikj', fik))/2.




W = np.empty_like(fik[:, :, 0])
for i, w in np.ndenumerate(W):
    W[i] = (ground_state_energy(fik[i]) - (es[i[0]] + es[i[1]])/2.)
"""

#def ground_state_energy(ps):
#    ist = stable_clusters(ps)
#    pindst = np.linalg.pinv(ss.cluster_occupancies[ist]).T.dot(ps)
#    return np.einsum('ij, i, j', Wfull[np.ix_(ist, ist)], pindst, pindst)

#ps = np.array([0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9])
#print(ground_state_energy(ps))

fs = np.linspace(0.0, 1.0, 101)
Es = np.empty_like(fs)
Es2 = np.empty_like(fs)
Es3 = np.empty_like(fs)
Es4 = np.empty_like(fs)
for i, f in enumerate(fs):
    ps = np.array([f, 1.-f,
                   f, 1.-f,
                   f, 1.-f,
                   f, 1.-f])
    ps2 = np.array([1.-f, f,
                    1.-f, f,
                    f, 1.-f,
                    f, 1.-f])
    pcl = ss._ideal_cluster_proportions(ps)
    #Ecl = ordered_cluster_energies(ps)
    Es[i] = ee(ps)
    Es3[i] = np.einsum('i, i', ss.cluster_energies_flat, pcl)

    pcl = ss._ideal_cluster_proportions(ps2)
    #Ecl = ordered_cluster_energies(ps2)
    Es2[i] = ee(ps2)
    Es4[i] = np.einsum('i, i', ss.cluster_energies_flat, pcl)

plt.plot(fs, Es/R, label='AAAA-BBBB')
plt.plot(fs, Es2/R, label='BBAA-AABB')
plt.plot(fs, Es3/R)
plt.plot(fs, Es4/R)
plt.legend()
plt.show()

exit()
vord = np.einsum('ij, il, kl->kij', ordering_vectors, ordering_vectors, midclps)

#po = np.einsum('ij, kj->ik', ordering_vectors, ss.cluster_occupancies)
#pomid = np.einsum('ij, kj->ik', ordering_vectors, midclps)
occs = np.einsum('ij, kj->ijk', ss.cluster_occupancies, midclps)

ideal_midcl_prps = np.prod([np.sum(occs[:, i:i+n_species], axis=1)
                            for i, n_species in ss.site_index_tuples],
                           axis=0).T

#pp = np.einsum('ij, j', ordering_vectors, ps)
#pf = pp*po.T
#stable_clusters = [all(ppf > -1.e-10) for ppf in pf]
#pfmid = pp*pomid.T


def ordered_cluster_energies(ps):
    np.random.seed(seed=20)
    delta_ps = np.random.rand(len(ps))*1.e-10

    pfmid = np.einsum('kij, j->ki', vord, ps+delta_ps)
    stable_mids = [all(ppf > -1.e-12) for ppf in pfmid]
    E_ord = lstsq(ideal_midcl_prps[stable_mids],
                  midclEs[stable_mids],
                  rcond=None)[0]
    return E_ord


fs = np.linspace(0.01, 0.99, 101)
Es = np.empty_like(fs)
Es2 = np.empty_like(fs)
Es3 = np.empty_like(fs)
Es4 = np.empty_like(fs)
for i, f in enumerate(fs):
    ps = np.array([f, 1.-f,
                   f, 1.-f,
                   f, 1.-f,
                   f, 1.-f])
    ps2 = np.array([1.-f, f,
                    1.-f, f,
                    f, 1.-f,
                    f, 1.-f])
    pcl = ss._ideal_cluster_proportions(ps)
    Ecl = ordered_cluster_energies(ps)
    Es[i] = np.einsum('i, i', Ecl, pcl)
    Es3[i] = np.einsum('i, i', ss.cluster_energies_flat, pcl)

    pcl = ss._ideal_cluster_proportions(ps2)
    Ecl = ordered_cluster_energies(ps2)
    Es2[i] = np.einsum('i, i', Ecl, pcl)
    Es4[i] = np.einsum('i, i', ss.cluster_energies_flat, pcl)

plt.plot(fs, Es)
plt.plot(fs, Es2)
plt.plot(fs, Es3)
plt.plot(fs, Es4)
plt.show()



exit()



print(ss.cluster_occupancies)
inds = np.array([ss.cluster_occupancies[i] for i in [0, 1, 3, 7, 15]])
E_inds = np.array([ss.cluster_energies_flat[i] for i in [0, 1, 3, 7, 15]])
print(inds)

A_ind_flat = np.linalg.lstsq(inds.T,
                             ss.cluster_occupancies.T,
                             rcond=None)[0].round(decimals=10).T
print(A_ind_flat)

Eord = ((np.einsum('ij, j', A_ind_flat, E_inds))/(4.*R))


print(Eord - ss.cluster_energies_flat/(4.*R))
print(Eord)
print(np.sum(Eord))
f = 0.
Eord[0] -= 0.
Eord[1] -= 0.
Eord[2] -= 0.
Eord[3] -= 0.
Eord[4] -= 2.
Eord[5] -= 0.
Eord[6] -= 2.
Eord[7] -= 0.


Eord[8] -= 4. + 4./3.  # this one?
Eord[9] -= 2.
Eord[10] -= 4.  # NOT this one.

Eord[11] -= 0.

Eord[12] -= 8.
Eord[13] -= 4. - 2./3.  # this one?
Eord[14] -= 6. - 2./3.  # this one? [8, 13, 14] must sum to 14

Eord[15] -= 0.
print(Eord)

#exit()
#Eord += 4./3.*ss.cluster_energies_flat/(4.*R)

Edis = ss.cluster_energies/(4.*R)
Eordering = Edis - Eord.reshape(2, 2, 2, 2)

print(Eordering.flatten())

Eord = 2.*Edis - Eord.reshape(2, 2, 2, 2)
print(Eord.flatten())

fs = np.linspace(0., 1., 101)
psA = np.array([1. + 0.*fs, 0. + 0.*fs])
psB = np.array([0. + 0.*fs, 1. + 0.*fs])
ps = np.array([fs, 1.-fs])
ps2 = np.array([1.-fs, fs])

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]



E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eordering, psA, psA, ps, ps2)
ax[0].plot(fs, E0, label='ordering (AABA-AAAB)')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eord, psA, psA, ps, ps2)
ax[0].plot(fs, E0, label='ordered')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Edis, psA, psA, ps, ps2)
ax[0].plot(fs, E0, label='disordered')

ax[0].legend()

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eordering, psA, ps, ps, psB)
ax[3].plot(fs, E0, label='ordering (ABBA-AAAB)')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eord, psA, ps, ps, psB)
ax[3].plot(fs, E0, label='ordered')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Edis, psA, ps, ps, psB)
ax[3].plot(fs, E0, label='disordered')

ax[3].legend()


E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eordering, psA, ps, ps, ps2)
ax[1].plot(fs, E0, label='ordering (ABBA-AAAB)')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eord, psA, ps, ps, ps2)
ax[1].plot(fs, E0, label='ordered')

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Edis, psA, ps, ps, ps2)
ax[1].plot(fs, E0, label='disordered')

ax[1].legend()
#ax[1].set_ylim(-2., 0.)
#ax[1].set_xlim(0.4, 0.6)


#fs = np.linspace(0., 0.5, 101)
ps = np.array([fs, 1.-fs])
ps2 = np.array([1.-fs, fs])

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eordering, ps, ps, ps2, ps2)
ax[2].plot(fs, E0)

E0 = np.einsum('ijkl, io, jo, ko, lo->o', Eord, ps, ps, ps2, ps2)
ax[2].plot(fs, E0)


E0 = np.einsum('ijkl, io, jo, ko, lo->o', Edis, ps, ps, ps2, ps2)
ax[2].plot(fs, E0)

plt.show()
exit()

"""
from numpy.linalg import pinv, lstsq
from itertools import product, permutations

M = np.zeros((2, 2, 2, 2, 2, 2, 2, 2))
N = np.zeros((2, 2, 2, 2, 2, 2, 2, 2))
for idx, E in np.ndenumerate(cluster_energies):
    M[idx[0], idx[1], idx[2], idx[3], :, :, :, :] += E/2.
    M[:, :, :, :, idx[0], idx[1], idx[2], idx[3]] += E/2.

    N[idx[0], idx[1], idx[2], idx[3], :, :, :, :] += E/2.
    N[:, :, :, :, idx[0], idx[1], idx[2], idx[3]] += E/2.

for idx, n in np.ndenumerate(N):
    if np.sum(np.abs(np.array(idx[:4]) - np.array(idx[4:]))) == 2:
        # Here we look for the ordering energy
        # for all clusters with the same composition.
        l = [list(set(p)) for p in np.array([idx[:4], idx[4:]]).T]
        es = np.squeeze(cluster_energies[np.ix_(*l)])
        N[idx] += ((es[0, 1] + es[1, 0]) - (es[0, 0] + es[1, 1]))

    if np.sum(np.abs(np.array(idx[:4]) - np.array(idx[4:]))) == 3:
        l = [list(set(p)) for p in np.array([idx[:4], idx[4:]]).T]
        es = np.squeeze(cluster_energies[np.ix_(*l)])
        print(es)

    if np.sum(np.abs(np.array(idx[:4]) - np.array(idx[4:]))) == 4:
        l = [list(set(p)) for p in np.array([idx[:4], idx[4:]]).T]
        es = np.squeeze(cluster_energies[np.ix_(*l)])
        N[idx] += np.min(es) - np.max(es)


pi, pj, pk, pl = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
pi, pj, pk, pl = [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]
pi, pj, pk, pl = [[0.25, 0.75], [0.75, 0.25], [0.25, 0.75], [0.75, 0.25]]

#N[0, 1, 1, 1, 0, 1, 0, 0]

fs = np.linspace(0.5, 1., 101)
Ns = np.empty_like(fs)
Ms = np.empty_like(fs)

for i, f in enumerate(fs):
    pi, pj, pk, pl = [[1.-f, f], [1.-f, f], [1.-f, f], [1.-f, f]]
    #pi, pj, pk, pl = [[f, 1.-f], [1.-f, f], [f, 1.-f], [1.-f, f]]
    #pi, pj, pk, pl = [[0., 1.], [1., 0.], [f, 1.-f], [1.-f, f]]
    Ns[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', N,
                      pi, pj, pk, pl,
                      pi, pj, pk, pl)
    Ms[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', M,
                    pi, pj, pk, pl,
                    pi, pj, pk, pl)
plt.plot(fs, Ns/R, label='ordered (p(AAAA))')
plt.plot(fs, Ms/R, label='disordered (p(AAAA))')
for i, f in enumerate(fs):
    #pi, pj, pk, pl = [[1.-f, f], [1.-f, f], [1.-f, f], [1.-f, f]]
    pi, pj, pk, pl = [[f, 1.-f], [1.-f, f], [f, 1.-f], [1.-f, f]]
    #pi, pj, pk, pl = [[0., 1.], [1., 0.], [f, 1.-f], [1.-f, f]]
    Ns[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', N,
                      pi, pj, pk, pl,
                      pi, pj, pk, pl)
    Ms[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', M,
                    pi, pj, pk, pl,
                    pi, pj, pk, pl)
plt.plot(fs, Ns/R, label='ordered (p(ABAB))')
plt.plot(fs, Ms/R, label='disordered (p(ABAB))')
plt.ylim(0.5, 1)
plt.ylim(-16, 0)
plt.legend()

plt.show()

exit()
"""



pi, pj, pk, pl = [[1., 0.], [0., 1.], [0.5, 0.5], [0.5, 0.5]]
print(np.einsum('ijkl, i, j, k, l',
                ss.cluster_energies,
                pi, pj, pk, pl)/R)


ss.set_composition_from_p_s(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

halfE = np.einsum('ijkl, mnop', ss.cluster_ones, ss.cluster_energies)
halfE = (np.einsum('ijklmnop', halfE) + np.einsum('mnopijkl', halfE))/2.

c = []
for n in ss.species_per_site:
    id = np.identity(n)
    ones = np.ones(n)
    c.append((np.einsum('iq, m', id, ones) + np.einsum('mq, i', id, ones))/2.)

A = np.einsum('imq, jnr, kos, lpt, imu, jnv, kow, lpx->ijklmnopqrstuvwx',
              c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3])


from numpy.linalg import pinv, lstsq
from itertools import product, permutations

perms = [permutations(s) for s in ['im', 'jn', 'ko', 'lp']]
perms = [[''.join(p) for p in t] for t in perms]
prod = list(product(*perms))
halfEmin = halfE.copy()
halfEmax = halfE.copy()
for p in prod:
    s = ''.join([s[0] for s in p])+''.join([s[1] for s in p])
    halfEmin = np.minimum(halfEmin, np.einsum('ijklmnop->'+s, halfE))
    halfEmax = np.maximum(halfEmax, np.einsum('ijklmnop->'+s, halfE))

halfEorder = halfEmin - halfEmax

print(halfEorder/R)
#exit()

p = np.array([0.5, 0.5])
print(halfEorder/R)
print(np.einsum('ijklmnop, i, j, k, l, m, n, o, p', halfEmin/R,
                p, p, p, p, p, p, p, p))

pa = [0., 1.]
pb = [0., 1.]
pc = [0., 1.]
pd = [0.75, 0.25]
pd = [0.5, 0.5]
print(np.einsum('ijklmnop, i, j, k, l, m, n, o, p', halfEmin/R,
                pa, pb, pc, pd, pa, pb, pc, pd))


W3 = np.zeros_like(halfEorder)
for idx, w in np.ndenumerate(halfEorder):
    f = np.sum(np.abs(np.array(idx[:4]) - np.array(idx[4:])))
    if f == 1:
        W3[idx] = -2.
    if f == 2:
        W3[idx] = -4.
    if f == 3:
        W3[idx] = -6.
    if f == 4:
        W3[idx] = -8.

pi, pj, pk, pl = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
pi, pj, pk, pl = [[0.75, 0.25], [0.25, 0.75], [0.75, 0.25], [0.25, 0.75]]
pi, pj, pk, pl = [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]
#pi, pj, pk, pl = [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]
#pi, pj, pk, pl = [[0.5, 0.5], [0.5, 0.5], [0., 1.], [0., 1.]]

fs = np.linspace(0., 1., 101)
es = np.empty_like(fs)
for i, f in enumerate(fs):
    #pi, pj, pk, pl = [[f, 1.-f], [1.-f, f], [f, 1.-f], [1.-f, f]]
    pi, pj, pk, pl = [[f, 1.-f], [f, 1.-f], [f, 1.-f], [f, 1.-f]]
    #pi, pj, pk, pl = [[f, 1.-f], [f, 1.-f], [1., 0.], [1., 0.]]
    es[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', W3,
                      pi, pj, pk, pl,
                      pi, pj, pk, pl) + np.einsum('ijkl, i, j, k, l',
                                                  ss.cluster_energies,
                                                  pi, pj, pk, pl)/R
plt.plot(fs, es)

for i, f in enumerate(fs):
    pi, pj, pk, pl = [[f, 1.-f], [1.-f, f], [f, 1.-f], [1.-f, f]]
    #pi, pj, pk, pl = [[f, 1.-f], [f, 1.-f], [f, 1.-f], [f, 1.-f]]
    #pi, pj, pk, pl = [[1.-f, f], [f, 1.-f], [1., 0.], [1., 0.]]
    es[i] = np.einsum('ijklmnop, i, j, k, l, m, n, o, p', W3,
                      pi, pj, pk, pl,
                      pi, pj, pk, pl) + np.einsum('ijkl, i, j, k, l',
                                                  ss.cluster_energies,
                                                  pi, pj, pk, pl)/R
plt.plot(fs, es)
plt.show()
exit()

halfEorder -= np.einsum('qrstuvwx, ijklmnopqrstuvwx->ijklmnop', W3, A)

for idx, w in np.ndenumerate(halfEorder):
    f = np.sum(np.abs(np.array(idx[:4]) - np.array(idx[4:])))
    if f == 2:
        print(w)

#print(W3)



exit()



c = []
for n in ss.species_per_site:
    id = np.identity(n)
    ones = np.ones(n)
    c.append((np.einsum('iq, m', id, ones) + np.einsum('mq, i', id, ones))/2.)

A = np.einsum('imq, jnr, kos, lpt, imu, jnv, kow, lpx->ijklmnopqrstuvwx',
              c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3])

B = lstsq(A.reshape(256, 256), Wmin.flatten(), rcond=None)
B = B[0].reshape((2, 2, 2, 2, 2, 2, 2, 2))

D = np.zeros((8, 8))
D[0, 3] = -1.
D[0, 5] = -1.
D[0, 7] = -1.
D[1, 2] = -1.
D[1, 4] = -1.
D[1, 6] = -1.
D[2, 5] = -1.
D[2, 7] = -1.
D[3, 4] = -1.
D[3, 6] = -1.
D[4, 7] = -1.
D[5, 6] = -1.

D[0,1] = -1.
D[2,3] = -1.
D[4,5] = -1.
D[6,7] = -1.

D *= 4.
pijkl = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#pijkl = np.array([0.75, 0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75])
#pijkl = np.array([0., 1., 0., 1., 1., 0., 1., 0.])

pijkl = np.array([0.5, 0.5, 0.5, 0.5, 1., 0., 1., 0.])
#pijkl = np.array([0.5, 0.5, 0.5, 0.5, 0., 1., 1., 0.])
#pijkl = np.array([0., 1., 1., 0., 0., 1., 1., 0.])
#pijkl = np.array([1., 0., 1., 0., 1., 0., 0., 1.])
#pijkl = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
#pijkl = np.array([0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25])
print(np.einsum('ij, i, j', D, pijkl, pijkl))
exit()
pi, pj, pk, pl = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
pi, pj, pk, pl = [[0.75, 0.25], [0.25, 0.75], [0.75, 0.25], [0.25, 0.75]]
pi, pj, pk, pl = [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]
print(np.einsum('ijklmnop, i, j, k, l, m, n, o, p', B,
                   pi, pj, pk, pl,
                   pi, pj, pk, pl)/R)
