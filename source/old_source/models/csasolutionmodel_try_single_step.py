import numpy as np
from numpy.linalg import pinv, lstsq, solve
from scipy.optimize import root, linprog, minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
import itertools
from sympy import Matrix

R = 8.31446

import warnings

warnings.filterwarnings('error')


def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps:
    log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    hi_limit = 1.e10
    f_eps = 1. - x/eps
    mask = np.logical_and(eps < x, x < hi_limit)
    f_eps = np.where(np.abs(x) < 1.e10, 1. - x/eps, 1. - hi_limit/eps)
    ln = np.where(x <= eps, np.log(eps) - f_eps - f_eps*f_eps/2.,
                  np.log(hi_limit))
    ln[mask] = np.log(x[mask])
    return ln


def inverseish(x, eps=1.e-5):
    """
    1st order series expansion of 1/x about eps: 2/eps - x/eps/eps
    Prevents infinities at x=0
    """
    hi_limit = 1.e10
    mask = np.logical_and(eps < x, x < hi_limit)
    oneoverx = np.where(x <= eps, 2./eps - x/eps/eps, hi_limit)
    oneoverx[mask] = 1./x[mask]
    return oneoverx


class CSAModel(object):
    """
    This is the base class for all Cluster/Site Approximation models
    """

    def __init__(self, cluster_energies, gamma, site_species,
                 compositional_interactions=np.array([[0., 0.],
                                                      [0., 0.]])):

        self.n_sites = len(cluster_energies.shape)
        self.species_per_site = np.array(cluster_energies.shape)
        self.site_start_indices = (np.cumsum(self.species_per_site)
                                   - self.species_per_site[0])
        self.site_index_tuples = np.array([self.site_start_indices,
                                           self.species_per_site]).T
        self.n_site_species = np.sum(self.species_per_site)

        if not len(site_species) == len(self.species_per_site):
            raise Exception('site_species must be a list of lists, '
                            'each second level list containing the number '
                            'of species on each site.')

        if not all([len(s) == self.species_per_site[i]
                    for i, s in enumerate(site_species)]):
            raise Exception('site_species must have the correct '
                            'number of species on each site')
        self.site_species = site_species

        self.site_species_flat = [species for site in site_species
                                  for species in site]
        self.n_clusters = len(self.site_species_flat)

        self.components = sorted(list(set(self.site_species_flat)))
        self.n_components = len(self.components)

        # Make correlation matrix between composition and site species
        self.site_species_compositions = np.zeros((len(self.site_species_flat),
                                                   len(self.components)),
                                                  dtype='int')
        for i, ss in enumerate(self.site_species_flat):
            self.site_species_compositions[i, self.components.index(ss)] = 1

        clusters = itertools.product(*[np.identity(n_species, dtype='int')
                                       for n_species
                                       in cluster_energies.shape])

        self.cluster_energies = cluster_energies
        self.cluster_energies_flat = cluster_energies.flatten()

        self.cluster_occupancies = np.array([np.hstack(cl) for cl in clusters])

        self.cluster_compositions = np.einsum('ij, jk->ik',
                                              self.cluster_occupancies,
                                              self.site_species_compositions)

        self.pivots = list(sorted(set([list(c).index(1)
                                       for c in self.cluster_occupancies.T])))
        self.n_ind = len(self.pivots)

        ind_cl_occs = np.array([self.cluster_occupancies[p]
                                for p in self.pivots], dtype='int')
        ind_cl_comps = np.array([self.cluster_compositions[p]
                                 for p in self.pivots],  dtype='int')
        self.independent_cluster_occupancies = ind_cl_occs
        self.independent_cluster_compositions = ind_cl_comps

        self.independent_interactions = np.einsum('ij, lk, jk->il',
                                                  ind_cl_comps, ind_cl_comps,
                                                  compositional_interactions)

        null = Matrix(self.independent_cluster_compositions.T).nullspace()
        rxn_matrix = np.array([np.array(v).T[0] for v in null])
        self.isochemical_reactions = rxn_matrix
        self.n_reactions = len(rxn_matrix)

        self._ps_to_p_ind = pinv(self.independent_cluster_occupancies.T)

        self.A_ind = lstsq(self.independent_cluster_occupancies.T,
                           self.cluster_occupancies.T,
                           rcond=None)[0].round(decimals=10).T

        self._AA = np.einsum('ik, ij -> ijk', self.A_ind, self.A_ind)
        self._AAA = np.einsum('il, ik, ij -> ijkl',
                              self.A_ind, self.A_ind, self.A_ind)

        self.gamma = gamma

        np.random.seed(seed=19)
        std = np.std(self.cluster_energies_flat)
        delta = np.random.rand(len(self.cluster_energies_flat))*std*1.e-10
        self._delta_cluster_energies = delta

    def set_state(self, temperature):
        self.temperature = temperature
        self.equilibrated_clusters = False

    def set_composition_from_p_ind(self, p_ind):
        self.p_ind = p_ind.astype('float64')
        self.p_s = self.independent_cluster_occupancies.T.dot(self.p_ind)
        self.ln_p_s = logish(self.p_s)
        self._ideal_c = logish(self._independent_ideal_cluster_proportions())
        self.equilibrated_clusters = False

    def set_composition_from_p_s(self, p_s):
        p_ind = self._ps_to_p_ind.dot(p_s)
        self.set_composition_from_p_ind(p_ind)

    def compositional_array(self, composition):
        # first, make the numpy array
        arr_c = np.zeros(len(self.components))
        for k, v in composition.items():
            arr_c[self.components.index(k)] = v
        return arr_c

    def ground_state_cluster_proportions_from_composition(self, c_arr,
                                                          normalize=True):

        # Apply a small random tweak to all energies to force the
        # solver to converge to the state with the minimum number of clusters
        # in the (common) circumstance where clusters are degenerate

        # Solve the linear programming problem
        # Revised simplex is slower than interior-point,
        # but is required to find an accurate solution
        # (especially with the small random tweaks applied)
        res = linprog(c=(self.cluster_energies_flat
                         + self._delta_cluster_energies),
                      A_eq=self.cluster_compositions.T,
                      b_eq=c_arr,
                      bounds=[(0., 1.) for l in self.cluster_energies_flat],
                      method='revised simplex')

        if not res.success:
            raise Exception('Minimum ground state energy not found.')

        # Normalise the proportions if required
        proportions = res.x
        if normalize:
            proportions /= np.sum(res.x)
        return proportions

    def maximum_entropy_cluster_proportions_from_composition(self, c_arr,
                                                             normalize=True):
        def minus_entropy(p_ind):
            p_s = self.independent_cluster_occupancies.T.dot(p_ind)
            norm_sum_p_s = self.n_sites*np.sum(p_s)
            S_s = np.sum(p_s*np.log(p_s/np.sum(norm_sum_p_s)))
            return S_s

        cons = (LinearConstraint(self.independent_cluster_occupancies.T,
                                 0., 1.),
                LinearConstraint(self.independent_cluster_compositions.T,
                                 c_arr, c_arr))

        res = minimize(minus_entropy, (0., 0.25, 0.25, 0.25, 0.25),
                       method='SLSQP', constraints=cons)

        # Normalise the proportions if required
        p_s = self.independent_cluster_occupancies.T.dot(res.x)
        proportions = self._ideal_cluster_proportions(p_s)

        if normalize:
            proportions /= np.sum(res.x)
        return proportions

    def equilibrate(self, composition, temperature):

        occs = self.independent_cluster_occupancies.T
        c_arr = self.compositional_array(composition)
        self.set_state(temperature)

        def energy(expc):
            c = logish(expc)
            p_cl = self._cluster_proportions(c, temperature)
            p_ind = self.A_ind.T.dot(p_cl)
            self.set_composition_from_p_ind(p_ind)
            self.c = c
            self.equilibrated_clusters = True
            self.set_cluster_proportions()

            dp_ind_dc = np.einsum('ijk, i -> kj', self._AA, p_cl)
            grad = self.molar_chemical_potentials.dot(dp_ind_dc)/expc  # certainly not the most efficient way to do this
            return self.molar_gibbs, grad

        def occupancy_constraints(expc):
            c = logish(expc)
            p_cl = self._cluster_proportions(c, temperature)
            p_ind = self.A_ind.T.dot(p_cl)
            return occs.dot(p_ind)

        def composition_constraints(expc):
            c = logish(expc)
            p_cl = self._cluster_proportions(c, temperature)
            p_ind = self.A_ind.T.dot(p_cl)
            return self.independent_cluster_compositions.T.dot(p_ind)

        # Find ordered and disordered states to use as starting points
        p_cl_grd = self.ground_state_cluster_proportions_from_composition(c_arr)
        p_cl_max = self.maximum_entropy_cluster_proportions_from_composition(c_arr)

        ideal_c = logish(p_cl_max[self.pivots])
        p_cl_near_ord = 0.05 * p_cl_max + 0.95 * p_cl_grd
        near_ord_c = (logish(p_cl_near_ord[self.pivots])
                      + (self.cluster_energies_flat[self.pivots]
                         / (self.temperature * R)))

        # bulk compositional constraints AND occupancies between 0 and 1
        # keep_feasible=True requires some future version of scipy
        cons = (NonlinearConstraint(occupancy_constraints,
                                    np.zeros(self.n_site_species),
                                    np.ones(self.n_site_species)),
                NonlinearConstraint(composition_constraints, c_arr, c_arr))

        # minimize using near-ground state as a starting point
        res = minimize(energy, np.exp(ideal_c),
                       method='SLSQP', constraints=cons,
                       args=(), jac=True)

        # store c and E
        c = self.c
        p_ind = self.p_ind
        E = res.fun

        # minimize using near-ground state as a starting point
        res = minimize(energy, np.exp(near_ord_c),
                       method='SLSQP', constraints=cons,
                       args=(), jac=True)

        if res.fun > E:
            self.set_composition_from_p_ind(p_ind)
            self.c = c
            self.equilibrated_clusters = True
            self.set_cluster_proportions()

    def _independent_ideal_cluster_proportions(self):
        occs = np.einsum('ij, j->ij', self.independent_cluster_occupancies,
                         self.p_s)
        return np.prod([np.sum(occs[:, i:i+n_species], axis=1)
                        for i, n_species in self.site_index_tuples],
                       axis=0)

    def _ideal_cluster_proportions(self, p_s):
        occs = np.einsum('ij, j->ij', self.cluster_occupancies, p_s)
        return np.prod([np.sum(occs[:, i:i+n_species], axis=1)
                        for i, n_species in self.site_index_tuples],
                       axis=0)

    def _ground_state_cluster_proportions(self, p_ind):
        # Solve the linear programming problem
        # Revised simplex is slower than interior-point,
        # but is required to find an accurate solution.
        # The small pseudorandom (using a fixed seed)
        # tweaks are applied so that a unique
        # set of cluster proportions is obtained.
        # (as the problem is degenerate).
        res = linprog(c=(self.cluster_energies_flat
                         + self._delta_cluster_energies),
                      A_eq=self.A_ind.T,
                      b_eq=p_ind,
                      bounds=[(0., 1.) for l in self.cluster_energies_flat],
                      method='revised simplex')
        # x0=self.ideal_cluster_proportions)

        if not res.success:
            print(self.ideal_cluster_proportions)
            print(self.A_ind.T.dot(self.ideal_cluster_proportions) - p_ind)
            print('p_s:', self.independent_cluster_occupancies.T.dot(p_ind))
            print(res)
            raise Exception('Minimum ground state energy not found (2).')
            exit()

        return res.x

    @property
    def ideal_cluster_proportions(self):
        return self._ideal_cluster_proportions(self.p_s)

    @property
    def ground_state_cluster_proportions(self):
        return self._ground_state_cluster_proportions(self.p_ind)

    def _cluster_proportions(self, c, temperature):
        lnval = (self.A_ind.dot(c)
                 - self.cluster_energies_flat / (self.gamma * R * temperature))
        return np.exp(np.where(lnval > 100, 100., lnval))

    def set_cluster_proportions(self):
        cf = self._cluster_proportions(self.c, self.temperature)
        self.cluster_proportions_flat = cf
        self.ln_cluster_proportions_flat = logish(cf)

        self.cluster_proportions = cf.reshape(self.species_per_site)
        self.ln_cluster_proportions = logish(self.cluster_proportions)

    def delta_proportions(self, c, temperature):
        p_cl = self._cluster_proportions(c, temperature)
        deltas = self.p_ind - self.A_ind.T.dot(p_cl)
        jac = -np.einsum('ijk, i -> kj', self._AA, p_cl)
        return deltas, jac

    def equilibrate_clusters(self):
        if np.max(self._ideal_c) > -1.e-10:  # pure endmember
            self.c = self._ideal_c
        else:  # Try the ideal cluster proportions
            sol = root(self.delta_proportions, self._ideal_c, jac=True,
                       args=(self.temperature),
                       method='lm', options={'ftol': 1.e-16})

            # print('ideal', self._ideal_c)
            if np.max(np.abs(sol.fun)) < 1.e-8:
                self.c = sol.x
            else:  # Try near-fully ordered cluster proportions

                p_cl_ord = self.ground_state_cluster_proportions
                p_cl_disord = self.ideal_cluster_proportions
                p_cl_near_ord = 0.05 * p_cl_disord + 0.95 * p_cl_ord
                near_ord_c = (logish(p_cl_near_ord[self.pivots])
                              + (self.cluster_energies_flat[self.pivots]
                                 / (self.temperature * R)))

                sol = root(self.delta_proportions, near_ord_c, jac=True,
                           args=(self.temperature),
                           method='lm', options={'ftol': 1.e-16})

                if np.max(np.abs(sol.fun)) < 1.e-8:
                    self.c = sol.x
                else:
                    # try to anneal
                    T = 10000.
                    n_steps = 4
                    good = True
                    while T > self.temperature + 0.1 or not good:
                        T_steps = np.logspace(np.log10(T),
                                              np.log10(self.temperature),
                                              n_steps)

                        guess_c = self._ideal_c
                        for i, T in enumerate(T_steps):
                            sol = root(self.delta_proportions, guess_c,
                                       args=(T), jac=True,
                                       method='lm', tol=1.e-10,
                                       options={'ftol': 1.e-16})

                            if np.max(np.abs(sol.fun)) < 1.e-10:
                                guess_c = sol.x
                                good = True

                            else:
                                n_steps *= 2

                                if n_steps > 100:
                                    print(sol)
                                    print(self._cluster_proportions(sol.x, T_steps[-1]))
                                    raise Exception('Clusters could not be '
                                                    'equilibrated, '
                                                    'even with annealing')
                                T = T_steps[i-1]
                                good = False
                                break

                    self.c = sol.x

        self.equilibrated_clusters = True
        self.set_cluster_proportions()

    def _inv_J(self):
        return np.einsum('ijk, i -> kj', self._AA,
                         self.cluster_proportions_flat)

    def _dpcl_dp_ind(self):
        dp_ind_dc = np.einsum('ijk, i -> kj', self._AA,
                              self.cluster_proportions_flat)
        dpcl_dc = np.einsum('ij, i -> ij', self.A_ind,
                            self.cluster_proportions_flat)

        # the returned solve is equivalent to
        # np.einsum('lj, jk -> lk', dpcl_dc, pinv(dp_ind_dc)))
        try:
            return solve(dp_ind_dc.T, dpcl_dc.T).T
        except np.linalg.LinAlgError:  # singular matrix
            return np.einsum('lj, jk -> lk', dpcl_dc, pinv(dp_ind_dc))

    def _d2pcl_dp_ind_dp_ind(self):

        pc = self.cluster_proportions_flat
        dpcl_dc = np.einsum('ij, i -> ij', self.A_ind, pc)
        dp_ind_dc = np.einsum('ijk, i -> jk', self._AA, pc)
        dpcl_dcdc = np.einsum('ijn, i -> ijn', self._AA, pc)
        dMdc = np.einsum('il, ij, ik, i -> jkl',
                         self.A_ind, self.A_ind, self.A_ind, pc)

        Minv = pinv(dp_ind_dc)

        D = np.einsum('lj, jk -> lk', dpcl_dc, Minv)
        E = dpcl_dcdc - np.einsum('il, mln-> imn', D, dMdc)
        F = np.einsum('imn, mk, np -> ikp', E, Minv, Minv)

        return F

    def non_ideal_hessian(self):
        q = np.eye(self.n_ind) - np.einsum('i, j->ij', np.ones(self.n_ind),
                                           self.p_ind)
        hess = np.einsum('ij, jk, mk->im', q, self.independent_interactions, q)
        hess += hess.T
        return hess

    def non_ideal_potentials(self):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003
        q = np.eye(self.n_ind) - np.einsum('i, j->ij', np.ones(self.n_ind),
                                           self.p_ind)
        # The following are equivalent to
        # np.einsum('ij, jk, ik->i', -q, self.Wx, q)
        Wint = -(q.dot(self.independent_interactions)*q).sum(-1)
        return Wint

    def non_ideal_energy(self):
        return np.einsum('i, j, ij', self.p_ind, self.p_ind, self.independent_interactions)

    def check_equilibrium(self):
        if self.equilibrated_clusters:
            pass
        else:
            raise Exception('This property of the solution requires '
                            'you to first call equilibrate_clusters()')

    @property
    def hessian_entropy(self):
        D = self._dpcl_dp_ind()
        F = self._d2pcl_dp_ind_dp_ind()

        total_site_atoms = np.sum(self.p_s)
        # total_clusters = np.sum(self.cluster_proportions_flat)

        d2S_dpcl2 = np.diag(inverseish(self.cluster_proportions_flat)) - 1.
        hess_S_n = -R*(np.einsum('im, ij, mk-> jk', d2S_dpcl2, D, D)
                       + np.einsum('i, ijk->jk',
                                   self.ln_cluster_proportions_flat, F))

        hess_S_i = -R*(np.einsum('ki, ji, i->jk',
                                 self.independent_cluster_occupancies,
                                 self.independent_cluster_occupancies,
                                 inverseish(self.p_s)) - total_site_atoms)

        hess_S_t = (self.gamma * hess_S_n
                    + (1./self.n_sites - self.gamma) * hess_S_i)

        return hess_S_t

    @property
    def partial_molar_entropies(self):
        self.check_equilibrium()
        D = self._dpcl_dp_ind()

        g_S_n = -R*(np.einsum('lk, l', D, self.ln_cluster_proportions_flat))
        g_S_i = -R*(np.einsum('ki, i',
                              self.independent_cluster_occupancies,
                              self.ln_p_s))
        g_S_t = self.gamma * g_S_n + (1./self.n_sites - self.gamma) * g_S_i
        return g_S_t

    @property
    def molar_entropy(self):
        """
        The entropy per mole of sites
        """
        self.check_equilibrium()

        # Cluster and site entropy contributions per cluster
        S_n = -R*np.sum(self.cluster_proportions * self.ln_cluster_proportions)
        S_i = -R*np.sum(self.p_s * self.ln_p_s)

        # remember gamma is the number of noninterfering clusters PER SITE
        S_t = self.gamma * S_n + (1./self.n_sites - self.gamma) * S_i

        return S_t

    @property
    def hessian_gibbs(self):
        F = self._d2pcl_dp_ind_dp_ind()

        hess_G = np.einsum('ijk, i -> jk', F, self.cluster_energies_flat)
        hess_S = self.hessian_entropy

        return hess_G - self.temperature * hess_S + self.non_ideal_hessian()

    @property
    def molar_chemical_potentials(self):
        """
        The chemical potentials per mole of sites
        """
        g_S_t = self.partial_molar_entropies

        D = self._dpcl_dp_ind()
        g_E = np.einsum('lk, l', D, self.cluster_energies_flat)
        chemical_potentials = g_E - self.temperature * g_S_t

        return chemical_potentials + self.non_ideal_potentials()

    @property
    def molar_gibbs(self):
        """
        The gibbs free energy per mole of sites
        """
        S_t = self.molar_entropy
        E_t = np.sum(self.cluster_proportions * self.cluster_energies)
        G_t = E_t - self.temperature * S_t

        return G_t + self.non_ideal_energy()
