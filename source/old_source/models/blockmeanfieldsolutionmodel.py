import numpy as np
from numpy.linalg import pinv, lstsq, solve
from scipy.linalg import block_diag
from scipy.optimize import root, linprog, minimize, LinearConstraint
from itertools import product
from sympy import Matrix

R = 8.31446


def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps:
    log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    mask = x > eps
    ln = np.where(x <= eps, np.log(eps) - f_eps - f_eps*f_eps/2., 0.)
    ln[mask] = np.log(x[mask])
    return ln


def inverseish(x, eps=1.e-5):
    """
    1st order series expansion of 1/x about eps: 2/eps - x/eps/eps
    Prevents infinities at x=0
    """
    mask = x > eps
    oneoverx = np.where(x <= eps, 2./eps - x/eps/eps, 0.)
    oneoverx[mask] = 1./x[mask]
    return oneoverx


class BlockMeanFieldModel(object):
    """
    This is the base class for all Block Mean Field models
    """

    def __init__(self, site_species,
                 site_species_energies, bond_energies,
                 cluster_bonds_per_cluster,
                 bridging_bonds_per_cluster,
                 fraction_bridging_bonds_in_basic_cluster,
                 compositional_interactions=None):

        self.gamma = 0.0
        self.site_species = site_species
        self.site_species_energies = site_species_energies
        self.bond_energies = bond_energies
        self.cluster_bonds = cluster_bonds_per_cluster
        self.bridging_bonds = bridging_bonds_per_cluster
        self.f_bridging_bonds_basic = fraction_bridging_bonds_in_basic_cluster

        self.n_species_per_site = np.array([len(s) for s in site_species])
        self.site_start_indices = (np.cumsum(self.n_species_per_site)
                                   - self.n_species_per_site[0])
        self.site_index_tuples = np.array([self.site_start_indices,
                                           self.n_species_per_site]).T

        site_bounds = [0]
        site_bounds.extend(list(np.cumsum(self.n_species_per_site)))
        self.site_bounds = np.array([site_bounds[:-1], site_bounds[1:]]).T

        self.n_sites = len(self.n_species_per_site)

        self.n_site_species = np.sum(self.n_species_per_site)
        self.n_ind = self.n_site_species - self.n_sites + 1

        if not len(site_species) == len(self.n_species_per_site):
            raise Exception('site_species must be a list of lists, '
                            'with the number of outermost lists equal to '
                            'the number of sites.')

        if not all([len(s) == self.n_species_per_site[i]
                    for i, s in enumerate(site_species)]):
            raise Exception('site_species must have the correct '
                            'number of species on each site')
        self.site_species = site_species
        self.site_species_flat = [species for site in site_species
                                  for species in site]
        self.components = sorted(list(set(self.site_species_flat)))
        self.n_components = len(self.components)

        if compositional_interactions is None:
            compositional_interactions = np.zeros((self.n_components,
                                                   self.n_components))

        # Make correlation matrix between composition and site species
        self.site_species_compositions = np.zeros((len(self.site_species_flat),
                                                   len(self.components)),
                                                  dtype='int')
        for i, ss in enumerate(self.site_species_flat):
            self.site_species_compositions[i, self.components.index(ss)] = 1

        clusters = product(*[np.identity(n_species, dtype='int')
                             for n_species in self.n_species_per_site])
        self.n_clusters = np.prod(self.n_species_per_site)

        self.cluster_occupancies = np.array([np.hstack(cl) for cl in clusters])
        self.cluster_compositions = np.einsum('ij, jk->ik',
                                              self.cluster_occupancies,
                                              self.site_species_compositions)
        self.pivots_flat = list(sorted(set([list(c).index(1)
                                            for c
                                            in self.cluster_occupancies.T])))

        ind_cl_occs = np.array([self.cluster_occupancies[p]
                                for p in self.pivots_flat], dtype='int')
        ind_cl_comps = np.array([self.cluster_compositions[p]
                                 for p in self.pivots_flat],  dtype='int')
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

        A_ind_shape = list(self.n_species_per_site)
        A_ind_shape.append(self.n_ind)
        self.A_ind = lstsq(self.independent_cluster_occupancies.T,
                           self.cluster_occupancies.T,
                           rcond=None)[0].round(decimals=10).T

        self._AA = np.einsum('ik, ij -> ijk', self.A_ind, self.A_ind)

        self.pivots = np.argwhere(np.sum(np.abs(self.A_ind), axis=-1) == 1)

        shp = np.concatenate((self.n_species_per_site,
                              self.n_species_per_site))
        self.cluster_identity_matrix = np.eye(self.n_clusters).reshape(shp)
        self.cluster_ones = np.ones(self.n_species_per_site)

        self.cluster_energies = (np.einsum('i, ki',
                                           self.site_species_energies,
                                           self.cluster_occupancies)
                                 + np.einsum('ij, ij, ki, kj->k',
                                             self.bond_energies,
                                             self.cluster_bonds,
                                             self.cluster_occupancies,
                                             self.cluster_occupancies))

        self.bridging_interactions = np.einsum('ij, ij, kj->ki',
                                               self.bond_energies,
                                               self.bridging_bonds,
                                               self.cluster_occupancies)

        np.random.seed(seed=19)
        std = np.std(self.cluster_energies)
        delta = np.random.rand(len(self.cluster_energies))*std*1.e-10
        self._delta_cluster_energies = delta

    def set_state(self, temperature):
        self.temperature = temperature
        self.equilibrated_clusters = False

    def set_composition_from_p_ind(self, p_ind, n_clusters_per_block):
        assert n_clusters_per_block >= 1.

        f_bridging_clusters = (self.f_bridging_bonds_basic
                               * np.power(n_clusters_per_block, -1./3.)) # should be -1./3.

        #f_bridging_clusters = np.power(f_bridging_clusters, 1.)
        self.block_size = n_clusters_per_block
        self.p_ind = p_ind.astype('float64')
        self.p_s = self.independent_cluster_occupancies.T.dot(self.p_ind)
        self.ln_p_s = logish(self.p_s)
        self.block_energies = ((self.cluster_energies
                                * (1. - f_bridging_clusters)
                                + np.einsum('ki, i->k',
                                            self.bridging_interactions,
                                            self.p_s)
                                * f_bridging_clusters)
                               * n_clusters_per_block)

        self._ideal_c = logish(self._independent_ideal_cluster_proportions())
        self.equilibrated_clusters = False

    def set_composition_from_p_s(self, p_s, n_clusters_per_block):
        p_ind = self._ps_to_p_ind.dot(p_s)
        self.set_composition_from_p_ind(p_ind, n_clusters_per_block)

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
        res = linprog(c=(self.cluster_energies
                         + self._delta_cluster_energies),
                      A_eq=self.cluster_compositions.T,
                      b_eq=c_arr,
                      bounds=[(0., 1.) for es in self.cluster_energies],
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

        x0 = [1./(self.n_ind-1.) if i > 0 else 0. for i in range(self.n_ind)]
        res = minimize(minus_entropy, x0, method='SLSQP', constraints=cons)

        # Normalise the proportions if required
        p_s = self.independent_cluster_occupancies.T.dot(res.x)
        proportions = self._ideal_cluster_proportions(p_s)

        if normalize:
            proportions /= np.sum(res.x)
        return proportions

    def equilibrate(self, composition, temperature):

        occs = self.independent_cluster_occupancies.T
        c_arr = self.compositional_array(composition)

        def energy(rxn_amounts, p_ind_start, rxn_matrix):
            p_ind = p_ind_start + rxn_matrix.T.dot(rxn_amounts)

            # tweak to ensure feasibility of solution
            p_s = occs.dot(p_ind)
            invalid = (p_s < 0.)
            if any(invalid):
                f = min(abs(p_s_max[invalid]
                            / (p_s_max[invalid] - p_s[invalid])))
                f -= 1.e-6  # a little extra nudge into the valid domain
                p_ind = f*p_ind + (1.-f)*p_ind_max
                """
                print('o', rxn_amounts)
                rxn_amounts[:] = pinv(rxn_matrix.T).dot(p_ind - p_ind_start)
                print('n', rxn_amounts)
                """
                """
                p_s = occs.dot(p_ind)
                invalid = (p_s < 0.)
                if any(invalid):
                    print('WHY!!! ARGH!!!!!')
                """

            self.set_composition_from_p_ind(p_ind)
            self.equilibrate_clusters()
            grad = rxn_matrix.dot(self.molar_chemical_potentials)
            return self.molar_helmholtz, grad

        self.set_state(temperature)

        # Find ordered and disordered states to use as starting points
        p_cl_grd = self.ground_state_cluster_proportions_from_composition(c_arr)
        p_cl_max = self.maximum_entropy_cluster_proportions_from_composition(c_arr)
        p_ind_max = self.A_ind.T.dot(p_cl_max)
        p_s_max = occs.dot(p_ind_max)

        # minimize using near-ground state as a starting point
        p_cl = 0.95*p_cl_grd + 0.05*p_cl_max
        p_ind0 = self.A_ind.T.dot(p_cl)

        # keep_feasible=True requires some future version of scipy
        cons = LinearConstraint(occs.dot(self.isochemical_reactions.T),
                                0.-occs.dot(p_ind0),
                                1.-occs.dot(p_ind0))

        guess = np.array([0. for i in range(self.n_reactions)])
        # minimize using near-ground state as a starting point
        res = minimize(energy,
                       guess,
                       method='SLSQP', constraints=cons,
                       args=(p_ind0, self.isochemical_reactions),
                       jac=True)

        # store c and E
        c = self.c
        p_ind = self.p_ind
        E = res.fun

        # minimize using near-maximum entropy as a starting point
        p_cl = 0.05*p_cl_grd + 0.95*p_cl_max
        p_ind0 = self.A_ind.T.dot(p_cl)

        # keep_feasible=True requires some future version of scipy
        cons = LinearConstraint(occs.dot(self.isochemical_reactions.T),
                                0.-occs.dot(p_ind0),
                                1.-occs.dot(p_ind0))

        guess = np.array([0. for i in range(self.n_reactions)])
        # minimize using near-ground state as a starting point
        res = minimize(energy,
                       guess,
                       method='SLSQP', constraints=cons,
                       args=(p_ind0, self.isochemical_reactions),
                       jac=True)

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
        res = linprog(c=(self.cluster_energies
                         + self._delta_cluster_energies),
                      A_eq=self.A_ind.T,
                      b_eq=p_ind,
                      bounds=[(0., 1.) for es in self.cluster_energies],
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
                 - (self.block_energies / (R * temperature * np.power(self.block_size, self.gamma))))
        return (self.block_size
                * np.exp(np.where(lnval > 100, 100., lnval)))

    def set_cluster_proportions(self):
        cf = self._cluster_proportions(self.c, self.temperature)
        self.cluster_proportions = cf
        self.ln_cluster_proportions = logish(cf)

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
                near_ord_c = (logish(p_cl_near_ord[self.pivots_flat])
                              + (self.block_energies[self.pivots_flat]
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
                                    print(self._cluster_proportions(sol.x,
                                                                    T_steps[-1]))
                                    raise Exception('Clusters could not be '
                                                    'equilibrated, '
                                                    'even with annealing')
                                T = T_steps[i-1]
                                good = False
                                break

                    self.c = sol.x

        self.equilibrated_clusters = True
        self.set_cluster_proportions()

    def check_equilibrium(self):
        if self.equilibrated_clusters:
            pass
        else:
            raise Exception('This property of the solution requires '
                            'you to first call equilibrate_clusters()')

    def _dpcl_dp_ind(self, cluster_proportions_flat):
        dp_ind_dc = np.einsum('ijk, i -> kj', self._AA,
                              self.cluster_proportions)
        dpcl_dc = np.einsum('ij, i -> ij', self.A_ind,
                            self.cluster_proportions)

        # the returned solve is equivalent to
        # np.einsum('lj, jk -> lk', dpcl_dc, pinv(dp_ind_dc)))
        try:
            return solve(dp_ind_dc.T, dpcl_dc.T).T
        except np.linalg.LinAlgError:  # singular matrix
            return np.einsum('lj, jk -> lk', dpcl_dc, pinv(dp_ind_dc))

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
        return np.einsum('i, j, ij', self.p_ind, self.p_ind,
                         self.independent_interactions)

    @property
    def partial_molar_entropies(self):
        self.check_equilibrium()

        D = self._dpcl_dp_ind(self.cluster_proportions)
        g_S_n = -R*(np.einsum('lk, l', D, self.ln_cluster_proportions))

        return g_S_n / self.block_size

    @property
    def molar_chemical_potentials(self):
        """
        The chemical potentials per mole of sites
        """
        g_S_t = self.partial_molar_entropies

        D = self._dpcl_dp_ind(self.cluster_proportions)

        dps = [np.eye(e-s) - np.einsum('i, j->ij', self.p_s[s:e], np.ones(e-s))
               for (s, e) in self.site_bounds]
        dps = block_diag(*dps)

        g_E = (np.einsum('lk, l->k', D, self.block_energies)
               + np.einsum('i, ik, km, lm->l',
                           self.cluster_proportions,
                           self.cluster_interactions,
                           dps,
                           self.independent_cluster_occupancies))

        chemical_potentials = g_E - self.temperature * g_S_t

        return chemical_potentials + self.non_ideal_potentials()

    @property
    def molar_entropy(self):
        """
        The entropy per mole of sites
        """
        self.check_equilibrium()

        # Cluster and site entropy contributions per cluster
        S_n = -R*np.sum(self.cluster_proportions
                        * self.ln_cluster_proportions) / self.block_size * np.power(self.block_size, self.gamma)

        return S_n

    @property
    def molar_internal_energy(self):
        E = np.sum(self.cluster_proportions
                   * self.block_energies) / self.block_size
        return E + self.non_ideal_energy()

    @property
    def molar_helmholtz(self):
        """
        The gibbs free energy per mole of sites
        """
        S_t = self.molar_entropy
        E_t = self.molar_internal_energy
        F_t = E_t - self.temperature * S_t

        return F_t
