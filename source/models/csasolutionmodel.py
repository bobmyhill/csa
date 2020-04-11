import numpy as np
from numpy.linalg import pinv, lstsq, solve
from scipy.optimize import root, linprog, minimize, LinearConstraint
import itertools
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


class CSAModel(object):
    """
    This is the base class for all Cluster/Site Approximation models
    """

    def __init__(self, cluster_energies, gamma, site_species):

        self.n_sites = len(cluster_energies.shape)
        self.species_per_site = np.array(cluster_energies.shape)

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
        self.set_site_species = sorted(list(set(self.site_species_flat)))

        # Make correlation matrix between composition and site species
        self.site_species_compositions = np.zeros((len(self.site_species_flat),
                                                   len(self.set_site_species)),
                                                  dtype='int')
        for i, ss in enumerate(self.site_species_flat):
            self.site_species_compositions[i,
                                           self.set_site_species.index(ss)] = 1

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

        self.independent_cluster_occupancies = np.array([self.cluster_occupancies[p] for p in self.pivots],
                                                        dtype='int')

        self.independent_cluster_compositions = np.array([self.cluster_compositions[p] for p in self.pivots],
                                                         dtype='int')

        null = Matrix(self.independent_cluster_compositions.T).nullspace()
        rxn_matrix = np.array([np.array(v).T[0] for v in null])
        self.isochemical_reactions = rxn_matrix

        self._ps_to_p_ind = pinv(self.independent_cluster_occupancies.T)

        self.A_ind = lstsq(self.independent_cluster_occupancies.T,
                           self.cluster_occupancies.T,
                           rcond=None)[0].round(decimals=10).T

        self._AA = np.einsum('ik, ij -> ijk', self.A_ind, self.A_ind)
        self._AAA = np.einsum('il, ik, ij -> ijkl',
                              self.A_ind, self.A_ind, self.A_ind)

        self.gamma = gamma

    def set_state(self, temperature):
        self.temperature = temperature
        self.equilibrated_clusters = False

    def set_composition_from_p_ind(self, p_ind):
        self.p_ind = p_ind
        self.p_s = self.independent_cluster_occupancies.T.dot(self.p_ind)
        self.ln_p_s = logish(self.p_s)

        self._ideal_c = logish(self._independent_ideal_cluster_proportions())

        self.equilibrated_clusters = False

    def set_composition_from_p_s(self, p_s):
        p_ind = self._ps_to_p_ind.dot(p_s)
        self.set_composition_from_p_ind(p_ind)

    def compositional_array(self, composition):
        # first, make the numpy array
        arr_c = np.zeros(len(self.set_site_species))
        for k, v in composition.items():
            arr_c[self.set_site_species.index(k)] = v
        return arr_c

    def ground_state_cluster_proportions(self, composition, normalize=True):

        # first, make the numpy array
        arr_c = self.compositional_array(composition)

        # Apply a small random tweak to all energies to force the
        # solver to converge to the state with the minimum number of clusters
        # in the (common) circumstance where clusters are degenerate
        np.random.seed(seed=19)
        std = np.std(self.cluster_energies_flat)
        c_rand = np.random.rand(len(self.cluster_energies_flat))*std*1.e-10

        # Solve the linear programming problem
        # Revised simplex is slower than interior-point,
        # but is required to find an accurate solution
        # (especially with the small random tweaks applied)
        res = linprog(c=self.cluster_energies_flat + c_rand,
                      A_eq=self.cluster_compositions.T,
                      b_eq=arr_c,
                      bounds=[(0., 1.) for l in self.cluster_energies_flat],
                      method='revised simplex')

        if not res.success:
            raise Exception('Minimum ground state energy not found.')

        # Normalise the proportions if required
        proportions = res.x
        if normalize:
            proportions /= np.sum(res.x)
        return proportions

    def maximum_entropy_cluster_proportions(self, composition, normalize=True):
        def minus_entropy(p_ind):
            p_s = self.independent_cluster_occupancies.T.dot(p_ind)
            norm_sum_p_s = self.n_sites*np.sum(p_s)
            S_s = np.sum(p_s*np.log(p_s/np.sum(norm_sum_p_s)))
            return S_s

        c_arr = self.compositional_array(composition)

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

        self.set_state(temperature)

        def energy(p_ind):
            self.set_composition_from_p_ind(p_ind)
            self.equilibrate_clusters()
            # J = self.molar_chemical_potentials
            return self.molar_gibbs

        c_arr = self.compositional_array(composition)

        cons = (LinearConstraint(self.independent_cluster_occupancies.T,
                                 0., 1.),
                LinearConstraint(self.independent_cluster_compositions.T,
                                 c_arr, c_arr))

        # Find ordered and disordered states to use as starting points
        p_cl_grd = self.ground_state_cluster_proportions(composition)
        p_cl_max = self.maximum_entropy_cluster_proportions(composition)

        # minimize using near-ground state as a starting point
        p_cl = 0.95*p_cl_grd + 0.05*p_cl_max
        res = minimize(energy,
                       self.A_ind.T.dot(p_cl),
                       method='SLSQP', constraints=cons,
                       jac=None)

        # store c and E
        c = self.c
        p_ind = self.p_ind
        E = res.fun

        # minimize using near-maximum entropy as a starting point
        p_cl = 0.05*p_cl_grd + 0.95*p_cl_max
        res = minimize(energy,
                       self.A_ind.T.dot(p_cl),
                       method='SLSQP', constraints=cons,
                       jac=None)

        if res.fun > E:
            self.set_composition_from_p_ind(p_ind)
            self.c = c
            self.equilibrated_clusters = True
            self.set_cluster_proportions()

    def _independent_ideal_cluster_proportions(self):
        occs = np.einsum('ij, j->ij', self.independent_cluster_occupancies,
                         self.p_s)

        i = 0
        proportions = np.ones(len(self.independent_cluster_occupancies))
        for n_species in self.species_per_site:
            proportions *= np.sum(occs[:, i:i+n_species], axis=1)
            i += n_species
        return proportions

    def _ideal_cluster_proportions(self, p_s):
        occs = np.einsum('ij, j->ij', self.cluster_occupancies, p_s)

        i = 0
        proportions = np.ones(len(self.cluster_occupancies))
        for n_species in self.species_per_site:
            proportions *= np.sum(occs[:, i:i+n_species], axis=1)
            i += n_species
        return proportions

    @property
    def ideal_cluster_proportions(self):
        return self._ideal_cluster_proportions(self.p_s)

    def _cluster_proportions(self, c, temperature):
        lnval = self.A_ind.dot(c) - self.cluster_energies_flat/(R*temperature)
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
        else:
            sol = root(self.delta_proportions, self._ideal_c, jac=True,
                       args=(self.temperature),
                       method='lm', options={'ftol': 1.e-16})
            # + np.random.rand(len(self._ideal_c))*1.e-2)

            # print('ideal', self._ideal_c)
            if np.max(np.abs(sol.fun)) < 1.e-10:
                self.c = sol.x
            else:
                # try to anneal
                T = 10000.
                n_steps = 4
                good = True
                while T > self.temperature + 100. or not good:
                    T_steps = np.logspace(np.log10(T),
                                          np.log10(self.temperature), n_steps)

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
        return solve(dp_ind_dc.T, dpcl_dc.T).T

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

        n_site_atoms = np.sum(self.p_s)
        # n_clusters = np.sum(self.cluster_proportions_flat)

        d2S_dpcl2 = np.diag(inverseish(self.cluster_proportions_flat)) - 1.
        hess_S_n = -R*(np.einsum('im, ij, mk-> jk', d2S_dpcl2, D, D)
                       + np.einsum('i, ijk->jk',
                                   self.ln_cluster_proportions_flat, F))

        hess_S_i = -R*(np.einsum('ki, ji, i->jk',
                                 self.independent_cluster_occupancies,
                                 self.independent_cluster_occupancies,
                                 inverseish(self.p_s)) - n_site_atoms)

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

        # remember gamm is the number of noninterfering clusters PER SITE
        S_t = self.gamma * S_n + (1./self.n_sites - self.gamma) * S_i

        return S_t

    @property
    def hessian_gibbs(self):
        F = self._d2pcl_dp_ind_dp_ind()

        hess_G = np.einsum('ijk, i -> jk', F, self.cluster_energies_flat)
        hess_S = self.hessian_entropy

        return self.gamma * hess_G - self.temperature * hess_S

    @property
    def molar_chemical_potentials(self):
        """
        The chemical potentials per mole of sites
        """
        g_S_t = self.partial_molar_entropies

        D = self._dpcl_dp_ind()
        g_E = np.einsum('lk, l', D, self.cluster_energies_flat)
        chemical_potentials = self.gamma * g_E - self.temperature * g_S_t

        return chemical_potentials

    @property
    def molar_gibbs(self):
        """
        The gibbs free energy per mole of sites
        """
        S_t = self.molar_entropy
        E_t = np.sum(self.cluster_proportions * self.cluster_energies)
        G_t = self.gamma * E_t - self.temperature * S_t

        return G_t
