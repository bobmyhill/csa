import numpy as np
from scipy.optimize import root
import itertools

R = 8.31446

def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps:
    log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    mask = x>eps
    ln = np.where(x<=eps, np.log(eps) - f_eps - f_eps*f_eps/2., 0.)
    ln[mask] = np.log(x[mask])
    return ln

class CSAModel(object):

    """
    This is the base class for all Cluster/Site Approximation models
    """
    def __init__(self, cluster_energies, gamma):

        self.n_sites = len(cluster_energies.shape)
        self.species_per_site = np.array(cluster_energies.shape)

        clusters = itertools.product(*[np.identity(n_species, dtype='int')
                                       for n_species in cluster_energies.shape])

        self.cluster_energies = cluster_energies.flatten()
        self.cluster_occupancies = np.array([np.hstack(cl) for cl in clusters])

        self.pivots = [i for i in range(self.species_per_site[0])]
        for i in range(1, self.n_sites):
            self.pivots.append(np.prod(self.species_per_site[0:i]))

        self.independent_cluster_occupancies = np.array([self.cluster_occupancies[p]
                                                         for p in self.pivots],
                                                        dtype='int')

        self._ps_to_p_ind = np.linalg.pinv(self.independent_cluster_occupancies.T)

        self.A_ind = np.linalg.lstsq(self.independent_cluster_occupancies.T,
                                     self.cluster_occupancies.T,
                                     rcond=None)[0].round(decimals=10).T

        self._AA = np.einsum('ik, ij -> ijk', self.A_ind, self.A_ind)

        self.gamma = gamma

    def set_composition_from_p_ind(self, p_ind):
        self.p_ind = p_ind

        self.p_s = self.independent_cluster_occupancies.T.dot(self.p_ind)
        self.ln_p_s =logish(self.p_s)

        self._ideal_c = logish(self._independent_ideal_cluster_proportions())

        self.equilibrated_clusters = False

    def set_composition_from_p_s(self, p_s):
        p_ind = self._ps_to_p_ind.dot(p_s)
        self.set_composition_from_p_ind(p_ind)


    def _independent_ideal_cluster_proportions(self):
        occs = np.einsum('ij, j->ij', self.independent_cluster_occupancies,
                         self.p_s)

        i = 0
        proportions = np.ones(len(self.independent_cluster_occupancies))
        for n_species in self.species_per_site:
            proportions *= np.sum(occs[:,i:i+n_species], axis=1)
            i += n_species
        return proportions

    @property
    def ideal_cluster_proportions(self):
        occs = np.einsum('ij, j->ij', self.cluster_occupancies, self.p_s)

        i = 0
        proportions = np.ones(len(self.cluster_occupancies))
        for n_species in self.species_per_site:
            proportions *= np.sum(occs[:,i:i+n_species], axis=1)
            i += n_species
        return proportions


    def set_state(self, temperature):
        self.temperature = temperature
        self.equilibrated_clusters = False

    def _cluster_proportions(self, c, T):
        lnval = self.A_ind.dot(c) - self.cluster_energies/(R*T)
        return np.exp(np.where(lnval>100, 100., lnval))

    def delta_proportions(self, c, temperature):
        p_cl = self._cluster_proportions(c, temperature)
        deltas = self.p_ind - self.A_ind.T.dot(p_cl)
        jac = -np.einsum('ijk, i -> kj', self._AA, p_cl)
        return deltas, jac

    def equilibrate_clusters(self):
        if np.max(self._ideal_c) > -1.e-10: # pure endmember
            self.c = self._ideal_c
            self.equilibrated_clusters = True
            self.cluster_proportions = self.ideal_cluster_proportions
            self.ln_cluster_proportions =logish(self.cluster_proportions)
        else:
            sol = root(self.delta_proportions, self._ideal_c, jac=True,
                       args=(self.temperature), method='lm', options={'ftol':1.e-16})# + np.random.rand(len(self._ideal_c))*1.e-2)

            #print('ideal', self._ideal_c)
            if np.max(np.abs(sol.fun)) < 1.e-10:
                self.c = sol.x
                self.equilibrated_clusters = True
                self.cluster_proportions = self._cluster_proportions(self.c,
                                                                     self.temperature)
                self.ln_cluster_proportions =logish(self.cluster_proportions)
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
                                   method='lm', tol=1.e-10, options={'ftol':1.e-16})

                        if np.max(np.abs(sol.fun)) < 1.e-10:
                            guess_c = sol.x
                            good = True

                        else:
                            n_steps *= 2

                            if n_steps > 100:
                                print(sol)
                                print(self._cluster_proportions(self.c, self.temperature))
                                raise Exception('Clusters could not be equilibrated, '
                                                'even with annealing')
                            T = T_steps[i-1]
                            good = False
                            break

                self.c = sol.x
                self.equilibrated_clusters = True
                self.cluster_proportions = self._cluster_proportions(self.c,
                                                                     self.temperature)
                self.ln_cluster_proportions =logish(self.cluster_proportions)
        #print('final', self.c)

    def _inv_J(self):
        return np.einsum('ijk, i -> kj', self._AA, self.cluster_proportions)

    def _dpcl_dp_ind(self):
        dp_ind_dc = np.einsum('ijk, i -> kj', self._AA, self.cluster_proportions)
        dpcl_dc = np.einsum('ij, i -> ij', self.A_ind, self.cluster_proportions)

        # the returned solve is equivalent to
        # np.einsum('lj, jk -> lk', dpcl_dc, np.linalg.pinv(dp_ind_dc)))
        return np.linalg.solve(dp_ind_dc.T, dpcl_dc.T).T

    def check_equilibrium(self):
        if self.equilibrated_clusters:
            pass
        else:
            raise Exception('This property of the solution requires '
                            'you to first call equilibrate_clusters()')

    @property
    def molar_chemical_potentials(self):
        """
        The chemical potentials on a per cluster basis
        """
        self.check_equilibrium()
        D = self._dpcl_dp_ind()

        g_E = np.einsum('lk, l', D, self.cluster_energies)
        g_S_n = -R*np.einsum('lk, l', D, self.ln_cluster_proportions)
        g_S_i = -R*(np.einsum('ik, i',
                              self.independent_cluster_occupancies.T,
                              self.ln_p_s))
        g_S_t = self.gamma * g_S_n + (1. - self.gamma) * g_S_i
        chemical_potentials = g_E - self.temperature*g_S_t

        return chemical_potentials

    @property
    def molar_gibbs(self):
        """
        The gibbs free energy per mole of clusters
        """
        S_t = self.molar_entropy
        E_t = np.sum(self.cluster_proportions * self.cluster_energies)
        G_t = E_t - self.temperature * S_t

        return G_t

    @property
    def molar_entropy(self):
        """
        The entropy per mole of clusters
        """
        self.check_equilibrium()

        S_n = -R*np.sum(self.cluster_proportions *
                        self.ln_cluster_proportions) # cluster entropy per cluster
        S_i = -R*np.sum(self.p_s * self.ln_p_s) # site entropy per cluster

        S_t = self.gamma * S_n + (1. - self.gamma) * S_i

        return S_t
