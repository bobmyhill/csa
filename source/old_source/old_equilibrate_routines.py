    """
    def equilibrate_nlopt(self, composition, temperature):

        self.set_state(temperature)

        def energy(p_ind_start, rxn_matrix):
            def energy_func(rxn_amounts, grad):
                p_ind = p_ind_start + rxn_matrix.T.dot(rxn_amounts)
                print(p_ind)
                self.set_composition_from_p_ind(p_ind)
                print(self.equilibrated_clusters)
                self.equilibrate_clusters()
                print(self.equilibrated_clusters)
                if grad.size > 0:
                    grad[:] = rxn_matrix.dot(self.molar_chemical_potentials)
                return self.molar_gibbs
            return energy_func

        def solution_bounds(occs_start, rxn_occs):
            # The site occupancies must all be positive (also less than 1).
            def constraints(result, x, grad):
                if grad.size > 0:
                    grad[:] = rxn_occs
                result[:] = occs_start + rxn_occs.dot(x)
            return constraints


        def compositional_constraints(i_comps, c_arr):
            # The composition must match that given by the user
            def constraints(result, x, grad):
                if grad.size > 0:
                    grad[:] = i_comps
                result[:] = i_comps.dot(x) - c_arr
                print(result)
            return constraints


        # Find ordered and disordered states to use as starting points
        p_cl_grd = self.ground_state_cluster_proportions_from_composition(composition)
        p_cl_max = self.maximum_entropy_cluster_proportions_from_composition(composition)

        # minimize using near-ground state as a starting point
        p_cl_start = 0.85*p_cl_grd + 0.15*p_cl_max
        p_ind_start = self.A_ind.T.dot(p_cl_start)

        rxn_matrix = self.isochemical_reactions

        opt = nlopt.opt(nlopt.LD_SLSQP, self.n_reactions)
        opt.set_min_objective(energy(p_ind_start, rxn_matrix))
        opt.set_initial_step(1.e-3)
        occs_start = self.independent_cluster_occupancies.T.dot(p_ind_start)
        rxn_occs = self.independent_cluster_occupancies.T.dot(rxn_matrix.T)
        opt.add_inequality_mconstraint(solution_bounds(occs_start, rxn_occs),
                                       [0 for i in range(self.n_site_species)])


        i_comps = self.independent_cluster_compositions.T
        c_arr = self.compositional_array(composition)
        opt.add_equality_mconstraint(compositional_constraints(i_comps, c_arr),
                                     [1.e-8 for i in range(len(c_arr))])


        opt.set_xtol_rel(1e-4)

        guess = np.array([0. for i in range(self.n_reactions)])
        x = opt.optimize(guess)
        minf = opt.last_optimum_value()

        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())

        exit()
    """

    """
    def equilibrate_old(self, composition, temperature):

        self.set_state(temperature)

        def energy(p_ind):
            self.set_composition_from_p_ind(p_ind)
            self.equilibrate_clusters()
            # J = self.molar_chemical_potentials
            return self.molar_gibbs

        c_arr = self.compositional_array(composition)

        cons = (LinearConstraint(self.independent_cluster_occupancies.T,
                                 0., 1., keep_feasible=True),  # requires some future version of scipy
                LinearConstraint(self.independent_cluster_compositions.T,
                                 c_arr, c_arr))

        # Find ordered and disordered states to use as starting points
        p_cl_grd = self.ground_state_cluster_proportions_from_composition(composition)
        p_cl_max = self.maximum_entropy_cluster_proportions_from_composition(composition)

        # minimize using near-ground state as a starting point
        p_cl_guess = 0.05*p_cl_grd + 0.95*p_cl_max
        res = minimize(energy,
                       self.A_ind.T.dot(p_cl_guess),
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
    """
