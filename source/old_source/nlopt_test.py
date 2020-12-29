import nlopt
import numpy as np


def myfunc(x, grad):
    if grad.size > 0:
        grad[:] = [0.0, 0.5 / np.sqrt(x[1])]
    return np.sqrt(x[1])


def constraint_func(a, b):
    def constraints(result, x, grad):
        if grad.size > 0:
            grad[:] = np.array([3 * a * (a*x[0] + b)**2, [-1.0, -1.0]]).T
        result[:] = (a*x[0] + b)**3 - x[1]
    return constraints


opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([-float('inf'), 0])
opt.set_min_objective(myfunc)

opt.add_inequality_mconstraint(constraint_func(np.array([2, -1]),
                                               np.array([0, 1])),
                               [1e-8, 1e-8])
opt.set_xtol_rel(1e-4)


# Can also set equality constraints and remove old constraints.

x = opt.optimize([1.234, 5.678])
minf = opt.last_optimum_value()

print("optimum at ", x[0], x[1])
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())
