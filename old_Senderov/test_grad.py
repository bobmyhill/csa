import numpy as np
import matplotlib.pyplot as plt

E = np.array([1.5, 2., 4., 7., 9., 11., 16., 20., 23.])

# Four sites, two elements
def calculate_energies(p):
    u = p[0]*p[2]*p[4]*p[6]*E[0] + p[1]*p[2]*p[4]*p[6]*E[1]
    return u

# for i in range(2):
# for j in range(2):
# for k in range(2):
# for l in range(2):
    # aijkl = 0
    # for m in range(2):
    # for n in range(2):
    # for o in range(2):
    # for p in range(2):

#    if i==m:
#        aijkl += o[j]o[k]o[l]d[i,m]p[n]p[o]p[p]*Emnop
#    if j==n:
#        aijkl += d[j,n]p[m]p[o]p[p]*Emnop
#    ...
#    ...


xs = np.linspace(0.001, 0.999, 101)
us = np.empty_like(xs)
for i, x in enumerate(xs):
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x, 1.-x, x])
    us[i] = calculate_energies(p)

plt.plot(xs, us)

for (x, c) in [(0.2, 'red'),
               (0.7, 'blue')]:
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x, 1.-x, x])

    E_0 = calculate_energies(p)

    x += 1.e-7
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x, 1.-x, x])
    E_1 = calculate_energies(p)
    plt.plot([0, 1], [E_1-x*(E_1 - E_0)/1.e-7, E_1+(1.-x)*(E_1 - E_0)/1.e-7], c=c)

    mu0246 = (p[0]*p[2]*p[4]*E[0] + p[0]*p[2]*p[6]*E[0] + p[0]*p[4]*p[6]*E[0]
              + p[2]*p[4]*p[6]*E[0] - 3.*(p[0]*p[2]*p[4]*p[6])*E[0]
              + p[1]*p[2]*p[4]*E[1] + p[1]*p[2]*p[6]*E[1]
              + p[1]*p[4]*p[6]*E[1] - 3.*p[1]*p[2]*p[4]*p[6]*E[1])
    mu1357 = (-3.*(p[0]*p[2]*p[4]*p[6])*E[0]
              + p[2]*p[4]*p[6]*E[1] - 3.*p[1]*p[2]*p[4]*p[6]*E[1]) #+ p[0]*E[2] - p[0]*p[3]*E[2] # i.e. variables are treated separately.

    plt.scatter([0, x, 1], [mu0246, E_0, mu1357], c=c)
#plt.plot([0, 1], [mu0, mu1])
plt.show()
exit()


# Three sites, two elements
def calculate_energies(p):
    u = p[0]*p[2]*p[4]*E[0] + p[1]*p[2]*p[4]*E[1]
    return u

xs = np.linspace(0.001, 0.999, 101)
us = np.empty_like(xs)
for i, x in enumerate(xs):
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x])
    us[i] = calculate_energies(p)

plt.plot(xs, us)

for (x, c) in [(0.2, 'red'),
               (0.7, 'blue')]:
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x])

    E_0 = calculate_energies(p)

    x += 1.e-7
    p = np.array([1.-x, x, 1.-x, x, 1.-x, x])
    E_1 = calculate_energies(p)
    plt.plot([0, 1], [E_1-x*(E_1 - E_0)/1.e-7, E_1+(1.-x)*(E_1 - E_0)/1.e-7], c=c)

    mu024 = (p[0]*p[2]*E[0] + p[0]*p[4]*E[0] + p[2]*p[4]*E[0] - 2.*(p[0]*p[2]*p[4])*E[0]
             + p[1]*p[2]*E[1] + p[1]*p[4]*E[1] - 2.*p[1]*p[2]*p[4]*E[1])
    mu135 = (-2.*(p[0]*p[2]*p[4])*E[0]
             + p[2]*p[4]*E[1] - 2.*p[1]*p[2]*p[4]*E[1]) #+ p[0]*E[2] - p[0]*p[3]*E[2] # i.e. variables are treated separately.

    plt.scatter([0, x, 1], [mu024, E_0, mu135], c=c)
#plt.plot([0, 1], [mu0, mu1])
plt.show()
exit()

# Two sites, Two elements
def calculate_energies(p):
    u = p[0]*p[2]*E[0] + p[1]*p[2]*E[1] + p[0]*p[3]*E[2]
    return u

xs = np.linspace(0.001, 0.999, 101)
us = np.empty_like(xs)
for i, x in enumerate(xs):
    p = np.array([1.-x, x, 1.-x, x])
    us[i] = calculate_energies(p)

plt.plot(xs, us)

for x in [0.2, 0.7]:
    p = np.array([1.-x, x, 1.-x, x])

    E_0 = calculate_energies(p)

    x += 1.e-7
    p = np.array([1.-x, x, 1.-x, x])
    E_1 = calculate_energies(p)
    plt.plot([0, 1], [E_1-x*(E_1 - E_0)/1.e-7, E_1+(1.-x)*(E_1 - E_0)/1.e-7])

    mu02 = p[0]*E[0] + p[2]*E[0] - (p[0]*p[2])*E[0]  + p[1]*E[1] - p[1]*p[2]*E[1] + p[3]*E[2] - p[0]*p[3]*E[2]
    mu13 = -(p[0]*p[2])*E[0] + p[2]*E[1] - p[1]*p[2]*E[1] + p[0]*E[2] - p[0]*p[3]*E[2] # i.e. variables are treated separately.

    plt.scatter([0, x, 1], [mu02, E_0, mu13])
#plt.plot([0, 1], [mu0, mu1])
plt.show()
