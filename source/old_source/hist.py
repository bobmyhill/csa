import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk
from scipy.integrate import cumtrapz

# Taken from Plischke and Bergesen, 1989
# Except there seems to be an error in the expression for q...
# The correct expression can be found on wiki
# (https://en.wikipedia.org/wiki/Square_lattice_Ising_model)
R = 8.31446

def coth(x):
     return 1./np.tanh(x)

def q(K, L):
    k = 1./(np.sinh(2.*K)*np.sinh(2.*L))
    m = 4.*k*np.power(1.+k, -2.)
    return m

def energy(T, J):
    beta = 1./(R*T)
    K = beta*J
    L = beta*J
    K1q = ellipk(q(K, L))
    u = -J*coth(2.*K)*(1. + 2./np.pi*(2.*(np.tanh(2.*K))**2 - 1.)*K1q)
    return u

def entropies_heat_capacities(energies, temperatures):
    c = np.gradient(energies, temperatures)
    s = cumtrapz(np.gradient(energies, temperatures)/temperatures,
                 temperatures, initial=0)
    return s, c

J = R
Tc = 2.*J/(R*np.log(1. + np.sqrt(2.))) # for K=L=betaJ

temperatures = np.linspace(Tc - 0.1, Tc + 0.1, 100001) # *J/R
temperatures = np.linspace(0.01, 10., 1000001)*J/R
energies = energy(temperatures, J)
entropies, heat_capacities = entropies_heat_capacities(energies, temperatures)
# Data from Thanh Ngo et al. (2014) and Beath and Ryan (2006)
T, E = np.loadtxt('FCC_AF_3d_Ising.dat', unpack=True)
S, C = entropies_heat_capacities(E, T)


#fig = plt.figure(figsize=(15, 5))
#ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

plt.scatter(T, S, label='3D Ising (AF FCC lattice)')
plt.plot(T, T*0. + np.log(2.), label='ideal')

#a = [-2.75, -3. ,  -3. ,  -3.75, -3.75, -3.,   -3. ,  -2.75, -3. ,  -2.75, -2.75, -3.  , -3.  , -2.75 ,-2.75, -3., -2.25, -3.,   -3.,   -3.25, -3.,   -3.25, -3.25, -3. ,  -3.,   -3.25, -3.25, -3., -3.25, -3.,   -3.,   -2.25]
a = np.array([-2.25, -3.,   -3.,   -3.25, -3. ,  -3.25, -3.25, -3. ,  -3. ,  -3.25, -3.25, -3., -3.25, -3.,   -3.,   -2.25])
#plt.hist(a, bins=11)

b = np.array([  0., -3., -3., -4., -3., -4., -4., -3., -3., -4., -4., -3., -4., -3., -3.,   0.])

c = (a+b)/2.

energies = np.hstack((a, a, a, b))

for i, energies in enumerate([a, np.hstack((a, a, a, a,
                                            a, a, a, a,
                                            a, a, a, a,
                                            a, a, a, a))]): # b, np.hstack((a, b)), 



    RTs = np.linspace(0.01, 2., 1001)

    ps = np.exp(np.einsum('i, j -> ij', -energies, 1./RTs))

    pfns = np.sum(ps, axis=0)

    ps = ps/pfns
    plnps = ps*np.log(ps)
    Ss = -np.sum(plnps, axis=0)

    Es = np.einsum('i, ij -> j', energies, ps)

    s = np.log(len(energies))/np.log(16)

    Cp = np.gradient(Es, RTs, edge_order=2)
    Ss2 = cumtrapz(Cp/RTs, RTs, initial=0.)
    if i > 2:
        plt.plot(RTs, Ss/4./s)
    else:
        plt.plot(RTs, Ss/4./s, linestyle=':')


    #if i==0:
    #    plt.plot(RTs, (Ss/4./s))
plt.show()
exit()






plt.hist(b, bins=21)
plt.hist(energies, bins=101)
plt.show()
