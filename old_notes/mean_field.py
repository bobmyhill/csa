import numpy as np
import matplotlib.pyplot as plt

def helmholtz(m, T, J, z, h):
    """
    T = # temperature
    J = # interaction energy
    N = # number of lattice sites
    m = # mean spin
    z = # number of nearest neighbours
    h = # external field
    """
    k = 8.31446
    beta = 1./(k*T)
    h_eff = h + m*J*z

    # Z = np.exp(-beta*J*m*m*N*z/2.)*np.power(2.*np.cosh(beta*(h + m*J*z)), N)
    lnZoverN = -beta*J*m*m*z/2. + np.log(np.cosh(beta*h_eff)) + np.log(2.)


    #lnZoverN = -beta*J*m*m*z/2. + np.log(np.exp((1. - m)*beta*h_eff) + np.exp((m)*-beta*h_eff))

    return -lnZoverN/beta

ms = np.linspace(-1., 1., 1001)
Ts = np.linspace(0.01, 1.5*12./8.31446, 101)

J = 1.
z = 12.

Fs = []
for T in Ts:
    #plt.plot(ms, helmholtz(ms, T, J, z, h=0.))
    Fs.append(np.min(helmholtz(ms, T, J, z, h=0.)))

for m in np.linspace(0., 1., 11):
    plt.plot(Ts, -np.gradient(helmholtz(m, Ts, J, z, h=0.), Ts), linestyle=':')
plt.plot(Ts, -np.gradient(Fs, Ts))

plt.show()
