import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d

d20 = np.loadtxt('Beath_Ryan_2006_FCC_Ising_antiferromagnet_w_axes_20.dat')
d24 = np.loadtxt('Beath_Ryan_2006_FCC_Ising_antiferromagnet_w_axes_24.dat')

#AF takes the negative of the energy
E24 = np.array(-d24[::-1,0])
ln_g24 = np.array(d24[::-1,1])*10000./np.log10(4.)
g24 = np.power(np.exp(1), d24[::-1,1])


ln_g24_spline=interp1d(E24, ln_g24, kind='quadratic')

Es = np.linspace(-2., 6., 10001)
ln_gs = ln_g24_spline(Es)

Ts = np.linspace(0.00001, 0.00015, 1001)
Ts = np.linspace(0.00002195, 0.0000220, 11)
E =  np.empty_like(Ts)
for i, T in enumerate(Ts):
    beta = 1./T
    ln_p = ln_gs - beta*Es
    ln_p -= np.max(ln_p)
    sump = sum(np.exp(ln_p))
    ln_p -= np.log(sump)
    p = np.exp(ln_p)
    E[i] = np.sum(p*Es)
    plt.plot(Es, np.exp(ln_p))

#plt.plot(Ts, E)
#plt.xlim(-2.1, 0.2)
plt.show()

exit()





scale = simps(g24, E24)
g24 /= scale

betas = 1./np.linspace(0.01, 5, 101)
us = np.empty_like(betas)

for i, beta in enumerate(betas):
    Z = simps(g24*np.exp(-beta*E24), E24)
    us[i] = simps(g24*E24*np.exp(-beta*E24), E24)/Z

"""
plt.plot(1./betas, us, label='24')
plt.legend()

plt.show()
exit()
"""
f = 1.04
plt.plot(d20[:,0], d20[:,1]*f, label='20')
plt.plot(d24[:,0], d24[:,1]*np.log10(4), label='24')
plt.legend()

plt.show()
