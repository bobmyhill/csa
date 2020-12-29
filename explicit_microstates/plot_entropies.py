import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad, cumtrapz
from scipy.special import erfi

R = 8.31446
expeps = 1.e-2


def einstein_energy(T, u_0, u_inf, s_inf):
    mask = (s_inf * T) > expeps * 2. * (u_inf - u_0)
    if type(T) is float:
        if not mask:
            return u_0
        else:
            x = 2. * (u_inf - u_0) / (s_inf * T)
            return u_inf + 2.*(u_inf - u_0)*(x * np.exp(x)
                                             / np.power(np.exp(x) - 1.0, 2.0)
                                             - ((1. / (np.exp(x) - 1.0)) + 0.5))
    else:
        x = np.zeros_like(T)
        u = np.zeros_like(T) + u_0
        x[mask] = 2. * (u_inf - u_0) / (s_inf * T[mask])
        u[mask] = u_inf + 2.*(u_inf - u_0)*(x[mask] * np.exp(x[mask])
                                            / np.power(np.exp(x[mask]) - 1.0, 2.0)
                                            - ((1. / (np.exp(x[mask]) - 1.0)) + 0.5))
        return u


def einstein_entropy(T, u_0, u_inf, s_inf):
    mask = (s_inf * T) > expeps * 2. * (u_inf - u_0)
    if type(T) is float:
        if not mask:
            return 0.
        else:
            x = 2. * (u_inf - u_0) / (s_inf * T)
            return s_inf * x * x * np.exp(x) / np.power(np.exp(x) - 1.0, 2.0)
    else:
        x = np.zeros_like(T)
        s = np.zeros_like(T)
        x[mask] = 2. * (u_inf - u_0) / (s_inf * T[mask])
        s[mask] = s_inf * x[mask] * x[mask] * np.exp(x[mask]) / np.power(np.exp(x[mask]) - 1.0, 2.0)

        return s

"""
print(einstein_energy(0.001, -3, -1, 4))
print(einstein_entropy(0.001, -3, -1, 4))

temperatures = np.linspace(0.0, 2., 1001)
plt.plot(temperatures, einstein_energy(temperatures, -3, -1, 4))
plt.plot(temperatures, einstein_entropy(temperatures, -3, -1, 4))
plt.show()
"""


def Efn(e_min, e_max, e_ideal, sd, T):

    erfn_min = erfi((e_min - e_ideal)/sd - sd/(2.*R*T))
    erfn_max = erfi((e_max - e_ideal)/sd - sd/(2.*R*T))

    fnexp = np.exp(-(sd**2 + 4*e_ideal*(R*T))/(4*(R*T)**2))

    e_min = 1/4*sd*(2*sd*np.exp((e_ideal - e_min)**2/sd**2 - e_min/(R*T))
                    + (np.sqrt(np.pi) * (sd**2 + 2 * e_ideal * (R*T))
                       * fnexp * erfn_min)
                    / (R*T))

    e_max = 1/4*sd*(2*sd*np.exp((e_ideal - e_max)**2/sd**2 - e_max/(R*T))
                    + (np.sqrt(np.pi) * (sd**2 + 2 * e_ideal * (R*T))
                       * fnexp * erfn_max)
                    / (R*T))

    Z = (sd * np.sqrt(np.pi) * (erfn_max - erfn_min) * fnexp / 2)
    return (e_max - e_min)/Z


def es(x, mean, sd, scale, T):
    return norm.pdf(x, mean, sd)*scale*x*np.exp(-x/(R*T))


def ps(x, mean, sd, scale, T):
    return norm.pdf(x, mean, sd)*scale*np.exp(-x/(R*T))


prms = [[-128, -64, -96, 5.28, 0.5],
        [-128, -68, -98, 5.06, 0.625],
        [-128, -80, -104, 4.16, 0.75],
        [-128, -100, -114, 2.56, 0.875]]

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

temperatures = np.linspace(0.01, 1.2, 10001)
energies = np.empty_like(temperatures)
helmholtz = np.empty_like(temperatures)

for (min, max, mean, sd, fraction) in prms:
    """
    for i, T in enumerate(temperatures):
        energies[i] = (quad(es, min, max, args=(mean, sd, 1., T/f))[0]
                       / quad(ps, min, max, args=(mean, sd, 1., T/f))[0])
    # plt.plot(temperatures, energies/32.)

    entropies = cumtrapz(np.gradient(energies, temperatures)/temperatures,
                         temperatures, initial=0)
    """
    S_ideal = -(fraction*np.log(fraction) + (1. - fraction)*np.log(1.-fraction))

    f2 = S_ideal/-np.log(0.5)
    f = 4.*f2 # this one modifies the entropy

    energies = Efn(min/f, max/f, mean/f, sd/f, temperatures)
    entropies = cumtrapz(np.gradient(energies, temperatures)/temperatures,
                         temperatures, initial=0)

    ax[0].plot(temperatures, entropies/32.*f, label=fraction, c='blue')
    ax[1].plot(temperatures, energies/32.*f, label=fraction, c='blue')

    Ss = einstein_entropy(temperatures, min/32., mean/32., S_ideal)
    ax[0].plot(temperatures, Ss, c='red')

    Es = einstein_energy(temperatures, min/32., mean/32., S_ideal)

    ax[1].plot(temperatures, Es, c='red')
    ax[2].plot(temperatures, Es - temperatures*Ss, label=fraction)

for p in [0.5, 0.625, 0.75, 0.875]:
    S = -(p*np.log(p) + (1.-p)*np.log(1.-p))
    ax[0].plot(temperatures, S + temperatures*0., linestyle=':')


#plt.xlim(0, 2.)
#plt.ylim(-0.5, 1.)
ax[0].legend()
ax[2].legend()
plt.show()
