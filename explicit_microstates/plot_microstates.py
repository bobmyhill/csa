import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import quad, cumtrapz

import sys
infile = sys.argv[1]
n_atoms = float(sys.argv[2])


energy_levels_and_degeneracies = np.loadtxt(infile)

energy_levels_and_degeneracies[:,0] /= -n_atoms

s = np.sum(energy_levels_and_degeneracies[:,1])
fs = energy_levels_and_degeneracies[:,1]/s
plt.scatter(energy_levels_and_degeneracies[:,0], fs)

def gauss(x, mean, sd, scale):
    return norm.pdf(x,mean,sd)*scale

mean_guess = energy_levels_and_degeneracies[np.argmax(fs),0]

de = 0.1
lims = np.array([np.min(energy_levels_and_degeneracies[:,0]), np.max(energy_levels_and_degeneracies[:,0])])
spread = np.max(np.abs(mean_guess - lims))
lims = [mean_guess-spread-de, mean_guess+spread+de]

popt, pcov = curve_fit(gauss, energy_levels_and_degeneracies[:,0], fs, p0=[mean_guess, 6, 4])
print(popt)

minmax = [min(energy_levels_and_degeneracies[:,0]),
          max(energy_levels_and_degeneracies[:,0])]
x = np.arange(minmax[0], minmax[1], 0.001) # range of x in spec
y = gauss(x, *popt)
plt.plot(x, y, linewidth=3)

#y = gauss(x, 80, 2.0, 3.84)
#plt.plot(-x/n_atoms, y, linestyle='--')

plt.xlabel('Mixing energy per atom')
plt.ylabel('Microstate probability')
plt.xlim(*lims)
plt.show()
exit()

R = 8.31446
def es(x, mean, sd, scale, T):
    return norm.pdf(x,mean,sd)*scale*x*np.exp(-x/(R*T))

def ps(x, mean, sd, scale, T):
    return norm.pdf(x,mean,sd)*scale*np.exp(-x/(R*T))

T = 1.
temperatures = np.linspace(0.1, 20., 1001)
energies = np.empty_like(temperatures)
f=4. # fudge factor
for i, T in enumerate(temperatures):
    energies[i] = (quad(es, -4.*32., -3.*32., args=(-112., 2., 1., T/f))[0]
                   / quad(ps, -4.*32., -3.*32., args=(-112., 2., 1., T/f))[0])
#plt.plot(temperatures, energies/32.)

entropies = cumtrapz(np.gradient(energies, temperatures)/temperatures, temperatures, initial=0)
plt.scatter(temperatures, f*(entropies-entropies[-1])/R/32. - 2./4.*np.log(0.5))

plt.xlim(0,2)
plt.ylim(-0.5, 0.7)
plt.show()
