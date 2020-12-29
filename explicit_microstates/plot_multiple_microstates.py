import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import quad, cumtrapz
import sys


def gauss(x, mean, sd, scale):
    return norm.pdf(x, mean, sd)*scale


n_files = int((len(sys.argv)-1)/2)

inn = []
for i in range(n_files):
    inn.append((sys.argv[2*i+1], float(sys.argv[2*i+2])))

for (infile, n_atoms) in inn:
    energy_levels_and_degeneracies = np.loadtxt(infile)

    energy_levels_and_degeneracies[:, 0] /= -n_atoms

    s = np.sum(energy_levels_and_degeneracies[:, 1])
    fs = energy_levels_and_degeneracies[:, 1]/s
    plt.scatter(energy_levels_and_degeneracies[:, 0], fs)

    mean_guess = energy_levels_and_degeneracies[np.argmax(fs), 0]

    de = 0.1
    lims = np.array([np.min(energy_levels_and_degeneracies[:, 0]),
                     np.max(energy_levels_and_degeneracies[:, 0])])
    spread = np.max(np.abs(mean_guess - lims))
    slims = [mean_guess-spread-de, mean_guess+spread+de]

    popt, pcov = curve_fit(gauss, energy_levels_and_degeneracies[:, 0],
                           fs, p0=[mean_guess, spread/2., 0.1])
    print(infile, lims*32, popt*32)

    minmax = [min(energy_levels_and_degeneracies[:, 0]),
              max(energy_levels_and_degeneracies[:, 0])]
    x = np.arange(minmax[0], minmax[1], 0.001)  # range of x in spec
    y = gauss(x, *popt)
    plt.plot(x, y, linewidth=3, label=infile)

    # y = gauss(x, 80, 2.0, 3.84)
    # plt.plot(-x/n_atoms, y, linestyle='--')

    plt.xlabel('Mixing energy per atom')
    plt.ylabel('Microstate probability')
    # plt.xlim(*slims)
plt.legend()
plt.show()
