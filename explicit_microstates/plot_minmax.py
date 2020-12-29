import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.5, 0.625, 0.75, 0.875, 1.])
mean1 = [-96, -98, -104, -114, -128]
mean2 = [-96, -90, -72, -42, 0]
min1 = [-128, -112-8, -88-8, -48-8, 0]
min2 = [-128, -128, -128, -128, -128]

a = [5.28, 5.06, 4.16, 2.56, 0.]


def quad(x, a, b, c):
    return a*x + b*(1 - x) + c*x*(1. - x)


xs = np.linspace(0.5, 1., 1001)

plt.scatter(x, a)
plt.plot(xs, quad(xs, 0., 0., 5.28*4.))
plt.show()

plt.scatter(x, mean1, label='mean1')
plt.scatter(x, mean2, label='mean2')
plt.scatter(x, min1, label='min1')
plt.scatter(x, min2, label='min2')
plt.plot(xs, quad(xs, -128, -128, 128.))
plt.plot(xs, quad(xs, -128, -128, 0.))
plt.plot(xs, quad(xs, 0., 0., -3.*128.))
plt.plot(xs, quad(xs, 0., 0., -4.*128.))
plt.show()
