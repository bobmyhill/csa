import numpy as np
import matplotlib.pyplot as plt

def E(gs, Es, RT):
    exps = np.exp(-Es/RT)
    pfn = np.einsum('i, i', gs, exps)
    return np.einsum('i, i, i', gs, Es, exps)/pfn

gi = np.array([1., 1.])
Ei = np.array([0., 2.])

RTs = np.linspace(0.1, 2., 101)
Es = np.empty_like(RTs)
for i, RT in enumerate(RTs):
    Es[i] = E(gi, Ei, RT)

plt.plot(RTs, Es, label='two step')

gi = np.array([1., 1., 1.])
Ei = np.array([0., 1., 2.])

Es = np.empty_like(RTs)
for i, RT in enumerate(RTs):
    Es[i] = E(gi, Ei, RT)

plt.plot(RTs, Es, label='three step')


gi = np.array([1., 3., 1.])
Ei = np.array([0., 1., 2.])

Es = np.empty_like(RTs)
for i, RT in enumerate(RTs):
    Es[i] = E(gi, Ei, RT)

plt.plot(RTs, Es, label='three step degen')
plt.legend()
plt.show()
