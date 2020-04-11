import numpy as np

# Al, Si, Al, Si etc. on sites alpha, beta1, beta2, beta3
cluster_occupancies = np.array([[1, 0, 0, 1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [0, 1, 1, 0, 0, 1, 0, 1],
                                [0, 1, 0, 1, 1, 0, 0, 1],
                                [0, 1, 0, 1, 0, 1, 1, 0],
                                [1, 0, 1, 0, 0, 1, 0, 1],
                                [1, 0, 0, 1, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1, 0, 0, 1],
                                [0, 1, 1, 0, 0, 1, 1, 0],
                                [0, 1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 0, 1],
                                [1, 0, 1, 0, 0, 1, 1, 0],
                                [1, 0, 0, 1, 1, 0, 1, 0],
                                [0, 1, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0]])

 # energies associated with change from preferred atoms (Al on alpha, Si on beta)
Wmbr = np.array([0, 1/4, 1/4, 0, 1/4, 0, 1/4, 0])

 # Al-Al nearest neighbours
Wb = np.zeros((8, 8))
Wb[0,2] = 1.
Wb[0,6] = 1.
Wb[4,2] = 1.
Wb[4,6] = 1.

f_e1 = np.einsum('ij, j -> i', cluster_occupancies, Wmbr)
f_e2 = np.einsum('ij, jk, ki -> i', cluster_occupancies, Wb, cluster_occupancies.T)

print(f_e1)
print(f_e2)
