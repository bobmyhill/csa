import numpy as np

# [A, B][C, D] A>C>D>B

# Clusters [AC, BD, AD]

# In the ground state, if fully connected, the clusters are NOT reciprocal
# (i.e. there are no exchange reactions possible

# p(AC) = p(C)
# p(BD) = p(B)
# p(AD) = 1 - p(C) - p(B) = p(A) + p(D) - 1

ps = [0.7, 0.3, 0.6, 0.4]

ps = [0.9, 0.1, 0.8, 0.2]

R = 8.31446
S_s = -R*np.sum([p*np.log(p) for p in ps])
S_c = -R*np.sum([ps[3]*np.log(ps[3]),
                 ps[1]*np.log(ps[1]),
                 (ps[0] + ps[2] - 1.)*np.log(ps[0] + ps[2] - 1.)])

# print((ps[0] + ps[3] - 1.), (1. - ps[1] - ps[2]))
print(S_s, S_c)

gamma = 1.
n_sites = 4.
print(gamma*S_c + (1./n_sites - gamma)*S_s)

ps2 = [0.89558667, 0.10441333, 0.00441333, 0.99558667,
       0.79618364, 0.20381636, 0.00381636, 0.99618364]

print(-R*np.sum([p*np.log(p) for p in ps2]))

# Extreme case of order-disorder (convex, [A,B][C,D][E,F][G,H])
# clusters = AlAlAlAl, MgSiMgSi

ps = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
pcls = [0.6, 0.4]

S_s = -R*np.sum([p*np.log(p) for p in ps])
S_c = -R*np.sum([0.5*np.log(0.5), 0.5*np.log(0.5)])
print(S_s, S_c)
gamma = 1.
n_sites = 4.
print(gamma*S_c + (1./n_sites - gamma)*S_s)


# Extreme case of order-disorder (non convex)
# [AlMg][AlSi][AlMg][AlSi]
# clusters = AlAlAlAl, MgSiMgSi


S_s = -R*np.sum([0.5*np.log(0.5) for i in range(8)])
S_c = -R*np.sum([0.5*np.log(0.5), 0.5*np.log(0.5)])
print(S_s, S_c)
gamma = 1.
n_sites = 4.
print(gamma*S_c + (1./n_sites - gamma)*S_s)
