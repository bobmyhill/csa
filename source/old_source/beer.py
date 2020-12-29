import numpy as np


glasses = np.linspace(1, 100000, 100000)

can_volume = 500.
glass_volume = 568.291

f_can = glasses*glass_volume/can_volume

plus_cans = (f_can + 0.5).round()
sub_cans = (f_can - 0.5).round()

v_plus = glasses*glass_volume - plus_cans*can_volume
v_sub = glasses*glass_volume - sub_cans*can_volume

min_plus = -600.
min_glasses = 0
for i, v in enumerate(v_plus):
    if v > min_plus:
        min_plus = v
        min_glasses = i+1
        print(min_glasses, v)
#print(min_glasses, min_plus)
print()

max_sub = 600.
max_glasses = 0
for i, v in enumerate(v_sub):
    if v < max_sub:
        max_sub = v
        max_glasses = i+1
        print(max_glasses, v)

#print(max_glasses*glass_volume/can_volume)
#print(max_glasses*glass_volume/can_volume)
