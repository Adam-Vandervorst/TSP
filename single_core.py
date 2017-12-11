"""single-core brute force TSP"""

from time import time
from itertools import permutations

import numpy as np
from numba import vectorize, int32

from dataset import n, matrix


@vectorize([int32(int32, int32, int32, int32, int32, int32,
                  int32, int32, int32, int32, int32, int32)], nopython=True)
def dist12(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11):
    return matrix[i11, i0] + matrix[i0, i1] + matrix[i1, i2]\
           + matrix[i2, i3] + matrix[i3, i4] + matrix[i4, i5]\
           + matrix[i5, i6] + matrix[i6, i7] + matrix[i7, i8]\
           + matrix[i8, i9] + matrix[i9, i10] + matrix[i10, i11]


data = iter(permutations(range(n)))
results = {}

t = time()
for batch_number in range(1600):
    batch = np.array([next(data) for i in range(299376)], dtype=np.int32).T
    distances = dist12(*batch)
    batch_min = np.min(distances)
    results[batch_number] = (batch_min, )
    del batch
    del distances
    if batch_number % 10 == 0:
        print(batch_number/16, '%', sep='')

print("Time taken:", (time()-t))

print(results)
print(min(results.values()))