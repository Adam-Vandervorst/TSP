"""multi-core brute force TSP"""

from time import time
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import vectorize, int32

from dataset import n, matrix

batches = 10
cores = 8

data = iter(permutations(range(n)))
results = [0]*batches


def dist12_nb(arr):
    @vectorize([int32(int32, int32, int32, int32, int32, int32,
                      int32, int32, int32, int32, int32, int32)], nopython=True)
    def dist12(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11):
        return matrix[i11, i0] + matrix[i0, i1] + matrix[i1, i2]\
               + matrix[i2, i3] + matrix[i3, i4] + matrix[i4, i5]\
               + matrix[i5, i6] + matrix[i6, i7] + matrix[i7, i8]\
               + matrix[i8, i9] + matrix[i9, i10] + matrix[i10, i11]
    return dist12(*arr.T)


with ThreadPoolExecutor(max_workers=cores) as executor:
    t = time()
    for batch_number in range(batches):
        batch = [0]*cores
        for core in range(cores):
            batch[core] = np.array([next(data) for i in range(299376)], dtype=np.int32)

        results[batch_number] = np.min(list(executor.map(dist12_nb, batch)))
    print((time()-t)/(batches*cores))
print(np.min(results))


