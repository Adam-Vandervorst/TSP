"""Cython multiprocessing test"""
from concurrent.futures import ProcessPoolExecutor
import cython
from cython_gsl cimport *

@cython.cdivision(True)
def mc_pi_cython(int n):
    cdef gsl_rng_type * T
    cdef gsl_rng * r
    cdef double s = 0.0
    cdef double x, y
    cdef int i

    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    for i in range(n):
        x = 2*gsl_rng_uniform(r) - 1
        y = 2*gsl_rng_uniform(r)- 1
        if (x**2 + y**2) < 1:
            s += 1
    return 4*s/n

with ProcessPoolExecutor(max_workers=8) as pool:
    res = pool.map(mc_pi_cython, [int(1e4) for i in range(int(1e4))])
print(res)