import cupy as cp
from scipy.lina

cpdef iddp_id()

cpdef iddr_aid(A, int k, rng):
    cdef int rows = A.shape[0], cols = A.shape[1]

    rmat = rng.random((cols, k))

    y = cp.matmul(A, rmat)







    



