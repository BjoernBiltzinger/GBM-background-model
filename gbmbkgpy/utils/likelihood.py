import numba
import math

@numba.njit(
    numba.float64(numba.float64[:, :], numba.int64[:, :]),
    parallel=False,
    fastmath=True,
)
def cstat_numba(M, counts):
    # Poisson loglikelihood statistic (Cash) is:
    # L = Sum ( M_i - D_i * log(M_i))
    val = 0.0
    for i in numba.prange(M.shape[0]):
        for j in numba.prange(M.shape[1]):
            #for k in range(M.shape[2]):
            val += M[i, j] - counts[i, j] * math.log(M[i, j])
    return val
