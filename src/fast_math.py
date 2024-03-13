from numba import njit, vectorize, int64, float64, complex128
import numpy as np


@njit
def combs(n: np.int64, k: np.int64) -> np.int64:
    return np.prod(np.arange(n+1-k, n+1)/np.arange(1, k+1))


@vectorize([int64(int64, int64)])
def combs_vect(n: np.uint16, k: np.int16) -> np.uint64:
    return combs(n, k)


@njit
def _MonopoleHarmonics(s: np.int64, l: np.int64, m: np.int64, theta: np.float64, phi: np.float64) -> np.complex128:
    norm = np.sqrt((2*l+s+1)*combs(2*l+s, l)/combs(2*l+s, m)/(4*np.pi))
    v_mod = np.sin(theta/2)
    u_mod = np.cos(theta/2)
    non_h = l*theta*0
    for k in range(2*l+s-m+1):
        non_h += (((-1)**k)*combs(l, k)*combs(l+s, 2*l+s-m-k) *
                  np.power(v_mod, 2*(l-k))*np.power(u_mod, 2*k))
    return norm * non_h * (((-1)**(l+s)) *
                           np.power(v_mod, l+s-m) *
                           np.power(u_mod, m-l)) * np.exp(1j*((m-l)*phi - s*phi/2))


@vectorize([complex128(int64, int64, int64, float64, float64)])
def MonopoleHarmonics(s: np.int64, l: np.int64, m: np.int64, theta: np.float64, phi: np.float64) -> np.array:
    return _MonopoleHarmonics(s, l, m, theta, phi)


@njit
def Variance(array: np.array, ddof: np.int64 = 0):
    mean = np.sum(array)/array.size
    return np.sum((array - mean)**2)/(array.size-ddof)


@njit
def JackknifeVariance(results: np.array, nbr_blocks: np.int64 = 100):
    mean = Variance(results, ddof=1)
    block_len = results.size//nbr_blocks
    var = 0
    for i in range(nbr_blocks):
        rm_block = np.hstack(
            (results[:block_len*i], results[block_len*(i+1):]))
        var += (Variance(rm_block, ddof=1)-mean)**2

    return mean, np.sqrt(var*(nbr_blocks-1)/nbr_blocks)


@njit
def JackknifeMean(results: np.array, nbr_blocks: np.int64 = 1000):
    mean = np.sum(results)/results.size
    block_len = results.size//nbr_blocks
    var = 0
    for i in range(nbr_blocks):
        rm_block = np.hstack(
            (results[:block_len*i], results[block_len*(i+1):]))
        var += (np.sum(rm_block)/rm_block.size-mean)**2

    return mean, np.sqrt(var*(nbr_blocks-1)/nbr_blocks)
