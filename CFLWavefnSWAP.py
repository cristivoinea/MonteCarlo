import numpy as np
from numba import njit, prange
from LaughlinWavefnSWAP import ReduceCM, ReduceNonholomorphic, ThetaFunction
from FreeFermionsWavefnSWAP import Kxs, Kys


@njit(parallel=True)
def StepOneAmplitudeCFL(Ns: np.uint16, t: np.complex128,
                        R_i: np.array, R_f: np.array, p: np.uint8,
                        kCM: np.uint8 = 0,
                        phi_1: np.float64 = 0, phi_t: np.float64 = 0
                        ) -> np.complex128:
    """
    Returns the ratio of wavefunctions for coordinates R_i
    to coordinates R_f, given that the particle with index p has moved.

    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    R_i : initial configuration of particles
    R_f : final configuration of particles
    p : indice of particle that moves

    Output:

    r : ratio of wavefunctions R_f/R_i
    """

    Ne = R_i.size
    m = Ns/Ne
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))

    (r_i, expo_CM_i) = ReduceCM(Ne, Ns, t, R_i, kCM, phi_1, phi_t)
    (r_f, expo_CM_f) = ReduceCM(Ne, Ns, t, R_f, kCM, phi_1, phi_t)
    amplitude = r_f/r_i
    expo_nonholomorphic_i = ReduceNonholomorphic(R_i, np.array([p]))
    expo_nonholomorphic_f = ReduceNonholomorphic(R_f, np.array([p]))
    expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

    for i in prange(Ne):
        if i != p:
            r *= (ThetaFunction((R_f[i]-R_f[p])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i]-R_i[p])/Lx, t, 1/2, 1/2))**m
        amplitude *= np.exp(expo/Ne)

    """
    jastrow_terms = np.ones(N, dtype=np.complex128)
    for i in prange(N):
        if i != p:
            jastrow_terms[i] = (ThetaFunction((R_f[i]-R_f[p])/Lx, t, 1/2, 1/2) /
                                ThetaFunction((R_i[i]-R_i[p])/Lx, t, 1/2, 1/2))**m
    for i in range(N):
        r *= jastrow_terms[i]
        r *= np.exp(expo/N)
    """

    return amplitude
