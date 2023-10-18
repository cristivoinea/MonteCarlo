from numba import njit, prange
import numpy as np
from FQHEWaveFunctions import ThetaFunction


@njit
def ReduceNonholomorphic(R: np.array, p: np.array
                         ) -> np.complex128:
    expo_nonholomorphic = 0
    for j in range(p.size):
        expo_nonholomorphic += (R[p[j]]**2 - np.abs(R[p[j]])**2)/4

    return expo_nonholomorphic


@njit
def ReduceCM(Ns: np.uint16, t: np.complex128,
             R: np.array, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0
             ) -> (np.complex128, np.complex128):
    """Using the properties of the theta function, splits the CM contribution
    to the wavefunction into a a theta function and exponential contribution."""
    Ne = R.size
    m = Ns/Ne
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (Ne-1)/2
    bCM = -phi_t/(2*np.pi) + m*(Ne-1)/2

    zCM = m*np.sum(R)/Lx
    c = np.imag(zCM)//np.imag(m*t)

    expo_CM = -1j*np.pi*c*(2*zCM - c*m*t + 2*bCM)

    r = ThetaFunction(zCM - m*t*c, m*t, aCM + c, bCM)

    return r, expo_CM


@njit(parallel=True)
def StepOneAmplitudeLaughlin(Ns: np.uint16, t: np.complex128,
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

    (r_i, expo_CM_i) = ReduceCM(Ns, t, R_i, kCM, phi_1, phi_t)
    (r_f, expo_CM_f) = ReduceCM(Ns, t, R_f, kCM, phi_1, phi_t)
    r = r_f/r_i
    expo_nonholomorphic_i = ReduceNonholomorphic(R_i, np.array([p]))
    expo_nonholomorphic_f = ReduceNonholomorphic(R_f, np.array([p]))
    expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

    for i in range(Ne):
        if i != p:
            r *= (ThetaFunction((R_f[i]-R_f[p])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i]-R_i[p])/Lx, t, 1/2, 1/2))**m
        r *= np.exp(expo/Ne)

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

    return r


@njit(parallel=True)
def StepOneAmplitudeLaughlinSWAP(Ns: np.uint16, t: np.complex128,
                                 swap_R_i: np.array, swap_R_f: np.array,
                                 p_swap_order: np.array,
                                 kCM: np.uint8 = 0,
                                 phi_1: np.float64 = 0, phi_t: np.float64 = 0):
    """
    """

    p_swap = np.zeros(2, dtype=np.uint8)
    if np.prod(p_swap_order) < 0:
        if p_swap_order[0] > 0:
            p_swap[1] = p_swap_order[0]-1
            p_swap[0] = np.abs(p_swap_order[1])-1
        else:
            p_swap[1] = p_swap_order[1]-1
            p_swap[0] = np.abs(p_swap_order[0])-1
        return (StepOneAmplitudeLaughlin(Ns, t, swap_R_i[:, 0], swap_R_f[:, 0], p_swap[0], kCM, phi_1, phi_t) *
                StepOneAmplitudeLaughlin(Ns, t, swap_R_i[:, 1], swap_R_f[:, 1], p_swap[1], kCM, phi_1, phi_t))

    else:
        if p_swap_order[0] > 0:
            p_swap = p_swap_order-1
            copy = 1
        else:
            p_swap = np.abs(p_swap_order)-1
            copy = 0

        N = swap_R_i.shape[0]
        m = Ns/N
        Lx = np.sqrt(2*np.pi*Ns/np.imag(t))

        diff_swap_R_f = swap_R_f[:, copy]
        diff_swap_R_i = swap_R_i[:, copy]

        (r_i, expo_CM_i) = ReduceCM(Ns, t, diff_swap_R_i, kCM, phi_1, phi_t)
        (r_f, expo_CM_f) = ReduceCM(Ns, t, diff_swap_R_f, kCM, phi_1, phi_t)
        r = r_f / r_i
        expo_nonholomorphic_i = ReduceNonholomorphic(diff_swap_R_i, p_swap)
        expo_nonholomorphic_f = ReduceNonholomorphic(diff_swap_R_f, p_swap)
        expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

        for j in range(N):
            if j != p_swap[0] and j != p_swap[1]:
                r *= ((ThetaFunction((diff_swap_R_f[j]-diff_swap_R_f[p_swap[0]])/Lx, t, 1/2, 1/2) *
                       ThetaFunction((diff_swap_R_f[j]-diff_swap_R_f[p_swap[1]])/Lx, t, 1/2, 1/2)) /
                      (ThetaFunction((diff_swap_R_i[j]-diff_swap_R_i[p_swap[0]])/Lx, t, 1/2, 1/2) *
                       ThetaFunction((diff_swap_R_i[j]-diff_swap_R_i[p_swap[1]])/Lx, t, 1/2, 1/2)))**m
            r *= np.exp(expo/N)

        r *= (ThetaFunction((diff_swap_R_f[p_swap[0]]-diff_swap_R_f[p_swap[1]])/Lx, t, 1/2, 1/2) /
              ThetaFunction((diff_swap_R_i[p_swap[0]]-diff_swap_R_i[p_swap[1]])/Lx, t, 1/2, 1/2))**m

        return r


@njit
def InitialModLaughlin(Ns: np.uint16, t: np.complex128, R: np.array,
                       swap_R: np.array, kCM: np.uint8 = 0,
                       phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                       ) -> np.float64:
    """
    """
    N = R.shape[0]
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))

    (mod_swap_0, expo_swap_0) = ReduceCM(
        Ns, t, swap_R[:, 0], kCM, phi_1, phi_t)
    (mod_swap_1, expo_swap_1) = ReduceCM(
        Ns, t, swap_R[:, 1], kCM, phi_1, phi_t)
    (mod_0, expo_0) = ReduceCM(Ns, t, R[:, 0], kCM, phi_1, phi_t)
    (mod_1, expo_1) = ReduceCM(Ns, t, R[:, 1], kCM, phi_1, phi_t)
    mod = mod_swap_0 * mod_swap_1 / (mod_0 * mod_1)
    expo = expo_swap_0 + expo_swap_1 - expo_0 - expo_1

    for i in range(N):
        for j in range(i+1, N):
            mod *= (ThetaFunction((swap_R[i, 0]-swap_R[j, 0])/Lx, t, 1/2, 1/2) /
                    ThetaFunction((R[i, 0]-R[j, 0])/Lx, t, 1/2, 1/2))**m
            mod *= (ThetaFunction((swap_R[i, 1]-swap_R[j, 1])/Lx, t, 1/2, 1/2) /
                    ThetaFunction((R[i, 1]-R[j, 1])/Lx, t, 1/2, 1/2))**m
        mod *= np.exp(expo/N)

    return np.abs(mod)


@njit
def InitialSignLaughlin(Ns: np.uint16, t: np.complex128, R: np.array,
                        swap_R: np.array, kCM: np.uint8 = 0,
                        phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                        ) -> np.float64:
    """
    """
    N = R.shape[0]
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))

    (sign_swap_0, expo_swap_0) = ReduceCM(
        Ns, t, swap_R[:, 0], kCM, phi_1, phi_t)
    sign_swap_0 /= np.abs(sign_swap_0)
    (sign_swap_1, expo_swap_1) = ReduceCM(
        Ns, t, swap_R[:, 1], kCM, phi_1, phi_t)
    sign_swap_1 /= np.abs(sign_swap_1)
    (sign_0, expo_0) = ReduceCM(Ns, t, R[:, 0], kCM, phi_1, phi_t)
    sign_0 /= np.abs(sign_0)
    (sign_1, expo_1) = ReduceCM(Ns, t, R[:, 1], kCM, phi_1, phi_t)
    sign_1 /= np.abs(sign_1)
    sign = np.conj(sign_swap_0 * sign_swap_1) * sign_0 * sign_1
    expo = 1j*np.imag(-expo_swap_0 - expo_swap_1 + expo_0 + expo_1)
    sign *= np.exp(expo)

    for i in range(N):
        for j in range(i+1, N):
            sign *= (np.conj(ThetaFunction((swap_R[i, 0]-swap_R[j, 0])/Lx, t, 1/2, 1/2) *
                             ThetaFunction((swap_R[i, 1]-swap_R[j, 1])/Lx, t, 1/2, 1/2)) *
                     (ThetaFunction((R[i, 0]-R[j, 0])/Lx, t, 1/2, 1/2) *
                     ThetaFunction((R[i, 1]-R[j, 1])/Lx, t, 1/2, 1/2)))**m
            sign /= np.abs(sign)

    return sign
