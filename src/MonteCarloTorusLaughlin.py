from numba import njit, vectorize, prange
import numpy as np


@njit  # (parallel=True)
def ThetaFunction(z: np.complex128, t: np.complex128, a: np.float64,
                  b: np.float64, n_max: np.uint32 = 60
                  ) -> np.complex128:
    index_a = np.arange(-n_max+a, n_max+a, 1)
    terms = np.exp(1j*np.pi*index_a*(t*(index_a) + 2*(z + b)))
    return np.sum(terms)


"""
@njit(parallel=True)
def ThetaFunction(z: np.complex128, t: np.complex128, a: np.float64,
                  b: np.float64, n_max: np.uint32 = 1000
                  ) -> np.complex128:
    theta = 0
    for i in prange(2*n_max):
        theta += np.exp(1j*np.pi*t*((i-n_max + a)*(i-n_max + a)) +
                        1j*2*np.pi*(i-n_max + a)*(z + b))
    return theta
"""


@vectorize
def ThetaFunctionVectorized(z, t: np.complex128, a: np.float64,
                            b: np.float64):
    return ThetaFunction(z, t, a, b)


@njit
def ReduceNonholomorphic(coords: np.array,
                         ) -> np.complex128:
    """Given the coordinates of the particles that move,
    returns the nonholomorphic Gaussian exponent in the wfn.

    Parameters:
    coords : array with updated coordiantes

    Output:
    expo_non_holomorphic : nonholomorphic Gaussian exponent"""
    expo_nonholomorphic = 0
    for j in range(coords.size):
        expo_nonholomorphic += (coords[j]**2 - np.abs(coords[j])**2)/4

    return expo_nonholomorphic


@njit
def ReduceCM(N: np.uint16, S: np.uint16, t: np.complex128,
             zCM: np.complex128, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0
             ) -> (np.complex128, np.complex128):
    """Using the properties of the theta function, splits the CM contribution
    to the wavefunction into a a theta function and exponential contribution."""
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_t/(2*np.pi) + m*(N-1)/2

    zCM *= m/Lx
    c = np.imag(zCM)//np.imag(m*t)

    expo_CM = -1j*np.pi*c*(2*zCM - c*m*t + 2*bCM)

    r = ThetaFunction(zCM - m*t*c, m*t, aCM + c, bCM)

    return r, expo_CM


@njit(parallel=True)
def UpdateJastrowsLaughlin(jastrows: np.array, coords: np.array, t: np.complex128,
                           Lx: np.float64, moved_particle: np.uint16):

    N = coords.size
    for i in prange(N):
        if i != moved_particle:
            jastrows[i, moved_particle] = ThetaFunction(
                (coords[i] - coords[moved_particle])/Lx, t, 1/2, 1/2)
            jastrows[moved_particle, i] = - jastrows[i, moved_particle]


@njit
def InitialJastrowsLaughlin(coords: np.array,
                            t: np.complex128, Lx: np.float64):

    N = coords.size
    jastrows = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(i+1, N):
            jastrows[i, j] = ThetaFunction(
                (coords[i] - coords[j])/Lx, t, 1/2, 1/2)
            jastrows[j, i] = - jastrows[i, j]
    for i in range(N):
        jastrows[i, i] = 1

    return jastrows


@njit(parallel=True)
def StepOneAmplitudeLaughlinOld(S: np.uint16, t: np.complex128,
                                coords_initial: np.array, coords_final: np.array,
                                p: np.uint8, kCM: np.uint8 = 0,
                                phi_1: np.float64 = 0, phi_t: np.float64 = 0
                                ) -> np.complex128:
    """
    Returns the ratio of initial and final wavefunctions, given
    that the particle with index p has moved.

    Parameters:
    N : number of particles
    S : number of flux quanta
    t : torus complex aspect ratio
    coords_initial : initial configuration of particles
    coords_final : final configuration of particles
    p : index of particle that moves

    Output:
    r : ratio of wavefunctions R_f/R_i
    """

    N = coords_initial.size
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(t))

    (r_i, expo_CM_i) = ReduceCM(N, S, t, np.sum(coords_initial),
                                kCM, phi_1, phi_t)
    (r_f, expo_CM_f) = ReduceCM(N, S, t, np.sum(coords_final),
                                kCM, phi_1, phi_t)
    r = r_f/r_i
    expo_nonholomorphic_i = ReduceNonholomorphic(np.array([coords_initial[p]]))
    expo_nonholomorphic_f = ReduceNonholomorphic(np.array([coords_final[p]]))
    expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

    for i in prange(N):
        if i != p:
            ratio = (ThetaFunction((coords_final[i]-coords_final[p])/Lx,
                                   t, 1/2, 1/2) /
                     ThetaFunction((coords_initial[i]-coords_initial[p])/Lx,
                                   t, 1/2, 1/2))**m
            r *= ratio
        r *= np.exp(expo/N)

    return r


@njit
def StepOneAmplitudeLaughlin(S: np.uint16, t: np.complex128,
                             coords_current: np.array, coords_new: np.array,
                             jastrows_current: np.array, jastrows_new: np.array,
                             moved_particle: np.uint8, kCM: np.uint8 = 0,
                             phi_1: np.float64 = 0, phi_t: np.float64 = 0
                             ) -> np.complex128:
    """
    Returns the ratio of initial and final wavefunctions, given
    that the particle with index p has moved.

    Parameters:
    N : number of particles
    S : number of flux quanta
    t : torus complex aspect ratio
    coords_current : current configuration of particles
    coords_new : new configuration of particles
    p : index of particle that moves

    Output:
    r : ratio of new/current wavefunctions
    """

    N = coords_current.size
    m = S/N

    (r_i, expo_CM_i) = ReduceCM(N, S, t, np.sum(coords_current),
                                kCM, phi_1, phi_t)
    (r_f, expo_CM_f) = ReduceCM(N, S, t, np.sum(coords_new),
                                kCM, phi_1, phi_t)
    r = r_f/r_i
    expo_nonholomorphic_i = ReduceNonholomorphic(
        np.array([coords_current[moved_particle]]))
    expo_nonholomorphic_f = ReduceNonholomorphic(
        np.array([coords_new[moved_particle]]))
    expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

    jastrows_relative = (jastrows_new[moved_particle, :] /
                         jastrows_current[moved_particle, :])

    return r*np.prod(np.power(jastrows_relative, m) * np.exp(expo/N))


@njit(parallel=True)
def StepOneAmplitudeLaughlinSWAP(S: np.uint16, t: np.complex128,
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
        return (StepOneAmplitudeLaughlin(S, t, swap_R_i[:, 0], swap_R_f[:, 0], p_swap[0], kCM, phi_1, phi_t) *
                StepOneAmplitudeLaughlin(S, t, swap_R_i[:, 1], swap_R_f[:, 1], p_swap[1], kCM, phi_1, phi_t))

    else:
        if p_swap_order[0] > 0:
            p_swap = p_swap_order-1
            copy = 1
        else:
            p_swap = np.abs(p_swap_order)-1
            copy = 0

        N = swap_R_i.shape[0]
        m = S/N
        Lx = np.sqrt(2*np.pi*S/np.imag(t))

        diff_swap_R_f = swap_R_f[:, copy]
        diff_swap_R_i = swap_R_i[:, copy]

        (r_i, expo_CM_i) = ReduceCM(N, S, t,
                                    np.sum(diff_swap_R_i), kCM, phi_1, phi_t)
        (r_f, expo_CM_f) = ReduceCM(N, S, t,
                                    np.sum(diff_swap_R_f), kCM, phi_1, phi_t)
        r = r_f / r_i
        expo_nonholomorphic_i = ReduceNonholomorphic(diff_swap_R_i[p_swap])
        expo_nonholomorphic_f = ReduceNonholomorphic(diff_swap_R_f[p_swap])
        expo = expo_CM_f - expo_CM_i + expo_nonholomorphic_f - expo_nonholomorphic_i

        for j in prange(N):
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
def InitialModLaughlin(S: np.uint16, t: np.complex128, R: np.array,
                       swap_R: np.array, kCM: np.uint8 = 0,
                       phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                       ) -> np.float64:
    """
    """
    N = R.shape[0]
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(t))

    (mod_swap_0, expo_swap_0) = ReduceCM(
        N, S, t, np.sum(swap_R[:, 0]), kCM, phi_1, phi_t)
    (mod_swap_1, expo_swap_1) = ReduceCM(
        N, S, t, np.sum(swap_R[:, 1]), kCM, phi_1, phi_t)
    (mod_0, expo_0) = ReduceCM(N, S, t, np.sum(R[:, 0]), kCM, phi_1, phi_t)
    (mod_1, expo_1) = ReduceCM(N, S, t, np.sum(R[:, 1]), kCM, phi_1, phi_t)
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
def InitialSignLaughlin(S: np.uint16, t: np.complex128, R: np.array,
                        swap_R: np.array, kCM: np.uint8 = 0,
                        phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                        ) -> np.float64:
    """
    """
    N = R.shape[0]
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(t))

    (sign_swap_0, expo_swap_0) = ReduceCM(
        N, S, t, np.sum(swap_R[:, 0]), kCM, phi_1, phi_t)
    sign_swap_0 /= np.abs(sign_swap_0)
    (sign_swap_1, expo_swap_1) = ReduceCM(
        N, S, t, np.sum(swap_R[:, 1]), kCM, phi_1, phi_t)
    sign_swap_1 /= np.abs(sign_swap_1)
    (sign_0, expo_0) = ReduceCM(N, S, t, np.sum(R[:, 0]), kCM, phi_1, phi_t)
    sign_0 /= np.abs(sign_0)
    (sign_1, expo_1) = ReduceCM(N, S, t, np.sum(R[:, 1]), kCM, phi_1, phi_t)
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


@njit
def LLLSymmetricGauge(z: np.complex128, S: np.uint8, t: np.complex128, n: np.uint8
                      ) -> np.complex128:
    """Returns the (unnormalised) n-th LLL wavefunction for the torus,
    sampled at point z.

    Parameters:
    S : number of flux quanta
    t : torus complex aspect ratio
    n : wavefunction index
    l : number of grid points along one axis"""
    Lx = np.sqrt(2*np.pi*S/np.imag(t))
    Ly = Lx*np.imag(t)
    k_n = (2*n-S)*np.pi/Lx
    w_n = (1+np.arange(S))/S - (np.pi*(1+2*S) + k_n*Lx*t)/(2*np.pi*S)

    psi = np.exp(1j*k_n*z + (z**2 - np.abs(z)**2)/4)
    psi *= np.prod(ThetaFunctionVectorized(z/Lx - w_n, t, 1/2, 1/2))

    return psi


@njit
def LLLSymmetricGaugeGrid(S: np.uint8, t: np.complex128, n: np.uint8,
                          l: np.uint16) -> np.array:
    """Returns the (unnormalised) n-th LLL wavefunction for the torus,
    sampled on an (l x l) grid.

    Parameters:
    S : number of flux quanta
    t : torus complex aspect ratio
    n : wavefunction index
    l : number of grid points along one axis"""
    Lx = np.sqrt(2*np.pi*S/np.imag(t))
    Ly = Lx*np.imag(t)
    psi = np.ones((l, l), dtype=np.complex128)
    k_n = (2*n-S)*np.pi/Lx
    w_n = (1+np.arange(S))/S - (np.pi*(1+2*S) + k_n*Lx*t)/(2*np.pi*S)

    x = np.linspace(-Lx/2, Lx/2, l)
    y = np.linspace(-Ly/2, Ly/2, l)
    for i in prange(x.size):
        for j in prange(y.size):
            z = x[i] + 1j*y[j]
            psi[i, j] *= np.exp(1j*k_n*z + (z**2 - np.abs(z)**2)/4)
            psi[i, j] *= np.prod(ThetaFunctionVectorized(z /
                                 Lx - w_n, t, 1/2, 1/2))

    return psi


@njit(parallel=True)
def LaughlinTorus(N: np.uint8, S: np.uint16, tau: np.complex128,
                  R: np.array, kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                  phi_tau: np.float64 = 0) -> np.complex128:
    # if we translate once, then only the reduced coordinates appear in the exponential
    # then I apply the magnetic translation, phases come out from the exponential and then
    # just run theta functions as is.
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(tau))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_tau/(2*np.pi) + m*(N-1)/2

    w = np.exp((np.sum(R**2) - np.sum(np.abs(R)**2))/4)
    w *= ThetaFunction(m*np.sum(R)/Lx, m*tau, aCM, bCM)
    for i in range(N):
        for j in range(i+1, N):
            w *= ThetaFunction((R[i]-R[j])/Lx, tau, 1/2, 1/2)**m

    return w


@njit(parallel=True)
def LaughlinTorusPhase(N: np.uint8, S: np.uint16, tau: np.complex128,
                       R: np.array, kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                       phi_tau: np.float64 = 0) -> np.complex128:
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(tau))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_tau/(2*np.pi) + m*(N-1)/2

    w = np.exp(1j*np.sum(np.real(R)*np.imag(R))/2)
    w *= np.exp(1j*np.angle(ThetaFunction(m*np.sum(R)/Lx, m*tau, aCM, bCM)))
    for i in range(N):
        for j in range(i+1, N):
            w *= np.exp(1j *
                        np.angle(ThetaFunction((R[i]-R[j])/Lx, tau, 1/2, 1/2)**m))

    return w


@njit(parallel=True)
def LaughlinTorusReduced(N: np.uint8, S: np.uint16, tau: np.complex128,
                         R: np.array, p: np.uint8, kCM: np.uint8 = 0,
                         phi_1: np.float64 = 0, phi_tau: np.float64 = 0
                         ) -> np.complex128:
    # if we translate once, then only the reduced coordinates appear in the exponential
    # then I apply the magnetic translation, phases come out from the exponential and then
    # just run theta functions as is.
    m = S/N
    Lx = np.sqrt(2*np.pi*S/np.imag(tau))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_tau/(2*np.pi) + m*(N-1)/2

    w = np.exp((R[p]**2 - np.abs(R[p])**2)/4)
    w *= ThetaFunction(m*np.sum(R)/Lx, m*tau, aCM, bCM)
    for i in prange(N):
        if i != p:
            w *= ThetaFunction((R[i]-R[p])/Lx, tau, 1/2, 1/2)**m

    return w
