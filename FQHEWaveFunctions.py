from numba import njit, vectorize, prange
import numpy as np

"""
@njit(parallel=True)
def ThetaFunction(z: np.complex128, t: np.complex128, a: np.float64,
                  b: np.float64, n_max: np.uint32 = 1000
                  ) -> np.complex128:
    index = np.arange(-n_max, n_max, 1)
    terms = np.exp(1j*np.pi*t*((index + a)**2) +
                   1j*2*np.pi*(index + a)*(z + b))
    return np.sum(terms)
"""


@njit(parallel=True)
def ThetaFunction(z: np.complex128, t: np.complex128, a: np.float64,
                  b: np.float64, n_max: np.uint32 = 1000
                  ) -> np.complex128:
    theta = 0
    for i in prange(2*n_max):
        theta += np.exp(1j*np.pi*t*((i-n_max + a)**2) +
                        1j*2*np.pi*(i-n_max + a)*(z + b))
    return theta


@vectorize
def ThetaFunctionVectorized(z, t: np.complex128, a: np.float64,
                            b: np.float64):
    return ThetaFunction(z, t, a, b)


@njit
def LLLSymmetricGauge(z: np.complex128, Ns: np.uint8, t: np.complex128, n: np.uint8
                      ) -> np.complex128:
    """Returns the (unnormalised) n-th LLL wavefunction for the torus,
    sampled at point z.

    Parameters:
    Ns : number of flux quanta
    t : torus complex aspect ratio
    n : wavefunction index
    l : number of grid points along one axis"""
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    k_n = (2*n-Ns)*np.pi/Lx
    w_n = (1+np.arange(Ns))/Ns - (np.pi*(1+2*Ns) + k_n*Lx*t)/(2*np.pi*Ns)

    psi = np.exp(1j*k_n*z + (z**2 - np.abs(z)**2)/4)
    psi *= np.prod(ThetaFunctionVectorized(z/Lx - w_n, t, 1/2, 1/2))

    return psi


@njit
def LLLSymmetricGaugeGrid(Ns: np.uint8, t: np.complex128, n: np.uint8,
                          l: np.uint16) -> np.array:
    """Returns the (unnormalised) n-th LLL wavefunction for the torus,
    sampled on an (l x l) grid.

    Parameters:
    Ns : number of flux quanta
    t : torus complex aspect ratio
    n : wavefunction index
    l : number of grid points along one axis"""
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    psi = np.ones((l, l), dtype=np.complex128)
    k_n = (2*n-Ns)*np.pi/Lx
    w_n = (1+np.arange(Ns))/Ns - (np.pi*(1+2*Ns) + k_n*Lx*t)/(2*np.pi*Ns)

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
def LaughlinTorus(N: np.uint8, Ns: np.uint16, tau: np.complex128,
                  R: np.array, kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                  phi_tau: np.float64 = 0) -> np.complex128:
    # if we translate once, then only the reduced coordinates appear in the exponential
    # then I apply the magnetic translation, phases come out from the exponential and then
    # just run theta functions as is.
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(tau))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_tau/(2*np.pi) + m*(N-1)/2

    w = np.exp((np.sum(R**2) - np.sum(np.abs(R)**2))/4)
    w *= ThetaFunction(m*np.sum(R)/Lx, m*tau, aCM, bCM)
    for i in range(N):
        for j in range(i+1, N):
            w *= ThetaFunction((R[i]-R[j])/Lx, tau, 1/2, 1/2)**m

    return w


@njit(parallel=True)
def LaughlinTorusPhase(N: np.uint8, Ns: np.uint16, tau: np.complex128,
                       R: np.array, kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                       phi_tau: np.float64 = 0) -> np.complex128:
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(tau))
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
def LaughlinTorusReduced(N: np.uint8, Ns: np.uint16, tau: np.complex128,
                         R: np.array, p: np.uint8, kCM: np.uint8 = 0,
                         phi_1: np.float64 = 0, phi_tau: np.float64 = 0
                         ) -> np.complex128:
    # if we translate once, then only the reduced coordinates appear in the exponential
    # then I apply the magnetic translation, phases come out from the exponential and then
    # just run theta functions as is.
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(tau))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_tau/(2*np.pi) + m*(N-1)/2

    w = np.exp((R[p]**2 - np.abs(R[p])**2)/4)
    w *= ThetaFunction(m*np.sum(R)/Lx, m*tau, aCM, bCM)
    for i in prange(N):
        if i != p:
            w *= ThetaFunction((R[i]-R[p])/Lx, tau, 1/2, 1/2)**m

    return w
