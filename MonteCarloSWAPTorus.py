from numba import njit, jit, prange
from numba.typed import List
import numpy as np
from FQHEWaveFunctions import ThetaFunction, ThetaFunctionVectorized, LaughlinTorus, LaughlinTorusReduced


@njit
def RandomPoint(Lx: np.float64, Ly: np.float64):
    return np.random.random()*Lx + 1j*np.random.random()*Ly


@njit
def RandomConfig(N: np.uint8, Lx: np.float64, Ly: np.float64
                 ) -> np.array:
    """Returns a random configuration of particles.

    Parameters:
    N : number of particles
    Lx, Ly : perpendicular dimensions of the torus

    Output: 
    R : random configuration of particles"""
    R = np.zeros(N, dtype=np.complex128)
    for p in range(N):
        R[p] = RandomPoint(Lx, Ly)

    return R


@njit
def RandomConfigSWAP(N: np.uint8, Lx: np.float64, Ly: np.float64,
                     boundary: np.array) -> np.array:
    """Returns two random configurations of particles, swappable with
    respect to region A.

    Parameters:

    N : number of particles
    Lx, Ly : perpendicular dimensions of the torus
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Output: 

    R : random configuration of particles"""
    R = np.zeros((N, 2), dtype=np.complex128)
    R[:, 0] = RandomConfig(N, Lx, Ly)
    R[:, 1] = RandomConfig(N, Lx, Ly)

    while np.sum(np.imag(R[:, 0]) < boundary) != np.sum(np.imag(R[:, 1]) < boundary):
        R[:, 1] = RandomConfig(N, Lx, Ly)

    return R


@njit(parallel=True)
def StepProbabilityLaughlin(N: np.uint8, Ns: np.uint16, t: np.complex128,
                            R0: np.array, R1: np.array, p: np.uint8, kCM: np.uint8 = 0,
                            phi_1: np.float64 = 0, phi_t: np.float64 = 0
                            ) -> np.complex128:
    # if we translate once, then only the reduced coordinates appear in the exponential
    # then I apply the magnetic translation, phases come out from the exponential and then
    # just run theta functions as is.
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_t/(2*np.pi) + m*(N-1)/2

    w = np.exp((R1[p]**2 - np.abs(R1[p])**2 - R0[p]**2 + np.abs(R0[p])**2)/4)
    w *= ThetaFunction(m*np.sum(R1)/Lx, m*t, aCM, bCM) / \
        ThetaFunction(m*np.sum(R0)/Lx, m*t, aCM, bCM)
    for i in prange(N):
        if i != p:
            w *= (ThetaFunction((R1[i]-R1[p])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R0[i]-R0[p])/Lx, t, 1/2, 1/2))**m

    return w


@njit(parallel=True)
def StepProbabilityP(N: np.uint8, Ns: np.uint16, t: np.complex128,
                     R_i: np.array, R_f: np.array, p: np.array,
                     kCM: np.uint8 = 0,
                     phi_1: np.float64 = 0, phi_t: np.float64 = 0
                     ) -> np.float64:
    """
    Returns the probability associated with stepping from coordinates R_i
    to coordinates R_f, when calculating the P term in the SWAP decomposition.
    Can also return the result for a fixed particle number sector.

    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    R_i : initial configuration of particles
    R_f : final configuration of particles
    p : indices of particles that move in each copy

    Output:

    r : probaiblity of taking step R_i -> R_f
    """

    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_t/(2*np.pi) + m*(N-1)/2

    r = np.exp((R_f[p[0], 0]**2 - np.abs(R_f[p[0], 0])**2 -
               R_i[p[0], 0]**2 + np.abs(R_i[p[0], 0])**2)/4)
    r *= np.exp((R_f[p[1], 1]**2 - np.abs(R_f[p[1], 1])**2 -
                R_i[p[1], 1]**2 + np.abs(R_i[p[1], 1])**2)/4)
    r *= (ThetaFunction(m*np.sum(R_f[:, 0])/Lx, m*t, aCM, bCM) *
          ThetaFunction(m*np.sum(R_f[:, 1])/Lx, m*t, aCM, bCM)) / \
        (ThetaFunction(m*np.sum(R_i[:, 0])/Lx, m*t, aCM, bCM) *
         ThetaFunction(m*np.sum(R_i[:, 1])/Lx, m*t, aCM, bCM))
    for i in range(N):
        if i != p[0]:
            r *= (ThetaFunction((R_f[i, 0]-R_f[p[0], 0])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i, 0]-R_i[p[0], 0])/Lx, t, 1/2, 1/2))**m
        if i != p[1]:
            r *= (ThetaFunction((R_f[i, 1]-R_f[p[1], 1])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i, 1]-R_i[p[1], 1])/Lx, t, 1/2, 1/2))**m

    return np.abs(r)**2


@njit
def StepProbabilityMod(N: np.uint8, Ns: np.uint16, t: np.complex128,
                       R_i: np.array, R_f: np.array, p: np.array,
                       kCM: np.uint8 = 0,
                       phi_1: np.float64 = 0, phi_t: np.float64 = 0
                       ) -> np.float64:
    """
    Returns the probability associated with stepping from coordinates R_i
    to coordinates R_f, when calculating the P term in the SWAP decomposition.
    Can also return the result for a fixed particle number sector.

    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    R_i : initial configuration of particles
    R_f : final configuration of particles
    p : indices of particles that move in each copy

    Output:

    r : probaiblity of taking step R_i -> R_f
    """

    return StepProbabilityP(N, Ns, t, R_i, R_f, p, kCM, phi_1, phi_t)


@njit
def _PartialContributionSWAP(Lx: np.float64, t: np.complex128,
                             R: np.array, swap_order: np.array, p: np.array):
    """
    """
    x = 1
    for j in range(2):
        for c in range(2):
            for i in range(R.shape[0]):
                if swap_order[i, c]*swap_order[p[j], j] > 0 and swap_order[i, c] != swap_order[p[j], j]:
                    x *= np.sign(np.abs(swap_order[p[j], j]) - np.abs(swap_order[i, c]))*ThetaFunction(
                        (R[i, c]-R[p[j], j])/Lx, t, 1/2, 1/2)

    if swap_order[p[1], 1]*swap_order[p[0], 0] > 0:
        x /= np.sign(np.abs(swap_order[p[1], 1]) - np.abs(swap_order[p[0], 0]))*ThetaFunction(
            (R[p[0], 0]-R[p[1], 1])/Lx, t, 1/2, 1/2)

    return x


@njit(parallel=True)
def StepProbabilitySign(Ns: np.uint16, t: np.complex128,
                        R_i: np.array, R_f: np.array, p: np.array,
                        swap_order_i: np.array, swap_order_f: np.array,
                        kCM: np.uint8 = 0,
                        phi_1: np.float64 = 0, phi_t: np.float64 = 0
                        ) -> np.float64:
    """
    Returns the probability associated with stepping from coordinates R_i
    to coordinates R_f, when calculating the P term in the SWAP decomposition.
    Can also return the result for a fixed particle number sector.

    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    R_i : initial configuration of particles
    R_f : final configuration of particles
    p : indices of particles that move in each copy

    Output:

    r : probaiblity of taking step R_i -> R_f
    """

    N = R_i.shape[0]
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_t/(2*np.pi) + m*(N-1)/2

    r = np.exp((R_f[p[0], 0]**2 - np.abs(R_f[p[0], 0])**2 -
                R_i[p[0], 0]**2 + np.abs(R_i[p[0], 0])**2)/2)
    r *= np.exp((R_f[p[1], 1]**2 - np.abs(R_f[p[1], 1])**2 -
                R_i[p[1], 1]**2 + np.abs(R_i[p[1], 1])**2)/2)

    sum_R_f = np.sum(R_f)
    sum_R_f_0 = np.sum(R_f[:, 0])
    sum_R_SWAP_f_0 = np.sum(np.ravel(R_f)[np.ravel(swap_order_f > 0)])
    sum_R_i = np.sum(R_i)
    sum_R_i_0 = np.sum(R_i[:, 0])
    sum_R_SWAP_i_0 = np.sum(np.ravel(R_i)[np.ravel(swap_order_i > 0)])
    r *= ((ThetaFunction(m*sum_R_f_0/Lx, m*t, aCM, bCM) *
          ThetaFunction(m*(sum_R_f-sum_R_f_0)/Lx, m*t, aCM, bCM) *
          ThetaFunction(m*sum_R_SWAP_f_0/Lx, m*t, aCM, bCM) *
          ThetaFunction(m*(sum_R_f-sum_R_SWAP_f_0)/Lx, m*t, aCM, bCM)) /
          (ThetaFunction(m*sum_R_i_0/Lx, m*t, aCM, bCM) *
           ThetaFunction(m*(sum_R_i-sum_R_i_0)/Lx, m*t, aCM, bCM) *
           ThetaFunction(m*sum_R_SWAP_i_0/Lx, m*t, aCM, bCM) *
           ThetaFunction(m*(sum_R_i-sum_R_SWAP_i_0)/Lx, m*t, aCM, bCM)))
    for i in range(N):
        if i != p[0]:
            r *= (ThetaFunction((R_f[i, 0]-R_f[p[0], 0])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i, 0]-R_i[p[0], 0])/Lx, t, 1/2, 1/2))
        if i != p[1]:
            r *= (ThetaFunction((R_f[i, 1]-R_f[p[1], 1])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i, 1]-R_i[p[1], 1])/Lx, t, 1/2, 1/2))

    r *= np.conj(_PartialContributionSWAP(Lx, t, R_f, swap_order_f, p))
    r /= np.conj(_PartialContributionSWAP(Lx, t, R_i, swap_order_i, p))

    return r**m


@njit
def ValueMod(Ns: np.uint16, t: np.complex128, R: np.array,
             which_A: np.array, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0,
             ) -> np.float64:
    """
    """

    m = Ns/R.shape[0]
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (R.shape[0]-1)/2
    bCM = -phi_t/(2*np.pi) + m*(R.shape[0]-1)/2
    alpha_1 = which_A[:, 0]
    beta_1 = (~which_A[:, 0])
    alpha_2 = which_A[:, 1]
    beta_2 = (~which_A[:, 1])

    v = (ThetaFunction(m*(np.sum(R[alpha_1, 0])+np.sum(R[beta_2, 1]))/Lx, m*t, aCM, bCM) *
         ThetaFunction(m*(np.sum(R[alpha_2, 1])+np.sum(R[beta_1, 0]))/Lx, m*t, aCM, bCM)) / \
        (ThetaFunction(m*np.sum(R[:, 0])/Lx, m*t, aCM, bCM) *
         ThetaFunction(m*np.sum(R[:, 1])/Lx, m*t, aCM, bCM))

    for i in np.flatnonzero(alpha_1):
        for j in np.flatnonzero(beta_2):
            v *= ThetaFunction((R[i, 0]-R[j, 1])/Lx, t, 1/2, 1/2)**m
        for j in np.flatnonzero(beta_1):
            v /= ThetaFunction((R[i, 0]-R[j, 0])/Lx, t, 1/2, 1/2)**m
    for i in np.flatnonzero(alpha_2):
        for j in np.flatnonzero(beta_1):
            v *= ThetaFunction((R[i, 1]-R[j, 0])/Lx, t, 1/2, 1/2)**m
        for j in np.flatnonzero(beta_2):
            v /= ThetaFunction((R[i, 1]-R[j, 1])/Lx, t, 1/2, 1/2)**m

    return np.abs(v)


@njit
def ValueSign(Ns: np.uint16, t: np.complex128, R: np.array,
              swap_order: np.array, kCM: np.uint8 = 0,
              phi_1: np.float64 = 0, phi_t: np.float64 = 0,
              ) -> np.complex128:
    """
    """
    R_SWAP = np.zeros((R.shape[0], R.shape[1]), dtype=np.complex128)
    for i in range(R.shape[0]):
        for c in range(2):
            if swap_order[i, c] > 0:
                R_SWAP[np.abs(swap_order[i, c])-1, 0] = R[i, c]
            else:
                R_SWAP[np.abs(swap_order[i, c])-1, 1] = R[i, c]

    v = (np.conj(LaughlinTorus(R.shape[0], Ns, t, R_SWAP[:, 0], kCM, phi_1, phi_t)) *
         np.conj(LaughlinTorus(R.shape[0], Ns, t, R_SWAP[:, 1], kCM, phi_1, phi_t)) *
         LaughlinTorus(R.shape[0], Ns, t, R[:, 0], kCM, phi_1, phi_t) *
         LaughlinTorus(R.shape[0], Ns, t, R[:, 1], kCM, phi_1, phi_t))

    return v


@njit
def PBCWithPhase(Lx: np.float64, Ly: np.float64, t: np.complex128,
                 z: np.complex128, phi_1: np.float64, phi_t: np.float64
                 ):
    """Check if the particle position wrapped around the torus
    after one step. When a step wraps around both directions,
    the algorithm applies """
    phi = 1
    w = np.copy(z)
    if np.imag(w) > Ly:
        phi += phi_t + Lx*(np.real(t)*np.imag(w) - np.imag(t)*np.real(w))/2
        w -= Lx*t
    elif np.imag(w) < 0:
        phi -= phi_t + Lx*(np.real(t)*np.imag(w) - np.imag(t)*np.real(w))/2
        w += Lx*t
    if np.real(w) > Lx:
        phi += phi_1 + Lx*np.imag(w)/2
        w -= Lx
    elif np.real(w) < 0:
        phi -= phi_1 + Lx*np.imag(w)/2
        w += Lx

    return w, np. exp(1j*phi)


@njit
def PBC(Lx: np.float64, Ly: np.float64, t: np.complex128,
        z: np.complex128) -> np.complex128:
    """Check if the particle position wrapped around the torus
    after one step. When a step wraps around both directions,
    the algorithm applies """
    if np.imag(z) > Ly:
        z -= Lx*t
    elif np.imag(z) < 0:
        z += Lx*t
    if np.real(z) > Lx:
        z -= Lx
    elif np.real(z) < 0:
        z += Lx

    return z


@njit
def PBCAllWithPhases(Lx: np.float64, Ly: np.float64, t: np.complex128,
                     z: np.array, phi_1: np.float64, phi_t: np.float64
                     ):
    """Check if particle positions wrapped around the torus
    after one step. Returns the position of the particle in the
    (Lx, Ly) unit cell and the corresponding phases."""
    w = np.ravel(z)
    phi = np.zeros(w.size)
    for p in range(w.size):
        if np.imag(w[p]) > Ly:
            phi[p] += phi_t + Lx * \
                (np.real(t)*np.imag(w[p]) - np.imag(t)*np.real(w[p]))/2
            w[p] -= Lx*t
        elif np.imag(w[p]) < 0:
            phi[p] -= phi_t + Lx * \
                (np.real(t)*np.imag(w[p]) - np.imag(t)*np.real(w[p]))/2
            w[p] += Lx*t
        if np.real(w[p]) > Lx:
            phi[p] += phi_1 + Lx*np.imag(w[p])/2
            w[p] -= Lx
        elif np.real(w[p]) < 0:
            phi[p] -= phi_1 + Lx*np.imag(w[p])/2
            w[p] += Lx

    return w.reshape(z.shape), np.exp(1j*phi).reshape(z.shape)


@njit
def PBCAll(Lx: np.float64, Ly: np.float64, t: np.complex128,
           R: np.array) -> np.array:
    """Check if particle positions wrapped around the torus
    after one step. Returns the position of the particle in the
    (Lx, Ly) unit cell and the corresponding phases."""
    w = np.ravel(R)
    w[np.imag(w) > Ly] -= Lx*t
    w[np.imag(w) < 0] += Lx*t
    w[np.real(w) > Lx] -= Lx
    w[np.real(w) < 0] += Lx

    return w.reshape(R.shape)


@njit
def TestPBC(N: np.uint8, Ns: np.uint16, t: np.complex128,
            kCM: np.uint8 = 0, phi_1: np.float64 = 0,
            phi_t: np.float64 = 0,):

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    R0 = RandomConfig(N, Lx, Ly)

    p = np.random.randint(0, N)
    R_test = np.copy(R0)
    R_test[p] += Lx
    r_1 = np.exp(-1j*Lx*np.imag(R0[p])/2)*LaughlinTorus(
        N, Ns, t, R_test, kCM, phi_1, phi_t)/LaughlinTorus(
        N, Ns, t, R0, kCM, phi_1, phi_t)
    print('t(Lx)*e^(-i phi_1) =', r_1*np.exp(-1j*phi_1))

    R_test = np.copy(R0)
    R_test[p] += Lx*t
    r_t = np.exp(1j*Ly*np.real(R0[p])/2)*LaughlinTorus(
        N, Ns, t, R_test, kCM, phi_1, phi_t)/LaughlinTorus(
        N, Ns, t, R0, kCM, phi_1, phi_t)
    print('t(Lx*tau)*e^(-i phi_t) =', r_t*np.exp(-1j*phi_t))


@njit
def AssignOrderSWAP(R: np.array, which_A: np.array,
                    ) -> np.array:
    """Assign an order to particles in the swapped copies.
    """

    swap_order = np.zeros((R.shape[0], R.shape[1]), dtype=np.int8)
    i_swap = 1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 0]:
            swap_order[i, 0] = i_swap
        else:
            while not which_A[j, 1]:
                j += 1
            swap_order[j, 1] = i_swap
            j += 1
        i_swap += 1

    i_swap = -1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 1]:
            swap_order[i, 1] = i_swap
        else:
            while not which_A[j, 0]:
                j += 1
            swap_order[j, 0] = i_swap
            j += 1
        i_swap -= 1

    return swap_order


@njit
def UpdateOrderSWAP(swap_order_i: np.array, p: np.array,
                    delta: np.uint8, where_moves: np.array):
    """
    Updates the order in the swapped copies after one move.
    """

    swap_order_f = np.copy(swap_order_i)
    if where_moves[p[0], delta] != 0:
        x = swap_order_f[p[1], 1]
        swap_order_f[p[1], 1] = swap_order_f[p[0], 0]
        swap_order_f[p[0], 0] = x

    return swap_order_f


@njit
def LocateParticles(Ly: np.float64, R: np.array,
                    boundary: np.float64, step_size: np.float64):
    """
    For a given configuration of particles, returns information
    about their positions relative to a region A. Also assigns a
    particle order in the teow copies of R_SWAP.

    Parameters:

    Ly : torus dimension along y-axis
    R : particles configuration
    boundary : dimensionful boundary between regions A and B 
                (the other one is implicit at y=0)
    step_size: initial step size in units of Lx

    Output:

    which_A : boolean array indicating which particles are in region A
    where_moves: array (N, 2, 4) specifying where each type of move take
                each particle in each of the copies. the specific move
                increases n_A by the value of where_moves.
                where_move[i,j,k]=m means move k
                (k=0 => +1 ; k=1 => -1; k=2 => +1j; k=3 => -1j) applied on
                particle i in copy j changes n_A in copy j by m.
    """

    y = np.imag(R)
    where_moves = np.zeros((R.shape[0], R.shape[1], 4), dtype=np.int8)

    under_edge_A = ((y - boundary > -step_size) & (y - boundary < 0))
    above_edge_A = (y < step_size)
    under_edge_B = (y > Ly-step_size)
    above_edge_B = ((y - boundary > 0) & (y - boundary < step_size))

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if above_edge_A[i, j]:
                where_moves[i, j, 3] = -1
            elif under_edge_A[i, j]:
                where_moves[i, j, 2] = -1
            if above_edge_B[i, j]:
                where_moves[i, j, 3] = 1
            if under_edge_B[i, j]:
                where_moves[i, j, 2] = 1

    which_A = ((y - boundary) < 0)

    return which_A, where_moves


@njit
def UpdateMoves(R: np.array, p: np.array, where_moves: np.array,
                Ly: np.float64, boundary: np.float64,
                step_size: np.float64,):
    """
    For a given new configuration of particles, updates information
    about their positions relative to a region A.

    Parameters:

    Ly : torus dimension along y-axis
    R : particles configuration
    p : indices of particles that move in each copy
    boundary : dimensionful boundary between regions A and B 
                (the other one is implicit at y=0)
    step_size: initial step size in units of Lx
    """
    y_0 = np.imag(R[p[0], 0])
    y_bd_0 = y_0 - boundary
    y_1 = np.imag(R[p[1], 1])
    y_bd_1 = y_1 - boundary

    if (y_0 < step_size):
        where_moves[p[0], 0, :] = np.array([0, 0, 0, -1])
    elif (y_bd_0 > -step_size) and (y_bd_0 < 0):
        where_moves[p[0], 0, :] = np.array([0, 0, -1, 0])
    elif (y_bd_0 > 0) and (y_bd_0 < step_size):
        where_moves[p[0], 0, :] = np.array([0, 0, 0, 1])
    elif (y_0 > Ly-step_size):
        where_moves[p[0], 0, :] = np.array([0, 0, 1, 0])
    else:
        where_moves[p[0], 0, :] = np.array([0, 0, 0, 0])

    if (y_1 < step_size):
        where_moves[p[1], 1, :] = np.array([0, 0, 0, -1])
    elif (y_bd_1 > -step_size) and (y_bd_1 < 0):
        where_moves[p[1], 1, :] = np.array([0, 0, -1, 0])
    elif (y_bd_1 > 0) and (y_bd_1 < step_size):
        where_moves[p[1], 1, :] = np.array([0, 0, 0, 1])
    elif (y_1 > Ly-step_size):
        where_moves[p[1], 1, :] = np.array([0, 0, 1, 0])
    else:
        where_moves[p[1], 1, :] = np.array([0, 0, 0, 0])


@njit
def ValidStep(which_A_f: np.array,
              where_moves: np.array):

    valid = True
    p1 = 0
    delta1 = 0

    valid_mask = (np.sum(which_A_f[:, 0]) == (
        np.sum(which_A_f[:, 1]) + where_moves))

    parameters_move = np.arange(where_moves.size)
    valid_moves = parameters_move[np.ravel(valid_mask)]

    if valid_moves.size:
        move = np.random.choice(valid_moves)
        p1 = move//4
        delta1 = np.array([1, -1, 1j, -1j])[move % 4]
    else:
        valid = False

    return valid, p1, delta1


@njit
def SwapConfig(N: np.uint8, R: np.array, which_1: List, which_2: List):
    R_SWAP = np.copy(R)
    for i in range(len(which_1)):
        r = R_SWAP[which_1[i], 0]
        R_SWAP[which_1[i], 0] = R_SWAP[which_2[i], 1]
        R_SWAP[which_2[i], 1] = r

    return R_SWAP


@njit
def StepOne(Lx: np.float64, Ly: np.float64, t: np.complex128,
            step_size: np.float64, R_i: np.array,
            which_A_i: np.array, boundary
            ) -> np.array:
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy.

    Parameters:

    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    R_i : initial position of all particles
    which_A_i : boolean array indicating which particles are initially in region A
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    R_f : final position of all particles
    p : indices of particles that move in each copy
    which_A_f : boolean array indicating which particles are finally in region A
    """

    R_f = np.copy(R_i)
    p = np.random.randint(0, R_i.shape[0], 2)
    delta = step_size * np.random.choice(np.array([1, -1, 1j, -1j]), 2)
    R_f[p[0], 0] = PBC(Lx, Ly, t, R_i[p[0], 0]+delta[0])
    R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[1], 1]+delta[1])

    which_A_f = np.copy(which_A_i)
    which_A_f[p[0], 0] = (np.imag(R_f[p[0], 0]) - boundary < 0)
    which_A_f[p[1], 1] = (np.imag(R_f[p[1], 1]) - boundary < 0)

    return R_f, p, which_A_f


@njit
def StepOneSWAPBiased(Lx: np.float64, Ly: np.float64, t: np.complex128,
                      step_size: np.float64, R_i: np.array,
                      which_A_i: np.array, where_moves: np.array,
                      boundary: np.float64,
                      ) -> np.array:
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy, ensuring that
    the copies are swappable with respect to region A.

    Parameters:

    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    R_i : initial position of all particles
    which_A_i : boolean array indicating which particles are initially in region A
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    R_f : final position of all particles
    p : indices of particles that move in each copy
    which_A_f : boolean array indicating which particles are finally in region A
    """

    R_f = np.copy(R_i)
    which_A_f = np.copy(which_A_i)
    p = np.zeros(2, dtype=np.uint8)
    delta = np.zeros(2, dtype=np.complex128)
    valid = False

    while not valid:
        p[0] = np.random.randint(R_i.shape[0])
        delta[0] = np.random.choice(np.array([1, -1, 1j, -1j]))

        R_f[p[0], 0] = PBC(Lx, Ly, t, R_i[p[0], 0]+delta[0]*step_size)
        which_A_f[p[0], 0] = (np.imag(R_f[p[0], 0]) - boundary < 0)
        valid, p[1], delta[1] = ValidStep(which_A_f, where_moves)

    R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[1], 1]+delta[1]*step_size)
    which_A_f[p[1], 1] = (np.imag(R_f[p[1], 1]) - boundary < 0)

    return R_f, p, which_A_f


@njit
def StepOneSWAP(Lx: np.float64, Ly: np.float64, t: np.complex128,
                step_size: np.float64, R_i: np.array,
                which_A_i: np.array, where_moves: np.array,
                boundary: np.float64,
                ) -> np.array:
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy, ensuring that
    the copies are swappable with respect to region A.

    Parameters:

    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    R_i : initial position of all particles
    which_A_i : boolean array indicating which particles are initially in region A
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    R_f : final position of all particles
    p : indices of particles that move in each copy
    which_A_f : boolean array indicating which particles are finally in region A
    """

    R_f = np.copy(R_i)
    which_A_f = np.copy(which_A_i)

    valid = False
    while not valid:
        p = np.random.randint(0, R_i.shape[0], 2)
        delta = np.random.randint(0, 4, 2)
        delta = np.random.choice(np.array([1, -1, 1j, -1j]), 2)

        if where_moves[p[0], delta[0]] == where_moves[p[1], delta[1]]:
            valid = True

        R_f[p[0], 0] = PBC(Lx, Ly, t, R_i[p[0], 0] +
                           np.array([1, -1, 1j, -1j])[delta[0]]*step_size)
        R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[0], 0] +
                           np.array([1, -1, 1j, -1j])[delta[0]]*step_size)
        which_A_f[p[0], 0] = (np.imag(R_f[p[0], 0]) - boundary < 0)
        valid, p[1], delta[1] = ValidStep(which_A_f, where_moves)

    R_f[p[0], 0] = PBC(Lx, Ly, t, R_i[p[0], 0] +
                       np.array([1, -1, 1j, -1j])[delta[0]]*step_size)
    R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[0], 0] +
                       np.array([1, -1, 1j, -1j])[delta[0]]*step_size)
    which_A_f[p[0], 0] = (np.imag(R_f[p[0], 0]) - boundary < 0)
    R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[1], 1]+delta[1]*step_size)
    which_A_f[p[1], 1] = (np.imag(R_f[p[1], 1]) - boundary < 0)

    return R_f, p, which_A_f


@njit
def StepOneSWAPRandom(Lx: np.float64, Ly: np.float64, t: np.complex128,
                      step_size: np.float64, R_i: np.array,
                      which_A_i: np.array, where_moves: np.array,
                      boundary: np.float64,
                      ) -> np.array:
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy, ensuring that
    the copies are swappable with respect to region A.

    Parameters:

    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    R_i : initial position of all particles
    which_A_i : boolean array indicating which particles are initially in region A
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    R_f : final position of all particles
    p : indices of particles that move in each copy
    which_A_f : boolean array indicating which particles are finally in region A
    """

    R_f = np.copy(R_i)
    which_A_f = np.copy(which_A_i)

    valid = False
    while not valid:
        p = np.random.randint(0, R_i.shape[0], 2)
        delta = np.random.randint(0, 4, 2)

        if where_moves[p[0], 0, delta[0]] == where_moves[p[1], 1, delta[1]]:
            valid = True

    R_f[p[0], 0] = PBC(Lx, Ly, t, R_i[p[0], 0] +
                       np.array([1, -1, 1j, -1j])[delta[0]]*step_size)
    R_f[p[1], 1] = PBC(Lx, Ly, t, R_i[p[1], 1] +
                       np.array([1, -1, 1j, -1j])[delta[1]]*step_size)
    which_A_f[p[0], 0] = (np.imag(R_f[p[0], 0]) - boundary < 0)
    which_A_f[p[1], 1] = (np.imag(R_f[p[1], 1]) - boundary < 0)

    return R_f, p, delta, which_A_f


@njit
def UpdateResult(result: np.array, update: np.complex128,
                 index: np.uint32, acceptance: np.float64,
                 accept_bit: np.uint8):
    result[index] = update
    new_acceptance = (acceptance*index + accept_bit)/(index + 1)

    if (index+1) % (result.shape[0]//20) == 0:
        print('Iteration', index+1, 'done, current acceptance ratio:',
              np.round(acceptance*100, 2), '%')

    return new_acceptance


@njit
def RunPSWAP(N: np.uint8, Ns: np.uint16, t: np.complex64,
             M: np.uint32, step_size: np.float64,
             boundary_dimensionless: np.array,
             kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0
             ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    term : which term in the mod/sign decomposition (p/mod/sign)
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    result: array of length M containing results at each step"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    acceptance: np.float64 = 0
    result = np.zeros(M, dtype=np.float64)

    step_size *= Lx
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    R_i = np.zeros((N, 2), dtype=np.complex128)
    R_i[:, 0] = RandomConfig(N, Lx, Ly)
    R_i[:, 1] = RandomConfig(N, Lx, Ly)
    which_A_i, _ = LocateParticles(Ly, R_i, boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = (np.sum(which_A_i[:, 0]) == np.sum(which_A_i[:, 1]))

    for i in range(M):
        accept_bit = 0
        R_f, p, which_A_f = StepOne(
            Lx, Ly, t, step_size, R_i, which_A_i, boundary)

        r = StepProbabilityP(N, Ns, t, R_i, R_f, p)

        if r > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            which_A_i = np.copy(which_A_f)
            update = (np.sum(which_A_i[:, 0]) == np.sum(which_A_i[:, 1]))

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result


@njit
def RunModSWAP(N: np.uint8, Ns: np.uint16, t: np.complex64,
               M: np.uint32, step_size: np.float64,
               boundary_dimensionless: np.float64,
               kCM: np.uint8 = 0,
               phi_1: np.float64 = 0, phi_t: np.float64 = 0
               ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    term : which term in the mod/sign decomposition (p/mod/sign)
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    result: array of length M containing results at each step"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    acceptance: np.float64 = 0
    result = np.zeros((M), dtype=np.float64)
    # result_R = np.zeros((M, N, 2), dtype=np.complex128)

    step_size *= Lx
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    R_i = RandomConfigSWAP(N, Lx, Ly, boundary)
    which_A_i, where_moves = LocateParticles(
        Ly, R_i, boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = ValueMod(Ns, t, R_i, which_A_i)

    for i in range(M):
        accept_bit = 0
        R_f, p, _, which_A_f = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, which_A_i, where_moves, boundary)

        if np.count_nonzero(which_A_f[:, 0]) != np.count_nonzero(which_A_f[:, 1]):
            print("not swappable!")
            print(R_i)
            print(R_f)
            print(where_moves[:, 0, :])
            print(where_moves[:, 1, :])
            print(which_A_i)
            print(which_A_f)
            break

        r = StepProbabilityMod(N, Ns, t, R_i, R_f, p)

        if r > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            which_A_i = np.copy(which_A_f)
            UpdateMoves(R_i, p, where_moves, Ly, boundary, step_size)
            update = ValueMod(Ns, t, R_i, which_A_i)

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result


@njit
def RunSignSWAP(N: np.uint8, Ns: np.uint16, t: np.complex64,
                M: np.uint32, step_size: np.float64,
                boundary_dimensionless: np.float64,
                kCM: np.uint8 = 0,
                phi_1: np.float64 = 0, phi_t: np.float64 = 0
                ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    term : which term in the mod/sign decomposition (p/mod/sign)
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:

    result: array of length M containing results at each step"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    acceptance: np.float64 = 0
    result = np.zeros((M), dtype=np.complex128)
    # result_R = np.zeros((M, N, 2), dtype=np.complex128)

    step_size *= Lx
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    R_i = RandomConfigSWAP(N, Lx, Ly, boundary)
    which_A_i, where_moves = LocateParticles(
        Ly, R_i, boundary, step_size)
    swap_order_i = AssignOrderSWAP(R_i, which_A_i)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    v_0 = ValueSign(Ns, t, R_i, swap_order_i)
    update = v_0/np.abs(v_0)

    for i in range(M):
        accept_bit = 0
        R_f, p, delta, which_A_f = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, which_A_i, where_moves, boundary)

        swap_order_f = UpdateOrderSWAP(
            swap_order_i, p, delta[0], where_moves[:, 0, :])

        # r = StepProbabilitySign(Ns, t, R_i, R_f, p, swap_order_i, swap_order_f)
        v_1 = ValueSign(Ns, t, R_f, swap_order_f)
        r = v_1/v_0

        if np.abs(r) > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            which_A_i = np.copy(which_A_f)
            UpdateMoves(R_i, p, where_moves, Ly, boundary, step_size)
            swap_order_i = np.copy(swap_order_f)
            update *= r/np.abs(r)
            v_0 = v_1

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result
