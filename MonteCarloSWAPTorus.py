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
def RatioStepOne(Ns: np.uint16, t: np.complex128,
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

    N = R_i.size
    m = Ns/N
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
    bCM = -phi_t/(2*np.pi) + m*(N-1)/2

    r = np.exp((R_f[p]**2 - np.abs(R_f[p])**2 -
               R_i[p]**2 + np.abs(R_i[p])**2)/4)
    # theta_cm = ThetaFunctionVectorized((m/Lx)*np.array([np.sum(R_f[:, 0]), np.sum(R_f[:, 1]),
    #                                                    np.sum(R_i[:, 0]), np.sum(R_i[:, 1])]),
    #                                   m*t, aCM, bCM)
    # r *= theta_cm[0]*theta_cm[1]/(theta_cm[2]*theta_cm[3])

    r *= (ThetaFunction(m*np.sum(R_f)/Lx, m*t, aCM, bCM) /
          ThetaFunction(m*np.sum(R_i)/Lx, m*t, aCM, bCM))

    for i in range(N):
        if i != p:
            r *= (ThetaFunction((R_f[i]-R_f[p])/Lx, t, 1/2, 1/2) /
                  ThetaFunction((R_i[i]-R_i[p])/Lx, t, 1/2, 1/2))**m

    return r


@njit(parallel=True)
def RatioStepOneSWAP(Ns: np.uint16, t: np.complex128,
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
        return (RatioStepOne(Ns, t, swap_R_i[:, 0], swap_R_f[:, 0], p_swap[0], kCM, phi_1, phi_t) *
                RatioStepOne(Ns, t, swap_R_i[:, 1], swap_R_f[:, 1], p_swap[1], kCM, phi_1, phi_t))

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
        aCM = phi_1/(2*np.pi*m) + kCM/m + (N-1)/2
        bCM = -phi_t/(2*np.pi) + m*(N-1)/2

        diff_swap_R_f = swap_R_f[:, copy]
        diff_swap_R_i = swap_R_i[:, copy]

        r = 1
        for j in range(2):
            r *= np.exp((diff_swap_R_f[p_swap[j]]**2 - np.abs(diff_swap_R_f[p_swap[j]])**2 -
                         diff_swap_R_i[p_swap[j]]**2 + np.abs(diff_swap_R_i[p_swap[j]])**2)/4)
        # theta_cm = ThetaFunctionVectorized((m/Lx)*np.array([np.sum(R_f[:, 0]), np.sum(R_f[:, 1]),
        #                                                    np.sum(R_i[:, 0]), np.sum(R_i[:, 1])]),
        #                                   m*t, aCM, bCM)
        # r *= theta_cm[0]*theta_cm[1]/(theta_cm[2]*theta_cm[3])

        r *= (ThetaFunction(m*np.sum(diff_swap_R_f)/Lx, m*t, aCM, bCM) /
              ThetaFunction(m*np.sum(diff_swap_R_i)/Lx, m*t, aCM, bCM))

        for j in range(N):
            if j != p_swap[0] and j != p_swap[1]:
                r *= ((ThetaFunction((diff_swap_R_f[j]-diff_swap_R_f[p_swap[0]])/Lx, t, 1/2, 1/2) *
                       ThetaFunction((diff_swap_R_f[j]-diff_swap_R_f[p_swap[1]])/Lx, t, 1/2, 1/2)) /
                      (ThetaFunction((diff_swap_R_i[j]-diff_swap_R_i[p_swap[0]])/Lx, t, 1/2, 1/2) *
                       ThetaFunction((diff_swap_R_i[j]-diff_swap_R_i[p_swap[1]])/Lx, t, 1/2, 1/2)))**m

        r *= (ThetaFunction((diff_swap_R_f[p_swap[0]]-diff_swap_R_f[p_swap[1]])/Lx, t, 1/2, 1/2) /
              ThetaFunction((diff_swap_R_i[p_swap[0]]-diff_swap_R_i[p_swap[1]])/Lx, t, 1/2, 1/2))**m

        return r


@njit
def ValueModOld(Ns: np.uint16, t: np.complex128, R: np.array,
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
def ValueMod(Ns: np.uint16, t: np.complex128, R: np.array,
             swap_R: np.array, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0,
             ) -> np.complex128:
    """
    Returns the mod term for a given configuration.
    """
    v = ((LaughlinTorus(R.shape[0], Ns, t, swap_R[:, 0], kCM, phi_1, phi_t) *
         LaughlinTorus(R.shape[0], Ns, t, swap_R[:, 1], kCM, phi_1, phi_t)) /
         (LaughlinTorus(R.shape[0], Ns, t, R[:, 0], kCM, phi_1, phi_t) *
         LaughlinTorus(R.shape[0], Ns, t, R[:, 1], kCM, phi_1, phi_t)))

    return np.abs(v)


@njit
def ValueSign(Ns: np.uint16, t: np.complex128, R: np.array,
              swap_R: np.array, kCM: np.uint8 = 0,
              phi_1: np.float64 = 0, phi_t: np.float64 = 0,
              ) -> np.complex128:
    """
    Returns the sign term for a given configuration.
    """
    v = (np.conj(LaughlinTorus(R.shape[0], Ns, t, swap_R[:, 0], kCM, phi_1, phi_t)) *
         np.conj(LaughlinTorus(R.shape[0], Ns, t, swap_R[:, 1], kCM, phi_1, phi_t)) *
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
def TestPBC(N: np.uint8, Ns: np.uint16, t: np.complex128,
            kCM: np.uint8 = 0, phi_1: np.float64 = 0,
            phi_t: np.float64 = 0,):

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    R0 = RandomConfig(N, Lx, Ly)

    p = np.random.randint(0, N)
    R_test = np.copy(R0)
    R_test[p] += Lx
    print('t(Lx)*e^(-i phi_1) =', (RatioStepOne(Ns, t, R0, R_test, p, kCM, phi_1, phi_t) *
                                   np.exp(-1j*Lx*np.imag(R0[p])/2)*np.exp(-1j*phi_1)))

    R_test = np.copy(R0)
    R_test[p] += Lx*t
    print('t(Lx*tau)*e^(-i phi_t) =', (RatioStepOne(Ns, t, R0, R_test, p, kCM, phi_1, phi_t) *
                                       np.exp(1j*Ly*np.real(R0[p])/2)*np.exp(-1j*phi_t)))


@njit
def AssignOrderSWAP(R: np.array, which_A: np.array,
                    ) -> np.array:
    """Assign an order to particles in the swapped copies. Also creates the
    vector R_SWAP containing the swapped configurations.
    R_SWAP[:,0] - config alpha_1, beta_2 (negative indices in swap_order)
    R_SWAP[:,1] - config alpha_2, beta_1 (positive indices in swap_order)
    """

    swap_order = np.zeros((R.shape[0], R.shape[1]), dtype=np.int8)
    swap_R = np.zeros((R.shape[0], R.shape[1]), dtype=np.complex128)
    i_swap = 1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 0]:
            swap_order[i, 0] = i_swap
            swap_R[i_swap-1, 1] = R[i, 0]
        else:
            while not which_A[j, 1]:
                j += 1
            swap_order[j, 1] = i_swap
            swap_R[i_swap-1, 1] = R[j, 1]
            j += 1
        i_swap += 1

    i_swap = -1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 1]:
            swap_order[i, 1] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[i, 1]
        else:
            while not which_A[j, 0]:
                j += 1
            swap_order[j, 0] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[j, 0]
            j += 1
        i_swap -= 1

    return swap_order, swap_R


@njit
def UpdateOrderSWAP(R_f: np.array, swap_order_i: np.array, swap_R_i: np.array,
                    p: np.array, delta: np.uint8, where_moves: np.array):
    """
    Updates the order in the swapped copies after one move.
    """

    swap_order_f = np.copy(swap_order_i)
    swap_R_f = np.copy(swap_R_i)
    if where_moves[p[0], delta] != 0:
        x = swap_order_f[p[1], 1]
        swap_order_f[p[1], 1] = swap_order_f[p[0], 0]
        swap_order_f[p[0], 0] = x

    for i in range(2):
        if swap_order_f[p[i], i] > 0:
            swap_R_f[swap_order_f[p[i], i]-1, 1] = R_f[p[i], i]
        else:
            swap_R_f[np.abs(swap_order_f[p[i], i])-1, 0] = R_f[p[i], i]

    return swap_order_f, swap_R_f


@njit
def LocateAndAssignOrderSWAP(Ly: np.float64, R: np.array,
                             boundary: np.float64, step_size: np.float64
                             ) -> (np.array, np.array, np.array):
    """
    For a given configuration of particles, returns information
    about their positions relative to a region A. Also assigns a
    particle order in the two copies of swap_R and returns them.
    R_SWAP[:,0] - config alpha_1, beta_2 (negative indices in swap_order)
    R_SWAP[:,1] - config alpha_2, beta_1 (positive indices in swap_order)

    Parameters:

    Ly : torus dimension along y-axis
    R : particles configuration
    boundary : dimensionful boundary between regions A and B 
                (the other one is implicit at y=0)
    step_size : initial step size in units of Lx

    Output:

    swap_order : order or particles in the swapped copies. positive indices go
                into swap_R[:,1] and negative into swap_R[:,0]
    swap_R : array containing the ordered positions of particles in the swapped
            copies. swap_R[:,0] containts alpha_1, beta_2
    where_moves : array (N, 2, 4) specifying where each type of move take
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

    swap_order = np.zeros((R.shape[0], R.shape[1]), dtype=np.int8)
    swap_R = np.zeros((R.shape[0], R.shape[1]), dtype=np.complex128)
    i_swap = 1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 0]:
            swap_order[i, 0] = i_swap
            swap_R[i_swap-1, 1] = R[i, 0]
        else:
            while not which_A[j, 1]:
                j += 1
            swap_order[j, 1] = i_swap
            swap_R[i_swap-1, 1] = R[j, 1]
            j += 1
        i_swap += 1

    i_swap = -1
    j = 0
    for i in range(R.shape[0]):
        if not which_A[i, 1]:
            swap_order[i, 1] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[i, 1]
        else:
            while not which_A[j, 0]:
                j += 1
            swap_order[j, 0] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[j, 0]
            j += 1
        i_swap -= 1

    return swap_order, swap_R, where_moves


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
def StepOne(Lx: np.float64, Ly: np.float64, t: np.complex128,
            step_size: np.float64, R_i: np.array
            ) -> (np.array, np.uint8):
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle.

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
    p : index of particles that moves
    which_A_f : boolean array indicating which particles are finally in region A
    """

    R_f = np.copy(R_i)
    p = np.random.randint(0, R_i.size)
    delta = step_size * np.random.choice(np.array([1, -1, 1j, -1j]))
    R_f[p] = PBC(Lx, Ly, t, R_i[p]+delta)

    return R_f, p


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
                      where_moves: np.array,
                      ) -> np.array:
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy, ensuring that
    the copies are swappable with respect to region A.

    Parameters:

    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    R_i : initial position of all particles
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Output:

    R_f : final position of all particles
    p : indices of particles that move in each copy
    """

    R_f = np.copy(R_i)

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

    return R_f, p, delta


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
def RunPParticleSector(N: np.uint8, Ns: np.uint16, t: np.complex64,
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
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:
    result: array of length M containing n_A at each step
    acceptance: final average acceptance of MC run
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    result = np.zeros(M, dtype=np.float64)
    R_i = RandomConfig(N, Lx, Ly)
    update = np.count_nonzero(((np.imag(R_i) - boundary) < 0))

    for i in range(M):
        accept_bit = 0
        R_f, p = StepOne(Lx, Ly, t, step_size, R_i)
        r = RatioStepOne(Ns, t, R_i, R_f, p)

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            update = np.count_nonzero(((np.imag(R_i) - boundary) < 0))

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result, acceptance


@njit
def RunDecoupledPSWAP(N: np.uint8, Ns: np.uint16, t: np.complex64,
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
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:
    result: array of length M containing n_A in eaach copy at each step
    acceptance: final average acceptance of MC run
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance = np.zeros(2, dtype=np.float64)
    step_size *= Lx
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    result = np.zeros((M, 2), dtype=np.float64)
    R_i = np.vstack((RandomConfig(N, Lx, Ly), RandomConfig(N, Lx, Ly))).T
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = np.array(
        [np.count_nonzero(((np.imag(R_i[:, 0]) - boundary) < 0)),
         np.count_nonzero(((np.imag(R_i[:, 1]) - boundary) < 0))])

    for i in range(M):
        # accept_bits = np.zeros(2, dtype=np.uint8)
        p = np.zeros(2, dtype=np.uint8)
        for j in prange(2):
            R_f[:, j], p[j] = StepOne(Lx, Ly, t, step_size, R_i[:, j])
            r = RatioStepOne(Ns, t, R_i[:, j], R_f[:, j], p[j])

            if np.abs(r)**2 > np.random.random():
                # accept_bits[j] = 1
                R_i[:, j] = np.copy(R_f[:, j])
                update[j] = np.count_nonzero(
                    ((np.imag(R_i[:, j]) - boundary) < 0))

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result, acceptance


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
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Returns:
    result: array of length M containing n_A in eaach copy at each step
    acceptance: final average acceptance of MC run
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    result = np.zeros((M, 2), dtype=np.float64)
    R_i = np.vstack((RandomConfig(N, Lx, Ly), RandomConfig(N, Lx, Ly))).T
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = np.array(
        [np.count_nonzero(((np.imag(R_i[:, 0]) - boundary) < 0)),
         np.count_nonzero(((np.imag(R_i[:, 1]) - boundary) < 0))])

    for i in range(M):
        accept_bit = 0
        p = np.zeros(2, dtype=np.uint8)
        for j in range(2):
            R_f[:, j], p[j] = StepOne(Lx, Ly, t, step_size, R_i[:, j])

        r = (RatioStepOne(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
             RatioStepOne(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))

        # r = 1
        # for j in prange(2):
        #    r *= RatioStepOne(Ns, t, R_i[:, j], R_f[:, j], p[j])

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            update = np.array(
                [np.count_nonzero(((np.imag(R_i[:, 0]) - boundary) < 0)),
                 np.count_nonzero(((np.imag(R_i[:, 1]) - boundary) < 0))])

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result, acceptance


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
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    result = np.zeros((M), dtype=np.complex128)
    R_i = RandomConfigSWAP(N, Lx, Ly, boundary)
    swap_order_i, swap_R_i, where_moves = LocateAndAssignOrderSWAP(
        Ly, R_i, boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = ValueMod(Ns, t, R_i, swap_R_i)

    for i in range(M):
        accept_bit = 0
        R_f, p, delta = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, where_moves)

        r = (RatioStepOne(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
             RatioStepOne(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            swap_order_f, swap_R_f = UpdateOrderSWAP(
                R_f, swap_order_i, swap_R_i, p, delta[0], where_moves[:, 0, :])
            p_swap_order = np.array(
                [swap_order_f[p[0], 0], swap_order_f[p[1], 1]])
            update *= np.abs(RatioStepOneSWAP(Ns, t, swap_R_i,
                             swap_R_f, p_swap_order)/r)
            R_i = np.copy(R_f)
            UpdateMoves(R_i, p, where_moves, Ly, boundary, step_size)
            swap_order_i = np.copy(swap_order_f)
            swap_R_i = np.copy(swap_R_f)

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result, acceptance


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
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Ly*boundary_dimensionless

    TestPBC(N, Ns, t, kCM, phi_1, phi_t)

    result = np.zeros((M), dtype=np.complex128)
    R_i = RandomConfigSWAP(N, Lx, Ly, boundary)
    swap_order_i, swap_R_i, where_moves = LocateAndAssignOrderSWAP(
        Ly, R_i, boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    sign_i = ValueSign(Ns, t, R_i, swap_R_i)
    update = sign_i/np.abs(sign_i)

    for i in range(M):
        accept_bit = 0
        R_f, p, delta = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, where_moves)

        swap_order_f, swap_R_f = UpdateOrderSWAP(
            R_f, swap_order_i, swap_R_i, p, delta[0], where_moves[:, 0, :])

        p_swap_order = np.array([swap_order_f[p[0], 0], swap_order_f[p[1], 1]])

        """
        r = 1
        for j in prange(3):
            if j == 2:
                r *= np.conj(RatioStepOneSWAP(Ns, t, swap_R_i,
                             swap_R_f, p_swap_order))
            else:
                r *= RatioStepOne(Ns, t, R_i[:, j], R_f[:, j], p[j])
        """

        r = (np.conj(RatioStepOneSWAP(Ns, t, swap_R_i, swap_R_f, p_swap_order)) *
             RatioStepOne(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
             RatioStepOne(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))

        if np.abs(r) > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            UpdateMoves(R_i, p, where_moves, Ly, boundary, step_size)
            swap_order_i = np.copy(swap_order_f)
            swap_R_i = np.copy(swap_R_f)
            update *= r/np.abs(r)

        acceptance = UpdateResult(
            result, update, i, acceptance, accept_bit)

    return result, acceptance
