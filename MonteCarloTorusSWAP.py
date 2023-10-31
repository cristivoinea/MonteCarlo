from numba import njit, prange
import numpy as np

from LaughlinWavefnSWAP import StepOneAmplitudeLaughlin, StepOneAmplitudeLaughlinSWAP, \
    InitialModLaughlin, InitialSignLaughlin
from FreeFermionsWavefnSWAP import StepOneAmplitudeFreeFermions, StepOneAmplitudeFreeFermionsSWAP, \
    InitialModFreeFermions, InitialSignFreeFermions, UpdateWavefnFreeFermions

from utilities import SaveConfig, SaveResult


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


def RandomConfigSWAP(Ne: np.uint8, Lx: np.float64, Ly: np.float64,
                     region_geometry: str, boundary: np.array) -> np.array:
    """Returns two random configurations of particles, swappable with
    respect to region A.

    Parameters:

    N : number of particles
    Lx, Ly : perpendicular dimensions of the torus
    boundary : dimensionless boundary between regions A and B 
                (the other one is implicit at y=0)

    Output: 

    R : random configuration of particles"""
    R = np.zeros((Ne, 2), dtype=np.complex128)
    R[:, 0] = RandomConfig(Ne, Lx, Ly)
    R[:, 1] = RandomConfig(Ne, Lx, Ly)

    while (np.count_nonzero(InsideRegion(Lx, Ly, R[:, 0], region_geometry, boundary)) !=
           np.count_nonzero(InsideRegion(Lx, Ly, R[:, 1], region_geometry, boundary))):
        R[:, 1] = RandomConfig(Ne, Lx, Ly)

    return R


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
def TestPBC(Ne: np.uint8, Ns: np.uint16, t: np.complex128,
            kCM: np.uint8 = 0, phi_1: np.float64 = 0,
            phi_t: np.float64 = 0,):

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    R0 = RandomConfig(Ne, Lx, Ly)

    p = np.random.randint(0, Ne)
    R_test = np.copy(R0)
    R_test[p] += Lx
    print('t(Lx)*e^(-i phi_1) =', (StepOneAmplitudeLaughlin(Ns, t, R0, R_test, p, kCM, phi_1, phi_t) *
                                   np.exp(-1j*Lx*np.imag(R0[p])/2)*np.exp(-1j*phi_1)))

    R_test = np.copy(R0)
    R_test[p] += Lx*t
    print('t(Lx*tau)*e^(-i phi_t) =', (StepOneAmplitudeLaughlin(Ns, t, R0, R_test, p, kCM, phi_1, phi_t) *
                                       np.exp(1j*Ly*np.real(R0[p])/2)*np.exp(-1j*phi_t)))


# @njit
def UpdateOrderSWAP(R_f: np.array, swap_order_i: np.array, swap_R_i: np.array,
                    p: np.array, delta: np.uint8, where_moves: np.array):
    """
    Updates the order of particles in the swapped copies after one step.

    Parameters:
    R_f : new configuration of particles after step
    swap_order_i : initial order or particles in the swapped copies. positive indices go
                into swap_R[:,1] and negative into swap_R[:,0]
    swap_R_i : array containing the initial ordered positions of particles in the swapped
            copies. swap_R[:,0] containts alpha_1, beta_2
    Output:

    swap_order_f : new order or particles in the swapped copies. positive indices go
                into swap_R[:,1] and negative into swap_R[:,0]
    swap_R_f: array containing the new ordered positions of particles in the swapped
            copies. swap_R[:,0] containts alpha_1, beta_2
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


def LocateParticlesSWAP(Lx: np.float64, Ly: np.float64, R: np.array,
                        region_geometry: str, boundary: np.float64,
                        step_size: np.float64
                        ) -> (np.array, np.array, np.array):
    """
    See description of LocateAndAssignOrderSquareSWAP.

    Parameters:

    Ly : torus dimension along y-axis
    R : particles configuration
    boundary : radius of subregion A (in the center of the torus)
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

    where_moves = np.zeros((R.shape[0], R.shape[1], 4), dtype=np.int8)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            TrackParticleMoves(R, i, j, where_moves, Lx, Ly,
                               region_geometry, boundary, step_size)

    swap_order, swap_R = AssignOrderSWAP(
        R, InsideRegion(Lx, Ly, R, region_geometry, boundary))

    return swap_order, swap_R, where_moves


def AssignOrderSWAP(R: np.array, inside_A: np.array
                    ) -> (np.array, np.array):
    """
    See description of LocateAndAssignOrderSquareSWAP.

    Parameters:

    R : particles configuration in the two copies
    inside_A : array specifying which particles in the two copies
                are in subregion A

    Output:

    swap_order : order or particles in the swapped copies. positive indices go
                into swap_R[:,1] and negative into swap_R[:,0]
    swap_R : array containing the ordered positions of particles in the swapped
            copies. swap_R[:,0] containts alpha_1, beta_2
    """
    swap_order = np.zeros((R.shape[0], R.shape[1]), dtype=np.int8)
    swap_R = np.zeros((R.shape[0], R.shape[1]), dtype=np.complex128)
    i_swap = 1
    j = 0
    for i in range(R.shape[0]):
        if not inside_A[i, 0]:
            swap_order[i, 0] = i_swap
            swap_R[i_swap-1, 1] = R[i, 0]
        else:
            while not inside_A[j, 1]:
                j += 1
            swap_order[j, 1] = i_swap
            swap_R[i_swap-1, 1] = R[j, 1]
            j += 1
        i_swap += 1

    i_swap = -1
    j = 0
    for i in range(R.shape[0]):
        if not inside_A[i, 1]:
            swap_order[i, 1] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[i, 1]
        else:
            while not inside_A[j, 0]:
                j += 1
            swap_order[j, 0] = i_swap
            swap_R[np.abs(i_swap)-1, 0] = R[j, 0]
            j += 1
        i_swap -= 1

    return swap_order, swap_R


def TrackParticleMoves(R: np.array, index: np.uint8, copy: np.uint8,
                       where_moves: np.array,
                       Lx: np.float64, Ly: np.float64,
                       region_geometry: str, boundary: np.float64,
                       step_size: np.float64,):
    """
    For a given new configuration of particles, updates information
    about their positions relative to a region A.

    Parameters:

    R : particles configuration
    index : index of particle
    copy : which system copy we are considering
    where_moves : array (N, 2, 4) specifying where each type of move take
                each particle in each of the copies. the specific move
                increases n_A by the value of where_moves.
                where_move[i,j,k]=m means move k
                (k=0 => +1 ; k=1 => -1; k=2 => +1j; k=3 => -1j) applied on
                particle i in copy j changes n_A in copy j by m.
    Ly : torus dimension along y-axis
    boundary : dimensionful boundary between regions A and B 
                (the other one is implicit at y=0)
    step_size: step size (multiplied by Ly)
    """

    if region_geometry == 'strip':
        y = np.imag(R[index, copy])
        inside_A = (y < boundary)
        inside_A_step_right = inside_A
        inside_A_step_left = inside_A
        inside_A_step_up = ((y + step_size < boundary)
                            | (y + step_size > Ly))
        inside_A_step_down = ((y - step_size < boundary)
                              & (y - step_size > 0))

    elif region_geometry == 'circle':
        r = R[index, copy] - (Lx + 1j*Ly)/2
        inside_A = (np.abs(r) < boundary)
        inside_A_step_right = (np.abs(r + step_size) < boundary)
        inside_A_step_left = (np.abs(r - step_size) < boundary)
        inside_A_step_up = (np.abs(r + 1j*step_size) < boundary)
        inside_A_step_down = (np.abs(r - 1j*step_size) < boundary)

    if inside_A:
        where_moves[index, copy, 0] = -(not inside_A_step_right)
        where_moves[index, copy, 1] = -(not inside_A_step_left)
        where_moves[index, copy, 2] = -(not inside_A_step_up)
        where_moves[index, copy, 3] = -(not inside_A_step_down)
    else:
        where_moves[index, copy, 0] = inside_A_step_right
        where_moves[index, copy, 1] = inside_A_step_left
        where_moves[index, copy, 2] = inside_A_step_up
        where_moves[index, copy, 3] = inside_A_step_down


def InsideRegion(Lx: np.float64, Ly: np.float64,
                 R: np.array, region_geometry: str,
                 boundary: np.float64):
    """"""
    if region_geometry == 'strip':
        y = np.imag(R)
        inside_A = (y < boundary)
    elif region_geometry == 'circle':
        r = R - (Lx + 1j*Ly)/2
        inside_A = (np.abs(r) < boundary)

    return inside_A


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
    delta : contains information about which step is taken
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


"""
@njit
def RunPParticleSector(N: np.uint8, Ns: np.uint16, t: np.complex64,
                       M: np.uint32, step_size: np.float64,
                       boundary_dimensionless: np.array,
                       kCM: np.uint8 = 0,
                       phi_1: np.float64 = 0, phi_t: np.float64 = 0
                       ):

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
    """


def RunPSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
             M: np.uint32, M0: np.uint32, step_size: np.float64,
             region_geometry: str, boundary: np.float64,
             state: str, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0,
             save_config: np.bool_ = True, save_result: np.bool_ = True
             ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    boundary : dimensionless boundary position between regions A and B 
                geometry = 'stripe' -> boundary specifies the height
                                        of the partition (y=0 implicit)
                geometry = 'circle' -> boundary specifies the circle radius
                                        of the partition (centered)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_result : indicate whether the full result vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    result = np.zeros((M), dtype=np.float64)
    R_i = np.vstack((RandomConfig(Ne, Lx, Ly), RandomConfig(Ne, Lx, Ly))).T
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = (np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 0], region_geometry, boundary)) ==
              np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 1], region_geometry, boundary)))

    if state == 'free_fermions':
        wf_i = np.zeros((2, 2), dtype=np.complex128)
        wf_f = np.zeros((2, 2), dtype=np.complex128)
        UpdateWavefnFreeFermions(wf_i, Lx, Ly, R_i)

    for i in range(M):
        accept_bit = 0
        p = np.zeros(2, dtype=np.uint8)
        for j in range(2):
            R_f[:, j], p[j] = StepOne(Lx, Ly, t, step_size, R_i[:, j])

        if state == 'laughlin':
            r = (StepOneAmplitudeLaughlin(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
                 StepOneAmplitudeLaughlin(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))
        elif state == 'free_fermions':
            UpdateWavefnFreeFermions(wf_f, Lx, Ly, R_f)
            r = ((wf_f[0, 0] * wf_f[0, 1]) / (wf_i[0, 0] * wf_i[0, 1]) *
                 np.exp(wf_f[1, 0] + wf_f[1, 1] - wf_i[1, 0] - wf_i[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            update = (np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 0], region_geometry, boundary)) ==
                      np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 1], region_geometry, boundary)))

            if state == 'free_fermions':
                wf_i = np.copy(wf_f)

        result[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           result, R_i, R_i)

    SaveResult(state, 'p', Ne, Ns, Lx, Ly, M0, t, step_size, region_geometry, boundary,
               result, save_result)


def RunModSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
               M: np.uint32, M0: np.uint32, step_size: np.float64,
               region_geometry: str, boundary: np.float64,
               state: str, kCM: np.uint8 = 0,
               phi_1: np.float64 = 0, phi_t: np.float64 = 0,
               save_config: np.bool_ = True, save_result: np.bool_ = True
               ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    boundary : dimensionless boundary position between regions A and B 
                geometry = 'stripe' -> boundary specifies the height
                                        of the partition (y=0 implicit)
                geometry = 'circle' -> boundary specifies the circle radius
                                        of the partition (centered)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_result : indicate whether the full result vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    result = np.zeros((M), dtype=np.float64)
    R_i = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
    swap_order_i, swap_R_i, where_moves = LocateParticlesSWAP(Lx, Ly, R_i, region_geometry,
                                                              boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    if state == 'laughlin':
        update = InitialModLaughlin(Ns, t, R_i, swap_R_i)
    elif state == 'free_fermions':
        update = InitialModFreeFermions(Lx, Ly, R_i, swap_R_i)
        wf_i = np.zeros((2, 2), dtype=np.complex128)
        wf_f = np.zeros((2, 2), dtype=np.complex128)
        UpdateWavefnFreeFermions(wf_i, Lx, Ly, R_i)
        swap_wf_i = np.zeros((2, 2), dtype=np.complex128)
        swap_wf_f = np.zeros((2, 2), dtype=np.complex128)
        UpdateWavefnFreeFermions(swap_wf_i, Lx, Ly, swap_R_i)

    for i in range(M):
        accept_bit = 0
        R_f, p, delta = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, where_moves)

        if state == 'laughlin':
            r = (StepOneAmplitudeLaughlin(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
                 StepOneAmplitudeLaughlin(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))
        elif state == 'free_fermions':
            UpdateWavefnFreeFermions(wf_f, Lx, Ly, R_f)
            r = ((wf_f[0, 0] * wf_f[0, 1]) / (wf_i[0, 0] * wf_i[0, 1]) *
                 np.exp(wf_f[1, 0] + wf_f[1, 1] - wf_i[1, 0] - wf_i[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            swap_order_f, swap_R_f = UpdateOrderSWAP(
                R_f, swap_order_i, swap_R_i, p, delta[0], where_moves[:, 0, :])
            p_swap_order = np.array(
                [swap_order_f[p[0], 0], swap_order_f[p[1], 1]])

            if state == 'laughlin':
                update *= np.abs(StepOneAmplitudeLaughlinSWAP(Ns, t, swap_R_i,
                                                              swap_R_f, p_swap_order) / r)
            elif state == 'free_fermions':
                UpdateWavefnFreeFermions(swap_wf_f, Lx, Ly, swap_R_f)
                swap_r = ((swap_wf_f[0, 0] * swap_wf_f[0, 1]) / (swap_wf_i[0, 0] * swap_wf_i[0, 1]) *
                          np.exp(swap_wf_f[1, 0] + swap_wf_f[1, 1] - swap_wf_i[1, 0] - swap_wf_i[1, 1]))
                update *= np.abs(swap_r / r)

                wf_i = np.copy(wf_f)
                swap_wf_i = np.copy(swap_wf_f)

            R_i = np.copy(R_f)
            for j in range(2):
                TrackParticleMoves(R_i, p[j], j, where_moves, Lx, Ly,
                                   region_geometry, boundary, step_size)
            swap_order_i = np.copy(swap_order_f)
            swap_R_i = np.copy(swap_R_f)

        result[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           result, R_i, swap_order_i)

    SaveResult(state, 'mod', Ne, Ns, Lx, Ly, M0, t, step_size, region_geometry, boundary,
               result, save_result)


def RunSignSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                M: np.uint32, M0: np.uint32, step_size: np.float64,
                region_geometry: str, boundary: np.float64,
                state: str, kCM: np.uint8 = 0,
                phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                save_config: np.bool_ = True, save_result: np.bool_ = True
                ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    boundary : dimensionless boundary position between regions A and B 
                geometry = 'stripe' -> boundary specifies the height
                                        of the partition (y=0 implicit)
                geometry = 'circle' -> boundary specifies the circle radius
                                        of the partition (centered)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_result : indicate whether the full result vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    result = np.zeros((M), dtype=np.complex128)
    R_i = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
    swap_order_i, swap_R_i, where_moves = LocateParticlesSWAP(Lx, Ly, R_i, region_geometry,
                                                              boundary, step_size)
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    if state == 'laughlin':
        update = InitialSignLaughlin(Ns, t, R_i, swap_R_i)
    elif state == 'free_fermions':
        update = InitialSignFreeFermions(Lx, Ly, R_i, swap_R_i)
        wf_i = np.zeros((2, 2), dtype=np.complex128)
        wf_f = np.zeros((2, 2), dtype=np.complex128)
        UpdateWavefnFreeFermions(wf_i, Lx, Ly, R_i)
        swap_wf_i = np.zeros((2, 2), dtype=np.complex128)
        swap_wf_f = np.zeros((2, 2), dtype=np.complex128)
        UpdateWavefnFreeFermions(swap_wf_i, Lx, Ly, swap_R_i)

    for i in range(M):
        accept_bit = 0
        R_f, p, delta = StepOneSWAPRandom(
            Lx, Ly, t, step_size, R_i, where_moves)

        swap_order_f, swap_R_f = UpdateOrderSWAP(
            R_f, swap_order_i, swap_R_i, p, delta[0], where_moves[:, 0, :])

        p_swap_order = np.array([swap_order_f[p[0], 0], swap_order_f[p[1], 1]])

        if state == 'laughlin':
            r = (np.conj(StepOneAmplitudeLaughlinSWAP(Ns, t, swap_R_i, swap_R_f, p_swap_order)) *
                 StepOneAmplitudeLaughlin(Ns, t, R_i[:, 0], R_f[:, 0], p[0]) *
                 StepOneAmplitudeLaughlin(Ns, t, R_i[:, 1], R_f[:, 1], p[1]))
        elif state == 'free_fermions':
            UpdateWavefnFreeFermions(wf_f, Lx, Ly, R_f)
            UpdateWavefnFreeFermions(swap_wf_f, Lx, Ly, swap_R_f)
            r = ((wf_f[0, 0] * wf_f[0, 1] * swap_wf_i[0, 0] * swap_wf_i[0, 1]) /
                 (wf_i[0, 0] * wf_i[0, 1] * swap_wf_f[0, 0] * swap_wf_f[0, 1]) *
                 np.exp(swap_wf_f[1, 0] + swap_wf_f[1, 1] - swap_wf_i[1, 0] - swap_wf_i[1, 1]
                        + wf_f[1, 0] + wf_f[1, 1] - wf_i[1, 0] - wf_i[1, 1]))

        if np.abs(r) > np.random.random():
            accept_bit = 1
            R_i = np.copy(R_f)
            for j in range(2):
                TrackParticleMoves(R_i, p[j], j, where_moves, Lx, Ly,
                                   region_geometry, boundary, step_size)
            swap_order_i = np.copy(swap_order_f)
            swap_R_i = np.copy(swap_R_f)
            update *= r/np.abs(r)

            if state == 'free_fermions':
                wf_i = np.copy(wf_f)
                swap_wf_i = np.copy(swap_wf_f)

        result[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           result, R_i, swap_order_i)

    SaveResult(state, 'sign', Ne, Ns, Lx, Ly, M0, t, step_size, region_geometry, boundary,
               result, save_result)
