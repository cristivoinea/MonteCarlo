from numba import njit, prange
import numpy as np

from WavefnLaughlin import StepOneAmplitudeLaughlin, StepOneAmplitudeLaughlinSWAP, \
    InitialModLaughlin, InitialSignLaughlin
from WavefnFreeFermions import StepOneAmplitudeFreeFermions, StepOneAmplitudeFreeFermionsSWAP, \
    InitialModFreeFermions, InitialSignFreeFermions, UpdateWavefnFreeFermions
from WavefnCFL import InitialWavefnCFL, InitialWavefnSwapCFL, \
    InitialModCFL, InitialSignCFL, TmpWavefnCFL, TmpWavefnSwapCFL, StepOneAmplitudeSwapCFL, \
    ResetJastrowsCFL, StepOneAmplitudeCFL, UpdateJastrowsCFL

from utilities import SaveConfig, SaveResults, fermi_sea_kx, fermi_sea_ky, LoadRun


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


@njit
def UpdateOrderSwap(to_swap_new: np.array, from_swap_new: np.array,
                    p: np.array, nbr_A_changes: bool):
    """
    Updates the order of particles in the swapped copies after one step.

    Parameters:
    to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:,0]
                and [Ne, 2*Ne) go into coords_swap[:,1]
    from_Swap : order of swapped particles in the original copies. [0,Ne) go into coords[:,0]
                and [Ne, 2*Ne) go into coords[:,1]
    p : array containing indices of moved particles in the two initial copies
    nbr_A_changes : indicates whether the number of particles in the subregion changed
    """

    Ne = from_swap_new.shape[0]

    if nbr_A_changes:
        i0 = to_swap_new[p[0], 0]
        i1 = to_swap_new[p[1], 1]

        to_swap_new[p[0], 0] = i1
        to_swap_new[p[1], 1] = i0

        from_swap_new[i0 % Ne, i0 // Ne] = p[1] + Ne
        from_swap_new[i1 % Ne, i1 // Ne] = p[0]


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


def AssignOrderToSwap(inside_A: np.array
                      ) -> (np.array):  # , np.array):
    """
    Given an array telling us whether each particle is inside the subregion A, 
    this method returns two arrays: 
    one containing the conversion from original -> swap indices and its
    inverse.

    Parameters:
    inside_A : array specifying which particles in the two copies
                are in subregion A

    Output:
    to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:,0]
                        and [Ne, 2*Ne) go into coords_swap[:,1]
    from_swap : order of swapped particles in the original copies. [0,Ne) go into coords[:,0]
                        and [Ne, 2*Ne) go into coords[:,1]
    """
    Ne = inside_A.shape[0]
    to_swap = np.zeros(
        (inside_A.shape[0], inside_A.shape[1]), dtype=np.uint8)
    # from_swap = np.zeros(
    #    (inside_A.shape[0], inside_A.shape[1]), dtype=np.uint8)

    i_swap = 0
    j = 0
    for i in range(Ne):
        if not inside_A[i, 1]:
            to_swap[i, 1] = i_swap
            # from_swap[i_swap, 0] = i + Ne
        else:
            while not inside_A[j, 0]:
                j += 1
            to_swap[j, 0] = i_swap
            # from_swap[i_swap, 0] = j
            j += 1
        i_swap += 1

    i_swap = 0
    j = 0
    for i in range(Ne):
        if not inside_A[i, 0]:
            to_swap[i, 0] = i_swap + Ne
            # from_swap[i_swap, 1] = i
        else:
            while not inside_A[j, 1]:
                j += 1
            to_swap[j, 1] = i_swap + Ne
            # from_swap[i_swap, 1] = j + Ne
            j += 1
        i_swap += 1

    return to_swap  # , from_swap


def OrderFromSwap(to_swap: np.array
                  ) -> np.array:
    """
    Given ordering of particles from the original copies to the swapped copies,
    returns the inverse mapping.

    Parameters:
    to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:,0]
                        and [Ne, 2*Ne) go into coords_swap[:,1]

    Output:
    from_swap : order of swapped particles in the original copies. [0,Ne) go into coords[:,0]
                        and [Ne, 2*Ne) go into coords[:,1]
    """
    Ne = to_swap.shape[0]
    from_swap = np.zeros(
        (to_swap.shape[0], to_swap.shape[1]), dtype=np.uint8)

    for cp in range(2):
        for i in range(Ne):
            from_swap[to_swap[i, cp] % Ne, to_swap[i, cp]//Ne] = i + cp*Ne

    return from_swap


def CheckCrossBoundary(coord_initial: np.complex128, coord_final: np.complex128,
                       Lx: np.float64, Ly: np.float64,
                       region_geometry: str, boundary: np.float64):
    """
    For a given new configuration of particles, updates information
    about their positions relative to a region A.

    Parameters:

    coords_initial/final : initial and final coordinates of the moving particle
    Lx, Ly : torus dimensions along the two perpendicular dimensions
    boundary : dimensionful boundary between regions A and B 
                (the other one is implicit at y=0)
    """

    if region_geometry == 'strip':
        inside_A_initial = (np.imag(coord_initial) < boundary)
        inside_A_final = (np.imag(coord_final) < boundary)

    elif region_geometry == 'circle':
        inside_A_initial = (np.abs(coord_initial - (Lx + 1j*Ly)/2) < boundary)
        inside_A_final = (np.abs(coord_final - (Lx + 1j*Ly)/2) < boundary)

    return inside_A_final - inside_A_initial


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


@njit
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


def StepOneCircle(Lx: np.float64, t: np.complex128,
                  step_size: np.float64, coords_tmp: np.array,
                  p: np.array):
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle in each copy, ensuring that
    the copies are swappable with respect to region A.
    """

    Ne = coords_tmp.shape[0]
    Ly = np.imag(t)*Lx
    # np.exp(1j*2*np.pi*np.random.random())
    p[0], p[1] = np.random.randint(0, Ne, 2)

    new_0 = coords_tmp[p[0], 0]
    new_1 = coords_tmp[p[1], 1]
    coords_tmp[p[0], 0] = PBC(Lx, Ly, t, new_0 +
                              step_size*np.random.choice(np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                                                   (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])))
    coords_tmp[p[1], 1] = PBC(Lx, Ly, t, new_1 +
                              step_size*np.random.choice(np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                                                   (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])))


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


@njit
def StepOneSwap(Lx: np.float64, t: np.complex128,
                step_size: np.float64, coords_new: np.array,
                p: np.array, region_geometry: str, boundary: np.float64
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

    Ne = coords_new.shape[0]
    Ly = np.imag(t)*Lx

    valid = False
    while not valid:
        p[0], p[1] = np.random.randint(0, Ne, 2)
        inside_A_current = InsideRegion(Lx, Ly, np.array([coords_new[p[0], 0], coords_new[p[1], 1]]),
                                        region_geometry, boundary)
        new_0 = PBC(Lx, Ly, t, coords_new[p[0], 0] +
                    step_size*np.random.choice(np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                                         (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])))
        new_1 = PBC(Lx, Ly, t, coords_new[p[1], 1] +
                    step_size*np.random.choice(np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                                         (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])))
        inside_A_new = InsideRegion(Lx, Ly, np.array([new_0, new_1]),
                                    region_geometry, boundary)
        if (int(inside_A_current[0]) - int(inside_A_new[0])) == (int(inside_A_current[1]) - int(inside_A_new[1])):
            valid = True
            nbr_A_changes = (inside_A_current[0] ^ inside_A_new[0])

    coords_new[p[0], 0] = new_0
    coords_new[p[1], 1] = new_1

    return nbr_A_changes


def RunPSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                         M: np.uint32, M0: np.uint32, step_size: np.float64,
                         region_geometry: str, region_size: np.float64,
                         save_config: np.bool_ = True, save_results: np.bool_ = True,
                         start_acceptance: np.float64 = -1
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
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, _, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                       step_size/Lx, 'free_fermions', 'p')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")
    else:
        acceptance: np.float64 = 0
        coords = np.vstack(
            (RandomConfig(Ne, Lx, Ly), RandomConfig(Ne, Lx, Ly))).T
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    coords_tmp = np.copy(coords)

    update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
              np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        StepOneCircle(Lx, t, step_size, coords_tmp, moved_particles)

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1]) / (slogdet[0, 0] * slogdet[0, 1]) *
             np.exp(slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1

            np.copyto(coords, coords_tmp)
            np.copyto(slogdet, slogdet_tmp)

            update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
                      np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, coords)

    SaveResults('free_fermions', 'p', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunPSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                M: np.uint32, M0: np.uint32, step_size: np.float64,
                region_geometry: str, region_size: np.float64,
                kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                save_config: np.bool_ = True, save_results: np.bool_ = True,
                start_acceptance: np.float64 = 0
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
    region_size : coverage of subregion A (percentage of total area)
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    load_prev : file name containing details of a previous run. has format
                M acceptance
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Ks = (fermi_sea_kx[Ne]*2*np.pi/Lx + 1j*fermi_sea_ky[Ne]*2*np.pi/Ly)

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, _, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                       step_size/Lx, 'cfl', 'p')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")
    else:
        acceptance: np.float64 = 0
        coords = np.vstack(
            (RandomConfig(Ne, Lx, Ly), RandomConfig(Ne, Lx, Ly))).T
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    coords_tmp = np.copy(coords)

    JK_slogdet = np.zeros((2, 2), dtype=np.complex128)
    jastrows = np.ones((Ne, Ne, Ne, 2), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 2), dtype=np.complex128)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[..., cp],
                                                 JK_matrix[..., cp])

    coords_tmp = np.copy(coords)
    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)
    JK_slogdet_tmp = np.copy(JK_slogdet)

    update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
              np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))

    for i in range(prev_iter, prev_iter+M):
        moved_particles = np.zeros(2, dtype=np.uint8)
        step_amplitude = 1
        StepOneCircle(Lx, t, step_size, coords_tmp, moved_particles)
        # for j in range(2):
        # coords_new[:, j], moved_particles[j] = StepOne(
        # Lx, Ly, t, step_size, coords_current[:, j])
        for cp in range(2):
            JK_slogdet_tmp[0, cp], \
                JK_slogdet_tmp[1, cp] = TmpWavefnCFL(t, Lx, coords_tmp[:, cp], Ks,
                                                     jastrows[..., cp],
                                                     jastrows_tmp[..., cp],
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])
            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  JK_slogdet[:, cp],
                                                  JK_slogdet_tmp[:, cp],
                                                  moved_particles[cp],
                                                  Ks, kCM, phi_1, phi_t)
        if np.abs(step_amplitude)**2 > np.random.random():
            accept_bit = 1
            for cp in range(2):
                coords[moved_particles[cp],
                       cp] = coords_tmp[moved_particles[cp], cp]
                ResetJastrowsCFL(jastrows_tmp[..., cp], jastrows[..., cp],
                                 JK_matrix_tmp[..., cp], JK_matrix[...,  cp], moved_particles[cp])

            np.copyto(JK_slogdet, JK_slogdet_tmp)
            update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
                      np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))

        else:
            accept_bit = 0
            for cp in range(2):
                coords_tmp[moved_particles[cp],
                           cp] = coords[moved_particles[cp], cp]
                ResetJastrowsCFL(jastrows[..., cp], jastrows_tmp[..., cp],
                                 JK_matrix[..., cp], JK_matrix_tmp[...,  cp], moved_particles[cp])

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('cfl', 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, coords)

    SaveResults('cfl', 'p', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunPSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
             M: np.uint32, M0: np.uint32, step_size: np.float64,
             region_geometry: str, boundary: np.float64,
             state: str, kCM: np.uint8 = 0,
             phi_1: np.float64 = 0, phi_t: np.float64 = 0,
             save_config: np.bool_ = True, save_results: np.bool_ = True
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
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    R_i = np.vstack((RandomConfig(Ne, Lx, Ly), RandomConfig(Ne, Lx, Ly))).T
    R_f = np.zeros((R_i.shape[0], R_i.shape[1]), dtype=np.complex128)

    update = (np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 0], region_geometry, boundary)) ==
              np.count_nonzero(InsideRegion(Lx, Ly, R_i[:, 1], region_geometry, boundary)))

    results = np.zeros((M), dtype=np.float64)

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

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           results, R_i, R_i)

    SaveResults(state, 'p', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, boundary)


def RunModSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                           M: np.uint32, M0: np.uint32, step_size: np.float64,
                           region_geometry: str, region_size: np.float64,
                           save_config: np.bool_ = True, save_results: np.bool_ = True,
                           start_acceptance: np.float64 = -1
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
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'free_fermions', 'mod')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]

    coords_tmp = np.copy(coords)
    coords_swap_tmp = np.copy(coords_swap)
    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    update = InitialModFreeFermions(coords, coords_swap, Kxs, Kys)
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    UpdateWavefnFreeFermions(slogdet_swap, coords_swap, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1]) / (slogdet[0, 0] * slogdet[0, 1]) *
             np.exp(slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                            moved_particles, nbr_A_changes)
            for cp in range(2):
                for j in range(Ne):
                    coords_swap_tmp[j, cp] = coords_tmp[from_swap_tmp[j, cp] %
                                                        Ne, from_swap_tmp[j, cp] // Ne]

            UpdateWavefnFreeFermions(
                slogdet_swap_tmp, coords_swap_tmp, Kxs, Kys)

            swap_r = ((slogdet_swap_tmp[0, 0] * slogdet_swap_tmp[0, 1]) /
                      (slogdet_swap[0, 0] * slogdet_swap[0, 1]) *
                      np.exp(slogdet_swap_tmp[1, 0] + slogdet_swap_tmp[1, 1] -
                             slogdet_swap[1, 0] - slogdet_swap[1, 1]))
            update *= np.abs(swap_r / r)

            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            np.copyto(coords_swap, coords_swap_tmp)

            np.copyto(slogdet, slogdet_tmp)
            np.copyto(slogdet_swap, slogdet_swap_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            np.copyto(coords_swap_tmp, coords_swap)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('free_fermions', 'mod', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunModSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                  M: np.uint32, M0: np.uint32, step_size: np.float64,
                  region_geometry: str, region_size: np.float64,
                  kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                  save_config: np.bool_ = True, save_results: np.bool_ = True,
                  start_acceptance: np.float64 = -1
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
    region_size : coverage of subregion A (percentage of total area)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Ks = (fermi_sea_kx[Ne] + 1j*fermi_sea_ky[Ne])*2*np.pi/Lx

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'cfl', 'mod')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    coords_tmp = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)

    JK_slogdet = np.zeros((2, 4), dtype=np.complex128)
    JK_slogdet_tmp = np.zeros((2, 4), dtype=np.complex128)
    jastrows = np.ones((2*Ne, 2*Ne, Ne), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[cp*Ne:(cp+1)*Ne,
                                                          cp*Ne:(cp+1)*Ne, :],
                                                 JK_matrix[:, :, cp])

    JK_slogdet[0, 2:4], \
        JK_slogdet[1, 2:4] = InitialWavefnSwapCFL(t, Lx, coords, Ks, jastrows,
                                                  JK_matrix[:, :, 2:], from_swap)
    update = InitialModCFL(Ne, Ns, t, coords, Ks, JK_slogdet,
                           from_swap, kCM, phi_1, phi_t)

    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)

    for i in range(prev_iter, prev_iter+M):

        step_amplitude = 1
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)
        for cp in range(2):
            JK_slogdet_tmp[0, cp], \
                JK_slogdet_tmp[1, cp] = TmpWavefnCFL(t, Lx, coords_tmp[:, cp], Ks,
                                                     jastrows[cp*Ne:(cp+1)*Ne,
                                                              cp*Ne:(cp+1)*Ne, :],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, :],
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])
            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  JK_slogdet[:, cp],
                                                  JK_slogdet_tmp[:, cp],
                                                  moved_particles[cp],
                                                  Ks, kCM, phi_1, phi_t)

        if step_amplitude*np.conj(step_amplitude) > np.random.random():
            accept_bit = 1
            UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                            moved_particles, nbr_A_changes)
            JK_slogdet_tmp[0, 2:4], \
                JK_slogdet_tmp[1, 2:4] = TmpWavefnSwapCFL(t, Lx, coords_tmp, Ks,
                                                          jastrows, jastrows_tmp,
                                                          JK_matrix_tmp[:, :,
                                                                        2:4], moved_particles,
                                                          to_swap_tmp, from_swap_tmp)

            step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                          JK_slogdet[:, 2:4],
                                                          JK_slogdet_tmp[:, 2:4],
                                                          moved_particles,
                                                          from_swap, from_swap_tmp,
                                                          kCM, phi_1, phi_t)

            update *= np.abs(step_amplitude_swap / step_amplitude)

            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            ResetJastrowsCFL(jastrows_tmp, jastrows,
                             JK_matrix_tmp[..., 0], JK_matrix[...,  0], moved_particles[0])
            ResetJastrowsCFL(jastrows_tmp, jastrows,
                             JK_matrix_tmp[..., 1], JK_matrix[...,  1], Ne+moved_particles[1])
            np.copyto(JK_matrix[..., 2], JK_matrix_tmp[..., 2])
            np.copyto(JK_matrix[..., 3], JK_matrix_tmp[..., 3])
            np.copyto(JK_slogdet, JK_slogdet_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            ResetJastrowsCFL(jastrows, jastrows_tmp,
                             JK_matrix[..., 0], JK_matrix_tmp[...,  0], moved_particles[0])
            ResetJastrowsCFL(jastrows, jastrows_tmp,
                             JK_matrix[..., 1], JK_matrix_tmp[...,  1], Ne+moved_particles[1])
            np.copyto(JK_matrix_tmp[..., 2], JK_matrix[..., 2])
            np.copyto(JK_matrix_tmp[..., 3], JK_matrix[..., 3])

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('cfl', 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('cfl', 'mod', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunModSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
               M: np.uint32, M0: np.uint32, step_size: np.float64,
               region_geometry: str, boundary: np.float64,
               state: str, kCM: np.uint8 = 0,
               phi_1: np.float64 = 0, phi_t: np.float64 = 0,
               save_config: np.bool_ = True, save_results: np.bool_ = True
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
    region_size : coverage of subregion A (percentage of total area)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    results = np.zeros((M), dtype=np.float64)
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

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           results, R_i, swap_order_i)

    SaveResults(state, 'mod', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, boundary)


def RunSignSWAP(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                M: np.uint32, M0: np.uint32, step_size: np.float64,
                region_geometry: str, boundary: np.float64,
                state: str, kCM: np.uint8 = 0,
                phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                save_config: np.bool_ = True, save_results: np.bool_ = True
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
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)
    # TestPBC(Ne, Ns, t, kCM, phi_1, phi_t)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary *= Ly

    results = np.zeros((M), dtype=np.complex128)
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

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig(state, 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, boundary,
                           results, R_i, swap_order_i)

    SaveResults(state, 'sign', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, boundary)


def RunSignSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                            M: np.uint32, M0: np.uint32, step_size: np.float64,
                            region_geometry: str, region_size: np.float64,
                            save_config: np.bool_ = True, save_results: np.bool_ = True,
                            start_acceptance: np.float64 = -1
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
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'free_fermions', 'sign')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.complex128)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]

    coords_tmp = np.copy(coords)
    coords_swap_tmp = np.copy(coords_swap)
    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    update = InitialSignFreeFermions(coords, coords_swap, Kxs, Kys)
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    UpdateWavefnFreeFermions(slogdet_swap, coords_swap, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)

        UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                        moved_particles, nbr_A_changes)
        for cp in range(2):
            for j in range(Ne):
                coords_swap_tmp[j, cp] = coords_tmp[from_swap_tmp[j, cp] %
                                                    Ne, from_swap_tmp[j, cp] // Ne]

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        UpdateWavefnFreeFermions(slogdet_swap_tmp, coords_swap_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1] * slogdet_swap[0, 0] * slogdet_swap[0, 1]) /
             (slogdet[0, 0] * slogdet[0, 1] * slogdet_swap_tmp[0, 0] * slogdet_swap_tmp[0, 1]) *
             np.exp(slogdet_swap_tmp[1, 0] + slogdet_swap_tmp[1, 1] - slogdet_swap[1, 0] - slogdet_swap[1, 1]
                    + slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r) > np.random.random():
            accept_bit = 1
            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            np.copyto(coords_swap, coords_swap_tmp)
            update *= r/np.abs(r)

            np.copyto(slogdet, slogdet_tmp)
            np.copyto(slogdet_swap, slogdet_swap_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            np.copyto(coords_swap_tmp, coords_swap)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('free_fermions', 'sign', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunSignSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                   M: np.uint32, M0: np.uint32, step_size: np.float64,
                   region_geometry: str, region_size: np.float64,
                   kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                   save_config: np.bool_ = True, save_results: np.bool_ = True,
                   start_acceptance: np.float64 = -1
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
    region_size : coverage of subregion A (percentage of total area)
    state : specifies the wavefunction of the system 
            ('laughlin','free_fermions','cfl')
    kCM, phi_1, phi_t : select the CM momentum and quasi-PBC manifold
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Ks = (fermi_sea_kx[Ne] + 1j*fermi_sea_ky[Ne])*2*np.pi/Lx

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'cfl', 'sign')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.complex128)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    coords_tmp = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)

    JK_slogdet = np.zeros((2, 4), dtype=np.complex128)
    JK_slogdet_tmp = np.zeros((2, 4), dtype=np.complex128)
    jastrows = np.ones((2*Ne, 2*Ne, Ne), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[cp*Ne:(cp+1)*Ne,
                                                          cp*Ne:(cp+1)*Ne, :],
                                                 JK_matrix[:, :, cp])
    JK_slogdet[0, 2:4], \
        JK_slogdet[1, 2:4] = InitialWavefnSwapCFL(t, Lx, coords, Ks, jastrows,
                                                  JK_matrix[:, :, 2:], from_swap)
    update = InitialSignCFL(Ne, Ns, t, coords, Ks, JK_slogdet[0, :],
                            from_swap, kCM, phi_1, phi_t)

    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)

    for i in range(prev_iter, prev_iter+M):
        step_amplitude = 1
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)
        for cp in range(2):

            JK_slogdet_tmp[0, cp], \
                JK_slogdet_tmp[1, cp] = TmpWavefnCFL(t, Lx, coords_tmp[:, cp], Ks,
                                                     jastrows[cp*Ne:(cp+1)*Ne,
                                                              cp*Ne:(cp+1)*Ne, :],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, :],
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])

            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  JK_slogdet[:, cp],
                                                  JK_slogdet_tmp[:, cp],
                                                  moved_particles[cp],
                                                  Ks, kCM, phi_1, phi_t)

        UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                        moved_particles, nbr_A_changes)

        JK_slogdet_tmp[0, 2:4], \
            JK_slogdet_tmp[1, 2:4] = TmpWavefnSwapCFL(t, Lx, coords_tmp, Ks,
                                                      jastrows, jastrows_tmp,
                                                      JK_matrix_tmp[...,
                                                                    2:4], moved_particles,
                                                      to_swap_tmp, from_swap_tmp)

        step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                      JK_slogdet[:, 2:4],
                                                      JK_slogdet_tmp[:, 2:4],
                                                      moved_particles,
                                                      from_swap, from_swap_tmp,
                                                      kCM, phi_1, phi_t)
        amplitude = step_amplitude*np.conj(step_amplitude_swap)

        if np.abs(amplitude) > np.random.random():
            accept_bit = 1

            update *= amplitude / np.abs(amplitude)
            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            ResetJastrowsCFL(jastrows_tmp, jastrows,
                             JK_matrix_tmp[..., 0], JK_matrix[...,  0], moved_particles[0])
            ResetJastrowsCFL(jastrows_tmp, jastrows,
                             JK_matrix_tmp[..., 1], JK_matrix[...,  1], Ne+moved_particles[1])
            np.copyto(JK_matrix[..., 2], JK_matrix_tmp[..., 2])
            np.copyto(JK_matrix[..., 3], JK_matrix_tmp[..., 3])
            np.copyto(JK_slogdet, JK_slogdet_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            ResetJastrowsCFL(jastrows, jastrows_tmp,
                             JK_matrix[..., 0], JK_matrix_tmp[...,  0], moved_particles[0])
            ResetJastrowsCFL(jastrows, jastrows_tmp,
                             JK_matrix[..., 1], JK_matrix_tmp[...,  1], Ne+moved_particles[1])
            np.copyto(JK_matrix_tmp[..., 2], JK_matrix[..., 2])
            np.copyto(JK_matrix_tmp[..., 3], JK_matrix[..., 3])

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('cfl', 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('cfl', 'sign', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)
