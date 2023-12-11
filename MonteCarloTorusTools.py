from numba import njit
import numpy as np


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
