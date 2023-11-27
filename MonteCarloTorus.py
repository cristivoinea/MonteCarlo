from numba import njit, jit, prange
from numba.typed import List
import numpy as np

from WavefnLaughlin import ThetaFunction, InitialJastrowsLaughlin, \
    UpdateJastrowsLaughlin, StepOneAmplitudeLaughlin, StepOneAmplitudeLaughlinOld
from WavefnCFL import StepOneAmplitudeCFL, ResetJastrowsCFL
from WavefnCFL import InitialWavefnCFL, TmpWavefnCFL
from utilities import SaveConfig, SaveResult, fermi_sea_kx, fermi_sea_ky


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


@njit
def CoulombRealSpaceNoBackground(N: np.uint8, Lx: np.float64, tau: np.complex128,
                                 R: np.array, coulomb_cutoff: np.uint8):
    energy = 0
    for i in range(N):
        for j in range(i+1, N):
            for m in range(coulomb_cutoff):
                for n in range(coulomb_cutoff):
                    energy += 1/(np.abs(R[i]-R[j]+m*Lx+n*Lx*tau))

    return energy


@njit(parallel=True)
def CoulombRealSpace(Lx: np.float64, Ly: np.float64,
                     coords: np.array, coulomb_cutoff: np.uint8 = 20):
    energy = 0
    Ne = coords.size
    # get meshgrid
    # add half of the plane plus half of the real line
    qx = np.zeros((coulomb_cutoff, 2*coulomb_cutoff+1), dtype=np.float64)
    qy = np.zeros((coulomb_cutoff, 2*coulomb_cutoff+1), dtype=np.float64)
    for i in range(coulomb_cutoff):
        for j in range(2*coulomb_cutoff+1):
            qx[i, j] = (j - coulomb_cutoff)*2*np.pi/Lx
            qy[i, j] = (coulomb_cutoff - i)*2*np.pi/Ly
    qx_line = np.arange(1, coulomb_cutoff+1, 1)*2*np.pi/Lx
    qy_line = np.zeros((coulomb_cutoff), dtype=np.float64)

    for i in prange(Ne):
        for j in range(i+1, Ne):
            x = np.real(coords[i]-coords[j])
            y = np.imag(coords[i]-coords[j])

            energy += np.sum(2*np.cos(qx*x + qy*y)/np.sqrt(qx**2+qy**2)) + \
                np.sum(2*np.cos(qx_line*x + qy_line*y) /
                       np.sqrt(qx_line**2+qy_line**2))

    return energy*2*np.pi/(Lx*Ly)


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
    w = z
    if np.imag(w) > Ly:
        w -= Lx*t
    elif np.imag(w) < 0:
        w += Lx*t
    if np.real(w) > Lx:
        w -= Lx
    elif np.real(w) < 0:
        w += Lx

    return w


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
def StepOne(Lx: np.float64, Ly: np.float64, t: np.complex128,
            step_size: np.float64, coords_current: np.array
            ) -> (np.array, np.uint8):
    """Provides a new Monte Carlo configuration by updating
    the coordinates of one particle.

    Parameters:
    Lx, Ly : perpendicular dimensions of the torus
    t : torus complex aspect ratio
    step_size : step size for each particle
    coords_current : initial position of all particles

    Output:
    coords_new : final position of all particles
    moved_particle : index of particles that moves
    """

    coords_new = np.copy(coords_current)
    moved_particle = np.random.randint(0, coords_current.size)
    delta = step_size * np.random.choice(np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                                   (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2]))
    coords_new[moved_particle] = PBC(
        Lx, Ly, t, coords_current[moved_particle]+delta)

    return coords_new, moved_particle


@njit
def StepOneOld(N: np.uint8, Ns: np.uint16, t: np.complex128, R0: np.array,
               step_size: np.float64, kCM: np.uint8,
               phi_1: np.float64, phi_t: np.float64):
    """Performs one Monte Carlo step by attempting to update the
    coordinates of one particle.
    Particle coordinates are always between (0,Lx) and (0,Ly)."""
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    p = np.random.randint(0, N)
    delta = step_size * np.exp(1j*2*np.pi*np.random.rand())
    # enforce periodic boundary conditions
    z_p = PBC(Lx, Ly, t, R0[p]+delta)
    # step
    R1 = np.copy(R0)
    R1[p] = z_p

    # check the weights of the two coordinates
    r = np.abs(StepProbabilityLaughlin(
        N, Ns, t, R0, R1, p, kCM, phi_1, phi_t))**2

    # perform the step according to the distribution
    eta = np.random.random()
    if r > eta:
        return R1, 1
    else:
        return R0, 0


@njit
def StepAll(N: np.uint8, Lx: np.float64, Ly: np.float64, t: np.complex128,
            R0: np.array, step_size: np.float64,
            phi_1: np.float64, phi_t: np.float64):
    """Performs one Monte Carlo step by attempting to update the
    coordinates of all particles.
    Particle coordinates are always between (0,Lx) and (0,Ly)."""
    delta = step_size * np.exp(1j*2*np.pi*np.random.rand(R0.size))
    # enforce periodic boundary conditions
    R1, phase = PBC(Lx, Ly, t, R0+delta, phi_1, phi_t)
    # step

    return R1, phase


def RunCoulombEnergyLaughlin(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                             M: np.uint32, M0: np.uint32, step_size: np.float64,
                             kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                             W: np.float64 = -1.95013246, coulomb_cutoff: np.uint8 = 20,
                             save_config: np.bool_ = True, save_result: np.bool_ = True,):
    """
    Parameters:
    Ne : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    W : value of unnormalized self-interaction energy
    coulomb_cutoff : cutoff for the infinite sum approx. of 
                    Coulomb energy"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx

    coords_current = RandomConfig(Ne, Lx, Ly)
    jastrows_current = InitialJastrowsLaughlin(coords_current, t, Lx)

    jastrows_new = np.zeros((Ne, Ne), dtype=np.complex128)

    update = CoulombRealSpace(
        Lx, Ly, coords_current, coulomb_cutoff)

    result = np.zeros(M, dtype=np.float64)

    for i in range(M):
        accept_bit = 0
        coords_new, moved_particle = StepOne(
            Lx, Ly, t, step_size, coords_current)
        np.copyto(jastrows_new, jastrows_current)

        UpdateJastrowsLaughlin(jastrows_new, coords_new, t, Lx, moved_particle)
        step_amplitude = StepOneAmplitudeLaughlin(Ns, t, coords_current, coords_new,
                                                  jastrows_current, jastrows_new,
                                                  moved_particle, kCM, phi_1, phi_t)

        if np.abs(step_amplitude)**2 > np.random.random():
            accept_bit = 1
            coords_current = np.copy(coords_new)
            jastrows_current = np.copy(jastrows_new)

            update = CoulombRealSpace(
                Lx, Ly, coords_current, coulomb_cutoff)

        result[i] = update

        acceptance = (acceptance*i + accept_bit)/(i + 1)
        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('laughlin', 'coulomb', Ne, Ns, Lx, Ly, t, step_size,
                           result, coords_current, coords_current)

    result = result/Ne + W/np.sqrt(Lx*Ly)

    SaveResult('laughlin', 'coulomb', Ne, Ns, Lx, Ly, M0, t, step_size,
               result, save_result)


def RunCoulombEnergyCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                        M: np.uint32, M0: np.uint32, step_size: np.float64,
                        kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                        W: np.float64 = -1.95013246, coulomb_cutoff: np.uint8 = 20,
                        save_config: np.bool_ = True, save_result: np.bool_ = True,):
    """
    Parameters:
    Ne : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    W : value of unnormalized self-interaction energy
    coulomb_cutoff : cutoff for the infinite sum approx. of 
                    Coulomb energy"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    Ks = (fermi_sea_kx[Ne]*2*np.pi/Lx + 1j*fermi_sea_ky[Ne]*2*np.pi/Ly)

    coords = RandomConfig(Ne, Lx, Ly)

    jastrows = np.ones((Ne, Ne, Ne), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne), dtype=np.complex128)
    JK_slogdet = np.zeros(2, dtype=np.complex128)
    JK_slogdet[0], \
        JK_slogdet[1] = InitialWavefnCFL(t, Lx, coords, Ks,
                                         jastrows, JK_matrix)
    JK_slogdet_tmp = np.zeros(2, dtype=np.complex128)

    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)

    update = CoulombRealSpace(
        Lx, Ly, coords, coulomb_cutoff)

    result = np.zeros(M, dtype=np.float64)

    for i in range(M):
        coords_tmp, moved_particle = StepOne(
            Lx, Ly, t, step_size, coords)

        JK_slogdet_tmp[0], JK_slogdet_tmp[1] = TmpWavefnCFL(t, Lx, coords_tmp, Ks,
                                                            jastrows, jastrows_tmp,
                                                            JK_matrix_tmp, moved_particle)
        step_amplitude = StepOneAmplitudeCFL(Ns, t, coords, coords_tmp,
                                             JK_slogdet, JK_slogdet_tmp,
                                             moved_particle, Ks, kCM, phi_1, phi_t)

        if np.abs(step_amplitude)**2 > np.random.random():
            accept_bit = 1
            coords[moved_particle] = coords_tmp[moved_particle]
            np.copyto(JK_slogdet, JK_slogdet_tmp)
            ResetJastrowsCFL(jastrows_tmp, jastrows,
                             JK_matrix_tmp, JK_matrix, moved_particle)

            update = CoulombRealSpace(
                Lx, Ly, coords, coulomb_cutoff)

        else:
            accept_bit = 0
            ResetJastrowsCFL(jastrows, jastrows_tmp, JK_matrix,
                             JK_matrix_tmp, moved_particle)

        result[i] = update

        acceptance = (acceptance*i + accept_bit)/(i + 1)
        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('cfl', 'coulomb', Ne, Ns, Lx, Ly, t, step_size,
                           result, coords, coords)

    result = result/Ne + W/np.sqrt(Lx*Ly)

    SaveResult('cfl', 'coulomb', Ne, Ns, Lx, Ly, M0, t, step_size,
               result, save_result)


def RunCoulombEnergyOld(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                        M: np.uint32, step_size: np.float64, kCM: np.uint8 = 0,
                        phi_1: np.float64 = 0, phi_t: np.float64 = 0,
                        W: np.float64 = -1.95013246, coulomb_cutoff: np.uint8 = 20):
    """
    Parameters:
    Ne : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    step_size: initial step size in units of Lx
    W : value of unnormalized self-interaction energy
    coulomb_cutoff : cutoff for the infinite sum approx. of 
                    Coulomb energy"""

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    R0 = RandomConfig(Ne, Lx, Ly)
    current_acceptance: np.float64 = 0
    all_values = np.zeros(M, dtype=np.float64)
    # coordinates = np.zeros((N, M), dtype=np.complex128)

    step_size *= Lx
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    for i in range(M):

        # coordinates[:,i] = R0
        R1, b = StepOneOld(Ne, Ns, t, R0, step_size, kCM, phi_1, phi_t)
        R0 = R1

        update = CoulombRealSpace(Lx, Ly, R0, coulomb_cutoff)
        all_values[i] = update

        current_acceptance = (current_acceptance*i+b)/(i+1)

        if (i+1) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(current_acceptance*100, 2), '%')

    return all_values/Ne + W/np.sqrt(Lx*Ly)


@njit
def UpdateResult(M: np.uint32, result: np.array, update: np.complex128,
                 index: np.uint32, acceptance: np.float64,
                 accept_bit: np.uint8):
    result[index] = update
    new_acceptance = (acceptance*index + accept_bit)/(index + 1)

    if (index+1) % (M//20) == 0:
        print('Iteration', index+1, 'done, current acceptance ratio:',
              np.round(acceptance*100, 2), '%')

    return new_acceptance
