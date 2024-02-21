import numpy as np
from numba import njit, prange
from WavefnLaughlin import ReduceCM, ReduceNonholomorphic, ThetaFunction


# @njit(parallel=True)
def InitialJastrowsCFL(t: np.complex128, Lx: np.float64,
                       coords: np.array, Ks: np.array,
                       jastrows: np.array, JK_coeffs: np.array):

    Ne = coords.size
    for k in prange(Ne):
        for i in range(Ne):
            for j in range(i+1, Ne):
                for l in range(JK_coeffs.shape[1]):
                    jastrows[i, j, k, l] = ThetaFunction(
                        (coords[i] - coords[j] +
                         JK_coeffs[0, l]*1j*Ks[k])/Lx,
                        t, 1/2, 1/2)
                    if JK_coeffs[0, l] != 0:
                        jastrows[j, i, k, l] = ThetaFunction(
                            (coords[j] - coords[i] +
                             JK_coeffs[0, l]*1j*Ks[k])/Lx,
                            t, 1/2, 1/2)
                    else:
                        jastrows[j, i, k, l] = -jastrows[i, j, k, l]


def InitialJKMatrixCFL(coords: np.array, Ks: np.array,
                       jastrows: np.array, JK_coeffs: np.array,
                       JK_matrix: np.array):

    Ne = coords.size
    for n in range(Ne):
        for m in range(Ne):
            JK_matrix[n, m] = np.exp(1j*(Ks[n] + np.conj(Ks[n]))*coords[m]/2)
            for l in range(JK_coeffs.shape[1]):
                JK_matrix[n, m] *= np.prod(
                    np.power(jastrows[m, :, n, l], JK_coeffs[1, l]))


def InitialWavefnCFL(t: np.complex128, Lx: np.float64,
                     coords: np.array, Ks: np.array,
                     jastrows: np.array, JK_coeffs: np.array,
                     JK_matrix: np.array
                     ) -> (np.complex128, np.float64):
    InitialJastrowsCFL(t, Lx, coords, Ks, jastrows, JK_coeffs)
    InitialJKMatrixCFL(coords, Ks, jastrows, JK_coeffs, JK_matrix)

    return np.linalg.slogdet(JK_matrix)


# @njit(parallel=True)
def InitialJastrowsSwapCFL(t: np.complex128, Lx: np.float64,
                           coords: np.array, Ks: np.array,
                           jastrows: np.array, JK_coeffs: np.array
                           ):

    Ne = coords.shape[0]
    for k in prange(Ne):
        for i in range(Ne):
            for j in range(Ne):
                for l in range(JK_coeffs.shape[1]):
                    jastrows[i, j+Ne, k, l] = ThetaFunction(
                        (coords[i, 0] - coords[j, 1] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    if JK_coeffs[0, l] != 0:
                        jastrows[j+Ne, i, k, l] = ThetaFunction(
                            (coords[j, 1] - coords[i, 0] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    else:
                        jastrows[j+Ne, i, k, l] = -jastrows[i, j+Ne, k, l]


@njit(parallel=True)
def InitialJKMatrixSwapCFL(coords: np.array, Ks: np.array, jastrows: np.array,
                           JK_coeffs: np.array, JK_matrix_swap: np.array, from_swap: np.array):

    Ne = coords.shape[0]
    for n in prange(Ne):
        for m in range(Ne):
            for swap_cp in range(2):
                JK_matrix_swap[n, m, swap_cp] = np.exp(
                    1j*(Ks[n] + np.conj(Ks[n]))*coords[from_swap[m, swap_cp] % Ne, from_swap[m, swap_cp] // Ne]/2)
                for j in range(Ne):
                    for l in range(JK_coeffs.shape[1]):
                        JK_matrix_swap[n, m, swap_cp] *= np.power(
                            jastrows[from_swap[m, swap_cp],
                                     from_swap[j, swap_cp], n, l],
                            JK_coeffs[1, l])


def InitialWavefnSwapCFL(t: np.complex128, Lx: np.float64,
                         coords: np.array, Ks: np.array,
                         jastrows: np.array, JK_coeffs: np.array,
                         JK_matrix_swap: np.array, from_swap: np.array):
    InitialJastrowsSwapCFL(t, Lx, coords, Ks, jastrows, JK_coeffs)
    InitialJKMatrixSwapCFL(coords, Ks, jastrows,
                           JK_coeffs, JK_matrix_swap, from_swap)

    JK_phase_0, JK_logdet_0 = np.linalg.slogdet(JK_matrix_swap[:, :, 0])
    JK_phase_1, JK_logdet_1 = np.linalg.slogdet(JK_matrix_swap[:, :, 1])
    return np.array([JK_phase_0, JK_phase_1]), np.array([JK_logdet_0, JK_logdet_1])


# @njit(parallel=True)
def UpdateJastrowsCFLOld(jastrows: np.array, coords: np.array, Ks: np.array,
                         t: np.complex128, Lx: np.float64, moved_particle: np.uint16):

    Ne = coords.size
    for k in prange(Ne):
        for i in prange(Ne):
            if i != moved_particle:
                jastrows[i, moved_particle, k] = ThetaFunction(
                    (coords[i] - coords[moved_particle] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)
                jastrows[moved_particle, i, k] = ThetaFunction(
                    (coords[moved_particle] - coords[i] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)


# @njit
def UpdateJKMatrixCFL(JK_matrix: np.array, coords: np.array,
                      jastrows_initial: np.array, jastrows_final: np.array,
                      moved_particle: np.uint16, Ks: np.array):
    Ne = coords.size
    for m in range(Ne):
        if m != moved_particle:
            JK_matrix[:, m] *= (jastrows_final[m, moved_particle, :] /
                                jastrows_initial[m, moved_particle, :])

    for n in range(Ne):
        JK_matrix[n, moved_particle] = (np.exp(1j*(Ks[n] + np.conj(Ks[n]))*coords[moved_particle]/2) *
                                        np.prod(jastrows_final[moved_particle, :, n]))


@njit(parallel=True)
def UpdateJastrowsCFL(t: np.complex128, Lx: np.float64,
                      coords_tmp: np.array, Ks: np.array,
                      jastrows: np.array, jastrows_tmp: np.array,
                      JK_coeffs: np.array,
                      JK_matrix_tmp: np.array, moved_particle: np.uint16):
    """
    Given new coordinates where we moved one single particle, this method
    updates the jastrow factors and the JK matrix.
    """
    Ne = coords_tmp.size

    JK_matrix_tmp[:, moved_particle] = np.exp(
        1j*np.real(Ks)*coords_tmp[moved_particle])
    for k in prange(Ne):
        for l in range(JK_coeffs.shape[1]):
            for i in range(Ne):
                if i != moved_particle:
                    jastrows_tmp[i, moved_particle, k, l] = ThetaFunction(
                        (coords_tmp[i] - coords_tmp[moved_particle] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                    if JK_coeffs[0, l] != 0:
                        jastrows_tmp[moved_particle, i, k, l] = ThetaFunction(
                            (coords_tmp[moved_particle] - coords_tmp[i] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    else:
                        jastrows_tmp[moved_particle, i, k, l] = - \
                            jastrows_tmp[i, moved_particle, k, l]

            JK_matrix_tmp[k, moved_particle] *= np.prod(jastrows_tmp[moved_particle,
                                                                     :, k, l]) ** JK_coeffs[1, l]
            JK_matrix_tmp[k, :] *= (jastrows_tmp[:, moved_particle, k, l] /
                                    jastrows[:, k, l]) ** JK_coeffs[1, l]


def TmpWavefnCFL(t: np.complex128, Lx: np.float64,
                 coords_tmp: np.array, Ks: np.array,
                 jastrows: np.array, jastrows_tmp: np.array,
                 JK_coeffs: np.array,
                 JK_matrix_tmp: np.array, moved_particle: np.uint16
                 ) -> (np.complex128, np.float64):

    UpdateJastrowsCFL(t, Lx, coords_tmp, Ks, jastrows[:, moved_particle, :, :], jastrows_tmp,
                      JK_coeffs, JK_matrix_tmp, moved_particle)
    return np.linalg.slogdet(JK_matrix_tmp)


@njit(parallel=True)
def UpdateJastrowsSwapCFL(t: np.complex128, Lx: np.float64,
                          coords: np.array, Ks: np.array,
                          jastrows: np.array, JK_coeffs: np.array,
                          moved_particles: np.array, to_swap: np.array):
    """figure out how to update only on this side. would like to do today but im not sure.
    also want to do the workshop problems today."""
    Ne = coords.shape[0]

    for k in prange(Ne):
        for i in range(Ne):
            for l in range(JK_coeffs.shape[1]):
                # update cross copy jastrows for particle in copy 2
                if (to_swap[moved_particles[1], 1] // Ne) == (to_swap[i, 0] // Ne):
                    jastrows[i, moved_particles[1]+Ne, k, l] = ThetaFunction(
                        (coords[i, 0] - coords[moved_particles[1], 1] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                    if JK_coeffs[0, l] != 0:
                        jastrows[moved_particles[1]+Ne, i, k, l] = ThetaFunction(
                            (coords[moved_particles[1], 1] - coords[i, 0] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    else:
                        jastrows[moved_particles[1]+Ne, i, k, l] = - \
                            jastrows[i, moved_particles[1]+Ne, k, l]

                # update cross copy jastrows for particle in copy 1
                if (to_swap[moved_particles[0], 0] // Ne) == (to_swap[i, 1] // Ne):
                    jastrows[i+Ne, moved_particles[0], k, l] = ThetaFunction(
                        (coords[i, 1] - coords[moved_particles[0], 0] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                    if JK_coeffs[0, l] != 0:
                        jastrows[moved_particles[0], i+Ne, k, l] = ThetaFunction(
                            (coords[moved_particles[0], 0] - coords[i, 1] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    else:
                        jastrows[moved_particles[0], i+Ne, k, l] = - \
                            jastrows[i+Ne, moved_particles[0], k, l]


@njit(parallel=True)
def UpdateJastrowsAllSwapCFL(t: np.complex128, Lx: np.float64,
                             coords: np.array, Ks: np.array,
                             jastrows: np.array, moved_particles: np.array):
    """figure out how to update only on this side. would like to do today but im not sure.
    also want to do the workshop problems today."""
    Ne = coords.shape[0]

    for k in prange(Ne):
        for i in range(Ne):
            # update cross copy jastrows for particle in copy 2
            jastrows[i, moved_particles[1]+Ne, k] = ThetaFunction(
                (coords[i, 0] - coords[moved_particles[1], 1] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)
            jastrows[moved_particles[1]+Ne, i, k] = ThetaFunction(
                (coords[moved_particles[1], 1] - coords[i, 0] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)

            # update cross copy jastrows for particle in copy 1
            jastrows[i+Ne, moved_particles[0], k] = ThetaFunction(
                (coords[i, 1] - coords[moved_particles[0], 0] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)
            jastrows[moved_particles[0], i+Ne, k] = ThetaFunction(
                (coords[moved_particles[0], 0] - coords[i, 1] + 2*1j*Ks[k])/Lx, t, 1/2, 1/2)


# @njit
def ResetJastrowsCFL(jastrows: np.array, jastrows_tmp: np.array,
                     JK_matrix: np.array, JK_matrix_tmp: np.array,
                     moved_particle: np.uint16):
    np.copyto(jastrows_tmp[moved_particle, :, ...],
              jastrows[moved_particle, :, ...])
    np.copyto(jastrows_tmp[:, moved_particle, ...],
              jastrows[:, moved_particle, ...])
    np.copyto(JK_matrix_tmp, JK_matrix)


# @njit
def UpdateJKMatrixSwapCFL(JK_matrix_swap: np.array, coords: np.array,
                          jastrows_initial: np.array, jastrows_final: np.array,
                          moved_particles: np.array, Ks: np.array,
                          from_swap: np.array):
    Ne = coords.shape[0]
    for m in range(Ne):
        if m not in moved_particles:
            for p in moved_particles:
                JK_matrix_swap[:, m] *= (jastrows_final[from_swap[m], from_swap[p], :] /
                                         jastrows_initial[from_swap[m], from_swap[p], :])

    for n in range(Ne):
        for p in moved_particles:
            JK_matrix_swap[n, p] = (np.exp(1j*(Ks[n] + np.conj(Ks[n])) *
                                           coords[from_swap[m] % Ne, from_swap[m] // Ne]/2))
            for j in range(Ne):
                if j != p:
                    JK_matrix_swap[n, p] *= jastrows_final[from_swap[p],
                                                           from_swap[j], n]


def TmpWavefnSwapCFL(t: np.complex128, Lx: np.float64,
                     coords_new: np.array, Ks: np.array,
                     jastrows_current: np.array, jastrows_new: np.array,
                     JK_coeffs: np.array,
                     JK_matrix_swap: np.array, moved_particles: np.array,
                     to_swap_new: np.array, from_swap_new: np.array
                     ) -> (np.array, np.array):

    UpdateJastrowsSwapCFL(t, Lx, coords_new, Ks, jastrows_new, JK_coeffs,
                          moved_particles, to_swap_new)
    # UpdateJastrowsAllSwapCFL(t, Lx, coords_new, Ks,
    #                         jastrows_new, moved_particles)
    """
    Ne = coords_new.shape[0]
    moved_swap = np.array(
        [to_swap_new[moved_particles[0], 0], to_swap_new[moved_particles[1], 1]])
    swap_cp = moved_swap // Ne

    if swap_cp[0] == swap_cp[1]:
        UpdateJKMatrixSwapCFL(JK_matrix_swap[:, :, swap_cp[0]], coords_new,
                              jastrows_current, jastrows_new, moved_swap % Ne,
                              Ks, from_swap_new[:, swap_cp[0]])
    else:
        UpdateJKMatrixSwapCFL(JK_matrix_swap[:, :, swap_cp[0]], coords_new,
                              jastrows_current, jastrows_new, np.array(
                                  [moved_swap[0] % Ne]),
                              Ks, from_swap_new[:, swap_cp[0]])
        UpdateJKMatrixSwapCFL(JK_matrix_swap[:, :, swap_cp[1]], coords_new,
                              jastrows_current, jastrows_new, np.array(
                                  [moved_swap[1] % Ne]),
                              Ks, from_swap_new[:, swap_cp[1]])
    """
    InitialJKMatrixSwapCFL(coords_new, Ks, jastrows_new, JK_coeffs,
                           JK_matrix_swap, from_swap_new)

    JK_phase_0, JK_logdet_0 = np.linalg.slogdet(JK_matrix_swap[:, :, 0])
    JK_phase_1, JK_logdet_1 = np.linalg.slogdet(JK_matrix_swap[:, :, 1])
    return np.array([JK_phase_0, JK_phase_1]), np.array([JK_logdet_0, JK_logdet_1])


# @njit
def ReducedAmplitudeCFL(Ns: np.uint16, t: np.complex128,
                        coords_current: np.array, coords_new: np.array,
                        moved_particle: np.uint8, Ks: np.array,
                        kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0
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
    p : index of partile that moves

    Output:

    r : ratio of wavefunctions R_f/R_i
    """

    Ne = coords_current.size
    m = Ns/Ne
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))

    (reduced_CM_current, exp_CM_current) = ReduceCM(Ne, Ns, t, np.sum(
        coords_current + 1j*Ks), kCM, phi_1, phi_t)
    # print(reduced_CM_current, exp_CM_current)
    (reduced_CM_new, exp_CM_new) = ReduceCM(Ne, Ns, t, np.sum(
        coords_new + 1j*Ks), kCM, phi_1, phi_t)
    # print(reduced_CM_new, exp_CM_new)
    reduced_amplitude = reduced_CM_new/reduced_CM_current
    exp_nonholomorphic = (ReduceNonholomorphic(np.array([coords_new[moved_particle]])) -
                          ReduceNonholomorphic(np.array([coords_current[moved_particle]])))
    # exp_wavevector = np.sum(Ks*(Ks + 2*np.conj(Ks))/4)
    reduced_exponent = (exp_CM_new - exp_CM_current +
                        exp_nonholomorphic)

    return reduced_amplitude, reduced_exponent


@njit
def GetExtraJastrowFactor(Ne, Ns, jastrows, moved_particle):

    jastrows_factor = 1
    if Ns//Ne == 3:
        jastrows_factor = np.prod(jastrows[moved_particle, :])
    elif Ns//Ne == 1:
        jastrows_factor = 1/np.prod(jastrows[moved_particle, :])

    return jastrows_factor


# @njit
def GetExtraJastrowFactorSwap(Ne, Ns, jastrows, from_swap):

    jastrows_factor = 1
    for n in prange(Ne):
        for m in range(n+1, Ne):
            for swap_cp in range(2):
                if Ns//Ne == 3:
                    jastrows_factor *= jastrows[from_swap[n, swap_cp],
                                                from_swap[m, swap_cp]]
                elif Ns//Ne == 1:
                    jastrows_factor /= jastrows[from_swap[n, swap_cp],
                                                from_swap[m, swap_cp]]

    return jastrows_factor


# @njit
def StepOneAmplitudeCFL(Ns: np.uint16, t: np.complex128,
                        coords: np.array, coords_tmp: np.array,
                        jastrows: np.array, jastrows_tmp: np.array,
                        JK_slogdet: np.array, JK_slogdet_tmp: np.array,
                        moved_particle: np.uint8, Ks: np.array,
                        kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0
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
    p : index of particle that moves

    Output:

    step_amplitude : ratio of wavefunctions new_current
    """

    Ne = coords.size

    (reduced_CM_current, exp_CM_current) = ReduceCM(Ne, Ns, t, np.sum(
        coords + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_new, exp_CM_new) = ReduceCM(Ne, Ns, t, np.sum(
        coords_tmp + 1j*Ks), kCM, phi_1, phi_t)

    reduced_amplitude = reduced_CM_new/reduced_CM_current
    exp_nonholomorphic = (ReduceNonholomorphic(np.array([coords_tmp[moved_particle]])) -
                          ReduceNonholomorphic(np.array([coords[moved_particle]])))
    # exp_wavevector = np.sum(Ks*(Ks + 2*np.conj(Ks))/4)
    reduced_exponent = (exp_CM_new - exp_CM_current +
                        exp_nonholomorphic)

    step_amplitude = reduced_amplitude*(JK_slogdet_tmp[0]/JK_slogdet[0])
    step_exponent = reduced_exponent + JK_slogdet_tmp[1] - JK_slogdet[1]

    if Ns//Ne % 2 == 0:
        return step_amplitude*np.exp(step_exponent)
    elif Ns//Ne == 1:
        return step_amplitude*(np.prod(jastrows[moved_particle, :]/jastrows_tmp[moved_particle, :] *
                               np.exp(step_exponent/Ne)))
    elif Ns//Ne == 3:
        return step_amplitude*(np.prod(jastrows_tmp[moved_particle, :]/jastrows[moved_particle, :] *
                               np.exp(step_exponent/Ne)))


@njit
def StepOneAmplitudeSwapCFL(Ns: np.uint16, t: np.complex128,
                            coords: np.array, coords_tmp: np.array, Ks: np.array,
                            jastrows: np.array, jastrows_tmp: np.array,
                            JK_slogdet: np.array, JK_slogdet_tmp: np.array,
                            moved_particles: np.array, from_swap, from_swap_tmp,
                            kCM: np.uint8 = 0, phi_1: np.float64 = 0, phi_t: np.float64 = 0
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
    p : index of particle that moves

    Output:

    step_amplitude : ratio of wavefunctions new_current
    """

    Ne = coords.shape[0]

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    coords_swap_new = np.zeros(
        (coords_tmp.shape[0], coords_tmp.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]
            coords_swap_new[i, cp] = coords_tmp[from_swap_tmp[i, cp] %
                                                Ne, from_swap_tmp[i, cp] // Ne]

    (reduced_CM_current_0, exp_CM_current_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_current_1, exp_CM_current_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    (reduced_CM_new_0, exp_CM_new_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap_new[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_new_1, exp_CM_new_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap_new[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    reduced_amplitude = ((reduced_CM_new_0 * reduced_CM_new_1) /
                         (reduced_CM_current_0 * reduced_CM_current_1))
    exp_nonholomorphic = (ReduceNonholomorphic(np.array([coords_tmp[moved_particles[0], 0], coords_tmp[moved_particles[1], 1]])) -
                          ReduceNonholomorphic(np.array([coords[moved_particles[0], 0], coords[moved_particles[1], 1]])))
    # exp_wavevector = np.sum(Ks*(Ks + 2*np.conj(Ks))/4)
    reduced_exponent = (exp_CM_new_0 + exp_CM_new_1 - exp_CM_current_0 - exp_CM_current_1 +
                        exp_nonholomorphic)

    step_amplitude = (reduced_amplitude*JK_slogdet_tmp[0, 0]*JK_slogdet_tmp[0, 1] /
                      (JK_slogdet[0, 0]*JK_slogdet[0, 1]))
    step_exponent = (reduced_exponent + JK_slogdet_tmp[1, 0] + JK_slogdet_tmp[1, 1]
                     - JK_slogdet[1, 0] - JK_slogdet[1, 1])

    if Ns//Ne % 2 == 0:
        return step_amplitude*np.exp(step_exponent)
    else:
        jastrows_factor = 1
        for n in prange(Ne):
            for m in range(n+1, Ne):
                for swap_cp in range(2):
                    if Ns//Ne == 3:
                        jastrows_factor *= (jastrows_tmp[from_swap_tmp[n, swap_cp],
                                                         from_swap_tmp[m, swap_cp]] /
                                            jastrows[from_swap[n, swap_cp],
                                                     from_swap[m, swap_cp]] *
                                            np.exp(step_exponent/(Ne*(Ne-1))))
                    elif Ns//Ne == 1:
                        jastrows_factor *= (jastrows[from_swap[n, swap_cp],
                                                     from_swap[m, swap_cp]] /
                                            jastrows_tmp[from_swap_tmp[n, swap_cp],
                                                         from_swap_tmp[m, swap_cp]] *
                                            np.exp(step_exponent/(Ne*(Ne-1))))

        return step_amplitude*jastrows_factor


def InitialModCFL(Ne, Ns, t, coords: np.array, Ks: np.array,
                  jastrows: np.array, JK_slogdet: np.array,
                  from_swap: np.array, kCM: np.uint8,
                  phi_1: np.float64, phi_t: np.float64):

    (reduced_CM_0, exp_CM_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_1, exp_CM_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] % Ne,
                                        from_swap[i, cp] // Ne]
    (reduced_CM_swap_0, exp_CM_swap_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_swap_1, exp_CM_swap_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    reduced_amplitude = (reduced_CM_swap_0*reduced_CM_swap_1 *
                         JK_slogdet[0, 2]*JK_slogdet[0, 3] /
                         (reduced_CM_0*reduced_CM_1 *
                          JK_slogdet[0, 0]*JK_slogdet[0, 1]))
    reduced_exponent = (exp_CM_swap_0 + exp_CM_swap_1 - exp_CM_0 - exp_CM_1 +
                        JK_slogdet[1, 2] + JK_slogdet[1, 3] -
                        JK_slogdet[1, 0] - JK_slogdet[1, 1])

    if Ns//Ne % 2 == 0:
        return np.abs(reduced_amplitude*np.exp(reduced_exponent))
    else:
        jastrows_factor = 1
        for n in prange(Ne):
            for m in range(n+1, Ne):
                for cp in range(2):
                    r = (jastrows[from_swap[n, cp],
                                  from_swap[m, cp]] /
                         jastrows[cp*Ne+n, cp*Ne+m])
                    if Ns//Ne == 3:
                        jastrows_factor *= r
                    elif Ns//Ne == 1:
                        jastrows_factor /= r

                jastrows_factor *= np.exp(reduced_exponent/(Ne*(Ne-1)/2))

        return np.abs(reduced_amplitude*jastrows_factor)


def InitialSignCFL(Ne, Ns, t, coords: np.array, Ks: np.array, jastrows: np.array,
                   JK_slogdet: np.array, from_swap: np.array,
                   kCM: np.uint8, phi_1: np.float64, phi_t: np.float64):

    (reduced_CM_0, exp_CM_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_1, exp_CM_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]

    (reduced_CM_swap_0, exp_CM_swap_0) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 0] + 1j*Ks), kCM, phi_1, phi_t)
    (reduced_CM_swap_1, exp_CM_swap_1) = ReduceCM(Ne, Ns, t, np.sum(
        coords_swap[:, 1] + 1j*Ks), kCM, phi_1, phi_t)

    reduced_amplitude = (np.conj(reduced_CM_swap_0*reduced_CM_swap_1*JK_slogdet[2]*JK_slogdet[3]) *
                         reduced_CM_0*reduced_CM_1*JK_slogdet[0]*JK_slogdet[1])
    reduced_amplitude /= np.abs(reduced_amplitude)
    reduced_exponent = 1j*np.imag(- exp_CM_swap_0 - exp_CM_swap_1
                                  + exp_CM_0 + exp_CM_1)

    if Ns//Ne % 2 == 0:
        return (reduced_amplitude*np.exp(reduced_exponent) /
                np.abs(reduced_amplitude*np.exp(reduced_exponent)))
    else:
        jastrows_factor = 1
        for n in prange(Ne):
            for m in range(n+1, Ne):
                for cp in range(2):
                    r = (np.conj(jastrows[from_swap[n, cp],
                                          from_swap[m, cp]]) *
                         jastrows[cp*Ne+n, cp*Ne+m])
                    r /= np.abs(r)
                    if Ns//Ne == 3:
                        jastrows_factor *= r
                    elif Ns//Ne == 1:
                        jastrows_factor /= r

                jastrows_factor *= np.exp(reduced_exponent/(Ne*(Ne-1)/2))

        return (reduced_amplitude*jastrows_factor /
                np.abs(reduced_amplitude*jastrows_factor))
