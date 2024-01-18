import numpy as np

from MonteCarloTorusTools import RandomConfig, InsideRegion, StepOneCircle, RandomConfigSWAP, \
    AssignOrderToSwap, OrderFromSwap, StepOneSwap, UpdateOrderSwap
from WavefnCFL import InitialWavefnCFL, InitialWavefnSwapCFL, \
    InitialModCFL, InitialSignCFL, TmpWavefnCFL, TmpWavefnSwapCFL, StepOneAmplitudeSwapCFL, \
    ResetJastrowsCFL, StepOneAmplitudeCFL, UpdateJastrowsCFL, GetExtraJastrowFactor
from utilities import SaveConfig, SaveResults, fermi_sea_kx, fermi_sea_ky, LoadRun


def RunPSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                M: np.uint32, M0: np.uint32, step_size: np.float64,
                region_geometry: str, region_size: np.float64, JK_coeffs: str,
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

    if (Ns//Ne) % 2 == 0:
        statistics = 'fermions'
    elif (Ns//Ne) % 2 == 1:
        statistics = 'bosons'

    JK_coeffs_parsed = np.array(list(JK_coeffs), dtype=int)
    if np.sum(JK_coeffs_parsed) != Ns/Ne:
        print("JK coefficients are incorrect for the given filling!")
        return 1
    else:
        coeffs, counts = np.unique(JK_coeffs_parsed, return_counts=True)
        JK_coeffs_unique = np.vstack((coeffs, counts))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, _, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                       step_size/Lx, 'cfl'+JK_coeffs, 'p')
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
    jastrows = np.ones(
        (Ne, Ne, Ne, JK_coeffs_unique.shape[1], 2), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 2), dtype=np.complex128)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[...,
                                                          cp], JK_coeffs_unique,
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
                                                     JK_coeffs_unique, JK_matrix_tmp[..., cp],
                                                     moved_particles[cp])
            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  jastrows[:, :, 0, 0, cp],
                                                  jastrows_tmp[:, :, 0, 0, cp],
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
                SaveConfig('cfl'+JK_coeffs, 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, coords)

    SaveResults('cfl'+JK_coeffs, 'p', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunModSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                  M: np.uint32, M0: np.uint32, step_size: np.float64,
                  region_geometry: str, region_size: np.float64, JK_coeffs: str,
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

    JK_coeffs_parsed = np.array(list(JK_coeffs), dtype=int)
    if np.sum(JK_coeffs_parsed) != Ns/Ne:
        print("JK coefficients are incorrect for the given filling!")
        return 1
    else:
        coeffs, counts = np.unique(JK_coeffs_parsed, return_counts=True)
        JK_coeffs_unique = np.vstack((coeffs, counts))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'cfl'+JK_coeffs, 'mod')
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
    coords_tmp = np.copy(coords)

    JK_slogdet = np.zeros((2, 4), dtype=np.complex128)
    JK_slogdet_tmp = np.zeros((2, 4), dtype=np.complex128)
    jastrows = np.ones(
        (2*Ne, 2*Ne, Ne, JK_coeffs_unique.shape[1]), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[cp*Ne:(cp+1)*Ne,
                                                          cp*Ne:(cp+1)*Ne, ...],
                                                 JK_coeffs_unique,
                                                 JK_matrix[:, :, cp])

    JK_slogdet[0, 2:4], \
        JK_slogdet[1, 2:4] = InitialWavefnSwapCFL(t, Lx, coords, Ks, jastrows,
                                                  JK_coeffs_unique,
                                                  JK_matrix[:, :, 2:], from_swap)
    update = InitialModCFL(Ne, Ns, t, coords, Ks, jastrows[:, :, 0, 0], JK_slogdet,
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
                                                              cp*Ne:(cp+1)*Ne, ...],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, ...],
                                                     JK_coeffs_unique,
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])
            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  jastrows[cp*Ne:(cp+1)*Ne,
                                                           cp*Ne:(cp+1)*Ne, 0, 0],
                                                  jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                               cp*Ne:(cp+1)*Ne, 0, 0],
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
                                                          JK_coeffs_unique,
                                                          JK_matrix_tmp[:, :,
                                                                        2:4], moved_particles,
                                                          to_swap_tmp, from_swap_tmp)

            step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                          jastrows[:, :, 0, 0],
                                                          jastrows_tmp[:,
                                                                       :, 0, 0],
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
                SaveConfig('cfl'+JK_coeffs, 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('cfl'+JK_coeffs, 'mod', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunSignSwapCFL(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                   M: np.uint32, M0: np.uint32, step_size: np.float64,
                   region_geometry: str, region_size: np.float64, JK_coeffs: str,
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

    JK_coeffs_parsed = np.array(list(JK_coeffs), dtype=int)
    if np.sum(JK_coeffs_parsed) != Ns/Ne:
        print("JK coefficients are incorrect for the given filling!")
        return 1
    else:
        coeffs, counts = np.unique(JK_coeffs_parsed, return_counts=True)
        JK_coeffs_unique = np.vstack((coeffs, counts))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'cfl'+JK_coeffs, 'sign')
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
    jastrows = np.ones(
        (2*Ne, 2*Ne, Ne, JK_coeffs_unique.shape[1]), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[cp*Ne:(cp+1)*Ne,
                                                          cp*Ne:(cp+1)*Ne, ...],
                                                 JK_coeffs_unique,
                                                 JK_matrix[:, :, cp])
    JK_slogdet[0, 2:4], \
        JK_slogdet[1, 2:4] = InitialWavefnSwapCFL(t, Lx, coords, Ks, jastrows,
                                                  JK_coeffs_unique,
                                                  JK_matrix[:, :, 2:], from_swap)
    update = InitialSignCFL(Ne, Ns, t, coords, Ks, jastrows[:, :, 0, 0],
                            JK_slogdet[0, :], from_swap,
                            kCM, phi_1, phi_t)

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
                                                              cp*Ne:(cp+1)*Ne, ...],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, ...],
                                                     JK_coeffs_unique,
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])

            step_amplitude *= StepOneAmplitudeCFL(Ns, t, coords[:, cp], coords_tmp[:, cp],
                                                  jastrows[cp*Ne:(cp+1)*Ne,
                                                           cp*Ne:(cp+1)*Ne, 0, 0],
                                                  jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                               cp*Ne:(cp+1)*Ne, 0, 0],
                                                  JK_slogdet[:, cp],
                                                  JK_slogdet_tmp[:, cp],
                                                  moved_particles[cp],
                                                  Ks, kCM, phi_1, phi_t)

        UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                        moved_particles, nbr_A_changes)

        JK_slogdet_tmp[0, 2:4], \
            JK_slogdet_tmp[1, 2:4] = TmpWavefnSwapCFL(t, Lx, coords_tmp, Ks,
                                                      jastrows, jastrows_tmp,
                                                      JK_coeffs_unique,
                                                      JK_matrix_tmp[...,
                                                                    2:4], moved_particles,
                                                      to_swap_tmp, from_swap_tmp)

        step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                      jastrows[:, :, 0, 0],
                                                      jastrows_tmp[:, :, 0, 0],
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
                SaveConfig('cfl'+JK_coeffs, 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('cfl'+JK_coeffs, 'sign', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)
