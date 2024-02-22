import numpy as np
from src.utilities import fermi_sea_kx, fermi_sea_ky
from src.MonteCarloTorusCFL import MonteCarloTorusCFL


def TestPeriodicityCFL(Ne, Ns, t, JK_coeffs, pfaffian_flag=False, nbr_copies=1,
                       kCM=0, phi_1=0, phi_t=0):
    test_passed = True
    cfl = MonteCarloTorusCFL(Ne, Ns, t, 1, 1, 'circle', 0.1, 0.1, JK_coeffs,
                             pfaffian_flag, nbr_copies, kCM, phi_1, phi_t)
    cfl.LoadRun('disorder')
    cfl.InitialWavefn()
    for moved_particle in range(cfl.coords.size):
        cfl.moved_particles[0] = moved_particle
        cfl.coords_tmp = np.copy(cfl.coords)
        cfl.coords_tmp[moved_particle] += cfl.Lx
        cfl.TmpWavefn()
        step_amplitude = cfl.StepAmplitude()

        t_L1 = (step_amplitude *
                np.exp(-1j*cfl.Lx*np.imag(cfl.coords[moved_particle])/2)*np.exp(-1j*cfl.phi_1))
        if np.abs(t_L1 - 1) > 1e-11:
            print('t(Lx) * exp(-i phi_1) = ', t_L1)
            test_passed = False

        cfl.RejectJastrowsTmp('disorder')

        cfl.coords_tmp[moved_particle] += 1j*cfl.Ly
        cfl.TmpWavefn()
        step_amplitude = cfl.StepAmplitude()

        t_L2 = (step_amplitude *
                np.exp(1j*cfl.Ly*np.real(cfl.coords[moved_particle])/2)*np.exp(-1j*phi_t))

        if np.abs(t_L2 - 1) > 1e-11:
            print('t(Lx*t) * exp(-i phi_t) = ', t_L2)
            test_passed = False

        cfl.RejectJastrowsTmp('disorder')


def TestStepUpdateCFL(Ne, Ns, t, JK_coeffs, pfaffian_flag=False, nbr_copies=1,
                      accept=True, kCM=0, phi_1=0, phi_t=0):
    test_passed = True
    cfl = MonteCarloTorusCFL(Ne, Ns, t, 1, 1, 'circle', 0.1, 0.1, JK_coeffs,
                             pfaffian_flag, nbr_copies, kCM, phi_1, phi_t)
    cfl.LoadRun('disorder')
    cfl.InitialWavefn()

    for i in range(100):
        cfl.StepOneParticle()
        cfl.TmpWavefn()
        if accept:
            cfl.AcceptJastrowsTmp('disorder')
        else:
            cfl.RejectJastrowsTmp('disorder')

        if np.abs(np.sum(cfl.coords-cfl.coords_tmp)) > 1e-13:
            print("Coordinates update failed!")
        if np.abs(np.sum(cfl.jastrows[cfl.moved_particles[0], ...]-cfl.jastrows_tmp[cfl.moved_particles[0], ...])) > 1e-13:
            print("Jastrows update failed!")
        if np.abs(np.sum(cfl.JK_matrix-cfl.JK_matrix_tmp)) > 1e-13:
            print("JK matrix update failed!")
        if np.abs(np.sum(cfl.JK_slogdet-cfl.JK_slogdet_tmp)) > 1e-13:
            print("JK slogdet update failed!")


def TestPeriodicityCFLold(Ne: np.array, Ns: np.uint16, t: np.complex128,
                          JK_coeffs: str,
                          kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                          phi_t: np.float64 = 0, nbr_tests: np.uint16 = 1):

    test_passed = True

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    coords = RandomConfig(Ne, Lx, Ly)
    Ks = (fermi_sea_kx[Ne]*2*np.pi/Lx + 1j*fermi_sea_ky[Ne]*2*np.pi/Ly)

    JK_coeffs_parsed = np.array(list(JK_coeffs), dtype=int)
    if np.sum(JK_coeffs_parsed) != Ns/Ne:
        print("JK coefficients are incorrect for the given filling!")
        return 1
    else:
        coeffs, counts = np.unique(JK_coeffs_parsed, return_counts=True)
        JK_coeffs_unique = np.vstack((coeffs, counts))

    jastrows = np.ones(
        (Ne, Ne, Ne, JK_coeffs_unique.shape[1]), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne), dtype=np.complex128)
    JK_slogdet = np.zeros(2, dtype=np.complex128)
    JK_slogdet[0], JK_slogdet[1] = InitialWavefnCFL(t, Lx, coords, Ks,
                                                    jastrows, JK_coeffs_unique,
                                                    JK_matrix)
    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)
    JK_slogdet_tmp = np.zeros(2, dtype=np.complex128)

    for _ in range(nbr_tests):
        for moved_particle in range(coords.size):
            coords_tmp = np.copy(coords)
            coords_tmp[moved_particle] += Lx

            JK_slogdet_tmp[0], JK_slogdet_tmp[1] = TmpWavefnCFL(t, Lx, coords_tmp, Ks,
                                                                jastrows, jastrows_tmp, JK_coeffs_unique,
                                                                JK_matrix_tmp, moved_particle)
            # jastrows_factor = (GetExtraJastrowFactor(Ne, Ns, jastrows_tmp[:, :, 0, 0], moved_particle) /
            #                   GetExtraJastrowFactor(Ne, Ns, jastrows[:, :, 0, 0], moved_particle))

            step_amplitude = StepOneAmplitudeCFL(Ns, t, coords, coords_tmp,
                                                 jastrows[:, :, 0, 0],
                                                 jastrows_tmp[:, :, 0, 0],
                                                 JK_slogdet, JK_slogdet_tmp,
                                                 moved_particle, Ks,
                                                 kCM, phi_1, phi_t)

            t_L1 = (step_amplitude *
                    np.exp(-1j*Lx*np.imag(coords[moved_particle])/2)*np.exp(-1j*phi_1))

            if np.abs(t_L1 - 1) > 1e-10:
                print('t(Lx) * exp(-i phi_1) = ', t_L1)
                test_passed = False

            ResetJastrowsCFL(jastrows, jastrows_tmp, JK_matrix,
                             JK_matrix_tmp, moved_particle)

            coords_tmp = np.copy(coords)
            coords_tmp[moved_particle] += Lx*t
            JK_slogdet_tmp[0], JK_slogdet_tmp[1] = TmpWavefnCFL(t, Lx, coords_tmp, Ks,
                                                                jastrows, jastrows_tmp, JK_coeffs_unique,
                                                                JK_matrix_tmp, moved_particle)
            # jastrows_factor = (GetExtraJastrowFactor(Ne, Ns, jastrows_tmp[:, :, 0, 0], moved_particle) /
            #                   GetExtraJastrowFactor(Ne, Ns, jastrows[:, :, 0, 0], moved_particle))

            step_amplitude = StepOneAmplitudeCFL(Ns, t, coords, coords_tmp,
                                                 jastrows[:, :, 0, 0],
                                                 jastrows_tmp[:, :, 0, 0],
                                                 JK_slogdet, JK_slogdet_tmp,
                                                 moved_particle, Ks,
                                                 kCM, phi_1, phi_t)

            t_L2 = (step_amplitude *
                    np.exp(1j*Ly*np.real(coords[moved_particle])/2)*np.exp(-1j*phi_t))

            if np.abs(t_L2 - 1) > 1e-10:
                print('t(Lx*t) * exp(-i phi_t) = ', t_L2)
                test_passed = False

            ResetJastrowsCFL(jastrows, jastrows_tmp, JK_matrix,
                             JK_matrix_tmp, moved_particle)

    if test_passed:
        print('Test passed!')
    else:
        print('Test failed!')


def TestPeriodicitySwapCFL(Ne: np.array, Ns: np.uint16, t: np.complex128,
                           JK_coeffs: str,
                           kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                           phi_t: np.float64 = 0, region_geometry: str = 'circle',
                           boundary: np.float64 = 0.2, nbr_tests: np.uint16 = 1):

    test_passed = True

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
    Ks = (fermi_sea_kx[Ne]*2*np.pi/Lx + 1j*fermi_sea_ky[Ne]*2*np.pi/Ly)

    JK_coeffs_parsed = np.array(list(JK_coeffs), dtype=int)
    if np.sum(JK_coeffs_parsed) != Ns/Ne:
        print("JK coefficients are incorrect for the given filling!")
        return 1
    else:
        coeffs, counts = np.unique(JK_coeffs_parsed, return_counts=True)
        JK_coeffs_unique = np.vstack((coeffs, counts))

    jastrows = np.ones(
        (2*Ne, 2*Ne, Ne, JK_coeffs_unique.shape[1]), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)
    JK_slogdet = np.zeros((2, 4), dtype=np.complex128)

    for cp in range(2):
        JK_slogdet[0, cp], \
            JK_slogdet[1, cp] = InitialWavefnCFL(t, Lx, coords[:, cp], Ks,
                                                 jastrows[cp*Ne:(cp+1)*Ne,
                                                          cp*Ne:(cp+1)*Ne, ...],
                                                 JK_coeffs_unique,
                                                 JK_matrix[:, :, cp])
    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)
    JK_slogdet_tmp = np.zeros((2, 4), dtype=np.complex128)

    to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                             region_geometry, boundary))
    from_swap = OrderFromSwap(to_swap)
    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)
    JK_slogdet[0, 2:4], \
        JK_slogdet[1, 2:4] = InitialWavefnSwapCFL(t, Lx, coords, Ks, jastrows,
                                                  JK_coeffs_unique,
                                                  JK_matrix[:, :, 2:], from_swap)

    for _ in range(nbr_tests):
        moved_particles = np.random.randint(0, Ne, 2)
        coords_tmp = np.copy(coords)

        for cp in range(2):
            coords_tmp[moved_particles[cp], cp] += Lx
            JK_slogdet_tmp[0, cp], \
                JK_slogdet_tmp[1, cp] = TmpWavefnCFL(t, Lx, coords_tmp[:, cp], Ks,
                                                     jastrows[cp*Ne:(cp+1)*Ne,
                                                              cp*Ne:(cp+1)*Ne, ...],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, ...],
                                                     JK_coeffs_unique,
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])
        JK_slogdet_tmp[0, 2:4], \
            JK_slogdet_tmp[1, 2:4] = TmpWavefnSwapCFL(t, Lx, coords_tmp, Ks,
                                                      jastrows, jastrows_tmp,
                                                      JK_coeffs_unique,
                                                      JK_matrix_tmp[:, :, 2:4],
                                                      moved_particles,
                                                      to_swap_tmp, from_swap_tmp)
        # jastrows_factor_swap = (GetExtraJastrowFactorSwap(Ne, Ns, jastrows_tmp[:, :, 0, 0], from_swap_tmp) /
        #                        GetExtraJastrowFactorSwap(Ne, Ns, jastrows[:, :, 0, 0], from_swap))
        step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                      jastrows[:, :, 0,
                                                               0], jastrows_tmp[:, :, 0, 0],
                                                      JK_slogdet[:, 2:4],
                                                      JK_slogdet_tmp[:, 2:4],
                                                      moved_particles,
                                                      from_swap, from_swap_tmp,
                                                      kCM, phi_1, phi_t)
        t_L1 = (step_amplitude_swap *
                np.exp(-1j*Lx*np.imag(coords[moved_particles[0], 0])/2) *
                np.exp(-1j*Lx*np.imag(coords[moved_particles[1], 1])/2) *
                np.exp(-2j*phi_1))

        if np.abs(t_L1 - 1) > 1e-10:
            print('t(Lx) * exp(-i phi_1) = ', t_L1)
            test_passed = False

        ResetJastrowsCFL(jastrows, jastrows_tmp,
                         JK_matrix[..., 0], JK_matrix_tmp[...,  0], moved_particles[0])
        ResetJastrowsCFL(jastrows, jastrows_tmp,
                         JK_matrix[..., 1], JK_matrix_tmp[...,  1], Ne+moved_particles[1])

        coords_tmp = np.copy(coords)

        for cp in range(2):
            coords_tmp[moved_particles[cp], cp] += Lx*t
            JK_slogdet_tmp[0, cp], \
                JK_slogdet_tmp[1, cp] = TmpWavefnCFL(t, Lx, coords_tmp[:, cp], Ks,
                                                     jastrows[cp*Ne:(cp+1)*Ne,
                                                              cp*Ne:(cp+1)*Ne, ...],
                                                     jastrows_tmp[cp*Ne:(cp+1)*Ne,
                                                                  cp*Ne:(cp+1)*Ne, ...],
                                                     JK_coeffs_unique,
                                                     JK_matrix_tmp[..., cp], moved_particles[cp])
        JK_slogdet_tmp[0, 2:4], \
            JK_slogdet_tmp[1, 2:4] = TmpWavefnSwapCFL(t, Lx, coords_tmp, Ks,
                                                      jastrows, jastrows_tmp,
                                                      JK_coeffs_unique,
                                                      JK_matrix_tmp[:, :, 2:4],
                                                      moved_particles,
                                                      to_swap_tmp, from_swap_tmp)
        # jastrows_factor_swap = (GetExtraJastrowFactorSwap(Ne, Ns, jastrows_tmp[:, :, 0, 0], from_swap_tmp) /
        #                        GetExtraJastrowFactorSwap(Ne, Ns, jastrows[:, :, 0, 0], from_swap))
        step_amplitude_swap = StepOneAmplitudeSwapCFL(Ns, t, coords, coords_tmp, Ks,
                                                      jastrows[:, :, 0,
                                                               0], jastrows_tmp[:, :, 0, 0],
                                                      JK_slogdet[:, 2:4],
                                                      JK_slogdet_tmp[:, 2:4],
                                                      moved_particles,
                                                      from_swap, from_swap_tmp,
                                                      kCM, phi_1, phi_t)

        t_L2 = (step_amplitude_swap *
                np.exp(1j*Ly*np.real(coords[moved_particles[0], 0])/2) *
                np.exp(1j*Ly*np.real(coords[moved_particles[1], 1])/2) *
                np.exp(-2j*phi_t))

        if np.abs(t_L2 - 1) > 1e-10:
            print('t(Lx*t) * exp(-i phi_t) = ', t_L2)
            test_passed = False

        ResetJastrowsCFL(jastrows, jastrows_tmp,
                         JK_matrix[..., 0], JK_matrix_tmp[...,  0], moved_particles[0])
        ResetJastrowsCFL(jastrows, jastrows_tmp,
                         JK_matrix[..., 1], JK_matrix_tmp[...,  1], Ne+moved_particles[1])

    if test_passed:
        print('Test passed!')
    else:
        print('Test failed!')


# add PBC test class
# add tests for jastrow update np.sum(jastrows-jastrows_tmp) and all the rest
