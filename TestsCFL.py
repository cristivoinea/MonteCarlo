import numpy as np
from utilities import fermi_sea_kx, fermi_sea_ky
from WavefnCFL import InitialWavefnCFL, TmpWavefnCFL, StepOneAmplitudeCFL, \
    ResetJastrowsCFL
from MonteCarloTorusSWAP import RandomConfig


def TestPeriodicityCFL(Ne: np.array, Ns: np.uint16, t: np.complex128,
                       kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                       phi_t: np.float64 = 0, nbr_tests: np.uint16 = 1):

    test_passed = True

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    coords = RandomConfig(Ne, Lx, Ly)
    Ks = (fermi_sea_kx[Ne]*2*np.pi/Lx + 1j*fermi_sea_ky[Ne]*2*np.pi/Ly)

    jastrows = np.ones((Ne, Ne, Ne), dtype=np.complex128)
    JK_matrix = np.zeros((Ne, Ne), dtype=np.complex128)
    JK_slogdet = np.zeros(2, dtype=np.complex128)
    JK_slogdet[0], JK_slogdet[1] = InitialWavefnCFL(t, Lx, coords, Ks,
                                                    jastrows, JK_matrix)
    jastrows_tmp = np.copy(jastrows)
    JK_matrix_tmp = np.copy(JK_matrix)
    JK_slogdet_tmp = np.zeros(2, dtype=np.complex128)

    for _ in range(nbr_tests):
        for moved_particle in range(coords.size):
            coords_tmp = np.copy(coords)
            coords_tmp[moved_particle] += Lx

            JK_slogdet_tmp[0], JK_slogdet_tmp[1] = TmpWavefnCFL(t, Lx, coords_tmp, Ks,
                                                                jastrows, jastrows_tmp,
                                                                JK_matrix_tmp, moved_particle)
            step_amplitude = StepOneAmplitudeCFL(Ns, t, coords, coords_tmp,
                                                 JK_slogdet, JK_slogdet_tmp,
                                                 moved_particle, Ks, kCM, phi_1, phi_t)

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
                                                                jastrows, jastrows_tmp,
                                                                JK_matrix_tmp, moved_particle)
            step_amplitude = StepOneAmplitudeCFL(Ns, t, coords, coords_tmp,
                                                 JK_slogdet, JK_slogdet_tmp,
                                                 moved_particle, Ks, kCM, phi_1, phi_t)
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
