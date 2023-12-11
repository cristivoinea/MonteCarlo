import numpy as np
from WavefnLaughlin import StepOneAmplitudeLaughlin


def TestPeriodicityLaughlin(coords: np.array, Ns: np.uint16, t: np.complex128,
                            kCM: np.uint8 = 0, phi_1: np.float64 = 0,
                            phi_t: np.float64 = 0, nbr_tests: np.uint16 = 1):

    Ne = coords.size
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)

    for _ in range(nbr_tests):
        for particle in range(coords.size):
            coords_new = np.copy(coords)
            coords_new[particle] += Lx
            t_L1 = (StepOneAmplitudeLaughlin(Ns, t, coords, coords_new, particle, kCM, phi_1, phi_t) *
                    np.exp(-1j*Lx*np.imag(coords[particle])/2)*np.exp(-1j*phi_1))

            if np.abs(t_L1 - 1) > 1e-10:
                print('t(Lx) * exp(-i phi_1) = ', t_L1)
                return None

            coords_new = np.copy(coords)
            coords_new[particle] += Lx*t
            t_L2 = (StepOneAmplitudeLaughlin(Ns, t, coords, coords_new, particle, kCM, phi_1, phi_t) *
                    np.exp(1j*Ly*np.real(coords[particle])/2)*np.exp(-1j*phi_t))

        if np.abs(t_L2 - 1) > 1e-10:
            print('t(Lx*t) * exp(-i phi_t) = ', t_L2)
            return None

    print('Test passed!')
