import numpy as np
from numba import njit

Kxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1, 2, 0, -2, 0,
                2, 1, -1, -2, -2, -1, 1, 2, 2, -2, -2, 2,
                3, 0, -3, 0, 3, 1, -1, -3, -3, -1, 1, 3,
                3, 2, -2, -3, -3, -2, 2, 3, 4, 0, -4, 0,
                4, 1, -1, -4, -4, -1, 1, 4, 3, -3, -3, 3,
                4, 2, -2, -4, -4, -2, 2, 4])
Kxs = Kxs.reshape((-1, 1))
Kys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2,
                1, 2, 2, 1, -1, -2, -2, -1, 2, 2, -2, -2,
                0, 3, 0, -3, 1, 3, 3, 1, -1, -3, -3, -1,
                2, 3, 3, 2, -2, -3, -3, -2, 0, 4, 0, -4,
                1, 4, 4, 1, -1, -4, -4, -1, 3, 3, -3, -3,
                2, 4, 4, 2, -2, -4, -4, -2])
Kys = Kys.reshape((-1, 1))


@njit
def UpdateWavefnFreeFermions(wavefunction: np.array,
                             Lx: np.float64, Ly: np.float64,
                             coords: np.array):
    """
    Returns the ratio of wavefunctions (free fermions) for coordinates 
    to coordinates R_f, given that the particle with index p has moved.

    Parameters:

    Lx : torus dimension along x-axis
    Ly : torus dimension along y-axis
    coords : configuration of particles
    wavefunction : wavefunction array that will be updated in place
    """

    Ne = coords.size
    for copy in range(2):
        phase, logdet = np.linalg.slogdet(np.exp(
            1j * ((2*np.pi/Lx) * Kxs[:Ne] * np.real(coords[:, copy]) +
                  (2*np.pi/Ly) * Kys[:Ne] * np.imag(coords[:, copy]))))
        wavefunction[0, copy] = phase
        wavefunction[1, copy] = logdet


@njit  # (parallel=True)
def StepOneAmplitudeFreeFermions(Lx: np.float64, Ly: np.float64,
                                 wavefunction_initial: np.complex128, coords_final: np.array,
                                 # particle: np.uint8
                                 ) -> (np.complex128, np.array):
    """
    Returns the ratio of wavefunctions (free fermions) for coordinates 
    to coordinates R_f, given that the particle with index p has moved.

    Parameters:

    Lx : torus dimension along x-axis
    Ly : torus dimension along y-axis
    coords_initial : initial configuration of particles
    coords_final : final configuration of particles
    particle : indice of particle that moves

    Output:

    r : ratio of wavefunctions R_f/R_i
    """

    Ne = coords_final.size
    wavefunction_final = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:Ne] * np.real(coords_final) +
              (2*np.pi/Ly) * Kys[:Ne] * np.imag(coords_final))))

    return ((wavefunction_final[0]/wavefunction_initial[0])*np.exp(
        wavefunction_final[1]-wavefunction_initial[1]), wavefunction_final)


@njit  # (parallel=True)
def StepOneAmplitudeFreeFermionsSWAP(Lx: np.float64, Ly: np.float64,
                                     coords_SWAP_initial: np.array, coords_SWAP_final: np.array,
                                     p_swap_order: np.uint8
                                     ) -> np.complex128:
    """
    """

    return (StepOneAmplitudeFreeFermions(Lx, Ly, coords_SWAP_initial[:, 0],
                                         coords_SWAP_final[:, 0]) *
            StepOneAmplitudeFreeFermions(Lx, Ly, coords_SWAP_initial[:, 1],
                                         coords_SWAP_final[:, 1]))


@njit
def InitialModFreeFermions(Lx: np.float64, Ly: np.float64, coords: np.array,
                           coords_SWAP: np.array,
                           ) -> np.float64:
    """
    """
    N = coords.size
    wavefunction_0 = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 0]))))
    wavefunction_1 = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 1]))))
    wavefunction_0_SWAP = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 0]))))
    wavefunction_1_SWAP = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 1]))))

    total = np.exp(wavefunction_0_SWAP[1] + wavefunction_1_SWAP[1] -
                   wavefunction_0[1] - wavefunction_1[1])

    return np.abs(total)


@njit
def InitialSignFreeFermions(Lx: np.float64, Ly: np.float64, coords: np.array,
                            coords_SWAP: np.array,
                            ) -> np.float64:
    """
    """
    N = coords.size
    wavefunction_0 = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 0]))))
    wavefunction_1 = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 1]))))
    wavefunction_0_SWAP = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 0]))))
    wavefunction_1_SWAP = np.linalg.slogdet(np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 1]))))

    return ((wavefunction_0[0] * wavefunction_1[0]) /
            (wavefunction_0_SWAP[0] * wavefunction_1_SWAP[0]))

    # total = (wavefunction_0_SWAP*wavefunction_1_SWAP *
    #         wavefunction_0*wavefunction_1)

    # return total/np.abs(total)
