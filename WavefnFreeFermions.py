import numpy as np
from numba import njit


# @njit
def UpdateWavefnFreeFermions(wavefunction: np.array,
                             coords: np.array, Kxs: np.array,
                             Kys: np.array):
    """
    Returns the ratio of wavefunctions (free fermions) for coordinates 
    to coordinates R_f, given that the particle with index p has moved.

    Parameters:

    Lx : torus dimension along x-axis
    Ly : torus dimension along y-axis
    coords : configuration of particles
    wavefunction : wavefunction array that will be updated in place
    """

    for copy in range(2):
        phase, logdet = np.linalg.slogdet(np.exp(
            1j * (Kxs * np.real(coords[:, copy]) +
                  Kys * np.imag(coords[:, copy]))))
        wavefunction[0, copy] = phase
        wavefunction[1, copy] = logdet


# @njit  # (parallel=True)
def StepOneAmplitudeFreeFermions(wavefunction_initial: np.complex128,
                                 coords_final: np.array, Kxs: np.array,
                                 Kys: np.array
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

    wavefunction_final = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords_final) +
              Kys * np.imag(coords_final))))

    return ((wavefunction_final[0]/wavefunction_initial[0])*np.exp(
        wavefunction_final[1]-wavefunction_initial[1]), wavefunction_final)


@njit  # (parallel=True)
def StepOneAmplitudeFreeFermionsSWAP(coords_SWAP_initial: np.array, coords_SWAP_final: np.array,
                                     Kxs: np.array, Kys: np.array, p_swap_order: np.uint8
                                     ) -> np.complex128:
    """
    """

    return (StepOneAmplitudeFreeFermions(coords_SWAP_initial[:, 0],
                                         coords_SWAP_final[:, 0], Kxs, Kys) *
            StepOneAmplitudeFreeFermions(coords_SWAP_initial[:, 1],
                                         coords_SWAP_final[:, 1], Kxs, Kys))


def InitialModFreeFermions(coords: np.array, coords_SWAP: np.array,
                           Kxs: np.array, Kys: np.array
                           ) -> np.float64:
    """
    """
    wavefunction_0 = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords[:, 0]) +
              Kys * np.imag(coords[:, 0]))))
    wavefunction_1 = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords[:, 1]) +
              Kys * np.imag(coords[:, 1]))))
    wavefunction_0_SWAP = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords_SWAP[:, 0]) +
              Kys * np.imag(coords_SWAP[:, 0]))))
    wavefunction_1_SWAP = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords_SWAP[:, 1]) +
              Kys * np.imag(coords_SWAP[:, 1]))))

    total = np.exp(wavefunction_0_SWAP[1] + wavefunction_1_SWAP[1] -
                   wavefunction_0[1] - wavefunction_1[1])

    return np.abs(total)


def InitialSignFreeFermions(coords: np.array, coords_SWAP: np.array,
                            Kxs: np.array, Kys: np.array
                            ) -> np.float64:
    """
    """
    wavefunction_0 = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords[:, 0]) +
              Kys * np.imag(coords[:, 0]))))
    wavefunction_1 = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords[:, 1]) +
              Kys * np.imag(coords[:, 1]))))
    wavefunction_0_SWAP = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords_SWAP[:, 0]) +
              Kys * np.imag(coords_SWAP[:, 0]))))
    wavefunction_1_SWAP = np.linalg.slogdet(np.exp(
        1j * (Kxs * np.real(coords_SWAP[:, 1]) +
              Kys * np.imag(coords_SWAP[:, 1]))))

    return ((wavefunction_0[0] * wavefunction_1[0]) /
            (wavefunction_0_SWAP[0] * wavefunction_1_SWAP[0]))

    # total = (wavefunction_0_SWAP*wavefunction_1_SWAP *
    #         wavefunction_0*wavefunction_1)

    # return total/np.abs(total)
