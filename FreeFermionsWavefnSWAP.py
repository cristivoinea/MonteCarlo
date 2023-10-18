import numpy as np
from numba import njit

Kxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1, 2, 0, -2, 0, 2,
                1, -1, -2, -2, -1, 1, 2, 2, -2, -2, 2, 3, 0, -3, 0,
                3, 1, -1, -3, -3, -1, 1, 3])
Kxs = Kxs.reshape((-1, 1))
Kys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2, 1, 2, 2, 1, -1, -2, -2, -1,
                2, 2, -2, -2, 0, 3, 0, -3, 1, 3, 3, 1, -1, -3, -3, -1])
Kys = Kys.reshape((-1, 1))


@njit
def StepOneAmplitudeFreeFermions(Lx: np.float64, Ly: np.float64,
                                 coords_initial: np.array, coords_final: np.array,
                                 # particle: np.uint8
                                 ) -> np.complex128:
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

    N = coords_initial.size
    wavefunction_initial = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_initial) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_initial)))
    wavefunction_final = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_final) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_final)))

    return np.linalg.det(wavefunction_final)/np.linalg.det(wavefunction_initial)


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
    wavefunction_0 = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 0])))
    wavefunction_1 = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 1])))
    wavefunction_0_SWAP = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 0])))
    wavefunction_1_SWAP = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 1])))

    total = ((np.linalg.det(wavefunction_0_SWAP)*np.linalg.det(wavefunction_1_SWAP)) /
             (np.linalg.det(wavefunction_0)*np.linalg.det(wavefunction_1)))

    return np.abs(total)


@njit
def InitialSignFreeFermions(Lx: np.float64, Ly: np.float64, coords: np.array,
                            coords_SWAP: np.array,
                            ) -> np.float64:
    """
    """
    N = coords.size
    wavefunction_0 = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 0])))
    wavefunction_1 = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords[:, 1])))
    wavefunction_0_SWAP = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 0]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 0])))
    wavefunction_1_SWAP = np.exp(
        1j * ((2*np.pi/Lx) * Kxs[:N] * np.real(coords_SWAP[:, 1]) +
              (2*np.pi/Ly) * Kys[:N] * np.imag(coords_SWAP[:, 1])))

    total = (np.linalg.det(wavefunction_0_SWAP)*np.linalg.det(wavefunction_1_SWAP) *
             np.linalg.det(wavefunction_0)*np.linalg.det(wavefunction_1))

    return total/np.abs(total)
