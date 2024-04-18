import numpy as np
from numba import njit
from scipy.special import sph_harm
from .MonteCarloSphere import MonteCarloSphere
from .fast_math import MonopoleHarmonics
from math import comb


# @njit
def njit_UpdateSlaterUnproj(S_eff, coords, moved_particle, slater, Ls):
    if S_eff == 0:
        slater[:, moved_particle] = sph_harm(
            Ls[:, 1], Ls[:, 0], coords[moved_particle, 1], coords[moved_particle, 0])
    else:
        slater[:, moved_particle] = MonopoleHarmonics(
            S_eff, np.array(Ls[:, 0]-np.float64(S_eff)/2, dtype=np.int64), np.array(
                Ls[:, 1] + Ls[:, 0], dtype=np.int64), coords[moved_particle, 0], coords[moved_particle, 1])


def njit_UpdateJastrows(S_eff: np.uint64, coords_tmp: np.array, spinors_tmp: np.array,
                        jastrows_tmp: np.array, slater_tmp: np.array,
                        Ls: np.array, p: np.uint16):
    jastrows_tmp[:, p, 0, 0] = (
        spinors_tmp[:, 0]*spinors_tmp[p, 1] -
        spinors_tmp[p, 0]*spinors_tmp[:, 1])
    jastrows_tmp[p, :, 0, 0] = - \
        jastrows_tmp[:, p, 0, 0]
    jastrows_tmp[p, p] = 1

    njit_UpdateSlaterUnproj(S_eff, coords_tmp, p, slater_tmp, Ls)


# @njit
def njit_UpdateJastrowsSwap(coords_tmp: np.array, spinors_tmp: np.array,
                            jastrows_tmp: np.array, moved_particles: np.array,
                            to_swap_tmp: np.array):
    N = coords_tmp.shape[0]//2
    """
    mask_1 = ((to_swap_tmp[moved_particles[1]] // N)
              == (to_swap_tmp[:N] // N))
    spinors_1 = spinors_tmp[:N][mask_1]
    jastrows_tmp[mask_1, moved_particles[1], 0, 0] = (
        spinors_1[:, 0]*spinors_tmp[moved_particles[1], 1] -
        spinors_tmp[moved_particles[1], 0]*spinors_1[:, 1])
    jastrows_tmp[moved_particles[1], mask_1, 0, 0] = - \
        jastrows_tmp[mask_1, moved_particles[1], 0, 0]
    jastrows_tmp[moved_particles[1], moved_particles[1], 0, 0] = 1

    mask_2 = ((to_swap_tmp[moved_particles[0]] // N)
              == (to_swap_tmp[N:] // N))
    spinors_2 = spinors_tmp[N:][mask_2]
    jastrows_tmp[mask_2+N, moved_particles[0], 0, 0] = (
        spinors_2[:, 0]*spinors_tmp[moved_particles[0], 1] -
        spinors_tmp[moved_particles[0], 0]*spinors_2[:, 1])
    jastrows_tmp[moved_particles[0], mask_2+N, 0, 0] = - \
        jastrows_tmp[mask_2+N, moved_particles[0], 0, 0]
    jastrows_tmp[moved_particles[0], moved_particles[0], 0, 0] = 1
    """
    for i in range(N):
        if (to_swap_tmp[moved_particles[1]] // N) == (to_swap_tmp[i] // N):
            jastrows_tmp[i, moved_particles[1], 0, 0] = (
                spinors_tmp[i, 0]*spinors_tmp[moved_particles[1], 1] -
                spinors_tmp[moved_particles[1], 0]*spinors_tmp[i, 1])
            jastrows_tmp[moved_particles[1], i, 0, 0] = - \
                jastrows_tmp[i, moved_particles[1], 0, 0]

            # update cross copy jastrows for particle in copy 1
        if (to_swap_tmp[moved_particles[0]] // N) == (to_swap_tmp[i+N] // N):
            jastrows_tmp[i+N, moved_particles[0], 0, 0] = (
                spinors_tmp[i+N, 0]*spinors_tmp[moved_particles[0], 1] -
                spinors_tmp[moved_particles[0], 0]*spinors_tmp[i+N, 1])
            jastrows_tmp[moved_particles[0], i+N, 0, 0] = - \
                jastrows_tmp[i+N, moved_particles[0], 0, 0]


@njit
def njit_StepAmplitudeTwoCopiesSwap(N: np.int64, vortices: np.int64, jastrows: np.array,
                                    jastrows_tmp: np.array, slogdet: np.array, slogdet_tmp: np.array,
                                    from_swap: np.array, from_swap_tmp: np.array) -> np.complex128:
    """
    Returns the ratio of wavefunctions for coordinates R_i
    to coordinates R_f, given that the particle with index p has moved."""
    step_amplitude = 1
    for copy in range(2):
        step_amplitude *= (slogdet_tmp[0, 2+copy] /
                           slogdet[0, 2+copy])

        for n in range(copy*N, (copy+1)*N):
            step_amplitude *= np.prod(np.exp((slogdet_tmp[1, 2+copy]-slogdet[1, 2+copy]) / (N*(N-1)/2)) *
                                      np.power(jastrows_tmp[from_swap_tmp[n], from_swap_tmp[n+1: (copy+1)*N], 0, 0] /
                                               jastrows[from_swap[n], from_swap[n+1: (copy+1)*N], 0, 0],  # nopep8
                                               vortices))

    return step_amplitude


class MonteCarloSphereCFL (MonteCarloSphere):
    spinors: np.array
    spinors_tmp: np.array
    jastrows: np.array
    jastrows_tmp: np.array
    slater: np.array
    slater_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array

    def InitialJastrows(self):
        for c in range(self.moved_particles.size):
            for i in range(c*self.N, (c+1)*self.N):
                spinors_slice = self.spinors[i+1:(c+1)*self.N, :]
                self.jastrows[i, i+1:(c+1)*self.N, 0, 0] = (
                    self.spinors[i, 0]*spinors_slice[:, 1] - spinors_slice[:, 0]*self.spinors[i, 1])
                self.jastrows[i+1:(c+1)*self.N, i, 0, 0] = - \
                    self.jastrows[i, i+1:(c+1)*self.N, 0, 0]

    def InitialJastrowsSwap(self):
        for i in range(self.N):
            self.jastrows[i, self.N:2*self.N, 0, 0] = (
                self.spinors[i, 0]*self.spinors[self.N:2*self.N, 1] -
                self.spinors[self.N:2*self.N, 0]*self.spinors[i, 1])
            self.jastrows[self.N:2*self.N, i, 0, 0] = - \
                self.jastrows[i, self.N:2*self.N, 0, 0]

    def InitialSlater(self, coords, slater):
        if self.S_eff == 0:
            for i in range(self.N):
                slater[i, :] = sph_harm(
                    self.Ls[i, 1], self.Ls[i, 0], coords[:, 1], coords[:, 0])
        else:
            for i in range(self.N):
                slater[i, :] = MonopoleHarmonics(self.S_eff, np.int64(self.Ls[i, 0]-self.S_eff/2), np.int64(
                    self.Ls[i, 1] + self.Ls[i, 0]), coords[:, 0], coords[:, 1])

    def InitialWavefn(self):
        nbr_copies = self.coords.shape[0]//self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.uint16)
        self.spinors[:, 0] = (np.cos(self.coords[..., 0]/2) *
                              np.exp(1j*self.coords[..., 1]/2))
        self.spinors[:, 1] = (np.sin(self.coords[..., 0]/2) *
                              np.exp(-1j*self.coords[..., 1]/2))
        self.InitialJastrows()
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords[copy*self.N:(copy+1)*self.N],
                               self.slater[..., copy])
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

        np.copyto(self.spinors_tmp, self.spinors)
        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def InitialWavefnSwap(self):
        coords_swap = self.coords[self.from_swap]
        self.InitialJastrowsSwap()
        for copy in range(2):
            self.InitialSlater(coords_swap[copy*self.N:(copy+1)*self.N],
                               self.slater[..., 2+copy])
            self.slogdet[:, 2+copy] = np.linalg.slogdet(
                self.slater[..., 2+copy])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows_tmp[p, :, ...], self.jastrows[p, :, ...])
            np.copyto(self.jastrows_tmp[:, p, ...], self.jastrows[:, p, ...])
            np.copyto(self.spinors_tmp[p], self.spinors[p])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows[p, :, ...], self.jastrows_tmp[p, :, ...])
            np.copyto(self.jastrows[:, p, ...], self.jastrows_tmp[:, p, ...])
            np.copyto(self.spinors[p], self.spinors_tmp[p])

        np.copyto(self.slater, self.slater_tmp)
        np.copyto(self.slogdet, self.slogdet_tmp)

    def TmpWavefn(self):
        for copy in range(self.moved_particles.size):
            p = self.moved_particles[copy]
            phase = np.exp(1j*self.coords_tmp[p, 1]/2)
            self.spinors_tmp[p, 0] = np.cos(
                self.coords_tmp[p, 0]/2)*phase
            self.spinors_tmp[p, 1] = np.sin(
                self.coords_tmp[p, 0]/2)/phase
            njit_UpdateJastrows(self.S_eff, self.coords_tmp[copy*self.N:(copy+1)*self.N],
                                self.spinors_tmp[copy *
                                                 self.N:(copy+1)*self.N],
                                self.jastrows_tmp[copy*self.N:(copy+1)*self.N,
                                                  copy*self.N:(copy+1)*self.N, ...],
                                self.slater_tmp[..., copy], self.Ls,
                                self.moved_particles[copy]-copy*self.N)
            self.slogdet_tmp[:, copy] = np.linalg.slogdet(
                self.slater_tmp[..., copy])

    def TmpWavefnSwap(self):
        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]
        njit_UpdateJastrowsSwap(self.coords_tmp, self.spinors_tmp,
                                self.jastrows_tmp, self.moved_particles,
                                self.to_swap_tmp)

        for copy in range(2):
            swap_copy = self.to_swap_tmp[self.moved_particles[copy]] // self.N
            njit_UpdateSlaterUnproj(self.S_eff, coords_swap_tmp[swap_copy*self.N:(swap_copy+1)*self.N],
                                    self.to_swap_tmp[self.moved_particles[copy]
                                                     ] % self.N,
                                    self.slater_tmp[..., 2+swap_copy], self.Ls)

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(self.slater_tmp[..., 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(self.slater_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.shape[0]//self.N
        for copy in range(nbr_copies):
            step_amplitude *= (self.slogdet_tmp[0, copy]/self.slogdet[0, copy])

            step_amplitude *= (np.prod(np.exp((self.slogdet_tmp[1, copy]-self.slogdet[1, copy])/self.N) *
                                       np.power(self.jastrows_tmp[self.moved_particles[copy],
                                                                  copy*self.N:(copy+1)*self.N, 0, 0] /
                                                self.jastrows[self.moved_particles[copy],
                                                              copy*self.N:(copy+1)*self.N, 0, 0],
                                                self.vortices)))
        return step_amplitude

    def StepAmplitudeTwoCopies(self) -> np.complex128:
        return self.StepAmplitude()

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved.
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (self.slogdet_tmp[0, 2+copy] /
                               self.slogdet[0, 2+copy])

            for n in range(copy*self.N, (copy+1)*self.N):
                step_amplitude *= np.prod(np.exp((self.slogdet_tmp[1, 2+copy]-self.slogdet[1, 2+copy]) / (self.N*(self.N-1)/2)) *
                                          np.power(self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[n+1: (copy+1)*self.N], 0, 0] /
                                                   self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.N], 0, 0],  # nopep8
                                                   self.vortices))

        return step_amplitude"""
        return njit_StepAmplitudeTwoCopiesSwap(self.N, self.vortices, self.jastrows, self.jastrows_tmp,
                                               self.slogdet, self.slogdet_tmp, self.from_swap, self.from_swap_tmp)

    def InitialMod(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (self.slogdet[0, copy+2] / self.slogdet[0, copy])

            for n in range(copy*self.N, (copy+1)*self.N):
                step_amplitude *= np.prod((np.exp((self.slogdet[1, copy+2] - self.slogdet[1, copy]) / (self.N*(self.N-1)/2)) *
                                           np.power((self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.N], 0, 0] /
                                                     self.jastrows[n, n+1:(copy+1)*self.N, 0, 0]), self.vortices)))

        return np.abs(step_amplitude)

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (np.conj(self.slogdet[0, copy+2])
                               * self.slogdet[0, copy])

            for n in range(copy*self.N, (copy+1)*self.N):
                step_amplitude *= np.prod(np.power((np.conj(
                    self.jastrows[self.from_swap[n], self.from_swap[n+1:(copy+1)*self.N], 0, 0]) *
                    self.jastrows[n, n+1:(copy+1)*self.N, 0, 0]), self.vortices))
                step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def DensityCF(self):
        cf_density = 0
        for i in range(self.N):
            if self.InsideRegion(self.coords[i]):
                cf_density += (np.prod(np.sqrt(self.N)*np.power(np.abs(self.jastrows[i, :, 0, 0]),
                                                                2*self.vortices)))

        return cf_density

    def __init__(self, N, S, nbr_iter, nbr_nonthermal,
                 step_size, region_theta=180, region_phi=360, nbr_copies=1,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.state = 'cfl'

        super().__init__(N, S, nbr_iter, nbr_nonthermal,
                         step_size, region_theta, region_phi,
                         save_results, save_config, acceptance_ratio)

        if self.S/(self.N-1) == np.floor(self.S/(self.N-1)):
            print("Initialising HLR-CFL.")
            self.vortices = self.S/(self.N-1)
        elif self.S == 2*self.N-1:
            print("Initialising Son-CFL.")
            self.vortices = 2

        self.S_eff = np.int64(self.S - self.vortices*(self.N-1))

        self.FillLambdaLevels()

        self.spinors = np.zeros((nbr_copies*N, 2), dtype=np.complex128)
        self.jastrows = np.ones(
            (nbr_copies*N, nbr_copies*N, 1, 1), dtype=np.complex128)
        self.slater = np.zeros(
            (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        self.spinors_tmp = np.copy(self.spinors)
        self.jastrows_tmp = np.copy(self.jastrows)
        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
