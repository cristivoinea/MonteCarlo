import numpy as np
# from numba import njit, prange
from scipy.special import sph_harm, lpmv, gamma, hyp2f1
from .MonteCarloSphere import MonteCarloSphere
from math import factorial, floor
from .utilities import LegendreDiagIntegral, LegendreOffDiagIntegral


class MonteCarloSphereFreeFermions (MonteCarloSphere):
    slater: np.array
    slater_tmp: np.array
    slater_inverse: np.array
    slater_inverse_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array

    def GetOverlapMatrix(self):
        overlap_matrix = np.zeros((self.N, self.N), dtype=np.complex128)
        x = np.cos(self.region_theta[1])

        norms = (2*self.Ls[:, 0]+1)/(4*np.pi)
        for i in range(self.N):
            if self.Ls[i, 1] > 0:
                for m in range(-int(self.Ls[i, 1]), int(self.Ls[i, 1])):
                    norms[i] /= (self.Ls[i, 0] + m + 1)
            elif self.Ls[i, 1] < 0:
                for m in range(int(self.Ls[i, 1]), -int(self.Ls[i, 1])):
                    norms[i] *= (self.Ls[i, 0] + m + 1)
        norms = np.sqrt(2*np.pi*norms)
        legendre = {}
        for l in range(floor(np.sqrt(self.N))+3):
            legendre[l] = np.zeros(2*l+1, dtype=np.float64)
            for m in range(-l, l+1):
                legendre[l][l+m] = lpmv(m, l, x)
        diag = LegendreDiagIntegral(x, floor(np.sqrt(self.N)), legendre)
        for i in range(self.N):
            overlap_matrix[i, i] = (diag[self.Ls[i, 0]][int(np.sum(self.Ls[i]))] *
                                    (norms[i]**2))
            for j in range(i+1, self.N):
                if self.Ls[i, 1] == self.Ls[j, 1]:
                    overlap_matrix[i, j] = (norms[i] * norms[j] *
                                            LegendreOffDiagIntegral(x, legendre, int(self.Ls[i, 0]),
                                                                    int(self.Ls[j, 0]), int(self.Ls[i, 1])))
                    overlap_matrix[j, i] = overlap_matrix[i, j]

        return overlap_matrix

    def InitialSlater(self, coords, slater):  # , slater_inverse):
        for i in range(self.N):
            slater[i, :] = sph_harm(
                self.Ls[i, 1], self.Ls[i, 0], coords[:, 1], coords[:, 0])
        # np.copyto(slater_inverse, np.linalg.inv(slater))

    def InitialWavefn(self):
        nbr_copies = self.coords.shape[0]//self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.int64)
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords[copy*self.N:(copy+1)*self.N],
                               self.slater[..., copy])  # , self.slater_inverse[..., copy])
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def InitialWavefnSwap(self):
        coords_swap = self.coords[self.from_swap]
        for copy in range(2):
            self.InitialSlater(coords_swap[copy*self.N:(copy+1)*self.N],
                               self.slater[..., 2+copy])  # , self.slater_inverse[..., 2+copy])
            self.slogdet[:, 2+copy] = np.linalg.slogdet(
                self.slater[..., 2+copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def TmpWavefn(self):
        for copy in range(self.moved_particles.size):
            p = self.moved_particles[copy]
            self.slater_tmp[:, p - copy*self.N, copy] = sph_harm(
                self.Ls[:, 1], self.Ls[:, 0],
                self.coords_tmp[p, 1], self.coords_tmp[p, 0])
            # d_det = (1+np.vdot(self.slater_inverse[p - copy*self.N, :, copy],
            #                   (self.slater_tmp[:, p - copy*self.N, copy] -
            #                   self.slater[:, p - copy*self.N, copy])))
            # self.slogdet_tmp[0, copy] = self.slogdet[0,
            #                                         copy]*d_det/np.abs(d_det)
            # self.slogdet_tmp[1, copy] = self.slogdet[1,
            #                                         copy]+np.log(np.abs(d_det))
            self.slogdet_tmp[:, copy] = np.linalg.slogdet(
                self.slater_tmp[..., copy])

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        np.copyto(self.slater, self.slater_tmp)
        np.copyto(self.slogdet, self.slogdet_tmp)
        # for copy in range(self.moved_particles.size):
        #    p = self.moved_particles[copy] - copy*self.N
        #    d_det = self.slogdet[0, copy]/self.slogdet_tmp[0, copy] * np.exp(
        #        self.slogdet[1, copy]-self.slogdet_tmp[1, copy])
        #    self.slater_inverse[..., copy] -= d_det*np.matmul(
        #        self.slater_inverse[..., copy], np.outer((self.slater_tmp[:, p, copy] -
        #                                                 self.slater[:, p, copy]), np.conj(self.slater_inverse[:, p, copy])))
        # print(self.slater_inverse[..., copy])

    def TmpWavefnSwap(self):

        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]
        for copy in range(2):
            swap_copy = self.to_swap_tmp[self.moved_particles[copy]] // self.N
            moved_in_swap = self.to_swap_tmp[self.moved_particles[copy]]
            self.slater_tmp[:, moved_in_swap % self.N, 2+swap_copy] = sph_harm(
                self.Ls[:, 1], self.Ls[:, 0],
                coords_swap_tmp[moved_in_swap, 1], coords_swap_tmp[moved_in_swap, 0])
            # self.InitialSlater(coords_swap_tmp[copy*self.N:(copy+1)*self.N],
            #                   self.slater_tmp[..., 2+copy])

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(self.slater_tmp[..., 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(self.slater_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.shape[0]//self.N
        for copy in range(nbr_copies):
            step_amplitude *= (self.slogdet_tmp[0, copy]/self.slogdet[0, copy])*np.exp(
                self.slogdet_tmp[1, copy]-self.slogdet[1, copy])
        return step_amplitude

    def StepAmplitudeTwoCopies(self) -> np.complex128:
        return self.StepAmplitude()

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved."""
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (self.slogdet_tmp[0, 2+copy]/self.slogdet[0, 2+copy])*np.exp(
                self.slogdet_tmp[1, 2+copy]-self.slogdet[1, 2+copy])
        return step_amplitude

    def InitialMod(self):
        """
        step_amplitude = 1
        step_exponent = 0

        for copy in range(2):

            step_amplitude *= (self.slogdet[0, copy+2] / self.slogdet[0, copy])
            step_exponent += (self.slogdet[1, copy+2] - self.slogdet[1, copy])

        return np.abs(step_amplitude*np.exp(step_exponent))"""

        return np.exp(self.slogdet[1, 2]+self.slogdet[1, 3] -
                      self.slogdet[1, 0] - self.slogdet[1, 1])

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (np.conj(self.slogdet[0, copy+2])
                               * self.slogdet[0, copy])

        step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def __init__(self, N, S, nbr_iter, nbr_nonthermal,
                 step_size, region_theta=180, region_phi=360, nbr_copies=1,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.state = 'free_fermions'

        super().__init__(N, S, nbr_iter, nbr_nonthermal,
                         step_size, region_theta, region_phi,
                         save_results, save_config, acceptance_ratio)

        self.S_eff = 0
        self.FillLambdaLevels()

        self.slater = np.zeros(
            (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slater_inverse = np.zeros(
            (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
