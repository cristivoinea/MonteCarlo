import numpy as np
# from numba import njit, prange
from scipy.special import sph_harm
from .MonteCarloSphere import MonteCarloSphere


class MonteCarloSphereFreeFermions (MonteCarloSphere):
    slater: np.array
    slater_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array

    def InitialSlater(self, coords, slater):
        for i in range(self.Ne):
            slater[i, :] = sph_harm(
                self.Ls[i, 1], self.Ls[i, 0], coords[:, 1], coords[:, 0])

    def InitialWavefn(self):
        nbr_copies = self.coords.shape[0]//self.Ne
        self.moved_particles = np.zeros(nbr_copies, dtype=np.uint16)
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords[copy*self.Ne:(copy+1)*self.Ne],
                               self.slater[..., copy])
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def InitialWavefnSwap(self):
        coords_swap = self.coords[self.from_swap]
        for copy in range(2):
            self.InitialSlater(coords_swap[copy*self.Ne:(copy+1)*self.Ne],
                               self.slater[..., 2+copy])
            self.slogdet[:, 2+copy] = np.linalg.slogdet(
                self.slater[..., 2+copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def TmpWavefn(self):
        nbr_copies = self.coords.shape[0]//self.Ne
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords_tmp[copy*self.Ne:(copy+1)*self.Ne],
                               self.slater_tmp[..., copy])
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

    def TmpWavefnSwap(self):

        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]
        for copy in range(2):
            self.InitialSlater(coords_swap_tmp[copy*self.Ne:(copy+1)*self.Ne],
                               self.slater_tmp[..., 2+copy])
            self.slogdet_tmp[:, 2+copy] = np.linalg.slogdet(
                self.slater_tmp[..., 2+copy])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.shape[0]//self.Ne
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
        step_amplitude = 1
        step_exponent = 0
        for copy in range(2):

            step_amplitude *= (self.slogdet[0, copy+2] / self.slogdet[0, copy])
            step_exponent += (self.slogdet[1, copy+2] - self.slogdet[1, copy])

        return np.abs(step_amplitude*np.exp(step_exponent))

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (np.conj(self.slogdet[0, copy+2])
                               * self.slogdet[0, copy])

        step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def __init__(self, Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                 region_size, linear_size, step_size, nbr_copies=1,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.state = 'free_fermions'

        super().__init__(Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                         step_size, region_size, linear_size,
                         save_results, save_config, acceptance_ratio)

        self.slater = np.zeros(
            (Ne, Ne, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
