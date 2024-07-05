import numpy as np
# from numba import njit, prange
from .MonteCarloTorus import MonteCarloTorus
from .utilities import fermi_sea_kx, fermi_sea_ky
from scipy.special import gamma, hyp0f1, j1


class MonteCarloTorusFreeFermions (MonteCarloTorus):
    slater: np.array
    slater_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array
    Ks: np.array

    def GetFermiSea(self):
        kF = {97: 30, 109: 35, 137: 41, 145: 45, 177: 53, 221: 68, 241: 75,
              277: 85, 341: 106, 421: 130, 481: 150, 553: 173, 657: 205,
              749: 238, 989: 315}
        cutoff = int(np.floor(np.sqrt(kF[self.N])))+1
        kx = ky = 0
        kxs = [0]
        kys = [0]
        for kx in range(1, cutoff):
            kxs.append(kx)
            kxs.append(-kx)
            kys.append(0)
            kys.append(0)

            kxs.append(0)
            kxs.append(0)
            kys.append(kx)
            kys.append(-kx)

        for kx in range(1, cutoff):
            for ky in range(1, cutoff):
                if kx**2 + ky**2 <= kF[self.N]:
                    kxs.append(kx)
                    kys.append(ky)

                    kxs.append(kx)
                    kys.append(-ky)

                    kxs.append(-kx)
                    kys.append(ky)

                    kxs.append(-kx)
                    kys.append(-ky)
        if (len(kxs) != self.N):
            print("error")
        return np.array(kxs), np.array(kys)

    def GetOverlapMatrix(self):
        if "square" in self.region_details:
            region_geometry = "square"
        elif "circle" in self.region_details:
            region_geometry = "circle"
        overlap_matrix = np.zeros((self.N, self.N), dtype=np.complex128)
        Kxs = np.real(self.Ks) * self.Lx/(2*np.pi)
        Kys = np.imag(self.Ks) * self.Lx/(2*np.pi)
        R = self.boundary/self.Lx
        for i in range(self.N):
            if region_geometry == "circle":
                overlap_matrix[i, i] = np.pi*R*R
                k_rel = np.sqrt((Kxs[i]-Kxs[:])**2 + (Kys[i]-Kys[:])**2)
                overlap_matrix[i,  i+1:] = (R/k_rel[i+1:] *
                                            j1(2*np.pi*k_rel[i+1:]*R))
                overlap_matrix[i+1:, i] = overlap_matrix[i, i+1:]
            elif region_geometry == "square":
                overlap_matrix[i, i] = 4*R*R
                overlap_matrix[i, i+1:] = 4*(np.sinc(2*R*(Kxs[i]-Kxs[i+1:])) *
                                             np.sinc(2*R*(Kys[i]-Kys[i+1:]))*R*R)
                overlap_matrix[i+1:, i] = overlap_matrix[i, i+1:]

        return overlap_matrix

    def InitialSlater(self, coords, slater):
        np.copyto(slater, np.exp(1j * (np.reshape(np.real(self.Ks), (-1, 1)) * np.real(coords) +
                                       np.reshape(np.imag(self.Ks), (-1, 1)) * np.imag(coords))))

    def InitialWavefn(self):
        nbr_copies = self.coords.size//self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.int64)
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords[copy*self.N:(copy+1)*self.N],
                               self.slater[..., copy])
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def InitialWavefnSwap(self):
        coords_swap = self.coords[self.from_swap]
        for copy in range(2):
            self.InitialSlater(coords_swap[copy*self.N:(copy+1)*self.N],
                               self.slater[..., 2+copy])
            self.slogdet[:, 2+copy] = np.linalg.slogdet(
                self.slater[..., 2+copy])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

    def TmpWavefn(self):
        for copy in range(self.moved_particles.size):
            p = self.moved_particles[copy]
            self.slater_tmp[:, p - copy*self.N, copy] = np.exp(
                1j * (np.real(self.Ks) * np.real(self.coords_tmp[p]) +
                      np.imag(self.Ks) * np.imag(self.coords_tmp[p])))
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
            swap_copy = self.to_swap_tmp[self.moved_particles[copy]] // self.N
            moved_in_swap = self.to_swap_tmp[self.moved_particles[copy]]
            self.slater_tmp[:, moved_in_swap % self.N, 2+swap_copy] = np.exp(
                1j * (np.real(self.Ks) * np.real(coords_swap_tmp[moved_in_swap]) +
                      np.imag(self.Ks) * np.imag(coords_swap_tmp[moved_in_swap])))

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(self.slater_tmp[..., 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(self.slater_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.size//self.N
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
        return np.exp(self.slogdet[1, 2]+self.slogdet[1, 3] -
                      self.slogdet[1, 0] - self.slogdet[1, 1])

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (np.conj(self.slogdet[0, copy+2])
                               * self.slogdet[0, copy])

        step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def __init__(self, N, S, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, area_size=0, linear_size=0, nbr_copies=1, t=1j,
                 save_results=True, save_last_config=True,
                 save_all_config=True, acceptance_ratio=0):

        self.state = 'free_fermions'

        super().__init__(N, S, t, nbr_iter, nbr_nonthermal, region_geometry,
                         step_size, area_size, linear_size,
                         save_results, save_last_config,
                         save_all_config, acceptance_ratio)

        if self.N in fermi_sea_kx:
            self.Ks = (fermi_sea_kx[self.N]*2*np.pi/self.Lx +
                       1j*fermi_sea_ky[self.N]*2*np.pi/self.Ly)
        else:
            kxs, kys = self.GetFermiSea()
            self.Ks = (kxs*2*np.pi/self.Lx +
                       1j*kys*2*np.pi/self.Ly)

        self.slater = np.zeros(
            (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
