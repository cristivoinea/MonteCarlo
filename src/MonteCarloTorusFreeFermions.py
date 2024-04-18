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
            overlap_matrix[i, i] = np.pi*R*R
            for j in range(i+1, self.N):
                k_rel = np.sqrt((Kxs[i]-Kxs[j])**2 + (Kys[i]-Kys[j])**2)
                if region_geometry == "square":
                    overlap_matrix[i, j] = np.sinc(
                        R*(Kxs[i]-Kxs[j]))*np.sinc(R*(Kys[i]-Kys[j]))*(R)**2
                elif region_geometry == "circle":
                    overlap_matrix[i, j] = R/k_rel * j1(2*np.pi*k_rel*R)
                    # overlap_matrix[i, j] = np.pi*R*R*hyp0f1(
                    #    2, -((Kxs[i]-Kxs[j])**2 + (Kys[i]-Kys[j])**2)*(R*np.pi)**2)/gamma(2)
                overlap_matrix[j, i] = overlap_matrix[i, j]

        return overlap_matrix

    def InitialSlater(self, coords, slater):
        np.copyto(slater, np.exp(1j * (np.reshape(np.real(self.Ks), (-1, 1)) * np.real(coords) +
                                       np.reshape(np.imag(self.Ks), (-1, 1)) * np.imag(coords))))

    def InitialWavefn(self):
        nbr_copies = self.coords.size//self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.uint16)
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
        nbr_copies = self.coords.size//self.N
        for copy in range(nbr_copies):
            self.InitialSlater(self.coords_tmp[copy*self.N:(copy+1)*self.N],
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
            self.InitialSlater(coords_swap_tmp[copy*self.N:(copy+1)*self.N],
                               self.slater_tmp[..., 2+copy])
            self.slogdet_tmp[:, 2+copy] = np.linalg.slogdet(
                self.slater_tmp[..., 2+copy])

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

    def __init__(self, N, S, t, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, area_size=0, linear_size=0, nbr_copies=1,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.state = 'free_fermions'

        super().__init__(N, S, t, nbr_iter, nbr_nonthermal, region_geometry,
                         step_size, area_size, linear_size,
                         save_results, save_config, acceptance_ratio)

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
