import numpy as np
from numba import njit, prange
from pfapack import pfaffian as pf
from .MonteCarloTorus import MonteCarloTorus
from .utilities import fermi_sea_kx, fermi_sea_ky
from .fast_math import ThetaFunction, ThetaFunctionVectorized


def njit_UpdateSlater(coords, moved_particle, slater, Ks):
    slater[:, moved_particle] = np.exp(1j * (np.real(Ks) * np.real(coords[moved_particle]) +
                                             np.imag(Ks) * np.imag(coords[moved_particle])))


@njit(parallel=True)
def njit_UpdateJastrows(t: np.complex128, Lx: np.float64,
                        coords_tmp: np.array, Ks: np.array,
                        jastrows: np.array, jastrows_tmp: np.array,
                        JK_coeffs: np.array, slater_tmp: np.array,
                        moved_particle: np.uint16):
    Ne = coords_tmp.size
    slater_tmp[:, moved_particle] = np.exp(
        1j*np.real(Ks)*coords_tmp[moved_particle])
    for k in prange(Ne):
        for i in range(Ne):
            if i != moved_particle:
                for l in range(JK_coeffs.shape[1]):
                    jastrows_tmp[i, moved_particle, k, l] = ThetaFunction(
                        (coords_tmp[i] - coords_tmp[moved_particle] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                    if JK_coeffs[0, l] != 0:
                        jastrows_tmp[moved_particle, i, k, l] = ThetaFunction(
                            (coords_tmp[moved_particle] - coords_tmp[i] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                    else:
                        jastrows_tmp[moved_particle, i, k, l] = - \
                            jastrows_tmp[i, moved_particle, k, l]

                    slater_tmp[k, i] *= np.power((jastrows_tmp[i, moved_particle, k, l] /
                                                  jastrows[i, moved_particle, k, l]), JK_coeffs[1, l])

                    slater_tmp[k, moved_particle] *= np.power(jastrows_tmp[moved_particle, i, k, l],
                                                              JK_coeffs[1, l])


@njit(parallel=True)
def njit_UpdatePfJastrows(t: np.complex128, Lx: np.float64,
                          coords_tmp: np.array, jastrows_tmp: np.array,
                          pf_jastrows_tmp: np.array,
                          pf_matrix_tmp: np.array,
                          moved_particle: np.uint16):
    Ne = coords_tmp.size
    for i in range(Ne):
        if i != moved_particle:
            pf_jastrows_tmp[i, moved_particle] = ThetaFunction(
                (coords_tmp[i] - coords_tmp[moved_particle])/Lx, t, 0, 0)
            pf_jastrows_tmp[moved_particle,
                            i] = pf_jastrows_tmp[i, moved_particle]

    pf_matrix_tmp[:, moved_particle] = (pf_jastrows_tmp[:, moved_particle] /
                                        jastrows_tmp[:, moved_particle, 0, 0])
    pf_matrix_tmp[moved_particle, :] = - pf_matrix_tmp[:, moved_particle]


@njit(parallel=True)
def njit_UpdateJastrowsSwap(t: np.complex128, Lx: np.float64, coords: np.array, Ks: np.array,
                            jastrows: np.array, JK_coeffs: np.array, moved_particles: np.array,
                            to_swap: np.array, flag_proj: np.bool_):
    Ne = coords.size//2
    if flag_proj:
        for k in prange(Ne):
            for i in range(Ne):
                for l in range(JK_coeffs.shape[1]):
                    # update cross copy jastrows for particle in copy 2
                    if (to_swap[moved_particles[1]] // Ne) == (to_swap[i] // Ne):
                        jastrows[i, moved_particles[1], k, l] = ThetaFunction(
                            (coords[i] - coords[moved_particles[1]] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                        if JK_coeffs[0, l] != 0:
                            jastrows[moved_particles[1], i, k, l] = ThetaFunction(
                                (coords[moved_particles[1]] - coords[i] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                        else:
                            jastrows[moved_particles[1], i, k, l] = - \
                                jastrows[i, moved_particles[1], k, l]

                    # update cross copy jastrows for particle in copy 1
                    if (to_swap[moved_particles[0]] // Ne) == (to_swap[i+Ne] // Ne):
                        jastrows[i+Ne, moved_particles[0], k, l] = ThetaFunction(
                            (coords[i+Ne] - coords[moved_particles[0]] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)

                        if JK_coeffs[0, l] != 0:
                            jastrows[moved_particles[0], i+Ne, k, l] = ThetaFunction(
                                (coords[moved_particles[0]] - coords[i+Ne] + JK_coeffs[0, l]*1j*Ks[k])/Lx, t, 1/2, 1/2)
                        else:
                            jastrows[moved_particles[0], i+Ne, k, l] = - \
                                jastrows[i+Ne, moved_particles[0], k, l]
    else:
        for i in range(Ne):
            if (to_swap[moved_particles[1]] // Ne) == (to_swap[i] // Ne):
                jastrows[i, moved_particles[1], 0, 0] = ThetaFunction(
                    (coords[i] - coords[moved_particles[1]])/Lx, t, 1/2, 1/2)
                jastrows[moved_particles[1], i, 0, 0] = - \
                    jastrows[i, moved_particles[1], 0, 0]

            # update cross copy jastrows for particle in copy 1
            if (to_swap[moved_particles[0]] // Ne) == (to_swap[i+Ne] // Ne):
                jastrows[i+Ne, moved_particles[0], 0, 0] = ThetaFunction(
                    (coords[i+Ne] - coords[moved_particles[0]])/Lx, t, 1/2, 1/2)
                jastrows[moved_particles[0], i+Ne, 0, 0] = - \
                    jastrows[i+Ne, moved_particles[0], 0, 0]


@njit  # (parallel=True)
def njit_UpdatePfJastrowsSwap(t: np.complex128, Lx: np.float64, coords: np.array,
                              pf_jastrows: np.array, moved_particles: np.array,
                              to_swap: np.array):
    Ne = coords.size//2
    for i in range(Ne):
        if (to_swap[moved_particles[1]] // Ne) == (to_swap[i] // Ne):
            pf_jastrows[i, moved_particles[1]] = ThetaFunction(
                (coords[i] - coords[moved_particles[1]])/Lx, t, 1/2, 1/2)
            pf_jastrows[moved_particles[1], i] = \
                pf_jastrows[i, moved_particles[1]]

        if (to_swap[moved_particles[0]] // Ne) == (to_swap[i+Ne] // Ne):
            pf_jastrows[i+Ne, moved_particles[0]] = ThetaFunction(
                (coords[i+Ne] - coords[moved_particles[0]])/Lx, t, 1/2, 1/2)
            pf_jastrows[moved_particles[0], i+Ne] = \
                pf_jastrows[i+Ne, moved_particles[0]]


@njit  # (parallel=True)
def njit_GetJKMatrixSwap(Ne, coords, Ks, from_swap, jastrows, slater, JK_coeffs, flag_proj):
    if flag_proj:
        for n in range(Ne):
            for m in range(2*Ne):
                slater[n, m % Ne, 2+m//Ne] = np.exp(
                    2j*np.real(Ks[n]) * coords[from_swap[m]]/2)
                for j in range(Ne):
                    for l in range(JK_coeffs.shape[1]):
                        slater[n, m % Ne, 2+m//Ne] *= np.power(
                            jastrows[from_swap[m],
                                     from_swap[j+(m//Ne)*Ne], n, l],
                            JK_coeffs[1, l])
    else:
        for m in range(2*Ne):
            slater[:, m % Ne, 2+m//Ne] = np.exp(
                1j * (np.real(Ks) * np.real(coords[from_swap[m]]) +
                      np.imag(Ks) * np.imag(coords[from_swap[m]])))


@njit  # (parallel=True)
def njit_GetPfMatrixSwap(Ne, from_swap, jastrows, pf_jastrows, pf_matrix):
    for copy in range(2):
        for i in range(Ne):
            for j in range(i+1, Ne):
                pf_matrix[i, j, 2+copy] = (pf_jastrows[from_swap[i+copy*Ne], from_swap[j+copy*Ne]] /
                                           jastrows[from_swap[i+copy*Ne], from_swap[j+copy*Ne], 0, 0])
                pf_matrix[j, i, 2+copy] = - pf_matrix[i, j, 2+copy]


class MonteCarloTorusCFL (MonteCarloTorus):
    kCM: np.uint8 = 0
    phi_1: np.float64 = 0
    phi_t: np.float64 = 0
    aCM: np.float64
    bCM: np.float64
    Ks: np.array
    JK_coeffs: str
    jastrows: np.array
    jastrows_tmp: np.array
    slater: np.array
    slater_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array
    flag_proj: np.bool_
    flag_pf: np.bool_
    pf_jastrows: np.array
    pf_jastrows_tmp: np.array
    pf_matrix: np.array
    pf_matrix_tmp: np.array
    pf: np.array
    pf_tmp: np.array

    def InitialJastrows(self):
        for copy in range(self.moved_particles.size):
            for i in range(copy*self.Ne, (copy+1)*self.Ne):
                for j in range(i+1, (copy+1)*self.Ne):
                    for k in range(self.Ne**self.flag_proj):
                        for l in range(self.JK_coeffs.shape[1]):
                            self.jastrows[i, j, k, l] = ThetaFunction(
                                (self.coords[i] - self.coords[j] +
                                 self.JK_coeffs[0, l]*1j*self.Ks[k])/self.Lx,
                                self.t, 1/2, 1/2)
                            if self.JK_coeffs[0, l] != 0:
                                self.jastrows[j, i, k, l] = ThetaFunction(
                                    (self.coords[j] - self.coords[i] +
                                     self.JK_coeffs[0, l]*1j*self.Ks[k])/self.Lx,
                                    self.t, 1/2, 1/2)
                            else:
                                self.jastrows[j, i, k, l] = - \
                                    self.jastrows[i, j, k, l]

                    if self.flag_pf:
                        self.pf_jastrows[i, j] = ThetaFunction(
                            (self.coords[i] - self.coords[j])/self.Lx, self.t, 0, 0)
                        self.pf_jastrows[j, i] = self.pf_jastrows[i, j]

                        self.pf_matrix[i-copy*self.Ne, j-copy*self.Ne, copy] = (self.pf_jastrows[i, j] /
                                                                                self.jastrows[i, j, 0, 0])
                        self.pf_matrix[j-copy*self.Ne, i-copy*self.Ne, copy] = - \
                            self.pf_matrix[i-copy*self.Ne,
                                           j-copy*self.Ne, copy]

    def InitialJKMatrix(self, coords, jastrows, slater):
        if self.flag_proj:
            for n in range(self.Ne):
                for m in range(self.Ne):
                    slater[n, m] = np.exp(
                        1j*(self.Ks[n] + np.conj(self.Ks[n]))*coords[m]/2)
                    for l in range(self.JK_coeffs.shape[1]):
                        slater[n, m] *= np.prod(
                            np.power(jastrows[m, :, n, l], self.JK_coeffs[1, l]))
        else:
            np.copyto(slater, np.exp(1j * (np.reshape(np.real(self.Ks), (-1, 1)) * np.real(coords) +
                                           np.reshape(np.imag(self.Ks), (-1, 1)) * np.imag(coords))))

    def InitialWavefn(self):
        self.moved_particles = np.zeros(
            self.coords.size//self.Ne, dtype=np.uint16)
        self.InitialJastrows()
        for copy in range(self.moved_particles.size):
            self.InitialJKMatrix(self.coords[copy*self.Ne:(copy+1)*self.Ne],
                                 self.jastrows[copy*self.Ne:(copy+1)*self.Ne,
                                               copy*self.Ne:(copy+1)*self.Ne,
                                               ...], self.slater[..., copy])
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

            if self.flag_pf:
                self.pf[copy] = pf.pfaffian(self.pf_matrix[..., copy])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

        if self.flag_pf:
            np.copyto(self.pf_jastrows_tmp, self.pf_jastrows)
            np.copyto(self.pf_matrix_tmp, self.pf_matrix)
            np.copyto(self.pf_tmp, self.pf)

        # print(self.pf_matrix, self.pf)

    def InitialJastrowsSwap(self):
        for i in range(self.Ne):
            for j in range(self.Ne, 2*self.Ne):
                for k in range(self.Ne**self.flag_proj):
                    for l in range(self.JK_coeffs.shape[1]):
                        self.jastrows[i, j, k, l] = ThetaFunction(
                            (self.coords[i] - self.coords[j] +
                             self.JK_coeffs[0, l]*1j*self.Ks[k])/self.Lx, self.t, 1/2, 1/2)
                        if self.JK_coeffs[0, l] != 0:
                            self.jastrows[j, i, k, l] = ThetaFunction(
                                (self.coords[j] - self.coords[i] +
                                 self.JK_coeffs[0, l]*1j*self.Ks[k])/self.Lx, self.t, 1/2, 1/2)
                        else:
                            self.jastrows[j, i, k, l] = - \
                                self.jastrows[i, j, k, l]

                if self.flag_pf:
                    self.pf_jastrows[i, j] = ThetaFunction(
                        (self.coords[i] - self.coords[j])/self.Lx, self.t, 0, 0)
                    self.pf_jastrows[j, i] = self.pf_jastrows[i, j]

    def InitialWavefnSwap(self):
        self.InitialJastrowsSwap()
        njit_GetJKMatrixSwap(self.Ne, self.coords, self.Ks, self.from_swap,
                             self.jastrows, self.slater, self.JK_coeffs, self.flag_proj)
        self.slogdet[:, 2] = np.linalg.slogdet(self.slater[:, :, 2])
        self.slogdet[:, 3] = np.linalg.slogdet(self.slater[:, :, 3])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

        if self.flag_pf:
            njit_GetPfMatrixSwap(self.Ne, self.from_swap, self.jastrows,
                                 self.pf_jastrows, self.pf_matrix)
            self.pf[2] = pf.pfaffian(self.pf_matrix[..., 2])
            self.pf[3] = pf.pfaffian(self.pf_matrix[..., 3])
            np.copyto(self.pf_matrix_tmp, self.pf_matrix)
            np.copyto(self.pf_tmp, self.pf)

    def TmpWavefn(self):
        for c in range(self.moved_particles.size):
            p = self.moved_particles[c]
            if not self.flag_proj:
                self.jastrows_tmp[c*self.Ne:(c+1)*self.Ne,
                                  p, 0, 0] = ThetaFunctionVectorized(
                    (self.coords_tmp[c*self.Ne:(c+1)*self.Ne] -
                     self.coords_tmp[p])/self.Lx, self.t, 1/2, 1/2, 100)
                self.jastrows_tmp[p, p, 0, 0] = 1
                self.jastrows_tmp[p, c*self.Ne:(c+1)*self.Ne, 0, 0] = - \
                    self.jastrows_tmp[c*self.Ne:(c+1)*self.Ne, p, 0, 0]
                self.slater_tmp[:, p-c*self.Ne, c] = np.exp(1j * (
                    np.real(self.Ks) * np.real(self.coords_tmp[p]) +
                    np.imag(self.Ks) * np.imag(self.coords_tmp[p])))
            else:
                njit_UpdateJastrows(self.t, self.Lx, self.coords_tmp[c*self.Ne:(c+1)*self.Ne],
                                    self.Ks, self.jastrows[c*self.Ne:(c+1)*self.Ne,
                                                           c*self.Ne:(c+1)*self.Ne, ...],
                                    self.jastrows_tmp[c*self.Ne:(c+1)*self.Ne,
                                                      c*self.Ne:(c+1)*self.Ne, ...],
                                    self.JK_coeffs, self.slater_tmp[..., c],
                                    p-c*self.Ne)

            self.slogdet_tmp[:, c] = np.linalg.slogdet(
                self.slater_tmp[..., c])

            if self.flag_pf:
                njit_UpdatePfJastrows(self.t, self.Lx, self.coords_tmp[c*self.Ne:(c+1)*self.Ne],
                                      self.jastrows_tmp[c*self.Ne:(c+1)*self.Ne,
                                                        c*self.Ne:(c+1)*self.Ne, ...],
                                      self.pf_jastrows_tmp[c*self.Ne:(c+1)*self.Ne,
                                                           c*self.Ne:(c+1)*self.Ne, ...],
                                      self.pf_matrix_tmp[..., c], p-c*self.Ne)
                self.pf_tmp[c] = pf.pfaffian(
                    self.pf_matrix_tmp[..., c])

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        for p in range(self.moved_particles.size):
            np.copyto(self.jastrows_tmp[self.moved_particles[p], :, ...],
                      self.jastrows[self.moved_particles[p], :, ...])
            np.copyto(self.jastrows_tmp[:, self.moved_particles[p], ...],
                      self.jastrows[:, self.moved_particles[p], ...])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

        if self.flag_pf:
            for p in range(self.moved_particles.size):
                np.copyto(self.pf_jastrows_tmp[self.moved_particles[p], :, ...],
                          self.pf_jastrows[self.moved_particles[p], :, ...])
                np.copyto(self.pf_jastrows_tmp[:, self.moved_particles[p], ...],
                          self.pf_jastrows[:, self.moved_particles[p], ...])
            np.copyto(self.pf_matrix_tmp, self.pf_matrix)
            np.copyto(self.pf_tmp, self.pf)

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        for p in range(self.moved_particles.size):
            np.copyto(self.jastrows[self.moved_particles[p], :, ...],
                      self.jastrows_tmp[self.moved_particles[p], :, ...])
            np.copyto(self.jastrows[:, self.moved_particles[p], ...],
                      self.jastrows_tmp[:, self.moved_particles[p], ...])

        np.copyto(self.slater, self.slater_tmp)
        np.copyto(self.slogdet, self.slogdet_tmp)

        if self.flag_pf:
            for p in range(self.moved_particles.size):
                np.copyto(self.pf_jastrows[self.moved_particles[p], :, ...],
                          self.pf_jastrows_tmp[self.moved_particles[p], :, ...])
                np.copyto(self.pf_jastrows[:, self.moved_particles[p], ...],
                          self.pf_jastrows_tmp[:, self.moved_particles[p], ...])
            np.copyto(self.pf_matrix, self.pf_matrix_tmp)
            np.copyto(self.pf, self.pf_tmp)

    def TmpWavefnSwap(self):
        njit_UpdateJastrowsSwap(self.t, self.Lx, self.coords_tmp, self.Ks, self.jastrows_tmp,
                                self.JK_coeffs, self.moved_particles,
                                self.to_swap_tmp, self.flag_proj)
        njit_GetJKMatrixSwap(self.Ne, self.coords_tmp, self.Ks, self.from_swap_tmp,
                             self.jastrows_tmp, self.slater_tmp, self.JK_coeffs, self.flag_proj)

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(
            self.slater_tmp[:, :, 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(
            self.slater_tmp[:, :, 3])

        if self.flag_pf:
            njit_UpdatePfJastrowsSwap(self.t, self.Lx, self.coords_tmp, self.pf_jastrows_tmp,
                                      self.moved_particles, self.to_swap)
            njit_GetPfMatrixSwap(self.Ne, self.from_swap_tmp, self.jastrows_tmp,
                                 self.pf_jastrows_tmp, self.pf_matrix_tmp)
            self.pf_tmp[2] = pf.pfaffian(self.pf_matrix_tmp[..., 2])
            self.pf_tmp[3] = pf.pfaffian(self.pf_matrix_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        step_exponent = 0

        for copy in range(self.moved_particles.size):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_tmp, expo_CM_tmp) = self.ReduceThetaFunctionCM(np.sum(
                self.coords_tmp[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))

            moved_coord = self.coords[self.moved_particles[copy]]
            moved_coord_tmp = self.coords_tmp[self.moved_particles[copy]]
            expo_nonholomorphic = 1j*(moved_coord_tmp*np.imag(moved_coord_tmp) -
                                      moved_coord*np.imag(moved_coord))/2

            step_amplitude *= (reduced_CM_tmp * self.slogdet_tmp[0, copy] /
                               (reduced_CM * self.slogdet[0, copy]))
            step_exponent = expo_CM_tmp - expo_CM + expo_nonholomorphic + \
                self.slogdet_tmp[1, copy] - self.slogdet[1, copy]

            if self.flag_proj:
                if self.Ns//self.Ne % 2 == 0:
                    step_amplitude *= np.exp(step_exponent)
                elif self.Ns//self.Ne == 1:
                    step_amplitude *= (np.prod(np.exp(step_exponent/self.Ne) *
                                               self.jastrows[self.moved_particles[copy],
                                                             copy*self.Ne:(copy+1)*self.Ne, 0, 0] /
                                               self.jastrows_tmp[self.moved_particles[copy],
                                                                 copy*self.Ne:(copy+1)*self.Ne, 0, 0]))
                elif self.Ns//self.Ne == 3:
                    step_amplitude *= (np.prod(np.exp(step_exponent/self.Ne) *
                                               self.jastrows_tmp[self.moved_particles[copy],
                                                                 copy*self.Ne:(copy+1)*self.Ne, 0, 0] /
                                               self.jastrows[self.moved_particles[copy],
                                                             copy*self.Ne:(copy+1)*self.Ne, 0, 0]))
            else:
                step_amplitude *= (np.prod(np.exp(step_exponent/self.Ne) *
                                           np.power(self.jastrows_tmp[self.moved_particles[copy],
                                                                      copy*self.Ne:(copy+1)*self.Ne, 0, 0] /
                                           self.jastrows[self.moved_particles[copy],
                                                         copy*self.Ne:(copy+1)*self.Ne, 0, 0], self.Ns/self.Ne)))

            if self.flag_pf:
                step_amplitude *= self.pf_tmp[copy]/self.pf[copy]

        return step_amplitude

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved."""
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        coords_swap_tmp = np.zeros(self.coords_tmp.size, dtype=np.complex128)

        coords_swap = self.coords[self.from_swap]
        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]

        step_amplitude = 1
        step_exponent = 0
        for swap_copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_tmp, expo_CM_tmp) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap_tmp[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))

            step_amplitude *= ((reduced_CM_tmp / reduced_CM) *
                               (self.slogdet_tmp[0, 2+swap_copy] / self.slogdet[0, 2+swap_copy]))
            step_exponent += (expo_CM_tmp - expo_CM +
                              self.slogdet_tmp[1, 2+swap_copy] - self.slogdet[1, 2+swap_copy])

            if self.flag_pf:
                step_amplitude *= self.pf_tmp[2+swap_copy]/self.pf[2+swap_copy]

        step_exponent += 1j*np.sum(self.coords_tmp[self.moved_particles]*np.imag(self.coords_tmp[self.moved_particles]) -
                                   self.coords[self.moved_particles]*np.imag(self.coords[self.moved_particles]))/2

        if self.flag_proj:
            if self.Ns//self.Ne % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in range(copy*self.Ne, (copy+1)*self.Ne):
                        jastrows_factor = np.prod(np.exp(step_exponent / (self.Ne*(self.Ne-1))) *
                                                  np.power(self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[n+1: (copy+1)*self.Ne], 0, 0] /
                                                           self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.Ne], 0, 0], self.Ns/self.Ne))
                        if self.Ns//self.Ne == 3:
                            step_amplitude *= jastrows_factor
                        elif self.Ns//self.Ne == 1:
                            step_amplitude /= jastrows_factor
        else:
            for copy in range(2):
                for n in range(copy*self.Ne, (copy+1)*self.Ne):
                    step_amplitude *= np.prod(np.exp(step_exponent / (self.Ne*(self.Ne-1))) *
                                              np.power(self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[n+1: (copy+1)*self.Ne], 0, 0] /
                                                       self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.Ne], 0, 0], self.Ns/self.Ne))

        return step_amplitude

    def InitialMod(self):
        step_amplitude = 1
        step_exponent = 0
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        for i in range(2*self.Ne):
            coords_swap[i] = self.coords[self.from_swap[i]]
        for copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))

            step_amplitude *= (reduced_CM_swap*self.slogdet[0, copy+2] /
                               (reduced_CM * self.slogdet[0, copy]))
            step_exponent += (expo_CM_swap - expo_CM +
                              self.slogdet[1, copy+2] - self.slogdet[1, copy])

            if self.flag_pf:
                step_amplitude *= self.pf[copy+2]/(self.pf[copy])

        if self.flag_proj:
            if self.Ns//self.Ne % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in range(copy*self.Ne, (copy+1)*self.Ne):
                        for m in range(n+1, (copy+1)*self.Ne):
                            jastrows_factor = (self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0] /
                                               self.jastrows[n, m, 0, 0])
                            if self.Ns//self.Ne == 3:
                                step_amplitude *= jastrows_factor
                            elif self.Ns//self.Ne == 1:
                                step_amplitude /= jastrows_factor
                            step_amplitude *= np.exp(step_exponent /
                                                     (self.Ne*(self.Ne-1)))
        else:
            for copy in range(2):
                for n in range(copy*self.Ne, (copy+1)*self.Ne):
                    for m in range(n+1, (copy+1)*self.Ne):
                        step_amplitude *= (np.exp(step_exponent / (self.Ne*(self.Ne-1))) *
                                           np.power((self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0] /
                                                     self.jastrows[n, m, 0, 0]), self.Ns/self.Ne))

        return np.abs(step_amplitude)

    def InitialSign(self):
        step_amplitude = 1
        step_exponent = 0
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        for i in range(2*self.Ne):
            coords_swap[i] = self.coords[self.from_swap[i]]
        for copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.flag_proj))

            step_amplitude *= (np.conj(reduced_CM_swap*self.slogdet[0, copy+2]) *
                               reduced_CM * self.slogdet[0, copy])
            step_amplitude /= np.abs(step_amplitude)
            step_exponent += 1j*np.imag(-expo_CM_swap + expo_CM)

            if self.flag_pf:
                step_amplitude *= np.conj(self.pf[copy+2]) * self.pf[copy]
                step_amplitude /= np.abs(step_amplitude)

        if self.flag_proj:
            if self.Ns//self.Ne % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in prange(copy*self.Ne, (copy+1)*self.Ne):
                        for m in range(n+1, (copy+1)*self.Ne):
                            jastrows_factor = (np.conj(self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]) *
                                               self.jastrows[n, m, 0, 0])
                            if self.Ns//self.Ne == 3:
                                step_amplitude *= jastrows_factor
                            elif self.Ns//self.Ne == 1:
                                step_amplitude /= jastrows_factor
                            step_amplitude *= np.exp(step_exponent /
                                                     (self.Ne*(self.Ne-1)))
                            step_amplitude /= np.abs(step_amplitude)
        else:
            for copy in range(2):
                for n in range(copy*self.Ne, (copy+1)*self.Ne):
                    for m in range(n+1, (copy+1)*self.Ne):
                        step_amplitude *= (np.exp(step_exponent / (self.Ne*(self.Ne-1))) *
                                           np.power((np.conj(self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]) *
                                                     self.jastrows[n, m, 0, 0]), self.Ns/self.Ne))
                        step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def __init__(self, Ne, Ns, t, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size=0.1, area_size=0, linear_size=0, JK_coeffs='2', flag_pf=False,
                 nbr_copies=2, kCM=0, phi_1=0, phi_t=0,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.kCM = kCM
        self.phi_1 = phi_1
        self.phi_t = phi_t
        self.aCM = phi_1/(2*np.pi*Ns/Ne) + kCM/(Ns/Ne) + (Ne-1)/2
        self.bCM = -phi_t/(2*np.pi) + (Ns/Ne)*(Ne-1)/2

        self.JK_coeffs = np.vstack((np.unique(
            np.array(list(JK_coeffs), dtype=int), return_counts=True)))

        if JK_coeffs == "0":
            self.flag_proj = False
        else:
            self.flag_proj = True
            if np.sum(np.sum(self.JK_coeffs[0, :])) != Ns/Ne:
                print("JK coefficients are incorrect for the given filling!")

        self.state = 'cfl'+JK_coeffs

        self.flag_pf = flag_pf
        if self.flag_pf:
            self.state += 'pf'

        super().__init__(Ne, Ns, t, nbr_iter, nbr_nonthermal, region_geometry,
                         step_size, area_size, linear_size,
                         save_results, save_config, acceptance_ratio)

        self.Ks = (fermi_sea_kx[self.Ne]*2*np.pi/self.Lx +
                   1j*fermi_sea_ky[self.Ne]*2*np.pi/self.Ly)

        self.jastrows = np.ones(
            (nbr_copies*Ne, nbr_copies*Ne, Ne**self.flag_proj,
             self.JK_coeffs.shape[1]), dtype=np.complex128)

        self.slater = np.zeros(
            (Ne, Ne, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        if self.flag_pf:
            self.pf_jastrows = np.ones(
                (nbr_copies*Ne, nbr_copies*Ne), dtype=np.complex128)
            self.pf_matrix = np.zeros(
                (Ne, Ne, 4**(nbr_copies-1)), dtype=np.complex128)
            self.pf = np.zeros(
                (4**(nbr_copies-1)), dtype=np.complex128)

            self.pf_jastrows_tmp = np.copy(self.pf_jastrows)
            self.pf_matrix_tmp = np.copy(self.pf_matrix)
            self.pf_tmp = np.copy(self.pf)

        self.jastrows_tmp = np.copy(self.jastrows)
        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
