import numpy as np
from numba import njit, prange
from .MonteCarloTorusBase import MonteCarloTorusBase
from .utilities import fermi_sea_kx, fermi_sea_ky, ThetaFunction


@njit  # (parallel=True)
def njit_UpdateJastrows(t: np.complex128, Lx: np.float64,
                        coords_tmp: np.array, Ks: np.array,
                        jastrows: np.array, jastrows_tmp: np.array,
                        JK_coeffs: np.array, JK_matrix_tmp: np.array,
                        moved_particle: np.uint16, projected: np.bool_):
    Ne = coords_tmp.size
    if projected:
        JK_matrix_tmp[:, moved_particle] = np.exp(
            1j*np.real(Ks)*coords_tmp[moved_particle])
        for k in range(Ne):
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

                        JK_matrix_tmp[k, i] *= np.power((jastrows_tmp[i, moved_particle, k, l] /
                                                        jastrows[i, moved_particle, k, l]), JK_coeffs[1, l])

                        JK_matrix_tmp[k, moved_particle] *= np.power(jastrows_tmp[moved_particle, i, k, l],
                                                                     JK_coeffs[1, l])
    else:
        for i in range(Ne):
            if i != moved_particle:
                jastrows_tmp[i, moved_particle, 0, 0] = ThetaFunction(
                    (coords_tmp[i] - coords_tmp[moved_particle])/Lx, t, 1/2, 1/2)
                jastrows_tmp[moved_particle, i, 0, 0] = - \
                    jastrows_tmp[i, moved_particle, 0, 0]
        JK_matrix_tmp[:, moved_particle] = np.exp(1j * (np.real(Ks) * np.real(coords_tmp[moved_particle]) +
                                                        np.imag(Ks) * np.imag(coords_tmp[moved_particle])))


@njit  # (parallel=True)
def njit_UpdateJastrowsSwap(t: np.complex128, Lx: np.float64, coords: np.array, Ks: np.array,
                            jastrows: np.array, JK_coeffs: np.array, moved_particles: np.array,
                            to_swap: np.array, projected: np.bool_):
    Ne = coords.size//2
    if projected:
        for k in range(Ne):
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
def njit_GetJKMatrixSwap(Ne, coords, Ks, from_swap, jastrows, JK_matrix, JK_coeffs, projected):
    if projected:
        for n in prange(Ne):
            for m in range(2*Ne):
                JK_matrix[n, m % Ne, 2+m//Ne] = np.exp(
                    2j*np.real(Ks[n]) * coords[from_swap[m]]/2)
                for j in range(Ne):
                    for l in range(JK_coeffs.shape[1]):
                        JK_matrix[n, m % Ne, 2+m//Ne] *= np.power(
                            jastrows[from_swap[m],
                                     from_swap[j+(m//Ne)*Ne], n, l],
                            JK_coeffs[1, l])
    else:
        for m in range(2*Ne):
            JK_matrix[:, m % Ne, 2+m//Ne] = np.exp(
                1j * (np.real(Ks) * np.real(coords[from_swap[m]]) +
                      np.imag(Ks) * np.imag(coords[from_swap[m]])))


"""
@njit
def njit_StepAmplitudeTwoCopiesSwap(Ne, coords, coords_tmp) -> np.complex128:
    coords_swap = np.zeros(coords.size, dtype=np.complex128)
    coords_swap_tmp = np.zeros(coords_tmp.size, dtype=np.complex128)
    for i in range(2*Ne):
        coords_swap[i] = self.coords[self.from_swap[i]]
        coords_swap_tmp[i] = self.coords_tmp[self.from_swap_tmp[i]]

    step_amplitude = 1
    step_exponent = 0
    for swap_copy in range(2):
        (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
            coords_swap[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.projected))
        (reduced_CM_tmp, expo_CM_tmp) = self.ReduceThetaFunctionCM(np.sum(
            coords_swap_tmp[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.projected))

        step_amplitude *= ((reduced_CM_tmp / reduced_CM) *
                                (self.JK_slogdet_tmp[0, 2+swap_copy] / self.JK_slogdet[0, 2+swap_copy]))
        step_exponent += (expo_CM_tmp - expo_CM +
                                self.JK_slogdet_tmp[1, 2+swap_copy] - self.JK_slogdet[1, 2+swap_copy])

    # add non-holomorphic contribution from exponential factor
    step_exponent += 1j*np.sum(self.coords_tmp[self.moved_particles]*np.imag(self.coords_tmp[self.moved_particles]) -
                                    self.coords[self.moved_particles]*np.imag(self.coords[self.moved_particles]))/2

    if self.projected:
        if self.Ns//self.Ne % 2 == 0:
            step_amplitude *= np.exp(step_exponent)
        else:
            for copy in range(2):
                for n in range(copy*self.Ne, (copy+1)*self.Ne):
                    for m in range(n+1, (copy+1)*self.Ne):
                        jastrows_factor = (self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[m], 0, 0] /
                                                self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0])
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
                                            np.power((self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[m], 0, 0] /
                                                        self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]), self.Ns/self.Ne))

    return step_amplitude"""


class MonteCarloTorusWithSpin (MonteCarloTorusBase):
    Ks: np.array
    jastrows: np.array
    jastrows_tmp: np.array
    JK_matrix: np.array
    JK_matrix_tmp: np.array
    JK_slogdet: np.array
    JK_slogdet_tmp: np.array

    def InitialJastrows(self, coords, jastrows):
        for spin in range(2):
            for i in range(self.Ne):
                for j in range(i+1, self.Ne):
                    jastrows[i, j, spin] = ThetaFunction(
                        (coords[i] - coords[j])/self.Lx, self.t, 1/2, 1/2)
                    jastrows[j, i, spin] = - jastrows[i, j]

    def InitialJKMatrix(self, coords, JK_matrix):
        np.copyto(JK_matrix, np.exp(1j * (np.reshape(np.real(self.Ks), (-1, 1)) * np.real(coords) +
                                          np.reshape(np.imag(self.Ks), (-1, 1)) * np.imag(coords))))

    def InitialWavefn(self):
        if self.coords.size == self.Ne:
            self.InitialJastrows(self.coords, self.jastrows)
            self.InitialJKMatrix(self.coords, self.jastrows, self.JK_matrix)
            self.JK_slogdet = np.linalg.slogdet(self.JK_matrix)
        elif self.coords.size == 2*self.Ne:
            self.moved_particles = np.zeros(2, dtype=np.uint16)
            for copy in range(2):
                self.InitialJastrows(self.coords[copy*self.Ne:(copy+1)*self.Ne],
                                     self.jastrows[copy*self.Ne:(copy+1)*self.Ne,
                                                   copy*self.Ne:(copy+1)*self.Ne,
                                                   ...])
                self.InitialJKMatrix(self.coords[copy*self.Ne:(copy+1)*self.Ne],
                                     self.jastrows[copy*self.Ne:(copy+1)*self.Ne,
                                                   copy*self.Ne:(copy+1)*self.Ne,
                                                   ...], self.JK_matrix[..., copy])
                self.JK_slogdet[:, copy] = np.linalg.slogdet(
                    self.JK_matrix[..., copy])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.JK_matrix_tmp, self.JK_matrix)
        np.copyto(self.JK_slogdet_tmp, self.JK_slogdet)

    def InitialJastrowsSwap(self):
        if self.projected:
            for k in range(self.Ne):
                for i in range(self.Ne):
                    for j in range(self.Ne, 2*self.Ne):
                        for l in range(self.JK_coeffs_unique.shape[1]):
                            self.jastrows[i, j, k, l] = ThetaFunction(
                                (self.coords[i] - self.coords[j] +
                                 self.JK_coeffs_unique[0, l]*1j*self.Ks[k])/self.Lx, self.t, 1/2, 1/2)
                            if self.JK_coeffs_unique[0, l] != 0:
                                self.jastrows[j, i, k, l] = ThetaFunction(
                                    (self.coords[j] - self.coords[i] +
                                     self.JK_coeffs_unique[0, l]*1j*self.Ks[k])/self.Lx, self.t, 1/2, 1/2)
                            else:
                                self.jastrows[j, i, k, l] = - \
                                    self.jastrows[i, j, k, l]
        else:
            for i in range(self.Ne):
                for j in range(self.Ne, 2*self.Ne):
                    self.jastrows[i, j, 0, 0] = ThetaFunction(
                        (self.coords[i] - self.coords[j])/self.Lx, self.t, 1/2, 1/2)
                    self.jastrows[j, i, 0, 0] = - self.jastrows[i, j, 0, 0]

    def InitialWavefnSwap(self):
        self.InitialJastrowsSwap()
        njit_GetJKMatrixSwap(self.Ne, self.coords, self.Ks, self.from_swap,
                             self.jastrows, self.JK_matrix, self.JK_coeffs_unique, self.projected)
        self.JK_slogdet[:, 2] = np.linalg.slogdet(self.JK_matrix[:, :, 2])
        self.JK_slogdet[:, 3] = np.linalg.slogdet(self.JK_matrix[:, :, 3])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.JK_matrix_tmp, self.JK_matrix)
        np.copyto(self.JK_slogdet_tmp, self.JK_slogdet)

    def TmpWavefn(self):
        for copy in range(2):
            njit_UpdateJastrows(self.t, self.Lx, self.coords_tmp[copy*self.Ne:(copy+1)*self.Ne],
                                self.Ks, self.jastrows[copy*self.Ne:(copy+1)*self.Ne,
                                                       copy*self.Ne:(copy+1)*self.Ne, ...],
                                self.jastrows_tmp[copy*self.Ne:(copy+1)*self.Ne,
                                                  copy*self.Ne:(copy+1)*self.Ne, ...],
                                self.JK_coeffs_unique, self.JK_matrix_tmp[..., copy],
                                self.moved_particles[copy]-copy*self.Ne, self.projected)
            self.JK_slogdet_tmp[:, copy] = np.linalg.slogdet(
                self.JK_matrix_tmp[..., copy])

    def RejectJastrowsTmp(self, swap_term):
        if self.moved_particles.size == 1:
            np.copyto(self.jastrows_tmp[self.moved_particle, :, ...],
                      self.jastrows[self.moved_particle, :, ...])
            np.copyto(self.jastrows_tmp[:, self.moved_particle, ...],
                      self.jastrows[:, self.moved_particle, ...])
            np.copyto(self.JK_matrix_tmp, self.JK_matrix)
        else:
            self.coords_tmp[self.moved_particles] = self.coords[self.moved_particles]
            for p in range(self.moved_particles.size):
                np.copyto(self.jastrows_tmp[self.moved_particles[p], :, ...],
                          self.jastrows[self.moved_particles[p], :, ...])
                np.copyto(self.jastrows_tmp[:, self.moved_particles[p], ...],
                          self.jastrows[:, self.moved_particles[p], ...])
                np.copyto(self.JK_matrix_tmp[..., p], self.JK_matrix[..., p])
            if swap_term != 'p':
                np.copyto(self.JK_matrix_tmp[..., 2], self.JK_matrix[..., 2])
                np.copyto(self.JK_matrix_tmp[..., 3], self.JK_matrix[..., 3])
                np.copyto(self.to_swap_tmp, self.to_swap)
                np.copyto(self.from_swap_tmp, self.from_swap)

            np.copyto(self.JK_slogdet_tmp, self.JK_slogdet)

    def AcceptJastrowsTmp(self, swap_term):
        self.acceptance_ratio += 1
        if self.moved_particles.size == 1:
            np.copyto(self.jastrows[self.moved_particle, :, ...],
                      self.jastrows_tmp[self.moved_particle, :, ...])
            np.copyto(self.jastrows[:, self.moved_particle, ...],
                      self.jastrows_tmp[:, self.moved_particle, ...])
            np.copyto(self.JK_matrix, self.JK_matrix_tmp)
        else:
            self.coords[self.moved_particles] = self.coords_tmp[self.moved_particles]
            for p in range(self.moved_particles.size):
                np.copyto(self.jastrows[self.moved_particles[p], :, ...],
                          self.jastrows_tmp[self.moved_particles[p], :, ...])
                np.copyto(self.jastrows[:, self.moved_particles[p], ...],
                          self.jastrows_tmp[:, self.moved_particles[p], ...])
                np.copyto(self.JK_matrix[..., p], self.JK_matrix_tmp[..., p])
            if swap_term != 'p':
                np.copyto(self.JK_matrix[..., 2], self.JK_matrix_tmp[..., 2])
                np.copyto(self.JK_matrix[..., 3], self.JK_matrix_tmp[..., 3])
                np.copyto(self.to_swap, self.to_swap_tmp)
                np.copyto(self.from_swap, self.from_swap_tmp)

            np.copyto(self.JK_slogdet, self.JK_slogdet_tmp)

    def TmpWavefnSwap(self) -> (np.array, np.array):

        njit_UpdateJastrowsSwap(self.t, self.Lx, self.coords_tmp, self.Ks, self.jastrows_tmp,
                                self.JK_coeffs_unique, self.moved_particles,
                                self.to_swap_tmp, self.projected)
        njit_GetJKMatrixSwap(self.Ne, self.coords_tmp, self.Ks, self.from_swap_tmp,
                             self.jastrows_tmp, self.JK_matrix_tmp, self.JK_coeffs_unique, self.projected)

        self.JK_slogdet_tmp[:, 2] = np.linalg.slogdet(
            self.JK_matrix_tmp[:, :, 2])
        self.JK_slogdet_tmp[:, 3] = np.linalg.slogdet(
            self.JK_matrix_tmp[:, :, 3])

    def StepAmplitudeTwoCopies(self) -> np.complex128:
        step_amplitude = 1
        step_exponent = 0
        for copy in range(2):

            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))
            (reduced_CM_tmp, expo_CM_tmp) = self.ReduceThetaFunctionCM(np.sum(
                self.coords_tmp[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))

            moved_coord = self.coords[self.moved_particles[copy]]
            moved_coord_tmp = self.coords_tmp[self.moved_particles[copy]]
            expo_nonholomorphic = 1j*(moved_coord_tmp*np.imag(moved_coord_tmp) -
                                      moved_coord*np.imag(moved_coord))/2

            step_amplitude *= (reduced_CM_tmp * self.JK_slogdet_tmp[0, copy] /
                               (reduced_CM * self.JK_slogdet[0, copy]))
            step_exponent = expo_CM_tmp - expo_CM + expo_nonholomorphic + \
                self.JK_slogdet_tmp[1, copy] - self.JK_slogdet[1, copy]

            if self.projected:
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

        return step_amplitude

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved."""
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        coords_swap_tmp = np.zeros(self.coords_tmp.size, dtype=np.complex128)
        for i in range(2*self.Ne):
            coords_swap[i] = self.coords[self.from_swap[i]]
            coords_swap_tmp[i] = self.coords_tmp[self.from_swap_tmp[i]]

        step_amplitude = 1
        step_exponent = 0
        for swap_copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.projected))
            (reduced_CM_tmp, expo_CM_tmp) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap_tmp[swap_copy*self.Ne:(swap_copy+1)*self.Ne] + 1j*self.Ks*self.projected))

            step_amplitude *= ((reduced_CM_tmp / reduced_CM) *
                               (self.JK_slogdet_tmp[0, 2+swap_copy] / self.JK_slogdet[0, 2+swap_copy]))
            step_exponent += (expo_CM_tmp - expo_CM +
                              self.JK_slogdet_tmp[1, 2+swap_copy] - self.JK_slogdet[1, 2+swap_copy])

        # add non-holomorphic contribution from exponential factor
        step_exponent += 1j*np.sum(self.coords_tmp[self.moved_particles]*np.imag(self.coords_tmp[self.moved_particles]) -
                                   self.coords[self.moved_particles]*np.imag(self.coords[self.moved_particles]))/2

        if self.projected:
            if self.Ns//self.Ne % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in range(copy*self.Ne, (copy+1)*self.Ne):
                        for m in range(n+1, (copy+1)*self.Ne):
                            jastrows_factor = (self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[m], 0, 0] /
                                               self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0])
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
                                           np.power((self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[m], 0, 0] /
                                                     self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]), self.Ns/self.Ne))

        return step_amplitude

    def InitialMod(self):
        step_amplitude = 1
        step_exponent = 0
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        for i in range(2*self.Ne):
            coords_swap[i] = self.coords[self.from_swap[i]]
        for copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))

            step_amplitude *= (reduced_CM_swap*self.JK_slogdet[0, copy+2] /
                               (reduced_CM * self.JK_slogdet[0, copy]))
            step_exponent += (expo_CM_swap - expo_CM +
                              self.JK_slogdet[1, copy+2] - self.JK_slogdet[1, copy])

        if self.projected:
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
                self.coords[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.Ne:(copy+1)*self.Ne] + 1j*self.Ks*self.projected))

            step_amplitude *= (np.conj(reduced_CM_swap*self.JK_slogdet[0, copy+2]) *
                               reduced_CM * self.JK_slogdet[0, copy])
            step_amplitude /= np.abs(step_amplitude)
            step_exponent += 1j*np.imag(-expo_CM_swap + expo_CM)

        if self.projected:
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

    def RunSwapP(self):
        """
        Computes the p-term in the entanglement entropy swap decomposition 
        (the probability of two copies being swappable).
        """
        self.LoadRun('p')
        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        update = (np.count_nonzero(inside_region[:self.Ne]) ==
                  np.count_nonzero(inside_region[self.Ne:]))

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            self.StepOneParticleStarTwoCopies()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitudeTwoCopies()

            if np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptJastrowsTmp('p')
                inside_region = self.InsideRegion(self.coords)
                update = (np.count_nonzero(inside_region[:self.Ne]) ==
                          np.count_nonzero(inside_region[self.Ne:]))
            else:
                self.RejectJastrowsTmp('p')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'p')

        self.SaveResults('p')

    def RunSwapMod(self):
        """
        Computes the mod-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('mod')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialMod()

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            nbr_in_region_changes = self.StepOneSwap()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitudeTwoCopies()

            if step_amplitude*np.conj(step_amplitude) > np.random.random():
                self.UpdateOrderSwap(nbr_in_region_changes)
                self.TmpWavefnSwap()
                step_amplitude_swap = self.StepAmplitudeTwoCopiesSwap()
                update *= np.abs(step_amplitude_swap / step_amplitude)
                self.AcceptJastrowsTmp('mod')

            else:
                self.RejectJastrowsTmp('mod')

            self.results[i] = update
            # if (i+1-self.load_iter) % (self.nbr_iter//100) == 0:
            # print(step_amplitude, step_amplitude_swap)
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'mod')

        self.SaveResults('mod')

    def RunSwapSign(self):
        """
        Computes the sign-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('sign')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialSign()

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            nbr_in_region_changes = self.StepOneSwap()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitudeTwoCopies()
            self.UpdateOrderSwap(nbr_in_region_changes)
            self.TmpWavefnSwap()
            step_amplitude *= np.conj(self.StepAmplitudeTwoCopiesSwap())

            if np.abs(step_amplitude) > np.random.random():
                update *= step_amplitude / np.abs(step_amplitude)
                self.AcceptJastrowsTmp('sign')

            else:
                self.RejectJastrowsTmp('sign')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'sign')

        self.SaveResults('sign')

    def __init__(self, Ne, Ns, t, nbr_iter, nbr_nonthermal, region_geometry,
                 region_size, step_size, kCM=0, phi_1=0, phi_t=0,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.state = 'spin_singlet'

        super().__init__(Ne, Ns, t, nbr_iter, nbr_nonthermal, region_geometry,
                         region_size, step_size, kCM, phi_1, phi_t,
                         save_results, save_config, acceptance_ratio)

        self.Ks = (fermi_sea_kx[self.Ne]*2*np.pi/self.Lx +
                   1j*fermi_sea_ky[self.Ne]*2*np.pi/self.Ly)

        self.jastrows = np.ones(
            (2*Ne, 2*Ne, Ne), dtype=np.complex128)
        self.JK_matrix = np.zeros((Ne, Ne, 4), dtype=np.complex128)

        self.jastrows_tmp = np.copy(self.jastrows)
        self.JK_matrix_tmp = np.copy(self.JK_matrix)

        self.JK_slogdet = np.zeros((2, 4), dtype=np.complex128)
        self.JK_slogdet_tmp = np.copy(self.JK_slogdet)
