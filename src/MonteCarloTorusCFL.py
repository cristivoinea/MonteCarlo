import numpy as np
from numba import njit, prange
from pfapack import pfaffian as pf
from .MonteCarloTorus import MonteCarloTorus
from .utilities import fermi_sea_kx, fermi_sea_ky
from .fast_math import ThetaFunction, ThetaFunctionVectorized, ReduceThetaFunctionCM


@njit
def njit_UpdateSlater(coords: np.array, moved_particle: np.array,
                      slater: np.array, Ks: np.array):
    slater[:, moved_particle] = np.exp(1j * (np.real(Ks) * np.real(coords[moved_particle]) +
                                             np.imag(Ks) * np.imag(coords[moved_particle])))


@njit(parallel=True)
def njit_UpdateJastrowsProj(t: np.complex128, Lx: np.float64,
                            coords_tmp: np.array, Ks: np.array,
                            jastrows: np.array, jastrows_tmp: np.array,
                            JK_coeffs: np.array, slater_tmp: np.array,
                            moved_particle: np.int64):
    N = coords_tmp.size
    slater_tmp[:, moved_particle] = np.exp(
        1j*np.real(Ks)*coords_tmp[moved_particle])
    for i in prange(N):
        if i != moved_particle:
            for l in range(JK_coeffs.shape[1]):
                jastrows_tmp[i, moved_particle, :, l] = ThetaFunctionVectorized(
                    (coords_tmp[i] - coords_tmp[moved_particle] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 75)

                if JK_coeffs[0, l] != 0:
                    jastrows_tmp[moved_particle, i, :, l] = ThetaFunctionVectorized(
                        (coords_tmp[moved_particle] - coords_tmp[i] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 75)
                else:
                    jastrows_tmp[moved_particle, i, :, l] = - \
                        jastrows_tmp[i, moved_particle, :, l]

                slater_tmp[:, i] *= np.power((jastrows_tmp[i, moved_particle, :, l] /
                                              jastrows[i, moved_particle, :, l]), JK_coeffs[1, l])

                slater_tmp[:, moved_particle] *= np.power(jastrows_tmp[moved_particle, i, :, l],
                                                          JK_coeffs[1, l])


@njit(parallel=True)
def njit_UpdatePfJastrows(t: np.complex128, Lx: np.float64,
                          coords_tmp: np.array, jastrows_tmp: np.array,
                          pf_jastrows_tmp: np.array,
                          pf_matrix_tmp: np.array,
                          moved_particle: np.uint16):
    N = coords_tmp.size
    for i in range(N):
        if i != moved_particle:
            pf_jastrows_tmp[i, moved_particle] = ThetaFunction(
                (coords_tmp[i] - coords_tmp[moved_particle])/Lx, t, 0, 0)
            pf_jastrows_tmp[moved_particle,
                            i] = pf_jastrows_tmp[i, moved_particle]

    pf_matrix_tmp[:, moved_particle] = (pf_jastrows_tmp[:, moved_particle] /
                                        jastrows_tmp[:, moved_particle, 0, 0])
    pf_matrix_tmp[moved_particle, :] = - pf_matrix_tmp[:, moved_particle]


@njit(parallel=True)
def njit_UpdateJastrowsSwapProj(t: np.complex128, Lx: np.float64, coords: np.array, Ks: np.array,
                                jastrows: np.array, JK_coeffs: np.array, moved_particles: np.array,
                                to_swap: np.array):
    N = coords.size//2
    for i in prange(N):
        for l in range(JK_coeffs.shape[1]):
            # update cross copy jastrows for particle in copy 2
            if (to_swap[moved_particles[1]] // N) == (to_swap[i] // N):
                jastrows[i, moved_particles[1], :, l] = ThetaFunctionVectorized(
                    (coords[i] - coords[moved_particles[1]] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 100)

                if JK_coeffs[0, l] != 0:
                    jastrows[moved_particles[1], i, :, l] = ThetaFunctionVectorized(
                        (coords[moved_particles[1]] - coords[i] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 100)
                else:
                    jastrows[moved_particles[1], i, :, l] = - \
                        jastrows[i, moved_particles[1], :, l]

                # update cross copy jastrows for particle in copy 1
            if (to_swap[moved_particles[0]] // N) == (to_swap[i+N] // N):
                jastrows[i+N, moved_particles[0], :, l] = ThetaFunctionVectorized(
                    (coords[i+N] - coords[moved_particles[0]] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 100)

                if JK_coeffs[0, l] != 0:
                    jastrows[moved_particles[0], i+N, :, l] = ThetaFunctionVectorized(
                        (coords[moved_particles[0]] - coords[i+N] + JK_coeffs[0, l]*1j*Ks)/Lx, t, 1/2, 1/2, 100)
                else:
                    jastrows[moved_particles[0], i+N, :, l] = - \
                        jastrows[i+N, moved_particles[0], :, l]


@njit
def njit_UpdateJastrowsSwapUnproj(t: np.complex128, Lx: np.float64, coords: np.array,
                                  jastrows: np.array, moved_particles: np.array,
                                  to_swap: np.array):
    N = coords.size//2
    for i in range(N):
        if (to_swap[moved_particles[1]] // N) == (to_swap[i] // N):
            jastrows[i, moved_particles[1], 0, 0] = ThetaFunction(
                (coords[i] - coords[moved_particles[1]])/Lx, t, 1/2, 1/2)
            jastrows[moved_particles[1], i, 0, 0] = - \
                jastrows[i, moved_particles[1], 0, 0]

        # update cross copy jastrows for particle in copy 1
        if (to_swap[moved_particles[0]] // N) == (to_swap[i+N] // N):
            jastrows[i+N, moved_particles[0], 0, 0] = ThetaFunction(
                (coords[i+N] - coords[moved_particles[0]])/Lx, t, 1/2, 1/2)
            jastrows[moved_particles[0], i+N, 0, 0] = - \
                jastrows[i+N, moved_particles[0], 0, 0]


@njit  # (parallel=True)
def njit_UpdatePfJastrowsSwap(t: np.complex128, Lx: np.float64, coords: np.array,
                              pf_jastrows: np.array, moved_particles: np.array,
                              to_swap: np.array):
    N = coords.size//2
    for i in range(N):
        if (to_swap[moved_particles[1]] // N) == (to_swap[i] // N):
            pf_jastrows[i, moved_particles[1]] = ThetaFunction(
                (coords[i] - coords[moved_particles[1]])/Lx, t, 1/2, 1/2)
            pf_jastrows[moved_particles[1], i] = \
                pf_jastrows[i, moved_particles[1]]

        if (to_swap[moved_particles[0]] // N) == (to_swap[i+N] // N):
            pf_jastrows[i+N, moved_particles[0]] = ThetaFunction(
                (coords[i+N] - coords[moved_particles[0]])/Lx, t, 1/2, 1/2)
            pf_jastrows[moved_particles[0], i+N] = \
                pf_jastrows[i+N, moved_particles[0]]


@njit  # (parallel=True)
def njit_GetJKMatrixSwap(N, coords, Ks, from_swap, jastrows, slater, JK_coeffs, flag_proj):
    if flag_proj:
        for n in range(N):
            for m in range(2*N):
                slater[n, m % N, 2+m//N] = np.exp(
                    2j*np.real(Ks[n]) * coords[from_swap[m]]/2)
                for j in range(N):
                    for l in range(JK_coeffs.shape[1]):
                        slater[n, m % N, 2+m//N] *= np.power(
                            jastrows[from_swap[m],
                                     from_swap[j+(m//N)*N], n, l],
                            JK_coeffs[1, l])
    else:
        for m in range(2*N):
            slater[:, m % N, 2+m//N] = np.exp(
                1j * (np.real(Ks) * np.real(coords[from_swap[m]]) +
                      np.imag(Ks) * np.imag(coords[from_swap[m]])))


@njit
def njit_UpdateSlaterUnproj(coords, to_swap, moved_particles, slater, Ks):
    N = Ks.size
    for c in range(2):
        swap_copy = to_swap[moved_particles[c]] // N
        swap_particle = to_swap[moved_particles[c]]
        slater[:, swap_particle % N, swap_copy] = np.exp(1j * (np.real(Ks) * np.real(coords[swap_particle]) +
                                                               np.imag(Ks) * np.imag(coords[swap_particle])))


@njit  # (parallel=True)
def njit_GetPfMatrixSwap(N, from_swap, jastrows, pf_jastrows, pf_matrix):
    for copy in range(2):
        for i in range(N):
            for j in range(i+1, N):
                pf_matrix[i, j, 2+copy] = (pf_jastrows[from_swap[i+copy*N], from_swap[j+copy*N]] /
                                           jastrows[from_swap[i+copy*N], from_swap[j+copy*N], 0, 0])
                pf_matrix[j, i, 2+copy] = - pf_matrix[i, j, 2+copy]


@njit
def njit_GetJastrowsSwap(jastrows: np.array, from_swap: np.array):
    jastrows_swap = np.copy(jastrows)
    for i in range(1, from_swap.size):
        for j in range(i+1, from_swap.size):
            jastrows_swap[i, j, ...] = jastrows[from_swap[i],
                                                from_swap[j], ...]


@njit
def njit_StepAmplitude(N, S, t, aCM, bCM, coords, coords_tmp,
                       moved_particles, simple_jastrows, simple_jastrows_tmp,
                       slogdet, slogdet_tmp, Ks, flag_proj) -> np.complex128:
    step_amplitude = 1
    for copy in range(moved_particles.size):
        step_exponent = 0
        (reduced_CM, expo_CM) = ReduceThetaFunctionCM(N, S, t, aCM, bCM, np.sum(
            coords[copy*N:(copy+1)*N] + 1j*Ks*flag_proj))
        (reduced_CM_tmp, expo_CM_tmp) = ReduceThetaFunctionCM(N, S, t, aCM, bCM, np.sum(
            coords_tmp[copy*N:(copy+1)*N] + 1j*Ks*flag_proj))

        step_amplitude *= ((reduced_CM_tmp / reduced_CM) *
                           (slogdet_tmp[0, copy] / slogdet[0, copy]))
        step_exponent += (expo_CM_tmp - expo_CM +
                          slogdet_tmp[1, copy] - slogdet[1, copy])

        step_exponent += 1j*(coords_tmp[moved_particles[copy]]*np.imag(coords_tmp[moved_particles[copy]]) -
                             coords[moved_particles[copy]]*np.imag(coords[moved_particles[copy]]))/2

        swap_copy = moved_particles[copy]//N
        if not flag_proj:
            step_amplitude *= np.power(np.prod(
                np.exp(step_exponent/S) *
                simple_jastrows_tmp[moved_particles[copy], swap_copy*N:(swap_copy+1)*N] /
                simple_jastrows[moved_particles[copy], swap_copy*N:(swap_copy+1)*N]), S/N)
        else:
            if S//N % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                jastrows_factor = np.prod(
                    np.exp(step_exponent/N) *
                    simple_jastrows_tmp[moved_particles[copy], swap_copy*N:(swap_copy+1)*N] /
                    simple_jastrows[moved_particles[copy], swap_copy*N:(swap_copy+1)*N])
                if S/N == 3:
                    step_amplitude *= jastrows_factor
                elif S/N == 1:
                    step_amplitude /= jastrows_factor

        if moved_particles[0]//N == moved_particles[1]//N:
            correction = np.power(
                simple_jastrows_tmp[moved_particles[0], moved_particles[1]] /
                simple_jastrows[moved_particles[0], moved_particles[1]], S/N)
            if not flag_proj:
                step_amplitude /= correction
            elif S/N == 3:
                step_amplitude /= correction
            elif S/N == 1:
                step_amplitude *= correction

        # if flag_pf:
        #        step_amplitude *= self.pf_tmp[copy]/self.pf[copy]

    return step_amplitude


@njit
def njit_StepAmplitudeTwoCopiesSwap(N, S, t, aCM, bCM, coords, coords_tmp, from_swap, from_swap_tmp,
                                    moved_particles, jastrows, jastrows_tmp,
                                    slogdet, slogdet_tmp, Ks, flag_proj) -> np.complex128:
    coords_swap = coords[from_swap]
    coords_swap_tmp = coords_tmp[from_swap_tmp]

    step_amplitude = 1
    for swap_copy in range(2):
        step_exponent = 0
        (reduced_CM, expo_CM) = ReduceThetaFunctionCM(N, S, t, aCM, bCM, np.sum(
            coords_swap[swap_copy*N:(swap_copy+1)*N] + 1j*Ks*flag_proj))
        (reduced_CM_tmp, expo_CM_tmp) = ReduceThetaFunctionCM(N, S, t, aCM, bCM, np.sum(
            coords_swap_tmp[swap_copy*N:(swap_copy+1)*N] + 1j*Ks*flag_proj))

        step_amplitude *= ((reduced_CM_tmp / reduced_CM) *
                           (slogdet_tmp[0, 2+swap_copy] / slogdet[0, 2+swap_copy]))
        step_exponent += (expo_CM_tmp - expo_CM +
                          slogdet_tmp[1, 2+swap_copy] - slogdet[1, 2+swap_copy])

        # if self.flag_pf:
        #        step_amplitude *= self.pf_tmp[2+swap_copy]/self.pf[2+swap_copy]

        step_exponent += 1j*(coords_tmp[moved_particles[swap_copy]]*np.imag(coords_tmp[moved_particles[swap_copy]]) -
                             coords[moved_particles[swap_copy]]*np.imag(coords[moved_particles[swap_copy]]))/2

        if not flag_proj:
            for n in range(swap_copy*N, (swap_copy+1)*N):
                step_amplitude *= np.prod(np.exp(step_exponent / (N*(N-1)/2)) *
                                          np.power(jastrows_tmp[from_swap_tmp[n], from_swap_tmp[n+1: (swap_copy+1)*N], 0, 0] /
                                                   jastrows[from_swap[n], from_swap[n+1: (swap_copy+1)*N], 0, 0], S/N))
        else:
            if S//N % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for n in range(swap_copy*N, (swap_copy+1)*N):
                    jastrows_factor = np.prod(np.exp(step_exponent / (N*(N-1))) *
                                              np.power(jastrows_tmp[from_swap_tmp[n], from_swap_tmp[n+1: (swap_copy+1)*N], 0, 0] /
                                                       jastrows[from_swap[n], from_swap[n+1: (swap_copy+1)*N], 0, 0], S/N))
                    if S/N == 3:
                        step_amplitude *= jastrows_factor
                    elif S/N == 1:
                        step_amplitude /= jastrows_factor

    return step_amplitude


@njit
def njit_TmpWavefn(N, t, Lx, coords_tmp, jastrows, jastrows_tmp, moved_particles,
                   slater_tmp, slogdet_tmp, Ks, JK_coeffs, flag_proj):
    for c in range(moved_particles.size):
        p = moved_particles[c]
        if not flag_proj:
            jastrows_tmp[c*N:(c+1)*N, p, 0, 0] = ThetaFunctionVectorized(
                (coords_tmp[c*N:(c+1)*N] -
                 coords_tmp[p])/Lx, t, 1/2, 1/2, 100)
            jastrows_tmp[p, c*N:(c+1)*N, 0, 0] = - \
                jastrows_tmp[c*N:(c+1)*N, p, 0, 0]
            jastrows_tmp[p, p, 0, 0] = 1
            njit_UpdateSlater(coords_tmp[c*N:(c+1)*N], p-c*N,
                              slater_tmp[:, :, c], Ks)
        else:
            njit_UpdateJastrowsProj(t, Lx, coords_tmp[c*N:(c+1)*N],
                                    Ks, jastrows[c*N:(c+1)*N,
                                                 c*N:(c+1)*N, ...],
                                    jastrows_tmp[c*N:(c+1)*N,
                                                 c*N:(c+1)*N, ...],
                                    JK_coeffs, slater_tmp[..., c],
                                    p-c*N)

        (slogdet_tmp[0, c], slogdet_tmp[1, c]) = np.linalg.slogdet(
            slater_tmp[:, :, c])


@njit
def njit_ApplyStep(moved_particles, jastrows, jastrows_tmp, slater, slater_tmp,
                   slogdet, slogdet_tmp):
    for p in range(moved_particles.size):
        jastrows[moved_particles[p],
                 :, ...] = jastrows_tmp[moved_particles[p], :, ...]
        jastrows[:, moved_particles[p], ...] = jastrows_tmp[:,
                                                            moved_particles[p], ...]
    slater[...] = np.copy(slater_tmp)
    slogdet[...] = np.copy(slogdet_tmp)


class MonteCarloTorusCFL (MonteCarloTorus):
    kCM: np.int64 = 0
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
            for i in range(copy*self.N, (copy+1)*self.N):
                for j in range(i+1, (copy+1)*self.N):
                    for k in range(self.N**self.flag_proj):
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

                        self.pf_matrix[i-copy*self.N, j-copy*self.N, copy] = (self.pf_jastrows[i, j] /
                                                                              self.jastrows[i, j, 0, 0])
                        self.pf_matrix[j-copy*self.N, i-copy*self.N, copy] = - \
                            self.pf_matrix[i-copy*self.N,
                                           j-copy*self.N, copy]

    def InitialJKMatrix(self, coords, jastrows, slater):
        if self.flag_proj:
            for n in range(self.N):
                for m in range(self.N):
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
            self.coords.size//self.N, dtype=np.int64)
        self.InitialJastrows()
        for copy in range(self.moved_particles.size):
            self.InitialJKMatrix(self.coords[copy*self.N:(copy+1)*self.N],
                                 self.jastrows[copy*self.N:(copy+1)*self.N,
                                               copy*self.N:(copy+1)*self.N,
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
        for i in range(self.N):
            for j in range(self.N, 2*self.N):
                for k in range(self.N**self.flag_proj):
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
        njit_GetJKMatrixSwap(self.N, self.coords, self.Ks, self.from_swap,
                             self.jastrows, self.slater, self.JK_coeffs, self.flag_proj)
        self.slogdet[:, 2] = np.linalg.slogdet(self.slater[:, :, 2])
        self.slogdet[:, 3] = np.linalg.slogdet(self.slater[:, :, 3])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)

        if self.flag_pf:
            njit_GetPfMatrixSwap(self.N, self.from_swap, self.jastrows,
                                 self.pf_jastrows, self.pf_matrix)
            self.pf[2] = pf.pfaffian(self.pf_matrix[..., 2])
            self.pf[3] = pf.pfaffian(self.pf_matrix[..., 3])
            np.copyto(self.pf_matrix_tmp, self.pf_matrix)
            np.copyto(self.pf_tmp, self.pf)

    def TmpWavefn(self):
        njit_TmpWavefn(self.N, self.t, self.Lx, self.coords_tmp, self.jastrows,
                       self.jastrows_tmp, self.moved_particles, self.slater_tmp,
                       self.slogdet_tmp, self.Ks, self.JK_coeffs, self.flag_proj)

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        njit_ApplyStep(self.moved_particles, self.jastrows_tmp, self.jastrows,
                       self.slater_tmp, self.slater, self.slogdet_tmp, self.slogdet)

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        njit_ApplyStep(self.moved_particles, self.jastrows, self.jastrows_tmp,
                       self.slater, self.slater_tmp, self.slogdet, self.slogdet_tmp)

    def TmpWavefnSwap(self):
        if self.flag_proj:
            njit_UpdateJastrowsSwapProj(self.t, self.Lx, self.coords_tmp, self.Ks, self.jastrows_tmp,
                                        self.JK_coeffs, self.moved_particles,
                                        self.to_swap_tmp)
            njit_GetJKMatrixSwap(self.N, self.coords_tmp, self.Ks, self.from_swap_tmp,
                                 self.jastrows_tmp, self.slater_tmp, self.JK_coeffs, self.flag_proj)
        else:
            njit_UpdateJastrowsSwapUnproj(self.t, self.Lx, self.coords_tmp[self.from_swap_tmp],
                                          self.jastrows_tmp, self.moved_particles, self.to_swap_tmp)
            njit_UpdateSlaterUnproj(self.coords_tmp, self.to_swap_tmp, self.moved_particles,
                                    self.slater_tmp[..., 2:], self.Ks)

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(
            self.slater_tmp[:, :, 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(
            self.slater_tmp[:, :, 3])

        if self.flag_pf:
            njit_UpdatePfJastrowsSwap(self.t, self.Lx, self.coords_tmp, self.pf_jastrows_tmp,
                                      self.moved_particles, self.to_swap)
            njit_GetPfMatrixSwap(self.N, self.from_swap_tmp, self.jastrows_tmp,
                                 self.pf_jastrows_tmp, self.pf_matrix_tmp)
            self.pf_tmp[2] = pf.pfaffian(self.pf_matrix_tmp[..., 2])
            self.pf_tmp[3] = pf.pfaffian(self.pf_matrix_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        return njit_StepAmplitude(
            self.N, self.S, self.t, self.aCM, self.bCM, self.coords,
            self.coords_tmp, self.moved_particles,
            self.jastrows[..., 0, 0], self.jastrows_tmp[..., 0, 0],
            self.slogdet[:, :2], self.slogdet_tmp[:, :2], self.Ks, self.flag_proj)

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved."""
        coords_swap = self.coords[self.from_swap]
        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]
        simple_jastrows_swap = self.jastrows[..., 0, 0][np.ix_(
            self.from_swap, self.from_swap)]
        simple_jastrows_swap_tmp = self.jastrows_tmp[..., 0, 0][np.ix_(
            self.from_swap_tmp, self.from_swap_tmp)]
        moved_particles_swap = self.to_swap[self.moved_particles]
        # moved_particles_swap_tmp is identical up to a swap of the two particles.
        # which results in the same amplitude
        return njit_StepAmplitude(
            self.N, self.S, self.t, self.aCM, self.bCM, coords_swap, coords_swap_tmp,
            moved_particles_swap, simple_jastrows_swap,
            simple_jastrows_swap_tmp, self.slogdet[:, 2:],
            self.slogdet_tmp[:, 2:], self.Ks, self.flag_proj)

    def InitialMod(self):
        step_amplitude = 1
        step_exponent = 0
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        for i in range(2*self.N):
            coords_swap[i] = self.coords[self.from_swap[i]]
        for copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.N:(copy+1)*self.N] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.N:(copy+1)*self.N] + 1j*self.Ks*self.flag_proj))

            step_amplitude *= (reduced_CM_swap*self.slogdet[0, copy+2] /
                               (reduced_CM * self.slogdet[0, copy]))
            step_exponent += (expo_CM_swap - expo_CM +
                              self.slogdet[1, copy+2] - self.slogdet[1, copy])

            if self.flag_pf:
                step_amplitude *= self.pf[copy+2]/(self.pf[copy])

        if self.flag_proj:
            if self.S//self.N % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in range(copy*self.N, (copy+1)*self.N):
                        for m in range(n+1, (copy+1)*self.N):
                            jastrows_factor = (self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0] /
                                               self.jastrows[n, m, 0, 0])
                            if self.S//self.N == 3:
                                step_amplitude *= jastrows_factor
                            elif self.S//self.N == 1:
                                step_amplitude /= jastrows_factor
                            step_amplitude *= np.exp(step_exponent /
                                                     (self.N*(self.N-1)))
        else:
            for copy in range(2):
                for n in range(copy*self.N, (copy+1)*self.N):
                    for m in range(n+1, (copy+1)*self.N):
                        step_amplitude *= (np.exp(step_exponent / (self.N*(self.N-1))) *
                                           np.power((self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0] /
                                                     self.jastrows[n, m, 0, 0]), self.S/self.N))

        return np.abs(step_amplitude)

    def InitialSign(self):
        step_amplitude = 1
        step_exponent = 0
        coords_swap = np.zeros(self.coords.size, dtype=np.complex128)
        for i in range(2*self.N):
            coords_swap[i] = self.coords[self.from_swap[i]]
        for copy in range(2):
            (reduced_CM, expo_CM) = self.ReduceThetaFunctionCM(np.sum(
                self.coords[copy*self.N:(copy+1)*self.N] + 1j*self.Ks*self.flag_proj))
            (reduced_CM_swap, expo_CM_swap) = self.ReduceThetaFunctionCM(np.sum(
                coords_swap[copy*self.N:(copy+1)*self.N] + 1j*self.Ks*self.flag_proj))

            step_amplitude *= (np.conj(reduced_CM_swap*self.slogdet[0, copy+2]) *
                               reduced_CM * self.slogdet[0, copy])
            step_amplitude /= np.abs(step_amplitude)
            step_exponent += 1j*np.imag(-expo_CM_swap + expo_CM)

            if self.flag_pf:
                step_amplitude *= np.conj(self.pf[copy+2]) * self.pf[copy]
                step_amplitude /= np.abs(step_amplitude)

        if self.flag_proj:
            if self.S//self.N % 2 == 0:
                step_amplitude *= np.exp(step_exponent)
            else:
                for copy in range(2):
                    for n in prange(copy*self.N, (copy+1)*self.N):
                        for m in range(n+1, (copy+1)*self.N):
                            jastrows_factor = (np.conj(self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]) *
                                               self.jastrows[n, m, 0, 0])
                            if self.S//self.N == 3:
                                step_amplitude *= jastrows_factor
                            elif self.S//self.N == 1:
                                step_amplitude /= jastrows_factor
                            step_amplitude *= np.exp(step_exponent /
                                                     (self.N*(self.N-1)))
                            step_amplitude /= np.abs(step_amplitude)
        else:
            for copy in range(2):
                for n in range(copy*self.N, (copy+1)*self.N):
                    for m in range(n+1, (copy+1)*self.N):
                        step_amplitude *= (np.exp(step_exponent / (self.N*(self.N-1))) *
                                           np.power((np.conj(self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]) *
                                                     self.jastrows[n, m, 0, 0]), self.S/self.N))
                        step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def CF(self, inside_region):
        count = 0
        # for i in range(self.N):
        #    if inside_region

    def __init__(self, N, S, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size=0.1, area_size=0, linear_size=0, JK_coeffs='2', flag_pf=False,
                 nbr_copies=2, t=1j, kCM=0, phi_1=0, phi_t=0,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.kCM = kCM
        self.phi_1 = phi_1
        self.phi_t = phi_t
        self.aCM = phi_1/(2*np.pi*S/N) + kCM/(S/N) + (N-1)/2
        self.bCM = -phi_t/(2*np.pi) + (S/N)*(N-1)/2

        self.JK_coeffs = np.vstack((np.unique(
            np.array(list(JK_coeffs), dtype=int), return_counts=True)))

        if JK_coeffs == "0":
            self.flag_proj = False
        else:
            self.flag_proj = True
            if np.sum(np.sum(self.JK_coeffs[0, :])) != S/N:
                print("JK coefficients are incorrect for the given filling!")

        self.state = 'cfl'+JK_coeffs

        self.flag_pf = flag_pf
        if self.flag_pf:
            self.state += 'pf'

        super().__init__(N, S, t, nbr_iter, nbr_nonthermal, region_geometry,
                         step_size, area_size, linear_size,
                         save_results, save_config, acceptance_ratio)

        self.Ks = (fermi_sea_kx[self.N]*2*np.pi/self.Lx +
                   1j*fermi_sea_ky[self.N]*2*np.pi/self.Ly)

        self.jastrows = np.ones(
            (nbr_copies*N, nbr_copies*N, N**self.flag_proj,
             self.JK_coeffs.shape[1]), dtype=np.complex128)

        self.slater = np.zeros(
            (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
        self.slogdet = np.zeros((2, 4**(nbr_copies-1)), dtype=np.complex128)

        if self.flag_pf:
            self.pf_jastrows = np.ones(
                (nbr_copies*N, nbr_copies*N), dtype=np.complex128)
            self.pf_matrix = np.zeros(
                (N, N, 4**(nbr_copies-1)), dtype=np.complex128)
            self.pf = np.zeros(
                (4**(nbr_copies-1)), dtype=np.complex128)

            self.pf_jastrows_tmp = np.copy(self.pf_jastrows)
            self.pf_matrix_tmp = np.copy(self.pf_matrix)
            self.pf_tmp = np.copy(self.pf)

        self.jastrows_tmp = np.copy(self.jastrows)
        self.slater_tmp = np.copy(self.slater)
        self.slogdet_tmp = np.copy(self.slogdet)
