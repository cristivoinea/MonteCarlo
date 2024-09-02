import numpy as np
from numba import njit
from .MonteCarloSphere import MonteCarloSphere
from scipy.special import betainc, beta


def njit_UpdateJastrows(spinors_tmp: np.array, jastrows_tmp: np.array, p: np.uint16):
    jastrows_tmp[:, p, 0, 0] = (
        spinors_tmp[:, 0] * spinors_tmp[p, 1] - spinors_tmp[p, 0] * spinors_tmp[:, 1]
    )
    jastrows_tmp[p, :, 0, 0] = -jastrows_tmp[:, p, 0, 0]
    jastrows_tmp[p, p] = 1


@njit
def njit_UpdateJastrowsSwap(
    coords_tmp: np.array,
    spinors_tmp: np.array,
    jastrows_tmp: np.array,
    moved_particles: np.array,
    to_swap_tmp: np.array,
):
    N = coords_tmp.shape[0] // 2
    for i in range(N):
        if (to_swap_tmp[moved_particles[1]] // N) == (to_swap_tmp[i] // N):
            jastrows_tmp[i, moved_particles[1], 0, 0] = (
                spinors_tmp[i, 0] * spinors_tmp[moved_particles[1], 1]
                - spinors_tmp[moved_particles[1], 0] * spinors_tmp[i, 1]
            )
            jastrows_tmp[moved_particles[1], i, 0, 0] = -jastrows_tmp[
                i, moved_particles[1], 0, 0
            ]

            # update cross copy jastrows for particle in copy 1
        if (to_swap_tmp[moved_particles[0]] // N) == (to_swap_tmp[i + N] // N):
            jastrows_tmp[i + N, moved_particles[0], 0, 0] = (
                spinors_tmp[i + N, 0] * spinors_tmp[moved_particles[0], 1]
                - spinors_tmp[moved_particles[0], 0] * spinors_tmp[i + N, 1]
            )
            jastrows_tmp[moved_particles[0], i + N, 0, 0] = -jastrows_tmp[
                i + N, moved_particles[0], 0, 0
            ]

"""
def njit_GetOverlapMatrix(N,S,region_theta, region_phi):
        overlap_matrix = np.zeros((N, N), dtype=np.complex128)

        x0 = np.cos(region_theta[0] / 2)
        x1 = np.cos(region_theta[1] / 2)
        d_phi = (region_phi[1] - region_phi[0]) / (2 * np.pi)

        if x0 == 1:
            for i in range(N):
                overlap_matrix[i, i] = d_phi * (1-betainc(i + 1, S - i + 1, x1**2))
                if d_phi != 0:
                    for j in range(i + 1, N):
                        overlap_matrix[i, j] = (np.exp(1j * region_phi[0] * (i - j))
                            - np.exp(1j * region_phi[1] * (i - j)) ) * (
                            1-betainc(i + 1, S - i + 1, x1**2)
                        )
                        overlap_matrix[j, i] = overlap_matrix[i, j]

        else:
            for i in range(N):
                overlap_matrix[i, i] = d_phi*(betainc(i + 1, self.S - i + 1, x0**2) -
                                        betainc(i + 1, self.S - i + 1, x1**2))
                if d_phi != 0:
                    for j in range(i + 1, self.N):
                        overlap_matrix[i, j] = (
                            (np.exp(1j * self.region_phi[0] * (i - j))
                            - np.exp(1j * self.region_phi[1] * (i - j)) ) # fmt: skip
                            * (betainc((i+j)/2 + 1, self.S - (i+j)/2 + 1, x0**2) -
                                        betainc((i+j)/2 + 1, self.S - (i+j)/2 + 1, x1**2)) *
                                        beta((i+j)/2 + 1, self.S - (i+j)/2 + 1)
                            / (2 * np.pi * (i - j) * np.sqrt(
                                beta(i + 1, self.S - i + 1) * beta(j + 1, self.S - j + 1) ) ) )  # fmt: skip
                        overlap_matrix[j, i] = overlap_matrix[i, j]

        return overlap_matrix
"""

@njit
def njit_StepAmplitudeTwoCopiesSwap(
    N: np.int64,
    vortices: np.int64,
    jastrows: np.array,
    jastrows_tmp: np.array,
    from_swap: np.array,
    from_swap_tmp: np.array,
) -> np.complex128:
    """
    Returns the ratio of wavefunctions for coordinates R_i
    to coordinates R_f, given that the particle with index p has moved."""
    step_amplitude = 1
    for copy in range(2):
        for n in range(copy * N, (copy + 1) * N):
            step_amplitude *= np.prod(
                np.power(
                    jastrows_tmp[
                        from_swap_tmp[n], from_swap_tmp[n + 1 : (copy + 1) * N], 0, 0
                    ]
                    / jastrows[
                        from_swap[n], from_swap[n + 1 : (copy + 1) * N], 0, 0
                    ],  # nopep8
                    vortices,
                )
            )

    return step_amplitude


class MonteCarloSphereLaughlin(MonteCarloSphere):
    spinors: np.array
    spinors_tmp: np.array
    jastrows: np.array
    jastrows_tmp: np.array

    def GetOverlapMatrix(self):
        overlap_matrix = np.zeros((self.N, self.N), dtype=np.complex128)

        x0 = np.cos(self.region_theta[0] / 2)
        x1 = np.cos(self.region_theta[1] / 2)
        d_phi = (self.region_phi[1] - self.region_phi[0]) / (2 * np.pi)

        for i in range(self.N):
            overlap_matrix[i, i] = d_phi*(betainc(i + 1, self.S - i + 1, x0**2) -
                                        betainc(i + 1, self.S - i + 1, x1**2))
            if d_phi != 1:
                js = np.arange(i+1, self.N)
                overlap_matrix[i, js] = (
                            (np.exp(1j * self.region_phi[1] * (i - js))
                            - np.exp(1j * self.region_phi[0] * (i - js)) ) # fmt: skip
                            * (betainc((i+js)/2 + 1, self.S - (i+js)/2 + 1, x0**2) -
                               betainc((i+js)/2 + 1, self.S - (i+js)/2 + 1, x1**2)) *
                                        beta((i+js)/2 + 1, self.S - (i+js)/2 + 1)
                            / (2 * 1j* np.pi * (i - js) * np.sqrt(
                                beta(i + 1, self.S - i + 1) * beta(js + 1, self.S - js + 1) ) ) )  # fmt: skip
                overlap_matrix[js, i] = np.conj(overlap_matrix[i, js])

        return overlap_matrix

    def ComputeEntropyED(self, entropy="s2"):
        if self.N != self.S + 1:
            print(
                "Filling is not equal to 1, cannot compute entropy through correlation matrix method."
            )
        else:
            overlap_matrix = self.GetOverlapMatrix()
            eigs = np.linalg.eigvalsh(overlap_matrix)

            if entropy == "s2":
                return -np.sum(np.log(eigs**2 + (1 - eigs) ** 2))
            elif entropy == "svN":
                return -np.sum(eigs*np.log(eigs) + (1-eigs)*np.log(1 - eigs))

    def ComputeParticleFluctuationsED(self):
        overlap_matrix = self.GetOverlapMatrix()
        return np.trace(np.matmul(overlap_matrix, (np.eye(self.N) - overlap_matrix)))

    def InitialJastrows(self):
        for copy in range(self.moved_particles.size):
            for i in range(copy * self.N, (copy + 1) * self.N):
                for j in range(i + 1, (copy + 1) * self.N):
                    self.jastrows[i, j, 0, 0] = (
                        self.spinors[i, 0] * self.spinors[j, 1]
                        - self.spinors[j, 0] * self.spinors[i, 1]
                    )
                    self.jastrows[j, i, 0, 0] = -self.jastrows[i, j, 0, 0]

    def InitialJastrowsSwap(self):
        for i in range(self.N):
            for j in range(self.N, 2 * self.N):
                self.jastrows[i, j, 0, 0] = (
                    self.spinors[i, 0] * self.spinors[j, 1]
                    - self.spinors[j, 0] * self.spinors[i, 1]
                )
                self.jastrows[j, i, 0, 0] = -self.jastrows[i, j, 0, 0]

    def InitialWavefn(self):
        nbr_copies = self.coords.shape[0] // self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.uint16)
        self.spinors[:, 0] = np.cos(self.coords[..., 0] / 2) * np.exp(
            1j * self.coords[..., 1] / 2
        )
        self.spinors[:, 1] = np.sin(self.coords[..., 0] / 2) * np.exp(
            -1j * self.coords[..., 1] / 2
        )
        self.InitialJastrows()

        np.copyto(self.spinors_tmp, self.spinors)
        np.copyto(self.jastrows_tmp, self.jastrows)

    def InitialWavefnSwap(self):
        self.InitialJastrowsSwap()
        np.copyto(self.jastrows_tmp, self.jastrows)

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows_tmp[p, :, ...], self.jastrows[p, :, ...])
            np.copyto(self.jastrows_tmp[:, p, ...], self.jastrows[:, p, ...])
            np.copyto(self.spinors_tmp[p], self.spinors[p])

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows[p, :, ...], self.jastrows_tmp[p, :, ...])
            np.copyto(self.jastrows[:, p, ...], self.jastrows_tmp[:, p, ...])
            np.copyto(self.spinors[p], self.spinors_tmp[p])

    def TmpWavefn(self):
        for copy in range(self.moved_particles.size):
            p = self.moved_particles[copy]
            phase = np.exp(1j * self.coords_tmp[p, 1] / 2)
            self.spinors_tmp[p, 0] = np.cos(self.coords_tmp[p, 0] / 2) * phase
            self.spinors_tmp[p, 1] = np.sin(self.coords_tmp[p, 0] / 2) / phase
            njit_UpdateJastrows(
                self.spinors_tmp[copy * self.N : (copy + 1) * self.N],
                self.jastrows_tmp[
                    copy * self.N : (copy + 1) * self.N,
                    copy * self.N : (copy + 1) * self.N,
                    ...,
                ],
                self.moved_particles[copy] - copy * self.N,
            )

    def TmpWavefnSwap(self):
        njit_UpdateJastrowsSwap(
            self.coords_tmp,
            self.spinors_tmp,
            self.jastrows_tmp,
            self.moved_particles,
            self.to_swap_tmp,
        )

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.shape[0] // self.N
        for copy in range(nbr_copies):

            step_amplitude *= np.prod(
                np.power(
                    self.jastrows_tmp[
                        self.moved_particles[copy],
                        copy * self.N : (copy + 1) * self.N,
                        0,
                        0,
                    ]
                    / self.jastrows[
                        self.moved_particles[copy],
                        copy * self.N : (copy + 1) * self.N,
                        0,
                        0,
                    ],
                    self.vortices,
                )
            )
        return step_amplitude

    def StepAmplitudeTwoCopies(self) -> np.complex128:
        return self.StepAmplitude()

    def StepAmplitudeTwoCopiesSwap(self) -> np.complex128:
        """
        Returns the ratio of wavefunctions for coordinates R_i
        to coordinates R_f, given that the particle with index p has moved.
        step_amplitude = 1
        for copy in range(2):

            for n in range(copy*self.N, (copy+1)*self.N):
                step_amplitude *= np.prod(np.power(self.jastrows_tmp[self.from_swap_tmp[n], self.from_swap_tmp[n+1: (copy+1)*self.N], 0, 0] /
                                                   self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.N], 0, 0],  # nopep8
                                                   self.vortices))

        return step_amplitude"""

        return njit_StepAmplitudeTwoCopiesSwap(
            self.N,
            self.vortices,
            self.jastrows,
            self.jastrows_tmp,
            self.from_swap,
            self.from_swap_tmp,
        )

    def InitialMod(self):
        step_amplitude = 1
        for copy in range(2):
            for n in range(copy * self.N, (copy + 1) * self.N):
                for m in range(n + 1, (copy + 1) * self.N):
                    step_amplitude *= np.power(
                        (
                            self.jastrows[self.from_swap[n], self.from_swap[m], 0, 0]
                            / self.jastrows[n, m, 0, 0]
                        ),
                        self.vortices,
                    )

        return np.abs(step_amplitude)

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):

            for n in range(copy * self.N, (copy + 1) * self.N):
                for m in range(n + 1, (copy + 1) * self.N):
                    step_amplitude *= np.power(
                        (
                            np.conj(
                                self.jastrows[
                                    self.from_swap[n], self.from_swap[m], 0, 0
                                ]
                            )
                            * self.jastrows[n, m, 0, 0]
                        ),
                        self.vortices,
                    )
                    step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def __init__(
        self,
        N,
        S,
        nbr_iter,
        nbr_nonthermal,
        step_size,
        region_theta=180,
        region_phi=360,
        nbr_copies=1,
        hardcore_radius=0, 
        save_results=True,
        save_last_config=True,
        save_all_config=True,
        acceptance_ratio=0,
    ):

        super().__init__(
            N,
            S,
            nbr_iter,
            nbr_nonthermal,
            step_size,
            region_theta,
            region_phi,
            hardcore_radius,
            save_results,
            save_last_config,
            save_all_config,
            acceptance_ratio,
        )

        if self.S / (self.N - 1) == np.floor(self.S / (self.N - 1)):
            self.vortices = self.S / (self.N - 1)
            if self.vortices == 1:
                print("Initialising IQHE.")
                self.state = "iqhe"
            else:
                print(f"Initialising Laughling state at filling 1/{self.vortices:.0f}.")
                self.state = "laughlin"
        else:
            print("This filling does not correspond to a Laughlin state on the sphere.")

        self.S_eff = np.int64(self.S - self.vortices * (self.N - 1))

        self.spinors = np.zeros((nbr_copies * N, 2), dtype=np.complex128)
        self.jastrows = np.ones(
            (nbr_copies * N, nbr_copies * N, 1, 1), dtype=np.complex128
        )

        self.spinors_tmp = np.copy(self.spinors)
        self.jastrows_tmp = np.copy(self.jastrows)
