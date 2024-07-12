import numpy as np
from numba import jit, njit, prange
from scipy.special import sph_harm
from .MonteCarloSphere import MonteCarloSphere
from .fast_math import MonopoleHarmonics, combs, combs_vect


@njit
def pos(a: np.int64, b: np.int64) -> np.int64:
    return (a+b)*(a+b+1)//2 + a


@njit
def inv_pos(j: np.int64) -> tuple[np.int64, np.int64]:
    if j == 0:
        return (0, 0)
    elif 1 <= j and j <= 2:
        return (j-1, 2-j)
    elif 3 <= j and j <= 5:
        return (j-3, 5-j)
    elif 6 <= j and j <= 9:
        return (j-6, 9-j)
    elif 10 <= j and j <= 14:
        return (j-10, 14-j)
    elif 15 <= j and j <= 20:
        return (j-15, 20-j)


@njit
def JastrowDerivative(ind: np.int64, a: np.int64, b: np.int64, f_table: np.array
                      ) -> np.complex128:
    match a+b:
        case 0:
            return 1
        case 1:
            return f_table[ind, pos(a, b)]
        case 2:
            match a:
                case 2:
                    return f_table[ind, pos(1, 0)]**2 - f_table[ind, pos(2, 0)]
                case 1:
                    return f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 1)] - f_table[ind, pos(1, 1)]
                case 0:
                    return f_table[ind, pos(0, 1)]**2 - f_table[ind, pos(0, 2)]
        case 3:
            match a:
                case 3:
                    return (f_table[ind, pos(1, 0)]**3 - 3*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 0)] +
                            2*f_table[ind, pos(3, 0)])
                case 2:
                    return ((f_table[ind, pos(0, 1)])*(f_table[ind, pos(1, 0)])**2 -
                            2*f_table[ind, pos(1, 0)]*f_table[ind, pos(1, 1)] -
                            f_table[ind, pos(0, 1)]*f_table[ind, pos(2, 0)]
                            + 2*f_table[ind, pos(2, 1)])
                case 1:
                    return ((f_table[ind, pos(1, 0)])*(f_table[ind, pos(0, 1)])**2 -
                            2*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 1)] -
                            f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 2)]
                            + 2*f_table[ind, pos(1, 2)])
                case 0:
                    return (f_table[ind, pos(0, 1)]**3 - 3*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 2)] +
                            2*f_table[ind, pos(0, 3)])
        case 4:
            match a:
                case 4:
                    return (f_table[ind, pos(1, 0)]**4 - 6*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(2, 0)] +
                            3*f_table[ind, pos(2, 0)]**2 + 8*f_table[ind, pos(1, 0)]*f_table[ind, pos(3, 0)] -
                            6*f_table[ind, pos(4, 0)])
                case 3:
                    return ((f_table[ind, pos(0, 1)])*(f_table[ind, pos(1, 0)])**3 -
                            3*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(1, 1)] -
                            3*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 0)] +
                            3*f_table[ind, pos(1, 1)]*f_table[ind, pos(2, 0)] +
                            6*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 1)] +
                            2*f_table[ind, pos(0, 1)]*f_table[ind, pos(3, 0)]
                            - 6*f_table[ind, pos(3, 1)])
                case 2:
                    return ((f_table[ind, pos(0, 1)]**2)*(f_table[ind, pos(1, 0)]**2) -
                            f_table[ind, pos(0, 2)]*(f_table[ind, pos(1, 0)]**2) -
                            4*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 1)] +
                            2*f_table[ind, pos(1, 1)]**2 +
                            4*f_table[ind, pos(1, 0)]*f_table[ind, pos(1, 2)] -
                            (f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(2, 0)] +
                            f_table[ind, pos(0, 2)]*f_table[ind, pos(2, 0)] +
                            4*f_table[ind, pos(0, 1)]*f_table[ind, pos(2, 1)] -
                            6*f_table[ind, pos(2, 2)])
                case 1:
                    return ((f_table[ind, pos(1, 0)])*(f_table[ind, pos(0, 1)])**3 -
                            3*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(1, 1)] -
                            3*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 2)] +
                            3*f_table[ind, pos(1, 1)]*f_table[ind, pos(0, 2)] +
                            6*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 2)] +
                            2*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 3)]
                            - 6*f_table[ind, pos(1, 3)])
                case 0:
                    return (f_table[ind, pos(0, 1)]**4 - 6*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(0, 2)] +
                            3*f_table[ind, pos(0, 2)]**2 + 8*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 3)] -
                            6*f_table[ind, pos(0, 4)])
        case 5:
            match a:
                case 5:
                    return (f_table[ind, pos(1, 0)]**5 - 10*(f_table[ind, pos(1, 0)]**3)*f_table[ind, pos(2, 0)] +
                            15*f_table[ind, pos(1, 0)]*(f_table[ind, pos(2, 0)]**2) +
                            20*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(3, 0)] -
                            20*f_table[ind, pos(2, 0)]*f_table[ind, pos(3, 0)] -
                            30*f_table[ind, pos(1, 0)]*f_table[ind, pos(4, 0)] +
                            24*f_table[ind, pos(5, 0)])
                case 4:
                    return ((f_table[ind, pos(0, 1)])*(f_table[ind, pos(1, 0)])**4 -
                            4*(f_table[ind, pos(1, 0)]**3)*f_table[ind, pos(1, 1)] -
                            6*f_table[ind, pos(0, 1)]*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(2, 0)] +
                            12*f_table[ind, pos(1, 0)]*f_table[ind, pos(1, 1)]*f_table[ind, pos(2, 0)] +
                            3*f_table[ind, pos(0, 1)]*(f_table[ind, pos(2, 0)]**2) +
                            12*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(2, 1)] -
                            12*f_table[ind, pos(2, 0)]*f_table[ind, pos(2, 1)] +
                            8*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 0)]*f_table[ind, pos(3, 0)] -
                            8*f_table[ind, pos(1, 1)]*f_table[ind, pos(3, 0)] -
                            24*f_table[ind, pos(1, 0)]*f_table[ind, pos(3, 1)] -
                            6*f_table[ind, pos(0, 1)]*f_table[ind, pos(4, 0)] +
                            24*f_table[ind, pos(4, 1)])
                case 3:
                    return ((f_table[ind, pos(0, 1)]**2)*(f_table[ind, pos(1, 0)]**3) -
                            f_table[ind, pos(0, 2)]*(f_table[ind, pos(1, 0)]**3) -
                            6*f_table[ind, pos(0, 1)]*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(1, 1)] +
                            6*f_table[ind, pos(1, 0)]*(f_table[ind, pos(1, 1)]**2) +
                            6*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(1, 2)] -
                            3*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 0)] +
                            3*f_table[ind, pos(0, 2)]*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 0)] +
                            6*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 1)]*f_table[ind, pos(2, 0)] -
                            6*f_table[ind, pos(1, 2)]*f_table[ind, pos(2, 0)] +
                            12*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 1)] -
                            12*f_table[ind, pos(1, 1)]*f_table[ind, pos(2, 1)] -
                            18*f_table[ind, pos(1, 0)]*f_table[ind, pos(2, 2)] +
                            2*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(3, 0)] -
                            2*f_table[ind, pos(0, 2)]*f_table[ind, pos(3, 0)] -
                            12*f_table[ind, pos(0, 1)]*f_table[ind, pos(3, 1)] +
                            24*f_table[ind, pos(3, 2)])
                case 2:
                    return ((f_table[ind, pos(1, 0)]**2)*(f_table[ind, pos(0, 1)]**3) -
                            f_table[ind, pos(2, 0)]*(f_table[ind, pos(0, 1)]**3) -
                            6*f_table[ind, pos(1, 0)]*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(1, 1)] +
                            6*f_table[ind, pos(0, 1)]*(f_table[ind, pos(1, 1)]**2) +
                            6*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(2, 1)] -
                            3*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 2)] +
                            3*f_table[ind, pos(2, 0)]*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 2)] +
                            6*f_table[ind, pos(1, 0)]*f_table[ind, pos(1, 1)]*f_table[ind, pos(0, 2)] -
                            6*f_table[ind, pos(2, 1)]*f_table[ind, pos(0, 2)] +
                            12*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 2)] -
                            12*f_table[ind, pos(1, 1)]*f_table[ind, pos(1, 2)] -
                            18*f_table[ind, pos(0, 1)]*f_table[ind, pos(2, 2)] +
                            2*(f_table[ind, pos(1, 0)]**2)*f_table[ind, pos(0, 3)] -
                            2*f_table[ind, pos(2, 0)]*f_table[ind, pos(0, 3)] -
                            12*f_table[ind, pos(1, 0)]*f_table[ind, pos(1, 3)] +
                            24*f_table[ind, pos(2, 3)])
                case 1:
                    return ((f_table[ind, pos(1, 0)])*(f_table[ind, pos(0, 1)])**4 -
                            4*(f_table[ind, pos(0, 1)]**3)*f_table[ind, pos(1, 1)] -
                            6*f_table[ind, pos(1, 0)]*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(0, 2)] +
                            12*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 1)]*f_table[ind, pos(0, 2)] +
                            3*f_table[ind, pos(1, 0)]*(f_table[ind, pos(0, 2)]**2) +
                            12*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(1, 2)] -
                            12*f_table[ind, pos(0, 2)]*f_table[ind, pos(1, 2)] +
                            8*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 3)] -
                            8*f_table[ind, pos(1, 1)]*f_table[ind, pos(0, 3)] -
                            24*f_table[ind, pos(0, 1)]*f_table[ind, pos(1, 3)] -
                            6*f_table[ind, pos(1, 0)]*f_table[ind, pos(0, 4)] +
                            24*f_table[ind, pos(1, 4)])
                case 0:
                    return (f_table[ind, pos(0, 1)]**5 - 10*(f_table[ind, pos(0, 1)]**3)*f_table[ind, pos(0, 2)] +
                            15*f_table[ind, pos(0, 1)]*(f_table[ind, pos(0, 2)]**2) +
                            20*(f_table[ind, pos(0, 1)]**2)*f_table[ind, pos(0, 3)] -
                            20*f_table[ind, pos(0, 2)]*f_table[ind, pos(0, 3)] -
                            30*f_table[ind, pos(0, 1)]*f_table[ind, pos(0, 4)] +
                            24*f_table[ind, pos(0, 5)])


@njit  # (parallel=True)
def njit_CalculateDerivativeTable(f_table: np.array,
                                  spinors: np.array, jastrows: np.array, p: np.int64):
    f_table[:, 0] = np.ones(f_table.shape[0])*(f_table.shape[0]-1)
    n = 1
    a = 0
    b = n
    for j in range(1, f_table.shape[1]):
        for i in range(f_table.shape[0]):
            f_table[i, j] = (np.dot(np.power(spinors[:, 1]/jastrows[i, :], a),
                                    np.power(-spinors[:, 0]/jastrows[i, :], b))  # -
                             )  # (spinors[i, 1]**a)*((-spinors[i, 0])**b))
        f_table[:, j] -= (spinors[:, 1]**a)*((-spinors[:, 0])**b)
        if a == n:
            n += 1
            a = 0
            b = n
        else:
            a += 1
            b -= 1

    f_table *= p


@njit  # (parallel=True)
def njit_UpdateDerivativeTable(f_table_tmp: np.array, moved_particle: np.int64,
                               spinors_tmp: np.array, spinors: np.array,
                               jastrows_tmp: np.array, jastrows: np.array, p: np.int64):
    n = 1
    a = 0
    b = n
    for j in range(1, f_table_tmp.shape[1]):
        f_table_tmp[moved_particle, j] = p*(np.dot(np.power(spinors_tmp[:, 1]/jastrows_tmp[moved_particle, :], a),
                                                   np.power(-spinors_tmp[:, 0]/jastrows_tmp[moved_particle, :], b)) -
                                            (spinors[moved_particle, 1]**a)*((-spinors[moved_particle, 0])**b))
        f_table_tmp[:, j] += p*(((spinors_tmp[moved_particle, 1]/jastrows_tmp[:, moved_particle])**a) *
                                ((-spinors_tmp[moved_particle, 0]/jastrows_tmp[:, moved_particle])**b) -
                                ((spinors[moved_particle, 1]/jastrows[:, moved_particle])**a) *
                                ((-spinors[moved_particle, 0]/jastrows[:, moved_particle])**b))
        if a == n:
            n += 1
            a = 0
            b = n
        else:
            a += 1
            b -= 1


@njit  # (parallel=True)
def njit_UpdateDerivativeTableLevel(f_table_tmp: np.array, level: np.int64,
                                    moved_particle: np.int64,
                                    spinors_tmp: np.array, spinors: np.array,
                                    jastrows_tmp: np.array, jastrows: np.array, p: np.int64):
    a = 0
    b = level
    j_start = pos(a, b)
    j_end = pos(b, a)
    for j in range(j_start, j_end+1):
        f_table_tmp[moved_particle, j] = p*(np.dot(np.power(spinors_tmp[:, 1]/jastrows_tmp[moved_particle, :], a),
                                                   np.power(-spinors_tmp[:, 0]/jastrows_tmp[moved_particle, :], b)) -
                                            (spinors[moved_particle, 1]**a)*((-spinors[moved_particle, 0])**b))
        f_table_tmp[:, j] += p*(((spinors_tmp[moved_particle, 1]/jastrows_tmp[:, moved_particle])**a) *
                                ((-spinors_tmp[moved_particle, 0]/jastrows_tmp[:, moved_particle])**b) -
                                ((spinors[moved_particle, 1]/jastrows[:, moved_particle])**a) *
                                ((-spinors[moved_particle, 0]/jastrows[:, moved_particle])**b))
        a += 1
        b -= 1


@njit  # (parallel=True)
def njit_UpdateSlaterProj(N: np.int64, S: np.int64, S_eff: np.int64,
                          spinors_tmp: np.array, spinors: np.array,
                          jastrows_tmp: np.array, jastrows: np.array, slater: np.array,
                          Ls: np.array, f_table_tmp: np.array, norms_CF: np.array,
                          moved_particle: np.int64, p: np.int64):

    for k in range(Ls.shape[0]):
        if Ls[k, 1] == 0:
            njit_UpdateDerivativeTableLevel(f_table_tmp, Ls[k, 0], moved_particle,
                                            spinors_tmp, spinors, jastrows_tmp, jastrows, p)
        for i in range(N):
            slater[k, i] = 0
            for s in range(max(0, Ls[k, 0]-Ls[k, 1]), min(S_eff+2*Ls[k, 0]-Ls[k, 1], Ls[k, 0])+1):
                # for s in range(0, Ls[k, 0]+1):
                # if (S_eff+2*Ls[k, 0]-Ls[k, 1]-s >= 0) and (S_eff+2*Ls[k, 0]-Ls[k, 1]-s <= S_eff+Ls[k, 0]):
                slater[k, i] += (((-1)**s)*combs(Ls[k, 0], s)*combs(S_eff+Ls[k, 0], S_eff+2*Ls[k, 0]-Ls[k, 1]-s) *
                                 ((spinors_tmp[i, 0]/spinors_tmp[i, 1])**s)*JastrowDerivative(i, s, Ls[k, 0]-s, f_table_tmp))
            # slater[k, i] *= (norms_CF[k] * (spinors[i, 0]**(Ls[k, 1]-Ls[k, 0])) *
            #                 (spinors[i, 1]**(S_eff + 2*Ls[k, 0] - Ls[k, 1])))
            slater[k, i] *= (norms_CF[k] * (spinors_tmp[i, 0]**(Ls[k, 1]-Ls[k, 0])) *
                             (spinors_tmp[i, 1]**(S_eff + 2*Ls[k, 0] - Ls[k, 1])))


@njit(parallel=True)
def njit_GetSlaterProj(N: np.int64, S: np.int64, S_eff: np.int64,
                       spinors: np.array, slater: np.array,
                       Ls: np.array, f_table: np.array,
                       norms_CF: np.array, combinations: np.array):
    L_max = np.int64(np.sqrt(N))-1
    spinors_ratio = np.zeros((N, L_max+1), dtype=np.complex128)
    for i in prange(N):
        spinors_ratio[i, :] = np.power(
            -spinors[i, 0]/spinors[i, 1], np.arange(L_max+1))
        for k in range(Ls.shape[0]):
            if Ls[k, 1] == 0:
                P_table = np.zeros(Ls[k, 0]+1, dtype=np.complex128)
                for l in range(Ls[k, 0]+1):
                    P_table[l] = JastrowDerivative(i, l, Ls[k, 0]-l, f_table)
            slater[k, i] = 0
            s_min = max(0, Ls[k, 0]-Ls[k, 1])
            s_max = min(S_eff+2*Ls[k, 0]-Ls[k, 1], Ls[k, 0])+1
            for s in range(s_min, s_max):
                slater[k, i] += (combinations[Ls[k, 0], s]*combinations[S_eff+Ls[k, 0], S_eff+2*Ls[k, 0]-Ls[k, 1]-s] *
                                 spinors_ratio[i, s]*P_table[s])  #
        slater[:, i] *= (norms_CF[:] * (spinors[i, 0]**(Ls[:, 1]-Ls[:, 0])) *
                         (spinors[i, 1]**(S_eff + 2*Ls[:, 0] - Ls[:, 1])))


def njit_UpdateSlaterUnproj(S_eff, coords, moved_particle, slater, Ls):
    if S_eff == 0:
        slater[:, moved_particle] = sph_harm(Ls[:, 1] - Ls[:, 0] - S_eff/2,
                                             Ls[:, 0] + S_eff/2,
                                             coords[moved_particle, 1], coords[moved_particle, 0])
    else:
        slater[:, moved_particle] = MonopoleHarmonics(
            S_eff, Ls[:, 0],  Ls[:, 1], coords[moved_particle, 0], coords[moved_particle, 1])


def njit_UpdateJastrows(S_eff: np.int64, coords_tmp: np.array, spinors_tmp: np.array,
                        jastrows_tmp: np.array, slater_tmp: np.array,
                        Ls: np.array, p: np.int64):
    jastrows_tmp[:, p, 0, 0] = (
        spinors_tmp[:, 0]*spinors_tmp[p, 1] -
        spinors_tmp[p, 0]*spinors_tmp[:, 1])
    jastrows_tmp[p, :, 0, 0] = - \
        jastrows_tmp[:, p, 0, 0]
    jastrows_tmp[p, p] = 1


# @njit
def njit_UpdateJastrowsSwap(coords_tmp: np.array, spinors_tmp: np.array,
                            jastrows_tmp: np.array, moved_particles: np.array,
                            to_swap_tmp: np.array):
    N = coords_tmp.shape[0]//2
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
def njit_StepAmplitudeTwoCopiesSwap(N: np.int64, nbr_vortices: np.int64, jastrows: np.array,
                                    jastrows_tmp: np.array, slogdet: np.array, slogdet_tmp: np.array,
                                    from_swap: np.array, from_swap_tmp: np.array, no_vortex: np.bool_
                                    ) -> np.complex128:
    """
    Returns the ratio of wavefunctions for coordinates R_i
    to coordinates R_f, given that the particle with index p has moved."""
    step_amplitude = 1
    for copy in range(2):
        step_amplitude *= (slogdet_tmp[0, 2+copy] /
                           slogdet[0, 2+copy])

        for n in range(copy*N, (copy+1)*N):
            vortices = np.power(jastrows_tmp[from_swap_tmp[n], from_swap_tmp[n+1: (copy+1)*N], 0, 0] /
                                              jastrows[from_swap[n], from_swap[n+1: (copy+1)*N], 0, 0],  # nopep8
                                              nbr_vortices)
            if no_vortex:
                vortices /= np.abs(vortices)
            step_amplitude *= np.prod(np.exp((slogdet_tmp[1, 2+copy]-slogdet[1, 2+copy]) / (N*(N-1)/2)) *
                                      vortices)

    return step_amplitude


@njit
def njit_DensityCF(N: np.int64, inside_region: np.array, jastrows: np.array, nbr_vortices: np.int64):
    cf_density = 0
    for i in range(N):
        if inside_region[i]:
            trunc_cf_operator = 1
            for j in range(N):
                if inside_region[j]:
                    trunc_cf_operator *= np.abs(jastrows[i, j]
                                                )**(2*nbr_vortices)
            cf_density += trunc_cf_operator
            # cf_density += (np.prod(np.sqrt(self.N)*np.power(np.abs(self.jastrows[i, :, 0, 0]),
            #                                                2*self.nbr_vortices)))

    return cf_density


class MonteCarloSphereCFL (MonteCarloSphere):
    spinors: np.array
    spinors_tmp: np.array
    jastrows: np.array
    jastrows_tmp: np.array
    slater: np.array
    slater_tmp: np.array
    slogdet: np.array
    slogdet_tmp: np.array
    flag_proj: np.bool_
    f_table: np.array
    nbr_vortices: np.int64
    no_vortex:  np.bool_
    combinations: np.array

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
        for i in range(self.N):
            if self.S_eff == 0:
                slater[i, :] = sph_harm(self.Ls[i, 1] - self.Ls[i, 0] - self.S_eff/2,
                                        self.Ls[i, 0] + self.S_eff/2, coords[:, 1], coords[:, 0])
            else:
                slater[i, :] = MonopoleHarmonics(
                    self.S_eff, self.Ls[i, 0], self.Ls[i, 1], coords[:, 0], coords[:, 1])

    def InitialWavefn(self):
        nbr_copies = self.coords.shape[0]//self.N
        self.moved_particles = np.zeros(nbr_copies, dtype=np.int64)
        self.spinors[:, 0] = (np.cos(self.coords[..., 0]/2) *
                              np.exp(1j*self.coords[..., 1]/2))
        self.spinors[:, 1] = (np.sin(self.coords[..., 0]/2) *
                              np.exp(-1j*self.coords[..., 1]/2))
        self.InitialJastrows()
        for copy in range(nbr_copies):
            if not self.flag_proj:
                self.InitialSlater(self.coords[copy*self.N:(copy+1)*self.N],
                                   self.slater[..., copy])
            else:
                njit_CalculateDerivativeTable(
                    self.f_table[..., copy], self.spinors[copy*self.N:(copy+1)*self.N, :],  # nopep8
                    self.jastrows[copy*self.N:(copy+1)*self.N, copy*self.N:(copy+1)*self.N, 0, 0], 1)
                njit_GetSlaterProj(
                    self.N, self.S, self.S_eff,
                    self.spinors[copy*self.N:(copy+1)*self.N],
                    self.slater[..., copy], self.Ls, self.f_table[..., copy],
                    self.norms_CF, self.combinations)
            self.slogdet[:, copy] = np.linalg.slogdet(
                self.slater[..., copy])

        np.copyto(self.spinors_tmp, self.spinors)
        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)
        np.copyto(self.f_table_tmp, self.f_table)

    def InitialWavefnSwap(self):
        coords_swap = self.coords[self.from_swap]
        self.InitialJastrowsSwap()
        for copy in range(2):
            if self.flag_proj:
                njit_CalculateDerivativeTable(
                    self.f_table[..., 2+copy], self.spinors[self.from_swap[copy*self.N:(copy+1)*self.N]],  # nopep8
                    self.jastrows[..., 0, 0][np.ix_(self.from_swap[copy*self.N:(copy+1)*self.N], self.from_swap[copy*self.N:(copy+1)*self.N])], 1)  # nopep8
                njit_GetSlaterProj(
                    self.N, self.S, self.S_eff,
                    self.spinors[self.from_swap[copy * self.N:(copy+1)*self.N]],  # nopep8
                    self.slater[..., 2+copy], self.Ls, self.f_table[..., 2+copy],  # nopep8
                    self.norms_CF, self.combinations)
            else:
                self.InitialSlater(coords_swap[copy*self.N:(copy+1)*self.N],
                                   self.slater[..., 2+copy])
            self.slogdet[:, 2+copy] = np.linalg.slogdet(
                self.slater[..., 2+copy])

        np.copyto(self.jastrows_tmp, self.jastrows)
        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)
        np.copyto(self.f_table_tmp, self.f_table)

    def RejectTmp(self, run_type):
        super().RejectTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows_tmp[p, :, ...], self.jastrows[p, :, ...])
            np.copyto(self.jastrows_tmp[:, p, ...], self.jastrows[:, p, ...])
            np.copyto(self.spinors_tmp[p], self.spinors[p])

        np.copyto(self.slater_tmp, self.slater)
        np.copyto(self.slogdet_tmp, self.slogdet)
        np.copyto(self.f_table_tmp, self.f_table)

    def AcceptTmp(self, run_type):
        super().AcceptTmp(run_type)

        for p in self.moved_particles:
            np.copyto(self.jastrows[p, :, ...], self.jastrows_tmp[p, :, ...])
            np.copyto(self.jastrows[:, p, ...], self.jastrows_tmp[:, p, ...])
            np.copyto(self.spinors[p], self.spinors_tmp[p])

        np.copyto(self.slater, self.slater_tmp)
        np.copyto(self.slogdet, self.slogdet_tmp)
        np.copyto(self.f_table, self.f_table_tmp)

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
            if self.flag_proj:
                """njit_UpdateSlaterProj(
                    self.N, self.S, self.S_eff,
                    self.spinors_tmp[copy*self.N:(copy+1) * self.N], self.spinors[copy*self.N:(copy+1)*self.N],  # nopep8
                    self.jastrows_tmp[copy*self.N:(copy+1)*self.N, copy*self.N:(copy+1)*self.N, 0, 0],  # nopep8
                    self.jastrows[copy*self.N:(copy+1)*self.N, copy*self.N:(copy+1)*self.N, 0, 0],  # nopep8
                    self.slater_tmp[..., copy], self.Ls, self.f_table_tmp[...,
                        copy], self.norms_CF,
                    self.moved_particles[copy]-copy*self.N, 1)"""
                njit_UpdateDerivativeTable(
                    self.f_table_tmp[..., copy], self.moved_particles[copy]-copy*self.N,  # nopep8
                    self.spinors_tmp[copy*self.N:(copy+1)*self.N, :], self.spinors[copy*self.N:(copy+1)*self.N, :],  # nopep8
                    self.jastrows_tmp[copy*self.N:(copy+1)*self.N,
                                                   copy*self.N:(copy+1)*self.N, 0, 0],
                    self.jastrows[copy*self.N:(copy+1)*self.N, copy*self.N:(copy+1)*self.N, 0, 0], 1)
                njit_GetSlaterProj(
                    self.N, self.S, self.S_eff,
                    self.spinors_tmp[copy*self.N:(copy+1)*self.N],
                    self.slater_tmp[..., copy], self.Ls, self.f_table_tmp[..., copy],  # nopep8
                    self.norms_CF, self.combinations)
            else:
                njit_UpdateSlaterUnproj(self.S_eff, self.coords_tmp[copy*self.N:(copy+1)*self.N],
                                        self.moved_particles[copy]-copy*self.N,
                                        self.slater_tmp[..., copy], self.Ls)

            self.slogdet_tmp[:, copy] = np.linalg.slogdet(
                self.slater_tmp[..., copy])

    def TmpWavefnSwap(self):
        coords_swap_tmp = self.coords_tmp[self.from_swap_tmp]
        njit_UpdateJastrowsSwap(self.coords_tmp, self.spinors_tmp,
                                self.jastrows_tmp, self.moved_particles,
                                self.to_swap_tmp)

        for copy in range(2):
            swap_copy = self.to_swap_tmp[self.moved_particles[copy]] // self.N
            if self.flag_proj:
                njit_UpdateDerivativeTable(
                            self.f_table_tmp[..., 2+swap_copy], self.to_swap_tmp[self.moved_particles[copy]] % self.N,  # nopep8
                            self.spinors_tmp[self.from_swap_tmp[swap_copy*self.N:(swap_copy+1)*self.N]],  # nopep8
                            self.spinors[self.from_swap[swap_copy*self.N:(swap_copy+1)*self.N]],  # nopep8
                            self.jastrows_tmp[..., 0, 0][np.ix_(self.from_swap_tmp[swap_copy*self.N:(swap_copy+1)*self.N], self.from_swap_tmp[swap_copy*self.N:(swap_copy+1)*self.N])],  # nopep8
                            self.jastrows[..., 0, 0][np.ix_(self.from_swap[swap_copy*self.N:(swap_copy+1)*self.N], self.from_swap[swap_copy*self.N:(swap_copy+1)*self.N])], 1)
            else:
                njit_UpdateSlaterUnproj(self.S_eff, coords_swap_tmp[swap_copy*self.N:(swap_copy+1)*self.N],
                                        self.to_swap_tmp[self.moved_particles[copy]] % self.N,  # nopep8
                                        self.slater_tmp[..., 2+swap_copy], self.Ls)

        if self.flag_proj:
            for copy in range(2):
                njit_GetSlaterProj(
                    self.N, self.S, self.S_eff,
                    self.spinors_tmp[self.from_swap_tmp[copy *
                        self.N:(copy+1)*self.N]],
                    self.slater_tmp[..., 2+copy], self.Ls, self.f_table_tmp[..., 2+copy],  # nopep8
                    self.norms_CF, self.combinations)

        self.slogdet_tmp[:, 2] = np.linalg.slogdet(self.slater_tmp[..., 2])
        self.slogdet_tmp[:, 3] = np.linalg.slogdet(self.slater_tmp[..., 3])

    def StepAmplitude(self) -> np.complex128:
        step_amplitude = 1
        nbr_copies = self.coords.shape[0]//self.N
        for copy in range(nbr_copies):
            step_amplitude *= (self.slogdet_tmp[0, copy]/self.slogdet[0, copy])

            vortices = np.power(self.jastrows_tmp[self.moved_particles[copy],
                                                  copy*self.N:(copy+1)*self.N, 0, 0] /
                                self.jastrows[self.moved_particles[copy],
                                              copy*self.N:(copy+1)*self.N, 0, 0],
                                self.nbr_vortices)

            if self.no_vortex:
                vortices /= np.abs(vortices)

            step_amplitude *= (np.prod(np.exp((self.slogdet_tmp[1, copy]-self.slogdet[1, copy])/self.N) *
                                       vortices))
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
                                                   self.nbr_vortices))

        return step_amplitude"""
        return njit_StepAmplitudeTwoCopiesSwap(self.N, self.nbr_vortices, self.jastrows, self.jastrows_tmp,
                                               self.slogdet, self.slogdet_tmp, self.from_swap, self.from_swap_tmp,
                                               self.no_vortex)

    def InitialMod(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (self.slogdet[0, copy+2] / self.slogdet[0, copy])

            for n in range(copy*self.N, (copy+1)*self.N):
                vortices = np.power((self.jastrows[self.from_swap[n], self.from_swap[n+1: (copy+1)*self.N], 0, 0] /
                                     self.jastrows[n, n+1:(copy+1)*self.N, 0, 0]), self.nbr_vortices)
                if self.no_vortex:
                    vortices /= np.abs(vortices)
                step_amplitude *= np.prod((np.exp((self.slogdet[1, copy+2] - self.slogdet[1, copy]) / (self.N*(self.N-1)/2)) *
                                          vortices))

        return np.abs(step_amplitude)

    def InitialSign(self):
        step_amplitude = 1
        for copy in range(2):
            step_amplitude *= (np.conj(self.slogdet[0, copy+2])
                               * self.slogdet[0, copy])

            for n in range(copy*self.N, (copy+1)*self.N):
                step_amplitude *= np.prod(np.power((np.conj(
                    self.jastrows[self.from_swap[n], self.from_swap[n+1:(copy+1)*self.N], 0, 0]) *
                    self.jastrows[n, n+1:(copy+1)*self.N, 0, 0]), self.nbr_vortices))
                step_amplitude /= np.abs(step_amplitude)

        return step_amplitude

    def DensityCF(self):
        return njit_DensityCF(self.N, self.InsideRegion(self.coords), self.jastrows[:, :, 0, 0], self.nbr_vortices)

    def __init__(self, N, S, nbr_iter, nbr_nonthermal,
                 step_size, region_theta=180, region_phi=360, nbr_copies=1,
                 JK_coeffs='0', no_vortex=False, save_results=True, save_last_config=True,
                 save_all_config=True, acceptance_ratio=0):

        super().__init__(N, S, nbr_iter, nbr_nonthermal,
                         step_size, region_theta, region_phi,
                         save_results, save_last_config,
                         save_all_config, acceptance_ratio)

        if self.S/(self.N-1) == np.floor(self.S/(self.N-1)):
            print("Initialising HLR-CFL.")
            self.nbr_vortices = self.S/(self.N-1)
        elif self.S == 2*self.N-1:
            print("Initialising Son-CFL.")
            self.nbr_vortices = 2

        self.S_eff = np.int64(self.S - self.nbr_vortices*(self.N-1))

        self.no_vortex = no_vortex

        self.FillLambdaLevels()

        self.state = 'cfl'+JK_coeffs

        self.JK_coeffs = np.int64(int(JK_coeffs))
        if self.JK_coeffs == 0:
            self.flag_proj = False
        else:
            self.flag_proj = True
            self.norms_CF = np.zeros(self.Ls.shape[0])
            for i in range(self.Ls.shape[0]):
                self.norms_CF[i] = ((-1)**(self.S_eff+2*self.Ls[i, 0]-self.Ls[i, 1]))*np.sqrt(
                    (self.S_eff+2*self.Ls[i, 0]+1) *
                    combs(self.S_eff+2*self.Ls[i, 0], self.Ls[i, 0]) /
                    (combs(self.S_eff+2*self.Ls[i, 0], self.Ls[i, 1])*4*np.pi) /
                    np.prod(np.arange(self.S+2, self.S+self.Ls[i, 0]+2)))
        self.f_table = np.zeros(
            (N, np.int64(np.sqrt(N)*(np.sqrt(N)+1))//2, 4**(nbr_copies-1)), dtype=np.complex128)
        self.f_table_tmp = np.zeros(
            (N, np.int64(np.sqrt(N)*(np.sqrt(N)+1))//2, 4**(nbr_copies-1)), dtype=np.complex128)
        self.P_table = np.zeros(
            (N, np.int64(np.sqrt(N)*(np.sqrt(N)+1))//2, 4**(nbr_copies-1)), dtype=np.complex128)
        self.combinations = np.zeros(
            (np.int64(np.sqrt(N)), np.int64(np.sqrt(N))))
        for c in range(np.int64(np.sqrt(N))):
            self.combinations[c, :(c+1)] = combs_vect(c, np.arange(c+1))

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
