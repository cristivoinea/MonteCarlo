import numpy as np
from numba import njit


def Stats(data: np.array) -> (np.float64, np.float64):
    data_copy = data
    mean = np.sum(data_copy)/data_copy.size
    var = np.var(data_copy, ddof=1)

    return mean, var


def FullStats(data: np.array
              ) -> (np.array, np.array):

    mean_vector = np.zeros(data.size, dtype=np.float64)
    std_vector = np.zeros(data.size, dtype=np.float64)
    for i in range(data.size):
        mean_vector[i] = np.sum(data[:i+1])/data[:i+1].size
        std_vector[i] = np.std(data[:i+1], ddof=1)

    return mean_vector, std_vector


def GetEntropy(Ne, Ns, M, M0, t, step, region_geometry):
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    data = np.loadtxt(
        f"laughlin_SWAP_Ne_{Ne}_Ns_{Ns}_t_1.00_step_{step:.3f}_{region_geometry}s.dat")
    boundaries = data[:, 0]*2*np.pi*Lx
    p = data[:, 1:3]
    mod = data[:, 3:5]
    sign = data[:, 5:7]

    S2_p = -np.log(p[:, 0])
    err_p = np.sqrt(p[:, 1]/(p[:, 0]**2))/np.sqrt(M-M0)
    S2_mod = -np.log(mod[:, 0])
    err_mod = np.sqrt(mod[:, 1]/(mod[:, 0]**2))/np.sqrt(M-M0)
    S2_sign = -np.log(sign[:, 0])
    err_sign = np.sqrt(sign[:, 1]/(sign[:, 0]**2))/np.sqrt(M-M0)

    S2 = S2_p + S2_mod + S2_sign
    err = np.sqrt(err_p**2 + err_mod**2 + err_sign**2)

    return boundaries, np.vstack((S2, err)).T, np.vstack((S2_p, err_p)).T, np.vstack((S2_mod, err_mod)).T, np.vstack((S2_sign, err_sign)).T


def SaveConfig(state: str, SWAP_term: str, Ne: np.uint8, Ns: np.uint16,
               Lx: np.float64, Ly: np.float64, t: np.complex128,
               step_size: np.float64, region_geometry: str, boundary: np.float64,
               result: np.array, R: np.array, swap_order: np.array):
    if SWAP_term == 'sign':
        np.save(f"{state}_full_{SWAP_term}_real_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.npy",
                np.real(result))
        np.save(f"{state}_full_{SWAP_term}_imag_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.npy",
                np.imag(result))

    else:
        np.save(f"{state}_full_{SWAP_term}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.npy",
                result)

    if SWAP_term == 'sign' or SWAP_term == 'mod':
        np.save(f"{state}_{SWAP_term}_order_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.npy",
                swap_order)

    np.save(f"{state}_{SWAP_term}_coords_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.npy",
            R)


def SaveResult(state: str, SWAP_term: str, Ne: np.uint8, Ns: np.uint16,
               Lx: np.float64, Ly: np.float64, M0: np.uint32, t: np.complex128,
               step_size: np.float64, region_geometry: str, boundary: np.float64,
               result: np.array, save_result: np.bool_):

    if save_result:
        if SWAP_term == 'sign':
            np.savetxt(f"{state}_{SWAP_term}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.dat",
                       np.vstack((Stats(np.real(result[int(M0):])),
                                  Stats(np.imag(result[int(M0):]))
                                  ))
                       )
        else:
            np.savetxt(f"{state}_{SWAP_term}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{boundary/Ly:.3f}_step_{step_size/Lx:.3f}.dat",
                       Stats(result[int(M0):]))

    else:
        if SWAP_term == 'sign':
            mean, var = Stats(np.real(result[int(M0):]))
            print(f"\nMean (real) = {mean} \n Var (real) = {var}")
            mean, var = Stats(np.imag(result[int(M0):]))
            print(f"\nMean (imag) = {mean} \n Var (imag) = {var}")
        else:
            mean, var = Stats(result[int(M0):])
            print(f"\nMean = {mean} \n Var = {var}")
