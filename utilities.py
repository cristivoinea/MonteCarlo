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
