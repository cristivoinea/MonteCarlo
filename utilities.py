import numpy as np
from numba import njit
from os.path import exists

fermi_sea_ky = {}
fermi_sea_kx = {}
fermi_sea_kx[32] = np.array([0, 1, 1, 0, 2, 2, 1, 0, -1, -1, 0, 1,
                             2, -1, -1, 2, 3, 3, 1, 0, -2, -2, 0, 1,
                             3, 2, -1, -2, -2, -1, 2, 3])
fermi_sea_ky[32] = np.array([0, 0, 1, 1, 0, 1, 2, 2, 1, 0, -1, -1,
                             2, 2, -1, -1, 0, 1, 3, 3, 1, 0, -2, -2,
                             2, 3, 3, 2, -1, -2, -2, -1])
fermi_sea_kx[12] = np.array([0, 1, 0, -1, 0, 1, -1, 1, 2, 2, 1, 0])
fermi_sea_ky[12] = np.array([0, 0, 1, 0, -1, 1, 1, -1, 0, 1, 2, 2])
fermi_sea_kx[69] = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1, 2, 0, -2, 0,
                             2, 1, -1, -2, -2, -1, 1, 2, 2, -2, -2, 2,
                             3, 0, -3, 0, 3, 1, -1, -3, -3, -1, 1, 3,
                             3, 2, -2, -3, -3, -2, 2, 3, 4, 0, -4, 0,
                             4, 1, -1, -4, -4, -1, 1, 4, 3, -3, -3, 3,
                             4, 2, -2, -4, -4, -2, 2, 4])
fermi_sea_ky[69] = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2,
                             1, 2, 2, 1, -1, -2, -2, -1, 2, 2, -2, -2,
                             0, 3, 0, -3, 1, 3, 3, 1, -1, -3, -3, -1,
                             2, 3, 3, 2, -2, -3, -3, -2, 0, 4, 0, -4,
                             1, 4, 4, 1, -1, -4, -4, -1, 3, 3, -3, -3,
                             2, 4, 4, 2, -2, -4, -4, -2])
fermi_sea_kx[37] = fermi_sea_kx[69][:37]
fermi_sea_ky[37] = fermi_sea_ky[69][:37]
fermi_sea_kx[21] = fermi_sea_kx[69][:21]
fermi_sea_ky[21] = fermi_sea_ky[69][:21]


def Stats(data: np.array) -> (np.float64, np.float64):
    data_copy = data
    mean = np.sum(data_copy)/data_copy.size
    var = np.var(data_copy, ddof=1)

    return mean, var


def FullStats(data: np.array
              ) -> (np.array, np.array):

    n = 500
    mean_vector = np.zeros(data.size-n, dtype=np.float64)
    # std_vector = np.zeros(data.size-n, dtype=np.float64)
    for i in range(mean_vector.size):
        mean_vector[i] = np.mean(data[:i+n+1])
        # std_vector[i] = np.std(data[:i+n+1], ddof=1)

    return mean_vector  # , std_vector


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


def SaveConfig(state: str, quantity: str, Ne: np.uint8, Ns: np.uint16,
               Lx: np.float64, Ly: np.float64, t: np.complex128,
               step_size: np.float64,
               region_geometry: str, region_size: np.float64,
               result: np.array, R: np.array, swap_order: np.array):
    if quantity == 'sign':
        np.save(f"{state}_{quantity}_results_real_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.npy",
                np.real(result))
        np.save(f"{state}_{quantity}_results_imag_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.npy",
                np.imag(result))

    else:
        np.save(f"{state}_{quantity}_results_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.npy",
                result)

    if quantity == 'sign' or quantity == 'mod':
        np.save(f"{state}_{quantity}_order_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.npy",
                swap_order)

    np.save(f"{state}_{quantity}_coords_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.npy",
            R)


def SaveResults(state: str, quantity: str, Ne: np.uint8, Ns: np.uint16,
                Lx: np.float64, Ly: np.float64,
                M0: np.uint32, t: np.complex128, step_size: np.float64,
                result: np.array, save_result: np.bool_,
                region_geometry: str = 'none', region_size: np.float64 = 0):

    if save_result:
        if quantity == 'sign':
            np.savetxt(f"{state}_{quantity}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.dat",
                       np.vstack((Stats(np.real(result[int(M0):])),
                                  Stats(np.imag(result[int(M0):]))
                                  ))
                       )
        else:
            np.savetxt(f"{state}_{quantity}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size/Lx:.3f}.dat",
                       Stats(result[int(M0):]))

    else:
        if quantity == 'sign':
            mean, var = Stats(np.real(result[int(M0):]))
            print(f"\nMean (real) = {mean} \n Var (real) = {var}")
            mean, var = Stats(np.imag(result[int(M0):]))
            print(f"\nMean (imag) = {mean} \n Var (imag) = {var}")
        else:
            mean, var = Stats(result[int(M0):])
            print(f"\nMean = {mean} \n Var = {var}")


def LoadRun(Ne: np.uint8, Ns: np.uint16, t: np.complex128, M: np.uint64,
            region_size: np.float64, region_geometry: str,
            step_size: np.float64, state: str, swap_term: str):
    error = None
    coords = 0
    order = 0
    results = 0

    file_coords = f"./{state}_{swap_term}_coords_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size:.3f}.npy"
    if exists(file_coords):
        coords = np.load(file_coords)
    else:
        error = f"{file_coords} missing!\n"

    if swap_term == 'mod' or swap_term == 'sign':
        file_order = f"./{state}_{swap_term}_order_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size:.3f}.npy"
        if exists(file_order):
            order = np.load(file_order)
        else:
            error += f"{file_order} missing!\n"

    if swap_term == 'sign':
        file_results_real = f"./{state}_{swap_term}_results_real_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size:.3f}.npy"
        file_results_imag = f"./{state}_{swap_term}_results_imag_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size:.3f}.npy"
        if exists(file_results_real) and exists(file_results_real):
            start_results = np.load(file_results_real) + \
                np.load(file_results_imag)
            prev_iter = start_results.size

            results = np.zeros((prev_iter+M), dtype=np.complex128)
            results[:prev_iter] = start_results
        else:
            error += "Results file not found!\n"

    else:
        file_results = f"./{state}_{swap_term}_results_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}_{region_size:.4f}_step_{step_size:.3f}.npy"
        if exists(file_results):
            start_results = np.load(file_results)
            prev_iter = start_results.size

            results = np.zeros((prev_iter+M), dtype=np.float64)
            results[:prev_iter] = start_results
        else:
            error += "Results file not found!\n"

    return coords, order, results, prev_iter, error
