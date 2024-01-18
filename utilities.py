import numpy as np
from numba import njit
from scipy.special import hyp0f1, gamma
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


def GetFermiSea(kF: np.float64):
    kx = ky = 0
    kx_list = [0]
    ky_list = [0]
    for kx in range(1, int(np.floor(kF))+1):
        kx_list.append(kx)
        kx_list.append(-kx)
        ky_list.append(0)
        ky_list.append(0)

        kx_list.append(0)
        kx_list.append(0)
        ky_list.append(kx)
        ky_list.append(-kx)

    for kx in range(1, int(np.floor(kF))+1):
        for ky in range(1, int(np.floor(kF))+1):
            if kx**2 + ky**2 <= kF**2:
                kx_list.append(kx)
                ky_list.append(ky)

                kx_list.append(kx)
                ky_list.append(-ky)

                kx_list.append(-kx)
                ky_list.append(ky)

                kx_list.append(-kx)
                ky_list.append(-ky)

    return np.array(kx_list), np.array(ky_list)


def GetEntropyFreeFermionsED(kF, m, region_geometry, linear_sizes):
    Kxs, Kys = GetFermiSea(kF)
    Ne = len(Kxs)
    Ns = m*Ne
    t = 1j

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)

    overlaps = np.zeros((Ne, Ne), dtype=np.float64)
    S = np.zeros(linear_sizes.size)
    for k in range(linear_sizes.size):
        for i in range(Ne):
            overlaps[i, i] = np.pi*linear_sizes[k]*linear_sizes[k]
            for j in range(i+1, Ne):
                if region_geometry == 'square':
                    overlaps[i, j] = np.sinc(
                        linear_sizes[k]*(Kxs[i]-Kxs[j]))*np.sinc(linear_sizes[k]*(Kys[i]-Kys[j]))*linear_sizes[k]*linear_sizes[k]
                elif region_geometry == 'circle':
                    overlaps[i, j] = np.pi*linear_sizes[k]*linear_sizes[k]*hyp0f1(
                        2, -((Kxs[i]-Kxs[j])**2 + (Kys[i]-Kys[j])**2)*(linear_sizes[k]*np.pi)**2)/gamma(2)
                overlaps[j, i] = overlaps[i, j]
        e = np.linalg.eigvalsh(overlaps)
        S[k] = -np.sum(np.log(e**2 + (1-e)**2))

    x = linear_sizes*np.sqrt(2*np.pi/Ns)*kF*Lx

    return x, S


def GetEntropyLaughlin(Ne, Ns, M, M0, t, step, region_geometry):
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


def LoadEntropy(Ne, Ns, M, M0, t, step_size, region_geometry, state):
    kf = {12: 2.5, 21: 5, 32: 8.5, 37: 10, 69: 20}
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    file = f"{state}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_step_{step_size}_{region_geometry}s.dat"

    if not exists(file):
        boundaries = np.arange(0.0250, 0.3001, 0.0125)
        data = np.zeros((boundaries.size, 7), dtype=np.float64)
        data[:, 0] = boundaries
        terms = ['p', 'mod', 'sign']
        for j in range(3):
            for i in range(boundaries.size):
                result = np.loadtxt(
                    f"../results/{state}/n_{Ne}/{terms[j]}/{state}_{terms[j]}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}_step_{step_size:.3f}.dat")
                if j == 2:
                    data[i, 1+2*j:3+2*j] = result[0, :]
                else:
                    data[i, 1+2*j:3+2*j] = result

        np.savetxt(file, data)

    data = np.loadtxt(file)

    entropy = np.zeros((data.shape[0], 8))
    x = np.sqrt(data[:, 0]/np.pi)*np.sqrt(2*kf[Ne]*np.pi/(Ns))*Lx

    entropy[:, 0] = -np.log(data[:, 1])
    entropy[:, 1] = np.sqrt((data[:, 2])/(data[:, 1])**2)/np.sqrt(M-M0)
    entropy[:, 2] = -np.log(data[:, 3])
    entropy[:, 3] = np.sqrt((data[:, 4])/(data[:, 3])**2)/np.sqrt(M-M0)
    entropy[:, 4] = -np.log(data[:, 5])
    entropy[:, 5] = np.sqrt((data[:, 6])/(data[:, 5])**2)/np.sqrt(M-M0)

    entropy[:, 6] = entropy[:, 0] + entropy[:, 2] + entropy[:, 4]
    entropy[:, 7] = np.sqrt(
        entropy[:, 1]**2 + entropy[:, 3]**2 + entropy[:, 5]**2)

    return x, entropy


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
