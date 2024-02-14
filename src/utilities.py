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


@njit  # (parallel=True)
def ThetaFunction(z: np.complex128, t: np.complex128, a: np.float64,
                  b: np.float64, n_max: np.uint32 = 75
                  ) -> np.complex128:
    index_a = np.arange(-n_max+a, n_max+a, 1)
    terms = np.exp(1j*np.pi*index_a*(t*(index_a) + 2*(z + b)))
    return np.sum(terms)


def Stats(data: np.array) -> (np.float64, np.float64):
    data_copy = data
    mean = np.sum(data_copy)/data_copy.size
    var = np.var(data_copy, ddof=1)

    return mean, var


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
                if state != 'cfl0':
                    single_file = f"../results/{state}/n_{Ne}/{terms[j]}/{state}_{terms[j]}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}_step_{step_size:.3f}.dat"
                else:
                    single_file = f"../results/cfl{int(Ns//Ne)}_unproj/n_{Ne}/{terms[j]}/{state}_{terms[j]}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}_step_{step_size:.3f}.dat"

                if exists(single_file):
                    result = np.loadtxt(single_file)
                    if j == 2:
                        data[i, 1+2*j:3+2*j] = result[0, :]
                    else:
                        data[i, 1+2*j:3+2*j] = result
                else:
                    data[i, 1+2*j:3+2*j] = np.array([0, 0])

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
