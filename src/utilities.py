import numpy as np
from numba import njit
from os.path import exists
from scipy.special import gamma, hyp2f1, factorial, factorial2


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


def Stats(data: np.array) -> tuple[np.float64, np.float64]:
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


def LoadDisorderOperator(Ne, Ns, M, M0, t, region_geometry, state):
    kf = {12: 2.5, 21: 5, 32: 8.5, 37: 10, 69: 20}
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    file = f"disorder_{state}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}s.dat"

    if not exists(file):
        boundaries = np.arange(0.08, 0.35, 0.01)
        data = np.zeros((boundaries.size, 3), dtype=np.float64)
        data[:, 0] = boundaries
        for i in range(boundaries.size):
            result = np.loadtxt(
                f"../results/disorder/{state}/n_{Ne}/disorder_{state}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}.dat")
            data[i, 1:3] = result

        np.savetxt(file, data)

    data = np.loadtxt(file)

    entropy = np.zeros((data.shape[0], 2))
    x = data[:, 0]*np.sqrt(2*kf[Ne]*np.pi/(Ns))*Lx

    entropy[:, 0] = -2*np.log(data[:, 1])
    entropy[:, 1] = np.sqrt((data[:, 2])/(data[:, 1])**2)/np.sqrt(M-M0)

    return x, entropy


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


def LoadEntropy(Ne, Ns, geometry, region_geometry, state, boundaries, t=1j):
    if geometry == "torus":
        kf = {12: 2.5, 21: 5, 32: 8.5, 37: 10, 69: 20}
        Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
        file = f"../data/{state}_{geometry}_entropy_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}s.dat"
    elif geometry == "sphere":
        file = f"../data/{state}_{geometry}_entropy_n_{Ne}_s_{Ns}.dat"

    if not exists(file):
        data = np.zeros((boundaries.size, 7), dtype=np.float64)
        data[:, 0] = boundaries
        terms = ['p', 'mod', 'sign']
        for j in range(3):
            for i in range(boundaries.size):
                if geometry == "torus":
                    result = np.loadtxt(
                        f"../../results/entropy/{state}/n_{Ne}/{terms[j]}/{terms[j]}_{state}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}.dat")
                elif geometry == "sphere":
                    result = np.loadtxt(
                        f"../../results/{geometry}/entropy/{state}/n_{Ne}/{terms[j]}/{state}_{geometry}_{terms[j]}_n_{Ne}_s_{Ns}_theta_0.000000_{boundaries[i]:.6f}.dat")
                # if j == 2:
                #    data[i, 1+2*j:3+2*j] = result[0, :]
                # else:
                data[i, 1+2*j:3+2*j] = result

        np.savetxt(file, data)

    data = np.loadtxt(file)

    entropy = np.zeros((data.shape[0], 8))
    if geometry == "torus":
        x = np.sqrt(data[:, 0]/np.pi)*np.sqrt(2*kf[Ne]*np.pi/(Ns))*Lx
    elif geometry == "sphere":
        x = np.sin(boundaries*np.pi/180)*np.sqrt(Ne-1)

    entropy[:, 0] = -np.log(data[:, 1])
    # np.sqrt((data[:, 2])/(data[:, 1])**2)/np.sqrt(M-M0)
    entropy[:, 1] = data[:, 2]/data[:, 1]
    entropy[:, 2] = -np.log(data[:, 3])
    # np.sqrt((data[:, 4])/(data[:, 3])**2)#/np.sqrt(M-M0)
    entropy[:, 3] = data[:, 4]/data[:, 3]
    entropy[:, 4] = -np.log(data[:, 5])
    # np.sqrt((data[:, 6])/(data[:, 5])**2)#/np.sqrt(M-M0)
    entropy[:, 5] = data[:, 6]/data[:, 5]

    entropy[:, 6] = entropy[:, 0] + entropy[:, 2] + entropy[:, 4]
    entropy[:, 7] = np.sqrt(
        entropy[:, 1]**2 + entropy[:, 3]**2 + entropy[:, 5]**2)

    return x, entropy


def LoadDisorderOperator(Ne, Ns, M, M0, t, region_geometry, state, boundaries):
    kf = {12: 2.5, 21: 5, 32: 8.5, 37: 10, 69: 20}
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    file = f"disorder_{state}_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_{region_geometry}s.dat"

    if not exists(file):
        data = np.zeros((boundaries.size, 3), dtype=np.float64)
        data[:, 0] = boundaries
        for i in range(boundaries.size):
            result = np.loadtxt(
                f"../results/disorder/{state}/n_{Ne}/disorder_{state}_Ne_{Ne}_Ns_{Ns}_t_1.00_circle_{boundaries[i]:.4f}.dat")
            data[i, 1:3] = result

        np.savetxt(file, data)

    data = np.loadtxt(file)

    entropy = np.zeros((data.shape[0], 2))
    x = data[:, 0]*np.sqrt(2*kf[Ne]*np.pi/(Ns))*Lx

    entropy[:, 0] = -2*np.log(data[:, 1])
    entropy[:, 1] = np.sqrt((data[:, 2])/(data[:, 1])**2)/np.sqrt(M-M0)

    return x, entropy


def LoadParticleFluctuations(Ne, Ns, geometry, state, boundaries, region_geometry='circle',
                             linear_size=True, t=1j, cf=False):
    kf = {12: 2.5, 21: 5, 32: 8.5, 37: 10, 69: 20}
    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    file = f"{state}_{geometry}_fluct_n_{Ne}_s_{Ns}.dat"

    if cf:
        cf_str = "cf_"
    else:
        cf_str = ""

    if not exists(file):
        data = np.zeros((boundaries.size, 3), dtype=np.float64)
        data[:, 0] = boundaries
        for i in range(boundaries.size):
            result = np.loadtxt(
                f"../../results/{geometry}/{cf_str}fluctuations/{state}/n_{Ne}/{state}_{geometry}_fluct_n_{Ne}_s_{Ns}_theta_0.000000_{boundaries[i]:.6f}.dat")
            data[i, 1:3] = result
        np.savetxt(file, data)

    data = np.loadtxt(file)

    fluctuations = np.zeros((data.shape[0], 2))
    if geometry == "sphere":
        if linear_size:
            x = np.sin(data[:, 0]*np.pi/180)*np.sqrt(Ne-1)
    elif geometry == "torus":
        if linear_size:
            x = data[:, 0] * np.sqrt(2*kf[Ne]*np.pi/(Ns))*Lx

    fluctuations[:, 0] = data[:, 1]
    fluctuations[:, 1] = data[:, 2]

    return x, fluctuations


def LegendreOffDiagIntegral(x, legendre, l, k, m):
    if np.abs(m) > k or np.abs(m) > l:
        return 0
    else:
        return ((x*legendre[l][l+m]*legendre[k][k+m]*(k-l) - (k-m+1)*legendre[l][l+m]*legendre[k+1][k+1+m]
                 + (l-m+1)*legendre[k][k+m]*legendre[l+1][l+1+m]) / (k*(k+1) - l*(l+1)))


def LegendreDiagIntegral(x, l_max, legendre_values):
    diag = {}
    for l in range(l_max+1):
        diag[l] = np.zeros(2*l+1, dtype=np.float64)
    diag[0][0] = (1-x)
    diag[1][0] = (2 - 3*x + x**3)/12
    diag[1][1] = (1-x**3)/3
    diag[1][2] = (2 - 3*x + x**3)/3
    for l in range(2, l_max+1):
        diag[l][2*l] = ((factorial2(2*l-1)**2)*(np.sqrt(np.pi)*gamma(1+l) /
                        (2*gamma(3/2+l)) - x*hyp2f1(1/2, -l, 3/2, x**2)))
        diag[l][0] = diag[l][2*l] / (factorial(2*l)**2)
        for m in range(-l+1, l):
            diag[l][l+m] = (((2*l-1) / ((2*l+1)*(l-m))) * ((l+m)*diag[l-1][l-1+m] + (l+1-m) *
                            LegendreOffDiagIntegral(x, legendre_values, l+1, l-1, m)) -
                            ((l+m-1)/(l-m))*LegendreOffDiagIntegral(x, legendre_values, l, l-2, m))

    return diag
