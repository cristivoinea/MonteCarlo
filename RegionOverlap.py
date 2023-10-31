import numpy as np
from scipy.integrate import dblquad


def OverlapMatrix(N, f, x_min, x_max, y_min, y_max):
    print(f"Circle radius {radii[i]}")
    M = np.zeros((N, N), dtype=np.complex128)
    real_errors = np.zeros((N, N), dtype=np.float64)
    imag_errors = np.zeros((N, N), dtype=np.float64)

    for n in range(N):
        print(f"Calculating diagonal overlap of state {n}..")
        real, real_err = dblquad(InnerProductReal, Lx/2-radii[i], Lx/2+radii[i],
                                 lambda x: (
            Ly/2 - np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
            lambda x: (
            Ly/2 + np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
            args=(n, n, N, t))
        M[n, n] = real
        real_errors[n, n] = real_err

    for n in range(N):
        for m in range(n+1, N):
            print(f"Calculating overlap of states {n,m}..")
            real, real_err = dblquad(InnerProductReal, x_min, x_max, y_min, y_max,
                                     args=args)
            imag, imag_err = dblquad(InnerProductImag, x_min, x_max, y_min, y_max,
                                     args=args)
            M[n, m] = (real+1j*imag)
            real_errors[n, m] = real_err
            imag_errors[n, m] = imag_err

            M[m, n] = (real-1j*imag)
            real_errors[m, n] = real_err
            imag_errors[m, n] = imag_err

    np.save(f"{state}_overlaps_N_{N}_circle_{radius:.3f}.npy", M)
    np.save(
        f"{state}_overlaps_real_err_N_{N}_circle_{radii_unscaled[i]:.3f}.npy", real_errors)
    np.save(
        f"{state}_overlaps_imag_err_N_{N}_circle_{radii_unscaled[i]:.3f}.npy", imag_errors)
