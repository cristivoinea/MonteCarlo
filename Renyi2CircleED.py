import argparse
import numpy as np
from LaughlinWavefnSWAP import LLLSymmetricGauge
from scipy.integrate import dblquad

parser = argparse.ArgumentParser(
    description="""Calculates the 2nd Renyi entanglement entropy in Slater determinant
    states for a region A of circular geometry.""")
parser.add_argument("-N", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("--r-start", action="store", required=True,
                    help="circle radius start value (units of Ly)")
parser.add_argument("--r-end", action="store", default=-1,
                    help="circle radius end value (units of Ly)")
parser.add_argument("--nbr-r", action="store", default=1,
                    help="number of circle radius values")
parser.add_argument("-S", action="store", required=True,
                    help="the wavefunction of the system ('laughlin','free_fermions','cfl')")

args = vars(parser.parse_args())

N = np.uint8(args["N"])

r_start = np.float64(args["r_start"])
r_end = np.float64(args["r_end"])
if r_end == -1:
    r_end = r_start
nbr_r = np.uint8(args["nbr_r"])

state = str(args["S"])
t = 1j
Lx = np.sqrt(2*np.pi*N/np.imag(t))
Ly = Lx*np.imag(t)

radii_unscaled = np.linspace(r_start, r_end, nbr_r, endpoint=True)
radii = radii_unscaled*Ly


def InnerProductReal(y, x, n, m, Ns, t):
    return np.real(LLLSymmetricGauge(x+1j*y, Ns, t, m)*np.conj(LLLSymmetricGauge(x+1j*y, Ns, t, n)))


def InnerProductImag(y, x, n, m, Ns, t):
    return np.imag(LLLSymmetricGauge(x+1j*y, Ns, t, m)*np.conj(LLLSymmetricGauge(x+1j*y, Ns, t, n)))


for i in range(radii.size):
    print(f"Circle radius {radii[i]}")
    M = np.zeros((N, N), dtype=np.complex128)
    real_errors = np.zeros((N, N), dtype=np.complex128)
    imag_errors = np.zeros((N, N), dtype=np.complex128)

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
            real, real_err = dblquad(InnerProductReal, Lx/2-radii[i], Lx/2+radii[i],
                                     lambda x: (
                                     Ly/2 - np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
                                     lambda x: (
                                     Ly/2 + np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
                                     args=(n, m, N, t))
            imag, imag_err = dblquad(InnerProductImag, Lx/2-radii[i], Lx/2+radii[i],
                                     lambda x: (
                                     Ly/2 - np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
                                     lambda x: (
                                     Ly/2 + np.sqrt(radii[i]**2 - (x-Lx/2)**2)),
                                     args=(n, m, N, t))
            M[n, m] = (real+1j*imag)
            real_errors[n, m] = real_err
            imag_errors[n, m] = imag_err
    np.save(f"{state}_overlaps_N_{N}_circle_{radii_unscaled[i]}.npy", M)
    np.save(
        f"{state}_overlaps_real_err_N_{N}_circle_{radii_unscaled[i]}.npy", real_err)
    np.save(
        f"{state}_overlaps_imag_err_N_{N}_circle_{radii_unscaled[i]}.npy", imag_err)
