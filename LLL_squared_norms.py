import argparse
import os
import numpy as np
from FQHEWaveFunctions import LLLSymmetricGauge
from scipy.integrate import dblquad

parser = argparse.ArgumentParser(
    description="Returns the squared norms of LLL wavefunctions on the torus.")
parser.add_argument("-Ns", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("--r-start", action="store", required=True,
                    help="aspect ratio start value")
parser.add_argument("--r-end", action="store", required=True,
                    help="aspect ratio end value")
parser.add_argument("--nbr-r", action="store", required=True,
                    help="number of aspect ratio values")

args = vars(parser.parse_args())


def inner_product_real(y, x, n, m, Ns, t):
    return np.real(LLLSymmetricGauge(x+1j*y, Ns, t, m)*np.conj(LLLSymmetricGauge(x+1j*y, Ns, t, n)))


def inner_product_imag(y, x, n, m, Ns, t):
    return np.imag(LLLSymmetricGauge(x+1j*y, Ns, t, m)*np.conj(LLLSymmetricGauge(x+1j*y, Ns, t, n)))


Ns = int(args["Ns"])
r_start = float(args["r-start"])
r_end = float(args["r-end"])
nbr_r = float(args["nbr-r"])

ts = 1j*np.linspace(r_start, r_end, nbr_r)
Lxs = np.sqrt(2*np.pi*Ns/np.imag(ts))
Lys = Lxs*np.imag(ts)

# get normalisation constants
for k in range(ts.size):
    print(
        f"Torus dimensions at t = {ts[k]} \nLx = ", Lxs[k], "\nLy = ", Lys[k])
    overlaps = np.zeros(Ns, dtype=np.complex128)
    errors = np.zeros(Ns, dtype=np.complex128)
    for i in range(Ns):
        print(f"Calculating norm of state {i}..")
        re, re_err = dblquad(
            inner_product_real, -Lxs[k]/2, Lxs[k]/2, -Lys[k]/2, Lys[k]/2, args=(i, i, Ns, ts[k]))
        overlaps[i] = re
        errors[i] = re_err
    np.savetxt(f"sq_norm_N_{Ns}_r_{np.imag(ts[k]):.3f}.dat", np.vstack(
        (overlaps, errors)).T)
