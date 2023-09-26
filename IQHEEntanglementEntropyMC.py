import argparse
import os
import numpy as np
from MonteCarloSWAPTorus import RunPSWAP, RunModSWAP, RunSignSWAP

parser = argparse.ArgumentParser(
    description="Calculates the entanglement entropy of the v=1 IQHE state using Monte Carlo.")
parser.add_argument("-N", action="store", required=True,
                    help="number of particles")
parser.add_argument("-M", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("--step-size", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("--r-start", action="store", required=True,
                    help="aspect ratio start value")
parser.add_argument("--r-end", action="store", required=True,
                    help="aspect ratio end value")
parser.add_argument("--nbr-r", action="store", required=True,
                    help="number of aspect ratio values")

parser.add_argument("--y-boundary", action="store", default=0.5,
                    help="y-location between bipartition")

args = vars(parser.parse_args())

N = np.uint8(args["N"])
M = np.uint32(args["M"])
step_size = np.float64(args["step_size"])

r_start = np.float64(args["r_start"])
r_end = np.float64(args["r_end"])
nbr_r = np.uint8(args["nbr_r"])
ts = np.complex128(1j*np.linspace(r_start, r_end, nbr_r, endpoint=True))
boundary = np.float64(args["y_boundary"])

for t in ts:
    result = RunPSWAP(N=N, Ns=N, t=t, M=M, step_size=step_size,
                      boundary_dimensionless=boundary)
    np.savetxt(f"p_N_{N}_r_{np.imag(t):.2f}.dat", result)

    result = RunModSWAP(N=N, Ns=N, t=t, M=M, step_size=step_size,
                        boundary_dimensionless=boundary)
    np.savetxt(f"mod_N_{N}_r_{np.imag(t):.2f}.dat", result)

    result = RunSignSWAP(N=N, Ns=N, t=t, M=M, step_size=step_size,
                         boundary_dimensionless=boundary)
    np.savetxt(f"sign_N_{N}_r_{np.imag(t):.2f}.dat", result)
