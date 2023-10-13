import argparse
import numpy as np
from MonteCarloSWAPTorus import RunPSWAP

parser = argparse.ArgumentParser(
    description="Calculates the entanglement entropy of the v=1 IQHE state using Monte Carlo.")
parser.add_argument("-Ne", action="store", required=True,
                    help="number of particles")
parser.add_argument("-Ns", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("-M", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("--step-start", action="store", required=True,
                    help="step size value at r_start")
parser.add_argument("--r-start", action="store", required=True,
                    help="aspect ratio start value")
parser.add_argument("--r-end", action="store", required=True,
                    help="aspect ratio end value")
parser.add_argument("--nbr-r", action="store", required=True,
                    help="number of aspect ratio values")

parser.add_argument("--y-boundary", action="store", default=0.5,
                    help="y-location between bipartition")

args = vars(parser.parse_args())

Ne = np.uint8(args["Ne"])
Ns = np.uint8(args["Ns"])
M = np.uint32(args["M"])
step_start = np.float64(args["step_start"])

r_start = np.float64(args["r_start"])
r_end = np.float64(args["r_end"])
nbr_r = np.uint8(args["nbr_r"])
ts = np.complex128(1j*np.linspace(r_start, r_end, nbr_r, endpoint=True))
boundary = np.float64(args["y_boundary"])

step_const = step_start/np.sqrt(ts[0])

for t in ts:
    step_size = step_const*np.sqrt(t)
    result, acceptance = RunPSWAP(N=Ne, Ns=Ns, t=t, M=M, step_size=step_size,
                                  boundary_dimensionless=boundary)
    final = np.vstack((np.array([acceptance, 0]), result))
    np.save(
        f"p_swap_Ne_{Ne}_Ns_{Ns}_t_{np.imag(t):.2f}_step_{step_size:.3f}.npy", final)
