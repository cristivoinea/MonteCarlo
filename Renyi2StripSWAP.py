import argparse
import numpy as np
from MonteCarloTorusSWAP import RunPSWAP, RunModSWAP, RunSignSWAP

parser = argparse.ArgumentParser(
    description="""Calculates the P,Mod,Sign terms in the SWAP representation of
    2nd Renyi entanglement entropy in Laughlin states at 1/m filling, using
    a region A of strip geometry.""")
parser.add_argument("-Ne", action="store", required=True,
                    help="number of particles")
parser.add_argument("-Ns", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("-M", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("-M0", action="store", default=-1,
                    help="number of non-thermal iterations")
parser.add_argument("--step", action="store", required=True,
                    help="step size value at square aspect ratio")
parser.add_argument("--r-start", action="store", required=True,
                    help="aspect ratio start value")
parser.add_argument("--r-end", action="store", default=-1,
                    help="aspect ratio end value")
parser.add_argument("--nbr-r", action="store", default=1,
                    help="number of aspect ratio values")
parser.add_argument("--y-boundary", action="store", default=0.5,
                    help="y-location between bipartition")
parser.add_argument("--SWAP-term", action="store", required=True,
                    help="term in the SWAP decomposition")
parser.add_argument("-S", action="store", required=True,
                    help="the wavefunction of the system ('laughlin','free_fermions','cfl')")

args = vars(parser.parse_args())

Ne = np.uint8(args["Ne"])
Ns = np.uint8(args["Ns"])
M = np.uint32(args["M"])
M0 = np.int32(args["M0"])
if M0 == -1:
    if M > 1e6:
        M0 = 1e5
    else:
        M0 = M//10

step = np.float64(args["step"])
r_start = np.float64(args["r_start"])
r_end = np.float64(args["r_end"])
if r_end == -1:
    r_end = r_start
nbr_r = np.uint8(args["nbr_r"])
ts = np.complex128(1j*np.linspace(r_start, r_end, nbr_r, endpoint=True))
boundary = np.float64(args["y_boundary"])

SWAP_term = str(args["SWAP_term"])
state = str(args["S"])

for t in ts:
    step_size = step*np.sqrt(np.imag(t))

    if SWAP_term == 'p':
        RunPSWAP(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step_size,
                 region_geometry='strip', boundary=boundary,
                 state=state)
    if SWAP_term == 'mod':
        RunModSWAP(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step_size,
                   region_geometry='strip', boundary=boundary,
                   state=state)
    if SWAP_term == 'sign':
        RunSignSWAP(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step_size,
                    region_geometry='strip', boundary=boundary,
                    state=state)
