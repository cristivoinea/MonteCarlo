import argparse
import numpy as np
from MonteCarloTorusSWAP import RunPSwapCFL, RunModSwapCFL, RunSignSwapCFL

parser = argparse.ArgumentParser(
    description="""Calculates the P,Mod,Sign terms in the SWAP representation of
    2nd Renyi entanglement entropy in the 1/2 CFL, using
    a region A of circular geometry.""")
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
                    help="circle radius start value (units of Ly)")
parser.add_argument("--r-end", action="store", default=-1,
                    help="circle radius end value (units of Ly)")
parser.add_argument("--nbr-r", action="store", default=1,
                    help="number of circle radius values")
parser.add_argument("--swap-term", action="store", required=True,
                    help="term in the SWAP decomposition")

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
radii = np.linspace(r_start, r_end, nbr_r, endpoint=True)

swap_term = str(args["swap_term"])
t = 1j

for r in radii:
    if swap_term == 'p':
        RunPSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                    region_geometry='circle', boundary=r)
    if swap_term == 'mod':
        RunModSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                      region_geometry='circle', boundary=r)
    if swap_term == 'sign':
        RunSignSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                       region_geometry='circle', boundary=r)
