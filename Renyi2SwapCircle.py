from MonteCarloTorusSWAP import RunPSwapCFL, RunModSwapCFL, RunSignSwapCFL, \
    RunPSwapFreeFermions, RunModSwapFreeFermions, RunSignSwapFreeFermions
import numpy as np
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="""Calculates the P,Mod,Sign terms in the SWAP representation of
    2nd Renyi entanglement entropy, using
    a region A of circular geometry.""")
parser.add_argument("-Ne", action="store", required=True,
                    help="number of particles")
parser.add_argument("-Ns", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("--nbr-iter", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("--nbr-nonthermal", action="store", default=-1,
                    help="number of non-thermal iterations")
parser.add_argument("--step", action="store", required=True,
                    help="step size value at square aspect ratio")
parser.add_argument("--A-start", action="store", required=True,
                    help="initial coverage of subregion A (percentage of total area)")
parser.add_argument("--A-end", action="store", default=-1,
                    help="final coverage of subregion A (percentage of total area)")
parser.add_argument("--nbr-A", action="store", default=1,
                    help="number of points")
parser.add_argument("--start-acceptance", action="store", default=-1,
                    help="loads a previous run with given acceptance")
parser.add_argument("--swap-term", action="store", required=True,
                    help="term in the SWAP decomposition")
parser.add_argument("--state", action="store", required=True,
                    help="type of state")

args = vars(parser.parse_args())

Ne = np.uint8(args["Ne"])
Ns = np.uint8(args["Ns"])
M = np.uint32(args["nbr_iter"])
M0 = np.int32(args["nbr_nonthermal"])
if M0 == -1:
    if M > 1e6:
        M0 = 1e5
    else:
        M0 = M//10

step = np.float64(args["step"])
A_start = np.float64(args["A_start"])
A_end = np.float64(args["A_end"])
if A_end == -1:
    A_end = A_start
nbr_A = np.uint8(args["nbr_A"])
A_sizes = np.linspace(A_start, A_end, nbr_A, endpoint=True)

start_acceptance = np.float64(args["start_acceptance"])
swap_term = str(args["swap_term"])
state = str(args["state"])
t = 1j

for region_size in A_sizes:
    start_time = datetime.now()

    if swap_term == 'p':
        if state == 'cfl':
            RunPSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                        region_geometry='circle', region_size=region_size,
                        start_acceptance=start_acceptance)
        elif state == 'free-fermions':
            RunPSwapFreeFermions(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                                 region_geometry='circle', region_size=region_size,
                                 start_acceptance=start_acceptance)
    elif swap_term == 'mod':
        if state == 'cfl':
            RunModSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                          region_geometry='circle', region_size=region_size,
                          start_acceptance=start_acceptance)
        elif state == 'free-fermions':
            RunModSwapFreeFermions(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                                   region_geometry='circle', region_size=region_size,
                                   start_acceptance=start_acceptance)
    elif swap_term == 'sign':
        if state == 'cfl':
            RunSignSwapCFL(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                           region_geometry='circle', region_size=region_size,
                           start_acceptance=start_acceptance)
        elif state == 'free-fermions':
            RunSignSwapFreeFermions(Ne=Ne, Ns=Ns, t=t, M=M, M0=M0, step_size=step,
                                    region_geometry='circle', region_size=region_size,
                                    start_acceptance=start_acceptance)

    end_time = datetime.now()
    print(f"Total time = {str(end_time - start_time)[:10]} s")
    print(f"Time / 5% = {str((end_time - start_time)/20)[:10]} s")
