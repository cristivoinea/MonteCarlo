import sys
import os

module_path = os.path.abspath(os.path.join('..'))  # nopep8
if module_path not in sys.path:
    sys.path.append(module_path)

from src.MonteCarloSphereCFL import MonteCarloSphereCFL  # nopep8
from src.MonteCarloSphereLaughlin import MonteCarloSphereLaughlin  # nopep8
from src.MonteCarloSphereFreeFermions import MonteCarloSphereFreeFermions  # nopep8
import numpy as np
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="""Calculates the P,Mod,Sign terms in the SWAP decomposition 
    of the 2nd Renyi entanglement entropy.""")
parser.add_argument("-N", action="store", required=True,
                    help="number of particles")
parser.add_argument("-S", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("--nbr-iter", action="store", required=True,
                    help="number of Monte Carlo iterations")
parser.add_argument("--nbr-nonthermal", action="store", default=-1,
                    help="number of non-thermal iterations")
parser.add_argument("--step", action="store", required=True,
                    help="step size value at square aspect ratio")
parser.add_argument("--theta", action="store", nargs='+', default=[180],
                    help="theta limits of the region (degrees)")
parser.add_argument("--phi", action="store", nargs='+', default=[360],
                    help="phi limits of the region (degrees)")
parser.add_argument("--acc-ratio", action="store", default=0,
                    help="loads a previous run with given acceptance")
parser.add_argument("--state", action="store", required=True,
                    help="type of state")
parser.add_argument("--JK-coeffs", action="store", default='0',
                    help="JK translation coefficients for enforcing PBC")
parser.add_argument("--CF", action="store_true",
                    help="composite fermions")

args = vars(parser.parse_args())

N = np.uint8(args["N"])
S = np.uint8(args["S"])
nbr_iter = np.uint32(args["nbr_iter"])
nbr_nonthermal = np.int32(args["nbr_nonthermal"])
if nbr_nonthermal == -1:
    if nbr_iter > 1e6:
        nbr_nonthermal = 1e5
    else:
        nbr_nonthermal = nbr_iter//10

step = np.float64(args["step"])

theta = args["theta"]
if len(theta) == 1:
    theta = np.float64(theta[0])
else:
    theta = np.array(theta, dtype=np.float64)
phi = args["phi"]
if len(phi) == 1:
    phi = np.float64(phi[0])
else:
    phi = np.array(phi, dtype=np.float64)

acceptance_ratio = np.float64(args["acc_ratio"])
state = str(args["state"])
if state == 'cfl':
    JK_coeffs = str(args["JK_coeffs"])

CF = bool(args["CF"])

start_time = datetime.now()
if state == "free_fermions":
    fqh = MonteCarloSphereFreeFermions(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                       step_size=step, region_theta=theta, region_phi=phi,
                                       nbr_copies=1, acceptance_ratio=acceptance_ratio)
elif state == 'cfl':
    fqh = MonteCarloSphereCFL(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                              step_size=step, region_theta=theta, region_phi=phi,
                              nbr_copies=1, acceptance_ratio=acceptance_ratio)
elif state == 'laughlin':
    fqh = MonteCarloSphereLaughlin(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                   step_size=step, region_theta=theta, region_phi=phi,
                                   nbr_copies=1, acceptance_ratio=acceptance_ratio)

fqh.RunParticleFluctuations(cf=CF)


end_time = datetime.now()
print(f"Total time = {str(end_time - start_time)[:10]} s")
print(f"Time / 5% = {str((end_time - start_time)/20)[:10]} s")
