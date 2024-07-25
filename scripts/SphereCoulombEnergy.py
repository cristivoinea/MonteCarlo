import sys
import os

module_path = os.path.abspath(os.path.join(".."))  # nopep8
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
    of the 2nd Renyi entanglement entropy."""
)
parser.add_argument("-N", action="store", required=True, help="number of particles")
parser.add_argument("-S", action="store", required=True, help="number of flux quanta")
parser.add_argument(
    "--nbr-iter", action="store", required=True, help="number of Monte Carlo iterations"
)
parser.add_argument(
    "--nbr-nonthermal",
    action="store",
    default=-1,
    help="number of non-thermal iterations",
)
parser.add_argument(
    "--step",
    action="store",
    required=True,
    help="step size value at square aspect ratio",
)
parser.add_argument(
    "--acc-ratio",
    action="store",
    default=0,
    help="loads a previous run with given acceptance",
)
parser.add_argument("--state", action="store", required=True, help="type of state")
parser.add_argument(
    "--JK-coeffs", action="store", default="0", help="JK projection coefficients"
)
parser.add_argument(
    "--no-vortex", action="store_true", help="use the mean-field CS wavefunction"
)
parser.add_argument(
    "--hardcore", default=0, help="enforce hardcore radius of particles"
)
parser.add_argument(
    "--save-all-config",
    action="store_true",
    help="save all sampled system configurations",
)


args = vars(parser.parse_args())

N = np.int64(args["N"])
S = np.int64(args["S"])
nbr_iter = np.int64(args["nbr_iter"])
nbr_nonthermal = np.int64(args["nbr_nonthermal"])
if nbr_nonthermal == -1:
    if nbr_iter > 1e6:
        nbr_nonthermal = 1e5
    else:
        nbr_nonthermal = nbr_iter // 10

step = np.float64(args["step"])

acceptance_ratio = np.float64(args["acc_ratio"])
state = str(args["state"])
if state == "cfl":
    JK_coeffs = str(args["JK_coeffs"])

save_all_config = np.bool_(args["save_all_config"])
no_vortex = np.bool_(args["no_vortex"])
hardcore_radius = np.float64(args["hardcore"])

start_time = datetime.now()
if state == "free_fermions":
    fqh = MonteCarloSphereFreeFermions(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                       step_size=step, nbr_copies=1, 
                                       acceptance_ratio=acceptance_ratio,
                                       save_all_config=save_all_config)  # fmt: skip
elif state == "cfl":
    fqh = MonteCarloSphereCFL(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                              step_size=step, nbr_copies=1, JK_coeffs=JK_coeffs, 
                              no_vortex=no_vortex, hardcore_radius=hardcore_radius,
                              acceptance_ratio=acceptance_ratio, save_all_config=save_all_config)  # fmt: skip
elif state == "laughlin":
    fqh = MonteCarloSphereLaughlin(N=N, S=S, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                   step_size=step, nbr_copies=1, acceptance_ratio=acceptance_ratio,
                                   save_all_config=save_all_config)  # fmt: skip

fqh.RunCoulombEnergy()

end_time = datetime.now()
if "day" in str(end_time - start_time):
    print(f"Total time = {str(end_time - start_time)[:18]} s")
else:
    print(f"Total time = {str(end_time - start_time)[:10]} s")
print(f"Time / 5% = {str((end_time - start_time)/20)[:10]} s")
