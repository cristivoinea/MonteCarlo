import sys
import os

module_path = os.path.abspath(os.path.join('..'))  # nopep8
if module_path not in sys.path:
    sys.path.append(module_path)

from src.MonteCarloTorusCFL import MonteCarloTorusCFL  # nopep8
from src.MonteCarloTorusFreeFermions import MonteCarloTorusFreeFermions  # nopep8
from src.utilities import ExtractSpFromFile
import numpy as np
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="""Calculates the subregion entanglement entropy, using
    a region A of circular geometry.""")
parser.add_argument("-N", action="store", required=True,
                    help="number of particles")
parser.add_argument("-S", action="store", required=True,
                    help="number of flux quanta")
parser.add_argument("-t", action="store", default=1,
                    help="aspect ratio")
parser.add_argument("--nbr-iter", action="store", default=10000,
                    help="number of Monte Carlo iterations")
parser.add_argument("--nbr-nonthermal", action="store", default=-1,
                    help="number of non-thermal iterations")
parser.add_argument("--step", action="store", default=0.1,
                    help="step size value at square aspect ratio")
parser.add_argument("--region-geometry", action="store", default='circle',
                    help="geometry of one of the bipartition regions")
parser.add_argument("--region-size", action="store", default=0.15,
                    help="linear size of the bipartition")
parser.add_argument("--acc-ratio", action="store", default=0,
                    help="loads a previous run with given acceptance")
parser.add_argument("--run", action="store", required=True,
                    help="type of Monte Carlo run")
parser.add_argument("--state", action="store", required=True,
                    help="type of state")
parser.add_argument("--JK-coeffs", action="store", default='0',
                    help="JK projection coefficients")
parser.add_argument("--save-all-config", action="store_true",
                    help="save all sampled system configurations")
parser.add_argument("--extract", action="store_true",
                    help="extract the SWAP probability from the existing data without running MC again")

args = vars(parser.parse_args())

N = np.int64(args["N"])
S = np.int64(args["S"])
t = np.float64(args["t"])
region_geometry = str(args["region_geometry"])
state = str(args["state"])
if state == 'cfl':
    JK_coeffs = str(args["JK_coeffs"])

extract = np.bool_(args["extract"])
if extract:
    if state == "free_fermions":
        ExtractSpFromFile(N, S, "torus", state, region_geometry)
    elif state == "cfl":
        ExtractSpFromFile(N, S, "torus", state+JK_coeffs, region_geometry)
else:
    nbr_iter = np.int64(args["nbr_iter"])
    nbr_nonthermal = np.int64(args["nbr_nonthermal"])
    if nbr_nonthermal == -1:
        if nbr_iter > 1e6:
            nbr_nonthermal = 1e5
        else:
            nbr_nonthermal = nbr_iter//10

    step = np.float64(args["step"])
    region_size = np.float64(args["region_size"])
    acceptance_ratio = np.float64(args["acc_ratio"])
    run_type = str(args["run"])

    save_all_config = np.bool_(args["save_all_config"])

    start_time = datetime.now()
    if state == "free_fermions":
        fqh = MonteCarloTorusFreeFermions(N=N, S=S, t=t*1j, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                          step_size=step, linear_size=region_size, region_geometry=region_geometry,
                                          nbr_copies=2, acceptance_ratio=acceptance_ratio,
                                          save_all_config=save_all_config)
    elif state == 'cfl':
        fqh = MonteCarloTorusCFL(N=N, S=S, t=t*1j, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                 step_size=step, linear_size=region_size, region_geometry=region_geometry,
                                 JK_coeffs=JK_coeffs, nbr_copies=2, acceptance_ratio=acceptance_ratio,
                                 save_all_config=save_all_config)

    if run_type == 'p':
        fqh.RunSwapP()
    elif run_type == 'mod':
        fqh.RunSwapMod()
    elif run_type == 'sign':
        fqh.RunSwapSign()

    end_time = datetime.now()
    if "day" in str(end_time - start_time):
        print(f"Total time = {str(end_time - start_time)[:18]} s")
    else:
        print(f"Total time = {str(end_time - start_time)[:10]} s")
    print(f"Time / 5% = {str((end_time - start_time)/20)[:10]} s")
