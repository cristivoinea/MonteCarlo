# from MonteCarloTorusSWAP import RunPSwapCFL, RunModSwapCFL, RunSignSwapCFL, \
#    RunPSwapFreeFermions, RunModSwapFreeFermions, RunSignSwapFreeFermions
import sys
import os

module_path = os.path.abspath(os.path.join('..'))  # nopep8
if module_path not in sys.path:
    sys.path.append(module_path)

from src.MonteCarloTorusCFL import MonteCarloTorusCFL  # nopep8
from src.MonteCarloSphereCFL import MonteCarloSphereCFL  # nopep8
from src.MonteCarloTorusFreeFermions import MonteCarloTorusFreeFermions  # nopep8
from src.MonteCarloSphereFreeFermions import MonteCarloSphereFreeFermions  # nopep8
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
parser.add_argument("--geometry", action="store", default='torus',
                    help="system geometry")
parser.add_argument("--region-geometry", action="store", default='circle',
                    help="geometry of one of the bipartition regions")
parser.add_argument("--linear", action="store_true",
                    help="enforce that the region size is the linear size of the subregion")
parser.add_argument("--pf", action="store_true",
                    help="add a Pfaffian to the wavefunction")
parser.add_argument("--acc-ratio", action="store", default=0,
                    help="loads a previous run with given acceptance")
parser.add_argument("--run", action="store", required=True,
                    help="type of Monte Carlo run")
parser.add_argument("--state", action="store", required=True,
                    help="type of state")
parser.add_argument("--JK-coeffs", action="store", default='0',
                    help="JK translation coefficients for enforcing PBC")

args = vars(parser.parse_args())

Ne = np.uint8(args["Ne"])
Ns = np.uint8(args["Ns"])
nbr_iter = np.uint32(args["nbr_iter"])
nbr_nonthermal = np.int32(args["nbr_nonthermal"])
if nbr_nonthermal == -1:
    if nbr_iter > 1e6:
        nbr_nonthermal = 1e5
    else:
        nbr_nonthermal = nbr_iter//10

step = np.float64(args["step"])
A_start = np.float64(args["A_start"])
A_end = np.float64(args["A_end"])
if A_end == -1:
    A_end = A_start
nbr_A = np.uint8(args["nbr_A"])
A_sizes = np.linspace(A_start, A_end, nbr_A, endpoint=True)
geometry = str(args["geometry"])
region_geometry = str(args["region_geometry"])
linear = bool(args["linear"])
pf = bool(args["pf"])
acceptance_ratio = np.float64(args["acc_ratio"])
run_type = str(args["run"])
state = str(args["state"])
if state == 'cfl':
    JK_coeffs = str(args["JK_coeffs"])
t = 1j

for region_size in A_sizes:
    start_time = datetime.now()
    if geometry == "torus":
        if state == 'cfl':
            fqh = MonteCarloTorusCFL(Ne=Ne, Ns=Ns, t=t, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                     region_geometry=region_geometry, region_size=region_size, step_size=step,
                                     nbr_copies=(2-(run_type == 'disorder')), flag_pf=pf,
                                     linear_size=linear, JK_coeffs=JK_coeffs, acceptance_ratio=acceptance_ratio)
        elif state == 'free_fermions':
            fqh = MonteCarloTorusFreeFermions(Ne=Ne, Ns=Ns, t=t, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                              region_geometry=region_geometry, region_size=region_size, step_size=step,
                                              nbr_copies=(2-(run_type == 'disorder')), linear_size=linear,
                                              acceptance_ratio=acceptance_ratio)
    elif geometry == "sphere":
        if state == "free_fermions":
            fqh = MonteCarloSphereFreeFermions(Ne=Ne, Ns=Ns, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                               region_geometry=region_geometry, region_size=region_size, step_size=step,
                                               nbr_copies=(2-(run_type == 'disorder')), linear_size=linear,
                                               acceptance_ratio=acceptance_ratio)
        elif state == 'cfl':
            fqh = MonteCarloSphereCFL(Ne=Ne, Ns=Ns, nbr_iter=nbr_iter, nbr_nonthermal=nbr_nonthermal,
                                      region_geometry=region_geometry, region_size=region_size, step_size=step,
                                      nbr_copies=(2-(run_type == 'disorder')), linear_size=linear,
                                      acceptance_ratio=acceptance_ratio)

    if run_type == 'p':
        fqh.RunSwapP()
    elif run_type == 'mod':
        fqh.RunSwapMod()
    elif run_type == 'sign':
        fqh.RunSwapSign()
    elif run_type == 'disorder':
        fqh.RunDisorderOperator()

    end_time = datetime.now()
    print(f"Total time = {str(end_time - start_time)[:10]} s")
    print(f"Time / 5% = {str((end_time - start_time)/20)[:10]} s")
