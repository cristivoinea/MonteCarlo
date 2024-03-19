from os.path import exists
from numba import njit
import numpy as np

from .utilities import Stats
from .MonteCarloBase import MonteCarloBase
from .fast_math import ThetaFunction


class MonteCarloTorus (MonteCarloBase):
    t: np.complex128
    Lx: np.float64
    Ly: np.float64

    def RandomPoint(self):
        return np.random.random()*self.Lx + 1j*np.random.random()*self.Ly

    def RandomConfig(self) -> np.array:
        """Returns a random configuration of particles."""
        R = np.zeros(self.Ne, dtype=np.complex128)
        for i in range(self.Ne):
            R[i] = self.RandomPoint()

        return R

    def RandomConfigSwap(self):
        """Returns two random configurations of particles, swappable
        with respect to region A.
        """
        self.coords = np.zeros(2*self.Ne, dtype=np.complex128)
        self.coords[:self.Ne] = self.RandomConfig()
        self.coords[self.Ne:] = self.RandomConfig()
        inside_region = self.InsideRegion(self.coords)

        while (np.count_nonzero(inside_region[:self.Ne]) !=
               np.count_nonzero(inside_region[self.Ne:])):
            self.coords[self.Ne:] = self.RandomConfig()
            inside_region = self.InsideRegion(self.coords)

    def BoundaryConditions(self, z) -> np.complex128:
        """Check if the particle position wrapped around the torus
        after one step. When a step wraps around both directions,
        the algorithm applies """
        z -= self.Lx*self.t*(np.imag(z) > self.Ly)
        z += self.Lx*self.t*(np.imag(z) < 0)
        z -= self.Lx*(np.real(z) > self.Lx)
        z += self.Lx*(np.real(z) < 0)

        return z

    def InsideRegion(self, coords):
        """
        Given an array of coordinates, returns a boolean array telling
        which particles are inside the subregion. Coordinates do not have
        to belong to all particles in the system.
        """
        if self.region_geometry == 'strip':
            y = np.imag(coords)
            inside_A = (y < self.boundary)
        elif self.region_geometry == 'circle':
            r = coords - (self.Lx + 1j*self.Ly)/2
            inside_A = (np.abs(r) < self.boundary)

        return inside_A

    def ReduceNonholomorphic(coords: np.array,
                             ) -> np.complex128:
        """Given the coordinates of the particles that move,
        returns the nonholomorphic Gaussian exponent in the wfn.

        Parameters:
        coords : array with updated coordiantes

        Output:
        expo_non_holomorphic : nonholomorphic Gaussian exponent"""
        expo_nonholomorphic = np.sum(
            np.power(coords, 2) - np.power(np.abs(coords), 2))/4
        """
        expo_nonholomorphic = 0
        for j in range(coords.size):
            expo_nonholomorphic += (coords[j]**2 - np.abs(coords[j])**2)/4
        """

        return expo_nonholomorphic

    def ReduceThetaFunctionCM(self, zCM: np.complex128) -> tuple[np.complex128, np.complex128]:
        """Using the properties of the theta function, splits the CM contribution
        to the wavefunction into a theta function and exponential contribution."""
        m = self.Ns/self.Ne

        zCM *= m/self.Lx
        c = np.imag(zCM)//np.imag(m*self.t)

        expo_CM = -1j*np.pi*c*(2*zCM - c*m*self.t + 2*self.bCM)

        reduced_CM = ThetaFunction(
            zCM - c*m*self.t, m*self.t, self.aCM + c, self.bCM)

        return reduced_CM, expo_CM

    """

    def SaveConfig(self, run_type: str):
        if run_type == 'sign':
            np.save(f"{run_type}_{self.state}_results_real_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    np.real(self.results))
            np.save(f"{run_type}_{self.state}_results_imag_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    np.imag(self.results))

        else:
            np.save(f"{run_type}_{self.state}_results_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    self.results)

        if run_type == 'sign' or run_type == 'mod':
            np.save(f"{run_type}_{self.state}_order_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    self.to_swap)

        np.save(f"{run_type}_{self.state}_coords_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy",
                self.coords)

    def SaveResults(self, run_type: str):

        if self.save_result:
            if run_type == 'sign':
                np.savetxt(f"{run_type}_{self.state}_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.dat",
                           np.vstack((Stats(np.real(self.results[int(self.nbr_nonthermal):])),
                                      Stats(
                                          np.imag(self.results[int(self.nbr_nonthermal):]))
                                      ))
                           )
            elif run_type == 'disorder':
                mean_re, var_re = Stats(
                    np.real(self.results[int(self.nbr_nonthermal):]))
                mean_im, var_im = Stats(
                    np.imag(self.results[int(self.nbr_nonthermal):]))
                mean = np.sqrt(mean_re**2 + mean_im**2)
                var = var_re*((mean_re/mean)**2) + var_im*((mean_im/mean)**2)
                np.savetxt(f"{run_type}_{self.state}_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.dat",
                           np.vstack((mean, var)))
            else:
                np.savetxt(f"{run_type}_{self.state}_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.dat",
                           Stats(self.results[int(self.nbr_nonthermal):]))

        else:
            if run_type == 'sign':
                mean, var = Stats(
                    np.real(self.results[int(self.nbr_nonthermal):]))
                print(f"\nMean (real) = {mean} \n Var (real) = {var}")
                mean, var = Stats(
                    np.imag(self.results[int(self.nbr_nonthermal):]))
                print(f"\nMean (imag) = {mean} \n Var (imag) = {var}")
            elif run_type == 'disorder':
                mean_re, var_re = Stats(
                    np.real(self.results[int(self.nbr_nonthermal):]))
                mean_im, var_im = Stats(
                    np.imag(self.results[int(self.nbr_nonthermal):]))
                mean = np.sqrt(mean_re**2 + mean_im**2)
                var = var_re*((mean_re/mean)**2) + var_im*((mean_im/mean)**2)
                print(f"\nMean = {mean} \n Var = {var}")
            else:
                mean, var = Stats(self.results[int(self.nbr_nonthermal):])
                print(f"\nMean = {mean} \n Var = {var}")

    def LoadRun(self, run_type: str):

        if self.acceptance_ratio > 0:
            file_coords = f"./{run_type}_{self.state}_coords_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy"
            if exists(file_coords):
                self.coords = np.load(file_coords)
            else:
                print(f"{file_coords} missing!\n")

            if run_type == 'mod' or run_type == 'sign':
                file_order = f"./{run_type}_{self.state}_order_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy"
                if exists(file_order):
                    self.to_swap = np.load(file_order)
                else:
                    print(f"{file_order} missing!\n")

            if run_type == 'sign' or run_type == 'disorder':
                file_results_real = f"./{run_type}_{self.state}_results_real_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy"
                file_results_imag = f"./{run_type}_{self.state}_results_imag_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy"
                if exists(file_results_real) and exists(file_results_real):
                    start_results = np.load(file_results_real) + \
                        np.load(file_results_imag)
                    self.load_iter = start_results.size

                    self.results = np.zeros(
                        (self.load_iter+self.nbr_nonthermal), dtype=np.complex128)
                    self.results[:self.load_iter] = start_results
                else:
                    print("Results file not found!\n")

            else:
                file_results = f"./{run_type}_{self.state}_results_Ne_{self.Ne}_Ns_{self.Ns}_t_{np.imag(self.t):.2f}_{self.region_geometry}_{self.region_size:.4f}.npy"
                if exists(file_results):
                    start_results = np.load(file_results)
                    self.load_iter = start_results.size

                    self.results = np.zeros(
                        (self.load_iter+self.nbr_nonthermal), dtype=np.float64)
                    self.results[:self.load_iter] = start_results
                else:
                    print("Results file not found!\n")

        else:
            if run_type == 'disorder':
                self.coords = self.RandomConfig()
            else:
                self.RandomConfigSwap()
            if run_type == 'sign' or run_type == 'disorder':
                self.results = np.zeros((self.nbr_iter), dtype=np.complex128)
            else:
                self.results = np.zeros((self.nbr_iter), dtype=np.float64)

        self.coords_tmp = np.copy(self.coords)
        if run_type == 'mod' or run_type == 'sign':
            self.AssignOrderToSwap()
            self.from_swap = self.OrderFromSwap(self.to_swap)
            self.to_swap_tmp = np.copy(self.to_swap)
            self.from_swap_tmp = np.copy(self.from_swap)
            """

    def __init__(self, Ne, Ns, t, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, area_size, linear_size,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.geometry = 'torus'
        self.t = t
        self.Lx = np.sqrt(2*np.pi*Ns/np.imag(self.t))
        self.Ly = self.Lx*np.imag(self.t)
        print(f"Torus dimensions \nLx = {self.Lx}, \nLy = {self.Ly}")

        if linear_size == 0 and area_size == 0:
            print("Region undefined, please define a subsystem.")
        elif linear_size != 0 and area_size != 0:
            print(
                "Region defined in two different ways, a single definition must be chosen.")
        elif linear_size != 0:
            self.boundary = self.Lx * linear_size
            region_size = linear_size
        else:
            self.boundary = self.Lx * np.sqrt(area_size/np.pi)
            region_size = area_size

        super().__init__(Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                         region_size, save_results, save_config, acceptance_ratio)

        self.step_size = step_size*self.Lx
        print(f"Step size = {step_size:.4f}*Lx = {self.step_size}")
        self.step_pattern = np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                      (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])
        self.acceptance_ratio = acceptance_ratio
