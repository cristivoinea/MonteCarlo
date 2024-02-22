import numpy as np
from os.path import exists

from .MonteCarloBase import MonteCarloBase
from .utilities import Stats


class MonteCarloSphere (MonteCarloBase):
    Ls: np.array

    def RandomPoint(self):
        return np.array([np.random.random()*np.pi, np.random.random()*2*np.pi])

    def RandomConfig(self) -> np.array:
        """Returns a random configuration of particles."""
        R = np.vstack((self.RandomPoint(), self.RandomPoint()))
        for _ in range(2, self.Ne):
            R = np.vstack((R, self.RandomPoint()))

        return R

    def RandomConfigSwap(self):
        """Returns two random configurations of particles, swappable
        with respect to region A.
        """
        self.coords = np.vstack((self.RandomConfig(), self.RandomConfig()))
        inside_region = self.InsideRegion(self.coords)

        while (np.count_nonzero(inside_region[:self.Ne]) !=
               np.count_nonzero(inside_region[self.Ne:])):
            self.coords[self.Ne:] = self.RandomConfig()
            inside_region = self.InsideRegion(self.coords)

    def BoundaryConditions(self, z) -> np.complex128:
        """Check if the particle position wrapped around the torus
        after one step. When a step wraps around both directions,
        the algorithm applies """
        z[..., 1] = z[..., 1] + np.pi*((z[..., 0] > np.pi) | (z[..., 0] < 0))
        z[..., 1] = z[..., 1] - 2*np.pi*(z[..., 1]//(2*np.pi))
        z[..., 0] = z[..., 0] + 2*np.pi*(z[..., 0] > np.pi) - 2*z[..., 0] * \
            ((z[..., 0] > np.pi) | (z[..., 0] < 0))

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
            inside_A = (coords[:, 0] < self.boundary)

        return inside_A

    def SaveConfig(self, run_type: str):
        if run_type == 'sign':
            np.save(f"{self.state}_{run_type}_results_real_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    np.real(self.results))
            np.save(f"{self.state}_{run_type}_results_imag_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    np.imag(self.results))

        else:
            np.save(f"{self.state}_{run_type}_results_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    self.results)

        if run_type == 'sign' or run_type == 'mod':
            np.save(f"{self.state}_{run_type}_order_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy",
                    self.to_swap)

        np.save(f"{self.state}_{run_type}_coords_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy",
                self.coords)

    def SaveResults(self, run_type: str):

        if self.save_result:
            if run_type == 'sign':
                np.savetxt(f"{self.state}_{run_type}_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.dat",
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
                np.savetxt(f"{self.state}_{run_type}_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.dat",
                           np.vstack((mean, var)))
            else:
                np.savetxt(f"{self.state}_{run_type}_{self.geometry}_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.dat",
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
            file_coords = f"./{run_type}_{self.state}_coords_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy"
            if exists(file_coords):
                self.coords = np.load(file_coords)
            else:
                print(f"{file_coords} missing!\n")

            if run_type == 'mod' or run_type == 'sign':
                file_order = f"./{run_type}_{self.state}_order_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy"
                if exists(file_order):
                    self.to_swap = np.load(file_order)
                else:
                    print(f"{file_order} missing!\n")

            if run_type == 'sign' or run_type == 'disorder':
                file_results_real = f"./{run_type}_{self.state}_results_real_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy"
                file_results_imag = f"./{run_type}_{self.state}_results_imag_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy"
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
                file_results = f"./{run_type}_{self.state}_results_Ne_{self.Ne}_Ns_{self.Ns}_{self.region_geometry}_{self.region_size:.4f}.npy"
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

    def __init__(self, Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, region_size, linear_size=False,
                 save_results=True, save_config=True, acceptance_ratio=0):

        super().__init__(Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                         region_size, save_results, save_config, acceptance_ratio)

        self.geometry = "sphere"
        self.Ls = np.zeros((self.Ne, 2))
        l = m = 0
        for i in range(self.Ne):
            self.Ls[i] = np.array([l, m])
            if m == l:
                l += 1
                m = -l
            else:
                m += 1

        self.step_size = step_size*np.pi
        if linear_size:
            self.boundary = np.arcsin(region_size)
        else:
            self.boundary = 1 - np.arccos(region_size)

        self.step_pattern = np.array([[0, 2], [0, -2], [1, 0], [-1, 0],
                                      [1, 1], [1, -1], [-1, 1], [-1, -1]])
