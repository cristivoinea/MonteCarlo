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
        R = np.zeros(self.N, dtype=np.complex128)
        for i in range(self.N):
            R[i] = self.RandomPoint()

        return R

    def RandomConfigSwap(self):
        """Returns two random configurations of particles, swappable
        with respect to region A.
        """
        self.coords = np.zeros(2*self.N, dtype=np.complex128)
        self.coords[:self.N] = self.RandomConfig()
        self.coords[self.N:] = self.RandomConfig()
        inside_region = self.InsideRegion(self.coords)

        while (np.count_nonzero(inside_region[:self.N]) !=
               np.count_nonzero(inside_region[self.N:])):
            self.coords[self.N:] = self.RandomConfig()
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
        else:
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
        m = self.S/self.N

        zCM *= m/self.Lx
        c = np.imag(zCM)//np.imag(m*self.t)

        expo_CM = -1j*np.pi*c*(2*zCM - c*m*self.t + 2*self.bCM)

        reduced_CM = ThetaFunction(
            zCM - c*m*self.t, m*self.t, self.aCM + c, self.bCM)

        return reduced_CM, expo_CM

    def __init__(self, N, S, t, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, area_size, linear_size,
                 save_results=True, save_last_config=True,
                 save_all_config=True, acceptance_ratio=0):

        self.geometry = 'torus'
        self.t = t
        self.Lx = np.sqrt(2*np.pi*S/np.imag(self.t))
        self.Ly = self.Lx*np.imag(self.t)
        # print(f"Torus dimensions \nLx = {self.Lx}, \nLy = {self.Ly}")

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

        region_details = "_" + region_geometry + f"_{region_size:.6f}"

        super().__init__(N, S, nbr_iter, nbr_nonthermal, region_details,
                         save_results, save_last_config,
                         save_all_config, acceptance_ratio)

        self.step_size = step_size*self.Lx
        # print(f"Step size = {step_size:.4f}*Lx = {self.step_size}")
        self.step_pattern = np.array([1, -1, 1j, -1j, (1+1j)*np.sqrt(2)/2, (1-1j)*np.sqrt(2)/2,
                                      (-1+1j)*np.sqrt(2)/2, (-1-1j)*np.sqrt(2)/2])
        self.acceptance_ratio = acceptance_ratio
