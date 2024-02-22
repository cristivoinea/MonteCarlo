import numpy as np

from .MonteCarloBase import MonteCarloBase


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
