import numpy as np
from os.path import exists

from .MonteCarloBase import MonteCarloBase
from .utilities import Stats
from .fast_math import JackknifeMean, JackknifeVariance


class MonteCarloSphere (MonteCarloBase):
    Ls: np.array

    def FillLambdaLevels(self):
        self.Ls = np.zeros((self.Ne, 2), dtype=np.float64)
        l = self.Ns_eff/2
        m = -l
        for i in range(self.Ne):
            self.Ls[i] = np.array([l, m])
            if m == l:
                l += 1
                m = -l
            else:
                m += 1

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

    def Jacobian(self):
        return np.prod(np.sin(self.coords_tmp[:, 0])/np.sin(self.coords[:, 0]))

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
            inside_A = (coords[..., 0] < self.boundary)

        return inside_A

    def StepOneParticle(self):
        self.moved_particles = np.random.randint(0, self.Ne, 1)
        p = self.moved_particles[0]
        self.coords_tmp[p, :] = self.BoundaryConditions(
            self.coords_tmp[p, :] + np.array([1, np.sin(self.coords_tmp[p, :][0])]) *
            self.step_size*np.random.default_rng().choice(self.step_pattern, 1))

    def StepOneParticleTwoCopies(self):
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A.
        """
        self.moved_particles[0] = np.random.randint(0, self.Ne)
        self.moved_particles[1] = np.random.randint(self.Ne, 2*self.Ne)
        measure = np.sin(self.coords_tmp[self.moved_particles][:, 0])
        meas_array = np.array([[1, measure[0]], [1, measure[1]]])
        self.coords_tmp[self.moved_particles] = \
            self.BoundaryConditions(self.coords_tmp[self.moved_particles] + meas_array *
                                    self.step_size*np.random.default_rng().choice(self.step_pattern, 2))

    def StepOneSwap(self) -> np.array:
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A.

        Parameters:

        coords_tmp : initial position of all particles

        Output:

        coords_final : final position of all particles
        p : indices of particles that move in each copy
        delta : contains information about which step is taken
        """

        valid = False
        while not valid:
            self.moved_particles[0] = np.random.randint(0, self.Ne)
            self.moved_particles[1] = np.random.randint(self.Ne, 2*self.Ne)
            inside_region = self.InsideRegion(
                self.coords_tmp[self.moved_particles])

            measure = np.sin(self.coords_tmp[self.moved_particles][:, 0])
            meas_array = np.array([[1, measure[0]], [1, measure[1]]])
            coords_step = self.BoundaryConditions(self.coords_tmp[self.moved_particles] + meas_array *
                                                  self.step_size*np.random.default_rng().choice(self.step_pattern, 2))
            inside_region_tmp = self.InsideRegion(coords_step)
            if ((int(inside_region[0]) - int(inside_region_tmp[0])) ==
                    (int(inside_region[1]) - int(inside_region_tmp[1]))):
                valid = True
                nbr_A_changes = (inside_region[0] ^ inside_region_tmp[0])

        self.coords_tmp[self.moved_particles] = coords_step

        return nbr_A_changes

    def __init__(self, Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                 step_size, theta_size, linear_size,
                 save_results=True, save_config=True, acceptance_ratio=0):

        self.step_size = np.arcsin(step_size)
        if linear_size == 0 and theta_size == 0:
            print("Region undefined, redefining as the whole system.")
            self.boundary = np.pi
            region_size = self.boundary
        elif linear_size != 0 and theta_size != 0:
            print(
                "Region defined in two different ways, a single definition must be chosen.")
        elif linear_size != 0:
            self.boundary = np.arcsin(linear_size)
            region_size = linear_size
        else:
            self.boundary = theta_size
            region_size = self.boundary

        super().__init__(Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                         region_size, save_results, save_config, acceptance_ratio)

        self.geometry = "sphere"

        self.step_pattern = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                                      [1, 1], [1, -1], [-1, 1], [-1, -1]])
