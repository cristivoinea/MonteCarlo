import numpy as np
from os.path import exists

from .MonteCarloBase import MonteCarloBase


class MonteCarloSphere (MonteCarloBase):
    Ls: np.array
    region_theta: np.ndarray
    region_phi: np.ndarray

    def FillLambdaLevels(self):
        Ls = np.zeros((self.N, 2), dtype=np.float64)
        l = self.S_eff/2
        m = -l
        for i in range(self.N):
            Ls[i] = np.array([l, m])
            if m == l:
                l += 1
                m = -l
            else:
                m += 1

        self.Ls = np.zeros((self.N, 2), dtype=np.int64)
        self.Ls[:, 0] = np.int64(Ls[:, 0] - self.S_eff/2)
        self.Ls[:, 1] = np.int64(Ls[:, 1] + Ls[:, 0])

    def RandomPoint(self):
        return np.array([np.arccos(1-2*np.random.random()), 2*np.random.random()*np.pi])

    def RandomConfig(self) -> np.array:
        """Returns a random configuration of particles."""
        R = np.vstack((self.RandomPoint(), self.RandomPoint()))
        for _ in range(2, self.N):
            R = np.vstack((R, self.RandomPoint()))
        # i = np.arange(self.N)+0.5
        # R = np.zeros((self.N, 2), dtype=np.float64)
        # R[:, 0] = np.arccos(1 - 2*i/self.N)
        # R[:, 1] = np.pi * (1 + 5**0.5) * i

        return R

    def RandomConfigSwap(self):
        """Returns two random configurations of particles, swappable
        with respect to region A.
        """
        self.coords = np.vstack((self.RandomConfig(), self.RandomConfig()))
        inside_region = self.InsideRegion(self.coords)

        while (np.count_nonzero(inside_region[:self.N]) !=
               np.count_nonzero(inside_region[self.N:])):
            self.coords[self.N:] = self.RandomConfig()
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

    def InsideRegion(self, coords, boundaries: str = "12"):
        """
        Given an array of coordinates, returns a boolean array telling
        which particles are inside the subregion. Coordinates do not have
        to belong to all particles in the system.
        """
        if self.region_geometry == 'circle':
            inside_A = (coords[..., 0] < self.boundary)
        else:
            if boundaries == "12":
                inside_A = ((coords[..., 0] > self.region_theta[0]) &
                            (coords[..., 0] < self.region_theta[1]) &
                            (coords[..., 1] > self.region_phi[0]) &
                            (coords[..., 1] < self.region_phi[1]))
            elif boundaries == "01":
                inside_A = ((coords[..., 0] < self.region_theta[0]) &
                            (coords[..., 1] < self.region_phi[0]) )
            elif boundaries == "02":
                inside_A = ((coords[..., 0] < self.region_theta[1]) &
                            (coords[..., 1] < self.region_phi[1]) )

        return inside_A

    def StepOneParticle(self):
        self.moved_particles = np.random.randint(0, self.N, 1)
        p = self.moved_particles[0]
        self.coords_tmp[p, :] = self.BoundaryConditions(
            self.coords_tmp[p, :] + np.array([1, np.sin(self.coords_tmp[p, 0])]) *
            self.step_size*np.random.default_rng().choice(self.step_pattern, 1))

    def StepOneParticleTwoCopies(self):
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A.
        """
        self.moved_particles[0] = np.random.randint(0, self.N)
        self.moved_particles[1] = np.random.randint(self.N, 2*self.N)
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
            self.moved_particles[0] = np.random.randint(0, self.N)
            self.moved_particles[1] = np.random.randint(self.N, 2*self.N)
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

    def __init__(self, N, S, nbr_iter, nbr_nonthermal,
                 step_size, region_theta=180, region_phi=360,
                 save_results=True, save_last_config=True,
                 save_all_config=True, acceptance_ratio=0):

        self.step_size = np.arcsin(step_size)

        region_details = ""
        if isinstance(region_theta, np.ndarray):
            self.region_theta = region_theta
        else:
            self.region_theta = np.array([0, region_theta], dtype=np.float64)
        if self.region_theta[1] != 180 or self.region_theta[0] != 0:
            region_details += f"_theta_{self.region_theta[0]:.6f}_{self.region_theta[1]:.6f}"

        if isinstance(region_phi, np.ndarray):
            self.region_phi = region_phi
        else:
            self.region_phi = np.array([0, region_phi], dtype=np.float64)
        if self.region_phi[1] != 360 or self.region_phi[0] != 0:
            region_details += f"_phi_{self.region_phi[0]:.6f}_{self.region_phi[1]:.6f}"

        super().__init__(N, S, nbr_iter, nbr_nonthermal, region_details,
                         save_results, save_last_config, save_all_config, acceptance_ratio)

        self.geometry = "sphere"

        # Convert degrees to radians for internal calculations
        self.region_theta = self.region_theta*np.pi/180
        self.region_phi = self.region_phi*np.pi/180

        self.step_pattern = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                                      [1, 1], [1, -1], [-1, 1], [-1, -1]])
