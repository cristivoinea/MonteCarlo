from os.path import exists
from numba import njit
import numpy as np

from .utilities import Stats, ThetaFunction


class MonteCarloBase:
    Ne: np.uint8
    Ns: np.uint16
    vortices: np.uint8
    Ns_eff: np.int64
    geometry: str
    region_geometry: str
    region_size: np.float64
    step_size: np.float64
    step_pattern: np.array
    boundary: np.float64
    state: str
    geometry: str

    nbr_iter: np.uint64
    load_iter: np.uint64 = 0
    nbr_nonthermal: np.uint64
    acceptance_ratio: np.float64 = 0
    results: np.array
    coords: np.array
    coords_tmp: np.array
    moved_particles: np.array

    save_config: np.bool_
    save_result: np.bool_

    def RandomPoint(self):
        """Returns random coordinates for one particle."""
        pass

    def RandomConfig(self) -> np.array:
        """Returns a random configuration of particles."""
        pass

    def RandomConfigSwap(self):
        """Returns two random configurations of particles, swappable
        with respect to region A.
        """
        pass

    def BoundaryConditions(self, z) -> np.complex128:
        """Check if the particle position wrapped around the torus
        after one step. When a step wraps around both directions,
        the algorithm applies """
        pass

    def UpdateOrderSwap(self, nbr_A_changes: bool):
        """
        Updates the order of particles in the swapped copies after one step.

        Parameters:
        to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:Ne]
                    and [Ne, 2*Ne) go into coords_swap[Ne:]
        from_swap : order of swapped particles in the original copies. [0,Ne) go into coords[:Ne]
                    and [Ne, 2*Ne) go into coords[Ne:]
        p : array containing indices of moved particles in the two initial copies
        nbr_A_changes : indicates whether the number of particles in the subregion changed
        """

        if nbr_A_changes:
            self.from_swap_tmp[self.to_swap_tmp[self.moved_particles[0]]
                               ] = self.moved_particles[1]
            self.from_swap_tmp[self.to_swap_tmp[self.moved_particles[1]]
                               ] = self.moved_particles[0]

            tmp = self.to_swap_tmp[self.moved_particles[0]]
            self.to_swap_tmp[self.moved_particles[0]
                             ] = self.to_swap_tmp[self.moved_particles[1]]
            self.to_swap_tmp[self.moved_particles[1]] = tmp

    def AssignOrderToSwap(self):
        """
        Given an array specifying whether each particle is inside the subregion A, 
        this method returns two arrays: 
        one containing the conversion from original -> swap indices and its
        inverse.
        Output:
        to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:,0]
                            and [Ne, 2*Ne) go into coords_swap[:,1]
        """
        self.to_swap = np.zeros((2*self.Ne), dtype=np.uint8)
        inside_region = self.InsideRegion(self.coords)
        i_swap = 0
        j = 0

        for i in range(self.Ne, 2*self.Ne):
            if not inside_region[i]:
                self.to_swap[i] = i_swap
            else:
                while not inside_region[j]:
                    j += 1
                self.to_swap[j] = i_swap
                j += 1
            i_swap += 1

        i_swap = self.Ne
        j = self.Ne
        for i in range(self.Ne):
            if not inside_region[i]:
                self.to_swap[i] = i_swap
            else:
                while not inside_region[j]:
                    j += 1
                self.to_swap[j] = i_swap
                j += 1
            i_swap += 1

        return self.to_swap

    def OrderFromSwap(self, to_swap: np.array
                      ) -> np.array:
        """
        Given ordering of particles from the original copies to the swapped copies,
        returns the inverse mapping.

        Parameters:
        to_swap : order of original particles in the swapped copies. [0,Ne) go into coords_swap[:,0]
                            and [Ne, 2*Ne) go into coords_swap[:,1]

        Output:
        from_swap : order of swapped particles in the original copies. [0,Ne) go into coords[:,0]
                            and [Ne, 2*Ne) go into coords[:,1]
        """
        from_swap = np.zeros((2*self.Ne), dtype=np.uint8)

        for i in range(2*self.Ne):
            from_swap[to_swap[i]] = i

        return from_swap

    def InsideRegion(self):
        """
        Given an array of coordinates, returns a boolean array telling
        which particles are inside the subregion. Coordinates do not have
        to belong to all particles in the system.
        """
        pass

    def StepOneParticle(self):
        self.moved_particles = np.random.randint(0, self.Ne, 1)
        self.coords_tmp[self.moved_particles] = self.BoundaryConditions(
            self.coords_tmp[self.moved_particles] +
            self.step_size*np.random.default_rng().choice(self.step_pattern, 1))

    def StepOneParticleTwoCopies(self):
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A.
        """
        self.moved_particles[0] = np.random.randint(0, self.Ne)
        self.moved_particles[1] = np.random.randint(self.Ne, 2*self.Ne)
        self.coords_tmp[self.moved_particles] = \
            self.BoundaryConditions(self.coords_tmp[self.moved_particles] +
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
            coords_step = self.BoundaryConditions(self.coords_tmp[self.moved_particles] +
                                                  self.step_size*np.random.default_rng().choice(self.step_pattern, 2))
            inside_region_tmp = self.InsideRegion(coords_step)
            if ((int(inside_region[0]) - int(inside_region_tmp[0])) ==
                    (int(inside_region[1]) - int(inside_region_tmp[1]))):
                valid = True
                nbr_A_changes = (inside_region[0] ^ inside_region_tmp[0])

        self.coords_tmp[self.moved_particles] = coords_step

        return nbr_A_changes

    def Checkpoint(self, current_iter, run_type):
        print('Iteration', current_iter+1, 'done, current acceptance ratio:',
              np.round(self.acceptance_ratio*100/(current_iter+1), 2), '%')
        if self.save_config:
            self.SaveConfig(run_type)

    def InitialWavefn(self):
        pass

    def InitialWavefnSwap(self):
        pass

    def TmpWavefn(self):
        pass

    def TmpWavefnSwap(self):
        pass

    def StepAmplitude(self):
        pass

    def StepAmplitudeTwoCopies(self):
        pass

    def StepAmplitudeTwoCopiesSwap(self):
        pass

    def AcceptTmp(self, run_type):
        self.acceptance_ratio += 1
        self.coords[self.moved_particles] = self.coords_tmp[self.moved_particles]

        if run_type == 'mod' or run_type == 'sign':
            np.copyto(self.to_swap, self.to_swap_tmp)
            np.copyto(self.from_swap, self.from_swap_tmp)

    def RejectTmp(self, run_type):
        self.coords_tmp[self.moved_particles] = self.coords[self.moved_particles]

        if run_type == 'mod' or run_type == 'sign':
            np.copyto(self.to_swap_tmp, self.to_swap)
            np.copyto(self.from_swap_tmp, self.from_swap)

    def InitialMod(self):
        pass

    def InitialSign(self):
        pass

    def Jacobian(self):
        return 1

    def CF(self):
        pass

    def RunDisorderOperator(self, composite=False):
        """
        Computes the disorder operator.
        """
        self.LoadRun('disorder')
        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        if composite:
            update = self.CF()
        else:
            update = np.exp(1j*(np.count_nonzero(inside_region))*np.pi/2)

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            self.StepOneParticle()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptTmp('disorder')
                inside_region = self.InsideRegion(self.coords)
                if composite == True:
                    update = self.CF()
                else:
                    update = np.exp(
                        1j*(np.count_nonzero(inside_region))*np.pi/2)

            else:
                self.RejectTmp('disorder')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'disorder')

        self.SaveResults('disorder')

    def RunParticleFluctuations(self, theta: np.float64 = 0):
        """
        Computes the particle number fluctuations.
        """
        self.LoadRun('nbr_particles')
        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        update = np.count_nonzero(inside_region)

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            self.StepOneParticle()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptTmp('nbr_particles')
                inside_region = self.InsideRegion(self.coords)
                update = np.count_nonzero(inside_region)

            else:
                self.RejectTmp('nbr_particles')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'nbr_particles')

        if theta != 0:
            self.result = np.exp(1j*self.result*theta)

        self.SaveResults('nbr_particles', theta)

    def RunSwapP(self):
        """
        Computes the p-term in the entanglement entropy swap decomposition 
        (the probability of two copies being swappable).
        """
        self.LoadRun('p')
        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        update = (np.count_nonzero(inside_region[:self.Ne]) ==
                  np.count_nonzero(inside_region[self.Ne:]))

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            self.StepOneParticleTwoCopies()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptTmp('p')
                inside_region = self.InsideRegion(self.coords)
                update = (np.count_nonzero(inside_region[:self.Ne]) ==
                          np.count_nonzero(inside_region[self.Ne:]))
            else:
                self.RejectTmp('p')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'p')

        self.SaveResults('p')

    def RunSwapMod(self):
        """
        Computes the mod-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('mod')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialMod()

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            nbr_in_region_changes = self.StepOneSwap()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*step_amplitude*np.conj(step_amplitude) > np.random.random():
                self.UpdateOrderSwap(nbr_in_region_changes)
                self.TmpWavefnSwap()
                step_amplitude_swap = self.StepAmplitudeTwoCopiesSwap()
                update *= np.abs(step_amplitude_swap / step_amplitude)
                self.AcceptTmp('mod')

            else:
                self.RejectTmp('mod')

            self.results[i] = update
            # if (i+1-self.load_iter) % (self.nbr_iter//100) == 0:
            # print(step_amplitude, step_amplitude_swap)
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'mod')

        self.SaveResults('mod')

    def RunSwapSign(self):
        """
        Computes the sign-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('sign')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialSign()

        for i in range(self.load_iter, self.load_iter+self.nbr_iter):
            nbr_in_region_changes = self.StepOneSwap()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()
            self.UpdateOrderSwap(nbr_in_region_changes)
            self.TmpWavefnSwap()
            step_amplitude *= np.conj(self.StepAmplitudeTwoCopiesSwap())

            if self.Jacobian()*np.abs(step_amplitude) > np.random.random():
                update *= step_amplitude / np.abs(step_amplitude)
                self.AcceptTmp('sign')

            else:
                self.RejectTmp('sign')

            self.results[i] = update
            if (i+1-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'sign')

        self.SaveResults('sign')

    def __init__(self, Ne, Ns, nbr_iter, nbr_nonthermal, region_geometry,
                 region_size, save_results=True, save_config=True,
                 acceptance_ratio=0):
        self.Ne = Ne
        self.Ns = Ns
        self.nbr_iter = nbr_iter
        self.nbr_nonthermal = nbr_nonthermal
        self.region_geometry = region_geometry
        self.region_size = region_size
        self.save_config = save_config
        self.save_result = save_results
        self.acceptance_ratio = acceptance_ratio
