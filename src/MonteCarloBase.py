from os.path import exists
from numba import njit
import numpy as np
from itertools import combinations

# from .utilities import Stats, ThetaFunction
from .fast_math import JackknifeMean, JackknifeVariance, ThetaFunction


class MonteCarloBase:
    N: np.int64
    S: np.int64
    vortices: np.int64
    S_eff: np.int64
    geometry: str
    region_details: str
    step_size: np.float64
    step_pattern: np.array
    boundary: np.float64
    boundary_delta: np.float64
    state: str
    geometry: str

    nbr_iter: np.int64
    load_iter: np.int64 = 0
    nbr_nonthermal: np.int64
    acceptance_ratio: np.float64 = 0
    results: np.array
    coords: np.array
    coords_tmp: np.array
    moved_particles: np.array

    save_last_config: np.bool_
    save_result: np.bool_
    save_all_config: np.bool_

    def ComputeEntropyED(self, entropy="s2"):
        overlap_matrix = self.GetOverlapMatrix()
        eigs = np.linalg.eigvalsh(overlap_matrix)

        if entropy == 's2':
            return -np.sum(np.log(eigs**2 + (1-eigs)**2))
        elif entropy == 's1':
            return -np.sum(eigs*np.log(eigs) + (1-eigs)*np.log(1-eigs))

    def ComputeParticleFluctuationsED(self):
        overlap_matrix = self.GetOverlapMatrix()
        return np.trace(np.matmul(overlap_matrix, (np.eye(self.N)-overlap_matrix)))

    def ComputeSwapProbability(self):
        overlap_matrix = self.GetOverlapMatrix()
        eigs = np.linalg.eigvalsh(overlap_matrix)
        p = 0
        for i in range(self.N+1):
            p_i = 0
            for c in combinations(range(self.N), i):
                t_c = 1
                for j in range(self.N):
                    if j in c:
                        t_c *= eigs[j]
                    else:
                        t_c *= (1-eigs[j])
                p_i += t_c
            p += p_i**2

        return p

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
        to_swap : order of original particles in the swapped copies. [0,N) go into coords_swap[:N]
                    and [N, 2*N) go into coords_swap[N:]
        from_swap : order of swapped particles in the original copies. [0,N) go into coords[:N]
                    and [N, 2*N) go into coords[N:]
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
        to_swap : order of original particles in the swapped copies. [0,N) go into coords_swap[:,0]
                            and [N, 2*N) go into coords_swap[:,1]
        """
        self.to_swap = np.zeros((2*self.N), dtype=np.int64)
        inside_region = self.InsideRegion(self.coords)
        i_swap = 0
        j = 0

        for i in range(self.N, 2*self.N):
            if not inside_region[i]:
                self.to_swap[i] = i_swap
            else:
                while not inside_region[j]:
                    j += 1
                self.to_swap[j] = i_swap
                j += 1
            i_swap += 1

        i_swap = self.N
        j = self.N
        for i in range(self.N):
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
        to_swap : order of original particles in the swapped copies. [0,N) go into coords_swap[:,0]
                            and [N, 2*N) go into coords_swap[:,1]

        Output:
        from_swap : order of swapped particles in the original copies. [0,N) go into coords[:,0]
                            and [N, 2*N) go into coords[:,1]
        """
        from_swap = np.zeros((2*self.N), dtype=np.int64)

        for i in range(2*self.N):
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
        self.moved_particles = np.random.randint(0, self.N, 1)
        self.coords_tmp[self.moved_particles] = self.BoundaryConditions(
            self.coords_tmp[self.moved_particles] +
            self.step_size*np.random.default_rng().choice(self.step_pattern, 1))

    def StepOneParticleTwoCopies(self):
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A.
        """
        self.moved_particles[0] = np.random.randint(0, self.N)
        self.moved_particles[1] = np.random.randint(self.N, 2*self.N)
        self.coords_tmp[self.moved_particles] = \
            self.BoundaryConditions(self.coords_tmp[self.moved_particles] +
                                    self.step_size*np.random.default_rng().choice(self.step_pattern, 2))

    def StepOneSwap(self) -> np.array:
        """Provides a new Monte Carlo configuration by updating
        the coordinates of one particle in each copy, ensuring that
        the copies are swappable with respect to region A. If self.N_reg is
        defined, the new configuration will preserve this quantity.
        """

        valid = False
        while not valid:
            self.moved_particles[0] = np.random.randint(0, self.N)
            self.moved_particles[1] = np.random.randint(self.N, 2*self.N)
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
        print('Iteration', current_iter, 'done, current acceptance ratio:',
              np.round(self.acceptance_ratio*100/current_iter, 2), '%')
        if self.save_last_config:
            self.SaveConfig(current_iter, run_type)

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

    def SaveConfig(self, current_iter: np.int64, run_type: str):
        # if run_type == 'sign':
        #    np.save(f"{self.state}_{run_type}_results_real_{self.geometry}_N_{self.N}_S_{self.S}_{self.region_geometry}_{self.region_size:.4f}.npy",
        #            np.real(self.results))
        #    np.save(f"{self.state}_{run_type}_results_imag_{self.geometry}_N_{self.N}_S_{self.S}_{self.region_geometry}_{self.region_size:.4f}.npy",
        #            np.imag(self.results))

        # else:
        np.save(f"{self.state}_{self.geometry}_{run_type}_results_N_{self.N}_S_{self.S}{self.region_details}.npy",
                self.results[:current_iter])

        save_coords = np.copy(self.coords)

        if run_type == 'sign' or run_type == 'mod':
            if len(self.coords.shape) == 1:
                save_coords = np.vstack((save_coords, self.to_swap)).T
            else:
                save_coords = np.hstack((save_coords, self.to_swap[:, None]))
            # np.save(f"{self.state}_{run_type}_order_{self.geometry}_N_{self.N}_S_{self.S}_{self.region_geometry}_{self.region_size:.4f}.npy",
            #        self.to_swap)

        np.save(f"{self.state}_{self.geometry}_{run_type}_coords_end_N_{self.N}_S_{self.S}{self.region_details}.npy",
                save_coords)

    def SaveResults(self, run_type: str, extra_param=0):
        if run_type == 'fluct':
            final_value = JackknifeVariance(
                np.real(self.results[np.int64(self.nbr_nonthermal):]))
        else:
            final_value = JackknifeMean(
                np.real(self.results[np.int64(self.nbr_nonthermal):]))
            if run_type == 'sign':
                imag_value = JackknifeMean(
                    np.imag(self.results[np.int64(self.nbr_nonthermal):]))

        if self.save_result:
            file_name = f"{self.state}_{self.geometry}_{run_type}_N_{self.N}_S_{self.S}{self.region_details}.dat"
            np.savetxt(file_name, final_value)

        else:
            print(f"\nMean = {final_value[0]} \n Var = {final_value[1]}")
            if run_type == 'sign':
                print(f"\nMean = {imag_value[0]} \n Var = {imag_value[1]}")
            elif run_type == 'disorder':
                mean_re, var_re = JackknifeMean(
                    np.real(self.results[int(self.nbr_nonthermal):]))
                mean_im, var_im = JackknifeMean(
                    np.imag(self.results[int(self.nbr_nonthermal):]))
                mean = np.sqrt(mean_re**2 + mean_im**2)
                var = var_re*((mean_re/mean)**2) + var_im*((mean_im/mean)**2)
                print("Check implementation", np.vstack((mean, var)))

    def GetFinalCoords(self):
        init_coords_file = f"{self.state}_{self.geometry}_p_coords_start_N_{self.N}_S_{self.S}{self.region_details}.npy"
        moved_ind_file = f"{self.state}_{self.geometry}_p_moved_ind_N_{self.N}_S_{self.S}{self.region_details}.npy"
        moved_coords_file = f"{self.state}_{self.geometry}_p_moved_coords_N_{self.N}_S_{self.S}{self.region_details}.npy"

        if not exists(init_coords_file):
            print("Initial coords. could not be retrieved.")
            return -1
        if not exists(moved_ind_file):
            print("Moved indices could not be retrieved.")
            return -1
        if not exists(moved_coords_file):
            print("Moved coords. could not be retrieved.")
            return -1

        init_coords = np.load(init_coords_file)
        moved_ind = np.load(moved_ind_file)
        moved_coords = np.load(moved_coords_file)

        coords = np.copy(init_coords)
        for i in range(moved_ind.shape[0]):
            coords[moved_ind[i, 0]] = moved_coords[i, 0]
            coords[moved_ind[i, 1]] = moved_coords[i, 1]

        np.save(
            f"{self.state}_{self.geometry}_p_coords_end_N_{self.N}_S_{self.S}{self.region_details}.npy", coords)

    def LoadRun(self, run_type: str):

        if self.acceptance_ratio > 0:
            file_coords_start = f"./{self.state}_{self.geometry}_{run_type}_coords_start_N_{self.N}_S_{self.S}{self.region_details}.npy"
            file_coords_end = f"./{self.state}_{self.geometry}_{run_type}_coords_end_N_{self.N}_S_{self.S}{self.region_details}.npy"

            if not exists(file_coords_start) and not exists(file_coords_end):
                print(f"{file_coords_start} and {file_coords_end} both missing!\n")
            else:
                if exists(file_coords_start) and not exists(file_coords_end):
                    self.GetFinalCoords()

                coords = np.load(file_coords_end)
                if run_type == "mod" or run_type == "sign":
                    if self.geometry == "sphere":
                        self.coords = coords[:, :-1]
                    elif self.geometry == "torus":
                        self.coords = coords[:, 0]
                    self.to_swap = np.int64(np.real(coords[:, -1]))
                else:
                    self.coords = coords

            file_results = f"./{self.state}_{self.geometry}_{run_type}_results_N_{self.N}_S_{self.S}{self.region_details}.npy"
            if exists(file_results):
                start_results = np.load(file_results)
                self.load_iter = start_results.size

                if run_type == 'sign' or run_type == 'disorder':
                    self.results = np.zeros(
                        (self.load_iter+self.nbr_iter), dtype=np.complex128)
                else:
                    self.results = np.zeros(
                        (self.load_iter+self.nbr_iter), dtype=np.float64)
                self.results[:self.load_iter] = start_results
            else:
                print("Results file not found!\n")
            print(f"Successfully loaded run of {self.load_iter} iterations.")
            self.acceptance_ratio *= self.load_iter

        else:
            if run_type == 'disorder' or run_type == 'fluct':
                self.coords = self.RandomConfig()
            else:
                self.RandomConfigSwap()
            if run_type == 'sign' or run_type == 'disorder':
                self.results = np.zeros((self.nbr_iter), dtype=np.complex128)
            else:
                self.results = np.zeros((self.nbr_iter), dtype=np.float64)

            if run_type == 'mod' or run_type == 'sign':
                self.AssignOrderToSwap()

        self.coords_tmp = np.copy(self.coords)
        if run_type == 'mod' or run_type == 'sign':
            self.from_swap = self.OrderFromSwap(self.to_swap)
            self.to_swap_tmp = np.copy(self.to_swap)
            self.from_swap_tmp = np.copy(self.from_swap)

    def RunDisorderOperator(self, composite=False):
        """
        Computes the disorder operator.
        """
        self.LoadRun('disorder')
        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        if composite:
            update = self.CF(inside_region)
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

    def DensityCF(self):
        return 1

    def RunParticleFluctuations(self, theta: np.float64 = 0, cf=False):
        """
        Computes the particle number fluctuations.
        """
        self.LoadRun('fluct')
        self.InitialWavefn()
        if cf:
            update = self.DensityCF()
        else:
            update = np.count_nonzero(self.InsideRegion(self.coords))

        for i in range(self.load_iter+1, self.load_iter+self.nbr_iter+1):
            self.StepOneParticle()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptTmp('fluct')
                if cf:
                    update = self.DensityCF()
                else:
                    update = np.count_nonzero(self.InsideRegion(self.coords))

            else:
                self.RejectTmp('fluct')

            self.results[i-1] = update
            if (i-self.load_iter) % (self.nbr_iter//20) == 0:
                if cf:
                    self.Checkpoint(i, 'fluct_cf')
                else:
                    self.Checkpoint(i, 'fluct')

        if theta != 0:
            self.result = np.exp(1j*self.result*theta)

        if cf:
            self.SaveResults('fluct_cf', theta)
        else:
            self.SaveResults('fluct', theta)

    def RunSwapP(self):
        """
        Computes the p-term in the entanglement entropy swap decomposition 
        (the probability of two copies being swappable).
        """
        self.LoadRun('p')

        if self.save_all_config:
            if self.acceptance_ratio > 0:
                prev_moved_particles = np.load(f"{self.state}_{self.geometry}_p_moved_ind_N_{self.N}_S_{self.S}{self.region_details}.npy",
                                               all_moved_particles)
                prev_moved_coords = np.save(f"{self.state}_{self.geometry}_p_moved_coords_N_{self.N}_S_{self.S}{self.region_details}.npy",
                                            all_moved_coords)
                load_iter = prev_moved_particles.shape[0]

                all_moved_particles = np.zeros(
                    (load_iter+self.nbr_iter, 2), dtype=np.uint8)
                all_moved_particles[:load_iter, 2] = prev_moved_particles
                all_moved_coords = np.zeros(
                    (load_iter+self.nbr_iter, 2), dtype=np.complex128)
                all_moved_coords[:load_iter, 2] = prev_moved_coords

            else:
                np.save(f"{self.state}_{self.geometry}_p_coords_start_N_{self.N}_S_{self.S}{self.region_details}.npy",
                        self.coords)
                all_moved_particles = np.zeros(
                    (self.nbr_iter, 2), dtype=np.uint8)
                if self.geometry == "torus":
                    all_moved_coords = np.zeros(
                        (self.nbr_iter, 2), dtype=np.complex128)
                elif self.geometry == "sphere":
                    all_moved_coords = np.zeros(
                        (self.nbr_iter, 2, 2), dtype=np.float64)

        self.InitialWavefn()
        inside_region = self.InsideRegion(self.coords)
        update = (np.count_nonzero(inside_region[:self.N]) ==
                  np.count_nonzero(inside_region[self.N:]))

        for i in range(self.load_iter+1, self.load_iter+self.nbr_iter+1):
            self.StepOneParticleTwoCopies()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()

            if self.Jacobian()*np.abs(step_amplitude)**2 > np.random.random():
                self.AcceptTmp('p')
                inside_region = self.InsideRegion(self.coords)
                update = (np.count_nonzero(inside_region[:self.N]) ==
                          np.count_nonzero(inside_region[self.N:]))
            else:
                self.RejectTmp('p')

            self.results[i-1] = update
            if (i-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'p')

            if self.save_all_config:
                all_moved_particles[i-1] = self.moved_particles
                all_moved_coords[i-1] = self.coords[self.moved_particles]

        if self.save_all_config:
            np.save(f"{self.state}_{self.geometry}_p_moved_ind_N_{self.N}_S_{self.S}{self.region_details}.npy",
                    all_moved_particles)
            np.save(f"{self.state}_{self.geometry}_p_moved_coords_N_{self.N}_S_{self.S}{self.region_details}.npy",
                    all_moved_coords)

        self.SaveResults('p')

    def RunSwapMod(self):
        """
        Computes the mod-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('mod')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialMod()

        for i in range(self.load_iter+1, self.load_iter+self.nbr_iter+1):
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

            self.results[i-1] = update
            if (i-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'mod')

        self.SaveResults('mod')

    def RunSwapIncrementalMod(self):
        """
        Computes the mod-term in the entanglement entropy swap decomposition .
        """
        self.LoadRun('mod')
        self.InitialWavefn()
        self.InitialWavefnSwap()
        update = self.InitialMod()

        for i in range(self.load_iter+1, self.load_iter+self.nbr_iter+1):
            nbr_in_region_changes = self.StepOneSwap()
            self.TmpWavefn()
            step_amplitude = self.StepAmplitude()
            self.UpdateOrderSwap(nbr_in_region_changes)
            self.TmpWavefnSwap()
            step_amplitude_swap = self.StepAmplitudeTwoCopiesSwap()

            if self.Jacobian()*np.abs(step_amplitude*step_amplitude_swap) > np.random.random():
                self.AcceptTmp('mod')
                inside_region = self.InsideRegion(
                    self.coords_tmp, boundaries="12")
                swappable_region_increment = (np.count_nonzero(inside_region[:self.N]) ==
                                              np.count_nonzero(inside_region[self.N:]))
                if swappable_region_increment:
                    self.UpdateOrderSwap(nbr_in_region_changes)
                    self.TmpWavefnSwapDelta()
                    step_amplitude_swap_delta = self.StepAmplitudeTwoCopiesSwapDelta()
                    update *= np.abs(step_amplitude_swap_delta /
                                     step_amplitude_swap)

            else:
                self.RejectTmp('mod')

            self.results[i-1] = update
            if (i-self.load_iter) % (self.nbr_iter//20) == 0:
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

        for i in range(self.load_iter+1, self.load_iter+self.nbr_iter+1):
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

            self.results[i-1] = update
            if (i-self.load_iter) % (self.nbr_iter//20) == 0:
                self.Checkpoint(i, 'sign')

        self.SaveResults('sign')

    def __init__(self, N, S, nbr_iter, nbr_nonthermal,
                 region_details, save_results=True, save_last_config=True,
                 save_all_config=True, acceptance_ratio=0):
        self.N = N
        self.S = S
        self.nbr_iter = nbr_iter
        self.nbr_nonthermal = nbr_nonthermal
        # self.region_geometry = ""
        self.region_details = region_details
        self.save_last_config = save_last_config
        self.save_all_config = save_all_config
        self.save_result = save_results
        self.acceptance_ratio = acceptance_ratio
