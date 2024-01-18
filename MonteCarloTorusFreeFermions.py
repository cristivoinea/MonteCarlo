import numpy as np

from MonteCarloTorusTools import RandomConfig, InsideRegion, StepOneCircle, RandomConfigSWAP, \
    AssignOrderToSwap, OrderFromSwap, StepOneSwap, UpdateOrderSwap
from WavefnFreeFermions import StepOneAmplitudeFreeFermions, StepOneAmplitudeFreeFermionsSWAP, \
    InitialModFreeFermions, InitialSignFreeFermions, UpdateWavefnFreeFermions
from utilities import SaveConfig, SaveResults, fermi_sea_kx, fermi_sea_ky, LoadRun


def RunPSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                         M: np.uint32, M0: np.uint32, step_size: np.float64,
                         region_geometry: str, region_size: np.float64,
                         save_config: np.bool_ = True, save_results: np.bool_ = True,
                         start_acceptance: np.float64 = -1
                         ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, _, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                       step_size/Lx, 'free_fermions', 'p')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")
    else:
        acceptance: np.float64 = 0
        coords = np.vstack(
            (RandomConfig(Ne, Lx, Ly), RandomConfig(Ne, Lx, Ly))).T
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    coords_tmp = np.copy(coords)

    update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
              np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        StepOneCircle(Lx, t, step_size, coords_tmp, moved_particles)

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1]) / (slogdet[0, 0] * slogdet[0, 1]) *
             np.exp(slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1

            np.copyto(coords, coords_tmp)
            np.copyto(slogdet, slogdet_tmp)

            update = (np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 0], region_geometry, boundary)) ==
                      np.count_nonzero(InsideRegion(Lx, Ly, coords[:, 1], region_geometry, boundary)))

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'p', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, coords)

    SaveResults('free_fermions', 'p', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunModSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                           M: np.uint32, M0: np.uint32, step_size: np.float64,
                           region_geometry: str, region_size: np.float64,
                           save_config: np.bool_ = True, save_results: np.bool_ = True,
                           start_acceptance: np.float64 = -1
                           ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'free_fermions', 'mod')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.float64)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]

    coords_tmp = np.copy(coords)
    coords_swap_tmp = np.copy(coords_swap)
    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    update = InitialModFreeFermions(coords, coords_swap, Kxs, Kys)
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    UpdateWavefnFreeFermions(slogdet_swap, coords_swap, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1]) / (slogdet[0, 0] * slogdet[0, 1]) *
             np.exp(slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r)**2 > np.random.random():
            accept_bit = 1
            UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                            moved_particles, nbr_A_changes)
            for cp in range(2):
                for j in range(Ne):
                    coords_swap_tmp[j, cp] = coords_tmp[from_swap_tmp[j, cp] %
                                                        Ne, from_swap_tmp[j, cp] // Ne]

            UpdateWavefnFreeFermions(
                slogdet_swap_tmp, coords_swap_tmp, Kxs, Kys)

            swap_r = ((slogdet_swap_tmp[0, 0] * slogdet_swap_tmp[0, 1]) /
                      (slogdet_swap[0, 0] * slogdet_swap[0, 1]) *
                      np.exp(slogdet_swap_tmp[1, 0] + slogdet_swap_tmp[1, 1] -
                             slogdet_swap[1, 0] - slogdet_swap[1, 1]))
            update *= np.abs(swap_r / r)

            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            np.copyto(coords_swap, coords_swap_tmp)

            np.copyto(slogdet, slogdet_tmp)
            np.copyto(slogdet_swap, slogdet_swap_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            np.copyto(coords_swap_tmp, coords_swap)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'mod', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('free_fermions', 'mod', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)


def RunSignSwapFreeFermions(Ne: np.uint8, Ns: np.uint16, t: np.complex64,
                            M: np.uint32, M0: np.uint32, step_size: np.float64,
                            region_geometry: str, region_size: np.float64,
                            save_config: np.bool_ = True, save_results: np.bool_ = True,
                            start_acceptance: np.float64 = -1
                            ):
    """
    Parameters:

    N : number of particles
    Ns : number of flux quanta
    t : torus complex aspect ratio
    M : total number of Monte Carlo interations
    M0 : number of non-thermal iterations
    step_size : initial step size in units of Lx
    region_geometry : geometry of subregion A ('stripe', 'circle')
    region_size : coverage of subregion A (percentage of total area)
    save_config : indicate whether the coordinate vector and the swap order
                    are saved (every 5% of the run)
    save_results : indicate whether the full results vector is saved, the
                    final mean and variance are saved, or just printed
    """

    Lx = np.sqrt(2*np.pi*Ns/np.imag(t))
    Ly = Lx*np.imag(t)
    print("Torus dimensions \nLx = ", Lx, "\nLy = ", Ly)

    acceptance: np.float64 = 0
    step_size *= Lx
    boundary = Lx * np.sqrt(region_size/np.pi)
    Kxs = np.reshape(fermi_sea_kx[Ne]*2*np.pi/Lx, (-1, 1))
    Kys = np.reshape(fermi_sea_ky[Ne]*2*np.pi/Ly, (-1, 1))

    if start_acceptance > 0:
        acceptance = start_acceptance
        coords, to_swap, results, prev_iter, error = LoadRun(Ne, Ns, t, M, region_size, region_geometry,
                                                             step_size/Lx, 'free_fermions', 'sign')
        if error is not None:
            print(error)
            return 1
        print(
            f"Previous run loaded successfully. Starting at iteration {prev_iter}")

    else:
        acceptance: np.float64 = 0
        coords = RandomConfigSWAP(Ne, Lx, Ly, region_geometry, boundary)
        to_swap = AssignOrderToSwap(InsideRegion(Lx, Ly, coords,
                                                 region_geometry, boundary))
        results = np.zeros((M), dtype=np.complex128)
        prev_iter = 0

    from_swap = OrderFromSwap(to_swap)

    coords_swap = np.zeros(
        (coords.shape[0], coords.shape[1]), dtype=np.complex128)
    for cp in range(2):
        for i in range(Ne):
            coords_swap[i, cp] = coords[from_swap[i, cp] %
                                        Ne, from_swap[i, cp] // Ne]

    coords_tmp = np.copy(coords)
    coords_swap_tmp = np.copy(coords_swap)
    to_swap_tmp = np.copy(to_swap)
    from_swap_tmp = np.copy(from_swap)

    update = InitialSignFreeFermions(coords, coords_swap, Kxs, Kys)
    slogdet = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap = np.zeros((2, 2), dtype=np.complex128)
    UpdateWavefnFreeFermions(slogdet, coords, Kxs, Kys)
    UpdateWavefnFreeFermions(slogdet_swap, coords_swap, Kxs, Kys)
    slogdet_tmp = np.zeros((2, 2), dtype=np.complex128)
    slogdet_swap_tmp = np.zeros((2, 2), dtype=np.complex128)

    moved_particles = np.zeros(2, dtype=np.uint8)

    for i in range(prev_iter, prev_iter+M):
        nbr_A_changes = StepOneSwap(
            Lx, t, step_size, coords_tmp, moved_particles, region_geometry, boundary)

        UpdateOrderSwap(to_swap_tmp, from_swap_tmp,
                        moved_particles, nbr_A_changes)
        for cp in range(2):
            for j in range(Ne):
                coords_swap_tmp[j, cp] = coords_tmp[from_swap_tmp[j, cp] %
                                                    Ne, from_swap_tmp[j, cp] // Ne]

        UpdateWavefnFreeFermions(slogdet_tmp, coords_tmp, Kxs, Kys)
        UpdateWavefnFreeFermions(slogdet_swap_tmp, coords_swap_tmp, Kxs, Kys)
        r = ((slogdet_tmp[0, 0] * slogdet_tmp[0, 1] * slogdet_swap[0, 0] * slogdet_swap[0, 1]) /
             (slogdet[0, 0] * slogdet[0, 1] * slogdet_swap_tmp[0, 0] * slogdet_swap_tmp[0, 1]) *
             np.exp(slogdet_swap_tmp[1, 0] + slogdet_swap_tmp[1, 1] - slogdet_swap[1, 0] - slogdet_swap[1, 1]
                    + slogdet_tmp[1, 0] + slogdet_tmp[1, 1] - slogdet[1, 0] - slogdet[1, 1]))

        if np.abs(r) > np.random.random():
            accept_bit = 1
            np.copyto(coords, coords_tmp)
            np.copyto(to_swap, to_swap_tmp)
            np.copyto(from_swap, from_swap_tmp)
            np.copyto(coords_swap, coords_swap_tmp)
            update *= r/np.abs(r)

            np.copyto(slogdet, slogdet_tmp)
            np.copyto(slogdet_swap, slogdet_swap_tmp)

        else:
            accept_bit = 0
            np.copyto(coords_tmp, coords)
            np.copyto(to_swap_tmp, to_swap)
            np.copyto(from_swap_tmp, from_swap)
            np.copyto(coords_swap_tmp, coords_swap)

        results[i] = update
        acceptance = (acceptance*i + accept_bit)/(i + 1)

        if (i+1-prev_iter) % (M//20) == 0:
            print('Iteration', i+1, 'done, current acceptance ratio:',
                  np.round(acceptance*100, 2), '%')
            if save_config:
                SaveConfig('free_fermions', 'sign', Ne, Ns, Lx, Ly, t, step_size, region_geometry, region_size,
                           results, coords, to_swap)

    SaveResults('free_fermions', 'sign', Ne, Ns, Lx, Ly, M0, t, step_size,
                results, save_results, region_geometry, region_size)
