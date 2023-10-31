from numba import njit

import numpy as np

@njit
def RandomPoint():
    return np.array([np.random.random()*np.pi, np.random.random()*2*np.pi])

@njit
def RandomConfig(N):
    R = np.zeros((N,2), dtype=np.float64)
    for particle in range(N):
        R[particle,:] = RandomPoint()
        
    return R

@njit
def LaughlinDensity(N, coords, electron, m, u, v):
    psi_partial = 1

    #compute the Laughlin factors that do not cancel
    for i in range(N):
        if i != electron:
            psi_partial *= u[i]*v[electron] - u[electron]*v[i]

    return  np.sin(coords[electron,0])*np.abs(psi_partial)**(2*m)

@njit
def CoulombRealSpace(N, u, v):
    Q = (3*N-3)/2
    exp_value = 0   
    for i in range(N):
        for j in range(i+1,N):
            exp_value += 1/(np.sqrt(Q)*2*np.abs(u[i]*v[j] - u[j]*v[i]))
        
    return exp_value

@njit
def Step(N, init_coords, step_size, SamplingFunction):
    #first, choose step direction (which electron moves)
    e = np.random.randint(0,N)
    delta = np.zeros((N,2), dtype=np.float64)
    delta[e,:] = np.array([step_size*(np.random.randint(0,3)-1), 
                               step_size*(np.random.randint(0,3)-1)])
        
    #udpate cooredinates on sphere
    new_coords = init_coords + delta
    if new_coords[e,0] > np.pi:
        new_coords[e,0] = 2*np.pi - new_coords[e,0]
    elif new_coords[e,0] < 0:
        new_coords[e,0] = -new_coords[e,0]

    if new_coords[e,1] > 2*np.pi:
        new_coords[e,1] = new_coords[e,1] - 2*np.pi
    elif new_coords[e,1] < 0:
        new_coords[e,1] = 2*np.pi + new_coords[e,1]

    #calculate spinor coordinates
    u_old = np.cos(init_coords[:,0]/2)*np.exp(1j*init_coords[:,1]/2)
    v_old = np.sin(init_coords[:,0]/2)*np.exp(-1j*init_coords[:,1]/2)

    u_new = np.cos(new_coords[:,0]/2)*np.exp(1j*new_coords[:,1]/2)
    v_new = np.sin(new_coords[:,0]/2)*np.exp(-1j*new_coords[:,1]/2)

    #check the weights of the two coordinates
    eta = np.random.random()
    r = SamplingFunction(N, new_coords, e, 3, u_new, v_new)/SamplingFunction(
        N, init_coords, e, 3, u_old, v_old)

    #return the new coordinate after checking
    if r > eta:
        return 1,new_coords, u_new, v_new
    else:
        return 0,init_coords, u_old, v_old

@njit
def Run(N, M, M0, step_size, SamplingFunction):
    R = RandomConfig(N)
    total_acceptance = np.zeros((M), dtype=np.float64)
    current_acceptance = 0
    total_EV = np.zeros((M-M0+1), dtype=np.float64)
    all_values = np.zeros((M-M0), dtype=np.float64)

    
    for i in range(M0):
        b, R_new ,u, v = Step(N, R, step_size, SamplingFunction)
        R = R_new

        current_acceptance += b
        total_acceptance[i] = current_acceptance/(i+1)

        step_size = step_size + (total_acceptance[i] - 0.5)*(np.pi/50)
            
    update = CoulombRealSpace(N, u, v)

    total_EV[0] = update

    for i in range(M0, M):
        b, R_new, u, v= Step(N, R, step_size, SamplingFunction)
        R = R_new
            
        update = CoulombRealSpace(N, u, v)
        all_values[i-M0] = update

        total_EV[i-M0+1] = (total_EV[i-M0]*(i-M0+1)+update)/(i-M0+2)
            
        current_acceptance += b
        total_acceptance[i] = current_acceptance/(i+1)

        step_size = step_size + (total_acceptance[i] - 0.5)*(np.pi/50)

        if i%5000 == 0:
            if total_acceptance[i] < 0.45 or total_acceptance[i] > 0.55:
                print('unable to stabilise acceptance', total_acceptance[i])
                break;
    
    return total_acceptance, total_EV, all_values