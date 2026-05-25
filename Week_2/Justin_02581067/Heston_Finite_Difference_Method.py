import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.interpolate as spi

def heston_fdm_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, 
                             N_S, N_V, N_T, S_max, V_max):
    
    dS = S_max / N_S
    dV = V_max / N_V
    dt = T / N_T
    
    S = np.linspace(0, S_max, N_S + 1)
    V = np.linspace(0, V_max, N_V + 1)

    C = np.zeros((N_S + 1, N_V + 1))

    for k in range(N_V + 1):
        C[:, k] = np.maximum(S - K, 0.0)

    N_total = (N_S + 1) * (N_V + 1)
    
    # Initialize with ones for boundary stability
    diag_center = np.ones(N_total)
    
    diag_S_up = np.zeros(N_total)
    diag_S_down = np.zeros(N_total)
    diag_V_up = np.zeros(N_total)
    diag_V_down = np.zeros(N_total)
    diag_SV_up_up = np.zeros(N_total)
    diag_SV_down_down = np.zeros(N_total)
    diag_SV_up_down = np.zeros(N_total)
    diag_SV_down_up = np.zeros(N_total)

    for j in range(1, N_S):
        for k in range(1, N_V):
            m = j + k * (N_S + 1)
            
            s = S[j]
            v = V[k]
            
            drift_S = (r * s) / (2 * dS)
            drift_V = (kappa * (theta - v)) / (2 * dV)
            
            gamma_S = (0.5 * v * s**2) / (dS**2)
            gamma_V = (0.5 * omega**2 * v) / (dV**2)
            
            cross_SV = (rho * omega * v * s) / (4 * dS * dV)
            
            diag_center[m] = 1.0 + dt * (2 * gamma_S + 2 * gamma_V + r)
            
            diag_S_up[m]   = dt * (-drift_S - gamma_S)
            diag_S_down[m] = dt * ( drift_S - gamma_S)
            diag_V_up[m]   = dt * (-drift_V - gamma_V)
            diag_V_down[m] = dt * ( drift_V - gamma_V)
            
            diag_SV_up_up[m]     = dt * (-cross_SV)
            diag_SV_down_down[m] = dt * (-cross_SV)
            diag_SV_up_down[m]   = dt * ( cross_SV)
            diag_SV_down_up[m]   = dt * ( cross_SV)

    offsets = [
        0, 
        1, -1, 
        (N_S + 1), -(N_S + 1), 
        (N_S + 1) + 1, -(N_S + 1) - 1, 
        -(N_S + 1) + 1, (N_S + 1) - 1
    ]
    
    # Properly slice the arrays to prevent offset drifting
    diagonals = [
        diag_center, 
        diag_S_up[:-1], 
        diag_S_down[1:], 
        diag_V_up[:-(N_S + 1)], 
        diag_V_down[(N_S + 1):], 
        diag_SV_up_up[:-(N_S + 2)], 
        diag_SV_down_down[(N_S + 2):], 
        diag_SV_up_down[N_S:], 
        diag_SV_down_up[:-N_S]
    ]
    
    A = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csr")

    # Use Fortran-order to map exactly to m = j + k * (N_S + 1)
    b = C.flatten(order='F')

    for t_step in range(N_T):
        x = spla.spsolve(A, b)
        
        C_new = x.reshape((N_S + 1, N_V + 1), order='F')

        C_new[0, :] = 0.0
        C_new[-1, :] = C_new[-2, :] + dS
        C_new[:, -1] = S
        C_new[:, 0] = C_new[:, 1]

        b = C_new.flatten(order='F')

    # Use top-level submodule reference
    interp_func = spi.RegularGridInterpolator((S, V), C_new)
    call_price = interp_func(np.array([S0, v0]))[0]
    
    return call_price

if __name__ == "__main__":

    S0 = 100.0          
    K = 100.0           
    T = 0.25            
    r = 0.04            
    kappa = 1.15        
    theta = 0.0348      
    omega = 0.39        
    rho = -0.64         
    v0 = 0.03482        
    
    N_S = 100           
    N_V = 50            
    N_T = 100           
    S_max = 300.0       
    V_max = 1.0         

    start_time = time.time()
    price = heston_fdm_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, 
                                     N_S, N_V, N_T, S_max, V_max)
    end_time = time.time()
    
    print(f"Heston FDM Call Price: {price:.4f}")
    print(f"Execution Time:        {end_time - start_time:.4f} seconds")