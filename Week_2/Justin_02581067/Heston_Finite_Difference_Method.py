import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def heston_fdm_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, 
                             N_S, N_V, N_T, S_max, V_max):
    """
    Prices a European Call option under the Heston Stochastic Volatility Model 
    using the Implicit Finite Difference Method on a 2D grid (S, V).

    Parameters:
    -----------
    S0, K, T, r     : Standard option parameters (Spot, Strike, Time, Risk-free rate)
    kappa, theta    : Heston mean reversion speed and long-term average variance
    omega, rho, v0  : Heston vol-of-vol, correlation, and initial variance
    N_S, N_V, N_T   : Number of grid steps for Stock, Variance, and Time
    S_max, V_max    : The upper boundaries for the computational grid
    
    Returns:
    --------
    call_price      : The interpolated option price at (S0, v0)
    """
    
    dS = S_max / N_S
    dV = V_max / N_V
    dt = T / N_T

    S = np.arange(0, S_max, N_S+1)
    V = np.arange(0, V_max, N_V+1)
    
    C = np.zeros((N_S + 1, N_V + 1))

    
    return 0.0 # Placeholder return

if __name__ == "__main__":

    S0 = 100.0          # Initial Stock Price
    K = 100.0           # Strike Price (At-The-Money)
    T = 0.25            # Time to Maturity (3 months)
    r = 0.04            # Risk-free rate (4%)
    
    # Heston specific parameters
    kappa = 1.15        # Mean reversion speed
    theta = 0.0348      # Long-term average variance
    omega = 0.39        # Volatility of volatility
    rho = -0.64         # Correlation (Leverage effect)
    v0 = 0.03482        # Initial variance (0.1866^2)
    

    N_S = 100           # Number of steps in the Stock price grid
    N_V = 50            # Number of steps in the Variance grid
    N_T = 100           # Number of time steps (backward from T to 0)
    
    # Grid Boundaries (To prevent computing to infinity)
    S_max = 300.0       # Truncate stock price at 3x the strike
    V_max = 1.0         # Truncate variance at 1.0 (100% volatility)

    print(f"Building FDM Grid: {N_S}x{N_V} nodes over {N_T} time steps...")
    
    start_time = time.time()
    price = heston_fdm_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, 
                                     N_S, N_V, N_T, S_max, V_max)
    end_time = time.time()
    
    print(f"\n--- Results ---")
    print(f"Heston FDM Call Price: {price:.4f}")
    print(f"Execution Time:        {end_time - start_time:.4f} seconds")