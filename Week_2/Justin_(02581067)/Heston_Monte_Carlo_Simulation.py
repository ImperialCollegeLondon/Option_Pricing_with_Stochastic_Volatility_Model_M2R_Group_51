from src.pricing.models import bsm_european, binomial_crr_european, mc_european_gbm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import time

import numpy as np
import time

def heston_mc_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, n_steps, n_sims):
    """
    Prices a European Call option under the Heston Stochastic Volatility Model 
    using a vectorized Euler-Maruyama Monte Carlo simulation.

    return (call_price, standard_error)

    Justin(Yue) Ju
    """
    dt = T/n_steps
    S = np.zeros(n_sims) + S0
    V = np.zeros(n_sims) + v0

    for i in range(n_steps):
        W_t = np.random.normal(0, np.sqrt(dt), n_sims)
        W_t_tlide = np.random.normal(0, np.sqrt(dt), n_sims)
        W_t1 = rho * W_t + np.sqrt(1 - rho**2) * W_t_tlide
        
        S = S + r*S*dt + np.sqrt(V)*S*np.sqrt(dt)*W_t
        V = V + kappa*(theta - V)*dt + omega*np.sqrt(V)*np.sqrt(dt)*W_t1
    
    payoffs = np.maximum(S - K, 0)
    discount_factor = np.exp(-r*T)
    call_price = discount_factor * np.mean(payoffs)
    standard_error = np.std(payoffs * discount_factor) / np.sqrt(n_sims)

    return call_price, standard_error

if __name__ == "__main__":
    S0 = 100.0       # Initial stock price
    K = 100.0        # Strike price (ATM)
    T = 0.25         # Time to maturity (3 months)
    r = 0.04         # Risk-free rate

    # Heston parameters (Bakshi, Cao & Chen 1997)
    kappa = 1.15     # Mean reversion speed
    theta = 0.0348   # Long-term average variance
    omega = 0.39     # Vol of vol
    rho = -0.64      # Correlation (leverage effect)
    v0 = 0.03482     # Initial variance

    # Simulation hyper-parameters
    n_steps = 100
    n_sims = 100000

    print(f"Simulating {n_sims:,} paths with {n_steps} steps...")

    start_time = time.time()
    price, se = heston_mc_european_call(S0, K, T, r, kappa, theta, omega, rho, v0, n_steps, n_sims)
    end_time = time.time()

    print(f"\n--- Results ---")
    print(f"Heston Call Price:  {price:.4f}")
    print(f"Standard Error:     ±{se:.4f}")
    print(f"Execution Time:     {end_time - start_time:.4f} seconds")  # keep minimal comment