import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pricing import heston_mc_european_call
import numpy as np

def heston_tree_european_call(S0, K, T, r, kappa, theta, omega, rho, V0, n, mv, mz, mode = "call"):
    """
        Prices a European Call/Pull option under the Heston Stochastic Volatility Model 
        using a tree based method.        

        S0: Initial stock price
        K: Exercise price at T
        T: Exercise time
        r: Risk-free interest rate
        kappa: Rate of mean reversion
        theta: Equilibrium line (baseline) of variance
        omega: Volatility of volatility
        rho: Correlation between dWt and dWt1
        V0: Initial Variance
        n: Time steps
        mv: Variance grid resolution
        mz: Log-price grid resolution
        mode: "call" or "put"
        Justin Ju
    """
    dt = T / n
    
    # Phase 1: Initialization & Grid Boundaries
    v_min, v_max = V0, V0
    z_min, z_max = np.log(S0), np.log(S0)

    for i in range(n):
        v_min_trunc = np.maximum(v_min, 0)
        v_max_trunc = np.maximum(v_max, 0)

        z_max_next = z_max + (r - v_max / 2) * dt + np.sqrt(v_max_trunc * dt)
        z_min_next = z_min + (r - v_max / 2) * dt - np.sqrt(v_max_trunc * dt)

        v_max_next = v_max + kappa * (theta - v_max_trunc) * dt + omega * np.sqrt(v_max_trunc * dt)
        v_min_next = v_min + kappa * (theta - v_min_trunc) * dt - omega * np.sqrt(v_min_trunc * dt) 

        z_max = z_max_next
        z_min = z_min_next
        v_max = v_max_next
        v_min = v_min_next

    v_min = np.maximum(v_min, 0)

    # Phase 2: State Space Discretization   
    dv = (v_max - v_min)/mv
    dz = (z_max - z_min)/mz

    print(f"dt = {dt}")
    print(f"dz = {dv}, z_max, z_min = {z_max}, {z_min}")
    print(f"dv = {dz}, v_max, v_min = {v_max}, {v_min}")
    print(f"dv/dt = {dv/dt}")
    print(f"dz/dt = {dz/dt}")

    V_nodes = np.linspace(v_min, v_max, mv + 1)
    Z_nodes = np.linspace(z_min, z_max, mz + 1)

    V_grid, Z_grid = np.meshgrid(V_nodes, Z_nodes, indexing='ij')
    V_plus_grid = np.maximum(V_grid, 0)

    U_next = np.zeros((mv+1, mz+1))

    if mode == "put":
        payoff_T = np.maximum(K - np.exp(Z_nodes), 0)
    elif mode == "call":
        payoff_T = np.maximum(- K + np.exp(Z_nodes), 0)
    else:
        raise(NotImplemented
              (f"Please choose mode between 'call' and 'put', {mode} is not implemented")
               )

    for i in range(mv+1):
        U_next[i,:] = payoff_T
    
    # Calculate the deterministic Drift & Diffusion for the whole grid once
    V_draft = kappa * (theta - V_plus_grid) * dt
    V_diff = omega * np.sqrt(V_plus_grid * dt)
    Z_drift = (r - 0.5 * V_plus_grid) * dt
    Z_diff = np.sqrt(V_plus_grid * dt)

    v_up = V_grid + V_draft + V_diff
    v_down = V_grid + V_draft - V_diff
    z_up = Z_grid + Z_drift + Z_diff
    z_down = Z_grid + Z_drift - Z_diff

    # Define the Risk-Neutral Probabilities
    q_up_up = 0.25 * (1 + rho)
    q_up_dn = 0.25 * (1 - rho)
    q_dn_up = 0.25 * (1 - rho)
    q_dn_dn = 0.25 * (1 + rho)

    def get_interpolated_prices(v_target, z_target):
        """Maps floating jump coordinates to physical grid corners and interpolates."""
        i = np.floor((v_target - v_min) / dv).astype(int)
        j = np.floor((z_target - z_min) / dz).astype(int)
        
        i = np.clip(i, 0, mv - 1)
        j = np.clip(j, 0, mz - 1)
        
        v_tilde = (v_target - V_nodes[i]) / dv
        z_tilde = (z_target - Z_nodes[j]) / dz
        
        v_tilde = np.clip(v_tilde, 0.0, 1.0)
        z_tilde = np.clip(z_tilde, 0.0, 1.0)
        
        c00 = (1 - v_tilde) * (1 - z_tilde)
        c10 = v_tilde * (1 - z_tilde)
        c01 = (1 - v_tilde) * z_tilde
        c11 = v_tilde * z_tilde
        
        prices = (c00 * U_next[i, j] +
                  c10 * U_next[i+1, j] +
                  c01 * U_next[i, j+1] +
                  c11 * U_next[i+1, j+1])
        return prices

    for k in range(n - 1, -1, -1): 
        val_up_up = q_up_up * get_interpolated_prices(v_up, z_up)
        val_up_dn = q_up_dn * get_interpolated_prices(v_up, z_down)
        val_dn_up = q_dn_up * get_interpolated_prices(v_down, z_up)
        val_dn_dn = q_dn_dn * get_interpolated_prices(v_down, z_down)

        U_current = np.exp(-r * dt) * (val_up_up + val_up_dn + val_dn_up + val_dn_dn)
        
        U_next = U_current

    final_price = get_interpolated_prices(V0, np.log(S0))
    
    return final_price

# ==========================================
# Main Execution / Testing Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Define standard Heston test parameters
    S0 = 100.0     # Initial stock price
    K = 100.0      # Strike price
    T = 1.0        # Time to maturity (1 year)
    r = 0.05       # Risk-free rate (5%)
    kappa = 2.0    # Mean reversion speed
    theta = 0.04   # Long-run variance (20% volatility)
    omega = 0.2    # Volatility of variance
    rho = -0.5     # Negative correlation (standard for equities)
    V0 = 0.04      # Initial variance
    
    print("==========================================")
    print(" Heston Model Benchmark: Tree vs MC")
    print("==========================================")
    
    # 2. Run the Tree Method
    n_tree = 1000  # Time steps
    mv = 70        # Variance grid resolution
    mz = 70      # Log-price grid resolution
    
    print(f"Running Tree Method ({n_tree} steps, {mv}x{mz} grid)...")
    start_tree = time.time()
    tree_price = heston_tree_european_call(
        S0, K, T, r, kappa, theta, omega, rho, V0, 
        n=n_tree, mv=mv, mz=mz, mode="call"
    )
    tree_time = time.time() - start_tree
    print(f"Tree Price: {tree_price:.5f} (computed in {tree_time:.4f} seconds)")

    # 3. Run the Monte Carlo Method
    n_steps_mc = 100   # Time steps for Euler discretization
    n_sims = 100000    # Number of simulated paths
    
    print(f"\nRunning Monte Carlo Method ({n_sims} simulations, {n_steps_mc} steps)...")
    start_mc = time.time()
    mc_price, mc_se = heston_mc_european_call(
        S0, K, T, r, kappa, theta, omega, rho, V0, 
        n_steps=n_steps_mc, n_sims=n_sims
    )
    mc_time = time.time() - start_mc
    print(f"MC Price:   {mc_price:.5f} (computed in {mc_time:.4f} seconds)")
    print(f"MC Std Err: {mc_se:.5f}")
    
    # 4. Compare Results
    diff = abs(tree_price - mc_price)
    print("\n==========================================")
    print(f"Absolute Difference: {diff:.5f}")
    print("==========================================")