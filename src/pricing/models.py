import time
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# ===========================================================================
# Shared Validation
# ===========================================================================

def _validate_inputs(S, K, T, r, sigma, option_type='call'):
    """Validate common option pricing inputs."""
    if S <= 0:      raise ValueError(f"S must be positive, got {S}")
    if K <= 0:      raise ValueError(f"K must be positive, got {K}")
    if T <= 0:      raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:  raise ValueError(f"sigma must be positive, got {sigma}")
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")


# ===========================================================================
# 1. Black-Scholes-Merton Closed-Form (European)
# ===========================================================================

def bsm_european(S, K, T, r, sigma, option_type='call'):
    """
    Price a European option using the Black-Scholes-Merton closed-form solution.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    option_type : str, optional
        'call' or 'put'. Default is 'call'.

    Returns
    -------
    float
        BSM option price.
    """
    _validate_inputs(S, K, T, r, sigma, option_type)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)


# ===========================================================================
# 2. Cox-Ross-Rubinstein Binomial Tree (European)
# ===========================================================================

def binomial_crr_european(S, K, T, r, sigma, n, option_type='call'):
    """
    Price a European option using the Cox-Ross-Rubinstein binomial tree.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    n : int
        Number of time steps.
    option_type : str, optional
        'call' or 'put'. Default is 'call'.

    Returns
    -------
    float
        CRR binomial tree option price.
    """
    _validate_inputs(S, K, T, r, sigma, option_type)

    dt = T / n
    u  = np.exp(sigma * np.sqrt(dt))
    d  = np.exp(-sigma * np.sqrt(dt))
    q  = (np.exp(r * dt) - d) / (u - d)

    j  = np.arange(n + 1)
    ST = S * (u**j) * (d**(n - j))

    if option_type == 'call':
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)

    discount = np.exp(-r * dt)
    for _ in range(n):
        C = discount * (q * C[1:] + (1 - q) * C[:-1])

    return float(C[0])


# ===========================================================================
# 3. Monte Carlo Simulation under GBM (European)
# ===========================================================================

def mc_european_gbm(S, K, T, r, sigma, n_sim, option_type='call', seed=None):
    """
    Price a European option via Monte Carlo simulation under GBM.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    n_sim : int
        Number of simulation paths.
    option_type : str, optional
        'call' or 'put'. Default is 'call'.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    price : float
        Monte Carlo estimated option price.
    std_err : float
        Standard error of the price estimate.
    """
    _validate_inputs(S, K, T, r, sigma, option_type)

    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(n_sim)
    ST  = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discount = np.exp(-r * T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs) / np.sqrt(n_sim)

    return price, std_err


# ===========================================================================
# 4. Implicit Finite Difference Method — Thomas Algorithm (European)
# ===========================================================================

def fd_european_implicit(S, K, T, r, sigma, n_steps, n_space,
                         S_max=None, option_type='call'):
    """
    Price a European option using the fully implicit finite difference
    method, solved via the Thomas (tridiagonal) algorithm.

    Unconditionally stable for all dt and dS choices, though accuracy
    is first-order in time O(dt) and second-order in space O(dS^2).
    For second-order time accuracy, use fd_european_cn (Crank-Nicolson).

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    n_steps : int
        Number of time steps.
    n_space : int
        Number of spatial (stock price) grid intervals.
    S_max : float or None, optional
        Upper stock price boundary. Defaults to 3*S if None.
    option_type : str, optional
        'call' or 'put'. Default is 'call'.

    Returns
    -------
    float
        Implicit FDM option price interpolated at spot S.
    """
    _validate_inputs(S, K, T, r, sigma, option_type)

    S_max = S_max if S_max is not None else 3.0 * S
    dt    = T / n_steps
    dS    = S_max / n_space
    grid  = np.linspace(0, S_max, n_space + 1)

    if option_type == 'call':
        V = np.maximum(grid - K, 0)
    else:
        V = np.maximum(K - grid, 0)

    m = n_space - 1
    i = np.arange(1, n_space)       # Interior node indices
    Si = grid[i]

    alpha = 0.5 * sigma**2 * Si**2 * dt / dS**2
    beta  = 0.5 * r * Si * dt / dS

    a = -(alpha - beta)             # Sub-diagonal
    b =  1 + r * dt + 2 * alpha    # Main diagonal
    c = -(alpha + beta)             # Super-diagonal

    for n in range(n_steps):
        d = V[1:n_space].copy()

        # Apply boundary conditions
        if option_type == 'call':
            V_lo = 0.0
            V_hi = S_max - K * np.exp(-r * (T - (n + 1) * dt))
        else:
            V_lo = K * np.exp(-r * (T - (n + 1) * dt))
            V_hi = 0.0

        d[0]  -= a[0]  * V_lo
        d[-1] -= c[-1] * V_hi

        V[1:n_space] = _solve_tridiagonal(a, b, c, d)
        V[0]        = V_lo
        V[n_space]  = V_hi

    return float(np.interp(S, grid, V))


def _solve_tridiagonal(lower, diag, upper, rhs):
    """
    Solve a tridiagonal linear system via the Thomas algorithm in O(n).

    Parameters
    ----------
    lower : np.ndarray
        Sub-diagonal (length n).
    diag : np.ndarray
        Main diagonal (length n).
    upper : np.ndarray
        Super-diagonal (length n).
    rhs : np.ndarray
        Right-hand side vector (length n).

    Returns
    -------
    np.ndarray
        Solution vector.
    """
    b = diag.copy().astype(float)
    d = rhs.copy().astype(float)
    n = len(b)

    for j in range(1, n):
        w    = lower[j] / b[j - 1]
        b[j] -= w * upper[j - 1]
        d[j] -= w * d[j - 1]

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for j in range(n - 2, -1, -1):
        x[j] = (d[j] - upper[j] * x[j + 1]) / b[j]

    return x

# ===========================================================================
# Week 2
# ===========================================================================

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
    sqrt_dt = np.sqrt(dt)
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        Z_t = np.random.standard_normal(n_sims)
        Z_t_tilde = np.random.standard_normal(n_sims)
        
        W_S = Z_t
        W_V = rho * Z_t + sqrt_1_minus_rho2 * Z_t_tilde
        
        V_max = np.maximum(V, 0.0)
        
        S = S + r * S * dt + np.sqrt(V_max) * S * sqrt_dt * W_S
        V = V + kappa * (theta - V_max) * dt + omega * np.sqrt(V_max) * sqrt_dt * W_V

    payoffs = np.maximum(S - K, 0)
    discount_factor = np.exp(-r*T)
    call_price = discount_factor * np.mean(payoffs)
    standard_error = np.std(payoffs * discount_factor) / np.sqrt(n_sims)

    return call_price, standard_error

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

    # Phase 2: State Space Discretization   
    
    dv = (v_max - v_min)/mv
    dz = (z_max - z_min)/mz

    V_nodes = np.linspace(v_min, v_max, mv + 1)
    Z_nodes = np.linspace(z_min, z_max, mz + 1)

    V_grid, Z_grid = np.meshgrid(V_nodes, Z_nodes, indexing='ij')
    V_plus_grid = np.maximum(V_grid, 0)

    U_next = np.zeros((mv+1, mz+1))

    if mode == "put":
        payoff_T = np.maximum(K - np.exp(V_nodes), 0)
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