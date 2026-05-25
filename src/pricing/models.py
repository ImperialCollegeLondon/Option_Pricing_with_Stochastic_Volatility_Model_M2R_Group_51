import time
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pandas as pd


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