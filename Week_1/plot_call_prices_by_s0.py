from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricing.models import (
    binomial_crr_european,
    bsm_european,
    fd_european_implicit,
    mc_european_gbm,
)


K = 100
T = 1.0
r = 0.05
sigma = 0.2

S0_values = np.linspace(50, 150, 41)

bsm_prices = []
crr_prices = []
mc_prices = []
fd_prices = []

for S0 in S0_values:
    bsm_prices.append(bsm_european(S0, K, T, r, sigma, option_type="call"))
    crr_prices.append(
        binomial_crr_european(S0, K, T, r, sigma, n=300, option_type="call")
    )
    mc_price, _ = mc_european_gbm(
        S0, K, T, r, sigma, n_sim=50_000, option_type="call", seed=51
    )
    mc_prices.append(mc_price)
    fd_prices.append(
        fd_european_implicit(
            S0, K, T, r, sigma, n_steps=200, n_space=400, option_type="call"
        )
    )

plt.figure(figsize=(9, 6))
plt.plot(S0_values, bsm_prices, label="Black-Scholes", linewidth=2)
plt.plot(S0_values, crr_prices, "--", label="CRR Binomial", linewidth=2)
plt.plot(S0_values, mc_prices, "o", label="Monte Carlo GBM", markersize=4)
plt.plot(S0_values, fd_prices, ":", label="Implicit Finite Difference", linewidth=2)

plt.xlabel("Initial stock price $S_0$")
plt.ylabel("Call option price at time 0")
plt.title("European Call Prices as a Function of $S_0$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
