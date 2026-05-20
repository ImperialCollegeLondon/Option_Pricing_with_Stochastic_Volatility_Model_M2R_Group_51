import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes
def bs_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Finite difference method
def fd_call_price(S0, K, T, r, sigma, Smax=20, Ns=100, Nt=100):
    dS = Smax / Ns
    dt = T / Nt
    S = np.linspace(0, Smax, Ns + 1) #current price
    
    V = np.maximum(S - K, 0) #V is the value of the option
    
    # backward iteration(time)
    for n in range(Nt):
        V_new = np.zeros(Ns + 1) 

        V_new[0] = 0
        V_new[Ns] = Smax - K * np.exp(-r * (T - (n + 1) * dt))

        m = Ns - 1
        a = np.zeros(m)
        b = np.zeros(m)
        c = np.zeros(m)
        d = np.zeros(m)
        
        for j in range(m):
            i = j + 1
            Si = S[i]
            alpha = 0.5 * sigma**2 * Si**2 * dt / (dS**2)
            beta = 0.5 * r * Si * dt / dS
            
            a[j] = - (alpha - beta)
            b[j] = 1 + r * dt + 2 * alpha
            c[j] = - (alpha + beta)
            d[j] = V[i]
        
        d[0] -= a[0] * V_new[0]
        a[0] = 0
        
        d[-1] -= c[-1] * V_new[Ns]
        c[-1] = 0
        
        # Thomas algorithm
        for j in range(1, m):
            w = a[j] / b[j-1]
            b[j] -= w * c[j-1]
            d[j] -= w * d[j-1]
        
        x = np.zeros(m)
        x[-1] = d[-1] / b[-1]
        for j in range(m-2, -1, -1):
            x[j] = (d[j] - c[j] * x[j+1]) / b[j]
        
        for j in range(m):
            V_new[j+1] = x[j]
        
        V = V_new
    
    # find the closest index
    idx = np.argmin(np.abs(S - S0))
    return V[idx]


# calculate option price and errors under different S0
def main():
    K = 5.0
    T = 1.0
    r = 0.05
    sigma = 0.3
    Smax = 20.0
    Ns = 100
    Nt = 100

    #let step length be 0.5, FDM will give the same result as BS as step length -> 0
    S0_values = np.linspace(2.0, 8.0, 9)
    
    results = []
    errors = []
    
    print("=" * 80)
    print("Table: European Call Option Prices at Time 0")
    print("=" * 80)
    print(f"{'S0':>6} {'BS Price':>12} {'FD Price':>12} {'Abs Error':>14} {'Rel Error (%)':>14}")
    print("-" * 80)
    
    for S0 in S0_values:
        bs = bs_call_price(S0, K, T, r, sigma)
        fd = fd_call_price(S0, K, T, r, sigma, Smax, Ns, Nt)
        abs_err = abs(bs - fd)
        rel_err = abs_err / bs * 100 #in percentage
        
        results.append({
            'S0': S0,
            'BS Price': bs,
            'FD Price': fd,
            'Abs Error': abs_err,
            'Rel Error (%)': rel_err
        })
        errors.append(abs_err)
        
        print(f"{S0:6.1f} {bs:12.6f} {fd:12.6f} {abs_err:14.6e} {rel_err:14.4f}")
    
    errors = np.array(errors)
    avg_abs_error = np.mean(errors)
    var_abs_error = np.var(errors)
    
    print("-" * 80)
    print(f"\n{'Average Absolute Error:':<30} {avg_abs_error:.6e}")
    print(f"{'Variance of Absolute Error:':<30} {var_abs_error:.6e}")
    
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(results)
    plt.plot(df['S0'], df['BS Price'], 'b-o', linewidth=2, label='Black-Scholes Formula')
    plt.plot(df['S0'], df['FD Price'], 'r--s', linewidth=2, label='Finite Difference Method')
    plt.xlabel('Initial Asset Price $S_0$', fontsize=12)
    plt.ylabel('Call Option Price $C(0, S_0)$', fontsize=12)
    plt.title(f'European Call Option Pricing: BS vs FDM\n(K={K}, T={T}, r={r}, σ={sigma})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('call_option_fd_vs_bs.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
