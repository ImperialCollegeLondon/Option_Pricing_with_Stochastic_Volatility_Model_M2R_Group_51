import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 参数设置 (沿用第五部分，sigma替换为Heston随机波动率参数)
# ============================================================
S0    = 5.0       # 初始股价
r     = 0.05      # 无风险利率
K     = 5.0       # 行权价
T     = 1.0       # 到期时间
n     = 252       # 模拟步数
M     = 10000     # 模拟路径数

# Heston 随机波动率参数
V0    = 0.09      # 初始方差
kappa = 2.0       # 均值回归速度
theta = 0.09      # 长期均值方差
omega = 0.3       # 波动率的波动率
rho   = -0.5      # 相关系数

# ============================================================
# 方法1: Monte Carlo 模拟 (Heston模型)
# ============================================================
def heston_mc(S0, V0, r, K, T, kappa, theta, omega, rho, n, M, seed=42):
    """Heston模型下的Monte Carlo定价"""
    np.random.seed(seed)
    h = T / n
    S = np.full(M, S0, dtype=float)
    V = np.full(M, V0, dtype=float)
    
    for _ in range(n):
        Z1 = np.random.standard_normal(M)
        Z2 = np.random.standard_normal(M)
        W  = Z1
        W1 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        V_pos = np.maximum(V, 0.0)  # 保证方差非负
        S = S + r * S * h + np.sqrt(V_pos) * S * np.sqrt(h) * W
        V = V + kappa * (theta - V_pos) * h + omega * np.sqrt(V_pos) * np.sqrt(h) * W1
    
    payoff = np.maximum(S - K, 0.0)
    price  = np.exp(-r * T) * np.mean(payoff)
    std_err = np.exp(-r * T) * np.std(payoff) / np.sqrt(M)
    return price, std_err


# ============================================================
# 方法2: 有限差分法 (Heston PDE, 隐式方法)
# ============================================================
def heston_fdm(S0, V0, r, K, T, kappa, theta, omega, rho,
               NS=80, NV=40, NT=200,
               Smax_mult=3.0, Vmax_mult=5.0):
    """
    用隐式有限差分法求解Heston PDE (20)
    返回在 (S0, V0) 处的期权价格
    """
    Smin, Smax = 0.0, Smax_mult * K
    Vmin, Vmax = 0.0, Vmax_mult * theta

    dS = (Smax - Smin) / NS
    dV = (Vmax - Vmin) / NV
    dt = T / NT

    # 网格
    S_grid = np.linspace(Smin, Smax, NS + 1)   # shape (NS+1,)
    V_grid = np.linspace(Vmin, Vmax, NV + 1)   # shape (NV+1,)

    # 终端条件: C(T, S, V) = (S - K)+
    C = np.zeros((NS + 1, NV + 1))
    for j in range(NS + 1):
        C[j, :] = max(S_grid[j] - K, 0.0)

    # 内部节点索引: j=1..NS-1, k=1..NV-1
    # 将二维(j,k)映射到一维索引 idx = j*(NV-1) + k  (k从1到NV-1)
    Nint = (NS - 1) * (NV - 1)   # 内部节点总数

    def idx(j, k):
        """内部节点(j,k) -> 线性索引, j=1..NS-1, k=1..NV-1"""
        return (j - 1) * (NV - 1) + (k - 1)

    # 时间回退
    for _ in range(NT):
        A = lil_matrix((Nint, Nint))
        b = np.zeros(Nint)

        for j in range(1, NS):
            Sj = S_grid[j]
            for k in range(1, NV):
                Vk = V_grid[k]
                ii = idx(j, k)

                # PDE系数
                a_S  = r * Sj / (2 * dS)
                a_SS = 0.5 * Vk * Sj**2 / dS**2
                a_V  = kappa * (theta - Vk) / (2 * dV)
                a_VV = 0.5 * omega**2 * Vk / dV**2
                a_SV = rho * omega * Sj * Vk / (4 * dS * dV)

                # 隐式: (C_new - C_old)/dt + L[C_new] = 0
                # => C_new/dt - L_disc[C_new] = C_old/dt
                # 整理为 A * C_new_vec = b

                diag_coeff = 1.0/dt + 2*a_SS + 2*a_VV + r

                A[ii, ii] = diag_coeff

                # j+1, k
                if j + 1 <= NS - 1:
                    A[ii, idx(j+1, k)] = -(a_SS + a_S)
                else:
                    # j = NS-1, j+1 = NS 是边界 S=Smax: C[NS,k] = C[NS-1,k] + dS
                    b[ii] += (a_SS + a_S) * (C[j, k] + dS)

                # j-1, k
                if j - 1 >= 1:
                    A[ii, idx(j-1, k)] = -(a_SS - a_S)
                else:
                    # j=1, j-1=0 是边界 S=0: C=0
                    pass  # 贡献为0

                # j, k+1
                if k + 1 <= NV - 1:
                    A[ii, idx(j, k+1)] = -(a_VV + a_V)
                else:
                    # k = NV-1, k+1 = NV 是边界 V=Vmax: C = Sj
                    b[ii] += (a_VV + a_V) * Sj

                # j, k-1
                if k - 1 >= 1:
                    A[ii, idx(j, k-1)] = -(a_VV - a_V)
                else:
                    # k=1, k-1=0 是边界 V=0: 用退化PDE近似，此处简化为C[j,0]
                    b[ii] += (a_VV - a_V) * C[j, 0]

                # 混合项 (j+1,k+1)
                if j+1 <= NS-1 and k+1 <= NV-1:
                    A[ii, idx(j+1, k+1)] = -a_SV
                # (j-1,k-1)
                if j-1 >= 1 and k-1 >= 1:
                    A[ii, idx(j-1, k-1)] = -a_SV
                # (j+1,k-1)
                if j+1 <= NS-1 and k-1 >= 1:
                    A[ii, idx(j+1, k-1)] = a_SV
                # (j-1,k+1)
                if j-1 >= 1 and k+1 <= NV-1:
                    A[ii, idx(j-1, k+1)] = a_SV

                # RHS
                b[ii] += C[j, k] / dt

        # 求解线性方程组
        A_csr = A.tocsr()
        C_vec = spsolve(A_csr, b)

        # 更新内部节点
        C_new = C.copy()
        for j in range(1, NS):
            for k in range(1, NV):
                C_new[j, k] = C_vec[idx(j, k)]

        # 更新边界
        # S=0: C=0
        C_new[0, :] = 0.0
        # S=Smax: dC/dS=1 => C[NS,k] = C[NS-1,k] + dS
        C_new[NS, :] = C_new[NS-1, :] + dS
        # V=Vmax: C=S
        C_new[:, NV] = S_grid
        # V=0: 退化PDE (简化处理: 用Black-Scholes with sigma=sqrt(theta))
        for j in range(NS + 1):
            Sj = S_grid[j]
            if Sj > 0:
                # 用BS公式近似V=0边界 (sigma=sqrt(theta))
                sig0 = np.sqrt(theta)
                tau  = max(T - _ * dt, 1e-10)
                d1   = (np.log(Sj/K) + (r + 0.5*sig0**2)*tau) / (sig0*np.sqrt(tau))
                d2   = d1 - sig0*np.sqrt(tau)
                from scipy.stats import norm
                C_new[j, 0] = Sj*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
            else:
                C_new[j, 0] = 0.0

        C = C_new

    # 插值得到 (S0, V0) 处的价格
    j0 = np.searchsorted(S_grid, S0) - 1
    k0 = np.searchsorted(V_grid, V0) - 1
    j0 = np.clip(j0, 0, NS - 1)
    k0 = np.clip(k0, 0, NV - 1)

    # 双线性插值
    wS = (S0 - S_grid[j0]) / dS
    wV = (V0 - V_grid[k0]) / dV
    wS = np.clip(wS, 0, 1)
    wV = np.clip(wV, 0, 1)

    price = ((1-wS)*(1-wV)*C[j0,   k0  ] +
             wS    *(1-wV)*C[j0+1, k0  ] +
             (1-wS)*wV    *C[j0,   k0+1] +
             wS    *wV    *C[j0+1, k0+1])
    return price, C, S_grid, V_grid


# ============================================================
# 方法3: Heston半解析公式 (作为基准 benchmark)
# ============================================================
def heston_semi_analytic(S0, V0, r, K, T, kappa, theta, omega, rho):
    """
    Heston (1993) 半解析公式
    使用特征函数数值积分
    """
    from scipy.integrate import quad

    def char_func(phi, S0, V0, r, T, kappa, theta, omega, rho, j):
        """Heston特征函数"""
        if j == 1:
            u = 0.5
            b = kappa - rho * omega
        else:
            u = -0.5
            b = kappa

        a   = kappa * theta
        x   = np.log(S0)
        d   = np.sqrt((rho*omega*phi*1j - b)**2 - omega**2*(2*u*phi*1j - phi**2))
        g   = (b - rho*omega*phi*1j + d) / (b - rho*omega*phi*1j - d)

        C_cf = (r*phi*1j*T +
                a/omega**2 * ((b - rho*omega*phi*1j + d)*T
                              - 2*np.log((1 - g*np.exp(d*T))/(1 - g))))
        D_cf = ((b - rho*omega*phi*1j + d)/omega**2 *
                (1 - np.exp(d*T)) / (1 - g*np.exp(d*T)))

        return np.exp(C_cf + D_cf*V0 + 1j*phi*x)

    def integrand_P(phi, S0, V0, r, K, T, kappa, theta, omega, rho, j):
        cf  = char_func(phi, S0, V0, r, T, kappa, theta, omega, rho, j)
        val = np.real(np.exp(-1j*phi*np.log(K)) * cf / (1j*phi))
        return val

    P1, _ = quad(integrand_P, 1e-6, 200, args=(S0,V0,r,K,T,kappa,theta,omega,rho,1),
                 limit=200, epsabs=1e-6)
    P2, _ = quad(integrand_P, 1e-6, 200, args=(S0,V0,r,K,T,kappa,theta,omega,rho,2),
                 limit=200, epsabs=1e-6)

    P1 = 0.5 + P1/np.pi
    P2 = 0.5 + P2/np.pi

    price = S0*P1 - K*np.exp(-r*T)*P2
    return price


# ============================================================
# 生成 Table: 不同S0下各方法的期权价格及误差
# ============================================================
print("="*75)
print("Computing Heston option prices for different S0 values...")
print("="*75)

S0_list = [3.0, 4.0, 5.0, 6.0, 7.0]

results = {
    'S0':         [],
    'Semi-Analytic (Benchmark)': [],
    'MC Price':   [],
    'MC Std Err': [],
    'MC Abs Err': [],
    'MC Rel Err': [],
    'FDM Price':  [],
    'FDM Abs Err':[],
    'FDM Rel Err':[],
}

for s0 in S0_list:
    print(f"  Processing S0 = {s0}...")

    # 基准: 半解析
    bench = heston_semi_analytic(s0, V0, r, K, T, kappa, theta, omega, rho)

    # Monte Carlo
    mc_price, mc_std = heston_mc(s0, V0, r, K, T, kappa, theta, omega, rho, n, M)

    # FDM
    fdm_price, _, _, _ = heston_fdm(s0, V0, r, K, T, kappa, theta, omega, rho,
                                     NS=80, NV=40, NT=200)

    results['S0'].append(s0)
    results['Semi-Analytic (Benchmark)'].append(bench)
    results['MC Price'].append(mc_price)
    results['MC Std Err'].append(mc_std)
    results['MC Abs Err'].append(abs(mc_price - bench))
    results['MC Rel Err'].append(abs(mc_price - bench)/bench * 100)
    results['FDM Price'].append(fdm_price)
    results['FDM Abs Err'].append(abs(fdm_price - bench))
    results['FDM Rel Err'].append(abs(fdm_price - bench)/bench * 100)

# 打印表格
print("\n" + "="*95)
print(f"{'S0':>5} | {'Benchmark':>12} | {'MC Price':>10} | {'MC AbsErr':>10} | "
      f"{'MC RelErr%':>10} | {'FDM Price':>10} | {'FDM AbsErr':>10} | {'FDM RelErr%':>11}")
print("-"*95)
for i in range(len(S0_list)):
    print(f"{results['S0'][i]:>5.1f} | "
          f"{results['Semi-Analytic (Benchmark)'][i]:>12.6f} | "
          f"{results['MC Price'][i]:>10.6f} | "
          f"{results['MC Abs Err'][i]:>10.6f} | "
          f"{results['MC Rel Err'][i]:>9.4f}% | "
          f"{results['FDM Price'][i]:>10.6f} | "
          f"{results['FDM Abs Err'][i]:>10.6f} | "
          f"{results['FDM Rel Err'][i]:>10.4f}%")
print("="*95)

# 平均误差和方差
print(f"\nMC  - Mean Abs Error: {np.mean(results['MC Abs Err']):.6f}, "
      f"Variance: {np.var(results['MC Abs Err']):.8f}")
print(f"FDM - Mean Abs Error: {np.mean(results['FDM Abs Err']):.6f}, "
      f"Variance: {np.var(results['FDM Abs Err']):.8f}")


# ============================================================
# 生成 Figure: 不同方法的期权价格 vs S0
# ============================================================
S0_range = np.linspace(1.0, 10.0, 30)

prices_bench = []
prices_mc    = []
prices_fdm   = []

print("\nGenerating figure data...")
for s0 in S0_range:
    b  = heston_semi_analytic(s0, V0, r, K, T, kappa, theta, omega, rho)
    mc, _ = heston_mc(s0, V0, r, K, T, kappa, theta, omega, rho, n, M)
    fd, _, _, _ = heston_fdm(s0, V0, r, K, T, kappa, theta, omega, rho,
                              NS=80, NV=40, NT=200)
    prices_bench.append(b)
    prices_mc.append(mc)
    prices_fdm.append(fd)

# ---- Figure 1: 期权价格 vs S0 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(S0_range, prices_bench, 'k-',  lw=2.5, label='Semi-Analytic (Benchmark)')
ax1.plot(S0_range, prices_mc,    'b--', lw=2.0, label='Monte Carlo')
ax1.plot(S0_range, prices_fdm,   'r-.',  lw=2.0, label='FDM (Implicit)')
ax1.axvline(x=K, color='gray', linestyle=':', lw=1.5, label=f'Strike K={K}')
ax1.set_xlabel('Initial Stock Price $S_0$', fontsize=13)
ax1.set_ylabel('Call Option Price $C(0, S_0, V_0)$', fontsize=13)
ax1.set_title('Heston Model: Call Option Price vs $S_0$', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# ---- Figure 2: 绝对误差 vs S0 ----
ax2 = axes[1]
abs_err_mc  = [abs(prices_mc[i]  - prices_bench[i]) for i in range(len(S0_range))]
abs_err_fdm = [abs(prices_fdm[i] - prices_bench[i]) for i in range(len(S0_range))]

ax2.plot(S0_range, abs_err_mc,  'b--', lw=2.0, label='MC Absolute Error')
ax2.plot(S0_range, abs_err_fdm, 'r-.', lw=2.0, label='FDM Absolute Error')
ax2.set_xlabel('Initial Stock Price $S_0$', fontsize=13)
ax2.set_ylabel('Absolute Error', fontsize=13)
ax2.set_title('Absolute Error vs $S_0$ (Benchmark: Semi-Analytic)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
'''
plt.savefig('heston_option_pricing.png', dpi=150, bbox_inches='tight')
'''
plt.show()
'''
print("\nFigure saved as 'heston_option_pricing.png'")
'''
