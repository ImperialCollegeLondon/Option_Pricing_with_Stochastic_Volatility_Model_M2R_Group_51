# run_test.py
from pricing.models import bsm_european, binomial_crr_european, mc_european_gbm


# 1. 测试 BSM 闭式解
try:
    price_bsm = bsm_european(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    print(f"✅ BSM European Call Price: {price_bsm:.4f}")
except Exception as e:
    print(f"❌ BSM 报错: {e}")

# 2. 测试 CRR 二叉树
try:
    price_crr = binomial_crr_european(S=100, K=100, T=1, r=0.05, sigma=0.2, n=100, option_type='call')
    print(f"✅ CRR Binomial Call Price: {price_crr:.4f}")
except Exception as e:
    print(f"❌ CRR 报错: {e}")

# 3. 测试 蒙特卡洛模拟
try:
    price_mc, stderr_mc = mc_european_gbm(S=100, K=100, T=1, r=0.05, sigma=0.2, n_sim=10000, option_type='call')
    print(f"✅ Monte Carlo Call Price: {price_mc:.4f} (Std Error: {stderr_mc:.4f})")
except Exception as e:
    print(f"❌ MC 报错: {e}")