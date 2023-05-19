import numpy as np
from scipy.special import gamma
import math

def extract_patch_energy(vec1, vec2):
    a = np.linalg.norm(vec1, ord=2, axis=1)
    energy = np.square(a)
    V = np.mean(energy)
    sigma = np.var(vec2, axis=1)
    q = energy / (sigma + 1e-6)
    result = np.mean(q)
    return result


def estimate_GGD_parameters(vec):
    f2 = []
    gam = np.arange(0.1, 5.0, 0.001)                                # 产生候选的α
    for k in range(0, 128):
        column = vec[:, k]
        r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)     # 根据候选的α计算r(γ)
        sigma_sq = np.mean((column - np.mean(column))**2)           # σ^2的零均值估计，非零均值需要计算均值然后按照(3)式
        sigma = np.sqrt(sigma_sq)
        E = np.mean(np.abs(column - np.mean(column)))
        r = sigma_sq/((E**2) + 1e-6)                                # 根据sigma和E计算r(γ)
        diff = np.abs(r - r_gam)                                    # 计算所有r_gam与估计值r的距离
        α = gam[np.argmin(diff, axis=0)]                            # 选取距离最小对应α即为所求
        # β = α * np.sqrt((gamma(1/α))/(gamma(3/α)))                  # 将α代入公式计算得到参数β
        f2 = np.append(f2, α)
        f2 = np.append(f2, sigma)
    return f2


def extract_log_normal_distribution_feature(vec):
    f3 = []
    for k in range(0, 128):
        column = vec[:, k]                              # 输出array1的第k列
        nonzero = column[np.nonzero(column)]            # 输出第k列中的非零值
        nonzero = np.abs(nonzero)
        log_nonzero = np.log(nonzero)
        mu = np.mean(log_nonzero)
        sig = np.std(log_nonzero)
        ex = np.exp(mu + sig**2 / 2)                    # 采用对数正态分布的数学期望来表征统计特征，ex=ⅇ^(μ+1/2*δ^2)
        f3 = np.append(f3, ex)
    return f3


def atoms_count(array):
    array1 = []
    count_all = np.count_nonzero(array)
    for k in range(0, 128):
        column = array[:, k]  # 输出array的第k列
        count = np.count_nonzero(column)  # 输出array第k列中的非零值到array1
        array1 = np.append(array1, count/count_all)
    return array1

