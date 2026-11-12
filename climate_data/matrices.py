"""仅依赖标准库的对称矩阵工具：特征分解、二次型与固定种子多元正态抽样。

不引入numpy等第三方依赖。特征分解使用循环Jacobi正交消去，仅适用于
小规模对称矩阵（本项目固定为4乘4），足以支撑半正定校验与奇异协方差抽样。
绝不向协方差添加对角抖动：零特征值方向在抽样中保持零方差。
"""
import math
import random


def mat(rows, cols, fill=0.0):
    return [[fill] * cols for _ in range(rows)]


def finite_matrix(v, n):
    """返回n乘n有限数值矩阵的副本，拒绝NaN/Infinity与错误形状。"""
    if not isinstance(v, list) or len(v) != n:
        raise ValueError('协方差必须为%d乘%d矩阵' % (n, n))
    out = mat(n, n)
    for i, row in enumerate(v):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError('协方差必须为%d乘%d矩阵' % (n, n))
        for j, x in enumerate(row):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError('矩阵元素必须为有限数值')
            out[i][j] = float(x)
    return out


def symmetric_eigen(a, sweeps=100, tol=1e-15):
    """对称矩阵的Jacobi特征分解，返回(特征值升序, 列正交特征向量矩阵)。

    特征向量以列存放：eigvec[r][k]为第k个特征向量的第r个坐标，
    满足 a = eigvec · diag(eigval) · eigvec转置。
    """
    n = len(a)
    eigen = [row[:] for row in a]
    eigvec = mat(n, n)
    for i in range(n):
        eigvec[i][i] = 1.0
    scale = math.sqrt(sum(eigen[i][j] * eigen[i][j] for i in range(n) for j in range(n)))
    if scale == 0.0:
        return [0.0] * n, eigvec
    for _ in range(sweeps):
        off = math.sqrt(sum(
            eigen[i][j] * eigen[i][j]
            for i in range(n) for j in range(i + 1, n)
        ))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = eigen[p][q]
                if abs(apq) <= tol * scale:
                    continue
                app = eigen[p][p]
                aqq = eigen[q][q]
                phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
                c = math.cos(phi)
                s = math.sin(phi)
                for k in range(n):
                    akp = eigen[k][p]
                    akq = eigen[k][q]
                    eigen[k][p] = c * akp - s * akq
                    eigen[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = eigen[p][k]
                    aqk = eigen[q][k]
                    eigen[p][k] = c * apk - s * aqk
                    eigen[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = eigvec[k][p]
                    vkq = eigvec[k][q]
                    eigvec[k][p] = c * vkp - s * vkq
                    eigvec[k][q] = s * vkp + c * vkq
    eigval = [eigen[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigval[i])
    sorted_val = [eigval[i] for i in order]
    sorted_vec = mat(n, n)
    for k, i in enumerate(order):
        for r in range(n):
            sorted_vec[r][k] = eigvec[r][i]
    return sorted_val, sorted_vec


def validate_symmetric_psd(v, n=4, tolerance=1e-12):
    """校验n乘n对称半正定矩阵；奇异但半正定接受，非法即拒绝。

    对称与半正定判定均使用随矩阵量级缩放的容差；不修补负特征值、
    不加对角噪声。返回(eigenvalues, eigenvectors)以便复用分解。
    """
    a = finite_matrix(v, n)
    magnitude = max((abs(x) for row in a for x in row), default=0.0)
    sym_tol = tolerance * (1.0 + magnitude)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > sym_tol:
                raise ValueError('协方差必须对称')
    eigval, eigvec = symmetric_eigen(a)
    psd_tol = tolerance * (1.0 + max(abs(x) for x in eigval))
    if min(eigval) < -psd_tol:
        raise ValueError('协方差不是半正定：存在负特征值')
    eigval = [0.0 if abs(x) <= psd_tol else x for x in eigval]
    if any(x < 0.0 for x in eigval):
        raise ValueError('协方差不是半正定：存在负特征值')
    return eigval, eigvec


def mvnorm_sample(mean, eigval, eigvec, draws, seed):
    """固定种子的多元正态抽样，奇异V的零特征值方向严格保持零方差。

    beta_draw = mean + Q · diag(sqrt(lambda)) · z，z为独立标准正态；
    lambda为0的方向不贡献任何离散。使用标准库random.Random(seed)，
    同一种子与draws在同一Python版本下结果确定。
    """
    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError('draws必须为正整数')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed必须为整数')
    rng = random.Random(seed)
    n = len(mean)
    out = []
    for _ in range(draws):
        z = [rng.gauss(0.0, 1.0) for _ in eigval]
        perturb = [0.0] * n
        for k, lam in enumerate(eigval):
            if lam <= 0.0:
                continue
            scale = math.sqrt(lam) * z[k]
            for r in range(n):
                perturb[r] += eigvec[r][k] * scale
        out.append([mean[r] + perturb[r] for r in range(n)])
    return out


def dot(u, w):
    return sum(x * y for x, y in zip(u, w))


def mat_vec(a, u):
    return [dot(row, u) for row in a]


def quadratic(v, u):
    """u转置 · V · u，半正定矩阵下结果非负。"""
    return dot(u, mat_vec(v, u))


def quantile(samples, p):
    """线性插值分位数（p属于[0,1]，最近秩法C=1）。"""
    if not 0.0 <= p <= 1.0:
        raise ValueError('分位数概率必须位于[0,1]')
    xs = sorted(samples)
    if len(xs) == 1:
        return xs[0]
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (h - lo)
