import numpy as np
import sympy
from party_solver.tools import difftools

def trust_region_optimization(f, vars, x0, epsilon=1e-5, max_iter=1000):
    """
    使用信赖区域优化方法求解函数f在变量vars上的最小值。

    参数:
    f: 待优化的函数表达式。
    vars: 函数f中的变量列表。
    x0: 迭代起始点的坐标。
    epsilon: 迭代停止的精度阈值，即当梯度模长小于epsilon时认为已经收敛。
    max_iter: 最大迭代次数。

    返回:
    x_val: 优化后的点的坐标。
    """

    x_val = np.array(x0, dtype=float)
    f_lamda = sympy.lambdify(vars, f, 'numpy')
    dta = 1.0
    eta1 = 0.1
    eta2 = 0.75
    dtabar = 2.0
    tau1 = 0.5
    tau2 = 2.0
    gk_last = np.inf
    iteration = 0

    for i in range(max_iter):
        # 计算当前点的梯度和 Hessian 矩阵
        gk = difftools.gradient(f, vars, x_val)
        Bk = difftools.hessian(f, vars, x_val)
        # 检查收敛条件
        if np.linalg.norm(gk) < epsilon:
            print(f'trust region optimization converged achieved after {i + 1} iterations')
            break

        if np.linalg.norm(gk) == np.linalg.norm(gk_last):
            iteration += 1

        if iteration == 10:
            break

        gk_last = gk
        # 调用信赖域子问题狗腿法
        d = dog_leg(gk, Bk, dta)

        # 计算预测减少量和实际减少量
        deltaf = f_lamda(*x_val) - f_lamda(*(x_val + d))
        deltam = -(np.dot(gk,d)+0.5*np.dot(d,np.dot(Bk,d)))
        rk = deltaf / deltam

        # 更新信赖域大小
        if rk <= eta1:
            dta = tau1 * dta
        elif rk >= eta2 and np.linalg.norm(d) == dta:
            dta = min(tau2 * dta, dtabar)
        else:
            dta = dta

        # 更新迭代点
        if rk > eta1:
            x_val = x_val + d

    return x_val

def dog_leg(gk, Bk, delta):
    # 求解信赖域子问题: min mk(d)=gk'*d+0.5*d'*Bk*d, s.t.||d||<=delta
    pB = -np.linalg.solve(Bk, gk)
    norm_pB = np.linalg.norm(pB)

    # 判断是否在信赖域内
    if norm_pB <= delta:
        return pB

    # 计算Cauchy点
    pU = -(np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    norm_pU = np.linalg.norm(pU)

    # 判断是否在信赖域内
    if norm_pU >= delta:
        return (delta / norm_pU) * pU

    # 计算Dogleg路径
    pB_pU = pB - pU
    t = ((delta ** 2 - norm_pU ** 2) / np.dot(pB_pU, pB_pU)) ** 0.5
    return pU + t * pB_pU



if __name__ == '__main__':
    # 示例使用
    x0 = [1, 1]
    x1, x2 = sympy.symbols('x1 x2')
    f = (x1+2) ** 2 + (x2-5) ** 2

    x= trust_region_optimization(f, (x1, x2), x0)
    print(x)
