import numpy as np
import sympy as sp
import copy

def nelder_mead_optimization_sympy(f, vars, x0, alpha=2, beta=0.5, delta=0.5, max_iter=10000, epsilon=1e-5):
    """
    使用Nelder-Mead优化方法对函数f进行优化。
    参数:
    f: 待优化的函数，接受一组变量，返回一个数值。
    vars: 函数f中的变量列表。
    x0: 初始猜测的优化变量值。
    alpha: 扩张系数，默认为2。
    beta: 收缩系数，默认为0.5。
    delta: 缩边系数，默认为0.5。
    max_iter: 最大迭代次数，默认为1000。
    epsilon: 迭代停止的误差阈值，默认为1e-5。
    返回:
    最终优化后的变量值。
    """
    f = sp.lambdify(vars, f)
    n = len(x0)
    x0 = np.array(x0)
    simplex = np.empty((n + 1, n), dtype=np.float64)

    # 初始化单纯形
    simplex[0] = x0

    for k in range(n):
        y = x0.tolist()
        if y[k] != 0:
            y[k] = (1 + 0.05) * y[k]
        else:
            y[k] = 0.00025
        simplex[k + 1] = y
    for k in range(max_iter):
        # 按函数值对单纯形顶点排序
        simplex = simplex[np.argsort([f(*x) for x in simplex])]
        best, second_worst, worst = simplex[0], simplex[1], simplex[-1]
        # 判断是否收敛
        if np.linalg.norm(best - worst) < epsilon:
            print(f"Nelder-Mead optimization convergence achieved after {k} iterations")
            break

        # 计算（不包括最差点）的重心
        centroid = np.mean(simplex[:-1], axis=0)

        # 反射
        reflected = centroid + (centroid - worst)
        f_reflected = f(*reflected)

        if f_reflected >= f(*best) and f_reflected < f(*second_worst):
            simplex[-1] = reflected
        elif f_reflected < f(*best):
            # 扩展
            expanded = centroid + alpha * (reflected - centroid)
            if f(*expanded) < f_reflected:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        else:
            if f_reflected < f(*worst):
                # 向外压缩
                contracted_out = centroid + beta * (reflected - centroid)
                if f(*contracted_out) < f_reflected:
                    simplex[-1] = contracted_out
                else:
                    # 缩小
                    for i in range(1, n + 1):
                        simplex[i] = best + delta * (simplex[i] - best)
            else:
                # 向内压缩
                contracted_in = centroid + beta * (worst - centroid)
                if f(*contracted_in) < f(*worst):
                    simplex[-1] = contracted_in
                else:
                    # 缩小
                    for i in range(1, n + 1):
                        simplex[i] = best + delta * (simplex[i] - best)


    return simplex[0]

if __name__ == "__main__":

    x, y, z = sp.symbols('x y z')
    f = x ** 2 + y ** 2 + z ** 2 + sp.cos(x)
    f_func = sp.lambdify((x, y, z), f)


    x0 = [1.0, 1.0, 1.0]


    optimal_x = nelder_mead_optimization_sympy(f, (x, y, z), x0)
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_func(*optimal_x))
