import sympy as sp
import numpy as np
from party_solver.tools import brent_method

def line_search(f, x_val, direction, vars):
    alpha = sp.Symbol('alpha')
    f_sub = f.subs(dict(zip(vars, x_val + alpha * direction)))
    f_func = sp.lambdify(alpha, f_sub)
    result = brent_method.brent_method(f_func,tol=1e-5)
    return result


def powell_optimization(f, vars, x0, epsilon=1e-5, max_iter=1000):
    """
    使用Powell优化方法最小化函数f。
    参数:
    f: 要最小化的函数。
    vars: 函数f中的变量列表。
    x0: 初始猜测的解。
    epsilon: 最小化过程的终止阈值。
    max_iter: 最大迭代次数。
    返回:
    x_val: 最小化后的解。
    """
    n = len(vars)
    x_val = np.array(x0, dtype=float)
    directions = np.eye(n)
    f_prev = float(f.subs(dict(zip(vars, x_val))))
    f_val = [f_prev]
    for k in range(max_iter):
        x_start = x_val.copy()
        # 正交方向依次搜索
        for i in range(n):
            direction = directions[i]
            alpha = line_search(f, x_val, direction, vars)
            x_val = x_val + alpha * direction
            f_val.append(float(f.subs(dict(zip(vars, x_val)))))
        new_direction = x_val - x_start
        # 判断是否达到收敛精度
        if np.linalg.norm(x_val - x_start) < epsilon:
            print(f"Powell optimization optimization convergence achieved after {k} iterations")
            break
        new_direction_norm = new_direction / np.linalg.norm(new_direction)
        x_val_next = x_val + new_direction
        f_curr = float(f.subs(dict(zip(vars, x_val_next))))
        f_val.append(f_curr)
        f0 = f_val[0]
        f2 = f_val[-2]
        f3 = f_val[-1]
        delta = [f_val[i - 1] - f_val[i] for i in range(1, len(f_val))]
        max_delta = max(delta[:-1])
        max_delta_index = delta.index(max_delta)
        # 检查新方向是否需要替换
        if f3 < f0 and (f0 - 2 * f2 + f3) * (f0 - f2 - max_delta)**2 < 0.5 * max_delta * (f0 - f3)**2:
            # 替换贡献最小方向
            directions[max_delta_index] = new_direction_norm
        elif f2 >= f3:
            x_val = x_val_next
        elif f2 < f3:
            pass
        f_prev = f_curr
        f_val = [f_prev]
    return x_val



if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    f_expr = x ** 2 + y ** 2 + z ** 2
    x0 = np.array([1.0, 1.0, 1.0])

    optimal_x = powell_optimization(f_expr, (x, y, z), x0)
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_expr.subs({x: optimal_x[0], y: optimal_x[1], z: optimal_x[2]}))
